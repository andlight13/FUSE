from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from .rgsp import StructuredAttribute


@dataclass(frozen=True)
class AtomicPredicate:
    attribute: str
    operator: str
    value: object
    upper: object | None = None

    def evaluate(self, structured: Mapping[str, Sequence]) -> np.ndarray:
        values = list(structured[self.attribute])
        if self.operator == "le":
            return np.asarray(values, dtype=np.float64) <= self.value
        if self.operator == "ge":
            return np.asarray(values, dtype=np.float64) >= self.value
        if self.operator == "between":
            numeric = np.asarray(values, dtype=np.float64)
            return (numeric >= self.value) & (numeric <= self.upper)
        if self.operator == "eq":
            return np.fromiter((value == self.value for value in values), dtype=bool, count=len(values))
        if self.operator == "contains":
            return np.fromiter((self.value in value for value in values), dtype=bool, count=len(values))
        raise ValueError(f"Unsupported predicate operator: {self.operator}")


@dataclass(frozen=True)
class ReferencePredicate:
    terms: tuple[AtomicPredicate, ...]

    def evaluate(self, structured: Mapping[str, Sequence]) -> np.ndarray:
        if not self.terms:
            raise ValueError("Reference predicate has no terms")
        result = self.terms[0].evaluate(structured)
        for term in self.terms[1:]:
            result &= term.evaluate(structured)
        return result


def _attribute_candidates(attribute: StructuredAttribute, values: Sequence, budget: int, quantiles: int) -> list[AtomicPredicate]:
    name = attribute.name
    if attribute.kind == "numerical":
        numeric = np.asarray(values, dtype=np.float64)
        if numeric.ndim != 1 or not np.isfinite(numeric).all():
            raise ValueError(f"Invalid numerical attribute: {name}")
        thresholds = np.unique(np.quantile(numeric, np.linspace(0, 1, quantiles + 2)[1:-1]))
        candidates = [AtomicPredicate(name, "le", float(value)) for value in thresholds]
        candidates += [AtomicPredicate(name, "ge", float(value)) for value in thresholds]
        candidates += [AtomicPredicate(name, "between", float(lower), float(upper)) for lower, upper in combinations(thresholds, 2)]
        return candidates
    if attribute.kind == "multi-valued":
        counts: dict[object, int] = {}
        for row in values:
            for value in set(row):
                counts[value] = counts.get(value, 0) + 1
        operator = "contains"
    else:
        counts = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        operator = "eq"
    ranked = sorted(counts, key=lambda value: (-counts[value], str(type(value)), str(value)))
    if len(ranked) > budget:
        frequent_count = max(1, (budget + 1) // 2)
        frequent = ranked[:frequent_count]
        remainder = ranked[frequent_count:]
        positions = np.linspace(0, len(remainder) - 1, budget - frequent_count, dtype=int)
        sampled = [remainder[position] for position in positions]
        ranked = [value for pair in zip(frequent, sampled) for value in pair]
        ranked += frequent[len(sampled):]
    return [AtomicPredicate(name, operator, value) for value in ranked]


def build_reference_predicates(
    structured: Mapping[str, Sequence],
    attributes: Sequence[StructuredAttribute],
    budget: int = 32,
    single_fraction: float = 0.75,
    quantiles: int = 10,
    random_state: int = 42,
) -> list[ReferencePredicate]:
    if budget < 1 or not 0 <= single_fraction <= 1 or quantiles < 2:
        raise ValueError("Invalid reference predicate parameters")
    attributes = tuple(attributes)
    if not attributes or len({item.name for item in attributes}) != len(attributes):
        raise ValueError("Structured attributes must have distinct names")
    n = len(structured[attributes[0].name])
    if n < 2 or any(len(structured[item.name]) != n for item in attributes):
        raise ValueError("Structured attributes must have the same row count of at least two")
    single_target = round(budget * single_fraction)
    conjunction_target = budget - single_target
    if conjunction_target and len(attributes) < 2:
        raise ValueError("Conjunctions require at least two structured attributes")
    pools = []
    attribute_budget = max(1, (single_target + len(attributes) - 1) // len(attributes))
    for item in attributes:
        candidates = _attribute_candidates(item, structured[item.name], attribute_budget, quantiles)
        pools.append([candidate for candidate in candidates if 0 < candidate.evaluate(structured).sum() < n])
    singles: list[ReferencePredicate] = []
    offsets = [0] * len(pools)
    expanded = False
    while len(singles) < single_target:
        advanced = False
        for index, pool in enumerate(pools):
            if offsets[index] < len(pool):
                singles.append(ReferencePredicate((pool[offsets[index]],)))
                offsets[index] += 1
                advanced = True
                if len(singles) == single_target:
                    break
        if not advanced:
            if expanded:
                raise ValueError("Insufficient nondegenerate single reference predicates")
            for index, item in enumerate(attributes):
                candidates = _attribute_candidates(item, structured[item.name], max(1, single_target), quantiles)
                pools[index] += [candidate for candidate in candidates if candidate not in pools[index] and 0 < candidate.evaluate(structured).sum() < n]
            expanded = True
    if not conjunction_target:
        return singles
    by_attribute = {item.name: [predicate.terms[0] for predicate in singles if predicate.terms[0].attribute == item.name] for item in attributes}
    active = [name for name, terms in by_attribute.items() if terms]
    if len(active) < 2:
        raise ValueError("Insufficient attributes for reference conjunctions")
    rng = np.random.default_rng(random_state)
    result = list(singles)
    seen = set(result)
    attempts = 0
    while len(result) < budget and attempts < max(1000, conjunction_target * 500):
        arity = min(2 + ((len(result) - single_target) % 4), len(active))
        names = rng.choice(active, size=arity, replace=False)
        terms = tuple(sorted((by_attribute[name][rng.integers(len(by_attribute[name]))] for name in names), key=lambda term: term.attribute))
        predicate = ReferencePredicate(terms)
        attempts += 1
        if predicate in seen:
            continue
        support = int(predicate.evaluate(structured).sum())
        if 0 < support < n:
            result.append(predicate)
            seen.add(predicate)
    if len(result) != budget:
        raise ValueError("Insufficient nondegenerate reference conjunctions")
    return result
