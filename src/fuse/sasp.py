from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .reference import ReferencePredicate, build_reference_predicates
from .rgsp import StructuredAttribute


class SASP:
    def __init__(
        self,
        n_regions: int = 64,
        min_region_size: int = 2,
        n_thresholds: int = 16,
        reference_budget: int = 32,
        single_fraction: float = 0.75,
        reference_quantiles: int = 10,
        random_state: int = 42,
    ):
        if n_regions < 1 or min_region_size < 1 or n_thresholds < 1:
            raise ValueError("Invalid partition parameters")
        self.n_regions = n_regions
        self.min_region_size = min_region_size
        self.n_thresholds = n_thresholds
        self.reference_budget = reference_budget
        self.single_fraction = single_fraction
        self.reference_quantiles = reference_quantiles
        self.random_state = random_state

    def fit(
        self,
        coordinates: np.ndarray,
        structured: Mapping[str, Sequence],
        attributes: Sequence[StructuredAttribute],
        reference_predicates: Sequence[ReferencePredicate] | None = None,
        tuple_ids: Sequence | None = None,
    ) -> SASP:
        z = np.asarray(coordinates, dtype=np.float64)
        if z.ndim != 2 or not z.shape[0] or not z.shape[1] or not np.isfinite(z).all():
            raise ValueError("Invalid semantic coordinates")
        n = len(z)
        self.n_dimensions_ = z.shape[1]
        if any(len(structured[item.name]) != n for item in attributes):
            raise ValueError("Semantic coordinates and structured attributes must have the same row count")
        if reference_predicates is None:
            reference_predicates = build_reference_predicates(
                structured,
                attributes,
                budget=self.reference_budget,
                single_fraction=self.single_fraction,
                quantiles=self.reference_quantiles,
                random_state=self.random_state,
            )
        self.reference_predicates_ = tuple(reference_predicates)
        if not self.reference_predicates_:
            raise ValueError("The reference predicate set is empty")
        masks = np.column_stack([predicate.evaluate(structured) for predicate in self.reference_predicates_]).astype(np.uint8)
        if masks.shape != (n, len(self.reference_predicates_)):
            raise ValueError("Invalid reference predicate masks")
        if tuple_ids is None:
            identifiers = np.arange(n)
        else:
            identifiers = np.asarray(tuple_ids)
            if identifiers.ndim != 1 or len(identifiers) != n or len(set(identifiers.tolist())) != n:
                raise ValueError("Tuple identifiers must be unique and aligned with the relation")
        leaves = {0: np.arange(n, dtype=np.int64)}
        self.splits_: dict[int, tuple[int, float, int, int]] = {}
        self.split_gains_: list[float] = []
        self.objective_history_ = [self._impurity(masks.sum(axis=0), n, n, masks.shape[1])]
        next_node = 1
        candidates = {0: self._best_split(z, masks, leaves[0])}
        while len(leaves) < self.n_regions:
            best_node = None
            best = None
            for node in sorted(candidates):
                candidate = candidates[node]
                if candidate is not None and (best is None or candidate[0] > best[0]):
                    best_node, best = node, candidate
            if best is None or best[0] <= 0:
                break
            gain, dimension, threshold = best
            indices = leaves.pop(best_node)
            del candidates[best_node]
            left = indices[z[indices, dimension] <= threshold]
            right = indices[z[indices, dimension] > threshold]
            left_node, right_node = next_node, next_node + 1
            next_node += 2
            leaves[left_node] = left
            leaves[right_node] = right
            candidates[left_node] = self._best_split(z, masks, left)
            candidates[right_node] = self._best_split(z, masks, right)
            self.splits_[best_node] = (dimension, threshold, left_node, right_node)
            self.split_gains_.append(gain)
            self.objective_history_.append(self.objective_history_[-1] - gain)
        self.leaf_regions_ = {node: region for region, node in enumerate(sorted(leaves))}
        self.region_ids_ = np.empty(n, dtype=np.int32)
        self.region_to_tuple_index_ = {}
        for node, region in self.leaf_regions_.items():
            indices = leaves[node]
            self.region_ids_[indices] = region
            self.region_to_tuple_index_[region] = identifiers[indices].copy()
        self.n_regions_ = len(leaves)
        self.objective_ = self.objective_history_[-1]
        return self

    @staticmethod
    def _impurity(counts: np.ndarray, size: int, total_size: int, n_predicates: int) -> float:
        local = counts / size
        return float((size / total_size) * np.sum(local * (1 - local)) / n_predicates)

    def _best_split(self, z: np.ndarray, masks: np.ndarray, indices: np.ndarray) -> tuple[float, int, float] | None:
        size = len(indices)
        if size < 2 * self.min_region_size:
            return None
        total_size, n_predicates = masks.shape
        total = masks[indices].sum(axis=0, dtype=np.int64)
        parent_impurity = self._impurity(total, size, total_size, n_predicates)
        best = None
        for dimension in range(z.shape[1]):
            values = z[indices, dimension]
            thresholds = np.unique(np.quantile(values, np.linspace(0, 1, self.n_thresholds + 2)[1:-1]))
            order = np.argsort(values, kind="stable")
            cut = np.searchsorted(values[order], thresholds, side="right")
            valid = (cut >= self.min_region_size) & (size - cut >= self.min_region_size)
            if not np.any(valid):
                continue
            cut = cut[valid]
            thresholds = thresholds[valid]
            cut, positions = np.unique(cut, return_index=True)
            thresholds = thresholds[positions]
            sorted_masks = masks[indices[order]]
            segment_counts = np.add.reduceat(sorted_masks, np.r_[0, cut], axis=0, dtype=np.int64)
            left_count = np.cumsum(segment_counts[:-1], axis=0, dtype=np.int64).astype(np.float64)
            right_count = total[None, :] - left_count
            left_probability = left_count / cut[:, None]
            right_probability = right_count / (size - cut)[:, None]
            left_impurity = (cut / total_size) * np.mean(left_probability * (1 - left_probability), axis=1)
            right_impurity = ((size - cut) / total_size) * np.mean(right_probability * (1 - right_probability), axis=1)
            gains = parent_impurity - left_impurity - right_impurity
            position = int(np.argmax(gains))
            candidate = (float(gains[position]), dimension, float(thresholds[position]))
            if best is None or candidate[0] > best[0]:
                best = candidate
        return best

    def predict(self, coordinates: np.ndarray) -> np.ndarray:
        if not hasattr(self, "leaf_regions_"):
            raise RuntimeError("SASP is not fitted")
        z = np.asarray(coordinates, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self.n_dimensions_ or not np.isfinite(z).all():
            raise ValueError("Invalid semantic coordinates")
        result = np.empty(len(z), dtype=np.int32)
        pending = [(0, np.arange(len(z), dtype=np.int64))]
        while pending:
            node, indices = pending.pop()
            if node in self.leaf_regions_:
                result[indices] = self.leaf_regions_[node]
            else:
                dimension, threshold, left, right = self.splits_[node]
                side = z[indices, dimension] <= threshold
                pending.append((left, indices[side]))
                pending.append((right, indices[~side]))
        return result

    def fit_predict(
        self,
        coordinates: np.ndarray,
        structured: Mapping[str, Sequence],
        attributes: Sequence[StructuredAttribute],
        reference_predicates: Sequence[ReferencePredicate] | None = None,
        tuple_ids: Sequence | None = None,
    ) -> np.ndarray:
        self.fit(coordinates, structured, attributes, reference_predicates, tuple_ids)
        return self.region_ids_.copy()
