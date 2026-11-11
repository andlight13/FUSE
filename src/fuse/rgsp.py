from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class StructuredAttribute:
    name: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in {"numerical", "categorical", "multi-valued", "boolean"}:
            raise ValueError(f"Unsupported structured attribute kind: {self.kind}")


class StructuredEncoder:
    def fit(self, structured: Mapping[str, Sequence], attributes: Sequence[StructuredAttribute]) -> StructuredEncoder:
        if not attributes or len({item.name for item in attributes}) != len(attributes):
            raise ValueError("Structured attributes must have distinct names")
        self.attributes_ = tuple(attributes)
        self.blocks_: dict[str, dict] = {}
        row_count = None
        for item in self.attributes_:
            values = list(structured[item.name])
            if row_count is None:
                row_count = len(values)
            if len(values) != row_count or row_count < 2:
                raise ValueError("Structured attributes must have the same row count of at least two")
            if item.kind == "numerical":
                numeric = np.asarray(values, dtype=np.float64)
                if numeric.ndim != 1 or not np.isfinite(numeric).all():
                    raise ValueError(f"Invalid numerical attribute: {item.name}")
                center = float(numeric.mean())
                scale = float(numeric.std())
                if scale <= 0:
                    raise ValueError(f"Constant structured attribute: {item.name}")
                raw = ((numeric - center) / scale)[:, None]
                state = {"center": center, "scale": scale}
            elif item.kind == "boolean":
                if any(value not in (True, False, 0, 1) for value in values):
                    raise ValueError(f"Invalid Boolean attribute: {item.name}")
                raw = np.asarray(values, dtype=np.float64)[:, None]
                state = {}
            else:
                if item.kind == "categorical":
                    if any(value is None for value in values):
                        raise ValueError(f"Missing categorical value: {item.name}")
                    categories = tuple(sorted(set(values), key=lambda value: (str(type(value)), str(value))))
                    raw = np.column_stack([np.equal(values, value).astype(np.float64) for value in categories])
                else:
                    if any(value is None or isinstance(value, (str, bytes)) for value in values):
                        raise ValueError(f"Invalid multi-valued attribute: {item.name}")
                    categories = tuple(sorted({entry for value in values for entry in value}, key=lambda value: (str(type(value)), str(value))))
                    raw = np.column_stack([np.fromiter((entry in value for value in values), dtype=np.float64, count=row_count) for entry in categories]) if categories else np.empty((row_count, 0))
                if not categories:
                    raise ValueError(f"Empty structured attribute: {item.name}")
                state = {"categories": categories}
            block_center = raw.mean(axis=0)
            block_scale = float(np.sqrt(np.mean(np.sum((raw - block_center) ** 2, axis=1))))
            if block_scale <= 0:
                raise ValueError(f"Constant structured attribute: {item.name}")
            state.update({"block_center": block_center, "block_scale": block_scale})
            self.blocks_[item.name] = state
        self.n_features_ = sum(len(state["block_center"]) for state in self.blocks_.values())
        return self

    def transform(self, structured: Mapping[str, Sequence]) -> np.ndarray:
        if not hasattr(self, "blocks_"):
            raise RuntimeError("StructuredEncoder is not fitted")
        blocks = []
        row_count = None
        for item in self.attributes_:
            values = list(structured[item.name])
            if row_count is None:
                row_count = len(values)
            if len(values) != row_count:
                raise ValueError("Structured attributes must have the same row count")
            state = self.blocks_[item.name]
            if item.kind == "numerical":
                numeric = np.asarray(values, dtype=np.float64)
                if numeric.ndim != 1 or not np.isfinite(numeric).all():
                    raise ValueError(f"Invalid numerical attribute: {item.name}")
                raw = ((numeric - state["center"]) / state["scale"])[:, None]
            elif item.kind == "boolean":
                if any(value not in (True, False, 0, 1) for value in values):
                    raise ValueError(f"Invalid Boolean attribute: {item.name}")
                raw = np.asarray(values, dtype=np.float64)[:, None]
            elif item.kind == "categorical":
                categories = state["categories"]
                if any(value not in categories for value in values):
                    raise ValueError(f"Unseen categorical value: {item.name}")
                raw = np.column_stack([np.equal(values, value).astype(np.float64) for value in categories])
            else:
                categories = state["categories"]
                if any(value is None or isinstance(value, (str, bytes)) or any(entry not in categories for entry in value) for value in values):
                    raise ValueError(f"Unseen multi-valued value: {item.name}")
                raw = np.column_stack([np.fromiter((entry in value for value in values), dtype=np.float64, count=row_count) for entry in categories])
            blocks.append((raw - state["block_center"]) / state["block_scale"])
        return np.concatenate(blocks, axis=1)

    def fit_transform(self, structured: Mapping[str, Sequence], attributes: Sequence[StructuredAttribute]) -> np.ndarray:
        return self.fit(structured, attributes).transform(structured)


class RGSP:
    def __init__(self, n_components: int = 3, n_slices: int = 10, n_projections: int = 200, random_state: int = 42, covariance_tolerance: float = 1e-10):
        if n_components < 1 or n_slices < 2 or n_projections < 1 or covariance_tolerance <= 0:
            raise ValueError("Invalid RGSP parameters")
        self.n_components = n_components
        self.n_slices = n_slices
        self.n_projections = n_projections
        self.random_state = random_state
        self.covariance_tolerance = covariance_tolerance

    def fit(self, embeddings: np.ndarray, structured: Mapping[str, Sequence], attributes: Sequence[StructuredAttribute]) -> RGSP:
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < self.n_components or not np.isfinite(x).all():
            raise ValueError("Invalid semantic embeddings")
        if self.n_slices > len(x):
            raise ValueError("The number of slices exceeds the number of tuples")
        self.encoder_ = StructuredEncoder()
        response = self.encoder_.fit_transform(structured, attributes)
        if len(response) != len(x):
            raise ValueError("Semantic embeddings and structured attributes must have the same row count")
        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_
        if len(x) >= x.shape[1]:
            covariance = centered.T @ centered / len(x)
            values, vectors = np.linalg.eigh(covariance)
            order = np.argsort(values)[::-1]
            values, vectors = values[order], vectors[:, order]
        else:
            _, singular_values, transposed = np.linalg.svd(centered, full_matrices=False)
            values = singular_values**2 / len(x)
            vectors = transposed.T
        keep = values > self.covariance_tolerance * values[0]
        if values[0] <= 0 or np.count_nonzero(keep) < self.n_components:
            raise ValueError("Insufficient semantic covariance rank")
        self.whitening_basis_ = vectors[:, keep] / np.sqrt(values[keep])
        whitened = centered @ self.whitening_basis_
        matrix = np.zeros((whitened.shape[1], whitened.shape[1]), dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_projections):
            direction = rng.normal(size=response.shape[1])
            direction /= np.linalg.norm(direction)
            order = np.argsort(response @ direction, kind="stable")
            for indices in np.array_split(order, self.n_slices):
                mean = whitened[indices].mean(axis=0)
                matrix += (len(indices) / len(x)) * np.outer(mean, mean)
        self.inverse_regression_matrix_ = matrix / self.n_projections
        eigenvalues, eigenvectors = np.linalg.eigh(self.inverse_regression_matrix_)
        order = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[order]
        self.components_ = eigenvectors[:, order[:self.n_components]]
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if not hasattr(self, "components_"):
            raise RuntimeError("RGSP is not fitted")
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean_) or not np.isfinite(x).all():
            raise ValueError("Invalid semantic embeddings")
        return (x - self.mean_) @ self.whitening_basis_ @ self.components_

    def fit_transform(self, embeddings: np.ndarray, structured: Mapping[str, Sequence], attributes: Sequence[StructuredAttribute]) -> np.ndarray:
        return self.fit(embeddings, structured, attributes).transform(embeddings)
