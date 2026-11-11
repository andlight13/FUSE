# FUSE

RGSP (Relationally Guided Semantic Projection) and SASP (Selectivity-Aware Semantic Partitioning).

## Method

RGSP encodes numerical, categorical, multi-valued, and Boolean structured attributes. The semantic embeddings are centered and whitened. Projective-resampling sliced inverse regression forms the inverse-regression matrix from random projections of the structured response and approximately equal-frequency slices. The leading eigenvectors yield the semantic coordinates $(z_1,\ldots,z_d)$.

SASP constructs reference relational predicates from offline structured-attribute statistics: numerical quantile ranges, categorical equalities, and conjunctions of two to five predicates. It recursively splits the semantic coordinates at local empirical quantiles, selecting the largest positive reduction in mean relational impurity subject to the minimum region size. The output includes region identifiers and a region-to-tuple index.

## Installation

```powershell
conda run -n CE python -m pip install -e .
```

## Usage

```python
from fuse import RGSP, SASP, StructuredAttribute

attributes = [
    StructuredAttribute("price", "numerical"),
    StructuredAttribute("category", "categorical"),
]
structured = {"price": prices, "category": categories}

rgsp = RGSP(n_components=3)
coordinates = rgsp.fit_transform(embeddings, structured, attributes)

sasp = SASP(n_regions=64, min_region_size=20)
region_ids = sasp.fit_predict(coordinates, structured, attributes, tuple_ids=tuple_ids)
region_to_tuple_index = sasp.region_to_tuple_index_
```

`embeddings` is a row-aligned NumPy matrix. Structured columns and `tuple_ids` use the same tuple order. RGSP accepts pretrained semantic embeddings; embedding model selection is external to RGSP. The SASP reference predicate set is constructed from these structured columns unless supplied explicitly. `predict` assigns new semantic coordinates to the fitted regions.

## Verification

```powershell
$env:PYTHONPATH = "src"
conda run -n CE python -m unittest discover -s tests -v
```
