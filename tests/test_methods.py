import unittest

import numpy as np

from fuse import RGSP, SASP, StructuredAttribute, StructuredEncoder, build_reference_predicates


class MethodTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        n = 240
        group = np.repeat(np.arange(4), n // 4)
        self.structured = {
            "value": group.astype(float) + rng.normal(0, 0.05, n),
            "category": np.asarray(["a", "b", "c", "d"])[group],
            "flag": group >= 2,
            "tags": [("low",) if value < 2 else ("high",) for value in group],
        }
        self.attributes = [
            StructuredAttribute("value", "numerical"),
            StructuredAttribute("category", "categorical"),
            StructuredAttribute("flag", "boolean"),
            StructuredAttribute("tags", "multi-valued"),
        ]
        self.embeddings = np.column_stack((group + rng.normal(0, 0.1, n), rng.normal(0, 1, n)))

    def test_rgsp_inverse_regression_and_block_normalization(self):
        response = StructuredEncoder().fit_transform(self.structured, self.attributes)
        for block in (response[:, :1], response[:, 1:5], response[:, 5:6], response[:, 6:]):
            self.assertAlmostEqual(float(np.mean(np.sum(block**2, axis=1))), 1.0, places=10)
        model = RGSP(n_components=1, n_slices=8, n_projections=40, random_state=7)
        z = model.fit_transform(self.embeddings, self.structured, self.attributes)
        self.assertEqual(z.shape, (240, 1))
        self.assertTrue(np.allclose(z, model.transform(self.embeddings)))
        self.assertTrue(np.allclose(model.inverse_regression_matrix_, model.inverse_regression_matrix_.T))
        self.assertGreater(abs(np.corrcoef(z[:, 0], self.structured["value"])[0, 1]), 0.9)

    def test_sasp_reference_predicates_and_partition(self):
        predicates = build_reference_predicates(self.structured, self.attributes, budget=20, single_fraction=0.75, random_state=5)
        self.assertEqual(len(predicates), 20)
        self.assertEqual(sum(len(predicate.terms) == 1 for predicate in predicates), 15)
        self.assertTrue(all(2 <= len(predicate.terms) <= 5 for predicate in predicates[15:]))
        model = RGSP(n_components=1, n_slices=8, n_projections=30, random_state=7)
        z = model.fit_transform(self.embeddings, self.structured, self.attributes)
        ids = np.arange(1000, 1240)
        partition = SASP(n_regions=4, min_region_size=20, n_thresholds=12)
        regions = partition.fit_predict(z, self.structured, self.attributes, predicates, ids)
        self.assertEqual(partition.n_regions_, 4)
        self.assertTrue(np.array_equal(regions, partition.predict(z)))
        self.assertEqual(sorted(np.concatenate(list(partition.region_to_tuple_index_.values()))), ids.tolist())
        self.assertTrue(all(len(index) >= 20 for index in partition.region_to_tuple_index_.values()))
        self.assertTrue(all(a > b for a, b in zip(partition.objective_history_, partition.objective_history_[1:])))
        masks = np.column_stack([predicate.evaluate(self.structured) for predicate in predicates])
        objective = 0.0
        for region in np.unique(regions):
            local = masks[regions == region].mean(axis=0)
            objective += np.mean(local * (1 - local)) * np.mean(regions == region)
        self.assertAlmostEqual(partition.objective_, objective, places=12)

    def test_frequency_ranked_categorical_reference_values(self):
        categories = [f"c{rank}" for rank in range(40) for _ in range(40 - rank)]
        structured = {"category": categories, "value": np.arange(len(categories), dtype=float)}
        attributes = [StructuredAttribute("category", "categorical"), StructuredAttribute("value", "numerical")]
        predicates = build_reference_predicates(structured, attributes, budget=8, single_fraction=1.0)
        selected = [predicate.terms[0].value for predicate in predicates if predicate.terms[0].attribute == "category"]
        self.assertEqual(len(selected), 4)
        self.assertIn("c0", selected)
        self.assertTrue(any(int(value[1:]) >= 20 for value in selected))

    def test_sasp_stops_without_positive_gain(self):
        z = np.zeros((8, 1))
        structured = {"category": ["a"] * 4 + ["b"] * 4}
        attributes = [StructuredAttribute("category", "categorical")]
        partition = SASP(n_regions=4, min_region_size=2)
        partition.fit(z, structured, attributes, reference_predicates=build_reference_predicates(structured, attributes, budget=2, single_fraction=1.0))
        self.assertEqual(partition.n_regions_, 1)
        self.assertEqual(len(partition.region_to_tuple_index_[0]), 8)


if __name__ == "__main__":
    unittest.main()
