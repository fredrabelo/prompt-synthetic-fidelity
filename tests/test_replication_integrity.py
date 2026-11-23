import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class ReplicationIntegrityTests(unittest.TestCase):
    def test_linkage_covers_all_records_and_expected_people(self):
        link = pd.read_csv(ROOT / "data" / "person_linkage.csv")
        self.assertEqual(len(link), 7526)
        self.assertEqual(link["external_key"].nunique(), 7526)
        self.assertEqual(link["person_id"].nunique(), 6027)
        self.assertEqual((link.groupby("person_id")["external_key"].nunique() > 1).sum(), 1499)

    def test_each_person_has_exactly_one_validation_fold(self):
        folds = pd.read_csv(ROOT / "results" / "cv_folds.csv")
        self.assertEqual(len(folds), 7526)
        self.assertEqual(folds["person_id"].nunique(), 6027)
        self.assertTrue((folds.groupby("person_id")["fold"].nunique() == 1).all())
        self.assertEqual(set(folds["fold"]), set(range(5)))

    def test_single_call_variance_shares_sum_to_one(self):
        partition = pd.read_csv(ROOT / "results" / "single_call_partition.csv")
        totals = partition.groupby("model")["share"].sum().to_numpy()
        np.testing.assert_allclose(totals, np.ones(4), atol=1e-8)

    def test_result_manifest_records_person_cluster_inference(self):
        manifest = json.loads((ROOT / "results" / "manifest.json").read_text())
        self.assertEqual(manifest["n_person_wave_records"], 7526)
        self.assertEqual(manifest["n_people"], 6027)
        self.assertEqual(manifest["linked_two_wave_people"], 1499)
        self.assertIn("Person", manifest["bootstrap_unit"])
        self.assertEqual(manifest["bootstrap_resamples"], 1000)


if __name__ == "__main__":
    unittest.main()
