"""tools/same_animal_groups.py: within-breed identity grouping, review band, overrides, split checks."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PHASE2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE2 / "tools"))

import same_animal_groups as sag  # noqa: E402


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _toy():
    # a1/a2 same animal (sim ~0.99); a3 in the review band (~0.88 to a1); b1/b2 exact duplicates
    # (same dup_group); c1 alone; d1 of breed B looks like a1 but must never link across breeds.
    base = _unit([1, 0, 0, 0])
    near = _unit([1, 0.14, 0, 0])
    band = _unit([1, -0.55, 0, 0])      # ~0.88 to a1 (review band), ~0.80 to a2 (below the band)
    far = _unit([0, 0, 1, 0])
    df = pd.DataFrame({"image": ["a1", "a2", "a3", "b1", "b2", "c1", "d1"],
                       "breed": ["A", "A", "A", "A", "A", "A", "B"],
                       "group": [0, 1, 2, 3, 3, 4, 5], "split": ["x"] * 7})   # split v1 column names
    feats = np.stack([base, near, band, far, far, _unit([0, 0, 0, 1]), base])
    return df, feats


class IdentityGroupsTest(unittest.TestCase):
    def test_links_duplicates_and_similar_within_breed_only(self):
        df, feats = _toy()
        g, pairs = sag.identity_groups(df, feats, 0.90)
        gid = dict(zip(g["image"], g["identity_group"]))
        self.assertEqual(gid["a1"], gid["a2"])
        self.assertNotEqual(gid["a1"], gid["a3"])           # review band, not linked
        self.assertEqual(gid["b1"], gid["b2"])               # dup group kept
        self.assertNotEqual(gid["a1"], gid["d1"])            # other breed never linked
        self.assertEqual(g["identity_group"].nunique(), 5)
        bands = dict(zip(zip(pairs["image_a"], pairs["image_b"]), pairs["band"]))
        self.assertEqual(bands[("a1", "a2")], "linked")
        self.assertEqual(bands[("a1", "a3")], "review")
        self.assertEqual(set(pairs["breed"]), {"A"})

    def test_overrides_link_and_unlink(self):
        df, feats = _toy()
        ov = pd.DataFrame({"image_a": ["a1", "a1"], "image_b": ["a2", "a3"], "decision": ["unlink", "link"]})
        g, _ = sag.identity_groups(df, feats, 0.90, ov)
        gid = dict(zip(g["image"], g["identity_group"]))
        self.assertNotEqual(gid["a1"], gid["a2"])
        self.assertEqual(gid["a1"], gid["a3"])

    def test_cross_partition_check(self):
        df, feats = _toy()
        g, _ = sag.identity_groups(df, feats, 0.90)
        split = pd.Series(["train", "test", "val", "train", "train", "val", "test"])
        c = sag.cross_partition_check(g, split)
        self.assertEqual(c["groups_crossing_partitions"], 1)       # a1 (train) / a2 (test)
        self.assertEqual(c["images_in_crossing_groups"], 2)
        self.assertFalse(c["leakage_free"])
        self.assertTrue(sag.cross_partition_check(g, pd.Series(["train"] * 7))["leakage_free"])

    def test_length_mismatch_raises(self):
        df, feats = _toy()
        with self.assertRaises(ValueError):
            sag.identity_groups(df, feats[:-1])

    def test_cli_precomputed_features_split_check_and_new_split(self):
        df, feats = _toy()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            df.to_csv(tmp / "images.csv", index=False)
            np.save(tmp / "feats.npy", feats)
            pd.DataFrame({"image": df["image"],
                          "split": ["train", "test", "val", "train", "train", "val", "test"]}).to_csv(tmp / "split.csv", index=False)
            out = subprocess.run([sys.executable, str(PHASE2 / "tools" / "same_animal_groups.py"),
                                  "--images-csv", str(tmp / "images.csv"), "--features", str(tmp / "feats.npy"),
                                  "--split", str(tmp / "split.csv"), "--new-split", "--out", str(tmp / "out")],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            params = json.loads((tmp / "out" / "grouping_params.json").read_text())
            self.assertEqual(params["threshold"], 0.90)
            self.assertEqual(params["identity_groups"], 5)
            self.assertEqual(params["cross_partition_check"]["groups_crossing_partitions"], 1)
            self.assertEqual(params["split_v2"]["crossing_groups"], 0)
            v2 = pd.read_csv(tmp / "out" / "split_v2.csv").set_index("image")
            self.assertEqual(set(v2.columns), {"breed", "group", "split"})
            self.assertEqual(v2.loc["a1", "split"], v2.loc["a2", "split"])

    def test_resnet_path_runs_without_downloaded_weights(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rng = np.random.default_rng(0)
            names = ["x1", "x2", "y1"]
            for n in names:
                Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(tmp / f"{n}.jpg")
            df = pd.DataFrame({"image": names, "breed": ["X", "X", "Y"]})
            feats, ident = sag.resnet50_features(df, tmp, "none", image_size=32, batch_size=2)
            self.assertEqual(feats.shape, (3, 2048))
            np.testing.assert_allclose(np.linalg.norm(feats, axis=1), 1.0, atol=1e-5)
            self.assertEqual(ident["weights"], "none")
            self.assertEqual(len(ident["state_dict_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
