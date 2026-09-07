"""Small end-to-end checks; uses temporary data, never the real benchmark outputs.

Run: python -m unittest discover -s example/imputation_benchmark_utils -p test_macaque_volume_resume.py
"""
import contextlib
import fcntl
import io
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

import macaque_lognormscale_npy_method_correlation as bench


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cache = self.root / "model" / "cache"
        cache.mkdir(parents=True)
        for name in ("st", "spalp"):
            (self.root / name).mkdir()
        meta = {"common_genes": ["g1", "g2", "g3"], "input_dim": 3, "slice_info": []}
        rng = np.random.default_rng(7)
        for i in range(3):
            stem = f"slice_{i}"
            meta["slice_info"].append({"batch": stem + ".h5ad", "file": stem + ".npz"})
            values = rng.integers(0, 8, size=(8, 3)).astype(np.float32)
            coords = np.array([[j // 2, i % 2, 0] for j in range(8)], dtype=np.float32)
            feat = sp.csr_matrix(values)
            np.savez(cache / (stem + ".npz"), feat_data=feat.data, feat_indices=feat.indices,
                     feat_indptr=feat.indptr, feat_shape=feat.shape, coords=coords)
            np.save(self.root / "st" / (stem + ".expression.float16.npy"),
                    (values + rng.normal(0, 0.5, values.shape)).astype(np.float16))
            with h5py.File(self.root / "spalp" / (stem + ".h5ad"), "w") as handle:
                handle.create_dataset("X", data=(values * 0.8)[:, [2, 0, 1]])
                handle.create_dataset("var/_index", data=np.array([b"g3", b"g1", b"g2"]))
                handle.create_dataset("obsm/ccf", data=coords)
        with (cache / "meta.pkl").open("wb") as handle:
            pickle.dump(meta, handle)

    def args(self, name):
        argv = ["benchmark", "--model-dir", str(self.root / "model"),
                "--stereotrack-dir", str(self.root / "st"), "--spalp-dir", str(self.root / "spalp"),
                "--output", str(self.root / name / "results.csv"),
                "--summary", str(self.root / name / "summary.csv"),
                "--voxel-size", "1", "--chunk-size", "2", "--checkpoint-every", "1"]
        with patch("sys.argv", argv):
            return bench.parse_args()

    def run_benchmark(self, args):
        log = io.StringIO()
        with contextlib.redirect_stdout(log), patch.object(bench, "parse_args", return_value=args):
            bench.main()
        return log.getvalue()

    def test_resume_all_stages_and_result_publication(self):
        baseline = self.args("baseline")
        self.run_benchmark(baseline)
        expected = pd.read_csv(baseline.output)
        # Stop after a durable checkpoint but before the next operation, including
        # the critical window between saving method results and publishing CSVs.
        for stage in ("reference.aggregate", "StereoTrack.aggregate", "SpaLP.aggregate",
                      "StereoTrack.results", "SpaLP.results"):
            with self.subTest(stage=stage):
                args = self.args(stage)
                original = bench.save_pickle

                def stop_after_save(path, value):
                    original(path, value)
                    if path.name == f"chunk_000000_{stage}.pkl":
                        raise SystemExit("simulated interruption")

                with patch.object(bench, "save_pickle", side_effect=stop_after_save):
                    with self.assertRaisesRegex(SystemExit, "simulated interruption"):
                        self.run_benchmark(args)
                if stage.startswith("SpaLP"):
                    partial = pd.read_csv(args.output)
                    self.assertEqual(len(partial), 2)
                    self.assertEqual(set(partial.method), {"StereoTrack"})
                    self.assertTrue(args.summary.exists())
                log = self.run_benchmark(args)
                self.assertIn("[resume]", log)
                if stage.endswith("aggregate"):
                    self.assertIn(f"{stage}.pkl: 1 files already aggregated", log)
                pd.testing.assert_frame_equal(expected, pd.read_csv(args.output))
                pd.testing.assert_frame_equal(pd.read_csv(baseline.summary), pd.read_csv(args.summary))
                self.assertFalse(pd.read_csv(args.output).duplicated(["method", "gene"]).any())
                self.assertFalse(list(Path(str(args.output) + ".checkpoints").glob("*.aggregate.pkl")))
                # Completed reruns recover missing exports without aggregating again.
                args.output.unlink()
                args.summary.unlink()
                with patch.object(bench, "aggregate_reference", side_effect=AssertionError("recomputed")):
                    self.run_benchmark(args)
                pd.testing.assert_frame_equal(expected, pd.read_csv(args.output))

    def test_reject_changed_settings_and_inputs(self):
        args = self.args("mismatch")
        self.run_benchmark(args)
        original_csv = args.output.read_bytes()
        args.voxel_size = 2
        with self.assertRaisesRegex(SystemExit, "settings differ"):
            self.run_benchmark(args)
        args.voxel_size = 1
        path = self.root / "st" / "slice_0.expression.float16.npy"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        with self.assertRaisesRegex(SystemExit, "settings differ"):
            self.run_benchmark(args)
        self.assertEqual(original_csv, args.output.read_bytes())

    def test_atomic_write_failure_preserves_previous_checkpoint(self):
        path = self.root / "state.pkl"
        bench.save_pickle(path, {"completed": 1})
        with patch.object(bench.os, "replace", side_effect=OSError("simulated write failure")):
            with self.assertRaises(OSError):
                bench.save_pickle(path, {"completed": 2})
        with path.open("rb") as handle:
            self.assertEqual(pickle.load(handle), {"completed": 1})
        self.assertFalse(list(self.root.glob(".*.tmp")))

    def test_lock_prevents_two_writers(self):
        args = self.args("locked")
        directory = Path(str(args.output) + ".checkpoints")
        directory.mkdir(parents=True)
        with (directory / "run.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(SystemExit, "Another benchmark"):
                self.run_benchmark(args)
        self.run_benchmark(args)

    def test_existing_legacy_results_are_preserved(self):
        args = self.args("legacy")
        args.output.parent.mkdir()
        args.output.write_text("legacy results\n")
        with self.assertRaisesRegex(SystemExit, "no checkpoint metadata"):
            self.run_benchmark(args)
        self.assertEqual(args.output.read_text(), "legacy results\n")


if __name__ == "__main__":
    unittest.main()
