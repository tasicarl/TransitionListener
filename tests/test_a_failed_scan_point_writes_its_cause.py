"""A scan's output says why a point failed, not only that it did.

`errormsg` is a registered result column, but neither scan writer can carry it. Both build a
float array, `np.full((N, len(outnames)), np.nan)`, over the observable names alone, and
`errormsg` is not among them. So a failed point reached the scan output as a bare code in
`error.txt`, and the sentence saying what went wrong existed only in the run's error log, which
is a different file in a different place and is not written per scan.

`_save_error_messages` writes it beside the numeric output, for both formats, since the hdf5
datasets cannot carry a string column either. Nothing is written when no point failed.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np

from transitionlistener.interface import scans


def a_failed_point(message, code=10):
    return {"error": np.array([code]), "errormsg": message}


def a_good_point(alpha=0.1):
    return {"strongestTransitionObservables": {"alpha": alpha}}


class ALineScanWritesTheCauseOfEachFailureTests(unittest.TestCase):
    """`save_results_1d`, both formats."""

    def save(self, points, output_format="txt"):
        directory = pathlib.Path(tempfile.mkdtemp())
        outpath = str(directory) + "/"
        scans.save_results_1d(
            list(points), len(points), np.linspace(1.0, 2.0, len(points)), outpath,
            {"alpha": "alpha"}, output_format, "lambda4", {"lambda1": 0.28894})
        return directory

    def read(self, directory):
        path = directory / "errormsg.txt"
        return path.read_text() if path.exists() else None

    def test_a_clean_scan_writes_no_message_file(self):
        """An empty file beside a clean scan would be a new thing to explain."""
        self.assertIsNone(self.read(self.save([a_good_point(), a_good_point()])))

    def test_a_failed_point_writes_its_message(self):
        written = self.read(self.save([a_good_point(), a_failed_point("T_bro below T_sym")]))
        self.assertIsNotNone(written)
        self.assertIn("T_bro below T_sym", written)

    def test_the_grid_position_of_the_failure_is_named(self):
        written = self.read(self.save(
            [a_good_point(), a_good_point(), a_failed_point("T_bro below T_sym")]))
        rows = [line for line in written.splitlines() if not line.startswith("#")]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].split("\t")[0], "2")

    def test_each_failure_is_one_row(self):
        written = self.read(self.save(
            [a_failed_point("first cause"), a_good_point(), a_failed_point("second cause")]))
        rows = [line for line in written.splitlines() if not line.startswith("#")]
        self.assertEqual(len(rows), 2)
        self.assertEqual([row.split("\t")[0] for row in rows], ["0", "2"])

    def test_a_multiline_message_stays_on_one_row(self):
        """Otherwise one failure looks like several, and the columns stop lining up."""
        written = self.read(self.save([a_failed_point("first line\nsecond line")]))
        rows = [line for line in written.splitlines() if not line.startswith("#")]
        self.assertEqual(len(rows), 1)
        self.assertIn("first line second line", rows[0])

    def test_the_hdf5_format_gets_the_messages_too(self):
        """Its datasets cannot carry a string column either."""
        written = self.read(self.save([a_failed_point("T_bro below T_sym")], "hdf5"))
        self.assertIsNotNone(written)
        self.assertIn("T_bro below T_sym", written)

    def test_the_numeric_error_column_is_still_written(self):
        """The message is added beside the code, not instead of it."""
        directory = self.save([a_good_point(), a_failed_point("T_bro below T_sym")])
        codes = np.loadtxt(directory / "error.txt")
        self.assertEqual(int(codes[1]), 10)


class AGridScanWritesTheCauseOfEachFailureTests(unittest.TestCase):
    """`save_results_2d`, where a point has two indices."""

    def save(self, rows, output_format="txt"):
        n = len(rows)
        directory = pathlib.Path(tempfile.mkdtemp())
        outpath = str(directory) + "/"
        flat = [point for row in rows for point in row]
        scans.save_results_2d(
            flat, n, np.linspace(1.0, 2.0, n), np.linspace(3.0, 4.0, n), outpath,
            {"alpha": "alpha"}, output_format,
            {"x": "lambda4", "y": "lambda5"}, {"lambda1": 0.28894})
        return directory

    def read(self, directory):
        path = directory / "errormsg.txt"
        return path.read_text() if path.exists() else None

    def test_a_clean_grid_writes_no_message_file(self):
        self.assertIsNone(self.read(self.save(
            [[a_good_point(), a_good_point()], [a_good_point(), a_good_point()]])))

    def test_both_indices_of_a_failed_point_are_named(self):
        written = self.read(self.save(
            [[a_good_point(), a_good_point()],
             [a_good_point(), a_failed_point("T_bro below T_sym")]]))
        rows = [line for line in written.splitlines() if not line.startswith("#")]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].split("\t")[0], "1,1")

    def test_every_failed_point_appears(self):
        written = self.read(self.save(
            [[a_failed_point("a"), a_good_point()],
             [a_failed_point("b"), a_failed_point("c")]]))
        rows = [line for line in written.splitlines() if not line.startswith("#")]
        self.assertEqual([row.split("\t")[0] for row in rows], ["0,0", "1,0", "1,1"])


class TheFileIsReadableTests(unittest.TestCase):
    """What the user opens."""

    def test_it_has_a_header_naming_its_two_columns(self):
        directory = pathlib.Path(tempfile.mkdtemp())
        scans.save_results_1d(
            [a_failed_point("T_bro below T_sym")], 1, np.array([1.0]), str(directory) + "/",
            {"alpha": "alpha"}, "txt", "lambda4", {"lambda1": 0.28894})
        first = (directory / "errormsg.txt").read_text().splitlines()[0]
        self.assertTrue(first.startswith("#"))
        self.assertIn("cause", first)


if __name__ == "__main__":
    unittest.main()
