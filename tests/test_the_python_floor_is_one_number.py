"""Every file that states the oldest supported Python has to state the same one.

The floor lives in five places that no build step reconciles: ``requires-python`` in
`pyproject.toml`, which is what pip enforces; the oldest version classifier beside it, which is
what PyPI advertises; ``python_min`` in the conda recipe, which is what the conda package
declares; and the interpreter the two GitHub workflows install, which is what the project is
actually tested and documented on. A release that raises some of them and not the rest ships a
package that refuses an interpreter it is still tested on, or is tested on one it refuses, and
nothing fails while it happens.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _search(path: pathlib.Path, pattern: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(pattern, text, flags=re.M)
    assert match, "%s has no line matching %r" % (path.name, pattern)
    return match.group(1)


class TestThePythonFloorIsOneNumber(unittest.TestCase):
    def setUp(self) -> None:
        self.declared = _search(ROOT / "pyproject.toml",
                                r'^requires-python = ">=([0-9]+\.[0-9]+)"')

    def test_the_declared_floor_is_a_major_minor_version(self) -> None:
        self.assertRegex(self.declared, r"^\d+\.\d+$")

    def test_the_oldest_classifier_is_the_declared_floor(self) -> None:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        versions = re.findall(r'"Programming Language :: Python :: (\d+\.\d+)"', text)
        self.assertTrue(versions, "pyproject.toml lists no Python version classifier")
        oldest = min(versions, key=lambda v: tuple(int(part) for part in v.split(".")))
        self.assertEqual(oldest, self.declared,
                         "the oldest version classifier is %s where requires-python declares "
                         "%s" % (oldest, self.declared))

    def test_every_file_states_the_declared_floor(self) -> None:
        found = {
            "conda/meta.yaml":
                _search(ROOT / "conda" / "meta.yaml",
                        r'^\{% set python_min = "([^"]+)" %\}'),
            "the test matrix of .github/workflows/ci.yml":
                _search(ROOT / ".github" / "workflows" / "ci.yml",
                        r'^\s*python-version: \["([^"]+)"'),
            "the documentation environment of .github/workflows/docs.yml":
                _search(ROOT / ".github" / "workflows" / "docs.yml",
                        r'^\s*python=([0-9]+\.[0-9]+)\s*$'),
        }
        for where, value in found.items():
            with self.subTest(where=where):
                self.assertEqual(value, self.declared,
                                 "%s states %s where requires-python declares %s"
                                 % (where, value, self.declared))


if __name__ == "__main__":
    unittest.main()
