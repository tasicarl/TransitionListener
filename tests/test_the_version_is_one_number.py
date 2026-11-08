"""Every file that states the released version has to state the same one.

The version lives in five places that no build step reconciles: `pyproject.toml`, which is
what an installed copy reports; the fallback in `src/transitionlistener/__init__.py`, which
is what a checkout reports before installation; the conda recipe; the citation file; and the
newest section heading of the changelog. A release that bumps some of them and not the rest
ships a package that reports one version and a citation that claims another, and nothing
fails while it happens.

That is not hypothetical. The fallback said `1.5` from v1 until the 2.2.0 release, so every
checkout reported 1.5 from `tl --version`, and the conda recipe carried the checksum of the
2.0.1 source archive through the 2.1.0 bump.

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


class TestTheVersionIsOneNumber(unittest.TestCase):
    def setUp(self) -> None:
        self.declared = _search(ROOT / "pyproject.toml", r'^version = "([^"]+)"')

    def test_the_declared_version_is_a_release_number(self) -> None:
        self.assertRegex(self.declared, r"^\d+\.\d+\.\d+$")

    def test_every_file_states_the_declared_version(self) -> None:
        found = {
            "the in-tree fallback of src/transitionlistener/__init__.py":
                _search(ROOT / "src" / "transitionlistener" / "__init__.py",
                        r'^    __version__ = "([^"]+)"'),
            "conda/meta.yaml": _search(ROOT / "conda" / "meta.yaml",
                                       r'^\{% set version = "([^"]+)" %\}'),
            "CITATION.cff": _search(ROOT / "CITATION.cff", r'^version: "([^"]+)"'),
        }
        for where, value in found.items():
            with self.subTest(where=where):
                self.assertEqual(value, self.declared,
                                 "%s states %s where pyproject.toml declares %s"
                                 % (where, value, self.declared))

    def test_the_newest_changelog_section_is_the_declared_version(self) -> None:
        heading = _search(ROOT / "CHANGELOG.md", r"^## \[([^\]]+)\]")
        self.assertEqual(
            heading, self.declared,
            "the newest CHANGELOG.md section is [%s] where pyproject.toml declares %s. "
            "Close the Unreleased section as part of the release." % (heading, self.declared))

    def test_the_documentation_release_is_the_declared_series(self) -> None:
        release = _search(ROOT / "docs" / "source" / "conf.py", r"^release = '([^']+)'")
        major, minor, _ = self.declared.split(".")
        self.assertEqual(release, "%s.%s" % (major, minor))


if __name__ == "__main__":
    unittest.main()
