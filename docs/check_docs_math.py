"""Compile every equation in the documentation with LaTeX.

The pages are rendered by MathJax, which fails silently: an equation with a stray brace or
an unknown command is dropped from the page and the reader sees the source or nothing at all.
Two such equations reached a review. LaTeX accepts a superset of what MathJax accepts, so an
equation LaTeX refuses is one the page will not show, and that is what this module tests.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage{amsmath,amssymb}
\begin{document}
"""


def pages() -> list[Path]:
    return sorted(p for p in SOURCE.rglob("*.rst"))


def equations(text: str) -> list[tuple[int, str]]:
    """Every display and inline equation, as (line number, LaTeX body)."""
    found: list[tuple[int, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^(\s*)\.\.\s+math::\s*(.*)$", lines[i])
        if m:
            indent, first = m.group(1), m.group(2).strip()
            start = i + 1
            body = [first] if first else []
            i += 1
            while i < len(lines):
                line = lines[i]
                if not line.strip():
                    body.append("")
                    i += 1
                    continue
                if len(line) - len(line.lstrip()) <= len(indent):
                    break
                if re.match(r"^\s*:\w+:", line):
                    i += 1
                    continue
                body.append(line.strip())
                i += 1
            # A blank line inside a math directive starts a second equation.
            chunk: list[str] = []
            lineno = start
            for part in body + [""]:
                if part:
                    chunk.append(part)
                elif chunk:
                    found.append((lineno, " ".join(chunk)))
                    chunk = []
            continue
        for m in re.finditer(r":math:`([^`]*)`", lines[i]):
            found.append((i + 1, m.group(1)))
        i += 1
    return found


def _compile(bodies: list[str], workdir: Path) -> bool:
    doc = PREAMBLE + "\n".join(r"\[" + b + r"\]" for b in bodies) + "\n\\end{document}\n"
    (workdir / "eq.tex").write_text(doc)
    run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "eq.tex"],
                         cwd=workdir, capture_output=True, text=True)
    return run.returncode == 0


def violations() -> list[str]:
    if shutil.which("pdflatex") is None:
        return []
    out: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for page in pages():
            found = equations(page.read_text())
            if not found:
                continue
            if _compile([b for _, b in found], work):
                continue
            # The page as a whole fails, so name the equations that fail on their own.
            for lineno, body in found:
                if not _compile([body], work):
                    out.append("%s:%d: LaTeX refuses this equation, so MathJax will not "
                               "render it: %s" % (page.relative_to(SOURCE), lineno, body[:90]))
    return out


if __name__ == "__main__":
    problems = violations()
    for p in problems:
        print(p)
    print("%d unrenderable equations in %d pages" % (len(problems), len(pages())))
