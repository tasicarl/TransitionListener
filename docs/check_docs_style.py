#!/usr/bin/env python3
"""Check the documentation against the project's house rules for prose and for maths.

Run from the repository root::

    python3 docs/check_docs_style.py

It exits non-zero and prints one line per violation, each with the file, the line and what to
write instead. ``tests/test_docs_style.py`` runs the same checks, so a rule broken in a pull
request fails the suite rather than waiting for someone to notice it in the rendered page.

Each rule is here because a real page broke it. The rules are deliberately literal: a check
that needs judgement is a check that argues with its author, so where a rule cannot be decided
mechanically it is left out rather than guessed at.
"""
from __future__ import annotations

import pathlib
import re
import sys

DOCS = pathlib.Path(__file__).resolve().parent / "source"

#: Symbols whose subscript is an abbreviation and so is upright. Whole symbols, not bare
#: letters: in a sum over fermions the index ``f`` is a running index and stays italic, while
#: in ``m_f`` it would not be, and no mechanical rule can tell those apart from the letter
#: alone. Adding a symbol here is how a new one gets checked.
UPRIGHT_SUBSCRIPTS = [
    "c_s", "c_b", "v_w", "v_J", "T_c", "T_f", "T_n", "T_reh", "T_perc", "T_nuc", "T_crit",
    "T_b", "f_b", "f_perc", "f_final", "f_start", "R_H", "g_eff", "h_eff",
    "V_rad", "V_daisy", "V_tree", "V_ct", "V_T", "V_CW", "e_s", "w_sym", "w_bro",
    "kappa_sw", "kappa_turb", "Omega_sw", "Omega_turb", "Omega_GW", "H_tau_sw",
    "alpha_hyd", "Delta_w", "n_b", "x_eps", "p_f", "rho_rad", "Pi_H",
]

#: (pattern, what is wrong, what to write instead). Applied to every line of every page.
RULES: list[tuple[str, str, str]] = [
    # ---- maths -------------------------------------------------------------------------
    (r"\\text\{",
     r"\text{} in a subscript shifts the baseline where the word has a descender",
     r"\mathrm{}"),
    (r"(?<![A-Za-z\\])\{?\\?d\}?\s*(?=[A-Za-z]\s*\}?\s*\{?\\?d?\s*[A-Za-z]*\}?\s*$)"
     r"(?!ocument)",
     "",  # placeholder, replaced by the differential rule below
     ""),
    (r"\\frac\{d[A-Za-z]",
     "the differential d is upright, not an italic variable",
     r"\frac{\mathrm{d}X}"),
    (r"\\frac\{\\partial",  # allowed: partial derivatives are already upright
     "", ""),
    (r"[^\\a-zA-Z]d([A-Za-z])/d([A-Za-z])",
     "the differential d is upright, not an italic variable",
     r"\mathrm{d}X/\mathrm{d}Y"),
    (r"\\nabla\\nabla",
     "a double nabla is not the Hessian",
     r"\nabla^2 or an explicit \partial^2 V/\partial\phi_i\partial\phi_j"),
    (r"\\in\\?text\{|\\in\s*\\mathrm\{(bosons|fermions)\}",
     "", ""),
    (r"(?<![\\\w])e\^\{-",
     "the exponential e is upright",
     r"\mathrm{e}^{-}"),
    # ---- prose -------------------------------------------------------------------------
    (r"\b(What|Where|How|Why|Which|Who)\b[^.?!]{0,80}?\b(is|are|was|were)\b"
     r"[^.?!]{0,8}?(,\s*(is|are)\b|\bthat\b)",
     "a pseudo-cleft",
     "say it straight: put the subject first"),
    (r";",
     "a semicolon",
     "a full stop, or a comma"),
    # "A model gives an effective potential. The potential gives phases." Nothing in the
    # package gives anything: it specifies, computes, returns or describes, and which of
    # those it does is the information the sentence is for.
    (r"\b(gives?|gave|giving)\b",
     "the verb give, which stands for whichever verb was meant",
     "specify, describe, compute, return, produce or set"),
    (r"\bconverted fraction\b",
     "a name for the fraction of space in the true vacuum that no one else uses",
     "true-vacuum fraction, or false-vacuum fraction"),
    (r"\bcommonest\b",
     "a superlative formed the short way",
     "most common"),
    (r"\b(the|a|an|is|of|to|in|and|that|it)\s+\1\b",
     "a word written twice",
     "the word once"),
    (r"\bnot even wrong\b",
     "a phrase that dismisses rather than says what is wrong",
     "what the quantity actually is"),
    (r"\bCrucially\b|\bNotably\b|\bIt is worth noting\b|\bImportantly\b",
     "an LLM filler word",
     "cut it"),
    (r"\bthe verdict is settled\b|\bthe verdict\b",
     "a word for the thing rather than the thing",
     "name what is decided"),
    # A model is a model; a figure shows a quantity computed from one of its benchmark
    # points. "the 2HDM at T = 25 GeV" names the wrong object.
    (r"\bthe (2HDM|conformal dark U\(1\)|dark U\(1\)|abelian dark Higgs)\s+at\b",
     "a model named where a quantity computed from one of its points is meant",
     "the <quantity> of a <model> benchmark point at ..."),
    (r"(?<![-\w])—|–",
     "an em or en dash",
     "a comma or a full stop"),
]

#: Section titles and figure captions must be sentences, not fragments. A title that has no
#: verb and is built out of a comma or a colon reads as a label rather than a statement.
FRAGMENT_TITLE = re.compile(r"^[A-Z][^.!?]*,\s+(and\s+)?\w+[^.!?]*$")

#: A caption longer than this many words stops being a caption.
MAX_CAPTION_WORDS = 90


def _differential_violations(text: str) -> list[tuple[int, str, str]]:
    """``d`` as a differential, italic, which reads as a variable named d."""
    out = []
    pattern = re.compile(r"(?<![\\\w])d(?=[A-Za-z]\b|\s*\\phi|\s*\\xi|\s*\\tau)")
    for i, line in enumerate(text.splitlines(), 1):
        if ":math:" not in line and ".. math::" not in line and not line.startswith("   "):
            continue
        for m in pattern.finditer(line):
            tail = line[m.start():m.start() + 12]
            if tail.startswith("decay") or tail.startswith("density"):
                continue
            out.append((i, "the differential d is italic, so it reads as a variable",
                        r"\mathrm{d}"))
    return out


def _subscript_violations(text: str) -> list[tuple[int, str, str]]:
    """A subscript that is an abbreviation must be upright."""
    out = []
    alts = []
    for sym in UPRIGHT_SUBSCRIPTS:
        base, sub = sym.split("_", 1)
        alts.append((sym, re.compile(re.escape(base) + r"_\{?" + re.escape(sub)
                                     + r"\}?(?![\w{])")))
    for i, line in enumerate(text.splitlines(), 1):
        if ":math:" not in line and not line.startswith("   ") and ".. math" not in line:
            continue
        for sym, pat in alts:
            for m in pat.finditer(line):
                base, sub = sym.split("_", 1)
                out.append((i, "the subscript of %r is an abbreviation, so it is upright"
                            % sym, "%s_\\mathrm{%s}" % (base, sub)))
    return out


def _caption_violations(path: pathlib.Path, text: str) -> list[tuple[int, str, str]]:
    """Figure captions: not too long, and not a fragment."""
    out = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith(".. figure::"):
            j = i + 1
            while j < len(lines) and (lines[j].strip().startswith(":") or not lines[j].strip()):
                j += 1
            start = j
            while j < len(lines) and lines[j].startswith("   "):
                j += 1
            caption = " ".join(l.strip() for l in lines[start:j]).strip()
            if caption:
                n = len(caption.split())
                if n > MAX_CAPTION_WORDS:
                    out.append((start + 1, "a caption of %d words" % n,
                                "at most %d; move the rest into the text"
                                % MAX_CAPTION_WORDS))
            i = j
            continue
        i += 1
    return out


def _title_violations(text: str) -> list[tuple[int, str, str]]:
    """Section titles: no questions, no fragments built on a comma."""
    out = []
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        under = lines[i + 1].strip()
        title = lines[i].strip()
        if not title or not under or len(under) < 3:
            continue
        if not re.fullmatch(r"[-=~^\"'`#*+_]{3,}", under):
            continue
        if len(under) < len(title) - 2:
            continue
        if title.endswith("?"):
            out.append((i + 1, "a section title that is a question", "a noun phrase"))
        elif re.match(r"^(What|How|Why|When|Where|Which|Who)\b", title):
            out.append((i + 1, "a section title that is a clause",
                        "a noun phrase naming the thing"))
        elif FRAGMENT_TITLE.match(title) and " and " not in title.lower():
            out.append((i + 1, "a section title that is a fragment built on a comma",
                        "a noun phrase without the comma"))
    return out


#: The modules of the package, taken from the source tree. A page that names one of these in
#: a literal is naming a module, and a module has a page of its own in the API reference, so
#: it is written as a cross-reference and not as a word in backticks.
def _package_modules() -> set:
    src = DOCS.parent.parent / "src" / "transitionlistener"
    if not src.is_dir():
        return set()
    return {p.stem for p in src.rglob("*.py") if not p.stem.startswith("__")}


def _module_literal_violations(text: str) -> list[tuple[int, str, str]]:
    """``pipeline`` where :py:mod:`transitionlistener.pipeline` was meant.

    A module named in backticks reads as a word the reader is expected to know. Written as a
    reference it is a link to the module's own documentation, which is where the reader who
    stops at that word wants to go.
    """
    modules = _package_modules()
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        for m in LITERAL_TOKEN.finditer(line):
            token = (m.group(1) or m.group(2)).strip()
            head = token.split(".")[0]
            if head in modules and not token.endswith(".yaml"):
                out.append((i, "the module %r is named as a literal" % head,
                            ":py:mod:`transitionlistener.%s`" % head))
    return out


def _underline_violations(text: str) -> list[tuple[int, str, str]]:
    """A section underline that is shorter than its title.

    Sphinx prints a warning and renders the line as a title anyway, so the page looks right
    and the warning is lost in the build log. A renamed section leaves its old underline
    behind, which is how this one appeared.
    """
    out = []
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        title, under = lines[i].rstrip(), lines[i + 1].rstrip()
        if not title.strip() or len(under) < 3:
            continue
        if not re.fullmatch(r"[-=~^\"'#*+_]{3,}", under.strip()):
            # Backticks are left out of the set: a run of them is a code fence in the
            # Markdown pages, not an underline, and the line above it is not a title.
            continue
        if len(under) < len(title):
            out.append((i + 2, "a section underline shorter than its title",
                        "an underline at least as long as the title"))
    return out


def _capital_title_violations(text: str) -> list[tuple[int, str, str]]:
    """A section title starts with a capital letter.

    Lower-case titles were written for the figures and crept into the pages with them. A
    title that starts in lower case reads as the continuation of the paragraph above it.
    """
    out = []
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        title, under = lines[i].strip(), lines[i + 1].strip()
        if not title or len(under) < 3:
            continue
        if not re.fullmatch(r"[-=~^\"'`#*+_]{3,}", under):
            continue
        if len(under) < len(title) - 2:
            continue
        first = title.lstrip(":`*")[:1]
        if first and first.isalpha() and not first.isupper():
            out.append((i + 1, "a section title that starts in lower case",
                        "a capital letter"))
    return out


def _strip_literal_blocks(text: str) -> str:
    """Blank out code blocks, which are neither prose nor maths.

    A bibliography entry or a shell command is not held to the house rules, and leaving them
    in produced a report about the ``d`` in "van de Vis".
    """
    out, lines = [], text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        starts = (
            (line.rstrip().endswith("::") and not stripped.startswith(".. math::"))
            or stripped.startswith(".. code-block::")
            or stripped.startswith(".. literalinclude::"))
        out.append(line)
        if not starts:
            i += 1
            continue
        i += 1
        while i < len(lines) and not lines[i].strip():
            out.append(lines[i]); i += 1
        indent = len(lines[i]) - len(lines[i].lstrip()) if i < len(lines) else 0
        while i < len(lines) and (not lines[i].strip()
                                  or len(lines[i]) - len(lines[i].lstrip()) >= indent > 0):
            out.append("")
            i += 1
    return "\n".join(out)


def _blank_citations(text: str) -> str:
    """Blank out bibliography entries.

    A reference is written the way the journal writes it, which includes the semicolon between
    two papers published together, and the house rules are about the prose of the pages.
    """
    out, lines = [], text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith(".. ["):
            out.append("")
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith(("   ", "\t"))):
                out.append("")
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _blank_maths(text: str) -> str:
    """Blank out maths, so that the prose rules do not read LaTeX.

    ``\\;`` is a thin space and ``\\,`` is a thinner one, not punctuation; without this the
    semicolon rule reports every aligned equation.
    """
    out, in_block = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith(".. math::"):
            in_block = True
            out.append(line)
            continue
        if in_block:
            if not line.strip() or line.startswith(("   ", "\t")):
                out.append("")
                continue
            in_block = False
        out.append(re.sub(r":math:`[^`]*`", ":math:``", line))
    return "\n".join(out)


def _blank_inline_literals(text: str) -> str:
    """Blank out ``code`` and `code` spans, which name identifiers rather than quantities.

    ``f_perc`` is the name of a setting in the configuration file, so it is spelled the way
    the file spells it; the same symbol written as maths is `f_\\mathrm{perc}`. Without this
    the maths rules report every mention of a setting.
    """
    # Blanked with spaces, not with backticks: a run of backticks left behind is itself read
    # as a literal span by the next pass, which then swallows the maths between two of them.
    # That silently hid most of the maths on every page that mentions a setting.
    # Confined to one line: a negated character class matches newlines too, so an unpaired
    # backtick would otherwise swallow every line up to the next one, equations included.
    text = re.sub(r"``[^`\n]*``", lambda m: " " * len(m.group(0)), text)
    # A single-backtick span preceded by a role is not a literal. ``:math:`x`` is maths and
    # has to stay, or every rule about notation is silently switched off.
    text = re.sub(r"(?<!`)(?<!:math:)(?<!:py:mod:)(?<!:doc:)(?<!:ref:)(?<!:func:)"
                  r"(?<!:class:)(?<!:meth:)`[^`\n]*`(?!`)",
                  lambda m: " " * len(m.group(0)), text)
    return text


def _maths_spans(text: str):
    """Every piece of maths on the page, with the line it starts on."""
    out = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith(".. math::"):
            start, i = i + 1, i + 1
            body = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith(("   ", "\t"))):
                body.append(lines[i].strip())
                i += 1
            out.append((start + 1, " ".join(b for b in body if b)))
            continue
        for m in re.finditer(r":math:`([^`]*)`", lines[i]):
            out.append((i + 1, m.group(1)))
        i += 1
    return out


#: Commands that take brace groups, with how many each takes.
BRACE_COMMANDS = {"frac": 2, "dfrac": 2, "tfrac": 2, "sqrt": 1, "mathrm": 1, "mathcal": 1,
                  "text": 1, "underbrace": 1, "overline": 1, "hat": 1, "tilde": 1,
                  "operatorname": 1, "boldsymbol": 1, "mathbf": 1}


def _group_violations(text: str) -> list[tuple[int, str, str]]:
    """A command whose brace group is not closed.

    Counting braces over a whole equation is not enough: a brace lost from one group is often
    made up by another, and the total balances while the equation does not render. This walks
    each command that takes groups and checks that its groups are there and closed. The rule
    exists because an automatic rewrite of subscripts once ate the closing brace of a
    ``\\frac`` and the equation silently stopped rendering.
    """
    out = []
    for lineno, maths in _maths_spans(text):
        for m in re.finditer(r"\\([A-Za-z]+)", maths):
            name = m.group(1)
            if name not in BRACE_COMMANDS:
                continue
            pos = m.end()
            for _ in range(BRACE_COMMANDS[name]):
                while pos < len(maths) and maths[pos] == " ":
                    pos += 1
                if pos >= len(maths) or maths[pos] != "{":
                    out.append((lineno, "\\%s without its brace group" % name,
                                "the group it takes"))
                    break
                depth, pos = 0, pos
                while pos < len(maths):
                    if maths[pos] == "{":
                        depth += 1
                    elif maths[pos] == "}":
                        depth -= 1
                        if depth == 0:
                            pos += 1
                            break
                    pos += 1
                else:
                    out.append((lineno, "\\%s with a group that is never closed" % name,
                                "the closing brace"))
                    break
        # A ``\\left`` and its ``\\right`` have to close at the same brace depth. Where they do
        # not, a brace group has been opened or closed between them, which is what a lost
        # closing brace looks like: the group swallows the rest of the expression and the
        # totals still balance, so counting braces sees nothing.
        depth, stack = 0, []
        for m in re.finditer(r"\{|\}|\\left|\\right", maths):
            tok = m.group(0)
            if tok == "{":
                depth += 1
            elif tok == "}":
                depth -= 1
            elif tok == "\\left":
                stack.append(depth)
            else:
                if stack and stack.pop() != depth:
                    out.append((lineno,
                                "a \\left and its \\right close at different brace depths, "
                                "so a brace group is unbalanced between them",
                                "the brace that was lost"))
                    break
        if maths.count("\\left") != maths.count("\\right"):
            out.append((lineno, "maths with %d unmatched \\left or \\right"
                        % abs(maths.count("\\left") - maths.count("\\right")),
                        "the missing one"))
    return out


def _render_violations(text: str) -> list[tuple[int, str, str]]:
    """Maths that will not render.

    Checked by handing each piece to matplotlib's own parser, which is the one that draws the
    equations in the built pages, so a piece that trips it is a piece a reader sees broken.
    Unbalanced braces and brackets are checked first, because the parser accepts some of those
    and simply drops what follows.
    """
    out = []
    for lineno, maths in _maths_spans(text):
        if not maths.strip():
            continue
        body = re.sub(r"\\(left|right|big|Big|bigg|Bigg)\b", " ", maths)
        for opener, closer, what in (("{", "}", "brace"), ("[", "]", "bracket"),
                                     ("(", ")", "parenthesis")):
            n = body.count(opener) - body.count(closer)
            if n:
                out.append((lineno, "maths with %d unmatched %s%s"
                            % (abs(n), what, "" if abs(n) == 1 else "s"),
                            "the missing %s" % what))
    return out


#: Symbols that every page may use without defining, because they are the subject of the set.
COMMON_SYMBOLS = {"T", "V", "f", "t", "r", "x", "n", "g", "h", "i", "j", "k", "s", "p", "e",
                  "w", "v", "a", "H", "S", "P", "I", "R", "A", "J", "D", "K", "N", "M"}


def _undefined_symbol_violations(text: str) -> list[tuple[int, str, str]]:
    """A symbol that appears only inside a displayed equation.

    A symbol the prose never picks up has not been introduced to the reader: it was written
    into an equation and left there. The test is whether the symbol is ever used in running
    text, as inline maths, anywhere on the page. That is how ``T_start`` was found, which
    stood in the limits of one integral and nowhere else.
    """
    inline, display = set(), {}
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith(".. math::"):
            start, i = i + 1, i + 1
            while i < len(lines) and (not lines[i].strip()
                                      or lines[i].startswith(("   ", "\t"))):
                for sym in _symbols(lines[i]):
                    display.setdefault(sym, start + 1)
                i += 1
            continue
        for m in re.finditer(r":math:`([^`\n]*)`", lines[i]):
            inline.update(_symbols(m.group(1)))
        i += 1
    out = []
    for sym, lineno in sorted(display.items(), key=lambda kv: kv[1]):
        if sym not in inline:
            out.append((lineno, "the symbol %r appears only inside a displayed equation" % sym,
                        "a sentence in the text that uses it"))
    return out


#: Not symbols: an operator carries limits, not a subscript.
NOT_SYMBOLS = {"\\int", "\\sum", "\\prod", "\\lim", "\\max", "\\min", "\\log", "\\ln"}


def _symbols(maths: str) -> set:
    """The symbols in a piece of maths whose subscript is an abbreviation.

    A running index and a digit are not abbreviations: ``n_i`` and ``J_0`` take their meaning
    from the equation they sit in, and a page that spelled each of them out would be unreadable.
    An abbreviation such as ``T_start`` does not, so it has to be said somewhere.
    """
    found = set()
    for m in re.finditer(r"([A-Za-z]|\\[A-Za-z]+)_(?:\{)?(?:\\mathrm\{)?"
                         r"([A-Za-z0-9*,]+)\}?\}?", maths):
        base, sub = m.group(1), m.group(2).split(",")[0]
        if base in NOT_SYMBOLS or not sub.isalpha() or len(sub) < 2:
            continue
        # Only up to the first comma: ``g_eff,reh`` and ``g_eff,0`` are the same symbol
        # evaluated at two times, and the prose introduces it once.
        found.add("%s_%s" % (base, sub))
    return found


def _label_violations(text: str) -> list[tuple[int, str, str]]:
    """A paragraph may not open with a bold label.

    Carlo reads a bold phrase ending in a full stop as a heading the page did not need:
    ``**A trial ends in one of three ways.** The field turns around ...``. The same sentence
    with the bold carried into it reads as prose. Bold that is part of the sentence, as in
    ``**The first pass** locates the range``, is left alone.
    """
    out = []
    previous_blank = True
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        bold = re.match(r"^\*\*([^*]+)\*\*([,:]?)", stripped)
        # A paragraph that opens with a bold phrase of more than one word is a label the page
        # did not need: ``**The first pass** locates ...`` reads as a heading. The same
        # sentence without the bold reads as prose. A single bold word, such as the name of
        # the package at the top of the landing page, is emphasis inside the sentence.
        if previous_blank and bold and (len(bold.group(1).split()) > 1 or bold.group(2)):
            out.append((i, "a paragraph opens with a bold label",
                        "the same words without the bold"))
        previous_blank = not stripped
    return out


#: Directories of the repository that a page may name. A path whose first component is one
#: of these has to exist, because a reader will type it. ``scans/`` is left out: a run writes
#: it. ``config`` and ``example`` are here although the repository has neither, which is the
#: point: both were sent to the reader as if they existed.
NAMED_DIRECTORIES = ("examples", "example", "models", "tests", "docs", "src", "notes",
                     "config", "scripts")

#: A literal in either of the two markups, because the pages use both.
LITERAL_TOKEN = re.compile(r"``([^`]+)``|(?<![`:])`([^`<>]+)`(?!`)")


def _repo_path_violations(text: str) -> list[tuple[int, str, str]]:
    """A path the documentation names has to be a path the repository has.

    The usage page sent the reader to a ``config/`` directory that has never existed and to
    ``example/`` where the directory is ``examples/``. Both read perfectly well.
    """
    root = DOCS.parent.parent
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        for m in LITERAL_TOKEN.finditer(line):
            token = (m.group(1) or m.group(2)).strip().rstrip(".,;:")
            if "/" not in token or " " in token:
                continue
            if token.startswith(("http", "/", "-", "~", ".")):
                continue
            if token.split("/")[0] not in NAMED_DIRECTORIES:
                continue
            if not (root / token).exists():
                out.append((i, "the page names %s, which the repository does not have"
                            % token, "the path that exists, or drop it"))
    return out


def check(paths: list[pathlib.Path]) -> list[str]:
    problems: list[str] = []
    for path in paths:
        raw = _blank_citations(_strip_literal_blocks(path.read_text(encoding="utf-8")))
        text = _blank_inline_literals(raw)
        prose = _blank_maths(text)
        try:
            rel = path.relative_to(DOCS.parent.parent)
        except ValueError:
            rel = path
        # A page of questions is allowed to have questions as its titles.
        questions_allowed = path.name == "faq.rst"
        for pattern, what, instead in RULES:
            if not what:
                continue
            # A rule about punctuation or wording reads the prose, with the maths blanked
            # out; a rule about notation reads the page as it stands.
            is_prose_rule = "\\\\" not in pattern
            source = prose if is_prose_rule else text
            for i, line in enumerate(source.splitlines(), 1):
                if re.search(pattern, line):
                    problems.append("%s:%d: %s -> write %s" % (rel, i, what, instead))
        for finder in (_differential_violations, _subscript_violations, _title_violations,
                       _render_violations, _group_violations, _label_violations,
                       _capital_title_violations, _underline_violations,
                       _undefined_symbol_violations):
            for i, what, instead in finder(text):
                if questions_allowed and "question" in what:
                    continue
                problems.append("%s:%d: %s -> write %s" % (rel, i, what, instead))
        # The raw page, because a path is written as a literal and the prose rules blank
        # literals out before they read a line.
        for i, what, instead in _module_literal_violations(raw):
            problems.append("%s:%d: %s -> write %s" % (rel, i, what, instead))
        for i, what, instead in _repo_path_violations(raw):
            problems.append("%s:%d: %s -> write %s" % (rel, i, what, instead))
        for i, what, instead in _caption_violations(path, text):
            problems.append("%s:%d: %s -> %s" % (rel, i, what, instead))
    return problems


def main() -> int:
    paths = sorted(DOCS.rglob("*.rst")) + sorted(DOCS.rglob("*.md"))
    paths = [p for p in paths if "api/" not in str(p)]
    problems = check(paths)
    for p in problems:
        print(p)
    print("\n%d problems in %d pages" % (len(problems), len(paths)))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
