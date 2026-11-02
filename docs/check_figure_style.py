#!/usr/bin/env python3
"""Check a matplotlib figure against the project's house rules for figures.

``make_algorithm_figures.py`` calls :func:`check_figure` on every figure it draws, so a figure
that breaks a rule is reported as it is built rather than after someone looks at it. The rules
are the ones that have actually been got wrong here:

* no text sits on top of a line, and every text has a margin around it that nothing crosses;
* no text sits on top of other text;
* the legend does not sit on top of a line;
* no text is clipped by the edge of the panel;
* a leader that has to cross a line crosses it steeply, not at a glancing angle;
* every curve runs to an edge of the panel rather than stopping inside it.

The checker rasterises each panel once and works on pixels, so it sees what a reader sees and
not what the author intended.
"""
from __future__ import annotations

import io

import numpy as np

#: A text's invisible margin, in points. Nothing may cross it.
TEXT_PAD_PT = 3.0

#: The fraction of a text's padded box that may be covered by ink before it counts as sitting
#: on something. One curve of the usual width crossing a two-line label covers about two per
#: cent of it, which is plainly visible, so the threshold is well below that and leaves only
#: the stray antialiased pixel.
MAX_BUSY_FRACTION = 0.004

#: A leader crossing a curve below this angle, in degrees, runs alongside it.
MIN_CROSSING_ANGLE_DEG = 35.0

#: How close to the panel edge a curve has to end to count as leaving it, as a fraction.
EDGE_TOLERANCE = 0.02


def _render(fig) -> tuple[np.ndarray, float]:
    """The figure as an alpha mask, and the dots per point."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    buf.seek(0)
    import matplotlib.image as mpimg

    img = mpimg.imread(buf)
    ink = img[..., :3].min(axis=2) < 0.97 if img.shape[-1] >= 3 else img < 0.97
    return ink, fig.dpi / 72.0


def _text_boxes(fig, renderer):
    """Every text in the figure, with its padded pixel box."""
    out = []
    for ax in fig.get_axes():
        texts = list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        texts += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        for t in texts:
            if not t.get_text().strip() or not t.get_visible():
                continue
            try:
                bb = _text_only_extent(t, renderer)
            except Exception:
                continue
            out.append((t, ax, bb))
    return out


def _text_only_extent(t, renderer):
    """The box of a label's words, without its leader.

    ``Annotation.get_window_extent`` covers the arrow as well, so for an annotation it returns
    a box stretching along the whole leader, and every curve the leader passes then counts as
    ink under the label. Calling the base class's method gives the words alone.
    """
    import matplotlib.text as mtext

    if isinstance(t, mtext.Annotation):
        return mtext.Text.get_window_extent(t, renderer=renderer)
    return t.get_window_extent(renderer=renderer)


def check_figure(fig, name: str = "figure") -> list[str]:
    """Return one message per rule the figure breaks, empty if it is clean."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems: list[str] = []
    height = fig.bbox.height
    pad = TEXT_PAD_PT * fig.dpi / 72.0

    ink, _ = _render(fig)
    # For the text-on-ink rule the leaders are rendered away rather than masked out. Masking
    # out a leader's bounding box also hides every curve inside that box, which is a large
    # rectangle for any leader that reaches across the panel: the rule then passes a text that
    # a curve runs straight through. This was got wrong once and is the reason for the second
    # render.
    boxes = _text_boxes(fig, renderer)

    leaders = [a for ax in fig.get_axes() for t in ax.texts
               if (a := getattr(t, "arrow_patch", None)) is not None]
    leader_vis = [a.get_visible() for a in leaders]
    for a in leaders:
        a.set_visible(False)
    texts = [t for ax in fig.get_axes() for t in ax.texts]
    text_vis = [t.get_visible() for t in texts]
    for t in texts:
        t.set_visible(False)
    ink_background, _ = _render(fig)
    for t, v in zip(texts, text_vis):
        t.set_visible(v)
    for a, v in zip(leaders, leader_vis):
        a.set_visible(v)
    fig.canvas.draw()

    # --- a text must not sit on ink, counting its margin --------------------------------
    for t, ax, bb in boxes:
        if t in (ax.xaxis.label, ax.yaxis.label) or t in ax.get_xticklabels() \
                or t in ax.get_yticklabels() or t is ax.title:
            continue
        # A white patch under a label is not a way past this rule. It hides whatever is
        # behind it, so the curve that was there is now missing from the figure, which is the
        # same defect seen from the other side. Labels go in clear space.
        x0 = int(max(0, bb.x0 - pad))
        x1 = int(min(ink.shape[1], bb.x1 + pad))
        y0 = int(max(0, height - bb.y1 - pad))
        y1 = int(min(ink.shape[0], height - bb.y0 + pad))
        if x1 <= x0 or y1 <= y0:
            continue
        # Measured against a render with every label and leader taken out, so that what is
        # counted is ink from the figure itself and nothing else. Subtracting the text's own
        # box from a full render instead leaves the antialiased edge of the glyphs in the
        # margin, which at a strict threshold reports a label standing in empty space.
        patch = ink_background[y0:y1, x0:x1]
        busy = patch.sum() / max(1, patch.size)
        if busy > MAX_BUSY_FRACTION:
            import os
            if os.environ.get("TL_FIGURE_DEBUG"):
                ys, xs = np.nonzero(patch)
                print("      DEBUG %r box x[%d,%d] y[%d,%d]; ink at x %s y %s"
                      % (t.get_text()[:20], x0, x1, y0, y1,
                         (xs.min() + x0, xs.max() + x0) if xs.size else None,
                         (ys.min() + y0, ys.max() + y0) if ys.size else None))
            problems.append("%s: text %r sits on ink, %.1f%% of its padded box"
                            % (name, t.get_text()[:34], 100 * busy))

    # --- texts must not sit on each other -----------------------------------------------
    # Boxes that merely touch are not an overlap: the tick labels at a corner of a panel
    # share an edge by construction, and reporting those would train the reader of this
    # report to ignore it.
    for i, (t1, _, b1) in enumerate(boxes):
        for t2, _, b2 in boxes[i + 1:]:
            if not b1.overlaps(b2):
                continue
            w = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            h = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if w <= 0 or h <= 0:
                continue
            smaller = min(b1.width * b1.height, b2.width * b2.height)
            if smaller > 0 and (w * h) / smaller > 0.20:
                problems.append("%s: text %r and text %r overlap"
                                % (name, t1.get_text()[:22], t2.get_text()[:22]))

    # --- the legend must not sit on a line ----------------------------------------------
    for ax in fig.get_axes():
        leg = ax.get_legend()
        if leg is None:
            continue
        bb = leg.get_window_extent(renderer=renderer)
        x0, x1 = int(bb.x0), int(bb.x1)
        y0, y1 = int(height - bb.y1), int(height - bb.y0)
        if x1 <= x0 or y1 <= y0:
            continue
        axbb = ax.get_window_extent(renderer=renderer)
        for line in ax.get_lines():
            pts = ax.transData.transform(np.column_stack(
                [np.asarray(line.get_xdata(), dtype=float),
                 np.asarray(line.get_ydata(), dtype=float)]))
            # Only the part of the curve that is actually drawn. A curve leaves the panel at
            # its limits and is clipped there; its points beyond that land wherever the
            # transform puts them, which can be inside a legend drawn outside the panel.
            drawn = ((pts[:, 0] >= axbb.x0) & (pts[:, 0] <= axbb.x1)
                     & (pts[:, 1] >= axbb.y0) & (pts[:, 1] <= axbb.y1))
            inside = (drawn & (pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                      & (height - pts[:, 1] >= y0) & (height - pts[:, 1] <= y1))
            if inside.any():
                problems.append("%s: the legend sits on the curve %r"
                                % (name, line.get_label()[:30]))
                break

    # --- a legend belongs inside its panel ------------------------------------------------
    for ax in fig.get_axes():
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(renderer=renderer)
        ab = ax.get_window_extent(renderer=renderer)
        if (lb.x0 < ab.x0 - 1 or lb.x1 > ab.x1 + 1
                or lb.y0 < ab.y0 - 1 or lb.y1 > ab.y1 + 1):
            problems.append("%s: the legend is not inside its panel" % name)

    # --- nothing the author placed may be clipped by the figure --------------------------
    # Tick labels are placed by matplotlib and the outermost of them legitimately reach the
    # edge; what this rule is about is a label put somewhere it does not fit.
    placed = {id(t) for ax in fig.get_axes() for t in ax.texts}
    for t, ax, bb in boxes:
        if id(t) not in placed:
            continue
        fb = fig.bbox
        if bb.x0 < fb.x0 - 0.5 or bb.x1 > fb.x1 + 0.5 \
                or bb.y0 < fb.y0 - 0.5 or bb.y1 > fb.y1 + 0.5:
            problems.append("%s: text %r is clipped by the figure edge"
                            % (name, t.get_text()[:30]))

    # --- a label placed in data coordinates belongs inside its panel ----------------------
    # Clipping against the figure is not enough: a label whose anchor is outside the axis
    # limits floats in the margin between panels, which looks deliberate and is not.
    for ax in fig.get_axes():
        axbb = ax.get_window_extent(renderer=renderer)
        for t in ax.texts:
            if not t.get_text().strip() or t.get_transform() is not ax.transData:
                continue
            bb = _text_only_extent(t, renderer)
            if (bb.x1 < axbb.x0 or bb.x0 > axbb.x1
                    or bb.y1 < axbb.y0 or bb.y0 > axbb.y1):
                problems.append("%s: the label %r is outside its panel"
                                % (name, t.get_text().splitlines()[0][:30]))

    # --- a leader crosses a curve steeply or not at all ----------------------------------
    for ax in fig.get_axes():
        for ann in ax.texts:
            arrow = getattr(ann, "arrow_patch", None)
            if arrow is None or ann.xy is None:
                continue
            try:
                tip = np.array(ax.transData.transform(ann.xy), dtype=float)
                tail = np.array(ax.transData.transform(
                    ann.get_position() if ann.xycoords == "data"
                    else ax.transData.inverted().transform(
                        ann.get_window_extent(renderer=renderer).p0)), dtype=float)
            except Exception:
                continue
            # The leader's own path, so that an elbow is tested as the two straight pieces it
            # is drawn as and not as the diagonal between its ends, which crosses things the
            # leader never goes near.
            try:
                verts = arrow.get_path().vertices
                leader = arrow.get_patch_transform().transform(verts)
                if leader.shape[0] < 2 or not np.isfinite(leader).all():
                    raise ValueError
            except Exception:
                leader = np.vstack([tail, tip])
            if np.hypot(*(tip - tail)) <= 1.0:
                continue
            flagged = False
            for line in ax.get_lines():
                xs = np.asarray(line.get_xdata(), dtype=float)
                ys = np.asarray(line.get_ydata(), dtype=float)
                if xs.size < 2:
                    continue
                pts = ax.transData.transform(np.column_stack([xs, ys]))
                seg = np.diff(pts, axis=0)
                for k in range(leader.shape[0] - 1):
                    a, b = leader[k], leader[k + 1]
                    if np.hypot(*(b - a)) < 2.0:
                        continue
                    hit = _segments_cross(a, b, pts[:-1], seg)
                    if hit is None:
                        continue
                    ang = _angle_between(b - a, seg[hit])
                    if ang < MIN_CROSSING_ANGLE_DEG:
                        problems.append(
                            "%s: the leader of %r crosses %r at %.0f degrees, below %.0f"
                            % (name, ann.get_text()[:24], line.get_label()[:20], ang,
                               MIN_CROSSING_ANGLE_DEG))
                        flagged = True
                        break
                if flagged:
                    break
    return problems


def _segments_cross(p, q, starts, deltas):
    """Index of the first segment that the leader p->q crosses, or None."""
    r = q - p
    denom = deltas[:, 0] * r[1] - deltas[:, 1] * r[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((p[0] - starts[:, 0]) * r[1] - (p[1] - starts[:, 1]) * r[0]) / denom
        u = ((p[0] - starts[:, 0]) * deltas[:, 1]
             - (p[1] - starts[:, 1]) * deltas[:, 0]) / denom
    good = np.isfinite(t) & np.isfinite(u) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    idx = np.flatnonzero(good)
    return int(idx[0]) if idx.size else None


def _angle_between(a, b) -> float:
    """The acute angle between two directions, in degrees."""
    na, nb = np.hypot(*a), np.hypot(*b)
    if na == 0 or nb == 0:
        return 90.0
    c = abs(float(np.dot(a, b)) / (na * nb))
    return float(np.degrees(np.arccos(min(1.0, c))))


def check_with_paper_style(fig, name: str = "figure") -> list[str]:
    """The project's older text checker, run as a second opinion.

    ``smbhb.paper_style.check_text`` predates this module and is used on the papers. Keeping
    both means a hole in one is covered by the other; this one found the defects first, and
    the hole that let a curve run through a label was in this module, not in that one.
    Returns an empty list where it is not importable, which is the case outside the author's
    own checkout.
    """
    import sys

    for candidate in ("/lhome/ific/t/tasicarl/AccurateAstro-SMBHB-PT-PTA",):
        if candidate not in sys.path:
            sys.path.append(candidate)
    try:
        from smbhb.paper_style import check_text
    except Exception:
        return []
    try:
        found = check_text(fig)
    except Exception as exc:
        return ["%s: paper_style.check_text could not run: %s" % (name, exc)]

    # That checker cannot hide a leader before it looks, so it counts an annotation's own
    # leader as something the label is sitting on. Only that case is dropped, and only for
    # labels that do have a leader; everything else it says is kept, because it has caught
    # what this module missed before.
    # Matched on the first line of the label, because that checker prints the text through
    # ``repr``, where a newline comes out as two characters and a plain membership test fails.
    # Both the label as written and the label with its maths delimiters removed, because that
    # checker prints a label through ``repr`` after stripping the dollars.
    leadered = set()
    for ax in fig.get_axes():
        for t in ax.texts:
            if getattr(t, "arrow_patch", None) is None or not t.get_text().strip():
                continue
            first = t.get_text().splitlines()[0].strip()
            leadered.add(first)
            leadered.add(first.strip("$"))
    out = []
    for m in (found or []):
        hits = [t for t in leadered if t and t in m]
        # Both of that checker's rules measure an annotation through
        # ``get_window_extent``, which covers the arrow as well as the words, so a leader
        # reaching across the panel makes the label look both large and busy. The rules in
        # this module measure the words alone and the leader separately, so a report about a
        # label that has a leader is left to them.
        if hits and ("busy background" in m or "text over text" in m):
            continue
        out.append("%s: paper_style says %s" % (name, m))
    return out


#: How tall a figure may be, as a multiple of its width, for one row of panels. A single
#: panel near square reads as a mistake on a page of running text. Each further row adds the
#: same allowance again.
#: A panel of this gallery is between 0.52 and 0.62 times as tall as it is wide. One that is
#: nearly square reads as a different kind of object on the page, which is what a reader
#: notices first, so the allowance sits just above the widest of them rather than at a value
#: no figure here needs.
MAX_ASPECT_PER_ROW = 0.68

#: The width a figure in the documentation must have, in pixels. Width and not the shorter
#: side: every figure is shown at the width of the text column, so a wide short one is not
#: coarse for being short. 1100 leaves something in hand when the reader zooms in.
MIN_WIDTH_PIXELS = 1100


def check_shape(fig, name: str = "figure") -> list[str]:
    """The figure's proportions, and that it will be sharp on the page."""
    w, h = fig.get_size_inches()
    out = []
    # One row of panels, two rows, three: the allowance grows with them, so a stack of panels
    # is not forced into the shape of a single one.
    rows = 1
    for ax in fig.get_axes():
        spec = ax.get_subplotspec()
        if spec is not None:
            rows = max(rows, spec.get_gridspec().nrows)
    limit = MAX_ASPECT_PER_ROW * rows
    if h / w > limit:
        out.append("%s: %.2f times as tall as it is wide, above %.2f for %d row%s"
                   % (name, h / w, limit, rows, "" if rows == 1 else "s"))
    if min(w, h) * fig.dpi < 1:
        out.append("%s: has no size" % name)
    return out


def check_image_resolution(path, name: str = "image") -> list[str]:
    """A shipped image has to be large enough to read when it is zoomed into."""
    try:
        import matplotlib.image as mpimg

        img = mpimg.imread(str(path))
    except Exception as exc:
        return ["%s: could not be read (%s)" % (name, exc)]
    h, w = img.shape[0], img.shape[1]
    if w < MIN_WIDTH_PIXELS:
        return ["%s: %d pixels wide, below %d" % (name, w, MIN_WIDTH_PIXELS)]
    return []


def check_curves_span(ax, name: str = "panel", skip: tuple = ()) -> list[str]:
    """Every curve reaches an edge of the panel rather than stopping inside it."""
    problems = []
    (xlo, xhi), (ylo, yhi) = ax.get_xlim(), ax.get_ylim()
    for line in ax.get_lines():
        label = line.get_label()
        if label.startswith("_") or label in skip:
            continue
        xs = np.asarray(line.get_xdata(), dtype=float)
        ys = np.asarray(line.get_ydata(), dtype=float)
        if xs.size < 3:
            continue
        for x, y, which in ((xs[0], ys[0], "starts"), (xs[-1], ys[-1], "ends")):
            # An end outside the limits is an end the panel clips, so the curve leaves the
            # frame and the rule is satisfied. Only an end that stops inside it is a defect.
            if not (min(xlo, xhi) <= x <= max(xlo, xhi)
                    and min(ylo, yhi) <= y <= max(ylo, yhi)):
                continue
            fx = abs(x - xlo) / abs(xhi - xlo), abs(x - xhi) / abs(xhi - xlo)
            fy = abs(y - ylo) / abs(yhi - ylo), abs(y - yhi) / abs(yhi - ylo)
            if min(fx + fy) > EDGE_TOLERANCE:
                problems.append("%s: the curve %r %s inside the panel, at (%.4g, %.4g)"
                                % (name, label[:24], which, x, y))
    return problems


def check_figure_wording(fig, name: str) -> list[str]:
    """The words inside a figure are held to the same rules as the words around it.

    A phrasing struck out of the prose stayed in a panel for a whole review round, because the
    prose checker reads the pages and nothing read the figures. This reads every text artist
    of the figure with the same patterns.
    """
    import pathlib
    import re
    import sys
    import matplotlib.text as mtext

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import check_docs_style as prose

    problems = []
    # A panel title starts with a capital letter, as a section title does. The lower-case
    # titles of these figures were written as if they continued the caption, and the caption
    # is not next to them in the built page.
    for ax in fig.get_axes():
        for title in (ax.get_title(), ax.get_title("left"), ax.get_title("right")):
            head = title.strip().lstrip("$\\")[:1]
            if head and head.isalpha() and not head.isupper():
                problems.append("%s: the panel title %r starts in lower case"
                                % (name, title.strip()[:48]))
    for text in fig.findobj(mtext.Text):
        body = text.get_text()
        if not body.strip():
            continue
        flat = " ".join(body.split())
        for pattern, what, instead in prose.RULES:
            if not what:
                continue
            # The maths rules read LaTeX source; a panel's label is mathtext and uses the
            # same commands, so both kinds apply.
            if re.search(pattern, flat):
                problems.append("%s: the label %r breaks a prose rule, %s -> write %s"
                                % (name, flat[:48], what, instead))
    return problems
