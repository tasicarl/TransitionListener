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
#: on something. A leader touching one corner is not the same as a curve running behind it.
MAX_BUSY_FRACTION = 0.02

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
                bb = t.get_window_extent(renderer=renderer)
            except Exception:
                continue
            out.append((t, ax, bb))
    return out


def check_figure(fig, name: str = "figure") -> list[str]:
    """Return one message per rule the figure breaks, empty if it is clean."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems: list[str] = []
    height = fig.bbox.height
    pad = TEXT_PAD_PT * fig.dpi / 72.0

    ink, _ = _render(fig)
    boxes = _text_boxes(fig, renderer)

    # --- a text must not sit on ink, counting its margin --------------------------------
    for t, ax, bb in boxes:
        if t in (ax.xaxis.label, ax.yaxis.label) or t in ax.get_xticklabels() \
                or t in ax.get_yticklabels() or t is ax.title:
            continue
        # A text drawn on an opaque patch covers whatever is behind it, which is what the
        # patch is for, so the rule has nothing to say about it.
        bbox_patch = t.get_bbox_patch()
        if bbox_patch is not None:
            fc = bbox_patch.get_facecolor()
            if len(fc) < 4 or fc[3] > 0.9:
                continue
        x0 = int(max(0, bb.x0 - pad))
        x1 = int(min(ink.shape[1], bb.x1 + pad))
        y0 = int(max(0, height - bb.y1 - pad))
        y1 = int(min(ink.shape[0], height - bb.y0 + pad))
        if x1 <= x0 or y1 <= y0:
            continue
        patch = ink[y0:y1, x0:x1].copy()
        own = ink[int(height - bb.y1):int(height - bb.y0), int(bb.x0):int(bb.x1)]
        # An annotation's own leader is allowed to touch its own box: that is what makes it a
        # leader. Its pixels are taken out before the box is judged.
        arrow = getattr(t, "arrow_patch", None)
        if arrow is not None:
            try:
                ab = arrow.get_window_extent(renderer=renderer)
                ax0 = int(max(x0, ab.x0)) - x0
                ax1 = int(min(x1, ab.x1)) - x0
                ay0 = int(max(y0, height - ab.y1)) - y0
                ay1 = int(min(y1, height - ab.y0)) - y0
                if ax1 > ax0 and ay1 > ay0:
                    patch[max(0, ay0):ay1, max(0, ax0):ax1] = False
            except Exception:
                pass
        busy = (patch.sum() - own.sum()) / max(1, patch.size - own.size)
        if busy > MAX_BUSY_FRACTION:
            problems.append("%s: text %r sits on ink, %.0f%% of its padded box"
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
        for line in ax.get_lines():
            pts = ax.transData.transform(np.column_stack(
                [np.asarray(line.get_xdata(), dtype=float),
                 np.asarray(line.get_ydata(), dtype=float)]))
            inside = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                      & (height - pts[:, 1] >= y0) & (height - pts[:, 1] <= y1))
            if inside.any():
                problems.append("%s: the legend sits on the curve %r"
                                % (name, line.get_label()[:30]))
                break

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
            d = tip - tail
            if not np.hypot(*d) > 1.0:
                continue
            for line in ax.get_lines():
                xs = np.asarray(line.get_xdata(), dtype=float)
                ys = np.asarray(line.get_ydata(), dtype=float)
                if xs.size < 2:
                    continue
                pts = ax.transData.transform(np.column_stack([xs, ys]))
                seg = np.diff(pts, axis=0)
                hit = _segments_cross(tail, tip, pts[:-1], seg)
                if hit is None:
                    continue
                ang = _angle_between(d, seg[hit])
                if ang < MIN_CROSSING_ANGLE_DEG:
                    problems.append(
                        "%s: the leader of %r crosses %r at %.0f degrees, below %.0f"
                        % (name, ann.get_text()[:24], line.get_label()[:20], ang,
                           MIN_CROSSING_ANGLE_DEG))
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
            fx = abs(x - xlo) / abs(xhi - xlo), abs(x - xhi) / abs(xhi - xlo)
            fy = abs(y - ylo) / abs(yhi - ylo), abs(y - yhi) / abs(yhi - ylo)
            if min(fx + fy) > EDGE_TOLERANCE:
                problems.append("%s: the curve %r %s inside the panel, at (%.4g, %.4g)"
                                % (name, label[:24], which, x, y))
    return problems
