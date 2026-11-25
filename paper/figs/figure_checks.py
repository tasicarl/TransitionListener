"""The automated golden-rule checks of the paper's figures, and the size check of the build.

`check_text` (text over text, text over a busy background, text outside the canvas) is adapted
from the figure checks of the binaryfog paper. It also sees the texts of inset axes
(`ax.child_axes`, which `fig.get_axes()` does not list), and its `busy` threshold is 0.5 per cent
of the pixels behind a label, so that a dotted line through a label is caught. The further checks
are

- `check_legends`: a legend over data, i.e. more than two points of any drawn curve, contour,
  filled boundary or patch outline inside a legend's bounding box;
- `check_clearance`: a text box closer than CLEAR points to the frame of an axes, or straddling it
  (a frame is the set of visible spines; the tick labels, axis labels and title of an axes are its
  own decoration and are tested against the other axes only);
- `check_sizes`: no text below a minimum point size;
- `check_curves`: a curve spans the horizontal range of its axes unless it ends physically
  (gid "physical_end"); `check_labels_inside`: labels and leaders inside their axes;
  `check_ticks_on_top`: ticks and frame above the data (freeze raises them); `check_label_nearest`: a
  label without a leader lies closer to its own curve than to any other plotted line, and a leader ends
  on its own curve; `check_leaders`: the leaders of the labels of an axes do not cross.

`gate(fig, name)` freezes the layout (draw, then layout engine "none"), runs every check, prints the
findings and raises if anything is found. Every figure calls it before it is saved.
"""
import os

import numpy as np
import matplotlib as mpl

CLEAR = 4.0              # points between a text box and an axes frame
MIN_PT = 6.0             # smallest text in a data figure, in points as printed


# --------------------------------------------------------------- helpers
def all_axes(fig):
    """Every axes of the figure, inset axes included (they are children of their parent axes
    and are not in fig.axes)."""
    out = []

    def walk(a):
        out.append(a)
        for c in getattr(a, "child_axes", []):
            walk(c)
    for a in fig.get_axes():
        walk(a)
    return out


def legends(fig):
    """Every legend: of each axes, added to an axes as an extra artist, and of the figure."""
    out = []
    for a in all_axes(fig):
        if a.get_legend() is not None:
            out.append(a.get_legend())
        out += [x for x in a.artists if isinstance(x, mpl.legend.Legend)]
    return out + list(fig.legends)


def _decoration(ax):
    """The texts that belong to an axes' frame: its tick labels, axis labels and title."""
    out = [ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title]
    for axis in (ax.xaxis, ax.yaxis):
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            out += [tick.label1, tick.label2]
    return out


# --------------------------------------------------------------- check_text
def _texts(fig):
    """Every visible, non-empty piece of text in the figure, including the tick labels on both sides
    and the texts of inset axes."""
    out = []
    for ax in all_axes(fig):
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            trf = axis.get_transform()
            lo, hi = sorted(trf.transform(np.asarray(lim, float)))
            pad = 1e-6 * (hi - lo)
            for tick in axis.get_major_ticks():
                v = float(trf.transform(np.asarray([tick.get_loc()], float))[0])
                if lo - pad <= v <= hi + pad:
                    out += [tick.label1, tick.label2]
        out.extend([ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title])
        out.extend(ax.texts)
    out.extend(fig.texts)
    if getattr(fig, "_suptitle", None) is not None:
        out.append(fig._suptitle)
    for lg in legends(fig):
        out.extend(lg.get_texts())
        if lg.get_title().get_text():
            out.append(lg.get_title())
    seen, uniq = set(), []
    for t in out:
        if id(t) not in seen and t.get_visible() and t.get_text().strip():
            seen.add(id(t)); uniq.append(t)
    return uniq


def _short(t):
    return t.get_text().replace(r"\mathdefault", "").replace("$", "")[:46]


def _where(fig, t):
    ax = getattr(t, "axes", None)
    if ax is None:
        return "figure"
    axs = all_axes(fig)
    return f"panel {axs.index(ax)}" if ax in axs else "figure"


def _boxes(fig):
    rend = fig.canvas.get_renderer()
    keep, boxes = [], []
    for t in _texts(fig):
        try:
            # the box of the text itself: an annotation's own extent includes its arrow
            bb = mpl.text.Text.get_window_extent(t, rend)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        keep.append(t); boxes.append(bb)
    return keep, boxes


def _turned_box(fig, t):
    """The corners of the box of the text t turned with it, in display pixels."""
    rend = fig.canvas.get_renderer()
    bb = mpl.text.Text.get_window_extent(t, rend)
    rot = t.get_rotation()
    t.set_rotation(0.0)
    b0 = mpl.text.Text.get_window_extent(t, rend)
    t.set_rotation(rot)
    c = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
    a = np.radians(rot)
    u, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    w, h = b0.width / 2, b0.height / 2
    return np.array([c + sx * w * u + sy * h * n for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def _rotated_gap(fig, s, t):
    """The separation of the turned boxes of two texts along the separating axes (negative when they overlap)."""
    A, B = _turned_box(fig, s), _turned_box(fig, t)
    best = -np.inf
    for P in (A, B):
        for k in range(4):
            e = P[(k + 1) % 4] - P[k]
            ax_ = np.array([-e[1], e[0]]) / np.hypot(*e)
            pa, pb = A @ ax_, B @ ax_
            best = max(best, max(pa.min() - pb.max(), pb.min() - pa.max()))
    return best


def check_text(fig, gap=1.0, busy=0.005, delta=12):
    """The golden rule, no text over a non-uniform area: text over text from the bounding boxes, text over
    anything else from the rendered pixels behind each label (with all text hidden), and text
    outside the canvas."""
    fig.canvas.draw()
    keep, boxes = _boxes(fig)
    bad = []
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            u, v = boxes[i], boxes[j]
            # the glyphs of one label set along a curve (attribute _group) touch each other by construction
            if getattr(keep[i], "_group", None) is not None and getattr(keep[i], "_group", None) == getattr(keep[j], "_group", 0):
                continue
            dx = max(u.x0, v.x0) - min(u.x1, v.x1)
            dy = max(u.y0, v.y0) - min(u.y1, v.y1)
            if dx < gap and dy < gap and (keep[i].get_rotation() or keep[j].get_rotation()):
                # a rotated label: its axis-aligned box is mostly empty corners; compare the turned boxes of the text
                d = _rotated_gap(fig, keep[i], keep[j])
                if d >= gap:
                    continue
                dx = dy = d
            if dx < gap and dy < gap:
                bad.append(f"text over text, {max(dx, dy):+.1f} px apart: {_where(fig, keep[i])} "
                           f"{_short(keep[i])!r} and {_where(fig, keep[j])} {_short(keep[j])!r}")
    # hide the glyphs only: an annotation keeps its leader, so that a leader through another label is seen
    vis = [t.get_visible() for t in keep]
    alph = [t.get_alpha() for t in keep]
    for t in keep:
        if isinstance(t, mpl.text.Annotation) and t.arrow_patch is not None:
            t.set_alpha(0.0)
        else:
            t.set_visible(False)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
    for t, v, a in zip(keep, vis, alph):
        t.set_visible(v)
        t.set_alpha(a)
    h = img.shape[0]
    for t, bb in zip(keep, boxes):
        bb = bb.padded(-1.0) if bb.width > 3 and bb.height > 3 else bb
        x0, x1 = int(np.floor(bb.x0)), int(np.ceil(bb.x1))
        y0, y1 = int(np.floor(bb.y0)), int(np.ceil(bb.y1))
        r0, r1 = max(h - y1, 0), min(h - y0, h)
        c0, c1 = max(x0, 0), min(x1, img.shape[1])
        patch = img[r0:r1, c0:c1]
        if patch.size == 0:
            continue
        flat = patch.reshape(-1, 3)
        rot = float(t.get_rotation()) % 180.0
        if min(rot, 180.0 - rot) > 0.5 and abs(rot - 90.0) > 0.5:
            # rotated text: only the pixels inside the rotated rectangle of the glyphs, not its axis-aligned box,
            # which takes in the curve a label runs along. The rectangle has the unrotated size of the text
            # and the centre of the box (a rotated rectangle and its axis-aligned box share the centre).
            from matplotlib.path import Path as _Path
            r_ = t.get_rotation(); t.set_rotation(0.0)
            b0 = t.get_window_extent(renderer=fig.canvas.get_renderer()); t.set_rotation(r_)
            w2, h2 = 0.5 * b0.width - 1.0, 0.5 * b0.height - 1.0
            cx, cy = 0.5 * (bb.x0 + bb.x1), 0.5 * (bb.y0 + bb.y1)
            ca, sa = np.cos(np.radians(r_)), np.sin(np.radians(r_))
            corners = [(cx + ca * u - sa * v, cy + sa * u + ca * v) for u, v in ((-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2))]
            ys, xs = np.mgrid[r0:r1, c0:c1]
            inside = _Path(corners).contains_points(np.column_stack([xs.ravel() + 0.5, (h - ys.ravel()) - 0.5]))
            flat = flat[inside]
            if flat.size == 0:
                continue
        cols, counts = np.unique(flat, axis=0, return_counts=True)
        mode = cols[np.argmax(counts)]
        frac = float(np.mean(np.max(np.abs(flat - mode), axis=1) > delta))
        if frac > busy:
            bad.append(f"text on a busy background ({100 * frac:.1f}% of its pixels): "
                       f"{_where(fig, t)} {_short(t)!r}")
    w, h = fig.get_size_inches() * fig.dpi
    for t, bb in zip(keep, boxes):
        over = max(-bb.x0, bb.x1 - w, -bb.y0, bb.y1 - h)
        if over > 0.5:
            bad.append(f"text outside the canvas by {over / fig.dpi * 72:.1f} pt, it will be "
                       f"clipped: {_where(fig, t)} {_short(t)!r}")
    fig.canvas.draw()
    return bad


# --------------------------------------------------------------- legend over data
def _polylines(artist):
    """The drawn geometry of an artist in display coordinates, as a list of (N, 2) arrays
    (a line, the boundary of a filled region, a patch outline, or the markers of a scatter)."""
    out = []
    if isinstance(artist, mpl.lines.Line2D):
        xy = artist.get_transform().transform(artist.get_xydata())
        if artist.get_linestyle() not in ("None", "none", "", " "):
            out.append(xy)
        if artist.get_marker() not in (None, "None", "none", "", " "):
            out.extend(p[None, :] for p in xy)
    elif isinstance(artist, mpl.collections.Collection):
        tr = artist.get_transform()
        for p in artist.get_paths():
            out.extend(tr.transform_path(p).to_polygons(closed_only=False))
        offs = artist.get_offsets()
        if len(artist.get_paths()) and offs is not None and len(offs) > 1:
            out.extend(p[None, :] for p in artist.get_offset_transform().transform(offs))
    elif isinstance(artist, mpl.patches.Patch):
        out.extend(artist.get_transform().transform_path(artist.get_path()).to_polygons(closed_only=False))
    return out


def _densify(xy, step=1.0):
    """Points along a polyline no further apart than `step` pixels. Vectorised (the label searches call it
    some 10^5 times per figure); the points are those of the segment-by-segment loop below, bit for bit
    (k/n as numpy.linspace computes it, the end of each segment exactly its vertex), which still serves
    polylines with non-finite points."""
    if len(xy) < 2:
        return xy
    a, d = xy[:-1], xy[1:] - xy[:-1]
    if np.isfinite(d).all():
        n = np.ceil(np.hypot(d[:, 0], d[:, 1]) / step).astype(int)
        m = np.maximum(n, 1)
        seg = np.repeat(np.arange(len(a)), m)
        k = np.arange(len(seg)) - np.repeat(np.cumsum(m) - m, m) + 1
        t = k * np.repeat(1.0 / m, m)
        t[k == np.repeat(m, m)] = 1.0
        pts = a[seg] + t[:, None] * d[seg]
        short = np.repeat(n <= 1, m)
        pts[short] = xy[1:][seg[short]]
        return np.concatenate([xy[:1], pts])
    pts = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = int(np.ceil(np.hypot(*(b - a)) / step))
        if n > 1:
            t = np.linspace(0, 1, n + 1)[1:, None]
            pts.append(a + t * (b - a))
        else:
            pts.append(b[None, :])
    return np.concatenate(pts)


def check_legends(fig, max_points=2):
    """Fail when more than `max_points` points of any drawn curve fall inside a legend."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bad = []
    for lg in legends(fig):
        bb = lg.get_window_extent(rend)
        for ax in all_axes(fig):
            for art in list(ax.lines) + list(ax.collections) + list(ax.patches):
                if not art.get_visible() or (art.get_alpha() is not None and art.get_alpha() == 0):
                    continue
                clip = art.get_clip_box() if art.get_clip_on() else None
                n = 0
                for xy in _polylines(art):
                    xy = xy[np.all(np.isfinite(xy), axis=1)]
                    if not len(xy):
                        continue
                    pts = _densify(xy)
                    if clip is not None:
                        pts = pts[(pts[:, 0] >= clip.x0) & (pts[:, 0] <= clip.x1)
                                  & (pts[:, 1] >= clip.y0) & (pts[:, 1] <= clip.y1)]
                    n += int(np.sum((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1)
                                    & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)))
                if n > max_points:
                    lab = art.get_label() if not str(art.get_label()).startswith("_") else type(art).__name__
                    bad.append(f"legend over data: {n} points of {lab!r} in panel "
                               f"{all_axes(fig).index(ax)} lie inside a legend")
    return bad


# --------------------------------------------------------------- clearance
def _frames(ax):
    """The visible spines of an axes as segments (x0, y0, x1, y1) in display coordinates."""
    if not ax.axison:
        return []
    b = ax.bbox
    seg = {"left": (b.x0, b.y0, b.x0, b.y1), "right": (b.x1, b.y0, b.x1, b.y1),
           "bottom": (b.x0, b.y0, b.x1, b.y0), "top": (b.x0, b.y1, b.x1, b.y1)}
    return [seg[k] for k, s in ax.spines.items() if k in seg and s.get_visible()]


def check_clearance(fig, clear=CLEAR):
    """No text box within `clear` points of an axes frame, and none across one."""
    fig.canvas.draw()
    keep, boxes = _boxes(fig)
    px = fig.dpi / 72.0
    bad = []
    for ax in all_axes(fig):
        own = {id(t) for t in _decoration(ax)}
        for t, bb in zip(keep, boxes):
            for x0, y0, x1, y1 in _frames(ax):
                dx = max(min(x0, x1) - bb.x1, bb.x0 - max(x0, x1), 0.0)
                dy = max(min(y0, y1) - bb.y1, bb.y0 - max(y0, y1), 0.0)
                d = np.hypot(dx, dy)
                if d == 0.0:
                    bad.append(f"text across a frame: {_where(fig, t)} {_short(t)!r} "
                               f"(frame of panel {all_axes(fig).index(ax)})")
                    break
                if id(t) not in own and d < clear * px:
                    bad.append(f"text {d / px:.1f} pt from a frame: {_where(fig, t)} {_short(t)!r} "
                               f"(frame of panel {all_axes(fig).index(ax)})")
                    break
    return bad


def check_sizes(fig, min_pt=MIN_PT):
    return [f"text of {t.get_fontsize():.1f} pt < {min_pt} pt: {_where(fig, t)} {_short(t)!r}"
            for t in _texts(fig) if t.get_fontsize() < min_pt - 1e-9]


# --------------------------------------------------------------- the gate
def ticks_on_top(fig):
    """Ticks and frame above every data element. Raises the axis and spine
    artists of every data axes above the highest zorder of its other non-text artists."""
    for ax in all_axes(fig):
        z = [a.get_zorder() for a in ax.get_children()
             if not isinstance(a, (mpl.text.Text, mpl.legend.Legend, mpl.axis.Axis, mpl.spines.Spine))
             and a is not ax.patch]
        top = max(z + [2.5]) + 0.1
        for a in [ax.xaxis, ax.yaxis] + list(ax.spines.values()):
            a.set_zorder(top)


def check_ticks_on_top(fig):
    bad = []
    for ax in all_axes(fig):
        if not ax.axison:
            continue
        z = [a.get_zorder() for a in ax.get_children()
             if not isinstance(a, (mpl.text.Text, mpl.legend.Legend, mpl.axis.Axis, mpl.spines.Spine))
             and a is not ax.patch]
        if z and min(ax.xaxis.get_zorder(), ax.yaxis.get_zorder()) <= max(z):
            bad.append(f"ticks below data in panel {all_axes(fig).index(ax)}")
    return bad


def check_curves(fig, tol=2.0, min_points=10):
    """A curve spans the horizontal range of its axes. Every visible line
    with a line style and at least min_points points reaches the left and the right edge of a data
    axes (within tol display pixels), unless it is declared to end where the quantity ends,
    line.set_gid("physical_end"). Such a curve must then leave the axes through the frame where it ends,
    its end point on or beyond the lower or upper edge, so that it is resolved until it leaves the axes
    and does not stop at the last computed parameter value. A curve whose input exists only on part of the horizontal range, such as an emulator fitted on a band of
    frequencies, declares it with line.set_gid("domain_end") and is exempt; the caption must state the range.
    Lines narrower than five per cent of the axes (vertical lines, markers, short leaders) are not curves."""
    bad = []
    for ax in all_axes(fig):
        if not ax.axison:
            continue
        bb = ax.get_window_extent()
        for ln in ax.get_lines():
            if not ln.get_visible() or ln.get_linestyle() in ("None", "none", "", " "):
                continue
            xy = np.asarray(ln.get_xydata(), float)
            xy = xy[np.all(np.isfinite(xy), axis=1)]
            if len(xy) < min_points:
                continue
            d = ln.get_transform().transform(xy)
            x0, x1 = d[:, 0].min(), d[:, 0].max()
            if x1 - x0 < 0.05 * bb.width:
                continue
            if ln.get_gid() == "domain_end":
                continue
            if ln.get_gid() == "physical_end":
                for x_end, side in ((x0, "left"), (x1, "right")):
                    inside = (x_end > bb.x0 + tol) if side == "left" else (x_end < bb.x1 - tol)
                    y_end = d[np.argmin(d[:, 0]) if side == "left" else np.argmax(d[:, 0]), 1]
                    if inside and bb.y0 + tol < y_end < bb.y1 - tol:
                        bad.append(f"curve with a physical end stops inside the axes (panel {all_axes(fig).index(ax)}, "
                                   f"{side} end {(y_end - bb.y0) * 72 / fig.dpi:.1f} pt above the lower edge): "
                                   f"resolve it until it leaves the frame")
                continue
            if x0 > bb.x0 + tol or x1 < bb.x1 - tol:
                bad.append(f"curve ends inside the axes (panel {all_axes(fig).index(ax)}, "
                           f"{(x0 - bb.x0) * 72 / fig.dpi:.1f} pt from the left edge, "
                           f"{(bb.x1 - x1) * 72 / fig.dpi:.1f} pt from the right): {ln.get_label()!r}")
    return bad


def check_labels_inside(fig, tol=0.5):
    """The labels and annotations of an axes, leaders included, lie inside
    its frame."""
    rend = fig.canvas.get_renderer()
    bad = []
    for ax in all_axes(fig):
        if not ax.axison:                  # a helper axes without a frame (side labels, a schematic canvas)
            continue
        bb = ax.get_window_extent()
        for t in ax.texts:
            if not t.get_visible() or not t.get_text().strip():
                continue
            boxes = [mpl.text.Text.get_window_extent(t, rend)]
            if isinstance(t, mpl.text.Annotation) and t.arrow_patch is not None:
                boxes.append(t.arrow_patch.get_window_extent(rend))
            for b in boxes:
                if b.x0 < bb.x0 - tol or b.x1 > bb.x1 + tol or b.y0 < bb.y0 - tol or b.y1 > bb.y1 + tol:
                    bad.append(f"label outside its axes: {_where(fig, t)} {_short(t)!r}")
                    break
    return bad


def plot_lines(ax):
    """The plotted lines of an axes: visible Line2D with a line style (vertical marker lines included)."""
    return [ln for ln in ax.get_lines() if ln.get_visible() and ln.get_linestyle() not in ("None", "none", "", " ")]


def line_distances(fig, ax, cx, cy, w, h, lines, angle=0.0):
    """Distance in points from a box (display centre cx, cy, width w, height h, rotated by angle degrees about its
    centre) to each line."""
    out = []
    c, s_ = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    for ln in lines:
        pts = np.concatenate([_densify(p, 1.0) for p in _polylines(ln)] or [np.empty((0, 2))])
        if not len(pts):
            out.append(np.inf); continue
        if angle:
            d0 = pts - (cx, cy)
            pts = np.column_stack([cx + c * d0[:, 0] + s_ * d0[:, 1], cy - s_ * d0[:, 0] + c * d0[:, 1]])
        dx = np.maximum(np.abs(pts[:, 0] - cx) - w / 2, 0.0)
        dy = np.maximum(np.abs(pts[:, 1] - cy) - h / 2, 0.0)
        out.append(float(np.min(np.hypot(dx, dy))) * 72 / fig.dpi)
    return out


def check_label_nearest(fig, margin=0.5, leader_tol=3.0):
    """The plotted line closest to a label is the one it names. A text that
    declares its curve (attribute `_labels`, set by the figure code) and has no leader must lie closer to
    that curve than to any other plotted line, vertical marker lines included; with a leader, the leader
    must end on that curve."""
    rend = fig.canvas.get_renderer()
    bad = []
    for ax in all_axes(fig):
        lines = plot_lines(ax)
        for t in ax.texts:
            own = getattr(t, "_labels", None)
            if own is None or not t.get_visible():
                continue
            arrow = getattr(t, "arrow_patch", None)
            if arrow is not None:
                end = np.asarray(t.xy if t.xycoords == "data" else t.xy, float)
                ex, ey = ax.transData.transform(end)
                d = line_distances(fig, ax, ex, ey, 0.0, 0.0, [own])[0]
                if d > leader_tol:
                    bad.append(f"leader of {_short(t)!r} ends {d:.1f} pt from its curve ({_where(fig, t)})")
                continue
            bb = mpl.text.Text.get_window_extent(t, rend)
            ang = t.get_rotation()
            if ang % 180:
                # rotated text: its own rectangle, not the axis-aligned box around it (same centre)
                t.set_rotation(0.0)
                b0 = mpl.text.Text.get_window_extent(t, rend)
                t.set_rotation(ang)
                ds = line_distances(fig, ax, (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2, b0.width, b0.height, lines, ang)
            else:
                ds = line_distances(fig, ax, (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2, bb.width, bb.height, lines)
            d_own = ds[lines.index(own)] if own in lines else np.inf
            others = [d for ln, d in zip(lines, ds) if ln is not own]
            if others and min(others) <= d_own + margin:
                near = [ln for ln, d in zip(lines, ds) if ln is not own][int(np.argmin(others))]
                bad.append(f"label {_short(t)!r} lies {d_own:.1f} pt from its curve but {min(others):.1f} pt from "
                           f"another line, {near.get_label()!r} ({_where(fig, t)})")
    return bad


def check_leaders(fig):
    """The leaders of the labels of an axes do not cross each other."""
    def cross(p, q, r, s_):
        o = lambda a, b, c: np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        return o(p, q, r) * o(p, q, s_) < 0 and o(r, s_, p) * o(r, s_, q) < 0
    fig.canvas.draw()
    bad = []
    for ax in all_axes(fig):
        legs = [(t, np.asarray(t.arrow_patch.get_path().vertices, float)) for t in ax.texts
                if isinstance(t, mpl.text.Annotation) and t.arrow_patch is not None and t.get_visible()
                and t.get_text()]      # an annotation without text is a drawn arrow, not the leader of a label
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                (ta, va), (tb, vb) = legs[i], legs[j]
                if any(cross(va[k], va[k + 1], vb[m], vb[m + 1])
                       for k in range(len(va) - 1) for m in range(len(vb) - 1)):
                    bad.append(f"the leaders of {_short(ta)!r} and {_short(tb)!r} cross ({_where(fig, ta)})")
    return bad


def freeze(fig):
    """Lay the figure out once and keep it: the checks hide text and redraw, and a layout engine
    would then move the axes into the space the labels occupied. Ticks and frames go on top."""
    ticks_on_top(fig)
    fig.canvas.draw()
    fig.set_layout_engine("none")


def gate(fig, name, min_pt=MIN_PT, extra=()):
    """Run every check and raise if any finds something. `extra` adds findings of the figure's
    own (for example labels that must end inside their bar)."""
    freeze(fig)
    bad = (check_text(fig) + check_legends(fig) + check_clearance(fig) + check_sizes(fig, min_pt)
           + check_curves(fig) + check_labels_inside(fig) + check_ticks_on_top(fig) + check_label_nearest(fig)
           + check_leaders(fig)
           + list(extra))
    if bad:
        print(f"  GOLDEN RULE violated in {name}, {len(bad)} place(s):")
        for b in bad:
            print(f"    {b}")
        if os.environ.get("FIGCHECK_WARN_ONLY"):          # for drafting only; the build itself fails
            print("  (FIGCHECK_WARN_ONLY set: saved anyway)")
            return bad
        raise SystemExit(f"{name}: {len(bad)} figure-check violation(s), not saved")
    print(f"  checks: {name} is clean (text, legends, clearance, sizes >= {min_pt} pt)")


# --------------------------------------------------------------- printed size
TEXTWIDTH_PT = 440.0                       # jcappub.sty: \textwidth = 440pt (TeX points)
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27        # 6.088 in = 438.4 PDF points


def pdf_width_bp(path):
    from pypdf import PdfReader
    box = PdfReader(path).pages[0].mediabox
    return float(box.width)


def size_table(items, tol=0.02):
    """items: (path, fraction of \\textwidth it is included at). Prints the ratio of the saved
    width to the included width and asserts that it is 1 within `tol`."""
    import os
    print(f"  {'figure':<22}{'included at':>12}{'saved [bp]':>12}{'included [bp]':>15}{'ratio':>8}")
    off = []
    for path, frac in items:
        w = pdf_width_bp(path)
        want = frac * TEXTWIDTH_IN * 72.0
        r = w / want
        print(f"  {os.path.basename(path):<22}{frac:>9.2f} tw{w:>12.1f}{want:>15.1f}{r:>8.3f}")
        if abs(r - 1) > tol:
            off.append(os.path.basename(path))
    assert not off, f"printed size differs from the drawn size by more than {100 * tol:.0f}%: {off}"
