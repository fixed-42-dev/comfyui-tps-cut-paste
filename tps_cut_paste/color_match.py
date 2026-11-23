"""Match the edited cut's color and texture to the original before it is pasted.

An edit model changes more than it was asked to: the whole cut comes back a
little darker or warmer, with its contrast and fine texture redrawn. Matching
statistics over the cut fails once the edit adds an object, because the object
pulls them (a red balloon over half the mask made a plain match brighten the
background by +0.04 in Oklab lightness, the wrong way).

So the difference between the aligned edit and the original is treated as a
height map: the edit's overall grade lifts or lowers a plateau, and what the
edit changed stands on it. The plateau is measured only on pixels the edit left
alone, found from outside the painted mask inward, and undone; the object and
what it did to its surroundings stay.

1. trust   - color and round-trip z-scores against local plateau and floor
             maps, with the painted mask as a prior: its inside never defines
             the plateau. Gives per-pixel trust and the change mask.
2. grade   - an Oklab offset per tone of the edit, fitted on trusted pixels,
             held flat beyond the tones it was measured on.
3. edge    - what the grade leaves at the mask edge, spread smoothly and faded
             inward with a profile; carries the correction when the plateau is
             uneven (an edit that relights the scene).
4. texture - a local gain per frequency band brings the detail energy back to
             the original's; the feather keeps the detail energy of both sides
             instead of averaging two different textures into a soft halo.

Frames here are float32 (h, w, 3) sRGB in [0, 1]. The proof of concept and the
tests behind the choices are in poc/color/ on the color-match-poc branch.
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

WORK = 1024             # long side the measurements run at
BAND = 80               # cut px outside the mask whose completeness is reported
EDGE_BAND = 48          # cut px either side of the mask edge the edge field is measured on

# ------------------------------------------------ color spaces

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M1i, _M2i = np.linalg.inv(_M1), np.linalg.inv(_M2)
LUMA = np.array([0.2126, 0.7152, 0.0722])


def _to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def to_oklab(rgb):
    return (np.cbrt(_to_linear(rgb) @ _M1.T) @ _M2.T).astype(np.float32)


def from_oklab(lab):
    return np.clip(_to_srgb(((lab @ _M2i.T) ** 3) @ _M1i.T), 0, 1).astype(np.float32)


# ------------------------------------------------ maps

def blur(x, sigma):
    """Gaussian blur of a 2D map. A wide one runs on a copy shrunk until sigma is
    about 4 px and is enlarged back: the fills below reach 200 px and more, and a
    full-size kernel that wide took over 90 % of the paste's time. The result is
    smooth on that scale, so the shrinking loses nothing visible."""
    x = np.ascontiguousarray(x, np.float32)
    f = int(sigma // 4)
    h, w = x.shape[:2]
    if f < 2 or min(h, w) < 4 * f:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    sw, sh = -(-w // f), -(-h // f)
    small = cv2.resize(x, (sw, sh), interpolation=cv2.INTER_AREA)
    # The area average (a box) and the linear enlargement (a tent) blur too:
    # their variances, k^2/12 and k^2/6 for a k px step, come off the Gaussian's.
    kx, ky = w / sw, h / sh
    sx = np.sqrt(max(sigma * sigma - kx * kx / 4.0, 0.25)) / kx
    sy = np.sqrt(max(sigma * sigma - ky * ky / 4.0, 0.25)) / ky
    # Mirrored at the border with the edge pixel repeated, so the mirror scales
    # with the copy; the default leaves the edge pixel out, which on the copy
    # leaves out k px and shifted a blur wider than the frame.
    small = cv2.GaussianBlur(small, (0, 0), sx, sigmaY=sy, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def nconv(value, weight, sigma):
    """Normalized convolution: a weighted local mean that fills gaps from around them."""
    num = blur(value * weight, sigma)
    den = blur(weight, sigma)
    return num / np.maximum(den, 1e-6), den


def fill(value, weight, sigma, scales=(1, 4, 16)):
    """Normalized convolution that reaches deep into large gaps: where support is
    thin the next coarser scale takes over, and finally the weighted median."""
    sel = weight > 0.5
    out = np.full(value.shape, np.median(value[sel]) if sel.any() else 0.0, np.float32)
    for s in reversed(scales):
        v, den = nconv(value, weight, sigma * s)
        a = den / (den + 0.05)
        out = a * v + (1 - a) * out
    return out


def signed_distance(mask):
    """Distance to the mask edge in px: positive outside, negative inside. The
    frame border is not an edge, so a side on the image border stays open."""
    m = (mask > 0).astype(np.uint8)
    out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5) if m.any() else np.full(m.shape, 1e6, np.float32)
    ins = cv2.distanceTransform(m, cv2.DIST_L2, 5) if (1 - m).any() else np.full(m.shape, 1e6, np.float32)
    return out - ins


def _flatness(rgb):
    y = (rgb @ LUMA).astype(np.float32)
    g = np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3))
    return cv2.GaussianBlur(g, (0, 0), 2.0)


def _soft(z, lo=2.0, hi=4.0):
    """1 below lo floors, 0 above hi."""
    return np.clip((hi - z) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ------------------------------------------------ stage 1: trust

def robust_z(x, sigma, floor_min, one_sided=False, iters=3, w0=None):
    """How many local floors each pixel sits from its local plateau.

    Seeded by a global median/MAD pass and refined locally; a pixel once
    excluded stays excluded, so an object as big as the window cannot raise its
    own floor and hide. w0 = 0 marks pixels that never define the plateau (the
    painted mask's inside); there it is filled in from outside."""
    x = x if x.ndim == 3 else x[..., None]
    base = np.ones(x.shape[:2], np.float32) if w0 is None else w0.astype(np.float32)
    sample = x[base > 0.5] if (base > 0.5).any() else x.reshape(-1, x.shape[-1])
    med = np.median(sample, 0)
    mad = 1.4826 * np.median(np.abs(sample - med), 0) + floor_min
    dev0 = (x - med) if one_sided else np.abs(x - med)
    w = np.minimum(base, _soft((dev0 / mad).max(-1)))
    for _ in range(iters):
        plateau = np.stack([fill(x[..., c], w, sigma) for c in range(x.shape[-1])], -1)
        dev = x - plateau if one_sided else np.abs(x - plateau)
        floor = np.stack([1.4826 * fill(np.abs(x[..., c] - plateau[..., c]), w, sigma)
                          for c in range(x.shape[-1])], -1) + floor_min
        z = (dev / floor).max(-1)
        w = np.minimum(w, _soft(z))
    return z, plateau


def _roundtrip_error(flows, a, b):
    fwd, bwd = flows(a, b)
    h, w = fwd.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    back = np.stack([cv2.remap(np.ascontiguousarray(bwd[i], np.float32), xx + fwd[0], yy + fwd[1],
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for i in range(2)])
    return np.hypot(fwd[0] + back[0], fwd[1] + back[1]).astype(np.float32)


def _clean(m, min_share=0.0005, grow=4):
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    m = np.isin(lab, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_share * m.size]).astype(np.uint8)
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(m, cnt, -1, 1, -1)                                   # fill holes
    if grow:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
    return m


def trust(original, processed, flows=None, prior=None, given=None, work=WORK):
    """Per-pixel trust (0-1), z maps and the change mask, at working size.

    given: a change mask from outside (painted, or from a segmentation model),
    (h, w) at the frame's size. It takes the place of the geometry evidence,
    which fails wherever the edit redrew the texture: the change mask is then
    the given one plus the color peaks, so light or shadow the change cast
    past it still counts as changed."""
    H, W = original.shape[:2]
    k = min(1.0, work / max(H, W))
    ww, wh = max(8, int(round(W * k / 8)) * 8), max(8, int(round(H * k / 8)) * 8)
    o = cv2.resize(original, (ww, wh), interpolation=cv2.INTER_AREA)
    p = cv2.resize(processed, (ww, wh), interpolation=cv2.INTER_AREA)
    sigma = max(ww, wh) / 80.0
    w0 = None
    if prior is not None and (prior > 0).any() and not (prior > 0).all():
        w0 = 1.0 - (cv2.resize((prior > 0).astype(np.uint8), (ww, wh), interpolation=cv2.INTER_NEAREST) > 0)
    z_col, plateau = robust_z(to_oklab(p) - to_oklab(o), sigma, floor_min=0.004, w0=w0)
    if given is not None:
        g_w = cv2.resize((given > 0.5).astype(np.uint8), (ww, wh), interpolation=cv2.INTER_NEAREST) > 0
        z_rt = np.zeros_like(z_col)
        change = _clean((z_col > 4) | g_w)
    else:
        z_rt = (robust_z(_roundtrip_error(flows, o, p), sigma, floor_min=0.05, one_sided=True, w0=w0)[0]
                if flows is not None else np.zeros_like(z_col))
        change = _clean((z_col > 4) | (z_rt > 4))
    g = _flatness(o)
    g50, g90 = np.percentile(g, 50), np.percentile(g, 90)
    w_flat = np.clip((g90 - g) / (g90 - g50 + 1e-6), 0.0, 1.0)
    clean = _soft(z_col) * _soft(z_rt) * (1 - change)
    return dict(size=(ww, wh), w=(clean * w_flat).astype(np.float32), w_tex=clean.astype(np.float32),
                change=change, o=o, p=p, flows=flows is not None, given=given is not None)


# ------------------------------------------------ stage 2: grade

def fit_grade(o, p, w, bins=16, min_weight=0.5):
    """Correction to add to the edit's Oklab, as a curve over the edit's lightness,
    flat beyond the tones it was measured on. None when nothing can be trusted."""
    lo, lp = to_oklab(o), to_oklab(p)
    sel = w >= min_weight
    n = int(sel.sum())
    if n < 200:
        return None, {"n": n}
    diff, L = (lo - lp)[sel], lp[..., 0][sel]
    info: dict = {"n": n, "L_range": (float(np.percentile(L, 1)), float(np.percentile(L, 99)))}
    if n < bins * 200:
        off = np.median(diff, 0)
        info.update(centres=np.array([info["L_range"][0], info["L_range"][1]]), corr=np.stack([off, off]))
    else:
        edges = np.quantile(L, np.linspace(0.01, 0.99, bins + 1))       # equal-count bins
        centres, corr = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            s = (L >= a) & (L < b)
            if s.sum() >= 200:
                centres.append(np.median(L[s]))
                corr.append(np.median(diff[s], 0))
        corr = np.array(corr)
        corr = np.stack([np.convolve(np.pad(corr[:, c], 1, mode="edge"), [0.25, 0.5, 0.25], "valid")
                         for c in range(3)], -1)
        info.update(centres=np.array(centres), corr=corr)
    c_, k_ = info["centres"], info["corr"]
    return (lambda Lq: np.stack([np.interp(Lq, c_, k_[:, i]) for i in range(3)], -1)), info


# ------------------------------------------------ stage 3: edge field

def profile(d_in, kind, width):
    """Weight of the edge correction by distance inside the mask edge (px)."""
    t = np.clip(d_in / max(width, 1e-6), 0.0, 1.0)
    if kind == "entire":
        return np.ones_like(d_in)
    if kind == "linear":
        return 1.0 - t
    if kind == "smooth":
        return 1.0 - t * t * (3 - 2 * t)
    if kind == "decay":
        return np.exp(-3.0 * np.maximum(d_in, 0) / max(width, 1e-6))
    raise ValueError(f"edge profile is smooth, linear, decay or entire, not {kind!r}")


def edge_field(original, graded, alpha, tr, kind="smooth", width=128, band=EDGE_BAND):
    """What the grade leaves at the edge of the pasted area, as an Oklab offset field
    (H, W, 3): measured on trusted pixels near the edge, spread smoothly, faded inward."""
    H, W = original.shape[:2]
    ww, wh = tr["size"]
    s = ww / W
    g = cv2.resize(graded, (ww, wh), interpolation=cv2.INTER_AREA)
    m = cv2.resize((alpha > 0.5).astype(np.uint8), (ww, wh), interpolation=cv2.INTER_NEAREST)
    d = signed_distance(m)
    r = to_oklab(tr["o"]) - to_oklab(g)
    wt = tr["w"] * (np.abs(d) <= band * s)
    if wt.sum() < 50:
        return np.zeros((H, W, 3), np.float32)
    sig = max(2.0, band * s)
    # Where nothing near the edge was trusted (the change reached past the mask
    # there), the field fades to nothing rather than borrowing a correction from
    # far away: it put a teal cast into a lantern's glow when it did.
    field = np.zeros_like(r)
    for c in range(3):
        f1, d1 = nconv(r[..., c], wt, sig)
        f3, d3 = nconv(r[..., c], wt, 3 * sig)
        a1, a3 = d1 / (d1 + 0.05), d3 / (d3 + 0.05)
        field[..., c] = a1 * f1 + (1 - a1) * a3 * f3
    prof = profile(-d / s, kind, width)
    prof[d > 0] = 1.0                                                     # outside: the feather fades it
    return cv2.resize(field * prof[..., None], (W, H), interpolation=cv2.INTER_CUBIC)


# ------------------------------------------------ stage 4: texture

def _levels(h, w, most=5):
    return int(max(0, min(most, np.floor(np.log2(max(min(h, w), 1))) - 4)))


def lap_pyr(x, levels):
    out, g = [], x.astype(np.float32)
    for _ in range(levels):
        down = cv2.pyrDown(g)
        out.append(g - cv2.pyrUp(down, dstsize=(g.shape[1], g.shape[0])))
        g = down
    return out, g


def lap_rebuild(bands, base):
    g = base
    for b in reversed(bands):
        g = cv2.pyrUp(g, dstsize=(b.shape[1], b.shape[0])) + b
    return g


def _at(m, like):
    return cv2.resize(m.astype(np.float32), (like.shape[1], like.shape[0]), interpolation=cv2.INTER_AREA)


def texture_gains(original, processed, weight, sigma=24.0, cap=(0.5, 2.0)):
    """Local gain per detail band of Oklab lightness so the processed detail energy
    matches the original's on trusted pixels. Returns the maps and the global gains."""
    n = _levels(*original.shape[:2])
    po, _ = lap_pyr(to_oklab(original)[..., 0], n)
    pp, _ = lap_pyr(to_oklab(processed)[..., 0], n)
    maps, glob = [], []
    for k, (a, b) in enumerate(zip(po, pp)):
        w = _at(weight, a)
        sel = w >= 0.5
        g = float(np.sqrt(np.mean(a[sel] ** 2) / max(np.mean(b[sel] ** 2), 1e-12))) if sel.sum() > 50 else 1.0
        glob.append(g)
        s = max(2.0, sigma / 2 ** k)
        ea, den = nconv(a * a, w, s)
        eb, _ = nconv(b * b, w, s)
        conf = np.clip(den / 0.2, 0, 1)
        gm = conf * np.sqrt(ea / np.maximum(eb, 1e-12)) + (1 - conf) * g
        maps.append(np.clip(gm, *cap).astype(np.float32))
    return maps, glob


def apply_texture(processed, gains):
    lab = to_oklab(processed)
    bands, base = lap_pyr(lab[..., 0], len(gains))
    lab[..., 0] = lap_rebuild([b * g for b, g in zip(bands, gains)], base)
    return from_oklab(lab)


def energy_blend(original, processed, alpha, sigma=4.0):
    """original * (1 - a) + processed * a, except that the detail bands of lightness
    are rescaled so their local energy is what each side would have on its own:
    two different textures averaged in a feather partly cancel (a plain feather
    lost a quarter of the fine detail), which shows as a soft halo."""
    a3 = alpha[..., None]
    plain = original * (1 - a3) + processed * a3
    n = _levels(*original.shape[:2])
    if n == 0:
        return plain
    lo, lp = to_oklab(original), to_oklab(processed)
    ab = lo[..., 1:] * (1 - a3) + lp[..., 1:] * a3
    bo, base_o = lap_pyr(lo[..., 0], n)
    bp, base_p = lap_pyr(lp[..., 0], n)
    g = lambda v: cv2.GaussianBlur(v, (0, 0), sigma)
    out = []
    for x, y in zip(bo, bp):
        a = _at(alpha, x)
        vo, vp, cov = g(x * x), g(y * y), g(x * y)
        mix = x * (1 - a) + y * a
        vm = (1 - a) ** 2 * vo + a * a * vp + 2 * a * (1 - a) * cov
        out.append(mix * np.clip(np.sqrt(((1 - a) * vo + a * vp) / np.maximum(vm, 1e-12)), 1.0, 2.0))
    ab_ = _at(alpha, base_o)
    L = lap_rebuild(out, base_o * (1 - ab_) + base_p * ab_)
    res = from_oklab(np.concatenate([L[..., None], ab], -1))
    keep = (alpha <= 0)[..., None]
    full = (alpha >= 1)[..., None]
    return np.where(keep, original, np.where(full, processed, res)).astype(np.float32)


# ------------------------------------------------ the whole match

def band_completeness(mask, band):
    """Share of the band a mask would have if the frame went on past its border."""
    d = signed_distance(mask)
    have = ((d > 0) & (d < band)).sum()
    P = int(band) + 8
    dp = signed_distance(cv2.copyMakeBorder(mask.astype(np.uint8), P, P, P, P, cv2.BORDER_REPLICATE))
    return float(have / max(((dp > 0) & (dp < band)).sum(), 1))


def match(original, processed, hard_mask, alpha, flows=None, color="grade + edge",
          grade_strength=1.0, edge_profile="smooth", edge_width=128, texture="match",
          given=None):
    """The processed cut with its color and texture matched to the original.

    original, processed: (h, w, 3) float32 sRGB, processed already aligned.
    hard_mask: the painted mask before grow and feather (the prior); alpha: the
    blend weight it will be pasted with; given: a change mask from outside (see
    `trust`). Returns the corrected cut, the change mask (h, w) and a report
    for the measurement view. report["correction"] applies the same correction
    to another frame of this size (see `apply_correction`)."""
    h, w = original.shape[:2]
    tr = trust(original, processed, flows, prior=hard_mask, given=given)
    ww, wh = tr["size"]
    s = ww / w
    change = cv2.resize(tr["change"], (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    hard_w = cv2.resize((hard_mask > 0.5).astype(np.uint8), (ww, wh), interpolation=cv2.INTER_NEAREST)
    d = signed_distance(hard_w)
    band = (d > 0) & (d < BAND * s)
    n_band = int((band & (tr["w"] >= 0.5)).sum())
    comp = band_completeness(hard_w, BAND * s)
    ch_w = tr["change"] > 0
    outside = float((ch_w & (hard_w == 0)).sum() / max(ch_w.sum(), 1))
    warnings = []
    conf = min(1.0, comp / 0.6) * min(1.0, n_band / max(0.01 * ww * wh, 1))
    # A change found past the mask may be a detection gone wrong, so it lowers
    # confidence; a given one was drawn on purpose and is only kept out.
    if outside > 0.05 and not tr["given"]:
        warnings.append(f"the change reaches past the mask ({outside * 100:.0f}% of it)")
        conf *= 0.5
    if comp < 0.3:
        warnings.append("little band around the mask: correction reduced")
    if not tr["flows"] and not tr["given"]:
        warnings.append("no optical flow: change found by color only")

    corr: dict = dict(grade=None, field=None, texture=None)
    out = processed
    info: dict = {"n": 0}
    gains: list | None = None
    if color in ("grade + edge", "grade"):
        curve, info = fit_grade(tr["o"], tr["p"], tr["w"])
        if curve is not None and grade_strength > 0:
            corr["grade"] = (curve, grade_strength * conf)
        elif curve is None:
            warnings.append("too few unchanged pixels to measure the grade")
        out = apply_correction(out, corr)
    if color in ("grade + edge", "edge"):
        corr["field"] = edge_field(original, out, alpha, tr, edge_profile, edge_width)
        out = from_oklab(to_oklab(out) + corr["field"])
    if texture == "match":
        gm, gains = texture_gains(original, out, cv2.resize(tr["w_tex"], (w, h), interpolation=cv2.INTER_LINEAR))
        if gm:
            corr["texture"] = gm
            out = apply_texture(out, gm)
    report: dict = dict(tr=tr, hard=hard_w, band=band, n_band=n_band, completeness=comp, outside=outside,
                  confidence=conf, warnings=warnings, info=info, field=corr["field"], gains=gains,
                  color=color, texture=texture, grade_strength=grade_strength, correction=corr)
    return out.astype(np.float32), change, report


def apply_correction(img, corr):
    """The correction `match` measured, applied to another (h, w, 3) frame of
    the same size: the grade by the frame's own lightness, then the edge field
    and the texture gains by position. Those two are smooth over tens of px,
    so a frame a few px off the measured one (the edit before alignment) takes
    them as well as the measured one."""
    out = img
    if corr["grade"] is not None:
        curve, k = corr["grade"]
        lab = to_oklab(out)
        out = from_oklab(lab + k * curve(lab[..., 0]))
    if corr["field"] is not None:
        out = from_oklab(to_oklab(out) + corr["field"])
    if corr["texture"] is not None:
        out = apply_texture(out, corr["texture"])
    return out.astype(np.float32)


# ------------------------------------------------ the measurement view

def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has only the small bitmap font
        return ImageFont.load_default()


def _diverging(x, lim=0.06):
    t = np.clip(x / lim, -1, 1)[..., None]
    blue, white, red = np.array([40, 90, 200]), np.array([245, 245, 245]), np.array([200, 50, 40])
    return np.where(t < 0, white + (blue - white) * -t, white + (red - white) * t).astype(np.uint8)


def render_view(original, processed, corrected, alpha, report, open_sides=(), height=560):
    """Where it measured, the plateau before and after, the grade curve and the
    numbers that set the strength. Returns (h, w, 3) float32 in [0, 1]."""
    tr = report["tr"]
    ww, wh = tr["size"]
    height = int(min(height, 1800 * wh / (3 * ww)))
    pw = max(1, int(round(ww * height / wh)))
    rs = lambda a, interp=cv2.INTER_AREA: cv2.resize(a, (pw, height), interpolation=interp)
    at_w = lambda a: cv2.resize(a, (ww, wh), interpolation=cv2.INTER_AREA)
    hard, change = report["hard"], tr["change"] > 0
    out_change = change & (hard == 0)

    p1 = rs(tr["o"]) * 0.45
    def tint(m, rgb, a):
        m = rs(m.astype(np.uint8), cv2.INTER_NEAREST) > 0
        p1[m] = p1[m] * (1 - a) + np.array(rgb) / 255 * a
    tint(report["band"] & (tr["w"] >= 0.5), (60, 200, 90), 0.55)
    tint(report["band"] & (tr["w"] < 0.5), (200, 190, 60), 0.35)
    tint(change & (hard > 0), (230, 60, 50), 0.6)
    tint(out_change, (255, 150, 0), 0.85)
    p1 = (np.clip(p1, 0, 1) * 255).astype(np.uint8)
    cnt, _ = cv2.findContours(rs(hard, cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(p1, cnt, -1, (255, 255, 255), 2)
    ends = {"top": ((0, 1), (pw - 1, 1)), "bottom": ((0, height - 2), (pw - 1, height - 2)),
            "left": ((1, 0), (1, height - 1)), "right": ((pw - 2, 0), (pw - 2, height - 1))}
    for side in open_sides:
        (ax, ay), (bx, by) = ends[side]
        n = max(2, int(max(abs(bx - ax), abs(by - ay)) // 12))
        for i in range(0, n, 2):
            u, v = i / n, min(1, (i + 1) / n)
            cv2.line(p1, (int(ax + (bx - ax) * u), int(ay + (by - ay) * u)),
                     (int(ax + (bx - ax) * v), int(ay + (by - ay) * v)), (0, 220, 255), 3)

    lo = to_oklab(tr["o"])[..., 0]
    def plateau(img_full, only=None):
        r = lo - to_oklab(at_w(img_full))[..., 0]
        img = _diverging(cv2.GaussianBlur(rs(r.astype(np.float32)), (0, 0), 2.0))
        m = rs(change.astype(np.uint8), cv2.INTER_NEAREST) > 0
        img[m] = (img[m] * 0.25 + 40).astype(np.uint8)
        if only is not None:
            img[rs((at_w(only) > 0.02).astype(np.uint8), cv2.INTER_NEAREST) == 0] = 32
        cv2.drawContours(img, cnt, -1, (0, 0, 0), 1)
        return img, r
    p2, _ = plateau(processed)
    p3, r_after = plateau(corrected, alpha)

    f, fs = _font(15), _font(13)
    curve_w, row_h, text_w = 420, 250, 720
    gimg = Image.new("RGB", (curve_w, row_h), (24, 24, 24))
    dr = ImageDraw.Draw(gimg)
    info = report["info"]
    if "corr" in info:
        cv_ = info["corr"]
        C0, C1 = min(-0.01, float(cv_.min()) - 0.01), max(0.02, float(cv_.max()) + 0.01)
        X = lambda L: 50 + L / 0.9 * (curve_w - 70)
        Y = lambda c: 30 + (C1 - c) / (C1 - C0) * (row_h - 70)
        lo_r, hi_r = info["L_range"]
        dr.rectangle([X(lo_r), Y(C1), X(hi_r), Y(C0)], fill=(40, 52, 40))
        dr.line([X(0), Y(0), X(0.9), Y(0)], fill=(110, 110, 110))
        for c, col in enumerate([(240, 240, 240), (220, 90, 200), (220, 200, 60)]):
            pts = [(X(L), Y(v)) for L, v in zip(info["centres"], cv_[:, c])]
            dr.line([(X(0), pts[0][1])] + pts + [(X(0.9), pts[-1][1])], fill=col, width=2)
        for v in np.arange(np.ceil(C0 / 0.02) * 0.02, C1, 0.02):
            dr.text((8, Y(v) - 7), f"{v:+.2f}", fill=(150, 150, 150), font=fs)
        dr.text((X(lo_r) + 4, Y(C0) - 18), "tones in the band", fill=(140, 190, 140), font=fs)
        dr.text((X(0.45) - 40, row_h - 34), "edit lightness", fill=(150, 150, 150), font=fs)
    else:
        dr.text((50, row_h // 2), "no grade measured", fill=(150, 150, 150), font=f)
    dr.text((50, 6), "grade by edit lightness: L white, a magenta, b yellow", fill=(230, 230, 230), font=fs)

    sm = cv2.GaussianBlur(r_after.astype(np.float32), (0, 0), max(2.0, 16.0 * ww / 1024))
    sel = report["band"] & (tr["w"] >= 0.5)
    left = float(np.ptp(np.percentile(sm[sel], [5, 95]))) if sel.sum() > 50 else float("nan")
    fld = report["field"]
    edge_rms = float(np.sqrt(np.mean(at_w(fld)[..., 0][hard > 0] ** 2))) if fld is not None and (hard > 0).any() else 0.0
    mid = f"{np.interp(0.5, info['centres'], info['corr'][:, 0]):+.3f}" if "corr" in info else "-"
    conf, warn = report["confidence"], report["warnings"]
    ok, orange, grey = (60, 200, 90), (255, 150, 0), (230, 230, 230)
    lines = [
        (f"color: {report['color']}, texture: {report['texture']}; painted mask white; "
         f"open sides cyan: {', '.join(open_sides) or 'none'}", grey),
        (f"band 0-{BAND} cut px: {report['n_band']} trusted px (green), "
         f"{report['completeness'] * 100:.0f}% of a full ring", ok),
        (f"changed (red): {change.mean() * 100:.1f}% of the cut"
         + (", from the given mask and color" if tr["given"] else "")
         + f"; outside the mask (orange): {report['outside'] * 100:.0f}% of it",
         orange if report["outside"] > 0.05 and not tr["given"] else (230, 60, 50)),
        (f"grade at mid tones: L {mid}; plateau left in the band (p5-p95): {left:.3f}", grey),
        (f"plateau: red = edit darker, blue = brighter (Oklab L, +-0.06); edge field rms {edge_rms:.3f}", grey),
        ("texture gain per band: " + (" ".join(f"{g:.2f}" for g in report["gains"]) if report["gains"] else "-"), grey),
        (f"confidence {conf:.2f} (scales the grade)", ok if conf >= 0.75 else orange),
    ] + [(f"WARNING: {w_}", orange) for w_ in warn]
    row_h = max(row_h, 16 + 30 * len(lines))
    timg = Image.new("RGB", (text_w, row_h), (24, 24, 24))
    dt = ImageDraw.Draw(timg)
    for i, (t, col) in enumerate(lines):
        dt.text((12, 12 + i * 30), t, fill=col, font=f)

    total_w = max(pw * 3 + 16, curve_w + 8 + text_w)
    canvas = Image.new("RGB", (total_w, 28 + height + 8 + row_h), (24, 24, 24))
    dc = ImageDraw.Draw(canvas)
    x0 = (total_w - (pw * 3 + 16)) // 2
    for i, (img, title) in enumerate([(p1, "measured"), (p2, "plateau before"), (p3, "after, pasted area")]):
        canvas.paste(Image.fromarray(img), (x0 + i * (pw + 8), 28))
        dc.text((x0 + i * (pw + 8) + 4, 6), title, fill=(230, 230, 230), font=f)
    canvas.paste(gimg, (0, 28 + height + 8))
    canvas.paste(timg, (curve_w + 8, 28 + height + 8))
    return np.asarray(canvas, np.float32) / 255.0
