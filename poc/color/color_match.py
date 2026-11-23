"""Color match after an edit model: prototype of stages 1-3.

1. trust   - which pixels the edit left alone: color z and round-trip z against
             2D plateau/floor maps, times flatness. Also gives a change mask.
2. grade   - the edit's overall grade, fitted on trusted pixels as an offset per
             tone in Oklab (as a function of the edit's lightness), held flat
             beyond the tones it was measured on.
3. edge    - what the grade leaves at the mask edge, as a smooth field, applied
             with a profile that fades inward from the edge.

Frames are float32 (h, w, 3) sRGB in [0, 1]; `processed` is already aligned to
`original` (TPS Paste align). Proof of concept, not part of the node pack.
"""
import numpy as np
import cv2

from common import srgb_to_oklab, oklab_to_srgb, nconv, signed_distance, flatness


# ------------------------------------------------ stage 1: trust

def _soft(z, lo=2.0, hi=4.0):
    """1 below lo floors, 0 above hi."""
    return np.clip((hi - z) / (hi - lo), 0.0, 1.0).astype(np.float32)


def fill(value, weight, sigma, scales=(1, 4, 16)):
    """Normalized convolution that reaches deep into large gaps: where the local
    support is thin, the next coarser scale takes over, and finally the weighted
    global median."""
    sel = weight > 0.5
    out = np.full(value.shape, np.median(value[sel]) if sel.any() else 0.0, np.float32)
    for s in reversed(scales):
        v, den = nconv(value, weight, sigma * s)
        a = den / (den + 0.05)
        out = a * v + (1 - a) * out
    return out


def robust_z(x, sigma, floor_min, one_sided=False, iters=3, w0=None):
    """How many local floors each pixel sits from its local plateau.

    Seeded by a global median/MAD pass, then refined locally; a pixel once
    excluded stays excluded, so an object as big as the window cannot raise its
    own floor and hide. w0 (0-1) is a prior: pixels with w0 = 0 (inside the
    painted mask) never help define the plateau, which is filled in there from
    outside. x: (h, w) or (h, w, c)."""
    x = x if x.ndim == 3 else x[..., None]
    base = np.ones(x.shape[:2], np.float32) if w0 is None else w0.astype(np.float32)
    flat_x = x[base > 0.5]
    med = np.median(flat_x, 0)
    mad = 1.4826 * np.median(np.abs(flat_x - med), 0) + floor_min
    dev0 = (x - med) if one_sided else np.abs(x - med)
    w = np.minimum(base, _soft((dev0 / mad).max(-1)))
    for _ in range(iters):
        plateau = np.stack([fill(x[..., c], w, sigma) for c in range(x.shape[-1])], -1)
        dev = x - plateau if one_sided else np.abs(x - plateau)
        floor = np.stack([1.4826 * fill(np.abs(x[..., c] - plateau[..., c]), w, sigma)
                          for c in range(x.shape[-1])], -1) + floor_min
        z = (dev / floor).max(-1)
        w = np.minimum(w, _soft(z))
    return z, plateau, floor, w


def roundtrip_error(flows, a, b):
    fwd, bwd = flows(a, b)
    h, w = fwd.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    back = np.stack([cv2.remap(bwd[i], xx + fwd[0], yy + fwd[1], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE) for i in range(2)])
    return np.hypot(fwd[0] + back[0], fwd[1] + back[1]).astype(np.float32)


def _clean(m, min_share=0.0005, grow=4):
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_share * m.size]
    m = np.isin(lab, keep).astype(np.uint8)
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # fill holes
    cv2.drawContours(m, cnt, -1, 1, -1)
    if grow:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
    return m


def trust(original, processed, flows=None, work=1024, prior=None):
    """Per-pixel trust (0-1) and a change mask, at working size.
    prior: the painted (hard) mask at full size; its inside never defines the plateau."""
    H, W = original.shape[:2]
    k = work / max(H, W)
    ww, wh = int(round(W * k / 8)) * 8, int(round(H * k / 8)) * 8
    o = cv2.resize(original, (ww, wh), interpolation=cv2.INTER_AREA)
    p = cv2.resize(processed, (ww, wh), interpolation=cv2.INTER_AREA)
    sigma = work / 80.0
    w0 = None
    if prior is not None:
        w0 = 1.0 - (cv2.resize(prior.astype(np.uint8), (ww, wh), interpolation=cv2.INTER_NEAREST) > 0)
    z_col, plateau, floor, _ = robust_z(srgb_to_oklab(p) - srgb_to_oklab(o), sigma, floor_min=0.004, w0=w0)
    if flows is not None:
        z_rt = robust_z(roundtrip_error(flows, o, p), sigma, floor_min=0.05, one_sided=True, w0=w0)[0]
    else:
        z_rt = np.zeros_like(z_col)
    g = flatness(o)
    g50, g90 = np.percentile(g, 50), np.percentile(g, 90)
    w_flat = np.clip((g90 - g) / (g90 - g50 + 1e-6), 0.0, 1.0)
    change = _clean((z_col > 4) | (z_rt > 4))
    w = _soft(z_col) * _soft(z_rt) * w_flat * (1 - change)
    return dict(size=(ww, wh), w=w.astype(np.float32), change=change, z_col=z_col, z_rt=z_rt,
                plateau=plateau, floor=floor, o=o, p=p)


# ------------------------------------------------ stage 2: grade

def fit_grade(o, p, w, mode="tone", bins=16, min_weight=0.5):
    """Correction to add to the processed frame's Oklab, as a function of its
    lightness. mode "offset": one constant per component; "tone": a curve of
    offsets over lightness, flat beyond the tones it was measured on."""
    lo, lp = srgb_to_oklab(o), srgb_to_oklab(p)
    sel = w >= min_weight
    diff = (lo - lp)[sel]
    L = lp[..., 0][sel]
    info = dict(n=int(sel.sum()), L_range=(float(np.percentile(L, 1)), float(np.percentile(L, 99))))
    if mode == "offset" or sel.sum() < bins * 200:
        off = np.median(diff, 0)
        info["offset"] = off
        return (lambda Lq: np.broadcast_to(off, Lq.shape + (3,))), info
    edges = np.quantile(L, np.linspace(0.01, 0.99, bins + 1))   # equal-count bins
    centres, corr = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (L >= a) & (L < b)
        if s.sum() >= 200:
            centres.append(np.median(L[s]))
            corr.append(np.median(diff[s], 0))
    centres, corr = np.array(centres), np.array(corr)
    corr = np.stack([np.convolve(np.pad(corr[:, c], 1, mode="edge"), [0.25, 0.5, 0.25], "valid")
                     for c in range(3)], -1)                     # light smoothing
    info.update(centres=centres, corr=corr, offset=np.median(diff, 0))
    curve = lambda Lq: np.stack([np.interp(Lq, centres, corr[:, c]) for c in range(3)], -1)  # flat beyond
    return curve, info


def apply_grade(processed, curve, strength=1.0):
    lab = srgb_to_oklab(processed)
    return np.clip(oklab_to_srgb(lab + strength * curve(lab[..., 0])), 0, 1).astype(np.float32)


# ------------------------------------------------ stage 3: edge field

def profile(d_in, kind, width):
    """Weight of the edge field by distance inside the mask edge (px)."""
    t = np.clip(d_in / max(width, 1e-6), 0.0, 1.0)
    if kind == "entire":
        return np.ones_like(d_in)
    if kind == "none":
        return np.zeros_like(d_in)
    if kind == "linear":
        return 1.0 - t
    if kind == "smooth":
        return 1.0 - t * t * (3 - 2 * t)
    if kind == "decay":
        return np.exp(-3.0 * np.maximum(d_in, 0) / max(width, 1e-6))
    raise ValueError(kind)


def edge_field(original, graded, mask, tr, band=48, kind="smooth", width=64):
    """The residual at the mask edge, spread smoothly and faded inward.
    mask: (H, W) blend region (> 0.5 = inside). Returns an Oklab offset field (H, W, 3)."""
    H, W = original.shape[:2]
    ww, wh = tr["size"]
    s = ww / W
    o = tr["o"]
    g = cv2.resize(graded, (ww, wh), interpolation=cv2.INTER_AREA)
    m = cv2.resize((mask > 0.5).astype(np.uint8), (ww, wh), interpolation=cv2.INTER_NEAREST)
    d = signed_distance(m)                                     # + outside, working px
    near = np.abs(d) <= band * s
    r = srgb_to_oklab(o) - srgb_to_oklab(g)
    wt = tr["w"] * near
    sig = max(2.0, band * s)
    field = np.zeros_like(r)
    den_total = None
    for c in range(3):
        f1, den1 = nconv(r[..., c], wt, sig)
        f3, _ = nconv(r[..., c], wt, 3 * sig)               # fills deep inside large masks
        a = np.clip(den1 / (0.05 + den1), 0, 1)
        field[..., c] = a * f1 + (1 - a) * f3
        den_total = den1
    prof = profile(-d / s, kind, width)                        # inside distance in full-size px
    prof[d > 0] = 1.0                                          # outside the edge: full (the blend fades it)
    field = field * prof[..., None]
    return cv2.resize(field, (W, H), interpolation=cv2.INTER_CUBIC), den_total


def apply_field(img, field):
    return np.clip(oklab_to_srgb(srgb_to_oklab(img) + field), 0, 1).astype(np.float32)


# ------------------------------------------------ stage 4: texture (Oklab L)

def lap_pyr(x, levels):
    """Laplacian pyramid: `levels` detail bands (finest first) + the residual base."""
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


def texture_gains(original, processed, weight, levels=5, local=True, sigma=24.0, cap=(0.5, 2.0)):
    """Per band, how much to scale the processed frame's detail so its energy
    matches the original's on trusted pixels. Returns one gain map per band
    (constant maps when local=False) and the global gains."""
    po, _ = lap_pyr(srgb_to_oklab(original)[..., 0], levels)
    pp, _ = lap_pyr(srgb_to_oklab(processed)[..., 0], levels)
    maps, glob = [], []
    for k, (a, b) in enumerate(zip(po, pp)):
        w = _at(weight, a)
        sel = w >= 0.5
        g = float(np.sqrt(np.mean(a[sel] ** 2) / max(np.mean(b[sel] ** 2), 1e-12)))
        glob.append(g)
        if local:
            s = max(2.0, sigma / 2 ** k)              # same footprint in full-size px at every level
            ea, den = nconv(a * a, w, s)
            eb, _ = nconv(b * b, w, s)
            gm = np.sqrt(ea / np.maximum(eb, 1e-12))
            conf = np.clip(den / 0.2, 0, 1)            # far from any trusted pixel -> the global gain
            gm = conf * gm + (1 - conf) * g
        else:
            gm = np.full(a.shape, g, np.float32)
        maps.append(np.clip(gm, *cap).astype(np.float32))
    return maps, glob


def apply_texture(processed, gains, profile_map=None):
    """Scale the processed frame's detail bands; profile_map (H, W) 0-1 fades the
    change (1 = full gain, 0 = untouched), as the edge profile does for color."""
    lab = srgb_to_oklab(processed)
    bands, base = lap_pyr(lab[..., 0], len(gains))
    out = []
    for b, g in zip(bands, gains):
        if profile_map is not None:
            p = _at(profile_map, b)
            g = 1.0 + (g - 1.0) * p
        out.append(b * g)
    lab[..., 0] = lap_rebuild(out, base)
    return np.clip(oklab_to_srgb(lab), 0, 1).astype(np.float32)


def blend(original, processed, alpha, mode="alpha", levels=5, sigma=4.0):
    """Paste processed over original through alpha (H, W).
    alpha     - plain mix, what TPS Paste does now
    multiband - Burt-Adelson: each band mixed with alpha blurred to its own scale
    energy    - plain mix of the base; the detail bands mixed, then rescaled so their
                local energy is what the two sides would have on their own:
                (1-a) var_o + a var_p, using the measured local covariance."""
    a3 = alpha[..., None]
    if mode == "alpha":
        return original * (1 - a3) + processed * a3
    lo, lp = srgb_to_oklab(original), srgb_to_oklab(processed)
    ab = lo[..., 1:] * (1 - a3) + lp[..., 1:] * a3
    bo, base_o = lap_pyr(lo[..., 0], levels)
    bp, base_p = lap_pyr(lp[..., 0], levels)
    out = []
    for k, (x, y) in enumerate(zip(bo, bp)):
        if mode == "multiband":
            a = cv2.GaussianBlur(_at(alpha, x), (0, 0), 1.0 + 0.5 * k) if k else _at(alpha, x)
            out.append(x * (1 - a) + y * a)
        elif mode == "energy":
            a = _at(alpha, x)
            s = max(1.5, sigma)
            g = lambda v: cv2.GaussianBlur(v, (0, 0), s)
            vo, vp, cov = g(x * x), g(y * y), g(x * y)
            mix = x * (1 - a) + y * a
            vm = (1 - a) ** 2 * vo + a * a * vp + 2 * a * (1 - a) * cov
            target = (1 - a) * vo + a * vp
            out.append(mix * np.clip(np.sqrt(target / np.maximum(vm, 1e-12)), 1.0, 2.0))
        else:
            raise ValueError(mode)
    if mode == "multiband":
        ab_ = cv2.GaussianBlur(_at(alpha, base_o), (0, 0), 1.0 + 0.5 * levels)
    else:
        ab_ = _at(alpha, base_o)
    base = base_o * (1 - ab_) + base_p * ab_
    L = lap_rebuild(out, base)
    return np.clip(oklab_to_srgb(np.concatenate([L[..., None], ab], -1)), 0, 1).astype(np.float32)
