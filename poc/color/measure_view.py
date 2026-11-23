"""A measurement preview, drawn the way TPS Paste would draw it (numpy, OpenCV, PIL
only): where the plateau was measured, what the edit changed, the plateau before
and after correction, the fitted grade, and the numbers that decide the strength.

usage: measure_view.py <results dir> <out dir> <prefix> <painted mask: file:<png> | ellipse:x0,y0,x1,y1>
"""
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw

from common import srgb_to_oklab, signed_distance
import color_match as cm

sys.path.insert(0, __file__.rsplit("/poc/", 1)[0])
from tps_cut_paste.cut_paste import _font


def diverging(x, lim):
    """Blue (edit brighter) - white - red (edit darker), as RGB uint8."""
    t = np.clip(x / lim, -1, 1)[..., None]
    blue, white, red = np.array([40, 90, 200]), np.array([245, 245, 245]), np.array([200, 50, 40])
    return np.where(t < 0, white + (blue - white) * -t, white + (red - white) * t).astype(np.uint8)


def band_completeness(rough, band_px):
    """Share of the band a mask would have if the image went on past its border."""
    d = signed_distance(rough)
    have = ((d > 0) & (d < band_px)).sum()
    P = band_px + 8
    dp = signed_distance(cv2.copyMakeBorder(rough, P, P, P, P, cv2.BORDER_REPLICATE))
    full = ((dp > 0) & (dp < band_px)).sum()
    return have / max(full, 1)


def render(O, E, rough, tr, info, G, field, gains, alpha=None, band_px=80, height=560):
    """O, E, G: (H, W, 3) frames (original, aligned edit, corrected edit); rough: painted
    mask; tr: stage-1 result; info: grade fit info; field: edge field (H, W, 3);
    gains: global texture gain per band."""
    H, W = O.shape[:2]
    # the cut: the painted mask's box, doubled (context_expand 2), clipped to the image
    ys, xs = np.nonzero(rough)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    hy, hx = np.ptp(ys), np.ptp(xs)
    y0, y1 = int(max(0, cy - hy)), int(min(H, cy + hy))
    x0, x1 = int(max(0, cx - hx)), int(min(W, cx + hx))
    open_sides = [s for s, on in [("top", y0 == 0), ("bottom", y1 == H), ("left", x0 == 0), ("right", x1 == W)] if on]
    crop = lambda a: a[y0:y1, x0:x1]
    s = height / (y1 - y0)
    pw = int(round((x1 - x0) * s))
    rs = lambda a, interp=cv2.INTER_AREA: cv2.resize(a, (pw, height), interpolation=interp)

    w = cv2.resize(tr["w"], (W, H), interpolation=cv2.INTER_LINEAR)
    change = cv2.resize(tr["change"].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    d = signed_distance(rough)
    band = (d > 0) & (d < band_px)
    out_change = change & (rough == 0)

    # panel 1: where it was measured
    base = rs(crop(O)) * 0.45
    def tint(img, m, rgb, a):
        m = rs(crop(m.astype(np.uint8)), cv2.INTER_NEAREST) > 0
        img[m] = img[m] * (1 - a) + np.array(rgb) / 255 * a
    tint(base, band & (w >= 0.5), (60, 200, 90), 0.55)          # band, trusted
    tint(base, band & (w < 0.5), (200, 190, 60), 0.35)          # band, not trusted
    tint(base, change & (rough > 0), (230, 60, 50), 0.6)        # changed, inside the painted mask
    tint(base, out_change, (255, 150, 0), 0.85)                 # changed, outside it
    p1 = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    cnt, _ = cv2.findContours(rs(crop(rough), cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(p1, cnt, -1, (255, 255, 255), 2)
    for side in open_sides:                                      # open sides: dashed cyan
        pts = {"top": ((0, 1), (pw - 1, 1)), "bottom": ((0, height - 2), (pw - 1, height - 2)),
               "left": ((1, 0), (1, height - 1)), "right": ((pw - 2, 0), (pw - 2, height - 1))}[side]
        (ax, ay), (bx, by) = pts
        n = int(max(abs(bx - ax), abs(by - ay)) // 12)
        for i in range(0, n, 2):
            p_, q_ = i / n, min(1, (i + 1) / n)
            cv2.line(p1, (int(ax + (bx - ax) * p_), int(ay + (by - ay) * p_)),
                     (int(ax + (bx - ax) * q_), int(ay + (by - ay) * q_)), (0, 220, 255), 3)

    # panels 2-3: the plateau (original - edit, Oklab L) before and after correction
    lo = srgb_to_oklab(O)[..., 0]
    before = lo - srgb_to_oklab(E)[..., 0]
    after = lo - srgb_to_oklab(G)[..., 0]
    def plateau(r, only=None):
        img = diverging(cv2.GaussianBlur(rs(crop(r.astype(np.float32))), (0, 0), 2.0), 0.06)
        m = rs(crop(change.astype(np.uint8)), cv2.INTER_NEAREST) > 0
        img[m] = (img[m] * 0.25 + 40).astype(np.uint8)          # the changed object, greyed out
        if only is not None:                                      # outside the pasted area nothing changes
            o_ = rs(crop((only > 0.02).astype(np.uint8)), cv2.INTER_NEAREST) == 0
            img[o_] = 32
        cv2.drawContours(img, cnt, -1, (0, 0, 0), 1)
        return img
    p2, p3 = plateau(before), plateau(after, alpha)

    # bottom row: the grade curve and the numbers
    curve_w, row_h = 420, 250
    gimg = Image.new("RGB", (curve_w, row_h), (24, 24, 24))
    dr = ImageDraw.Draw(gimg)
    f, fs = _font(15), _font(13)
    cv_ = info["corr"]
    L0, L1 = 0.0, 0.9
    C0 = min(-0.01, float(cv_.min()) - 0.01); C1 = max(0.02, float(cv_.max()) + 0.01)
    X = lambda L: 50 + (L - L0) / (L1 - L0) * (curve_w - 70)
    Y = lambda c: 30 + (C1 - c) / (C1 - C0) * (row_h - 70)
    lo_r, hi_r = info["L_range"]
    dr.rectangle([X(lo_r), Y(C1), X(hi_r), Y(C0)], fill=(40, 52, 40))
    dr.line([X(L0), Y(0), X(L1), Y(0)], fill=(110, 110, 110))
    for c, col in enumerate([(240, 240, 240), (220, 90, 200), (220, 200, 60)]):
        pts = [(X(L), Y(v)) for L, v in zip(info["centres"], info["corr"][:, c])]
        pts = [(X(L0), pts[0][1])] + pts + [(X(L1), pts[-1][1])]         # flat beyond the measured tones
        dr.line(pts, fill=col, width=2)
    dr.text((50, 6), "grade by edit lightness: L white, a magenta, b yellow", fill=(230, 230, 230), font=fs)
    dr.text((X(lo_r) + 4, Y(C0) - 18), "tones in the band", fill=(140, 190, 140), font=fs)
    for v in np.arange(np.ceil(C0 / 0.02) * 0.02, C1, 0.02):
        dr.text((8, Y(v) - 7), f"{v:+.2f}", fill=(150, 150, 150), font=fs)
    dr.text((X(0.0), row_h - 34), "0", fill=(150, 150, 150), font=fs)
    dr.text((X(0.45) - 30, row_h - 34), "edit lightness", fill=(150, 150, 150), font=fs)

    comp = band_completeness(rough, band_px)
    nb = int((band & (w >= 0.5)).sum())
    sm = cv2.GaussianBlur(after.astype(np.float32), (0, 0), 16.0)       # the plateau, not the texture
    spread = np.ptp(np.percentile(sm[band & (w >= 0.5)], [5, 95]))
    edge_rms = float(np.sqrt(np.mean(field[..., 0][rough > 0] ** 2)))
    conf = min(1.0, comp / 0.6) * min(1.0, nb / 20000) * (0.5 if out_change.sum() > 0.05 * change.sum() else 1.0)
    lines = [
        ("painted mask (white), open sides (cyan dashes): " + (", ".join(open_sides) or "none"), (230, 230, 230)),
        (f"band 0-{band_px} px: {nb} trusted px, {comp*100:.0f}% of a full ring", (60, 200, 90)),
        (f"changed: {change.sum()/H/W*100:.1f}% of the image; outside the painted mask: "
         f"{out_change.sum()/max(change.sum(),1)*100:.0f}% of it", (255, 150, 0) if out_change.any() else (230, 60, 50)),
        (f"grade at mid tones: L {np.interp(0.5, info['centres'], info['corr'][:, 0]):+.3f}; "
         f"plateau left in the band p5-p95: {spread:.3f}", (230, 230, 230)),
        (f"plateau: red = edit darker, blue = brighter (Oklab L, +-0.06); edge field rms {edge_rms:.3f}", (230, 230, 230)),
        ("texture gain per band: " + " ".join(f"{g:.2f}" for g in gains), (230, 230, 230)),
        (f"confidence {conf:.2f}" + ("  WARNING: change reaches past the mask" if out_change.sum() > 0.05 * change.sum() else ""),
         (255, 150, 0) if conf < 0.75 else (60, 200, 90)),
    ]
    text_w = 700
    total_w = max(pw * 3 + 16, curve_w + 8 + text_w)
    timg = Image.new("RGB", (text_w, row_h), (24, 24, 24))
    dt = ImageDraw.Draw(timg)
    for i, (t, col) in enumerate(lines):
        dt.text((12, 12 + i * 32), t, fill=col, font=f)

    canvas = Image.new("RGB", (total_w, 28 + height + 8 + row_h), (24, 24, 24))
    dc = ImageDraw.Draw(canvas)
    left = (total_w - (pw * 3 + 16)) // 2
    for i, (img, title) in enumerate([(p1, "measured"), (p2, "plateau before"), (p3, "after, pasted area")]):
        canvas.paste(Image.fromarray(img), (left + i * (pw + 8), 28))
        dc.text((left + i * (pw + 8) + 4, 6), title, fill=(230, 230, 230), font=f)
    canvas.paste(gimg, (0, 28 + height + 8))
    canvas.paste(timg, (curve_w + 8, 28 + height + 8))
    return canvas


if __name__ == "__main__":
    from PIL import Image as _I
    RES, OUT, prefix, spec = sys.argv[1:5]
    O = cv2.imread(f"{RES}/{prefix}_orig_editframe.png")[..., ::-1].astype(np.float32) / 255
    E = np.load(f"{RES}/{prefix}_aligned.npy").astype(np.float32)
    H, W = O.shape[:2]
    from tps_cut_paste.flow import Flow
    from tps_cut_paste import seam_align
    flows = seam_align.raft_flows(Flow())
    if spec.startswith("file:"):
        ell = np.asarray(_I.open(f"{RES}/../{spec[5:]}"))[..., 3] == 0
        rough = cv2.resize(ell.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        x0, y0, x1, y1 = map(int, spec.split(":")[1].split(","))
        ch = cv2.resize(cm.trust(O, E, flows)["change"].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        sub = np.zeros_like(ch); sub[y0:y1, x0:x1] = ch[y0:y1, x0:x1]
        ys, xs = np.nonzero(sub)
        rough = np.zeros((H, W), np.uint8)
        cv2.ellipse(rough, (int((xs.min() + xs.max()) // 2), int((ys.min() + ys.max()) // 2)),
                    (int(np.ptp(xs) * 0.68), int(np.ptp(ys) * 0.62)), 0, 0, 360, 1, -1)
    grown = cv2.dilate(rough, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(np.float32)
    alpha = np.clip(cv2.GaussianBlur(grown, (0, 0), 8.0), 0, 1)
    tr = cm.trust(O, E, flows, prior=rough)
    curve, info = cm.fit_grade(tr["o"], tr["p"], tr["w"], mode="tone")
    G = cm.apply_grade(E, curve)
    field, _ = cm.edge_field(O, G, alpha, tr, band=48, kind="smooth", width=192)
    Gf = cm.apply_field(G, field)
    w_tex = cv2.resize((cm._soft(tr["z_col"]) * cm._soft(tr["z_rt"]) * (1 - tr["change"])).astype(np.float32), (W, H))
    _, gains = cm.texture_gains(O, Gf, w_tex, local=False)
    render(O, E, rough, tr, info, Gf, field, gains, alpha=alpha).save(f"{OUT}/view_{prefix}.png")
    print("saved", f"{OUT}/view_{prefix}.png")
