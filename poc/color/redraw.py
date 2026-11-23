"""Does the plateau stay even when the edit redraws more? Residual after the global
tone-curve grade, mapped over the frame, plus the color scores with a rough mask.

usage: redraw.py <results dir> <out dir> <prefix>:<rough kind> ...
  rough kind: ellipse:x0,y0,x1,y1 (an ellipse around change inside that box, edit px)
              | hull:<grow px> (the change mask, grown and filled)
              | file:<png next to results>
"""
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import srgb_to_oklab, oklab_to_srgb, signed_distance
import color_match as cm

RES, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, __file__.rsplit("/poc/", 1)[0])
from tps_cut_paste.flow import Flow
from tps_cut_paste import seam_align
flows = seam_align.raft_flows(Flow())

blocks_all, rings_all, names = [], [], []
for spec in sys.argv[3:]:
    prefix, kind = spec.split(":", 1)
    O = cv2.imread(f"{RES}/{prefix}_orig_editframe.png")[..., ::-1].astype(np.float32) / 255
    E = np.load(f"{RES}/{prefix}_aligned.npy").astype(np.float32)
    H, W = O.shape[:2]
    tr0 = cm.trust(O, E, flows)                        # no prior: for building the stand-in painted mask
    change = cv2.resize(tr0["change"].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    # the rough mask a user would paint
    if kind.startswith("maskpng:"):
        _, fn, g = kind.split(":")
        mk = np.asarray(Image.open(f"{RES}/{fn}").convert("L")) > 128
        mk = cv2.resize(mk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mk, connectivity=8)
        mk = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 0.01 * H * W]).astype(np.uint8)
        g = int(g)
        rough = cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1,) * 2))
    elif kind.startswith("ellipse:"):
        x0, y0, x1, y1 = map(int, kind.split(":")[1].split(","))
        sub = np.zeros_like(change); sub[y0:y1, x0:x1] = change[y0:y1, x0:x1]
        ys, xs = np.nonzero(sub)
        rough = np.zeros((H, W), np.uint8)
        cv2.ellipse(rough, ((xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2),
                    (int(np.ptp(xs) * 0.68), int(np.ptp(ys) * 0.62)), 0, 0, 360, 1, -1)
    elif kind.startswith("hull:"):
        g = int(kind.split(":")[1])
        n, lab, st, _ = cv2.connectedComponentsWithStats(change, connectivity=8)
        big = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 0.01 * H * W]).astype(np.uint8)
        rough = cv2.dilate(big, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1,) * 2))
        rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1,) * 2))
    else:
        ell = np.asarray(Image.open(f"{RES}/../{kind.split(':')[1]}"))[..., 3] == 0
        rough = cv2.resize(ell.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    no_prior = change.mean()
    tr = cm.trust(O, E, flows, prior=rough)             # the painted mask as prior (hard, no feather)
    ww, wh = tr["size"]
    change = cv2.resize(tr["change"].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    print(f"\n[{prefix}] stage-1 change mask: {no_prior*100:.1f}% of frame without the prior, {change.mean()*100:.1f}% with it")
    grown = cv2.dilate(rough, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(np.float32)
    alpha = np.clip(cv2.GaussianBlur(grown, (0, 0), 8.0), 0, 1)
    obj_in_mask = (change & rough).sum() / max(rough.sum(), 1)
    change_outside = (change & (1 - rough)).sum() / max(change.sum(), 1)

    # residual plateau after the global tone curve, on trusted pixels
    curve, info = cm.fit_grade(tr["o"], tr["p"], tr["w"], mode="tone")
    G = cm.apply_grade(E, curve)
    r = (srgb_to_oklab(O) - srgb_to_oklab(G))[..., 0]
    w = cv2.resize(tr["w"], (W, H), interpolation=cv2.INTER_LINEAR)
    B = 64
    blk = np.full((H // B, W // B), np.nan)
    for i in range(H // B):
        for j in range(W // B):
            m = w[i*B:(i+1)*B, j*B:(j+1)*B] >= 0.5
            if m.mean() > 0.3:
                blk[i, j] = np.median(r[i*B:(i+1)*B, j*B:(j+1)*B][m])
    d = signed_distance(rough)
    edges = np.arange(-160, 481, 32)
    ring = [np.median(r[(d >= a) & (d < b) & (w >= 0.5)]) if ((d >= a) & (d < b) & (w >= 0.5)).sum() > 300 else np.nan
            for a, b in zip(edges[:-1], edges[1:])]
    blocks_all.append(blk); rings_all.append((edges[:-1] + 16, ring)); names.append(prefix.replace("fin_", ""))
    print(f"\n== {prefix}: change mask {change.mean()*100:.1f}% of frame (stage 1), rough mask {rough.mean()*100:.1f}%, "
          f"object fills {obj_in_mask*100:.0f}% of it, change outside the rough mask {change_outside*100:.0f}%")
    print(f"  residual plateau after the tone curve, Oklab L, 64 px blocks: p5 {np.nanpercentile(blk, 5):+.4f}  "
          f"median {np.nanmedian(blk):+.4f}  p95 {np.nanpercentile(blk, 95):+.4f}  (spread {np.nanpercentile(blk, 95) - np.nanpercentile(blk, 5):.4f})")
    print("  ring profile of that residual from the rough mask edge (px: value):",
          " ".join(f"{c:+d}:{v:+.3f}" for c, v in zip(edges[:-1] + 16, ring) if np.isfinite(v)))

    # color scores
    ch_full = cv2.dilate(change, np.ones((9, 9), np.uint8)) > 0
    seam = (alpha > 0.05) & (alpha < 0.95) & ~ch_full
    inside = (alpha >= 0.95) & ~ch_full
    Lo = srgb_to_oklab(O)
    ys, xs = np.nonzero(rough)
    cy, cx, hy, hx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2, int(np.ptp(ys)), int(np.ptp(xs))
    box = (slice(max(0, cy - hy), min(H, cy + hy)), slice(max(0, cx - hx), min(W, cx + hx)))
    le = srgb_to_oklab(E); lo_b, le_b = Lo[box].reshape(-1, 3), le[box].reshape(-1, 3)
    variants = {"none": E,
                "cut-area match (Reinhard)": np.clip(oklab_to_srgb((le - le_b.mean(0)) / le_b.std(0) * lo_b.std(0) + lo_b.mean(0)), 0, 1),
                "grade: tone curve": G}
    for k_, wd in [("smooth", 64), ("smooth", 192), ("entire", 0)]:
        f, _ = cm.edge_field(O, G, alpha, tr, band=48, kind=k_, width=wd)
        variants[f"tone + edge {k_} {wd}"] = cm.apply_field(G, f)
    # the grade fitted only in a band around the mask (local), not the whole frame
    dw = signed_distance(cv2.resize(rough, (ww, wh), interpolation=cv2.INTER_NEAREST)) * (W / ww)
    for bw in (80, 160):
        wb = tr["w"] * ((dw > 0) & (dw < bw))
        cb, _ = cm.fit_grade(tr["o"], tr["p"], wb, mode="tone", bins=8)
        Gb = cm.apply_grade(E, cb)
        variants[f"band {bw} grade"] = Gb
        f, _ = cm.edge_field(O, Gb, alpha, tr, band=48, kind="smooth", width=192)
        variants[f"band {bw} grade + edge smooth 192"] = cm.apply_field(Gb, f)
    obj = change > 0
    for name, P in variants.items():
        comp = O * (1 - alpha[..., None]) + P * alpha[..., None]
        dl = srgb_to_oklab(comp) - Lo
        dp = srgb_to_oklab(P) - Lo
        de = lambda x, m: np.median(np.linalg.norm(x[m], axis=-1)) if m.any() else np.nan
        bias = lambda x, m: np.median(x[m, 0]) if m.any() else np.nan
        moved = (srgb_to_oklab(P)[obj & (rough > 0)].mean(0) - le[obj & (rough > 0)].mean(0))[0]
        print(f"  {name:28s} seam dE {de(dl, seam):.4f} (L {bias(dl, seam):+.4f})  inside dE {de(dp, inside):.4f} "
              f"(L {bias(dp, inside):+.4f})  object moved L {moved:+.4f}")
    vis = O.copy()
    cnt, _ = cv2.findContours(rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    vis8 = (vis * 255).astype(np.uint8).copy()
    vis8[change > 0] = (vis8[change > 0] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
    cv2.drawContours(vis8, cnt, -1, (0, 255, 255), 6)
    Image.fromarray(vis8).resize((W // 4, H // 4)).save(f"{OUT}/redraw_{prefix}_masks.png")

fig, ax = plt.subplots(1, len(names) + 1, figsize=(4 * len(names) + 5, 6.5), gridspec_kw={"width_ratios": [1] * len(names) + [1.6]})
for a_, blk, nm in zip(ax, blocks_all, names):
    m = a_.imshow(blk, cmap="coolwarm", vmin=-0.06, vmax=0.06); a_.set_title(f"{nm}: residual after tone curve", fontsize=10); a_.axis("off")
plt.colorbar(m, ax=ax[:-1], fraction=0.02, label="original - graded edit, Oklab L")
for (c, v), nm in zip(rings_all, names):
    ax[-1].plot(c, v, "o-", label=nm)
ax[-1].axvline(0, color="grey", ls="--"); ax[-1].axhline(0, color="grey", lw=0.5)
ax[-1].set_xlabel("distance from rough mask edge, px (negative = inside)"); ax[-1].set_title("ring profile of the residual"); ax[-1].legend()
plt.savefig(f"{OUT}/redraw_plateaus.png", dpi=70, bbox_inches="tight"); plt.close()
print("\nfigures in", OUT)
