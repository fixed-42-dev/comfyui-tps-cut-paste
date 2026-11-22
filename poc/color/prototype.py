"""Run the prototype on the aligned sparrow pair and compare correction variants.

usage: prototype.py <results dir> <out dir> [prefix, default fin_mona] [rough mask png next to results]
"""
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import srgb_to_oklab, oklab_to_srgb, load_pair
import color_match as cm

RES, OUT = sys.argv[1], sys.argv[2]
PREFIX = sys.argv[3] if len(sys.argv) > 3 else "fin_mona"
ROUGH = sys.argv[4] if len(sys.argv) > 4 else "demo_mona_lisa_shoulder.png"
O, E, change_ref, rough = load_pair(RES, PREFIX, ROUGH)
H, W = O.shape[:2]

# the blend TPS Paste would use with a rough mask: grow 8, feather 24
grown = cv2.dilate(rough, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(np.float32)
alpha = np.clip(cv2.GaussianBlur(grown, (0, 0), 8.0), 0, 1)

# ---- stage 1
sys.path.insert(0, __file__.rsplit("/poc/", 1)[0])
from tps_cut_paste.flow import Flow
from tps_cut_paste import seam_align
tr = cm.trust(O, E, seam_align.raft_flows(Flow()))
ww, wh = tr["size"]
ref_w = cv2.resize(change_ref, (ww, wh), interpolation=cv2.INTER_AREA) > 0.5
ch = tr["change"] > 0
iou = (ch & ref_w).sum() / max((ch | ref_w).sum(), 1)
print(f"stage 1 at {ww}x{wh}: change mask {ch.mean()*100:.2f}% of frame, IoU with the earlier mask {iou:.2f}; "
      f"trusted (w>=0.5) {(tr['w'] >= 0.5).mean()*100:.0f}%")

# evaluation regions at full size
changed_full = cv2.resize(tr["change"].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
changed_full |= change_ref > 0.01
changed_full = cv2.dilate(changed_full.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
seam = (alpha > 0.05) & (alpha < 0.95) & ~changed_full
inside = (alpha >= 0.95) & ~changed_full
bird = change_ref > 0.5
Lo = srgb_to_oklab(O)

def score(name, P):
    comp = O * (1 - alpha[..., None]) + P * alpha[..., None]
    dlab = srgb_to_oklab(comp) - Lo
    dp = srgb_to_oklab(P) - Lo
    de = lambda x, m: np.median(np.linalg.norm(x[m], axis=-1))
    bias = lambda x, m: np.median(x[m, 0])
    lb = srgb_to_oklab(P)[bird].mean(0) - srgb_to_oklab(E)[bird].mean(0)
    print(f"  {name:34s} seam dE {de(dlab, seam):.4f} (L bias {bias(dlab, seam):+.4f})  "
          f"inside unchanged dE {de(dp, inside):.4f} (L bias {bias(dp, inside):+.4f})  bird moved L {lb[0]:+.4f}")
    return comp

variants = {}
variants["none"] = E

# baseline: what a default color match does - Reinhard on the cut area (mask box x2)
ys, xs = np.nonzero(rough)
cy, cx, hy, hx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2, int(np.ptp(ys)), int(np.ptp(xs))
box = (slice(max(0, cy - hy), min(H, cy + hy)), slice(max(0, cx - hx), min(W, cx + hx)))
lo_b, le_b = Lo[box].reshape(-1, 3), srgb_to_oklab(E)[box].reshape(-1, 3)
le = srgb_to_oklab(E)
variants["cut-area match (Reinhard)"] = np.clip(oklab_to_srgb((le - le_b.mean(0)) / le_b.std(0) * lo_b.std(0) + lo_b.mean(0)), 0, 1)

# ---- stage 2
curve_off, info_off = cm.fit_grade(tr["o"], tr["p"], tr["w"], mode="offset")
curve_tone, info_tone = cm.fit_grade(tr["o"], tr["p"], tr["w"], mode="tone")
print(f"stage 2: {info_tone['n']} samples, lightness range {info_tone['L_range'][0]:.2f}..{info_tone['L_range'][1]:.2f}; "
      f"global offset Oklab {np.round(info_off['offset'], 4)}")
print("  tone curve (edit L -> correction L, a, b):")
for c, v in zip(info_tone["centres"], info_tone["corr"]):
    print(f"    L {c:.3f}: {v[0]:+.4f} {v[1]:+.4f} {v[2]:+.4f}")
variants["grade: offset"] = cm.apply_grade(E, curve_off)
variants["grade: tone curve"] = cm.apply_grade(E, curve_tone)
G = variants["grade: tone curve"]

# ---- stage 3
for kind, width in [("smooth", 64), ("smooth", 192), ("decay", 128), ("entire", 0)]:
    f, _ = cm.edge_field(O, G, alpha, tr, band=48, kind=kind, width=width)
    variants[f"tone + edge {kind} {width}"] = cm.apply_field(G, f)
f, _ = cm.edge_field(O, E, alpha, tr, band=48, kind="smooth", width=64)
variants["edge only, smooth 64 (no grade)"] = cm.apply_field(E, f)
ff, _ = cm.edge_field(O, G, alpha, tr, band=48, kind="entire", width=0)
print(f"stage 3 field after the tone grade (Oklab L): p5 {np.percentile(ff[..., 0][alpha > 0.5], 5):+.4f}, "
      f"p95 {np.percentile(ff[..., 0][alpha > 0.5], 95):+.4f}")

print("\nscores (Oklab; seam = blend transition on unchanged pixels; inside = unchanged under the mask)")
comps = {k: score(k, v) for k, v in variants.items()}

# ---- figures
yy, xx = np.nonzero(alpha > 0.01)
y0, y1, x0, x1 = max(0, yy.min() - 60), min(H, yy.max() + 60), max(0, xx.min() - 60), min(W, xx.max() + 60)
show = ["none", "cut-area match (Reinhard)", "grade: offset", "grade: tone curve", "tone + edge smooth 64", "edge only, smooth 64 (no grade)"]
fig, ax = plt.subplots(2, len(show), figsize=(4 * len(show), 8.5))
for i, k in enumerate(show):
    c = comps[k][y0:y1, x0:x1]
    ax[0, i].imshow(c); ax[0, i].set_title(k, fontsize=9); ax[0, i].axis("off")
    dl = (srgb_to_oklab(comps[k]) - Lo)[y0:y1, x0:x1, 0]
    ax[1, i].imshow(dl, cmap="coolwarm", vmin=-0.05, vmax=0.05); ax[1, i].axis("off")
    ax[1, i].contour(alpha[y0:y1, x0:x1], levels=[0.05, 0.95], colors="k", linewidths=0.5)
ax[1, 0].set_title("pasted - original, Oklab L (+-0.05)", fontsize=9)
plt.tight_layout(); plt.savefig(OUT + f"/proto_{PREFIX}_variants.png", dpi=70); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
for c, (nm, col) in enumerate(zip("Lab", ["k", "m", "y"])):
    ax.plot(info_tone["centres"], info_tone["corr"][:, c], col + "o-", label=f"Oklab {nm}")
    ax.axhline(info_off["offset"][c], color=col, ls=":", lw=1)
ax.axvspan(*info_tone["L_range"], color="grey", alpha=0.1)
ax.set_xlabel("edit lightness (Oklab L)"); ax.set_ylabel("correction"); ax.legend(); ax.set_title("fitted grade (dotted = single offset)")
plt.tight_layout(); plt.savefig(OUT + f"/proto_{PREFIX}_curve.png", dpi=80); plt.close()
print("figures in", OUT)
