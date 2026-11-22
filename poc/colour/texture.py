"""Stage 4 on an aligned pair: spectrum match and seam-aware blending.

usage: texture.py <results dir> <out dir> [prefix] [rough mask png next to results]
"""
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import srgb_to_oklab, load_pair
import colour_match as cm

RES, OUT = sys.argv[1], sys.argv[2]
PREFIX = sys.argv[3] if len(sys.argv) > 3 else "fin_mona"
ROUGH = sys.argv[4] if len(sys.argv) > 4 else "demo_mona_lisa_shoulder.png"
O, E, change_ref, rough = load_pair(RES, PREFIX, ROUGH)
H, W = O.shape[:2]
grown = cv2.dilate(rough, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(np.float32)
alpha = np.clip(cv2.GaussianBlur(grown, (0, 0), 8.0), 0, 1)

sys.path.insert(0, __file__.rsplit("/poc/", 1)[0])
from tps_cut_paste.flow import Flow
from tps_cut_paste import seam_align
tr = cm.trust(O, E, seam_align.raft_flows(Flow()))

# colour first (stages 2-3), as decided
curve, _ = cm.fit_grade(tr["o"], tr["p"], tr["w"], mode="tone")
G = cm.apply_grade(E, curve)
f, _ = cm.edge_field(O, G, alpha, tr, band=48, kind="smooth", width=64)
G = cm.apply_field(G, f)

# texture trust: like colour trust but without the flatness term (texture lives on edges)
w_tex = cm._soft(tr["z_col"]) * cm._soft(tr["z_rt"]) * (1 - tr["change"])
w_tex = cv2.resize(w_tex.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
maps_g, glob = cm.texture_gains(O, G, w_tex, local=False)
maps_l, _ = cm.texture_gains(O, G, w_tex, local=True)
print("global gain per band (finest first):", " ".join(f"{g:.2f}" for g in glob))
print("local gain spread per band (p5..p95):", "  ".join(f"{np.percentile(m, 5):.2f}..{np.percentile(m, 95):.2f}" for m in maps_l))
Tg = cm.apply_texture(G, maps_g)
Tl = cm.apply_texture(G, maps_l)

variants = {
    "raw edit, plain alpha": (E, "alpha"),
    "colour, plain alpha": (G, "alpha"),
    "colour + global gain, alpha": (Tg, "alpha"),
    "colour + local gain, alpha": (Tl, "alpha"),
    "colour + local gain, multiband": (Tl, "multiband"),
    "colour + local gain, energy": (Tl, "energy"),
}

# regions
changed = cv2.resize(tr["change"].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
changed |= change_ref > 0.01
changed = cv2.dilate(changed.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
inside = (alpha >= 0.95) & ~changed
Lo = srgb_to_oklab(O)[..., 0]
po, _ = cm.lap_pyr(Lo, 5)
blur = lambda v, s: cv2.GaussianBlur(v, (0, 0), s)
def bands_hp(L):   # fine (1-2 px) and mid (4-12 px) detail
    return L - blur(L, 1.0), blur(L, 2.0) - blur(L, 6.0)
hp_o = bands_hp(Lo)
rms_o = [np.sqrt(blur(b * b, 6.0)) for b in hp_o]
abins = [0.0, 0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.01]

print("\nband energy inside (unchanged under the mask), composite / original, finest band first; target 1.00")
print("then the seam: fine and mid detail energy, composite / original, by blend weight (unchanged pixels); dip = lowest")
profiles = {}
for name, (P, mode) in variants.items():
    comp = cm.blend(O, P, alpha, mode)
    Lc = srgb_to_oklab(comp)[..., 0]
    pc, _ = cm.lap_pyr(Lc, 5)
    ratios = []
    for a_, c_ in zip(po, pc):
        m = cv2.resize(inside.astype(np.uint8), (a_.shape[1], a_.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        ratios.append(np.sqrt(np.mean(c_[m] ** 2) / np.mean(a_[m] ** 2)))
    hp_c = bands_hp(Lc)
    prof = []
    for bo, bc in zip(rms_o, hp_c):
        rc = np.sqrt(blur(bc * bc, 6.0))
        prof.append([np.median((rc / np.maximum(bo, 1e-6))[(alpha >= a) & (alpha < b) & ~changed]) for a, b in zip(abins[:-1], abins[1:])])
    profiles[name] = np.array(prof)
    dE = np.median(np.linalg.norm((srgb_to_oklab(comp) - srgb_to_oklab(O))[(alpha > 0.05) & (alpha < 0.95) & ~changed], axis=-1))
    print(f"  {name:32s} inside {' '.join(f'{r:.2f}' for r in ratios)} | seam dip fine {min(prof[0][1:-1]):.2f} "
          f"mid {min(prof[1][1:-1]):.2f} | seam dE {dE:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
centres = [(a + b) / 2 for a, b in zip(abins[:-1], abins[1:])]
centres[-1] = 1.0
for name, p in profiles.items():
    for i in range(2):
        ax[i].plot(centres, p[i], "o-", label=name, lw=1.5)
for i, t in enumerate(["fine detail (1-2 px)", "mid detail (4-12 px)"]):
    ax[i].axhline(1, color="grey", lw=0.5); ax[i].set_xlabel("blend weight (0 = original, 1 = edit)")
    ax[i].set_ylabel("local detail energy, composite / original"); ax[i].set_title(f"Across the seam: {t}")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + f"/tex_{PREFIX}_seam.png", dpi=80); plt.close()

# zoomed crop across the seam: original | colour, alpha | local gain, alpha | local gain, energy
ring = (alpha > 0.3) & (alpha < 0.7) & ~changed & (blur(Lo, 8.0) > 0.45)   # a bright, textured stretch of seam
score_map = np.where(ring, blur(rms_o[0], 16.0), 0)
cy, cx = np.unravel_index(np.argmax(score_map), score_map.shape)
s = 150
y0, x0 = max(0, cy - s), max(0, cx - s)
box = (slice(y0, min(H, cy + s)), slice(x0, min(W, cx + s)))
names = ["colour, plain alpha", "colour + local gain, alpha", "colour + local gain, energy"]
comps = [cm.blend(O, variants[k][0], alpha, variants[k][1]) for k in names]
fig, ax = plt.subplots(2, 4, figsize=(16, 8.4))
ax[0, 0].imshow(O[box]); ax[0, 0].set_title("original", fontsize=10)
ax[1, 0].imshow(alpha[box], cmap="gray", vmin=0, vmax=1); ax[1, 0].set_title("blend weight", fontsize=10)
for j, (k, c) in enumerate(zip(names, comps), 1):
    ax[0, j].imshow(np.clip(c[box], 0, 1)); ax[0, j].set_title(k, fontsize=10)
    hp = bands_hp(srgb_to_oklab(c)[..., 0])[0]
    ratio = np.sqrt(blur(hp * hp, 6.0)) / np.maximum(rms_o[0], 1e-6)
    m = ax[1, j].imshow(ratio[box], cmap="RdBu", vmin=0.5, vmax=1.5)
    ax[1, j].contour(alpha[box], levels=[0.05, 0.5, 0.95], colors="k", linewidths=0.5)
    ax[1, j].set_title("fine detail energy / original", fontsize=10)
for a_ in ax.ravel():
    a_.axis("off")
plt.colorbar(m, ax=ax[1, :], fraction=0.015)
plt.savefig(OUT + f"/tex_{PREFIX}_crop.png", dpi=75); plt.close()
print("figures in", OUT, f"(crop centre x={cx}, y={cy})")
