"""Measure before prototyping (doc checklist 1-5) on the aligned sparrow pair.

usage: measure.py <results dir with fin_mona_*> <out dir>
"""
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from common import (srgb_to_linear, linear_to_srgb, srgb_to_oklab, oklab_to_srgb, LUMA,
                    nconv, signed_distance, flatness, load_pair)

RES, OUT = sys.argv[1], sys.argv[2]
O, E, change, rough = load_pair(RES)
h, w = O.shape[:2]
d = E - O
dist = signed_distance(rough)                       # + outside the rough mask
unchanged = change < 0.01
flat = flatness(O) < np.percentile(flatness(O), 60)
trusted = unchanged & flat
Lo, Le = srgb_to_oklab(O), srgb_to_oklab(E)
dl = Le - Lo
rng = np.random.default_rng(0)
print(f"frame {w}x{h}; unchanged {unchanged.mean()*100:.1f}%, flat {flat.mean()*100:.0f}%, trusted {trusted.mean()*100:.1f}%")

# ================================================= 1. ring profile
print("\n== 1. ring profile: median difference by distance from the rough mask edge (16 px rings)")
edges = np.arange(-320, 496, 16)
rows = []
for a, b in zip(edges[:-1], edges[1:]):
    ring = (dist >= a) & (dist < b)
    t = ring & trusted
    rows.append((a + 8, ring.sum(), t.sum(),
                 np.median(dl[ring, 0]) if ring.any() else np.nan,
                 np.median(dl[t, 0]) if t.sum() > 200 else np.nan,
                 *(np.median(d[t], 0) if t.sum() > 200 else [np.nan] * 3),
                 np.median(dl[t, 1]) if t.sum() > 200 else np.nan,
                 np.median(dl[t, 2]) if t.sum() > 200 else np.nan))
rows = np.array(rows, float)
print("  dist   all-px dL   trusted dL    R       G       B     Oklab a  Oklab b  (n trusted)")
for r in rows[::3]:
    print(f"  {r[0]:+5.0f}   {r[3]:+.4f}     {r[4]:+.4f}   {r[5]:+.4f} {r[6]:+.4f} {r[7]:+.4f}  {r[8]:+.4f}  {r[9]:+.4f}  ({int(r[2])})")
out_rows = rows[rows[:, 0] > 0]
for width in (40, 80, 160, 320, 480):
    s = trusted & (dist > 0) & (dist < width)
    print(f"  band 0..{width:3d} px outside: trusted dL median {np.median(dl[s, 0]):+.4f}  RGB {np.median(d[s], 0).round(4)}  n={s.sum()}")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
ax[0].plot(rows[:, 0], rows[:, 3], "k-", lw=1, label="all pixels, Oklab dL")
ax[0].plot(rows[:, 0], rows[:, 4], "C0-", lw=2, label="trusted (unchanged & flat), dL")
ax[0].axvline(0, color="grey", ls="--", lw=0.8); ax[0].axhline(0, color="grey", lw=0.5)
ax[0].set_xlabel("distance from rough mask edge, px (negative = inside)"); ax[0].set_ylabel("median after - before")
ax[0].legend(); ax[0].set_title("Ring profile, lightness")
for c, col in enumerate("rgb"):
    ax[1].plot(rows[:, 0], rows[:, 5 + c], col, lw=1.5, label=f"{col.upper()} trusted")
ax[1].plot(rows[:, 0], rows[:, 8], "m--", lw=1, label="Oklab a"); ax[1].plot(rows[:, 0], rows[:, 9], "y--", lw=1, label="Oklab b")
ax[1].axvline(0, color="grey", ls="--", lw=0.8); ax[1].axhline(0, color="grey", lw=0.5)
ax[1].set_xlabel("distance from rough mask edge, px"); ax[1].legend(fontsize=8); ax[1].set_title("Ring profile, per channel")
plt.tight_layout(); plt.savefig(OUT + "/1_ring_profile.png", dpi=80); plt.close()

# ================================================= 2. representations
print("\n== 2. representations: fit on half the 0..80 px band, test on the other half and on everything farther out")
cy, cx = np.argwhere(rough > 0).mean(0)
ang = np.arctan2(*np.mgrid[0:h, 0:w] - np.array([cy, cx])[:, None, None])
sector = (np.floor((ang + np.pi) / (np.pi / 8)).astype(int) % 2) == 0
band = trusted & (dist > 0) & (dist < 80)
sets = {"train": band & sector, "test band": band & ~sector, "test far": trusted & (dist >= 80)}
pick = {k: np.argwhere(v) for k, v in sets.items()}
pick = {k: v[rng.choice(len(v), min(len(v), 60000), replace=False)] for k, v in pick.items()}
X = {k: O[v[:, 0], v[:, 1]].astype(np.float64) for k, v in pick.items()}
Y = {k: E[v[:, 0], v[:, 1]].astype(np.float64) for k, v in pick.items()}
print(f"  train {len(X['train'])}  test band {len(X['test band'])}  test far {len(X['test far'])}  (subsampled)")
print(f"  train tones: luma p1..p99 {np.percentile(X['train'] @ LUMA, 1):.2f}..{np.percentile(X['train'] @ LUMA, 99):.2f}; "
      f"far {np.percentile(X['test far'] @ LUMA, 1):.2f}..{np.percentile(X['test far'] @ LUMA, 99):.2f}")

def sat(x, s):
    y = (x @ LUMA)[:, None]
    return y + s * (x - y)

def m_offset(p, x): return x + p[:3]
def m_affine(p, x): return x * p[:3] + p[3:6]
def m_matrix(p, x): return x @ p[:9].reshape(3, 3).T + p[9:12]
def m_lin_gain(p, x): return linear_to_srgb(srgb_to_linear(x) * p[:3])
def m_lin_affine(p, x): return linear_to_srgb(srgb_to_linear(x) * p[:3] + p[3:6])
def m_oklab_affine(p, x):
    lab = srgb_to_oklab(x)
    return oklab_to_srgb(lab * p[:3] + p[3:6])
def m_oklab_full(p, x):
    lab = srgb_to_oklab(x)
    return oklab_to_srgb(lab @ p[:9].reshape(3, 3).T + p[9:12])
def m_cdl(p, x):
    y = np.clip(x * p[:3] + p[3:6], 1e-6, None) ** p[6:9]
    return sat(y, p[9])

models = {
    "none": (None, []),
    "sRGB offset": (m_offset, [0, 0, 0]),
    "sRGB gain+offset": (m_affine, [1, 1, 1, 0, 0, 0]),
    "sRGB 3x3+offset": (m_matrix, list(np.eye(3).ravel()) + [0, 0, 0]),
    "linear gain": (m_lin_gain, [1, 1, 1]),
    "linear gain+offset": (m_lin_affine, [1, 1, 1, 0, 0, 0]),
    "Oklab gain+offset": (m_oklab_affine, [1, 1, 1, 0, 0, 0]),
    "Oklab 3x3+offset": (m_oklab_full, list(np.eye(3).ravel()) + [0, 0, 0]),
    "CDL (slope,offset,power,sat)": (m_cdl, [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]),
}
fits = {}
print(f"  {'model':30s} params  train   test band  test far   far bias (R,G,B)")
for name, (f, p0) in models.items():
    if f is None:
        pred = lambda x: x; p = []
    else:
        r = least_squares(lambda p: (f(p, X["train"]) - Y["train"]).ravel(), p0, loss="huber", f_scale=0.02)
        p = r.x; pred = lambda x, f=f, p=p: f(p, x)
    fits[name] = (pred, p)
    err = {k: np.abs(pred(X[k]) - Y[k]) for k in X}
    bias = np.median(Y["test far"] - pred(X["test far"]), 0)
    print(f"  {name:30s} {len(p):3d}   {np.median(err['train']):.4f}  {np.median(err['test band']):.4f}     "
          f"{np.median(err['test far']):.4f}    {bias.round(4)}")
print("  (median |after - predicted| per channel value; bias = median signed residual far out)")

# tone dependence of the far residual for three candidates
fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
xs = X["test far"]; yl = xs @ LUMA; bins = np.linspace(0, 0.9, 19)
for i, name in enumerate(["none", "sRGB gain+offset", "Oklab gain+offset"]):
    res = Y["test far"] - fits[name][0](xs)
    for c, col in enumerate("rgb"):
        med = [np.median(res[(yl >= a) & (yl < b), c]) if ((yl >= a) & (yl < b)).sum() > 100 else np.nan
               for a, b in zip(bins[:-1], bins[1:])]
        ax[i].plot((bins[:-1] + bins[1:]) / 2, med, col, lw=2)
    ax[i].axhline(0, color="grey", lw=0.5); ax[i].set_title(f"far residual by tone: {name}")
    ax[i].axvspan(np.percentile(X["train"] @ LUMA, 1), np.percentile(X["train"] @ LUMA, 99), color="grey", alpha=0.1)
    ax[i].set_xlabel("before luma (grey = tones in the training band)")
ax[0].set_ylabel("median after - predicted")
plt.tight_layout(); plt.savefig(OUT + "/2_representations.png", dpi=80); plt.close()

# ================================================= 3. 2D plateau and floor maps
print("\n== 3. plateau and floor maps (quarter size, sigma 8 = ~32 px at edit size), self-weighting from all pixels")
q = (w // 4, h // 4)
dq = cv2.resize(dl, q, interpolation=cv2.INTER_AREA)            # Oklab difference
chg_q = cv2.resize(change, q, interpolation=cv2.INTER_AREA) > 0.5
unch_q = cv2.resize(change, q, interpolation=cv2.INTER_AREA) < 0.01
def global_weights(x):
    """Seed: distance from the global median in global MADs. A local mean seeded
    with all pixels is dragged by an object as big as its window and never recovers."""
    med = np.median(x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x, 0)
    mad = 1.4826 * np.median(np.abs(x - med).reshape(-1, x.shape[-1]) if x.ndim == 3 else np.abs(x - med), 0) + 1e-4
    zz = (np.abs(x - med) / mad)
    zz = zz.max(-1) if x.ndim == 3 else zz
    return np.clip(1.0 - (zz - 2.0) / 2.0, 0.0, 1.0).astype(np.float32)
wgt = global_weights(dq)
print(f"  seed (global median/MAD): weight<0.5 on {(wgt < 0.5)[chg_q].mean()*100:.0f}% of the bird, {(wgt < 0.5)[unch_q].mean()*100:.2f}% outside")
for it in range(4):
    plateau = np.stack([nconv(dq[..., c], wgt, 8)[0] for c in range(3)], -1)
    dev = np.abs(dq - plateau)
    floor = np.stack([1.4826 * nconv(dev[..., c], wgt, 8)[0] for c in range(3)], -1) + 1e-4
    z = (dev / floor).max(-1)
    wgt = np.minimum(wgt, np.clip(1.0 - (z - 2.0) / 2.0, 0.0, 1.0)).astype(np.float32)   # full below 2 floors, none above 4; once out, stays out
    print(f"  iteration {it}: z>4 covers {(z > 4).mean()*100:.2f}% of frame; "
          f"of the bird {(z > 4)[chg_q].mean()*100:.0f}%, outside it {(z > 4)[unch_q].mean()*100:.2f}%")
fixed = np.abs(cv2.resize(d, q, interpolation=cv2.INTER_AREA)).mean(-1) > 0.08
print(f"  fixed color threshold 0.08: of the bird {fixed[chg_q].mean()*100:.0f}%, outside it {fixed[unch_q].mean()*100:.2f}%")
print(f"  floor (Oklab L) over the frame: p5 {np.percentile(floor[..., 0], 5):.4f}, median {np.median(floor[..., 0]):.4f}, "
      f"p95 {np.percentile(floor[..., 0], 95):.4f}  (ratio p95/p5 {np.percentile(floor[..., 0], 95) / np.percentile(floor[..., 0], 5):.1f})")
print(f"  plateau (Oklab L): p5 {np.percentile(plateau[..., 0], 5):+.4f}, median {np.median(plateau[..., 0]):+.4f}, p95 {np.percentile(plateau[..., 0], 95):+.4f}")
# the round-trip error floor
sys.path.insert(0, __file__.rsplit("/poc/", 1)[0])
from tps_cut_paste.flow import Flow
from tps_cut_paste import seam_align
fw, fh = 672, 1024
fwd, bwd = seam_align.raft_flows(Flow())(cv2.resize(O, (fw, fh), interpolation=cv2.INTER_AREA),
                                         cv2.resize(E, (fw, fh), interpolation=cv2.INTER_AREA))
yy, xx = np.mgrid[0:fh, 0:fw].astype(np.float32)
back = np.stack([cv2.remap(bwd[i], xx + fwd[0], yy + fwd[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for i in range(2)])
err = cv2.resize(np.hypot(fwd[0] + back[0], fwd[1] + back[1]), q, interpolation=cv2.INTER_AREA)
ew = global_weights(err)
for it in range(3):
    ep, _ = nconv(err, ew, 8); ef = 1.4826 * nconv(np.abs(err - ep), ew, 8)[0] + 0.02
    ez = (err - ep) / ef
    ew = np.clip(1.0 - (ez - 2.0) / 2.0, 0, 1).astype(np.float32)
print(f"  round-trip error floor: p5 {np.percentile(ef, 5):.3f}, median {np.median(ef):.3f}, p95 {np.percentile(ef, 95):.3f} px; "
      f"z>4 of the bird {(ez > 4)[chg_q].mean()*100:.0f}%, outside {(ez > 4)[unch_q].mean()*100:.2f}%; "
      f"fixed 1.5 px: bird {(err > 1.5)[chg_q].mean()*100:.0f}%, outside {(err > 1.5)[unch_q].mean()*100:.2f}%")
fig, ax = plt.subplots(1, 5, figsize=(20, 6))
for a_, im, t, kw in [(ax[0], plateau[..., 0], "plateau map, Oklab L", dict(cmap="coolwarm", vmin=-0.04, vmax=0.04)),
                      (ax[1], floor[..., 0], "floor map, Oklab L", dict(cmap="magma", vmin=0, vmax=0.05)),
                      (ax[2], np.clip(z, 0, 8), "color z (floors above plateau)", dict(cmap="inferno", vmin=0, vmax=8)),
                      (ax[3], np.clip(ez, 0, 8), "round-trip z", dict(cmap="inferno", vmin=0, vmax=8)),
                      (ax[4], fixed.astype(float), "fixed threshold 0.08", dict(cmap="gray"))]:
    m = a_.imshow(im, **kw); a_.set_title(t); a_.axis("off"); plt.colorbar(m, ax=a_, fraction=0.04)
plt.tight_layout(); plt.savefig(OUT + "/3_floor_maps.png", dpi=70); plt.close()

# ================================================= 4. texture
print("\n== 4. texture: detail energy per Laplacian level (Oklab L), unchanged pixels")
def lap_pyr(x, n=6):
    out, g = [], x.astype(np.float32)
    for _ in range(n):
        down = cv2.pyrDown(g); up = cv2.pyrUp(down, dstsize=(g.shape[1], g.shape[0]))
        out.append(g - up); g = down
    return out
po, pe = lap_pyr(Lo[..., 0]), lap_pyr(Le[..., 0])
m0 = unchanged.astype(np.uint8)
print("  level  px-scale  energy before   after    after/before")
for i, (a_, b_) in enumerate(zip(po, pe)):
    mi = cv2.resize(m0, (a_.shape[1], a_.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    eo, ee = np.median(np.abs(a_[mi])), np.median(np.abs(b_[mi]))
    print(f"  {i}      {2**i:3d}       {eo:.5f}     {ee:.5f}    {ee / eo:.2f}")
fl0 = cv2.resize((flat & unchanged).astype(np.uint8), (po[0].shape[1], po[0].shape[0])) > 0
print(f"  noise (MAD of level 0 on flat unchanged): before {1.4826*np.median(np.abs(po[0][fl0])):.5f}, "
      f"after {1.4826*np.median(np.abs(pe[0][fl0])):.5f}")

# ================================================= 5. SSIM split
print("\n== 5. SSIM split (Oklab L, Gaussian window sigma 1.5)")
x, y = Lo[..., 0].astype(np.float32), Le[..., 0].astype(np.float32)
g = lambda a: cv2.GaussianBlur(a, (0, 0), 1.5)
mx, my = g(x), g(y)
vx, vy, cxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
sx, sy = np.sqrt(np.maximum(vx, 0)), np.sqrt(np.maximum(vy, 0))
C1, C2 = (0.01) ** 2, (0.03) ** 2
lum = (2 * mx * my + C1) / (mx * mx + my * my + C1)
con = (2 * sx * sy + C2) / (vx + vy + C2)
stru = (cxy + C2 / 2) / (sx * sy + C2 / 2)
bird = change > 0.5
for nm, a_ in [("luminance", lum), ("contrast", con), ("structure", stru)]:
    print(f"  {nm:10s} median unchanged {np.median(a_[unchanged]):.3f}  bird {np.median(a_[bird]):.3f}  "
          f"share < 0.5: unchanged {(a_[unchanged] < 0.5).mean()*100:.1f}%  bird {(a_[bird] < 0.5).mean()*100:.0f}%")
fig, ax = plt.subplots(1, 3, figsize=(15, 7))
for a_, im, t in zip(ax, [lum, con, stru], ["luminance term", "contrast term", "structure term"]):
    m = a_.imshow(cv2.resize(im, q, interpolation=cv2.INTER_AREA), cmap="viridis", vmin=0, vmax=1); a_.set_title(t); a_.axis("off")
plt.colorbar(m, ax=ax, fraction=0.02); plt.savefig(OUT + "/5_ssim_split.png", dpi=70); plt.close()
print("\nfigures in", OUT)
