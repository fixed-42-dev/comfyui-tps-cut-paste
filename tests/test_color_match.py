"""Color and texture match: the grade is measured where the edit left the picture
alone, a large object does not pull it, the painted mask finds a large recolor,
the feather keeps detail, and nothing outside the paste changes.

No optical flow here (changes are found by color only), so no model is loaded.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from tps_cut_paste import color_match as cm  # noqa: E402
from tps_cut_paste.cut_paste import cut, paste  # noqa: E402

RNG = np.random.default_rng(7)


def scene(h=320, w=320, seed=0):
    """Smooth color fields plus a fine texture, like a painting's surface."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w] / max(h, w)
    base = np.stack([0.45 + 0.3 * np.sin(5 * x + 2 * y), 0.4 + 0.25 * np.cos(4 * y),
                     0.3 + 0.2 * np.sin(3 * (x + y))], -1)
    tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.8)
    return np.clip(base + 0.04 * tex[..., None], 0, 1).astype(np.float32)


def disk(h, w, cx, cy, r):
    y, x = np.mgrid[0:h, 0:w]
    return ((x - cx) ** 2 + (y - cy) ** 2) < r * r


def graded(img, dL=-0.03):
    """What an edit model does everywhere: darker, more so in the shadows."""
    lab = cm.to_oklab(img)
    lab[..., 0] += dL * (1.2 - lab[..., 0])
    return cm.from_oklab(lab)


def test_grade_is_measured_off_the_object():
    o = scene()
    mask = disk(320, 320, 160, 160, 90)
    e = graded(o)
    obj = disk(320, 320, 160, 160, 64)                 # a bright red object, half the mask
    e[obj] = [0.85, 0.12, 0.1]
    alpha = mask.astype(np.float32)
    fixed, change, _ = cm.match(o, e, mask, alpha, color="grade", texture="off")
    keep = mask & ~cv2.dilate(obj.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    before = np.median((cm.to_oklab(e) - cm.to_oklab(o))[keep, 0])
    after = np.median((cm.to_oklab(fixed) - cm.to_oklab(o))[keep, 0])
    assert before < -0.015
    assert abs(after) < 0.004
    # a plain mean-and-spread match over the cut is pulled off by the object
    lo, le = cm.to_oklab(o)[mask], cm.to_oklab(e)[mask]
    naive = (cm.to_oklab(e)[keep] - le.mean(0)) / le.std(0) * lo.std(0) + lo.mean(0)
    assert abs(np.median(naive[:, 0] - cm.to_oklab(o)[keep, 0])) > 0.02
    # and the object is found
    assert change[obj].mean() > 0.9
    assert change[~mask].mean() < 0.01


def test_painted_mask_finds_a_large_recolor():
    o = scene(seed=1)
    mask = np.zeros((320, 320), bool)
    mask[40:300, 40:280] = True                        # 60 % of the frame
    lab = cm.to_oklab(o)
    lab[mask, 1] += 0.08                               # recolored, same folds and texture
    e = cm.from_oklab(lab)
    with_prior = cm.trust(o, e, prior=mask)["change"]
    assert with_prior[mask].mean() > 0.8
    far = ~cv2.dilate(mask.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    assert with_prior[far].mean() < 0.01            # beyond the change mask's small safety grow


def test_feather_keeps_detail():
    h, w = 256, 256
    rng = np.random.default_rng(3)
    a_tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.7)
    b_tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.7)
    o = np.clip(0.5 + 0.05 * a_tex[..., None], 0, 1).repeat(3, -1).astype(np.float32)
    p = np.clip(0.5 + 0.05 * b_tex[..., None], 0, 1).repeat(3, -1).astype(np.float32)
    alpha = np.full((h, w), 0.5, np.float32)
    alpha[:, :32] = 0.0
    hp = lambda img: (lambda L: L - cv2.GaussianBlur(L, (0, 0), 1.0))(cm.to_oklab(img)[..., 0])
    ref = hp(o)[:, 64:-16].std()
    plain = hp(o * 0.5 + p * 0.5)[:, 64:-16].std() / ref
    kept_img = cm.energy_blend(o, p, alpha)
    kept = hp(kept_img)[:, 64:-16].std() / ref
    assert plain < 0.8
    assert kept > 0.9
    assert np.array_equal(kept_img[:, :32], o[:, :32])  # where alpha is 0, the original


def test_the_frame_border_is_not_an_edge():
    m = np.zeros((100, 100), np.uint8)
    m[20:80, 0:40] = 1                                 # touches the left border
    d = cm.signed_distance(m)
    assert d[50, 0] < -10                              # a border pixel is deep inside, not on an edge
    assert cm.profile(np.array([0.0, 64.0, 128.0]), "smooth", 128).tolist() == [1.0, 0.5, 0.0]


def test_paste_changes_nothing_outside_with_everything_on():
    H, W = 384, 384
    src = torch.from_numpy(scene(H, W, seed=2))[None]
    mask = torch.from_numpy(disk(H, W, 190, 200, 60).astype(np.float32))[None]
    c, _, info, _ = cut(src, mask, 512, 1.8, 6.0, 0, 16, True, 0.1)
    edited = torch.from_numpy(graded(c[0].numpy()))[None]
    out, _, pasted_cut, _, pm, change, view = paste(
        src, edited, info, "mask", 8, 24, 32, "none", color="grade + edge", texture="match",
        feather_detail="keep")
    assert torch.equal(out[0][pm[0] == 0], src[0][pm[0] == 0])
    assert change.shape == (1, H, W) and view.dim() == 4 and view.shape[-1] == 3
    assert pasted_cut.shape == c.shape
    # inside, the darkening is undone
    inner = pm[0] > 0.99
    d_before = np.median(cm.to_oklab(edited[0].numpy())[..., 0] - cm.to_oklab(c[0].numpy())[..., 0])
    d_after = np.median((cm.to_oklab(out[0].numpy()) - cm.to_oklab(src[0].numpy()))[inner.numpy(), 0])
    assert d_before < -0.01 and abs(d_after) < 0.006


def test_off_measures_nothing():
    H, W = 256, 256
    src = torch.from_numpy(scene(H, W))[None]
    mask = torch.from_numpy(disk(H, W, 128, 128, 40).astype(np.float32))[None]
    c, _, info, _ = cut(src, mask, 256, 1.8, 6.0, 0, 16, True, 0.1)
    *_, change, view = paste(src, c, info, "mask", 8, 24, 32, "none")
    assert change.abs().sum() == 0 and view.shape[0] == 1


@pytest.mark.parametrize("sigma", [3.0, 12.8, 51.2, 204.8])
def test_wide_blur_matches_a_full_gaussian(sigma):
    """The shrunk blur behind the fills stays within a few thousandths of a
    full-size Gaussian, on a map with hard edges and gaps like a trust weight."""
    h, w = 600, 451
    rng = np.random.default_rng(3)
    m = (disk(h, w, 200, 300, 120) | (rng.random((h, w)) > 0.97)).astype(np.float32)
    full = cv2.GaussianBlur(m, (0, 0), sigma)
    assert np.abs(cm.blur(m, sigma) - full).max() < 0.005
