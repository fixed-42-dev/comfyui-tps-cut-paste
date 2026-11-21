"""TPS Cut and TPS Paste geometry: a cut pasted back unchanged returns the
source, nothing outside the cut changes, and every cut-size rule holds.

All with align "none", so no optical flow is needed; `test_seam_align.py`
covers alignment.
"""
import logging
import math

import pytest

torch = pytest.importorskip("torch")

from tps_cut_paste.cut_paste import cut as _cut, paste  # noqa: E402

ARGS = (1024, 1.8, 6.0, 0, 16, True, 0.1)   # target_size .. mask_threshold


def cut(*args):
    """The geometry: cut, cut mask and warp_info, without the preview."""
    return _cut(*args)[:3]


def scene(H, W):
    y, x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    img = torch.stack([0.5 + 0.4 * torch.sin(9 * x + 3 * y), 0.5 + 0.4 * torch.cos(7 * y),
                       0.5 + 0.3 * torch.sin(5 * (x + y))], -1)
    return img[None]


def disk(H, W, cx, cy, r):
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    return (((x - cx) ** 2 + (y - cy) ** 2) < r * r).float()[None]


def rect(H, W, x0, y0, x1, y1):
    m = torch.zeros(1, H, W)
    m[0, y0:y1, x0:x1] = 1
    return m


def source_box(info):
    """x0, y0, x1, y1 in source px of the first image's cut."""
    w, h, s = info["cut_w"], info["cut_h"], info["scale"]
    x0, y0 = info["t"][0].tolist()
    return x0, y0, x0 + w * s, y0 + h * s


@pytest.mark.parametrize("H,W,cx,cy,r,rot", [
    (768, 1024, 400, 300, 60, 0),          # small face, upscaled
    (768, 1024, 500, 380, 90, 25),         # rotated 25 degrees
    (4320, 7680, 3000, 2000, 700, 0),      # 8K, downscaled cut
], ids=["upscaled", "rotated", "8k"])
def test_round_trip(H, W, cx, cy, r, rot):
    img, m = scene(H, W), disk(H, W, cx, cy, r)
    c, cm, info = cut(img, m, 1024, 1.8, 6.0, rot, 16, True, 0.1)
    out = paste(img, c, info, "mask", 8, 24, 32, "none")[0]
    inner = disk(H, W, cx, cy, r * 0.8)[0].bool()
    assert (out[0] - img[0]).abs().amax(-1)[inner].max().item() < 1e-3
    marked = c * 0 + torch.tensor([1.0, 0, 0])               # paint the cut red
    out2 = paste(img, marked, info, "mask", 8, 24, 32, "none")[0]
    far = (1 - disk(H, W, cx, cy, r * 3)[0]).bool()
    assert (out2[0] - img[0]).abs().amax(-1)[far].max().item() == 0
    assert out2[0, cy, cx].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-3)


def test_batch_keeps_each_center():
    img = scene(512, 512).repeat(5, 1, 1, 1)
    m = torch.cat([disk(512, 512, 200 + 10 * i, 250, 50) for i in range(5)])
    c, cm, info = cut(img, m, 768, 1.8, 6.0, 0, 16, True, 0.1)
    out = paste(img, c, info, "mask", 8, 24, 32, "none")[0]
    assert c.shape == (5, 768, 768, 3) and out.shape == img.shape
    xs = info["t"][:, 0].tolist()
    assert xs == pytest.approx([xs[0] + 10 * i for i in range(5)], abs=1e-3)


def test_channels():
    """Qwen Image 2.1's VAE decodes RGBA from an RGB source."""
    img, m = scene(512, 512), disk(512, 512, 256, 256, 80)
    c, cm, info = cut(img, m, 512, 1.8, 6.0, 0, 16, True, 0.1)
    alpha = torch.ones_like(c[..., :1])
    rgb_out = paste(img, c, info, "mask", 8, 24, 32, "none")[0]
    rgba_out = paste(img, torch.cat([c, alpha], -1), info, "mask", 8, 24, 32, "none")[0]
    assert rgba_out.shape == img.shape
    assert torch.equal(rgba_out, rgb_out), "RGBA processed must blend exactly like RGB"
    rgba_src = torch.cat([img, torch.full_like(img[..., :1], 0.5)], -1)
    c4, _, info4 = cut(rgba_src, m, 512, 1.8, 6.0, 0, 16, True, 0.1)
    kept = paste(rgba_src, c4[..., :3], info4, "mask", 8, 24, 32, "none")[0]
    assert kept.shape == rgba_src.shape
    assert torch.equal(kept[..., 3], rgba_src[..., 3]), "source alpha must survive an RGB processed"


def test_mask_on_image_border():
    """The cut stays inside the image and the border pixels are replaced."""
    img, m = scene(1433, 1434), rect(1433, 1434, 300, 0, 900, 500)
    c, cm, info = cut(img, m, *ARGS)
    x0, y0, x1, y1 = source_box(info)
    assert x0 >= -0.01 and y0 >= -0.01 and x1 <= 1434.01 and y1 <= 1433.01, (x0, y0, x1, y1)
    marked = torch.zeros_like(c)
    marked[..., 0] = 1
    blended, bm, *_ = paste(img, marked, info, "mask", 8, 24, 32, "none")
    assert blended[0, 0, 600, 0] > 0.999 and blended[0, 0, 600, 1] < 0.001
    assert bm[0, 0, c.shape[2] // 2 - 40] > 0.999, "the side on the image border must not fade"
    assert (blended[0, 1000:, :, 1] - img[0, 1000:, :, 1]).abs().max() == 0, \
        "pixels below the cut must not change"
    assert bm[0, -1].max() < 0.02, "the other sides still fade out"


@pytest.mark.parametrize("box,expect", [
    ((800, 700, 1200, 1300), (2, 3)), ((500, 800, 1500, 1300), (16, 9)),
    ((800, 300, 1000, 1700), (9, 16)), ((200, 950, 1800, 1100), (21, 9))])
def test_nearest_ratio_only_adds_area(box, expect):
    img = scene(2000, 2000)
    c, cm, info = cut(img, rect(2000, 2000, *box), 1024, 1.1, 6.0, 0, 16, True, 0.1)
    x0, y0, x1, y1 = source_box(info)
    assert x0 <= box[0] and y0 <= box[1] and x1 >= box[2] and y1 >= box[3], "mask must stay covered"
    assert (x1 - x0) / (y1 - y0) / (expect[0] / expect[1]) == pytest.approx(1, abs=0.03)


def test_ratio_that_does_not_fit_is_skipped():
    img = scene(500, 1000)
    c, cm, info = cut(img, rect(500, 1000, 0, 0, 1000, 500), 1024, 1.1, 6.0, 0, 16, True, 0.1)
    x0, y0, x1, y1 = source_box(info)
    assert y1 - y0 <= 512 and x1 - x0 <= 1024, (x0, y0, x1, y1)


def test_small_image_low_max_upscale_shrinks_cut():
    img = scene(512, 512)
    c, cm, info = cut(img, rect(512, 512, 250, 250, 262, 262), 1024, 1.1, 1.0, 0, 16, True, 0.1)
    assert (info["cut_w"], info["cut_h"]) == (512, 512)


def test_nearest_ratio_off_warns_over_2_to_1(caplog):
    img = scene(2000, 2000)
    with caplog.at_level(logging.WARNING):
        c, cm, info = cut(img, rect(2000, 2000, 200, 950, 1800, 1100), 1024, 1.1, 6.0, 0, 16,
                          False, 0.1)
    assert len(caplog.records) == 1 and "over 2:1" in caplog.records[0].getMessage()
    assert info["cut_w"] / info["cut_h"] > 10
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        cut(img, rect(2000, 2000, 800, 700, 1200, 1300), 1024, 1.1, 6.0, 0, 16, False, 0.1)
    assert not caplog.records


def test_min_target_size():
    """Over max_upscale the cut shrinks and keeps the region, down to min_target_size."""
    img = scene(2000, 2000)
    c, cm, info = cut(img, rect(2000, 2000, 950, 950, 1050, 1050), 1024, 1.1, 6.0, 0, 16, True,
                      0.1, 512)
    assert (info["cut_w"], info["cut_h"]) == (656, 656)
    assert abs(info["cut_w"] * info["scale"] - 110) < 2       # keeps context_expand
    m = rect(2000, 2000, 990, 990, 1010, 1010)
    c, cm, info = cut(img, m, 1024, 1.1, 6.0, 0, 16, True, 0.1, 512)
    assert (info["cut_w"], info["cut_h"]) == (512, 512)
    assert abs(1 / info["scale"] - 6) < 0.01                  # the region grew, the limit held
    c, cm, info = cut(img, m, 768, 1.1, 6.0, 0, 16, True, 0.1, 1024)
    assert info["cut_w"] == 768, "the floor can't pass target_size"
    c, cm, info = cut(img, rect(2000, 2000, 900, 800, 1100, 1400), 1024, 1.1, 6.0, 0, 16, True,
                      0.1, 768)
    assert max(info["cut_w"], info["cut_h"]) == 1024, "under the limit: no effect"


def test_cut_area_preview():
    img = scene(768, 1024).repeat(2, 1, 1, 1)
    m = torch.cat([disk(768, 1024, 400, 300, 60), disk(768, 1024, 980, 700, 40)])
    prev = _cut(img, m, *ARGS)[3]
    assert prev.shape == (2, 768, 1024, 3)
    assert (prev - img).abs().amax() > 0.5, "the preview must draw on the image"
    prev = _cut(scene(4320, 7680), disk(4320, 7680, 3000, 2000, 700),
                1024, 1.8, 6.0, 20, 16, True, 0.1)[3]
    assert max(prev.shape[1:3]) == 2048


def test_small_spots_ignored():
    """Leftover specks don't stretch the cut and leave the cut mask."""
    img = scene(1200, 1600)
    face = disk(1200, 1600, 700, 500, 150)
    specks = face.clone()
    specks[0, 1000:1006, 100:106] = 1
    specks[0, 50:53, 1500:1503] = 1
    base = _cut(img, face, *ARGS)
    kept = _cut(img, specks, *ARGS)
    assert torch.equal(kept[2]["t"], base[2]["t"]) and kept[2]["scale"] == base[2]["scale"]
    assert torch.equal(kept[1], base[1]), "specks must leave the cut mask"
    off = _cut(img, specks, *ARGS, 512, False)
    assert off[2]["scale"] > base[2]["scale"] * 2, "with the toggle off the specks still count"
    two = disk(1200, 1600, 500, 500, 100) + disk(1200, 1600, 1100, 500, 90)
    a = _cut(img, two, *ARGS)
    b = _cut(img, two, *ARGS, 512, False)
    assert a[2]["scale"] == b[2]["scale"], "spots over the share must stay"


def test_paste_area_and_paste_mask():
    """paste_mask is exactly the blend applied to each source pixel."""
    img = scene(1200, 1600).repeat(2, 1, 1, 1)
    m = torch.cat([disk(1200, 1600, 700, 500, 150), disk(1200, 1600, 1500, 120, 100)])
    c, cm, info = cut(img, m, *ARGS)
    out, bm, _, area, pm = paste(img, c.flip(-1), info, "mask", 8, 24, 32, "none")
    assert pm.shape == (2, 1200, 1600) and area.shape[0] == 2 and area.shape[-1] == 3
    assert (out - img).abs().amax(-1)[pm == 0].max() == 0, "nothing outside paste_mask may change"
    assert pm[0, 500, 700] > 0.99 and pm[0, 1100, 100] == 0
    paste(img, c.flip(-1), info, "full_cut", 8, 24, 32, "none")


def test_align_without_flow_is_an_error():
    img, m = scene(512, 512), disk(512, 512, 256, 256, 80)
    c, cm, info = cut(img, m, 512, 1.8, 6.0, 0, 16, True, 0.1)
    with pytest.raises(ValueError, match="needs optical flow"):
        paste(img, c, info, "mask", 8, 24, 32, "seam")


@pytest.mark.parametrize("rot", [0, 25])
def test_round_trip_keeps_detail(rot):
    """Stripes 4 source px apart survive an upscaled cut pasted back unchanged.
    The Gaussian prefilter the paste used before kept 77 % of their contrast,
    visibly softening a detailed face; Lanczos-3 keeps all of it."""
    H, W, cx, cy, r = 768, 1024, 400, 300, 60
    x = torch.arange(W).float()[None, :, None].expand(H, W, 3)
    img = (0.5 + 0.3 * torch.sin(2 * math.pi * x / 4 + 0.3))[None]
    c, cm, info = cut(img, disk(H, W, cx, cy, r), 1024, 1.8, 6.0, rot, 16, True, 0.1)
    out = paste(img, c, info, "mask", 8, 24, 32, "none")[0]
    inner = disk(H, W, cx, cy, r * 0.8)[0].bool()
    kept = out[0, ..., 0][inner].std() / img[0, ..., 0][inner].std()
    assert kept > 0.95
