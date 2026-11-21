"""Alignment math without torch: OpenCV DIS flow stands in for RAFT.

The processed cut has an eye-like blob moved 4 px right of the original's, one
deep in the mask's core and one on the blend edge. seam: lined up at the edge,
left where the edit put it in the core. full: lined up everywhere.
"""
import cv2
import numpy as np
import pytest

from tps_cut_paste import seam_align

H = W = 512


def with_blob(img, cx, cy):
    out = img.copy()
    y, x = np.mgrid[0:H, 0:W]
    out[np.hypot(x - cx, y - cy) < 14] = [0.05, 0.05, 0.05]
    return out


def dis_flows():
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def gray(image):
        return cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    def flows(a, b):
        return (dis.calc(gray(a), gray(b), None).transpose(2, 0, 1),
                dis.calc(gray(b), gray(a), None).transpose(2, 0, 1))
    return flows


def centre_x(img, cx, cy):
    win = img[cy - 30:cy + 30, cx - 30:cx + 30].mean(-1) < 0.2
    ys, xs = np.nonzero(win)
    return xs.mean() + cx - 30


@pytest.fixture(scope="module")
def pair():
    tex = cv2.GaussianBlur(np.random.default_rng(0).random((H, W, 3)).astype(np.float32),
                           (0, 0), 2.0)
    original = with_blob(with_blob(tex, 256, 256), 256, 110)
    processed = with_blob(with_blob(tex, 260, 256), 260, 110)
    y, x = np.mgrid[0:H, 0:W]
    alpha = np.clip((150 - np.hypot(x - 256, y - 256)) / 40, 0, 1).astype(np.float32)
    return original, processed, alpha


def aligned(pair, mode):
    original, processed, alpha = pair
    extra = seam_align.displacement(original, processed, dis_flows(), alpha, mode, seam_band=48)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(processed, gx + extra[..., 0], gy + extra[..., 1], cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)


def test_seam_keeps_the_core_and_lines_up_the_edge(pair):
    out = aligned(pair, "seam")
    assert centre_x(out, 258, 256) == pytest.approx(260, abs=0.25)   # as generated
    assert centre_x(out, 258, 110) == pytest.approx(256, abs=1.5)    # back on the original


def test_full_lines_up_everything(pair):
    out = aligned(pair, "full")
    assert centre_x(out, 258, 256) == pytest.approx(256, abs=1.5)
    assert centre_x(out, 258, 110) == pytest.approx(256, abs=1.5)


def test_too_few_points_leaves_the_cut_alone(pair):
    original, processed, alpha = pair

    def lost(a, b):     # a flow that never comes back
        size = seam_align._working_size(W, H)
        return np.full((2, size[1], size[0]), 50.0, np.float32), np.zeros((2, size[1], size[0]),
                                                                           np.float32)
    assert seam_align.displacement(original, processed, lost, alpha, "seam") is None


def test_unknown_mode(pair):
    original, processed, alpha = pair
    with pytest.raises(ValueError, match="seam or full"):
        seam_align.displacement(original, processed, dis_flows(), alpha, "half")


def test_raft_flows_scales_to_0_255():
    seen = []

    def flow(first, second):
        seen.append((first.max(), second.max()))
        return "forward", "backward"
    assert seam_align.raft_flows(flow)(np.ones((8, 8, 3)), np.full((8, 8, 3), 0.5)) == \
        ("forward", "backward")
    assert seen == [(255.0, 127.5)]
