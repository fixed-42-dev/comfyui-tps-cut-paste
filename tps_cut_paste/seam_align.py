"""Line the processed cut back up with the original before it is pasted.

A detailer at moderate denoise moves features a little: an eye a few pixels
over, the jaw line slightly lower. Blend that back with a feathered mask and
the old edge and the new edge both show through the feather - a ghost. Here
the processed cut is moved so that it agrees with the original where the
blend fades, and is left alone where the blend is whole.

The original and processed cuts share one frame, so no flat fit is needed:
RAFT flow both ways, grid points whose round trip comes back, and a
thin-plate-spline field (`tps.field`) blending their offsets into a smooth
displacement. `seam` scales each point's offset by how far it is from the
mask's core, so new features inside survive; `full` pulls everything back.

The sampling settings below started equal to SameFace's measurement
(github.com/fixed-42/comfyui-sameface), where the same flow and spline were
first used.
"""
import cv2
import numpy as np

from .tps import field

WORKING_SIZE = 512      # long side RAFT sees; multiples of 8
GRID_STEP = 16          # working px between sampled points
CONSISTENCY_PX = 1.5    # forward-backward agreement for a point to count
MIN_POINTS = 8


def raft_flows(flow):
    """`flows(a, b)` for `displacement` from the pack's RAFT (`flow.Flow`), which
    takes 0-255 pixels: these frames are float in [0, 1]."""
    def flows(first, second):
        return flow(np.asarray(first, np.float32) * 255.0, np.asarray(second, np.float32) * 255.0)
    return flows


# ------------------------------------------------ matched points

def sample_matches(flow_ab, flow_ba, step=GRID_STEP, consistency=CONSISTENCY_PX):
    """Grid points in the first frame, where flow puts them in the second,
    and whether the round trip comes back. Flows are (2, h, w)."""
    _, h, w = flow_ab.shape
    half = step // 2
    xs, ys = np.meshgrid(np.arange(half, w - half, step), np.arange(half, h - half, step))
    xs, ys = xs.ravel(), ys.ravel()
    forward = np.column_stack([xs + flow_ab[0, ys, xs], ys + flow_ab[1, ys, xs]])
    bx = np.clip(np.rint(forward[:, 0]).astype(int), 0, w - 1)
    by = np.clip(np.rint(forward[:, 1]).astype(int), 0, h - 1)
    back = np.column_stack([forward[:, 0] + flow_ba[0, by, bx],
                            forward[:, 1] + flow_ba[1, by, bx]])
    origin = np.column_stack([xs, ys]).astype(float)
    error = np.hypot(*(back - origin).T)
    inside = ((forward >= 0) & (forward <= [w - 1, h - 1])).all(axis=1)
    return origin, forward, (error <= consistency) & inside


def _working_size(w, h):
    k = WORKING_SIZE / max(w, h)
    return max(8, int(round(w * k / 8)) * 8), max(8, int(round(h * k / 8)) * 8)


def seam_weight(alpha, seam_band):
    """1 where the blend is partial or absent, 0 in the mask's solid core."""
    core = (alpha >= 0.99).astype(np.uint8)
    if seam_band > 0:
        core = cv2.erode(core, np.ones((3, 3), np.uint8), iterations=int(seam_band))
        core = cv2.GaussianBlur(core.astype(np.float32), (0, 0), max(1.0, seam_band / 3.0))
    return 1.0 - np.clip(core.astype(np.float32), 0.0, 1.0)


def displacement(original, processed, flows, alpha, mode="seam", seam_band=48):
    """Where to read the processed cut from, per cut pixel, so it lines up
    with the original: sample processed at (pixel + returned field).

    original, processed: (h, w, 3) float32 in [0, 1]. alpha: (h, w) final blend
    weight. `flows(a, b)` takes two (h', w', 3) float32 frames and returns the
    forward and backward flow, each (2, h', w'). Returns None when too few
    points can be trusted, which leaves the cut as it is.
    """
    h, w = original.shape[:2]
    ww, wh = _working_size(w, h)
    small_o = cv2.resize(original, (ww, wh), interpolation=cv2.INTER_AREA)
    small_p = cv2.resize(processed, (ww, wh), interpolation=cv2.INTER_AREA)
    flow_ab, flow_ba = flows(small_o, small_p)
    origin, landed, keep = sample_matches(flow_ab, flow_ba)
    if keep.sum() < MIN_POINTS:
        return None

    scale = np.array([w / ww, h / wh])
    origin = (origin[keep] + 0.5) * scale - 0.5      # pixel centres, cut frame
    landed = (landed[keep] + 0.5) * scale - 0.5
    offsets = landed - origin                          # source - target
    if mode == "seam":
        weight = seam_weight(alpha, seam_band)
        ix = np.clip(np.rint(origin[:, 0]).astype(int), 0, w - 1)
        iy = np.clip(np.rint(origin[:, 1]).astype(int), 0, h - 1)
        offsets = offsets * weight[iy, ix][:, None]
    elif mode != "full":
        raise ValueError(f"mode is seam or full, not {mode!r}")
    return field(origin, offsets, (w, h)).astype(np.float32)
