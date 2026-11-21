"""The thin-plate spline: a smooth displacement field from scattered offsets.

Every matched point says how far the image is off at its place. A pixel moves
by a smooth blend of what the points near it say: all the way at a point, less
with distance, not at all once the neighbouring points are reached if those did
not move. It is a thin-plate spline - a thin sheet pinned at each point and
left to bend as gently as it can between them (scipy's `RBFInterpolator`,
kernel `thin_plate_spline`, degree 1). No triangles, so none of the creases a
triangle mesh leaves along its edges.

**It gives way.** The sheet is not forced through every point. One point that
moved against still neighbours is a wrong match far more often than a feature;
`SMOOTHING` lets the sheet miss each point a little, which flattens a spike and
leaves a movement several points share alone.

**It stops.** The blend is eased to nothing just beyond the points, so what lies
outside them is not moved.

Taken unchanged from SameFace's smooth warp (github.com/fixed-42/comfyui-sameface).
"""
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

# How readily the sheet gives way, with the points' spacing as the unit of
# length. Zero passes through every point. Chosen on hand-made cases: at 1.0
# a lone point moving against still neighbours is followed about half way,
# two neighbours moving together about 70%, and a patch of points moving
# together all of it.
SMOOTHING = 1.0
# How far beyond the points the blend takes to ease to nothing, in spacings.
FADE = 2.0


def spacing_of(points):
    """How far a point typically is from its nearest neighbour."""
    points = np.asarray(points, float)
    if len(points) < 2:
        return 0.0
    nearest, _ = cKDTree(points).query(points, k=2)
    return float(np.median(nearest[:, 1]))


def field(points, offsets, frame_size, smoothing=SMOOTHING, fade=FADE):
    """How far to move at every pixel of a frame: `offsets` at `points`,
    blended smoothly between them and eased to nothing beyond them.

    Returns `(height, width, 2)`, in the units the offsets are given in.
    """
    points, offsets = np.asarray(points, float), np.asarray(offsets, float)
    width, height = (int(round(side)) for side in frame_size)
    spacing = spacing_of(points)
    if spacing <= 0 or not np.any(offsets):
        return np.zeros((height, width, 2))

    # The sheet, solved with the spacing as the unit so `smoothing` means the
    # same thing whatever size the image is.
    sheet = RBFInterpolator(points / spacing, offsets, kernel="thin_plate_spline",
                            smoothing=smoothing, degree=1)
    # Evaluated on a coarse grid and enlarged: the sheet is smooth on the
    # scale of the points, so nothing is lost and a megapixel frame costs the
    # same as a small one.
    step = max(2.0, spacing / 4.0)
    xs = np.arange(0.0, width + step, step)
    ys = np.arange(0.0, height + step, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coarse = sheet(np.column_stack([grid_x.ravel(), grid_y.ravel()]) / spacing)
    coarse = coarse.reshape(len(ys), len(xs), 2).astype(np.float32)
    # A coarse sample sits at a pixel, not at the centre of a cell, so the
    # enlargement is an explicit resampling at each pixel's own position.
    pixel_x, pixel_y = np.meshgrid(np.arange(width, dtype=np.float32) / step,
                                   np.arange(height, dtype=np.float32) / step)
    blended = cv2.remap(coarse, pixel_x, pixel_y, cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE)

    # Eased to nothing beyond the points. Within a spacing of a point the
    # blend is whole; from there it fades over `fade` spacings.
    away = np.ones((height, width), np.uint8)
    inside = ((points >= 0) & (points < [width, height])).all(axis=1)
    marks = np.round(points[inside]).astype(int)
    away[np.clip(marks[:, 1], 0, height - 1), np.clip(marks[:, 0], 0, width - 1)] = 0
    distance = cv2.distanceTransform(away, cv2.DIST_L2, 5)
    weight = np.clip(1.0 - (distance - spacing) / (fade * spacing), 0.0, 1.0)
    weight = weight * weight * (3.0 - 2.0 * weight)          # no corner where it starts
    return blended.astype(float) * weight[..., None]
