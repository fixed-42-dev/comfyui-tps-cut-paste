"""The thin-plate-spline field: how far each pixel is moved.

- **it stops** - beyond the points nothing is moved;
- **it gives way** - a lone point that moved against still neighbours is a bad
  match more often than a feature, and is not followed all the way;
- points moving together carry everything between them.

No GPU and no photographs: the points are made by hand. Taken from SameFace's
smooth-warp tests.
"""
import numpy as np
import pytest

from tps_cut_paste.tps import field

SIZE = 200


def grid(lo=60, hi=140, step=10):
    return np.array([[x, y] for y in range(lo, hi + 1, step)
                     for x in range(lo, hi + 1, step)], float)


def at(found, x, y):
    return found[int(y), int(x)]


# ---------- the field: how far each pixel is moved ----------


def test_where_nothing_moved_nothing_is_moved():
    found = field(grid(), np.zeros((len(grid()), 2)), (SIZE, SIZE))

    assert np.abs(found).max() < 1e-6


def test_one_point_moves_and_its_neighbours_hold_still():
    """The Liquify picture: full at the point, fading, gone by the anchors."""
    points = grid()
    offsets = np.zeros_like(points)
    centre = int(np.flatnonzero((points == [100, 100]).all(axis=1))[0])
    offsets[centre] = [4.0, 0.0]

    found = field(points, offsets, (SIZE, SIZE), smoothing=0.0)

    here = at(found, 100, 100)[0]
    half = at(found, 105, 100)[0]
    next_door = at(found, 110, 100)[0]
    assert here == pytest.approx(4.0, abs=0.3)
    assert 0.5 < half < here                       # less with distance
    assert abs(next_door) < 0.3                    # the anchor holds
    assert abs(at(found, 100, 100)[1]) < 0.2       # and only the way it was told


def test_points_that_all_move_together_carry_everything_between_them():
    points = grid()
    offsets = np.tile([3.0, -2.0], (len(points), 1))

    found = field(points, offsets, (SIZE, SIZE))

    for x, y in ((100, 100), (75, 125), (95, 65)):
        assert at(found, x, y) == pytest.approx([3.0, -2.0], abs=0.15)


def test_it_stops_outside_the_points():
    """The face is moved; what is beyond it is left as the flat fit put it."""
    points = grid()
    offsets = np.tile([3.0, -2.0], (len(points), 1))

    found = field(points, offsets, (SIZE, SIZE))

    for x, y in ((5, 5), (195, 100), (100, 190)):
        assert np.abs(at(found, x, y)).max() == 0.0
    # ...and gets there gradually. Within a spacing of the last point it is
    # still whole; a little further out it is part of the way.
    assert at(found, 148, 100)[0] == pytest.approx(3.0, abs=0.15)
    assert 0.5 < at(found, 158, 100)[0] < 2.9


def test_a_lone_spike_is_not_followed_all_the_way():
    """One point moving against still neighbours is usually a bad match.
    With smoothing the warp gives way to it; without, it passes through."""
    points = grid()
    offsets = np.zeros_like(points)
    centre = int(np.flatnonzero((points == [100, 100]).all(axis=1))[0])
    offsets[centre] = [6.0, 0.0]

    exact = at(field(points, offsets, (SIZE, SIZE), smoothing=0.0), 100, 100)[0]
    eased = at(field(points, offsets, (SIZE, SIZE)), 100, 100)[0]

    assert exact == pytest.approx(6.0, abs=0.4)
    assert 0.5 < eased < exact * 0.8


def test_smoothing_leaves_a_broad_movement_alone():
    """What it gives way to is the spike, not the feature: points moving
    together are followed with smoothing on."""
    points = grid()
    offsets = np.zeros_like(points)
    moved = (np.abs(points - [100, 100]) <= 20).all(axis=1)      # a 5 x 5 patch
    offsets[moved] = [4.0, 0.0]

    found = field(points, offsets, (SIZE, SIZE))

    assert at(found, 100, 100)[0] == pytest.approx(4.0, abs=0.5)
