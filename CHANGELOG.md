# Changelog

## 0.2.0

- **TPS Color Match**, a new node between the edit and TPS Paste, matches the edit's
  color and texture to the original: `color` (grade + edge / grade / edge / off),
  `grade_strength`, `edge_profile`, `edge_width` and `texture` (match / off). The
  edit's overall grade is measured only where the edit left the picture alone,
  outside-in from the painted mask, so a new object neither pulls it nor gets
  corrected away.
- Outputs the matched cut (in the cut's own layout, so TPS Paste aligns it as
  before), `change_mask` (what the edit changed, cut space) and `measure_view`
  (where it measured, the plateau before and after, the grade, confidence and
  warnings). Bypass it to compare.
- Optional `change_mask` input for when the change found is wrong (an edit that
  redraws the texture past the mask): painted, or from a segmentation model, at
  source or cut size. Light the change cast past it is still found by color.
- **TPS Paste** gains `feather_detail` (keep / plain): the feather keeps the detail
  energy of both sides instead of a soft halo. Saved 0.1.0 workflows load
  unchanged.
- Test workflows in `example_workflows/tests/`.

## 0.1.0

First public release, as two V3 nodes:

- **TPS Cut** and **TPS Paste** (node IDs `TPSCut`, `TPSPaste`; link type
  `TPS_WARP_INFO`). Earlier private builds named them True Warp Cut / Paste; their
  saved workflows do not load into these nodes.
- Alignment uses RAFT optical flow and a smoothed thin-plate spline, with the
  spline and RAFT loader taken from SameFace.
- Licence: PolyForm Noncommercial 1.0.0.
