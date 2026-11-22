# Changelog

## 0.2.0

- **TPS Paste matches color and texture** to the original before blending, on by
  default: `color` (grade + edge / grade / edge / off), `grade_strength`,
  `edge_profile`, `edge_width`, `texture` (match / off) and `feather_detail`
  (keep / plain). The edit's overall grade is measured only where the edit left the
  picture alone, outside-in from the painted mask, so a new object neither pulls it
  nor gets corrected away.
- New outputs `change_mask` (what the edit changed, source space) and `measure_view`
  (where it measured, the plateau before and after, the grade, confidence and
  warnings). They come after the existing outputs, so saved links stay.
- `aligned_cut` is now the cut as pasted, after color and texture.
- Optical flow (RAFT) also runs when `align` is `none` but color or texture is on.
- Test workflows in `example_workflows/tests/`.

## 0.1.0

First public release, as two V3 nodes:

- **TPS Cut** and **TPS Paste** (node IDs `TPSCut`, `TPSPaste`; link type
  `TPS_WARP_INFO`). Earlier private builds named them True Warp Cut / Paste; their
  saved workflows do not load into these nodes.
- Alignment uses RAFT optical flow and a smoothed thin-plate spline, with the
  spline and RAFT loader taken from SameFace.
- Licence: PolyForm Noncommercial 1.0.0.
