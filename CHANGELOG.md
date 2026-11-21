# Changelog

## 0.1.0

First public release, as two V3 nodes:

- **TPS Cut** and **TPS Paste** (node IDs `TPSCut`, `TPSPaste`; link type
  `TPS_WARP_INFO`). Earlier private builds named them True Warp Cut / Paste; their
  saved workflows do not load into these nodes.
- Alignment uses RAFT optical flow and a smoothed thin-plate spline, with the
  spline and RAFT loader taken from SameFace.
- Licence: PolyForm Noncommercial 1.0.0.
