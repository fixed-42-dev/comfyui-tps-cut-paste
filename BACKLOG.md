# Backlog

## Make `target_size` mean total resolution

`target_size` sets the long side of the cut, so the pixel count drops as the ratio
narrows. At 1024, 1:1 gets 1.05 MP but 21:9 gets only 0.44 MP.

Proposal: treat `target_size` as total area, so 1024 means about 1024×1024 pixels at
any ratio. For example, 3:4 becomes 880×1184 and 21:9 becomes 1568×672.

To decide:
- Test on the real models whether equal area gives better results across ratios.
- Non-square cuts get larger, so time and VRAM rise compared with today.
- Decide what `min_target_size` means under this change, as it's also a long side today.

## Compare `align = seam` with `none` on real edits

Never measured. Run the same edits through both and compare the feather band, by
eye and by a drift measurement. Test together with the paste grid density below.

## Paste grid density

Alignment samples a point every 16 working px on a 512 working frame, so on a 1024
cut that is one point per 32 cut px, against a `seam_band` of 48. Check whether a
denser grid lines up the feather band better.
