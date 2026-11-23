# comfyui-tps-cut-paste

Three ComfyUI nodes for editing a masked area at a fixed working size and pasting
it back aligned:

- **TPS Cut** — similarity warp of the masked region into a cut of `target_size`.
- **TPS Color Match** — between the edit and the paste: undoes the overall color and
  texture change the edit model made, measured only where the edit left the picture
  alone.
- **TPS Paste** — inverse warp and masked blend back into the source, after aligning
  the edited cut to the original with RAFT optical flow and a smoothed thin-plate
  spline (TPS).

Only the cut passes through VAE and sampler, so VRAM scales with `target_size`, not
with the source resolution. The paste writes only the source rectangle the cut
covers; every pixel outside the blend mask is returned unchanged.

## Method

### Cut

Coordinates are continuous, pixel *i* centred at *i* + 0.5. The cut-to-source map is
a similarity transform

    source_point = A · cut_point + t,   A = s · R(θ)

(uniform scale *s*, rotation θ = `rotation_deg`, translation *t*), sampled with
`torch.nn.functional.grid_sample`, bicubic, at sub-pixel precision. Whenever the
map downsamples by more than 5 %, the input is lowpassed first to avoid aliasing:
a separable Lanczos-3 windowed sinc with its cutoff at the output's Nyquist
frequency for the picture, and a Gaussian (σ = 0.45 × the downsampling factor, in
input pixels) for masks, which must not ring. This applies to the cut of a large
region and to the paste of an upscaled small one. A Gaussian on the picture as
well cut the contrast of detail at a quarter of the output sampling rate to 77 %,
so a 4× cut pasted back unchanged came out visibly softer than the source;
Lanczos-3 keeps it in full (`tests/test_cut_paste.py`), with similar aliasing.

Region selection, in order:

1. Mask thresholded at `mask_threshold`; with `ignore_small_spots`, 8-connected
   components smaller than `small_spot_share` of the total mask area are dropped
   (the largest always stays). Dropped spots are also removed from `cut_mask`.
2. Mask bounding box × `context_expand`.
3. With `nearest_ratio`, widened to the nearest of 1:1, 3:2, 4:3, 16:9 (either
   orientation) or 21:9. Without it, a ratio over 2:1 logs a warning.
4. Scaled so the long side is `target_size`, rounded to `size_multiple`. If that
   needs more than `max_upscale`, the cut shrinks instead, down to
   `min_target_size`; below that the region grows to take in more context.
5. Slid to stay inside the image. A rotated region is not clipped to the image
   size and only its centre is fitted, so its corners can still cross the border.

A batch shares one cut size; each image keeps its own centre.

### Paste

Blend mask (cut space): cut mask → grow/erode by `grow` px → Gaussian feather of
`feather` px → × a smoothstep ramp of `edge_feather` px from the cut border. Sides of the
cut lying on the image border get no ramp. `blend = full_cut` uses the ramp alone;
an optional `blend_mask` input replaces the stored mask.

The aligned, processed cut is inverse-warped into the source rectangle the cut
covers and blended `out = src · (1 − α) + cut · α`.

### Alignment (`align`)

A detailer at moderate denoise moves features by a few pixels. Blended through a
feathered mask, the original and the moved edge both show in the feather band. The
alignment resamples the processed cut so it agrees with the original where α < 1:

1. The original region is resampled into cut space with the same (A, t), so the two
   cuts share one frame and no global fit is needed.
2. Both are resized to a long side of 512 (multiples of 8) and RAFT
   (`torchvision` `raft_large`, `C_T_SKHT_V2` weights) gives forward and backward
   flow.
3. Grid points every 16 working px; a point is kept if its forward-backward round
   trip returns within 1.5 px and it lands inside the frame. Fewer than 8 kept points
   → no alignment (reported in `paste_area`).
4. Offsets *d* = landed − origin, mapped back to cut pixels.
   - `seam`: *d* is weighted by w = 1 − core, where core is the α ≥ 0.99 region
     eroded by `seam_band` px and Gaussian-blurred (σ = max(1, `seam_band`/3)).
     Points in the solid core keep the generated geometry; points in the feather band
     are pulled onto the original.
   - `full`: *d* unweighted; the whole cut takes the original's layout.
5. A thin-plate spline (`scipy.interpolate.RBFInterpolator`, kernel
   `thin_plate_spline`, degree 1) is fitted to the offsets with coordinates in units
   of the median nearest-neighbour spacing, smoothing 1.0. In those units a lone
   outlier among still neighbours is followed about halfway, two agreeing points
   about 70 %, a coherent patch fully. The spline is evaluated on a grid of
   max(2, spacing/4) px and upsampled bicubically, then eased to zero over 2
   spacings beyond the outermost points (smoothstep).
6. The processed cut is resampled at pixel + `align_strength` · field (bicubic,
   border padding).

The TPS field, the round-trip sampling and the RAFT loader come from
[SameFace](https://github.com/fixed-42/comfyui-sameface), which uses them to measure
and repair drift between a before and an after image.


### Color and texture (TPS Color Match; TPS Paste `feather_detail`)

An edit model changes more than the object it was asked for: the whole cut comes
back slightly darker or warmer, with fine texture smoothed and mid-scale contrast
raised. Matching statistics over the cut does not undo that once the edit adds an
object, because the object pulls them: with a red balloon filling half the mask, a
mean-and-spread match brightened the unchanged sky by +0.037 Oklab L, the wrong way.

The aligned difference *processed − original* is treated as a height map: the
edit's overall change lifts or lowers a plateau, and what the edit actually changed
stands on it. All of this runs in Oklab, on the original and the processed cut
lined up with it (a `full` align of its own), at a long side of at most 1024. The
correction is then applied to the processed cut as it came, so TPS Paste can align it
in any mode: the grade goes by each pixel's own lightness, and the edge field and
texture gains are smooth over tens of pixels, so the few pixels the alignment moves
do not matter. TPS Paste's own alignment then runs on a cut whose color already
matches.

1. **Trust.** For the color difference and for the RAFT round-trip error, a local
   plateau (normalized-convolution median of the difference) and floor (its median
   absolute deviation) are estimated, seeded by a global median/MAD pass; a pixel
   more than 4 floors off is changed, and one excluded once stays excluded, so an
   object as big as the window cannot raise its own floor. The painted mask (before
   `grow` and feather) is a prior: its inside never helps define the plateau, which
   is filled in there from outside. Without it a large recolor (a dress, 44 % of the
   frame) was not found at all, since it keeps the folds and looks like a local
   grade. Trust = soft weights from both z-scores × flatness, outside the change
   mask.

   The round trip fails wherever the edit redrew the texture, not only on what it
   added: a lantern edit that repainted the whole sky had a 2.5 px round-trip error
   there (0.08 px below), and the change mask flooded the sky. A `change_mask` input
   then takes the place of the round trip: the change is the given mask plus the
   color peaks, so light the change cast past the given mask still counts. On the
   balloon, the color peaks alone overlap a SAM 2.1 segmentation at IoU 0.93; on the
   lantern, the given mask plus color peaks is the lantern and its glow, without the
   sky.
2. **Grade** (`color` grade): the median Oklab difference of trusted pixels in 16
   equal-count bins of the edit's lightness, lightly smoothed, applied as a function
   of each pixel's lightness and held flat beyond the measured tones (a fitted gain
   overshot by +0.06 beyond them). Scaled by `grade_strength` and by a confidence
   from the band's completeness and size, halved when a found change reaches past
   the mask (not a given one).
3. **Edge field** (`color` edge): what the grade leaves within 48 cut px of the
   blend edge (the painted mask grown as TPS Paste grows it by default, 8 px), spread by normalized convolution and faded inward by `edge_profile`
   over `edge_width`; where nothing near the edge was trusted it fades to nothing
   rather than borrowing a correction from far away. Distances count real mask edges
   only, so sides on the image border stay open. It carries the correction when the
   plateau is uneven (an edit that relights the scene); `edge` alone keeps the
   edit's own grade inside.
4. **Texture** (`texture` match): a local gain per Laplacian band of lightness,
   √(E_original / E_processed) over trusted pixels (clamped 0.5–2), applied to the
   processed cut.
5. **Feather** (TPS Paste `feather_detail` keep): two different textures mixed in a feather
   partly cancel; a plain mix lost about a quarter of the fine detail at α ≈ 0.5,
   a soft halo along the seam. The detail bands are rescaled locally to the energy
   (1 − α)·var_o + α·var_p using the measured covariance; α = 0 and α = 1 pixels are
   returned exactly.

`measure_view` draws the measurement: the band and its trusted pixels, the change
inside and outside the mask, open sides, the plateau before and after, the fitted
grade curve, band completeness, confidence and warnings. `change_mask` is the change
mask in cut space.

## Nodes

**TPS Cut** — `image, mask → cut, cut_mask, warp_info, cut_area`

| Input | Default | |
|---|---|---|
| `target_size` | 1024 | long side of the cut |
| `context_expand` | 1.8 | region = mask bbox × this |
| `max_upscale` | 6.0 | upscale limit; over it the cut shrinks |
| `rotation_deg` | 0 | rotate the cut (e.g. upright a tilted face) |
| `size_multiple` | 16 | cut sides rounded to this |
| `nearest_ratio` | on | snap the region to a standard ratio |
| `mask_threshold` | 0.1 | |
| `min_target_size` | 512 | floor for the shrunk cut |
| `ignore_small_spots` / `small_spot_share` | on / 0.05 | drop small mask components |

`cut_area`: the source (long side ≤ 2048) with the mask in red and one box per step —
yellow bbox, cyan × `context_expand`, magenta ratio, green final cut; ignored spots in
grey. The legend gives each size, the zoom, and whether `max_upscale` or the image
border changed the region.

**TPS Color Match** — `processed, image, warp_info [, change_mask, blend_mask] → processed, change_mask, measure_view`

| Input | Default | |
|---|---|---|
| `color` | grade + edge | `grade + edge`, `grade`, `edge` or `off` |
| `grade_strength` | 1.0 | share of the measured grade undone |
| `edge_profile` | smooth | `smooth`, `linear`, `decay` or `entire` |
| `edge_width` | 128 | cut px over which the edge correction fades inward |
| `texture` | match | `match` or `off` |
| `change_mask` | — | what the edit changed, when the one found is wrong: the size of the source (painted on it, or on the pasted result) or of the cut |
| `blend_mask` | — | the `blend_mask` given to TPS Paste, if any |

The output is the processed cut at cut size, in its own layout (not aligned).
`processed` is the first input, so a bypassed node passes the edit through. With
`color` and `texture` off nothing is measured. `change_mask` and `measure_view` are
described under Color and texture.

**TPS Paste** — `image, processed, warp_info [, blend_mask] → image, blend_mask, aligned_cut, paste_area, paste_mask`

| Input | Default | |
|---|---|---|
| `blend` | mask | `mask` or `full_cut` |
| `grow` | 8 | cut px, negative erodes |
| `feather` | 24 | cut px of Gaussian softening |
| `edge_feather` | 32 | cut px ramp at the cut border |
| `align` | seam | `seam`, `full` or `none` |
| `align_strength` | 1.0 | scales the field |
| `seam_band` | 48 | cut px of core erosion for `seam` |
| `feather_detail` | keep | `keep` or `plain` |

`aligned_cut` is the processed cut as pasted, after alignment. `paste_area` shows the result
around the paste (long side 1024–2048): red mask, yellow after `grow`, cyan/white the
2 %/98 % α contours, green the cut border and `edge_feather` line, magenta the `seam`
core boundary; the legend reports when too few points matched. `paste_mask` is α at
full image size.


## Verification

- `align = none`: a cut pasted back unchanged returns the source within 1e-3 inside
  the mask; a cut painted over changes nothing outside the paste mask (exact
  equality). Upscaled, rotated and 8K-downscaled cases, in `tests/test_cut_paste.py`.
- Alignment math on synthetic frames with OpenCV DIS flow standing in for RAFT: a blob
  moved 4 px is lined up at the blend edge and left in the core with `seam`, lined up
  everywhere with `full` (`tests/test_seam_align.py`).
- TPS field properties: zero offsets give zero field, a lone spike is damped, a
  coherent patch is followed, and the field is exactly zero beyond the fade
  (`tests/test_tps.py`).
- `seam` versus `none` has not been compared on real edits.
- Color and texture on synthetic scenes (`tests/test_color_match.py`): a darkening is
  undone off a bright object that a mean-and-spread match is pulled by; the painted
  mask finds a large recolor; a given change mask is kept out while a glow past it is
  still found; the correction carries to a shifted frame; the feather keeps detail
  energy (plain < 0.8, keep > 0.9); with everything on, nothing outside the paste
  mask changes.
- On Qwen Image 2.1 edits of the Mona Lisa at 2K (proof of concept on branch
  `color-match-poc`): seam ΔE (Oklab, unchanged pixels in the feather) 0.0116 → 0.0096
  (sparrow), 0.0117 → 0.0104 (balloon; a cut-area match made it 0.0159), 0.0133 →
  0.0100 (red dress), 0.0277 → 0.0133 (lantern relight). Detail energy inside the
  mask back to 0.98–1.05 of the original per band; fine-detail dip in the feather
  0.75 → 0.96.
- An edit that relights and redraws beyond the painted mask floods the change mask
  found; `measure_view` warns when more than 5 % of it lies outside the mask. Given
  the lantern as `change_mask` (a SAM 2.1 segmentation), the lantern edit pastes
  without the dark shadow the found mask leaves.

## Requirements

ComfyUI's torch, torchvision and Pillow, plus `numpy`, `scipy`,
`opencv-python-headless`. The RAFT-large weights (~20 MB) download from
download.pytorch.org on the first TPS Color Match, or TPS Paste with `align` on.

## Install

Clone into `ComfyUI/custom_nodes/` and restart, or install `comfyui-tps-cut-paste`
from the Comfy Registry.

## Test

```
python -m pytest -q
```

`example_workflows/tests/` holds three ComfyUI workflows that run a Qwen Image 2.1 edit
and paste it twice, with and without TPS Color Match, side by side with
`measure_view`: a sparrow (small object), a balloon (a large object) and a lantern
(an edit that relights and redraws past its mask; its note shows how to give it a
`change_mask`). Each needs its painted input image in ComfyUI's `input/`
folder (`demo_mona_lisa_*.png`).

## Licence

[PolyForm Noncommercial 1.0.0](LICENSE): free for any noncommercial purpose; commercial
use needs permission.
