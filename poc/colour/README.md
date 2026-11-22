# Colour and texture match: proof of concept

Not part of the node pack. Measurements and a prototype for matching an edit
model's output to the original before TPS Paste blends it back. Design notes,
numbers and decisions: the "Colour matching after edit models" doc.

The idea: after alignment, the difference between edit and original is a
plateau (the edit's overall grade) with the changed object standing on it.
Measure the plateau only where the edit changed nothing, outside-in from the
painted mask, and undo it; leave the object and its effect on its surroundings.

| File | What it does |
| --- | --- |
| `colour_match.py` | The prototype. Stage 1 trust (colour and round-trip z-scores against 2D plateau and floor maps, painted mask as prior, change mask), stage 2 grade (Oklab offset per tone, flat beyond the measured tones), stage 3 edge field with profiles, stage 4 texture (local gain per Laplacian band, energy-preserving blend) |
| `common.py` | Colour spaces (sRGB, linear, Oklab), normalized convolution, distances, loading a test pair |
| `measure.py` | The five measurements: ring profile, representations, floor maps, texture spectrum, SSIM split |
| `prototype.py` | Stages 1-3 against no correction and a cut-area (Reinhard) match |
| `texture.py` | Stage 4: band gains and blends, seam energy profile |
| `redraw.py` | Plateau evenness and scores on redraw-heavy edits, with the painted mask as prior |
| `border_check.py` | How a mask touching the image border shortens the band |

Inputs are aligned pairs written by `finish_test.py` (outside this repo):
`<prefix>_orig_editframe.png`, `<prefix>_aligned.npy`, `<prefix>_soft.npy`, plus
an RGBA image whose transparent area is the painted mask. Run from this folder
with ComfyUI's Python or any env with numpy, opencv, scipy, torch, torchvision
and matplotlib, e.g.

    python prototype.py <results dir> <out dir> fin_balloon demo_mona_lisa_balloon.png
