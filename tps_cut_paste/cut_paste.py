"""Cut and paste: edit only a masked area, and put it back where it was.

Only the warped cut (at target_size) ever reaches VAE/KSampler, so VRAM scales
with target_size, not with the source resolution. Pasting back touches only the
source sub-rectangle the cut covers.

Transform: similarity warp (rotation + uniform scale + translation), applied with
torch grid_sample at sub-pixel precision, with a lowpass prefilter whenever the
warp downsamples (cut of a large region, or paste-back of an upscaled small face)
to avoid aliasing: Lanczos-3 for pictures, which keeps detail up to the new pixel
size, and a Gaussian for masks, which must not ring.

Convention: continuous pixel coords, pixel i has center i + 0.5.
    source_point = A @ cut_point + t

`cut`, `color_match_cut` and `paste` are the TPS Cut, TPS Color Match and TPS
Paste nodes without ComfyUI; `warp_info` is the dict `cut` returns for the others.
"""
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from . import color_match, seam_align

# TPS Paste's default grow and feather (cut px): TPS Color Match measures the
# edge correction at the edge they make.
PASTE_GROW, PASTE_FEATHER = 8, 24

# width, height. 21:9 is landscape only; tall regions stop at 9:16.
RATIOS = ((1, 1), (3, 2), (2, 3), (4, 3), (3, 4), (16, 9), (9, 16), (21, 9))

# The cut area preview is drawn at most this long, so lines and text read the same
# on any source size and an 8K source costs little.
PREVIEW_LONG_SIDE = 2048
PREVIEW_COLORS = {
    "mask": (255, 60, 60),
    "bbox": (255, 220, 0),
    "context": (0, 200, 255),
    "ratio": (255, 80, 255),
    "cut": (40, 255, 100),
    "ignored": (150, 150, 165),
}


# ---------------------------------------------------------------- helpers

def _gauss1d(sigma, device, dtype):
    r = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(x, sigma):
    """x: [N,C,H,W]. Separable Gaussian, replicate padding."""
    if sigma <= 0.05:
        return x
    k = _gauss1d(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), ky, groups=c)
    return x


def _grow(m, px):
    """m: [N,1,H,W]. Positive px dilates, negative erodes, by a square. A square's
    max is a row max then a column max, which keeps a wide grow cheap (the seam
    band's 97 x 97 in one pass took seconds)."""
    if px == 0:
        return m
    r = abs(px)
    k = 2 * r + 1
    sign = 1 if px > 0 else -1
    x = F.max_pool2d(sign * m, (1, k), stride=1, padding=(0, r))
    return sign * F.max_pool2d(x, (k, 1), stride=1, padding=(r, 0))


def _pixel_grid(h, w, device, dtype):
    ys = torch.arange(h, device=device, dtype=dtype) + 0.5
    xs = torch.arange(w, device=device, dtype=dtype) + 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)  # [h,w,2] as (x,y)


def _corners(w, h, device, dtype):
    return torch.tensor([[0, 0], [w, 0], [0, h], [w, h]], device=device, dtype=dtype)


def lanczos_blur(x, factor, a=3):
    """x: [N,C,H,W]. Separable Lanczos-windowed sinc lowpass with its cutoff at the
    Nyquist frequency of a grid `factor` px apart, replicate padding.

    A Gaussian strong enough to stop aliasing (sigma 0.45 x factor) also cuts the
    contrast of detail at a quarter of the new sampling rate to 77 %, so a
    4x-upscaled cut pasted back unchanged came out visibly softer than the source;
    this keeps that detail in full, with aliasing close to the Gaussian's."""
    fc = 0.5 / factor
    r = int(math.ceil(a * factor))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    w = torch.sinc(2 * fc * k) * torch.sinc(2 * fc * k / a) * (k.abs() < a * factor)
    w = w / w.sum()
    c = x.shape[1]
    kx = w.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = w.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kx, groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), ky, groups=c)


def warp_sample(src, A, t, out_h, out_w, mode, padding, downscale=1.0, kernel="lanczos"):
    """
    src: [1,C,H,W]. Output pixel u samples src at A @ u + t.
    `downscale` is the input px per output px; above 1.05 the input is lowpassed
    first, with `kernel` "lanczos" (pictures) or "gauss" (masks: no ringing).
    Only the source sub-rect the output touches is blurred/sampled.
    """
    _, c, H, W = src.shape
    dev, dt = src.device, src.dtype
    pts_c = _corners(out_w, out_h, dev, dt) @ A.T + t
    margin = 3.0 * downscale + 4.0
    x0 = int(math.floor(pts_c[:, 0].min().item() - margin))
    y0 = int(math.floor(pts_c[:, 1].min().item() - margin))
    x1 = int(math.ceil(pts_c[:, 0].max().item() + margin))
    y1 = int(math.ceil(pts_c[:, 1].max().item() + margin))
    x0 = min(max(x0, 0), W - 1)
    y0 = min(max(y0, 0), H - 1)
    x1 = max(min(x1, W), x0 + 1)
    y1 = max(min(y1, H), y0 + 1)

    sub = src[:, :, y0:y1, x0:x1]
    if downscale > 1.05:
        sub = (lanczos_blur(sub, downscale) if kernel == "lanczos"
               else gaussian_blur(sub, 0.45 * downscale))

    pts = _pixel_grid(out_h, out_w, dev, dt) @ A.T + t
    pts = pts - torch.tensor([x0, y0], device=dev, dtype=dt)
    sw, sh = x1 - x0, y1 - y0
    grid = torch.empty_like(pts)
    grid[..., 0] = pts[..., 0] * (2.0 / sw) - 1.0
    grid[..., 1] = pts[..., 1] * (2.0 / sh) - 1.0
    return F.grid_sample(sub, grid.unsqueeze(0), mode=mode,
                         padding_mode=padding, align_corners=False)


def _mask_bbox(m, thr):
    ys, xs = torch.nonzero(m > thr, as_tuple=True)
    if ys.numel() == 0:
        return None
    return xs.min().item(), ys.min().item(), xs.max().item() + 1, ys.max().item() + 1


def _drop_small_spots(m, thr, min_share):
    """m: [H,W]. Removes mask spots (8-connected, above thr) smaller than min_share of
    the total mask area; the largest spot always stays. Returns the new mask, the
    removed pixels ([H,W] bool, or None) and the removed spots' boxes."""
    from scipy import ndimage

    on = (m > thr).cpu().numpy()
    labels, n = ndimage.label(on, structure=np.ones((3, 3), dtype=bool))
    if n <= 1:
        return m, None, []
    areas = np.bincount(labels.ravel())[1:]
    small = areas < min_share * areas.sum()
    small[areas.argmax()] = False
    if not small.any():
        return m, None, []
    boxes = [(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)
             for sl, is_small in zip(ndimage.find_objects(labels), small) if is_small]
    drop = torch.from_numpy(np.concatenate([[False], small])[labels]).to(m.device)
    keep = torch.from_numpy(on).to(m.device) & ~drop
    # Also clear the spot's soft edge below thr, without touching a kept spot.
    near = _grow(drop.to(m.dtype)[None, None], 2)[0, 0] > 0
    return m.masked_fill(near & ~keep, 0), drop, boxes


def _expand_to_ratio(rw, rh, max_w, max_h):
    """Grow one side to the nearest listed ratio that still fits the limits.
    Keeps the region as it is if none fits."""
    current = rw / rh
    for w, h in sorted(RATIOS, key=lambda r: abs(math.log(r[0] / r[1] / current))):
        ratio = w / h
        new_w, new_h = (rh * ratio, rh) if ratio > current else (rw, rw / ratio)
        if new_w <= max_w and new_h <= max_h:
            return new_w, new_h
    return rw, rh


def _fit_center(center, half, size):
    """Slide a region along one axis so it stays inside the image."""
    if half * 2 >= size:
        return size / 2
    return min(max(center, half), size - half)


def _sides_on_image_border(A, t, w, h, W, H):
    """Which sides of the cut lie on (or past) the edge of the source image."""
    tl, tr, bl, br = (_corners(w, h, A.device, A.dtype) @ A.T + t).tolist()

    def on_border(a, b):
        return (max(a[0], b[0]) <= 1 or min(a[0], b[0]) >= W - 1
                or max(a[1], b[1]) <= 1 or min(a[1], b[1]) >= H - 1)

    sides = {"top": (tl, tr), "bottom": (bl, br), "left": (tl, bl), "right": (tr, br)}
    return {name for name, (a, b) in sides.items() if on_border(a, b)}


def _edge_ramp(h, w, width, device, dtype, open_sides=()):
    """1 in the interior, smoothstep to 0 at the cut border. Sides in open_sides
    stay at 1: they lie on the image border, so there is no seam to hide."""
    if width <= 0:
        return torch.ones(h, w, device=device, dtype=dtype)
    g = _pixel_grid(h, w, device, dtype)
    distances = {"left": g[..., 0], "right": w - g[..., 0],
                 "top": g[..., 1], "bottom": h - g[..., 1]}
    d = torch.full((h, w), float(width), device=device, dtype=dtype)
    for side, distance in distances.items():
        if side not in open_sides:
            d = torch.minimum(d, distance)
    x = (d / width).clamp(0, 1)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- cut area preview

def _ratio_label(w, h):
    """'4:3' when w/h is a listed ratio, else '1.37:1'."""
    for rw, rh in RATIOS:
        if abs(math.log((w / h) / (rw / rh))) < 0.01:
            return f"{rw}:{rh}"
    return f"{max(w, h) / min(w, h):.2f}:1" if w >= h else f"1:{h / w:.2f}"


def _box_quad(cx, cy, w, h, cos_t, sin_t):
    """Corners (tl, tr, br, bl) of a w x h box centered at cx, cy, rotated like the cut."""
    pts = []
    for u, v in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append((cx + cos_t * u - sin_t * v, cy + sin_t * u + cos_t * v))
    return pts


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has only the small bitmap font
        return ImageFont.load_default()


def _render_preview(img, mask, shapes, lines, focus, ignored=None):
    """img: [H,W,C], mask, ignored: [H,W], all in source px. ignored is tinted gray.
    shapes: (color, points, width_factor) outlines in source px, drawn in order.
    lines: (color or None, text) legend rows. focus: source point the legend avoids."""
    H, W = mask.shape
    k = min(1.0, PREVIEW_LONG_SIDE / max(H, W))
    h, w = max(1, round(H * k)), max(1, round(W * k))
    rgb = img[..., :3] if img.shape[-1] >= 3 else img[..., :1].expand(-1, -1, 3)
    if ignored is None:
        ignored = torch.zeros_like(mask)
    x = torch.cat([rgb, mask[..., None], ignored[..., None].to(mask.dtype)], -1)
    x = x.permute(2, 0, 1)[None].float()
    if k < 1:
        x = F.interpolate(x, size=(h, w), mode="area")
    x = x[0].permute(1, 2, 0).clamp(0, 1)
    rgb = x[..., :3]
    for name, m, strength in (("mask", x[..., 3:4], 0.5), ("ignored", x[..., 4:5], 0.8)):
        tint = torch.tensor(PREVIEW_COLORS[name], dtype=rgb.dtype, device=rgb.device) / 255
        rgb = rgb * (1 - strength * m) + tint * (strength * m)
    canvas = Image.fromarray((rgb * 255).round().byte().cpu().numpy()).convert("RGBA")

    lw = max(2, round(max(h, w) / 500))
    draw = ImageDraw.Draw(canvas)
    for color, pts, factor in shapes:
        pts = [(px * k, py * k) for px, py in pts]
        if len(pts) == 1:  # a center marker
            r = lw * 2.5
            (px, py), = pts
            draw.ellipse((px - r, py - r, px + r, py + r), outline=color, width=lw)
        else:
            closed = pts + [pts[0]] if len(pts) > 2 else pts
            draw.line(closed, fill=color, width=round(lw * factor), joint="curve")

    # The legend goes in the corner farthest from the cut.
    font = _font(max(12, round(max(h, w) / 75)))
    box_w, box_h = _legend_size(lines, font)
    fx, fy = focus[0] * k, focus[1] * k
    left = 0 if fx > w / 2 else w - box_w
    top = 0 if fy > h / 2 else h - box_h
    canvas = _draw_legend(canvas, lines, font, left, top)
    return torch.from_numpy(np.asarray(canvas).astype(np.float32) / 255)


def _legend_layout(lines, font):
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    pad = round(font.size * 0.5) if hasattr(font, "size") else 6
    rows = [draw.textbbox((0, 0), text, font=font) for _, text in lines]
    row_h = max(b[3] - b[1] for b in rows) + pad
    swatch = row_h - pad
    box_w = max(b[2] - b[0] for b in rows) + swatch + 3 * pad
    return pad, rows, row_h, swatch, box_w, row_h * len(lines) + pad


def _legend_size(lines, font):
    return _legend_layout(lines, font)[-2:]


def _draw_legend(canvas, lines, font, left, top):
    """Legend rows (color or None, text) on a dark box. Returns an RGB image."""
    pad, rows, row_h, swatch, box_w, box_h = _legend_layout(lines, font)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle((left, top, left + box_w, top + box_h), fill=(0, 0, 0, 170))
    for i, (color, text) in enumerate(lines):
        y = top + pad + i * row_h
        if color is not None:
            od.rectangle((left + pad, y, left + pad + swatch, y + swatch), fill=color)
        od.text((left + 2 * pad + swatch, y - rows[i][1]), text, font=font, fill=(255, 255, 255))
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


# ---------------------------------------------------------------- paste area

PASTE_COLORS = {
    "mask": (255, 60, 60),
    "grow": (255, 220, 0),
    "paste": (0, 200, 255),
    "full": (255, 255, 255),
    "cut": (40, 255, 100),
    "seam": (255, 80, 255),
}


PASTE_DIM = 0.3  # brightness left where nothing is pasted


def _outline(field, level, r):
    """field: [h,w]. Pixels within r of the boundary of field >= level."""
    on = (field >= level).float()[None, None]
    return (_grow(on, r) - _grow(on, -r))[0, 0] > 0


def _render_paste_area(bg, alpha, contours, quads, lines):
    """bg: [h,w,3] final image in display px; alpha: [h,w] paste strength, shown as
    brightness: unpasted pixels are darkened, fully pasted ones are left as they are.
    contours: (color, field [h,w], level). quads: (color, points, width_factor).
    The legend goes in a strip under the image, so it never covers the paste."""
    h, w = alpha.shape
    lw = max(2, round(max(h, w) / 500))
    rgb = bg * (PASTE_DIM + (1 - PASTE_DIM) * alpha[..., None])
    for color, field, level in contours:
        ring = _outline(field, level, max(1, lw // 2))
        rgb[ring] = torch.tensor(color, dtype=rgb.dtype, device=rgb.device) / 255
    img = Image.fromarray((rgb.clamp(0, 1) * 255).round().byte().cpu().numpy())
    draw = ImageDraw.Draw(img)
    for color, pts, factor in quads:
        draw.line(pts + [pts[0]], fill=color, width=max(1, round(lw * factor)), joint="curve")

    size = max(12, round(max(h, w) / 60))
    font = _font(size)
    while _legend_size(lines, font)[0] > w and size > 9:
        size -= 1
        font = _font(size)
    box_h = _legend_size(lines, font)[1]
    canvas = Image.new("RGB", (w, h + box_h), (0, 0, 0))
    canvas.paste(img, (0, 0))
    canvas = _draw_legend(canvas, lines, font, 0, h)
    return torch.from_numpy(np.asarray(canvas).astype(np.float32) / 255)


# ---------------------------------------------------------------- cut

def cut(image, mask, target_size=1024, context_expand=1.8, max_upscale=6.0, rotation_deg=0.0,
        size_multiple=16, nearest_ratio=True, mask_threshold=0.1, min_target_size=512,
        ignore_small_spots=True, small_spot_share=0.05):
    """The cut, its mask, the warp_info for `paste` and the cut area preview."""
    img = image.float()
    B, H, W, C = img.shape
    dev = img.device
    m = mask.float().to(dev)
    if m.dim() == 2:
        m = m[None]
    if tuple(m.shape[-2:]) != (H, W):
        m = F.interpolate(m[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    if m.shape[0] == 1 and B > 1:
        m = m.expand(B, -1, -1)
    if m.shape[0] != B:
        raise ValueError(f"mask batch {m.shape[0]} != image batch {B}")

    # Removed spots leave the cut mask too, so they are neither framed nor pasted.
    ignored = [(None, [])] * B
    if ignore_small_spots and small_spot_share > 0:
        kept = []
        for b in range(B):
            mb, drop, spot_boxes = _drop_small_spots(m[b], mask_threshold, small_spot_share)
            kept.append(mb)
            ignored[b] = (drop, spot_boxes)
        m = torch.stack(kept)

    cxs, cys, bws, bhs, boxes = [], [], [], [], []
    for b in range(B):
        box = _mask_bbox(m[b], mask_threshold)
        boxes.append(box)
        x0, y0, x1, y1 = box or (0, 0, W, H)
        cxs.append((x0 + x1) / 2)
        cys.append((y0 + y1) / 2)
        bws.append(x1 - x0)
        bhs.append(y1 - y0)

    # One shared cut size for the batch so the sampler sees a uniform tensor.
    # A rotated region isn't aligned with the image, so only its position is fitted.
    rotated = rotation_deg % 360 != 0
    max_w, max_h = (math.inf, math.inf) if rotated else (W, H)
    rw = min(max(bws) * context_expand, max_w)
    rh = min(max(bhs) * context_expand, max_h)
    ctx_w, ctx_h = rw, rh
    ctx_capped = rw < max(bws) * context_expand or rh < max(bhs) * context_expand
    if nearest_ratio:
        rw, rh = _expand_to_ratio(rw, rh, max_w, max_h)
    elif max(rw, rh) / min(rw, rh) > 2:
        logging.warning("TPS Cut: region ratio %.2f:1 is over 2:1; models may handle it "
                        "poorly. Turn on nearest_ratio to use a standard ratio.",
                        max(rw, rh) / min(rw, rh))

    def rnd(v):
        return max(size_multiple, int(round(v / size_multiple)) * size_multiple)

    def rnd_down(v):
        return max(size_multiple, int(v / size_multiple) * size_multiple)

    def cut_size(long_side):
        if rw >= rh:
            return rnd(long_side), rnd(long_side * rh / rw)
        return rnd(long_side * rw / rh), rnd(long_side)

    cw, ch = cut_size(target_size)
    s = max(rw / cw, rh / ch)  # source px per cut px
    upscale_limited = s < 1.0 / max_upscale
    if upscale_limited:
        # Over the limit the cut shrinks to keep this region. At the floor the
        # region grows instead, so the sampler never gets too small a latent.
        s = 1.0 / max_upscale
        cw, ch = cut_size(max(max(rw, rh) / s, min(min_target_size, target_size)))
        if not rotated:  # the extra context can't reach past the image
            cw, ch = min(cw, rnd_down(W / s)), min(ch, rnd_down(H / s))
        s = max(s, rw / cw, rh / ch)
    th = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(th), math.sin(th)
    R = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=torch.float32, device=dev)
    A = s * R
    c_cut = torch.tensor([cw / 2, ch / 2], dtype=torch.float32, device=dev)
    half_w = s * (abs(cos_t) * cw + abs(sin_t) * ch) / 2
    half_h = s * (abs(sin_t) * cw + abs(cos_t) * ch) / 2
    # Unrotated cuts stay inside the image, so the mask can read its edge pixels.
    mask_padding = "zeros" if rotated else "border"

    cuts, cmasks, ts, centers = [], [], [], []
    for b in range(B):
        centers.append((_fit_center(cxs[b], half_w, W), _fit_center(cys[b], half_h, H)))
        center = torch.tensor(centers[b], dtype=torch.float32, device=dev)
        t = center - A @ c_cut
        src = img[b:b + 1].permute(0, 3, 1, 2)
        cut = warp_sample(src, A, t, ch, cw, "bicubic", "border", s).clamp(0, 1)
        cm = warp_sample(m[b:b + 1, None], A, t, ch, cw, "bilinear", mask_padding, s,
                         "gauss").clamp(0, 1)
        cuts.append(cut.permute(0, 2, 3, 1))
        cmasks.append(cm[:, 0])
        ts.append(t)

    info = {
        "A": A, "t": torch.stack(ts), "scale": s,
        "cut_w": cw, "cut_h": ch, "H": H, "W": W,
        "cut_mask": torch.cat(cmasks),
    }
    steps = {
        "boxes": boxes, "cxs": cxs, "cys": cys, "centers": centers,
        "context": (ctx_w, ctx_h, ctx_capped), "ratio": (rw, rh),
        "upscale_limited": upscale_limited, "ignored": ignored,
        "small_spot_share": small_spot_share,
    }
    preview = _cut_preview(img, m, A, ts, cw, ch, s, steps, context_expand, nearest_ratio,
                            max_upscale, min_target_size, rotation_deg, mask_threshold)
    return (torch.cat(cuts).to(image.dtype), torch.cat(cmasks), info, preview.to(image.dtype))


def _cut_preview(img, m, A, ts, cw, ch, s, steps, context_expand, nearest_ratio,
             max_upscale, min_target_size, rotation_deg, mask_threshold):
    """Draw how the mask became the cut area, one image per batch item."""
    th = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(th), math.sin(th)
    ctx_w, ctx_h, ctx_capped = steps["context"]
    rw, rh = steps["ratio"]
    reg_w, reg_h = cw * s, ch * s
    c = PREVIEW_COLORS
    size = lambda w, h: f"{w:.0f}x{h:.0f}"
    out = []
    for b in range(img.shape[0]):
        box, cx, cy = steps["boxes"][b], steps["cxs"][b], steps["cys"][b]
        fx, fy = steps["centers"][b]
        shapes, lines = [], []  # shapes are listed inside out, drawn outside in
        drop, spot_boxes = steps["ignored"][b]
        if box is not None:
            lines.append((c["mask"], f"mask (above {mask_threshold})"))
        if spot_boxes:
            pad = max(img.shape[1], img.shape[2]) * 0.006
            for sx0, sy0, sx1, sy1 in spot_boxes[:100]:
                sx0, sy0, sx1, sy1 = sx0 - pad, sy0 - pad, sx1 + pad, sy1 + pad
                shapes.append((c["ignored"], [(sx0, sy0), (sx1, sy0), (sx1, sy1), (sx0, sy1)], 1))
            n = len(spot_boxes)
            lines.append((c["ignored"], f"ignored {n} spot{'s' if n > 1 else ''} under "
                                        f"{steps['small_spot_share']:.0%} of the mask area"))
        if box is None:
            lines.append((c["bbox"], f"no mask above {mask_threshold}: using the whole image"))
        else:
            x0, y0, x1, y1 = box
            shapes.append((c["bbox"], [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], 1))
            lines.append((c["bbox"], f"1. mask bbox {size(x1 - x0, y1 - y0)}"))
        note = ", capped at the image" if ctx_capped else ""
        shapes.append((c["context"], _box_quad(cx, cy, ctx_w, ctx_h, cos_t, sin_t), 1))
        lines.append((c["context"], f"2. x {context_expand:g} context_expand: {size(ctx_w, ctx_h)}{note}"))
        if not nearest_ratio:
            lines.append((None, f"3. nearest_ratio off: stays {_ratio_label(rw, rh)}"))
        elif rw < ctx_w * 1.01 and rh < ctx_h * 1.01:  # a rounding nudge isn't a step
            label = _ratio_label(rw, rh)
            why = "already a listed ratio" if ":" in label and "." not in label else "no listed ratio fits"
            lines.append((None, f"3. nearest_ratio: stays {label} ({why})"))
        else:
            shapes.append((c["ratio"], _box_quad(cx, cy, rw, rh, cos_t, sin_t), 1.8))
            lines.append((c["ratio"], f"3. nearest_ratio {_ratio_label(rw, rh)}: {size(rw, rh)}"))
        quad = (_corners(cw, ch, A.device, A.dtype) @ A.T + ts[b])[[0, 1, 3, 2]].tolist()
        shapes.append((c["cut"], [tuple(p) for p in quad], 2.6))
        zoom = f"{1 / s:.2f}x up" if s < 1 else f"{s:.2f}x down"
        lines.append((c["cut"], f"4. cut area {size(reg_w, reg_h)} -> {cw}x{ch} px ({zoom})"))
        if steps["upscale_limited"]:
            if reg_w > rw * 1.02 or reg_h > rh * 1.02:
                lines.append((None, f"   max_upscale {max_upscale:g} hit; area grew to keep "
                                    f"min_target_size {min_target_size}"))
            else:
                lines.append((None, f"   max_upscale {max_upscale:g} hit; cut shrank, area kept"))
        # Each box is thinner than the one drawn before it, so a step that
        # matches the next one shows as a stripe inside its line.
        shapes.reverse()
        if abs(fx - cx) > 0.5 or abs(fy - cy) > 0.5:
            shapes.append((c["cut"], [(fx, fy)], 1))
            shapes.append((c["bbox"], [(cx, cy)], 1))
            shapes.append((c["cut"], [(cx, cy), (fx, fy)], 1))
            lines.append((None, "   slid to stay inside the image"))
        if rotation_deg % 360 != 0:
            lines.append((None, f"   rotated {rotation_deg:g} deg"))
        out.append(_render_preview(img[b], m[b], shapes, lines, (fx, fy), drop))
    return torch.stack(out)


# ---------------------------------------------------------------- paste

def _cut_mask(m, ch, cw, dev):
    """A cut-space mask ([N,H,W] or [H,W]) as [N,1,ch,cw]."""
    m = m.float().to(dev)
    if m.dim() == 2:
        m = m[None]
    m = m[:, None]
    if tuple(m.shape[-2:]) != (ch, cw):
        m = F.interpolate(m, size=(ch, cw), mode="bilinear", align_corners=False)
    return m


def _blend_alpha(info, blend_mask, blend, grow, feather, edge_feather, open_sides, dev):
    """The blend weight in cut space, [B,1,ch,cw], and the (start, grown) masks
    for the paste area (None for full_cut)."""
    cw, ch = info["cut_w"], info["cut_h"]
    stages = None
    if blend == "full_cut":
        bm = torch.ones(1, 1, ch, cw, device=dev)
    else:
        bm = _cut_mask(blend_mask if blend_mask is not None else info["cut_mask"], ch, cw, dev)
        grown = _grow(bm, grow)
        stages = (bm, grown)
        bm = gaussian_blur(grown, feather / 3.0)
    bm = torch.cat([bm[min(b, bm.shape[0] - 1)][None]
                    * _edge_ramp(ch, cw, edge_feather, dev, torch.float32, sides)
                    for b, sides in enumerate(open_sides)]).clamp(0, 1)
    return bm, stages


def _processed_at_cut(processed, n_ch, ch, cw, dev):
    proc = processed[..., :n_ch].float().to(dev).permute(0, 3, 1, 2)
    if tuple(proc.shape[-2:]) != (ch, cw):
        proc = F.interpolate(proc, size=(ch, cw), mode="bicubic",
                             antialias=True, align_corners=False).clamp(0, 1)
    return proc


def paste(image, processed, warp_info, blend="mask", grow=8, feather=24, edge_feather=32,
          align="seam", align_strength=1.0, seam_band=48, blend_mask=None, flows=None,
          feather_detail="plain"):
    """The source with the processed cut pasted back: the image, the blend mask in cut
    space, the cut as pasted (after align), the paste area preview and the paste mask.

    `flows(a, b)` gives forward and backward optical flow between two float [0, 1]
    frames (see `seam_align`); it is needed when `align` is not "none".

    feather_detail "keep" keeps the detail energy of both sides in the feather
    (`color_match.energy_blend`); "plain" here, "keep" in the node. Color and
    texture are matched before, by `color_match` (the TPS Color Match node)."""
    info = warp_info
    img = image.float()
    B, H, W, C = img.shape
    dev = img.device
    cw, ch, s = info["cut_w"], info["cut_h"], info["scale"]
    A = info["A"].to(dev)
    ts = info["t"].to(dev)
    if ts.shape[0] != B:
        raise ValueError(f"warp_info batch {ts.shape[0]} != image batch {B}")

    # Channel counts can differ: Qwen Image 2.1's VAE decodes RGBA even from RGB.
    # Only channels both sides have are blended; the source keeps the rest.
    n_ch = min(C, processed.shape[-1])
    proc = _processed_at_cut(processed, n_ch, ch, cw, dev)
    open_sides = [_sides_on_image_border(A, ts[b], cw, ch, W, H) for b in range(B)]
    bm, stages = _blend_alpha(info, blend_mask, blend, grow, feather, edge_feather, open_sides, dev)

    A_inv = torch.linalg.inv(A)
    out = img.clone()
    paste_mask = torch.zeros(B, H, W, device=dev)
    aligned, did_align, rects = [], [], []
    for b in range(B):
        t = ts[b]
        p_src = proc[min(b, proc.shape[0] - 1)][None]
        aligned_from = p_src
        m_src = bm[b:b + 1]
        if align != "none" and align_strength > 0:
            original = warp_sample(img[b:b + 1].permute(0, 3, 1, 2), A, t, ch, cw,
                                   "bicubic", "border", s).clamp(0, 1)
            p_src = _align_cut(original, p_src, m_src, align, align_strength, seam_band, flows)
        did_align.append(p_src is not aligned_from)  # _align_cut returns its input on failure
        aligned.append(p_src)
        pc = _corners(cw, ch, dev, torch.float32) @ A.T + t
        rx0 = max(0, int(math.floor(pc[:, 0].min().item())) - 1)
        ry0 = max(0, int(math.floor(pc[:, 1].min().item())) - 1)
        rx1 = min(W, int(math.ceil(pc[:, 0].max().item())) + 1)
        ry1 = min(H, int(math.ceil(pc[:, 1].max().item())) + 1)
        rects.append((rx0, ry0, rx1, ry1))
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        rw, rh = rx1 - rx0, ry1 - ry0
        r0 = torch.tensor([rx0, ry0], dtype=torch.float32, device=dev)
        t_loc = A_inv @ (r0 - t)

        # On an open side the blend stays above 0 up to the last pixel, so padding
        # with zeros there would darken the image edge.
        alpha_padding = "border" if open_sides[b] else "zeros"
        # The paste shrinks the cut by 1/s.
        p = warp_sample(p_src, A_inv, t_loc, rh, rw, "bicubic", "border", 1 / s).clamp(0, 1)
        a = warp_sample(m_src, A_inv, t_loc, rh, rw, "bilinear", alpha_padding, 1 / s,
                        "gauss").clamp(0, 1)
        p = p[0].permute(1, 2, 0)
        a = a[0].permute(1, 2, 0)
        region = out[b, ry0:ry1, rx0:rx1, :n_ch]
        mixed = region * (1 - a) + p * a
        if feather_detail == "keep" and n_ch >= 3:
            kept = color_match.energy_blend(region[..., :3].float().cpu().numpy(),
                                            p[..., :3].float().cpu().numpy(),
                                            a[..., 0].float().cpu().numpy())
            mixed = torch.cat([torch.from_numpy(kept).to(dev), mixed[..., 3:]], -1)
        out[b, ry0:ry1, rx0:rx1, :n_ch] = mixed
        paste_mask[b, ry0:ry1, rx0:rx1] = a[..., 0]

    aligned_cut = torch.cat(aligned).permute(0, 2, 3, 1)
    notes = [f"feather detail {feather_detail}"]
    area = _paste_area(out, A, ts, cw, ch, s, rects, bm, stages, open_sides, did_align,
                       blend, blend_mask is not None, grow, feather, edge_feather,
                       align, align_strength, seam_band, notes)
    return (out.to(image.dtype), bm[:, 0], aligned_cut.to(image.dtype),
            area.to(image.dtype), paste_mask)


def _given_at_cut(change_mask, info, b, H, W, dev):
    """A given change mask for batch item b as a (ch, cw) numpy array. A mask the
    size of the source (painted on it, or on the pasted result) is taken through
    the cut's warp; one the size of the cut is used as it is."""
    cw, ch, s = info["cut_w"], info["cut_h"], info["scale"]
    m = change_mask.float().to(dev)
    if m.dim() == 2:
        m = m[None]
    m = m[min(b, m.shape[0] - 1)]
    if tuple(m.shape) == (H, W) and (H, W) != (ch, cw):
        m = warp_sample(m[None, None], info["A"].to(dev), info["t"][b].to(dev), ch, cw,
                        "bilinear", "zeros", s, "gauss")[0, 0]
    elif tuple(m.shape) != (ch, cw):
        if abs(m.shape[1] / m.shape[0] - cw / ch) > 0.01 * cw / ch:
            raise ValueError(f"change_mask is {m.shape[1]}x{m.shape[0]}: it must be the size "
                             f"of the source ({W}x{H}) or of the cut ({cw}x{ch})")
        m = F.interpolate(m[None, None], size=(ch, cw), mode="bilinear", align_corners=False)[0, 0]
    return m.clamp(0, 1).cpu().numpy()


def color_match_cut(image, processed, warp_info, change_mask=None, blend_mask=None,
                    flows=None, color="grade + edge", grade_strength=1.0,
                    edge_profile="smooth", edge_width=128, texture="match"):
    """The processed cut with its color and texture matched to the source's, before
    it is pasted: the corrected cut (at cut size, in the processed cut's own
    layout), the change mask in cut space and the measurement view.

    It is measured with the processed cut lined up with the original (a full
    align, by `flows`), on pixels the edit left alone (see `color_match`), and
    applied to the processed cut as it came, so TPS Paste can still align it in
    any mode. change_mask, when given, says what the edit changed (see
    `color_match.trust`). The edge correction is measured at the edge of the
    painted mask (blend_mask, or the cut mask) grown as TPS Paste grows it by
    default."""
    info = warp_info
    img = image.float()
    B, H, W, C = img.shape
    dev = img.device
    cw, ch, s = info["cut_w"], info["cut_h"], info["scale"]
    A = info["A"].to(dev)
    ts = info["t"].to(dev)
    if ts.shape[0] != B:
        raise ValueError(f"warp_info batch {ts.shape[0]} != image batch {B}")
    n_ch = processed.shape[-1]
    proc = _processed_at_cut(processed, n_ch, ch, cw, dev)
    if n_ch < 3 or C < 3 or (color == "off" and texture == "off"):
        return (proc.permute(0, 2, 3, 1).to(processed.dtype), torch.zeros(proc.shape[0], ch, cw),
                _off_view(dev, B).to(processed.dtype))
    open_sides = [_sides_on_image_border(A, ts[b], cw, ch, W, H) for b in range(B)]
    painted = _cut_mask(blend_mask if blend_mask is not None else info["cut_mask"], ch, cw, dev)
    bm, _ = _blend_alpha(info, blend_mask, "mask", PASTE_GROW, PASTE_FEATHER, 0, open_sides, dev)
    to_np = lambda x: x[0, :3].permute(1, 2, 0).float().cpu().numpy().astype(np.float32)

    outs, changes, views = [], [], []
    for b in range(max(B, proc.shape[0])):
        bi = min(b, B - 1)
        p_src = proc[min(b, proc.shape[0] - 1)][None]
        m_src = bm[bi:bi + 1]
        original = warp_sample(img[bi:bi + 1].permute(0, 3, 1, 2), A, ts[bi], ch, cw,
                               "bicubic", "border", s).clamp(0, 1)
        lined = _align_cut(original, p_src, m_src, "full", 1.0, 0, flows) if flows else p_src
        given = None if change_mask is None else _given_at_cut(change_mask, info, bi, H, W, dev)
        o_np, l_np, p_np = to_np(original), to_np(lined), to_np(p_src)
        alpha = m_src[0, 0].cpu().numpy()
        fixed_lined, change, report = color_match.match(
            o_np, l_np, painted[min(bi, painted.shape[0] - 1), 0].cpu().numpy() > 0.5, alpha,
            flows, color, grade_strength, edge_profile, edge_width, texture, given=given)
        views.append(color_match.render_view(o_np, l_np, fixed_lined, alpha, report,
                                             sorted(open_sides[bi])))
        fixed = torch.from_numpy(color_match.apply_correction(p_np, report["correction"]))
        fixed = fixed.to(dev).permute(2, 0, 1)[None]
        outs.append(torch.cat([fixed, p_src[:, 3:]], 1) if n_ch > 3 else fixed)
        changes.append(torch.from_numpy(change))
    out = torch.cat(outs).permute(0, 2, 3, 1)
    return (out.to(processed.dtype), torch.stack(changes).to(dev),
            _stack_views(views, dev).to(processed.dtype))


def _stack_views(views, dev):
    """The measurement views of a batch, padded to one size so they stack."""
    h = max(v.shape[0] for v in views)
    w = max(v.shape[1] for v in views)
    out = torch.full((len(views), h, w, 3), 24 / 255, device=dev)
    for i, v in enumerate(views):
        out[i, :v.shape[0], :v.shape[1]] = torch.from_numpy(v).to(dev)
    return out


def _off_view(dev, B):
    img = Image.new("RGB", (560, 80), (24, 24, 24))
    ImageDraw.Draw(img).text((16, 28), "color and texture match off: nothing measured",
                             fill=(200, 200, 200), font=_font(16))
    v = torch.from_numpy(np.asarray(img, np.float32) / 255.0).to(dev)
    return v[None].expand(B, -1, -1, -1)


def _paste_area(out, A, ts, cw, ch, s, rects, bm, stages, open_sides, did_align, blend,
                has_override, grow, feather, edge_feather, align, align_strength, seam_band,
                notes=()):
    """The final image around each paste, zoomed so cut-px settings are visible,
    with every blend step marked. One image per batch item."""
    B, H, W, C = out.shape
    dev = out.device
    A_inv = torch.linalg.inv(A)
    # One view size for the batch, so the images stack; each is centered on its paste.
    RW = min(W, max(max(r[2] - r[0] for r in rects), 1) * 1.25)
    RH = min(H, max(max(r[3] - r[1] for r in rects), 1) * 1.25)
    k = min(2048, max(1024, RW, RH)) / max(RW, RH)  # display px per source px
    dw, dh = max(1, round(RW * k)), max(1, round(RH * k))
    c = PASTE_COLORS
    results = []
    for b in range(B):
        x0, y0, x1, y1 = rects[b]
        r0 = torch.tensor([_fit_center((x0 + x1) / 2, RW / 2, W) - RW / 2,
                           _fit_center((y0 + y1) / 2, RH / 2, H) - RH / 2],
                          dtype=torch.float32, device=dev)
        rgb = out[b, ..., :3] if C >= 3 else out[b, ..., :1].expand(-1, -1, 3)
        bg = warp_sample(rgb.permute(2, 0, 1)[None], torch.eye(2, device=dev) / k, r0,
                         dh, dw, "bilinear", "border", 1 / k)[0].permute(1, 2, 0).clamp(0, 1)
        Ad, td = A_inv / k, A_inv @ (r0 - ts[b])

        def to_disp(f):  # cut-space [1,1,ch,cw] -> display [dh,dw]
            return warp_sample(f, Ad, td, dh, dw, "bilinear", "zeros", 1 / (s * k), "gauss")[0, 0]

        def to_pts(pts):  # cut-space points -> display points
            p = torch.tensor(pts, dtype=torch.float32, device=dev) @ A.T + ts[b]
            return [tuple(q) for q in ((p - r0) * k).tolist()]

        alpha = to_disp(bm[b:b + 1])
        contours, quads, lines = [], [], []
        if stages is None:
            lines.append((c["full"], "full_cut: the whole cut is pasted"))
            lines.append((None, "grow and feather don't apply"))
        else:
            start, grown = (st[min(b, st.shape[0] - 1)][None] for st in stages)
            contours.append((c["mask"], to_disp(start), 0.5))
            lines.append((c["mask"], "blend_mask input" if has_override else "cut mask"))
            if grow:
                contours.append((c["grow"], to_disp(grown), 0.5))
                lines.append((c["grow"], f"grow {grow:+d} cut px" + (" (erodes)" if grow < 0 else "")))
            else:
                lines.append((None, "grow 0"))
        contours.append((c["paste"], alpha, 0.02))
        contours.append((c["full"], alpha, 0.98))
        feather_txt = f", feather {feather} cut px" if stages is not None else ""
        lines.append((c["paste"], f"brightness = paste strength: cyan 2%, white 98% (fully new){feather_txt}"))

        quads.append((c["cut"], to_pts([(0, 0), (cw, 0), (cw, ch), (0, ch)]), 1.5))
        e, sides = edge_feather, open_sides[b]
        if e > 0 and 2 * e < min(cw, ch):
            ex0, ey0 = (0 if "left" in sides else e), (0 if "top" in sides else e)
            ex1, ey1 = (cw if "right" in sides else cw - e), (ch if "bottom" in sides else ch - e)
            quads.append((c["cut"], to_pts([(ex0, ey0), (ex1, ey0), (ex1, ey1), (ex0, ey1)]), 1))
        open_txt = f"; none on {', '.join(sorted(sides))} (image border)" if sides else ""
        lines.append((c["cut"], f"cut border, edge fade {e} cut px{open_txt}"))

        strength = f" at {align_strength:g}" if align_strength < 1 else ""
        if align == "none" or align_strength <= 0:
            lines.append((None, "align off"))
        elif not did_align[b]:
            lines.append((None, f"align {align}: too few matches, not aligned"))
        elif align == "full":
            lines.append((c["seam"], f"align full{strength}: the whole cut follows the original layout"))
        else:
            core = (bm[b:b + 1] >= 0.99).float()  # as seam_align.seam_weight
            if seam_band > 0:
                core = gaussian_blur(_grow(core, -seam_band), max(1.0, seam_band / 3.0))
            contours.append((c["seam"], to_disp(core), 0.5))
            lines.append((c["seam"], f"align seam{strength}: inside the magenta line stays as "
                                     f"generated (band {seam_band} cut px)"))
        lines.extend((None, n) for n in notes)
        results.append(_render_paste_area(bg, alpha, contours, quads, lines))

    # Legend fonts can shrink per image to fit, so pad to the tallest.
    tallest = max(r.shape[0] for r in results)
    return torch.stack([F.pad(r, (0, 0, 0, 0, 0, tallest - r.shape[0])) for r in results])


def _align_cut(original, processed, alpha, mode, strength, seam_band, flows):
    """original, processed: [1,C,h,w]; alpha: [1,1,h,w]. Returns processed
    resampled so its features meet the original's where the blend fades."""
    if flows is None:
        raise ValueError(f"align {mode!r} needs optical flow; none was given")
    to_np = lambda x: x[0, :3].permute(1, 2, 0).float().cpu().numpy().astype("float32")
    extra = seam_align.displacement(to_np(original), to_np(processed), flows,
                                    alpha[0, 0].float().cpu().numpy(), mode, seam_band)
    if extra is None:
        return processed
    _, _, h, w = processed.shape
    dev = processed.device
    extra = torch.from_numpy(extra).to(dev) * strength
    ys = torch.arange(h, device=dev, dtype=torch.float32)
    xs = torch.arange(w, device=dev, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    sx = gx + extra[..., 0]   # pixel-index coords, as the flow measured them
    sy = gy + extra[..., 1]
    grid = torch.stack([(2 * sx + 1) / w - 1, (2 * sy + 1) / h - 1], dim=-1)[None]
    return F.grid_sample(processed, grid, mode="bicubic", padding_mode="border",
                         align_corners=False).clamp(0, 1)
