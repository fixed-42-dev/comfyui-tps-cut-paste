"""The ComfyUI nodes: TPS Cut, TPS Color Match and TPS Paste.

TPS Cut takes the masked area out at a size the sampler suits; only the cut is
edited; TPS Color Match undoes the color and texture change the edit model made
where it was not asked to; TPS Paste puts the cut back, first lining it up with
the original by RAFT optical flow and a thin-plate-spline warp, and leaves
everything outside the mask as it was.
"""
from functools import lru_cache

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .tps_cut_paste import cut_paste, seam_align
from .tps_cut_paste.flow import Flow

WarpInfo = io.Custom("TPS_WARP_INFO")
CATEGORY = "TPS Cut Paste"


@lru_cache(maxsize=1)
def _flow():
    return Flow()


class TPSCut(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TPSCut",
            display_name="TPS Cut",
            category=CATEGORY,
            description=(
                "Cuts the masked area out for editing: framed with some context, scaled "
                "to target_size and optionally turned upright, so only the cut goes "
                "through VAE and sampler and VRAM depends on target_size, not on the "
                "image. Edit the cut, then put it back with TPS Paste."),
            search_aliases=["crop", "cut and paste", "detailer", "inpaint crop",
                            "crop and stitch"],
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Int.Input("target_size", default=1024, min=256, max=4096, step=16,
                             tooltip="Long side of the cut sent to the sampler."),
                io.Float.Input("context_expand", default=1.8, min=1.0, max=6.0, step=0.05,
                               tooltip="Region size = mask bbox x this."),
                io.Float.Input("max_upscale", default=6.0, min=1.0, max=32.0, step=0.5,
                               tooltip="Upscale limit. Over it the cut gets smaller, down to "
                                       "min_target_size."),
                io.Float.Input("rotation_deg", default=0.0, min=-180.0, max=180.0, step=0.5,
                               tooltip="Rotate the cut, e.g. to make a tilted face upright."),
                io.Int.Input("size_multiple", default=16, min=8, max=128, step=8),
                io.Boolean.Input("nearest_ratio", default=True,
                                 tooltip="Expand the region to the nearest of 1:1, 3:2, 4:3, "
                                         "16:9 (either way round) or 21:9."),
                io.Float.Input("mask_threshold", default=0.1, min=0.0, max=1.0, step=0.01),
                io.Int.Input("min_target_size", default=512, min=256, max=4096, step=16,
                             tooltip="Smallest long side when max_upscale shrinks the cut. "
                                     "Below it the region takes in more context."),
                io.Boolean.Input("ignore_small_spots", default=True,
                                 tooltip="Drop mask spots smaller than small_spot_share of the "
                                         "total mask area, so leftover specks don't stretch "
                                         "the cut. The largest spot always stays."),
                io.Float.Input("small_spot_share", default=0.05, min=0.0, max=1.0, step=0.01,
                               tooltip="Share of the total mask area under which a spot is "
                                       "ignored."),
            ],
            outputs=[
                io.Image.Output("cut"),
                io.Mask.Output("cut_mask"),
                WarpInfo.Output("warp_info"),
                io.Image.Output("cut_area", tooltip="The source with the mask and each step "
                                                    "from mask to cut area drawn on it."),
            ],
        )

    @classmethod
    def execute(cls, image, mask, target_size, context_expand, max_upscale, rotation_deg,
                size_multiple, nearest_ratio, mask_threshold, min_target_size,
                ignore_small_spots, small_spot_share) -> io.NodeOutput:
        return io.NodeOutput(*cut_paste.cut(
            image, mask, target_size, context_expand, max_upscale, rotation_deg,
            size_multiple, nearest_ratio, mask_threshold, min_target_size,
            ignore_small_spots, small_spot_share))


class TPSPaste(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TPSPaste",
            display_name="TPS Paste",
            category=CATEGORY,
            description=(
                "Puts the edited cut back where TPS Cut took it from, blended through the "
                "mask, with everything outside the mask left as it was. align lines the "
                "edited cut up with the original first (RAFT optical flow and a "
                "thin-plate-spline warp), so the old and new edges don't both show "
                "through the feather. Put TPS Color Match before it to undo the color and "
                "texture change the edit model made everywhere."),
            search_aliases=["stitch", "cut and paste", "detailer", "inpaint stitch",
                            "crop and stitch"],
            inputs=[
                io.Image.Input("image"),
                io.Image.Input("processed"),
                WarpInfo.Input("warp_info"),
                io.Combo.Input("blend", options=["mask", "full_cut"], default="mask"),
                io.Int.Input("grow", default=8, min=-128, max=128,
                             tooltip="Cut px. Negative erodes."),
                io.Int.Input("feather", default=24, min=0, max=256,
                             tooltip="Cut px of mask softening."),
                io.Int.Input("edge_feather", default=32, min=0, max=256,
                             tooltip="Cut px fade at the cut border, hides seams."),
                io.Combo.Input("align", options=["seam", "full", "none"], default="seam",
                               tooltip="RAFT flow + thin-plate-spline warp. seam: line the "
                                       "processed cut up with the original where the blend "
                                       "fades, keep new features inside. full: pull all "
                                       "features back to the original layout."),
                io.Float.Input("align_strength", default=1.0, min=0.0, max=1.0, step=0.05),
                io.Int.Input("seam_band", default=48, min=0, max=512,
                             tooltip="Cut px inside the mask core that also get aligned "
                                     "(seam mode)."),
                io.Combo.Input("feather_detail", options=["keep", "plain"], default="keep",
                               tooltip="keep: in the feather, keep the detail energy of "
                                       "both sides. A plain mix of two different textures "
                                       "partly cancels and leaves a soft halo along the "
                                       "seam."),
                io.Mask.Input("blend_mask", optional=True,
                              tooltip="Cut-space mask overriding the stored one."),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Mask.Output("blend_mask", tooltip="Blend strength in cut space."),
                io.Image.Output("aligned_cut", tooltip="The processed cut as pasted, after "
                                                       "align."),
                io.Image.Output("paste_area", tooltip="The final image around the paste, "
                                                      "marked with the mask, grow, the blend "
                                                      "band, the edge fade and the align area."),
                io.Mask.Output("paste_mask", tooltip="Blend strength applied to each source "
                                                     "pixel."),
            ],
        )

    @classmethod
    def execute(cls, image, processed, warp_info, blend, grow, feather, edge_feather,
                align, align_strength, seam_band, feather_detail="keep",
                blend_mask=None) -> io.NodeOutput:
        flows = seam_align.raft_flows(_flow()) if align != "none" else None
        return io.NodeOutput(*cut_paste.paste(
            image, processed, warp_info, blend, grow, feather, edge_feather, align,
            align_strength, seam_band, blend_mask, flows=flows, feather_detail=feather_detail))


class TPSColorMatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TPSColorMatch",
            display_name="TPS Color Match",
            category=CATEGORY,
            description=(
                "Goes between the edit and TPS Paste. Undoes what the edit model changed "
                "everywhere (an overall grade, smoothed or sharpened detail), measured only "
                "where the edit left the picture alone, so the new object keeps its own "
                "look and the seam disappears. When it takes the wrong area for changed, "
                "give it a change_mask. Bypass it to compare."),
            search_aliases=["color match", "color correction", "grade",
                            "texture match", "seam"],
            inputs=[
                # processed is the first image, so a bypassed node passes the edit on.
                io.Image.Input("processed", tooltip="The edited cut."),
                io.Image.Input("image", tooltip="The source TPS Cut took the cut from."),
                WarpInfo.Input("warp_info"),
                io.Combo.Input("color", options=["grade + edge", "grade", "edge", "off"],
                               default="grade + edge",
                               tooltip="Undo the edit's overall color change. grade: an "
                                       "Oklab correction per tone, measured on pixels the "
                                       "edit left alone, outside-in from the mask. edge: "
                                       "match what is left at the mask edge and fade it "
                                       "inward; it carries the correction when the edit "
                                       "relit the scene unevenly. edge alone keeps the "
                                       "edit's own grade inside."),
                io.Float.Input("grade_strength", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="How much of the measured grade to undo. It is "
                                       "also scaled down when the measurement is unsure "
                                       "(see measure_view)."),
                io.Combo.Input("edge_profile", options=["smooth", "linear", "decay", "entire"],
                               default="smooth",
                               tooltip="How the edge correction fades inward from the mask "
                                       "edge over edge_width. entire: the whole pasted "
                                       "area."),
                io.Int.Input("edge_width", default=128, min=8, max=2048,
                             tooltip="Cut px over which the edge correction fades inward."),
                io.Combo.Input("texture", options=["match", "off"], default="match",
                               tooltip="Match the edit's detail energy per frequency band "
                                       "to the original's (edit models smooth fine texture "
                                       "and boost mid-scale contrast)."),
                io.Combo.Input("mode", options=["edit", "background"], default="edit",
                               tooltip="edit: an edit model's overall change is undone and "
                                       "what it changed is kept. background: for inpainting "
                                       "that redraws the whole mask (high denoise): "
                                       "change_mask is the object, and the redraw around it "
                                       "takes back the original's broad tone at the same "
                                       "place; grade_strength scales it, the edge settings "
                                       "do nothing."),
                io.Mask.Input("change_mask", optional=True,
                              tooltip="What the edit changed, when the one found is wrong; "
                                      "in background mode, the object to keep (required). "
                                      "Painted, or from a segmentation model, the size of "
                                      "the source (painted on it or on the pasted result) "
                                      "or of the cut. In edit mode, light or shadow the "
                                      "change cast past it is still found by color."),
                io.Mask.Input("blend_mask", optional=True,
                              tooltip="The blend_mask given to TPS Paste, if any: the edge "
                                      "correction is measured at its edge."),
            ],
            outputs=[
                io.Image.Output("processed", tooltip="The edited cut, color and texture "
                                                     "matched, at cut size."),
                io.Mask.Output("change_mask", tooltip="What the edit changed, in cut space: "
                                                      "kept out of the measurement."),
                io.Image.Output("measure_view", tooltip="Where color was measured, what the "
                                                        "edit changed, the plateau before and "
                                                        "after, the fitted grade, confidence "
                                                        "and warnings."),
            ],
        )

    @classmethod
    def execute(cls, processed, image, warp_info, color="grade + edge", grade_strength=1.0,
                edge_profile="smooth", edge_width=128, texture="match", mode="edit",
                change_mask=None, blend_mask=None) -> io.NodeOutput:
        flows = (seam_align.raft_flows(_flow())
                 if mode == "edit" and (color != "off" or texture != "off") else None)
        return io.NodeOutput(*cut_paste.color_match_cut(
            image, processed, warp_info, change_mask, blend_mask, flows, color,
            grade_strength, edge_profile, edge_width, texture, mode))


class TPSCutPasteExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TPSCut, TPSColorMatch, TPSPaste]


async def comfy_entrypoint() -> TPSCutPasteExtension:
    return TPSCutPasteExtension()
