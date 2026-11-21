"""TPS Cut Paste: edit a masked area only, and paste it back aligned by RAFT
optical flow and a thin-plate-spline warp."""
# Loaded by ComfyUI as a package. Outside one - pytest collecting the repository
# root - there are no nodes to register, and the tests import `tps_cut_paste`.
if __package__:
    from .nodes import comfy_entrypoint

    __all__ = ["comfy_entrypoint"]
