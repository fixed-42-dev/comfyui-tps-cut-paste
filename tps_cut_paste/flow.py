"""RAFT optical flow, loaded once and moved to the GPU only while in use.

torchvision's `raft_large` with the `C_T_SKHT_V2` weights (BSD-3-Clause). They
download on first use into the torch cache; after that they load from it with
no network.
"""
import threading

import numpy as np
import torch


def _devices():
    """Where to run and where to rest, as ComfyUI would have it."""
    try:
        import comfy.model_management as management
        return management.get_torch_device(), management.unet_offload_device()
    except ImportError:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.device("cpu")


class _OnDemand:
    """Loaded on first use, kept on the offload device, lent to the GPU."""

    def __init__(self):
        self._lock = threading.Lock()
        self._loaded = None

    def _load(self):
        raise NotImplementedError

    def _modules(self):
        raise NotImplementedError

    def __enter__(self):
        self._lock.acquire()
        try:
            if self._loaded is None:
                self._loaded = self._load()
            self.device, self._rest = _devices()
            for module in self._modules():
                module.to(self.device)
        except BaseException:
            self._lock.release()
            raise
        return self

    def __exit__(self, *_):
        for module in self._modules():
            module.to(self._rest)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._lock.release()


class Flow(_OnDemand):
    """`flows(first, second)` -> forward and backward flow, with RAFT."""

    def _load(self):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        self.weights = Raft_Large_Weights.C_T_SKHT_V2
        return {"raft": raft_large(weights=self.weights, progress=False).eval()}

    def _modules(self):
        return (self._loaded["raft"],)

    def _tensor(self, rgb):
        # [0, 1] then into [-1, 1], which is what the weights' transforms do.
        # Feeding raw uint8 would be silently wrong rather than an error.
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
        return tensor.sub_(0.5).div_(0.5).unsqueeze(0).to(self.device)

    def __call__(self, first, second):
        with self, torch.no_grad():
            first, second = self._tensor(first), self._tensor(second)
            raft = self._loaded["raft"]
            forward = raft(first, second)[-1][0].cpu().numpy()
            backward = raft(second, first)[-1][0].cpu().numpy()
        return forward, backward
