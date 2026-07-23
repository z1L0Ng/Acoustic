"""Restricted loader for the audited Patch-Mix author checkpoint."""

from __future__ import annotations

import argparse
import codecs
import collections
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


_ALLOWED_GLOBALS = {
    ("argparse", "Namespace"): argparse.Namespace,
    ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy", "ndarray"): np.ndarray,
    ("_codecs", "encode"): codecs.encode,
    ("numpy", "dtype"): np.dtype,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch", "FloatStorage"): torch.FloatStorage,
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        key = (module, name)
        if key not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"forbidden checkpoint global: {module}.{name}")
        return _ALLOWED_GLOBALS[key]


_RESTRICTED_PICKLE = SimpleNamespace(
    __name__="patch_mix_restricted_pickle",
    Unpickler=RestrictedUnpickler,
    load=pickle.load,
    loads=pickle.loads,
    dump=pickle.dump,
    dumps=pickle.dumps,
)


def restricted_torch_load(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        pickle_module=_RESTRICTED_PICKLE,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"expected dict checkpoint, got {type(checkpoint)!r}")
    return checkpoint
