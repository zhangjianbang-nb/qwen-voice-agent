"""Lazy, cached loaders for the on-box models.

This deployment runs perception *locally*: ASR is FunASR's SenseVoice and the
speaker embedding is FunASR's CAM++.  Both are strictly better than reaching a
server here — they are faster than an HTTP round trip (SenseVoice does 3.7 s of
audio in 0.10 s on this CPU, CAM++ in 0.04 s) and they cannot fail because the
network did.

Both are heavy (they pull in torch, ~1 GB resident), so two rules apply:

* **nothing is imported at package-import time** — ``funasr`` is imported inside
  the loader, so ``import qwen_voice_agent`` still costs only numpy (acceptance
  criterion A1);
* **a loaded model is cached by directory**, so the cost is paid once per process
  rather than once per utterance.  Reloading a 900 MB checkpoint per turn would
  dominate the turn latency.

ModelScope stores a downloaded model as
``<cache>/models/<org>--<name>/snapshots/<rev>``, which is what
:func:`resolve_model_dir` walks.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

__all__ = [
    "modelscope_cache",
    "resolve_model_dir",
    "sensevoice_dir",
    "campplus_dir",
    "load_funasr",
    "loaded_models",
]

_CACHE_ENV = "MODELSCOPE_CACHE"
_DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "modelscope")

# The two checkpoints this deployment uses, by their ModelScope folder name.
SENSEVOICE = "iic--SenseVoiceSmall"
CAMPLUS = "iic--speech_campplus_sv_zh-cn_16k-common"

_LOADED: Dict[str, Any] = {}


def modelscope_cache() -> str:
    """Root of the ModelScope cache, honouring ``MODELSCOPE_CACHE``."""
    return os.environ.get(_CACHE_ENV) or _DEFAULT_CACHE


def resolve_model_dir(name: str) -> Optional[str]:
    """Return the newest snapshot directory of a downloaded model, or ``None``."""
    snapshots = os.path.join(modelscope_cache(), "models", name, "snapshots")
    if not os.path.isdir(snapshots):
        return None
    for rev in sorted(os.listdir(snapshots), reverse=True):
        path = os.path.join(snapshots, rev)
        if os.path.isdir(path):
            return path
    return None


def sensevoice_dir() -> Optional[str]:
    return resolve_model_dir(SENSEVOICE)


def campplus_dir() -> Optional[str]:
    return resolve_model_dir(CAMPLUS)


def load_funasr(model_dir: str, *, device: str = "cpu", **kwargs) -> Any:
    """Load (and cache) a FunASR ``AutoModel`` from a local directory.

    The cache key is ``(model_dir, device)`` so a second caller — the voiceprint
    embedder and the ASR backend both go through here — reuses the first load.

    The directory is checked *before* FunASR is handed it: given a name it does not
    find locally, FunASR treats it as a hub id and silently starts a network
    download, so a typo'd path turns into a slow, confusing 404 instead of a local
    error.
    """
    if not os.path.isdir(model_dir):
        raise ValueError(f"no such local model directory: {model_dir}")
    key = f"{model_dir}|{device}"
    cached = _LOADED.get(key)
    if cached is not None:
        return cached
    try:
        from funasr import AutoModel  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "local FunASR models need the 'funasr' package; install the 'local' extra"
        ) from exc
    model = AutoModel(
        model=model_dir,
        device=device,
        disable_update=True,
        disable_pbar=True,
        **kwargs,
    )
    _LOADED[key] = model
    return model


def loaded_models() -> Dict[str, Any]:
    """The models currently resident (diagnostics for ``doctor``)."""
    return dict(_LOADED)
