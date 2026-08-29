"""Model-specific resolution declarations — see arg_groups/overrides.py."""

from typing import Any

from sglang.srt.arg_groups.overrides import (
    _register_for,
    resolving_view,
)
from sglang.srt.utils.common import is_sm100_supported


@_register_for("GraniteMoeHybridForCausalLM")
def _granite_moe_hybrid_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    has_mamba = any(
        layer_type == "mamba" for layer_type in getattr(hf_config, "layer_types", [])
    )
    if has_mamba and is_sm100_supported() and cfg.attention_backend is None:
        return {"attention_backend": "flashinfer"}
    return {}
