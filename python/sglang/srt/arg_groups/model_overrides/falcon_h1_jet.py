"""Model-specific resolution declarations — see arg_groups/overrides.py."""

from typing import Any

from sglang.srt.arg_groups.overrides import (
    _register_for,
    resolving_view,
)
from sglang.srt.utils.common import is_sm100_supported


@_register_for(
    "FalconH1ForCausalLM", "JetNemotronForCausalLM", "JetVLMForConditionalGeneration"
)
def _falcon_h1_jet_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if is_sm100_supported() and cfg.attention_backend is None:
        return {"attention_backend": "triton"}
    return {}
