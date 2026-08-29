"""Model-specific resolution declarations — see arg_groups/overrides.py."""

from typing import Any, Dict

from sglang.srt.arg_groups.overrides import (
    is_attention_backend_not_set,
    logger,
    register_model_override_predicate,
    resolving_view,
)
from sglang.srt.utils.common import is_blackwell_supported, is_sm90_supported


@register_model_override_predicate(
    lambda arch: "Step3p5ForCausalLM" in arch
    or "Step3p7ForConditionalGeneration" in arch
)
def _step3p_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    if is_attention_backend_not_set(cfg):
        if is_blackwell_supported():
            logger.info("Auto-select fa4 attention backend for Step3p7 on Blackwell.")
            overrides["attention_backend"] = "fa4"
        elif is_sm90_supported():
            logger.info("Auto-select fa3 attention backend for Step3p7 on Hopper.")
            overrides["attention_backend"] = "fa3"
    if cfg.speculative_algorithm == "EAGLE":
        logger.info(
            "Enable multi-layer EAGLE speculative decoding for Step3p5ForCausalLM model."
        )
        overrides["enable_multi_layer_eagle"] = True
    if cfg.enable_hierarchical_cache:
        logger.warning(
            "Reset swa_full_tokens_ratio to 1.0 for Step3p5ForCausalLM model with hierarchical cache"
        )
        overrides["swa_full_tokens_ratio"] = 1.0
        logger.warning(
            "Disable hybrid SWA memory for Step3p5ForCausalLM model with hierarchical cache"
        )
        overrides["disable_hybrid_swa_memory"] = True
    return overrides
