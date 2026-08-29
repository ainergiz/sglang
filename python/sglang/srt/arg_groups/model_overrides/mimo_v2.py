"""Model-specific resolution declarations — see arg_groups/overrides.py."""

from typing import Any, Dict

from sglang.srt.arg_groups.overrides import (
    _register_for,
    logger,
    resolving_view,
)
from sglang.srt.utils.common import get_quantization_config, is_sm100_supported


@_register_for("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM")
def _mimo_v2_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    if cfg.speculative_algorithm == "EAGLE":
        logger.info("Enable multi-layer EAGLE speculative decoding for MiMoV2 model.")
        overrides["enable_multi_layer_eagle"] = True

    # On Blackwell "auto" falls through to the triton fused-MoE runner, ~12%
    # slower at bs=1 decode. FP4 checkpoints use flashinfer_mxfp4 instead.
    if (
        is_sm100_supported()
        and cfg.moe_runner_backend == "auto"
        and get_quantization_config(hf_config) == "fp8"
    ):
        overrides["moe_runner_backend"] = "flashinfer_trtllm"
        logger.info("MiMoV2 FP8 on SM100: moe_runner_backend=flashinfer_trtllm.")
    return overrides
