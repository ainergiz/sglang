"""Model-specific resolution declarations — see arg_groups/overrides.py."""

from typing import Any, Dict

from sglang.srt.arg_groups.overrides import (
    _register_for,
    logger,
    resolving_view,
)
from sglang.srt.utils.common import get_quantization_config, is_sm100_supported


@_register_for(
    "Qwen3MoeForCausalLM",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3NextForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "InternS2PreviewForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
)
def _qwen3_moe_family_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    if is_sm100_supported():
        quant_method = get_quantization_config(hf_config)
        quantization = cfg.quantization
        if (
            quantization is None
            and not server_args._quantization_explicitly_unset
            and quant_method is not None
        ):
            overrides["quantization"] = quant_method
            quantization = quant_method
        if (
            (quantization in ("fp8", "modelopt_fp4") or quantization is None)
            and cfg.moe_a2a_backend == "none"
            and cfg.moe_runner_backend == "auto"
        ):
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on sm100 for "
                f"{hf_config.architectures[0]}"
            )
    return overrides
