"""Model-specific resolution declarations — see arg_groups/overrides.py."""

from typing import Any, Dict

from sglang.srt.arg_groups.overrides import (
    _register_for,
    logger,
    resolving_view,
)
from sglang.srt.utils.common import is_sm100_supported


@_register_for("Glm4MoeForCausalLM")
def _glm4_moe_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    if is_sm100_supported():
        quantization_config = getattr(hf_config, "quantization_config", None)
        quant_method = (
            quantization_config.get("quant_method")
            if quantization_config is not None
            else None
        )
        quantization = cfg.quantization
        if (
            quantization is None
            and not server_args._quantization_explicitly_unset
            and quant_method is not None
        ):
            overrides["quantization"] = quant_method
            quantization = quant_method
        if (
            quantization in {"modelopt_fp4", None}
            and cfg.moe_a2a_backend == "none"
            and cfg.moe_runner_backend == "auto"
        ):
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on sm100 for Glm4MoeForCausalLM"
            )
    logger.info(
        "Enable TF32 matmul for Glm4MoeForCausalLM model to improve gate gemm performance."
    )
    overrides["enable_tf32_matmul"] = True
    return overrides
