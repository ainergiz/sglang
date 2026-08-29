"""Model-specific resolution declarations — see arg_groups/overrides.py."""

from typing import Any

from sglang.srt.arg_groups.overrides import (
    _register_for,
    logger,
    model_config_of,
    resolving_view,
)
from sglang.srt.utils.common import is_sm100_supported


@_register_for("MiniMaxM2ForCausalLM")
def _minimax_m2_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides = {"enable_tf32_matmul": True}
    logger.info(
        "Enable TF32 matmul for MiniMaxM2ForCausalLM model to improve gate gemm performance."
    )
    if (
        is_sm100_supported()
        and cfg.moe_runner_backend == "auto"
        and model_config_of(server_args).quantization == "modelopt_fp4"
    ):
        overrides["moe_runner_backend"] = "flashinfer_trtllm_routed"
        logger.info(
            "Use flashinfer_trtllm_routed as MoE runner backend on SM10X "
            "for MiniMaxM2ForCausalLM with modelopt_fp4."
        )
    return overrides
