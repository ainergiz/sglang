"""Model-specific resolution declarations — see arg_groups/overrides.py."""

from typing import Any, Dict

from sglang.srt.arg_groups.overrides import (
    _register_for,
    logger,
    resolving_view,
)
from sglang.srt.utils.common import is_hip, is_sm90_supported, is_sm100_supported


@_register_for("Llama4ForConditionalGeneration", "Llama4ForCausalLM")
def _llama4_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if cfg.device == "cpu":
        return {}
    overrides: Dict[str, Any] = {}
    # Auto-select attention backend for Llama4 if not specified
    if cfg.attention_backend is None:
        if is_sm100_supported():
            backend, platform = "trtllm_mha", "sm100"
        elif is_sm90_supported():
            backend, platform = "fa3", "sm90"
        elif is_hip():
            backend, platform = "aiter", "hip"
        elif cfg.device == "xpu":
            backend, platform = "intel_xpu", "xpu"
        else:
            backend, platform = "triton", "other platforms"
        logger.warning(
            f"Use {backend} as attention backend on {platform} for Llama4 model"
        )
        overrides["attention_backend"] = backend
    if is_sm100_supported() and cfg.moe_runner_backend == "auto":
        if cfg.quantization in {"fp8", "modelopt_fp8"}:
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on SM100 for Llama4"
            )
    return overrides
