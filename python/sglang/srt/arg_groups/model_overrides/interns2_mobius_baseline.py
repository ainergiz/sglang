"""Model-specific resolution declarations — see arg_groups/overrides.py."""

from typing import Any

from sglang.srt.arg_groups.overrides import (
    _register_for,
    resolving_view,
)


@_register_for("InternS2MobiusForConditionalGeneration")
def _interns2_mobius_baseline_overrides(server_args: Any, hf_config: Any) -> dict:
    """Select the only MoE runner validated for the 2,560-expert baseline."""
    cfg = resolving_view(server_args)
    if cfg.moe_runner_backend == "auto":
        return {"moe_runner_backend": "triton_kernel"}
    return {}
