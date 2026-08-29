"""Model-specific resolution declarations — see arg_groups/overrides.py."""

from typing import Any

from sglang.srt.arg_groups.overrides import (
    _register_for,
    logger,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.utils.common import is_hip


@_register_for("Qwen3VLForConditionalGeneration")
def _qwen3vl_overrides(server_args: Any, hf_config: Any) -> dict:

    cfg = resolving_view(server_args)
    if is_hip() and envs.SGLANG_USE_AITER_UNIFIED_ATTN.get() and cfg.page_size is None:
        logger.info(
            "Setting page_size=16 for aiter unified attention on Qwen3VLForConditionalGeneration."
        )
        return {"page_size": 16}
    return {}
