"""Per-model resolution declarations.

Importing this package registers every family with the `overrides` registry;
`collect_model_override_declarations` imports it before it collects.
Module names mirror `models/`.

The import order is the **registration order**, which is the definition order
these families had in `overrides.py`: an architecture may match several
providers and the later one wins downstream, so alphabetising this list
would change resolution.
"""

from sglang.srt.arg_groups.model_overrides import kimi_k3 as kimi_k3
from sglang.srt.arg_groups.model_overrides import (
    kimi_k3_moe_runner as kimi_k3_moe_runner,
)
from sglang.srt.arg_groups.model_overrides import deepseek_family as deepseek_family
from sglang.srt.arg_groups.model_overrides import mimo_v2 as mimo_v2
from sglang.srt.arg_groups.model_overrides import minimax_m2 as minimax_m2
from sglang.srt.arg_groups.model_overrides import minimax_m3 as minimax_m3
from sglang.srt.arg_groups.model_overrides import gemma2_gemma3 as gemma2_gemma3
from sglang.srt.arg_groups.model_overrides import exaone as exaone
from sglang.srt.arg_groups.model_overrides import gpt_oss as gpt_oss
from sglang.srt.arg_groups.model_overrides import llama4 as llama4
from sglang.srt.arg_groups.model_overrides import gemma4 as gemma4
from sglang.srt.arg_groups.model_overrides import moss_vl as moss_vl
from sglang.srt.arg_groups.model_overrides import minicpm_sala as minicpm_sala
from sglang.srt.arg_groups.model_overrides import minicpm_v4_6 as minicpm_v4_6
from sglang.srt.arg_groups.model_overrides import falcon_h1_jet as falcon_h1_jet
from sglang.srt.arg_groups.model_overrides import (
    granite_moe_hybrid as granite_moe_hybrid,
)
from sglang.srt.arg_groups.model_overrides import lfm2 as lfm2
from sglang.srt.arg_groups.model_overrides import deepseek_v4 as deepseek_v4
from sglang.srt.arg_groups.model_overrides import inkling as inkling
from sglang.srt.arg_groups.model_overrides import nemotron_h as nemotron_h
from sglang.srt.arg_groups.model_overrides import qwen3_5_hybrid as qwen3_5_hybrid
from sglang.srt.arg_groups.model_overrides import (
    interns2_mobius_baseline as interns2_mobius_baseline,
)
from sglang.srt.arg_groups.model_overrides import qwen3vl as qwen3vl
from sglang.srt.arg_groups.model_overrides import qwen3_moe_family as qwen3_moe_family
from sglang.srt.arg_groups.model_overrides import glm4_moe as glm4_moe
from sglang.srt.arg_groups.model_overrides import olmo2 as olmo2
from sglang.srt.arg_groups.model_overrides import step3p as step3p
from sglang.srt.arg_groups.model_overrides import (
    muse_glimmer_fp4_gemm_runner as muse_glimmer_fp4_gemm_runner,
)
