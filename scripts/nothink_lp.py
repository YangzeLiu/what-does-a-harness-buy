"""vLLM custom logits processor: hard-ban the <think> token so a model served with
enable_thinking=false cannot re-open a reasoning block (Qwen's enable_thinking=false is a
soft switch: the chat template ends the prompt with an empty <think></think>, but the model
may still emit <think> later in long agentic contexts -- Claude Code trials showed ~40% of
turns doing so). Applied to every request, all harnesses alike.
Load with:  --logits-processors nothink_lp:BanThinkLogitsProcessor   (PYTHONPATH must include this dir)
Token ids default to Qwen3.6/3.8 <think>=248068; the host-b launch sets NOTHINK_BAN_IDS=248068,248069 to ban
</think> as well: with only <think> banned the model still wrote its thought as text and closed it with
</think>, which vLLM's streaming qwen3 reasoning parser then re-labelled as reasoning (CC trial: 57 blocks, 22.9k chars).
"""
import os
import torch
from vllm.v1.sample.logits_processor import LogitsProcessor


class BanThinkLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        ids = [int(x) for x in os.environ.get("NOTHINK_BAN_IDS", "248068").split(",") if x.strip()]
        self.ban_ids = torch.tensor(ids, dtype=torch.long, device=device)

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update) -> None:
        return None

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        logits[:, self.ban_ids] = float("-inf")
        return logits
