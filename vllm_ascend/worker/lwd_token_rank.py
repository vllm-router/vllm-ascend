# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD 云侧的唯一排名编码：分数降序，同分时按全局 token ID 升序。"""

import torch


class LwdTokenRankEncoder:
    """通过比较和归约确定排名，无需在云侧对完整词表排序。"""

    def __init__(self):
        self.vocab_ids = None

    def encode(self, logits: torch.Tensor, selected_ids: torch.Tensor) -> torch.Tensor:
        # 输入列必须按全局 token ID 升序排列；边侧用稳定降序排序恢复。
        # 缓存词表索引，避免每个解码步重新分配并生成 arange。
        vocab_size = logits.shape[-1]
        if self.vocab_ids is None or self.vocab_ids.numel() != vocab_size or self.vocab_ids.device != logits.device:
            self.vocab_ids = torch.arange(vocab_size, device=logits.device, dtype=torch.int64)
        selected_ids = selected_ids.long()
        # 投机拒绝后的 -1 行不会交付，但仍占传输段的位置。先用合法索引
        # 取分数，最后将这些行的排名置零，避免 gather(-1) 越界。
        valid = selected_ids != -1
        safe_ids = selected_ids.masked_fill(~valid, 0)
        selected_logits = logits.gather(1, safe_ids.unsqueeze(1))
        # 唯一排名 = 更高分的候选数 + 同分且 ID 更小的候选数。
        # 两类候选互斥，合并布尔掩码后仅归约一次；不改写 logits 或采样结果。
        before = (logits > selected_logits) | ((logits == selected_logits) & (self.vocab_ids < safe_ids.unsqueeze(1)))
        ranks = before.sum(dim=1, dtype=torch.int32)
        return ranks.masked_fill(~valid, 0)
