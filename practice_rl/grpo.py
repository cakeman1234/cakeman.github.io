from typing import List

import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class TokenStep:
    token_id : int
    token_text : str
    log_prob : float
    position : int

@dataclass
class Trajectory:
    query : str
    token_steps : List[TokenStep]
    generated_text : str
    reward : float
    final_answer: str
    full_input_ids: List[str]   # full input : prompt + generated + information
    generated_positions: List[int]  # positions of generated tokens in the full input


class GRPOLoss(nn.Module):
    def __init__(self,
                 ref_model: nn.Module,
                 old_log_probs: List[torch.Tensor],
                 groupe_size: int = 4,
                 clip_eps: float = 0.2,
                 kl_coef: float = 0.01,
                 adv_eps: float = 1e-8,
                 ):
        super().__init__()
        self.ref_model = ref_model
        self.old_log_probs = old_log_probs
        self.groupe_size = groupe_size
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.adv_eps = adv_eps

   
    def compute_kl_divergence(self, new_lp: torch.Tensor, ref_lp: torch.Tensor) -> torch.Tensor:
        delta = ref_lp - new_lp
        kl = torch.exp(delta) - delta - 1
        return kl

    def forward(self, 
                new_logprobs: List[torch.Tensor], 
                old_logprobs: List[torch.Tensor], 
                advantages: torch.Tensor, 
                ref_logprobs: List[torch.Tensor]) -> torch.Tensor:
        all_loss = []
        
        # 遍历组内的每个回答
        for i in range(len(new_logprobs)):
            new_lp = new_logprobs[i]
            old_lp = old_logprobs[i]
            ref_lp = ref_logprobs[i]
            adv = advantages[i]

            if (len(new_lp) != len(old_lp)):
                continue
            
            # compute ratio
            ratio = torch.exp(new_lp - old_lp)

            # Ai -> {Ai, Ai, Ai, Ai} for each token in the answer
            token_adv = adv.expand_as(ratio)

            # 计算 PPO 裁剪项
            surr1 = ratio * token_adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * token_adv

            # compute kl penalty
            kl_loss = self.compute_kl_divergence(new_lp, ref_lp)

            # 计算最终的损失，包含 PPO 损失和 KL 散度惩罚
            token_loss = -torch.min(surr1, surr2) + self.kl_coef * kl_loss

            # 对单条回答内部求平均
            all_loss.append(token_loss.mean())

        # 对组内多条回答求平均
        return torch.stack(all_loss).mean()
    

def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
    
    if len(rewards_tensor) == 1:
        return torch.zeros_like(rewards_tensor)

    mean_reward = torch.mean(rewards_tensor)
    std_reward = torch.std(rewards_tensor, unbiased=False) + 1e-8
    advantages = (rewards_tensor - mean_reward) / std_reward
    return advantages
