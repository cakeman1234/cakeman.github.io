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
    full_input_ids: List[int]   # full input : prompt + generated + information
    generated_positions: List[int]  # positions of generated tokens in the full input


class GRPOLoss(nn.Module):
    def __init__(self,
                 group_size: int = 4,
                 clip_eps: float = 0.2,
                 kl_coef: float = 0.01,
                 adv_eps: float = 1e-8,
                 ):
        super().__init__()
        self.group_size = group_size
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

        if len(new_logprobs) != len(old_logprobs) or len(new_logprobs) != len(ref_logprobs):
                raise ValueError(...)
        if len(advantages) != len(new_logprobs):
            raise ValueError(...)
        
        # 遍历组内的每个回答
        for i in range(len(new_logprobs)):
            
            
            new_lp = new_logprobs[i].reshape(-1)
            old_lp = old_logprobs[i].reshape(-1)
            ref_lp = ref_logprobs[i].reshape(-1)
            adv = advantages[i]

            if (len(new_lp) != len(old_lp) or len(new_lp) != len(ref_lp)):
                raise ValueError(f"Length of new_logprobs, old_logprobs, and ref_logprobs must be the same for each answer. Got {len(new_lp)}, {len(old_lp)}, and {len(ref_lp)}.")
            
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
    
class GRPOTrainer():
    def __init__(self,
                policy_model: nn.Module,
                ref_model: nn.Module,
                optimizer,
                loss_fn: GRPOLoss,
                group_size: int = 4,
                device: str = "cpu",
                adv_eps: float = 1e-8):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.group_size = group_size
        self.device = device
        self.adv_eps = adv_eps

    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        if len(rewards_tensor) == 0:
            raise ValueError("rewards cannot be empty.")

        if len(rewards_tensor) % self.group_size != 0:
            raise ValueError(
                f"Number of rewards ({len(rewards_tensor)}) must be divisible by group_size ({self.group_size})."
            )

        grouped_rewards = rewards_tensor.view(-1, self.group_size)  # shape: (num_groups, group_size)

        group_mean = grouped_rewards.mean(dim=1, keepdim=True)      # shape: (num_groups, 1)
        group_std = grouped_rewards.std(dim=1, unbiased=False, keepdim=True)  # shape: (num_groups, 1)

        grouped_advantages = (grouped_rewards - group_mean) / (group_std + self.adv_eps)
        advantages = grouped_advantages.reshape(-1)  # shape: (num_groups * group_size,)

        return advantages
    
    def compute_logprobs(self, model, trajectories: List[Trajectory]) :
        all_logprobs = []

        for traj in trajectories:
            # 把input_ids转换成tensor，shape为(1, seq_len)
            input_ids = torch.tensor(traj.full_input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            
            outputs = model(input_ids)
            logits = outputs.logits # shape: (1, seq_len, vocab_size)
            logits = logits.squeeze(0) # shape: (seq_len, vocab_size)

            log_probs = torch.log_softmax(logits, dim=-1) # shape: (seq_len, vocab_size)

            # 抽取generated token的log_probs
            token_logprobs = []
            for pos in traj.generated_positions:
                if (pos == 0):
                    raise ValueError(
                        f"position can not be zero"
                    )
                
                token_id = input_ids[0, pos]    # 这个位置上的真实token
                token_lp = log_probs[pos - 1, token_id]  # 第 pos-1 个位置的 logits / log_probs，用来预测第 pos 个 token
                token_logprobs.append(token_lp)
            
            token_logprobs = torch.stack(token_logprobs)  # shape: (num_generated_tokens,)
            all_logprobs.append(token_logprobs)
        
        return all_logprobs
    
    def update_step(self, trajectories: List[Trajectory], old_logprobs: List[torch.Tensor]) -> torch.Tensor:
        if len(trajectories) == 0:
            raise ValueError("trajectories cannot be empty.")
         
        if len(old_logprobs) != len(trajectories):
            raise ValueError(
                f"old_logprobs batch size ({len(old_logprobs)}) must match trajectories batch size ({len(trajectories)})."
            )

        # 设置模型模式
        self.policy_model.train()
        self.ref_model.eval()

        rewards = [traj.reward for traj in trajectories]
        advantages = self.compute_advantages(rewards)

        new_logprobs = self.compute_logprobs(self.policy_model, trajectories)
        
        with torch.no_grad():
            ref_logprobs = self.compute_logprobs(self.ref_model, trajectories)

        loss = self.loss_fn(new_logprobs, old_logprobs, advantages, ref_logprobs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
