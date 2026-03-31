"""
Actor–Critic agent with Torch MLP and masked softmax policy for 6x6 Checkers.
Both sides share the same parameters (self-play); observations are player-centric
from the environment.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from mycheckersenv import BOARD_SIZE, MAX_ACTIONS

OBS_DIM = BOARD_SIZE * BOARD_SIZE * 4


class ActorCritic(nn.Module):
    """Shared trunk, separate policy and value heads for actor and critic."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(OBS_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, MAX_ACTIONS)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs_flat)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def act(
        self,
        observation: Dict[str, Any],
        device: torch.device,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        obs = torch.as_tensor(
            observation["observation"].reshape(-1),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        mask = torch.as_tensor(
            observation["action_mask"],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        logits, value = self.forward(obs)
        logits = logits.masked_fill(mask == 0, float("-inf"))
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    def evaluate(
        self,
        obs_flat: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs_flat)
        logits = logits.masked_fill(mask == 0, float("-inf"))
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return logp, values, entropy


def batch_obs(
    observations: List[Dict[str, Any]], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    obs_mat = np.stack([o["observation"].reshape(-1) for o in observations], axis=0)
    mask_mat = np.stack([o["action_mask"] for o in observations], axis=0)
    return (
        torch.as_tensor(obs_mat, dtype=torch.float32, device=device),
        torch.as_tensor(mask_mat, dtype=torch.float32, device=device),
    )


class SelfPlayActorCritic:
    """On-policy episodic A2C-style update after each full game."""

    def __init__(
        self,
        lr: float = 3e-4,
        device: str | torch.device | None = None,
        hidden: int = 256,
        gamma: float = 0.99,
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.net = ActorCritic(hidden=hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def act(self, observation: Dict[str, Any], deterministic: bool = False) -> Tuple[int, float, float]:
        return self.net.act(observation, self.device, deterministic=deterministic)

    def update_on_episode(
        self,
        observations: List[Dict[str, Any]],
        actions: List[int],
        rewards: List[float],
        dones: List[bool],
    ) -> Dict[str, float]:
        """Monte Carlo returns; advantage = return - V(s). One gradient step on full episode."""
        returns: List[float] = []
        g = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                g = 0.0
            g = r + self.gamma * g
            returns.append(g)
        returns = list(reversed(returns))

        obs_t, mask_t = batch_obs(observations, self.device)
        act_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, values_old = self.net.forward(obs_t)
        adv = ret_t - values_old
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        logp, values, ent = self.net.evaluate(obs_t, mask_t, act_t)
        policy_loss = -(logp * adv).mean()
        value_loss = F.mse_loss(values, ret_t)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * ent

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(ent.item()),
            "mean_return": float(np.mean(returns)),
        }

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd, strict: bool = True):
        self.net.load_state_dict(sd, strict=strict)
