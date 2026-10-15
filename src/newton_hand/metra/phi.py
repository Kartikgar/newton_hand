"""METRA representation ``phi`` and its temporal-distance constraint.

Park et al., "METRA: Scalable Unsupervised RL with Metric-Aware Abstraction"
(ICLR 2024). ``phi`` maps a state to R^D and is trained to maximize
``(phi(s') - phi(s))^T z`` subject to ``||phi(s') - phi(s)|| <= 1`` for every
observed transition. The constraint is enforced with a learned Lagrange
multiplier ``lambda``, following the authors' implementation:

    penalty  = min(eps, 1 - ||phi(s') - phi(s)||^2)
    L_phi    = -E[(phi(s') - phi(s))^T z + lambda.detach() * penalty]
    L_lambda =  log(lambda) * E[penalty.detach()]

A transition that breaks the constraint makes ``penalty`` negative, which pulls
``phi`` back and raises ``lambda``; once every transition satisfies it, the
``eps`` clamp stops ``phi`` from being paid for slack.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class MetraPhi(nn.Module):
    """MLP ``phi`` with a fixed per-dimension input scale, plus ``log(lambda)``."""

    def __init__(
        self,
        input_dim: int,
        skill_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (512, 512),
        input_scale: torch.Tensor | None = None,
        learning_rate: float = 1.0e-4,
        dual_lambda_init: float = 30.0,
        dual_learning_rate: float = 1.0e-4,
        constraint_eps: float = 1.0e-3,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.skill_dim = int(skill_dim)
        self.constraint_eps = float(constraint_eps)

        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, self.skill_dim))
        self.net = nn.Sequential(*layers)

        scale = torch.ones(self.input_dim) if input_scale is None else input_scale.float()
        self.register_buffer("input_scale", scale.reshape(self.input_dim))
        self.log_dual_lambda = nn.Parameter(torch.tensor(math.log(dual_lambda_init)))

        self.to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(learning_rate))
        self.dual_optimizer = torch.optim.Adam([self.log_dual_lambda], lr=float(dual_learning_rate))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states * self.input_scale)

    @torch.no_grad()
    def reward(
        self, states: torch.Tensor, next_states: torch.Tensor, skills: torch.Tensor
    ) -> torch.Tensor:
        """``(phi(s') - phi(s))^T z``, shape ``(N,)``."""
        return ((self(next_states) - self(states)) * skills).sum(dim=-1)

    def update(
        self, states: torch.Tensor, next_states: torch.Tensor, skills: torch.Tensor
    ) -> dict[str, float]:
        """One gradient step on ``phi`` and one on ``lambda``."""
        delta = self(next_states) - self(states)
        objective = (delta * skills).sum(dim=-1)
        squared_distance = delta.pow(2).sum(dim=-1)
        penalty = torch.clamp(1.0 - squared_distance, max=self.constraint_eps)

        dual_lambda = self.log_dual_lambda.exp()
        loss_phi = -(objective + dual_lambda.detach() * penalty).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss_phi.backward()
        self.optimizer.step()

        loss_dual = self.log_dual_lambda * penalty.detach().mean()
        self.dual_optimizer.zero_grad(set_to_none=True)
        loss_dual.backward()
        self.dual_optimizer.step()

        return {
            "phi_loss": float(loss_phi.detach()),
            "objective": float(objective.detach().mean()),
            "lambda": float(dual_lambda.detach()),
            "phi_step_sq": float(squared_distance.detach().mean()),
            # Transitions over the unit step: the constraint is being broken.
            "constraint_violation": float((squared_distance.detach() > 1.0).float().mean()),
        }

    def training_state(self) -> dict:
        return {
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "dual_optimizer": self.dual_optimizer.state_dict(),
        }

    def load_training_state(self, state: dict) -> None:
        self.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.dual_optimizer.load_state_dict(state["dual_optimizer"])
