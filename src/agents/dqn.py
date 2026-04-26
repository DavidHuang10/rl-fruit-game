# Generated with Claude Code (claude-sonnet-4-6). Architecture and hyperparameters directed by David Huang.
# Double DQN: online net selects actions, target net evaluates them.

from __future__ import annotations

import copy
import pathlib

import torch
import torch.nn as nn

from agents.network import SuikaQNetwork, obs_to_tensor
from agents.replay_buffer import DictReplayBuffer


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DQNAgent:
    """
    Double DQN agent for SuikaEnv.

    Hyperparameters (all overridable at construction):
        gamma          0.99   discount
        lr             3e-4   Adam learning rate (cosine-decayed over total_grad_steps)
        weight_decay   1e-5   Adam L2 regularisation
        batch_size     128
        grad_clip      10.0   global gradient norm clip
        target_sync    1000   hard-copy online->target every N gradient steps
        eps_start      1.0    initial epsilon for epsilon-greedy
        eps_end        0.05   final epsilon
        eps_decay_steps 100_000  env steps over which epsilon decays linearly
        total_grad_steps computed from training budget; used for LR schedule
    """

    def __init__(
        self,
        device: torch.device | None = None,
        gamma: float = 0.99,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 128,
        grad_clip: float = 10.0,
        target_sync: int = 1000,
        eps_end: float = 0.05,
        eps_decay_steps: int = 100_000,
        total_grad_steps: int = 125_000,   # 500k env steps / 4
    ) -> None:
        self.device = device or _auto_device()
        self.gamma  = gamma
        self.batch_size = batch_size
        self.grad_clip  = grad_clip
        self.target_sync = target_sync
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        self.online = SuikaQNetwork().to(self.device)
        self.target = copy.deepcopy(self.online)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_grad_steps, eta_min=1e-5
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss — robust to large reward outliers

        self._env_steps  = 0
        self._grad_steps = 0

    # ── epsilon ───────────────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        frac = min(self._env_steps / self.eps_decay_steps, 1.0)
        return 1.0 - frac * (1.0 - self.eps_end)

    # ── action selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: dict, eval_mode: bool = False) -> int:
        """Epsilon-greedy action selection. eval_mode forces greedy."""
        if not eval_mode and torch.rand(1).item() < self.epsilon:
            return int(torch.randint(0, 32, (1,)).item())
        tensors = obs_to_tensor(obs, self.device)
        self.online.eval()
        q = self.online(**tensors)          # [1, 32]
        self.online.train()
        return int(q.argmax(dim=-1).item())

    def tick_env_step(self, n: int = 1) -> None:
        """Call once per environment step (even random warmup ones)."""
        self._env_steps += n

    # ── learning ──────────────────────────────────────────────────────────────

    def update(self, buffer: DictReplayBuffer) -> float:
        """Sample a batch, compute Double DQN loss, step optimizer. Returns scalar loss."""
        batch = buffer.sample(self.batch_size)

        obs_t  = obs_to_tensor(batch["obs"],      self.device)
        obs_tp = obs_to_tensor(batch["next_obs"], self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long,    device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        dones   = torch.tensor(batch["dones"],   dtype=torch.float32, device=self.device)

        # Q(s, a) from online network
        q_all = self.online(**obs_t)                             # [B, 32]
        q_sa  = q_all.gather(1, actions.unsqueeze(1)).squeeze(1) # [B]

        # Double DQN target: online net picks a*, target net evaluates it
        with torch.no_grad():
            a_star = self.online(**obs_tp).argmax(dim=1)         # [B]  — online selects
            q_next = self.target(**obs_tp)                       # [B, 32]
            q_next_sa = q_next.gather(1, a_star.unsqueeze(1)).squeeze(1)  # [B]
            target = rewards + self.gamma * (1.0 - dones) * q_next_sa    # [B]

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        self._grad_steps += 1
        if self._grad_steps % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online":      self.online.state_dict(),
            "target":      self.target.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "env_steps":   self._env_steps,
            "grad_steps":  self._grad_steps,
        }, path)

    def load(self, path: str | pathlib.Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self._env_steps  = ckpt["env_steps"]
        self._grad_steps = ckpt["grad_steps"]
