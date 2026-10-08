"""PPO for continuous battery dispatch.

Standard PPO-clip (Schulman et al. 2017), written out rather than imported so the bits
that matter for this problem -- state-dependent action bounds, reward scaling across
price regimes -- are visible instead of buried in a wrapper.

Includes the refinements that actually matter in practice: GAE at lambda 0.95,
per-batch advantage normalisation (keeps the step scale-free as reward magnitude shifts
between calm and volatile weeks), gradient clipping, and early stop on approximate KL so
one bad batch can't wreck the policy.

The log-std is state-independent and starts wide. Battery arbitrage has a nasty local
optimum at "do nothing" -- always safe, earns zero -- and an agent that collapses
exploration early never gets out of it.

Actions are tanh-squashed into [-1, 1]. The log-prob correction for the squash is applied
explicitly; leaving it out is a common bug that biases the entropy term and quietly kills
exploration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from lowland.config import RANDOM_SEED
from lowland.utils import get_logger, resolve_device, set_seed

log = get_logger(__name__)


@dataclass
class PPOConfig:
    """Hyperparameters for :class:`PPOAgent`."""

    hidden: int = 128
    lr: float = 3e-4
    gamma: float = 0.995          # near-1: storage value is realised many hours later
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.004
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 8
    minibatch: int = 256
    rollout_steps: int = 4096
    n_updates: int = 120
    target_kl: float = 0.02
    init_log_std: float = -0.3


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic with a Gaussian policy over the pre-squash action."""

    def __init__(self, obs_dim: int, cfg: PPOConfig) -> None:
        super().__init__()
        h = cfg.hidden
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
        )
        self.mu = nn.Linear(h, 1)
        self.v = nn.Linear(h, 1)
        # State-independent log-std: fewer parameters and far more stable early on than
        # letting the network output its own variance, which tends to collapse.
        self.log_std = nn.Parameter(torch.ones(1) * cfg.init_log_std)

        # Orthogonal init with a small final-layer gain keeps the initial policy close to
        # zero-mean, i.e. "do nothing", which is the right prior for a trading agent.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.orthogonal_(self.v.weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        z = self.trunk(obs)
        return self.mu(z), self.v(z).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mu, value = self(obs)
        std = self.log_std.exp()
        if deterministic:
            u = mu
            return torch.tanh(u).squeeze(-1), None, value

        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)
        # Change-of-variables correction for the tanh squash.
        logp = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)
        return a.squeeze(-1), logp, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        mu, value = self(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        a = actions.clamp(-0.999999, 0.999999).unsqueeze(-1)
        u = torch.atanh(a)
        logp = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)
        return logp, value, dist.entropy().sum(-1)


@dataclass
class PPOAgent:
    """PPO trainer and policy wrapper."""

    obs_dim: int
    cfg: PPOConfig = field(default_factory=PPOConfig)
    device: str = field(default_factory=lambda: resolve_device())
    net: ActorCritic | None = None
    history_: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        set_seed(RANDOM_SEED)
        self.net = ActorCritic(self.obs_dim, self.cfg).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)

    # ---- rollout ---------------------------------------------------------------------

    def collect(self, env) -> dict[str, torch.Tensor]:
        """Gather ``rollout_steps`` transitions, resetting whenever an episode ends."""
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        obs = env.reset()
        ep_returns, ep_ret = [], 0.0

        for _ in range(self.cfg.rollout_steps):
            ot = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                a, logp, v = self.net.act(ot)
            a_np = float(a.item())

            next_obs, r, done, _ = env.step(a_np)

            obs_buf.append(obs)
            act_buf.append(a_np)
            logp_buf.append(float(logp.item()))
            rew_buf.append(r)
            val_buf.append(float(v.item()))
            done_buf.append(float(done))

            ep_ret += r
            obs = next_obs
            if done:
                ep_returns.append(ep_ret)
                ep_ret = 0.0
                obs = env.reset()

        with torch.no_grad():
            last_v = float(
                self.net(
                    torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                )[1].item()
            )

        adv, ret = self._gae(
            np.array(rew_buf, dtype=np.float32),
            np.array(val_buf, dtype=np.float32),
            np.array(done_buf, dtype=np.float32),
            last_v,
        )
        t = lambda x, d=torch.float32: torch.as_tensor(np.array(x), dtype=d, device=self.device)  # noqa: E731
        return {
            "obs": t(obs_buf),
            "act": t(act_buf),
            "logp": t(logp_buf),
            "adv": t(adv),
            "ret": t(ret),
            "ep_return": float(np.mean(ep_returns)) if ep_returns else float("nan"),
        }

    def _gae(self, rew: np.ndarray, val: np.ndarray, done: np.ndarray, last_v: float):
        """Generalised advantage estimation."""
        n = len(rew)
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for i in reversed(range(n)):
            next_v = last_v if i == n - 1 else val[i + 1]
            next_nonterminal = 1.0 - done[i]
            delta = rew[i] + self.cfg.gamma * next_v * next_nonterminal - val[i]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * gae
            adv[i] = gae
        return adv, adv + val

    # ---- update ----------------------------------------------------------------------

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs, act, old_logp = batch["obs"], batch["act"], batch["logp"]
        adv, ret = batch["adv"], batch["ret"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        n_batches = 0

        for _ in range(self.cfg.n_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.cfg.minibatch):
                mb = idx[s : s + self.cfg.minibatch]
                mbt = torch.as_tensor(mb, device=self.device)

                logp, value, ent = self.net.evaluate(obs[mbt], act[mbt])
                ratio = (logp - old_logp[mbt]).exp()

                unclipped = ratio * adv[mbt]
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv[mbt]
                pi_loss = -torch.min(unclipped, clipped).mean()
                v_loss = 0.5 * (value - ret[mbt]).pow(2).mean()
                loss = pi_loss + self.cfg.value_coef * v_loss - self.cfg.entropy_coef * ent.mean()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (logp - old_logp[mbt])).mean().item()
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.mean().item()
                stats["kl"] += approx_kl
                n_batches += 1

            if n_batches and stats["kl"] / n_batches > self.cfg.target_kl:
                # A single over-large step can wreck a policy that took many updates to
                # build; stopping the epoch loop early is cheap insurance.
                break

        return {k: v / max(n_batches, 1) for k, v in stats.items()}

    def train(self, env, *, log_every: int = 10) -> PPOAgent:
        for update in range(self.cfg.n_updates):
            batch = self.collect(env)
            stats = self.update(batch)
            stats["update"] = update
            stats["ep_return"] = batch["ep_return"]
            stats["log_std"] = float(self.net.log_std.item())
            self.history_.append(stats)
            if update % log_every == 0 or update == self.cfg.n_updates - 1:
                log.info(
                    "update %3d  ep_return=%8.3f  pi=%+.4f  v=%.4f  H=%.3f  kl=%.4f  logstd=%.2f",
                    update, stats["ep_return"], stats["pi_loss"], stats["v_loss"],
                    stats["entropy"], stats["kl"], stats["log_std"],
                )
        return self

    # ---- inference --------------------------------------------------------------------

    def policy(self, deterministic: bool = True):
        """Return a ``policy_fn(obs) -> action`` usable by :meth:`BatteryTradingEnv.rollout`."""

        def fn(obs: np.ndarray) -> float:
            with torch.no_grad():
                ot = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                a, _, _ = self.net.act(ot, deterministic=deterministic)
            return float(a.item())

        return fn

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def save(self, path) -> None:
        torch.save({"state_dict": self.net.state_dict(), "obs_dim": self.obs_dim}, path)

    def load(self, path) -> PPOAgent:
        blob = torch.load(path, map_location=self.device)
        self.net.load_state_dict(blob["state_dict"])
        return self
