"""Gym-style battery trading environment.

MPC is already near-optimal given a forecast, and this project shows that. RL is here for
the part that isn't in the forecast -- spikes whose timing is unpredictable, and the
state-dependence of the decision (how much room is left, how many cycles are spent). A
learned policy also runs in microseconds instead of re-solving a MILP every hour. The
catch, measured rather than hidden: no optimality guarantee, and MPC beats it when the
forecast is good.

Observation is only what's genuinely available at decision time: SoC and cycles used
today, the current price normalised by a trailing window, the forecast quantile band over
the next 12 h, and cyclical time. The trailing-window normalisation is the bit that lets
one policy work across both the EUR 30-50/MWh years and 2022 without retraining -- EUR 80
is cheap in a EUR 200 week and expensive in a EUR 40 one.

Reward is realised margin minus degradation. Realised, not forecast -- otherwise the
agent gets paid for trusting a forecast that turned out wrong.

No gymnasium dependency: the loop is small enough to read, and this avoids pinning a
fast-moving RL library.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lowland.config import DEFAULT_BATTERY, BatterySpec
from lowland.utils import get_logger

log = get_logger(__name__)


@dataclass
class BatteryEnvConfig:
    """Environment hyperparameters."""

    spec: BatterySpec = DEFAULT_BATTERY
    episode_hours: int = 168            # one week
    dt_h: float = 1.0
    forecast_horizon: int = 12
    price_window: int = 168             # trailing window for price normalisation
    congestion_lambda: float = 0.0
    #: Reward scaling. Raw euro rewards run to thousands, which destabilises the value
    #: function; dividing by a reference scale keeps returns in a sane range without
    #: changing the optimal policy.
    reward_scale: float = 1000.0
    #: Penalty applied when the agent requests an infeasible action, so it learns the
    #: physical limits instead of relying on the clipping that enforces them.
    infeasibility_penalty: float = 0.02


class BatteryTradingEnv:
    """A minimal Gym-style environment; no external RL dependency required.

    Implements ``reset`` / ``step`` with the usual semantics. Keeping it dependency-free
    avoids pinning a specific gymnasium/stable-baselines version and makes the training
    loop fully auditable, which matters more here than framework compatibility.
    """

    def __init__(
        self,
        prices: pd.Series,
        cfg: BatteryEnvConfig | None = None,
        *,
        forecast_quantiles: np.ndarray | None = None,
        congestion_weight: pd.Series | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.cfg = cfg or BatteryEnvConfig()
        self.prices = prices.to_numpy(dtype=np.float32)
        self.index = prices.index
        self.rng = rng or np.random.default_rng(0)

        self.fq = forecast_quantiles
        self.cw = (
            congestion_weight.to_numpy(dtype=np.float32)
            if congestion_weight is not None
            else np.zeros_like(self.prices)
        )

        local = pd.DatetimeIndex(self.index).tz_convert("Europe/Amsterdam")
        self.hour_sin = np.sin(2 * np.pi * local.hour / 24).astype(np.float32)
        self.hour_cos = np.cos(2 * np.pi * local.hour / 24).astype(np.float32)
        self.dow_sin = np.sin(2 * np.pi * local.dayofweek / 7).astype(np.float32)
        self.dow_cos = np.cos(2 * np.pi * local.dayofweek / 7).astype(np.float32)

        # Trailing price statistics, computed once. Using a causal rolling window means
        # the normalisation itself never looks ahead.
        s = pd.Series(self.prices)
        self.p_mu = s.rolling(self.cfg.price_window, min_periods=24).mean().bfill().to_numpy(np.float32)
        self.p_sd = (
            s.rolling(self.cfg.price_window, min_periods=24).std().bfill().fillna(1.0).to_numpy(np.float32)
        )
        self.p_sd = np.maximum(self.p_sd, 5.0)  # floor avoids division blow-up in flat weeks

        self.t0 = 0
        self.t = 0
        self.soc = self.cfg.spec.soc_init
        self.throughput_today = 0.0
        self._day_marker = 0

    # ---- spaces ---------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        # soc, cycles_today, price_z, price_level, 3 x forecast band, time (4), progress
        return 7 + 3 * self.cfg.forecast_horizon

    @property
    def action_dim(self) -> int:
        return 1

    # ---- core ------------------------------------------------------------------------

    def reset(self, start: int | None = None) -> np.ndarray:
        n = len(self.prices)
        span = self.cfg.episode_hours
        max_start = max(1, n - span - self.cfg.forecast_horizon - 1)
        self.t0 = int(self.rng.integers(0, max_start)) if start is None else int(start)
        self.t = self.t0
        self.soc = self.cfg.spec.soc_init
        self.throughput_today = 0.0
        self._day_marker = self.t // 24
        return self._observe()

    def _forecast_band(self) -> np.ndarray:
        """Normalised low/median/high forecast path over the next ``forecast_horizon`` hours."""
        H = self.cfg.forecast_horizon
        mu, sd = self.p_mu[self.t], self.p_sd[self.t]
        idx = np.arange(self.t + 1, self.t + 1 + H)
        idx = np.clip(idx, 0, len(self.prices) - 1)

        if self.fq is not None:
            band = self.fq[idx]                      # (H, Q)
            lo = band[:, 0]
            med = band[:, band.shape[1] // 2]
            hi = band[:, -1]
        else:
            # Without a probabilistic forecast, fall back to perfect foresight of the
            # path with zero width. Used only for ablation runs.
            med = self.prices[idx]
            lo = hi = med

        return np.concatenate([(lo - mu) / sd, (med - mu) / sd, (hi - mu) / sd]).astype(np.float32)

    def _observe(self) -> np.ndarray:
        spec = self.cfg.spec
        mu, sd = self.p_mu[self.t], self.p_sd[self.t]
        obs = np.concatenate(
            [
                np.array(
                    [
                        (self.soc - spec.soc_min) / max(spec.soc_max - spec.soc_min, 1e-6),
                        self.throughput_today / max(spec.max_cycles_per_day * spec.energy_mwh, 1e-6),
                        (self.prices[self.t] - mu) / sd,
                        np.tanh(self.prices[self.t] / 200.0),
                        self.hour_sin[self.t],
                        self.hour_cos[self.t],
                        (self.t - self.t0) / max(self.cfg.episode_hours, 1),
                    ],
                    dtype=np.float32,
                ),
                self._forecast_band(),
            ]
        )
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        """Apply ``action`` in ``[-1, 1]``: positive discharges, negative charges."""
        spec = self.cfg.spec
        dt = self.cfg.dt_h
        a = float(np.clip(action, -1.0, 1.0))

        requested = a * spec.power_mw
        charge = discharge = 0.0

        if requested >= 0:
            # Discharge is limited by available energy above the floor.
            max_d = min(spec.power_mw, (self.soc - spec.soc_min) * spec.eta_discharge / dt)
            discharge = float(np.clip(requested, 0.0, max(max_d, 0.0)))
        else:
            max_c = min(spec.power_mw, (spec.soc_max - self.soc) / (spec.eta_charge * dt))
            charge = float(np.clip(-requested, 0.0, max(max_c, 0.0)))

        # Daily cycle budget.
        budget = spec.max_cycles_per_day * spec.energy_mwh
        remaining = max(budget - self.throughput_today, 0.0)
        if discharge * dt > remaining:
            discharge = remaining / dt

        infeasible = abs(abs(requested) - (charge + discharge)) / max(spec.power_mw, 1e-6)

        price = float(self.prices[self.t])
        gross = price * (discharge - charge) * dt
        deg = spec.degradation_cost_eur_per_mwh * (discharge + charge) * dt
        congestion = self.cfg.congestion_lambda * float(self.cw[self.t]) * (charge - discharge) * dt
        profit = gross - deg - congestion

        self.soc = float(
            np.clip(
                self.soc + spec.eta_charge * charge * dt - (discharge / spec.eta_discharge) * dt,
                spec.soc_min,
                spec.soc_max,
            )
        )
        self.throughput_today += discharge * dt

        self.t += 1
        if self.t // 24 != self._day_marker:
            self._day_marker = self.t // 24
            self.throughput_today = 0.0

        done = (self.t - self.t0) >= self.cfg.episode_hours or self.t >= len(self.prices) - 1
        reward = profit / self.cfg.reward_scale - self.cfg.infeasibility_penalty * infeasible

        info = {
            "profit_eur": profit,
            "gross_eur": gross,
            "degradation_eur": deg,
            "congestion_eur": congestion,
            "charge_mw": charge,
            "discharge_mw": discharge,
            "soc_mwh": self.soc,
            "price": price,
            "timestamp": self.index[self.t - 1],
        }
        obs = self._observe() if not done else np.zeros(self.obs_dim, dtype=np.float32)
        return obs, float(reward), done, info

    # ---- evaluation ------------------------------------------------------------------

    def rollout(self, policy_fn, start: int = 0, n_hours: int | None = None) -> pd.DataFrame:
        """Run a policy deterministically over a fixed span and return the schedule."""
        n_hours = n_hours or self.cfg.episode_hours
        saved = self.cfg.episode_hours
        self.cfg.episode_hours = n_hours
        obs = self.reset(start=start)
        rows = []
        done = False
        while not done:
            a = policy_fn(obs)
            obs, _, done, info = self.step(a)
            rows.append(info)
        self.cfg.episode_hours = saved
        return pd.DataFrame(rows).set_index("timestamp")
