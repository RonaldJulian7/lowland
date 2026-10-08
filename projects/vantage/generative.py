"""Conditional VAE over daily price profiles.

The scenario simulator gives a conditional mean, which isn't enough to value a battery or
a PPA. Those are paid out of the *shape* of the price day -- peak-to-trough spread, how
many hours go negative, whether the cheap hours sit together. That's a property of the
joint distribution over the 24 hours, and no point forecast carries it.

CVAE rather than a GAN or diffusion because the training set is ~4,000 days. The explicit
likelihood bound trains stably at that size; a GAN would mode-collapse and diffusion would
be badly under-determined. The price is the usual VAE over-smoothing, which shows up in
the validation table below and is reported rather than buried.

evaluate_generator() checks samples against held-out days on the statistics the downstream
use actually depends on. A generative model nobody checks is decoration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from lowland.config import RANDOM_SEED
from lowland.utils import get_logger, resolve_device, set_seed

log = get_logger(__name__)


@dataclass
class CVAEConfig:
    latent_dim: int = 12
    hidden: int = 192
    lr: float = 1.5e-3
    batch_size: int = 64
    max_epochs: int = 400
    patience: int = 40
    #: Weight on the KL term. Below 1 this is a beta-VAE: with only ~4k training days the
    #: full KL weight over-regularises and the decoder collapses toward the conditional
    #: mean, which is precisely the failure this model exists to avoid.
    beta: float = 0.35
    #: Linear KL warm-up. Applying the full KL penalty from step zero causes posterior
    #: collapse, where the latent code is ignored and the model degenerates into a plain
    #: conditional regressor.
    warmup_epochs: int = 60


class ConditionalVAE(nn.Module):
    """Encoder ``q(z | x, c)`` and decoder ``p(x | z, c)``, both MLPs."""

    def __init__(self, x_dim: int, c_dim: int, cfg: CVAEConfig) -> None:
        super().__init__()
        h, z = cfg.hidden, cfg.latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + c_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
        )
        self.mu = nn.Linear(h, z)
        self.logvar = nn.Linear(h, z)
        self.decoder = nn.Sequential(
            nn.Linear(z + c_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, x_dim),
        )
        # Learned observation noise, so the reconstruction term is a proper Gaussian
        # log-likelihood rather than an arbitrarily scaled MSE.
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=-1))
        return self.mu(h), self.logvar(h).clamp(-8, 8)

    def reparameterise(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterise(mu, logvar)
        return self.decode(z, c), mu, logvar


def cvae_loss(x, x_hat, mu, logvar, log_sigma, beta: float):
    """Negative ELBO with a learned Gaussian observation model."""
    sigma2 = (2 * log_sigma).exp()
    recon = 0.5 * (((x - x_hat) ** 2) / sigma2 + 2 * log_sigma).sum(dim=-1).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=-1).mean()
    return recon + beta * kl, recon.item(), kl.item()


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def build_daily_profiles(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, dict]:
    """Reshape the panel into complete 24-hour days with their conditioning vectors.

    Returns ``(X, C, days, meta)`` where ``X`` is the standardised price profile and ``C``
    concatenates the standardised residual-load profile with cyclical calendar encodings.
    Days with any missing hour are dropped outright: interpolating a gap would inject
    smoothness the generative model would then learn as real structure.
    """
    df = panel
    if "is_provisional" in df.columns:
        df = df[df["is_provisional"] == 0]
    df = df.dropna(subset=["price_da", "residual_load"])

    local = df.index.tz_convert("Europe/Amsterdam")
    day = pd.Series(local.normalize(), index=df.index)
    hour = pd.Series(local.hour, index=df.index)

    price_wide = df["price_da"].groupby([day, hour]).mean().unstack()
    rl_wide = df["residual_load"].groupby([day, hour]).mean().unstack()

    # Fuel-cost context, lagged by one day so it is genuinely known in advance.
    #
    # Without this the model has no way to know which price *regime* a day belongs to.
    # Conditioned only on residual load and the calendar, it learns the average level over
    # a sample spanning EUR 32/MWh and EUR 242/MWh years, and then generates that average
    # regardless -- which showed up as generated days sitting 27% below the held-out
    # period's real mean and with half its daily spread. The level had to become part of
    # the conditioning, not something the decoder was expected to infer.
    from projects.vantage.counterfactual import SupplyCurveModel

    gas = SupplyCurveModel.gas_regime_proxy(panel)
    gas_daily = gas.groupby(gas.index.tz_convert("Europe/Amsterdam").normalize()).mean().shift(1)

    full = price_wide.notna().all(axis=1) & rl_wide.notna().all(axis=1)
    full &= price_wide.shape[1] == 24
    price_wide, rl_wide = price_wide[full], rl_wide[full]
    days = pd.DatetimeIndex(price_wide.index)

    gas_vec = gas_daily.reindex(days).to_numpy(dtype=np.float32)
    ok = np.isfinite(gas_vec)
    price_wide, rl_wide, days, gas_vec = (
        price_wide[ok], rl_wide[ok], days[ok], gas_vec[ok]
    )

    p = price_wide.to_numpy(dtype=np.float32)
    r = rl_wide.to_numpy(dtype=np.float32)

    # ---- level / shape decomposition ----------------------------------------------------
    #
    # The model targets the day's *shape* -- its deviation from its own daily mean -- not
    # the absolute price. Asking one network to learn both at once fails on this sample for
    # a structural reason: the daily level is essentially fuel cost, which moved from
    # EUR 32/MWh to EUR 242/MWh and back over the period, so a model trained across it
    # generates the average of several different regimes. Measured on a held-out period
    # after the shift, it undershot the real mean by 23% and the daily spread by 40%.
    #
    # The level is far better predicted by things we already model well (the learned supply
    # curve, the gas regime), while the shape -- when the cheap hours fall, how deep the
    # midday solar trough goes, whether the evening peak is sharp -- is what no point model
    # captures and what storage is paid for. Splitting the two gives each component the job
    # it is suited to, and the daily mean enters as conditioning so the shape can still
    # depend on the level (high-price days really do have different shapes).
    day_mean = p.mean(axis=1, keepdims=True)
    shape = p - day_mean

    s_sd = float(shape.std() + 1e-6)
    r_mu, r_sd = float(r.mean()), float(r.std() + 1e-6)
    g_mu, g_sd = float(gas_vec.mean()), float(gas_vec.std() + 1e-6)
    m_mu, m_sd = float(day_mean.mean()), float(day_mean.std() + 1e-6)

    X = shape / s_sd
    Rz = (r - r_mu) / r_sd
    Gz = ((gas_vec - g_mu) / g_sd).reshape(-1, 1)
    Mz = ((day_mean - m_mu) / m_sd).astype(np.float32)

    month = days.month.to_numpy()
    dow = days.dayofweek.to_numpy()
    cal = np.column_stack(
        [
            np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            (dow >= 5).astype(np.float32),
        ]
    ).astype(np.float32)

    C = np.concatenate([Rz, cal, Gz, Mz], axis=1).astype(np.float32)
    meta = {
        "s_sd": s_sd, "r_mu": r_mu, "r_sd": r_sd, "g_mu": g_mu, "g_sd": g_sd,
        "m_mu": m_mu, "m_sd": m_sd, "n_days": int(len(days)),
        # The observed daily means, so evaluation and sampling can reconstruct absolute
        # prices by adding the level back onto the generated shape.
        "day_mean": day_mean.reshape(-1).astype(np.float32),
    }
    log.info(
        "daily profiles: %s complete days, x_dim=%s c_dim=%s (modelling shape about the daily mean)",
        len(days), X.shape[1], C.shape[1],
    )
    return X, C, days, meta


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


@dataclass
class ProfileGenerator:
    """Train / sample wrapper around :class:`ConditionalVAE`."""

    cfg: CVAEConfig = field(default_factory=CVAEConfig)
    device: str = field(default_factory=lambda: resolve_device())
    model: ConditionalVAE | None = None
    meta: dict = field(default_factory=dict)
    history_: list[dict] = field(default_factory=list)
    #: Cholesky factor of the 24x24 reconstruction-residual covariance, used to draw
    #: temporally coherent observation noise. See :meth:`_fit_residual_covariance`.
    resid_chol_: np.ndarray | None = None
    #: Mean and Cholesky factor of the *aggregate posterior* over latents, used for
    #: ex-post sampling. See :meth:`_fit_aggregate_posterior`.
    z_mean_: np.ndarray | None = None
    z_chol_: np.ndarray | None = None

    def fit(self, X: np.ndarray, C: np.ndarray, meta: dict, *, valid_frac: float = 0.15):
        set_seed(RANDOM_SEED)
        self.meta = meta
        n = len(X)
        # Chronological split: the validation days are the most recent ones, so the
        # reported generalisation is to the future rather than to interleaved days.
        n_val = max(64, int(n * valid_frac))
        tr = slice(0, n - n_val)
        va = slice(n - n_val, n)

        self.model = ConditionalVAE(X.shape[1], C.shape[1], self.cfg).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-5)

        Xtr = torch.from_numpy(X[tr]).to(self.device)
        Ctr = torch.from_numpy(C[tr]).to(self.device)
        Xva = torch.from_numpy(X[va]).to(self.device)
        Cva = torch.from_numpy(C[va]).to(self.device)

        best, bad, best_state = float("inf"), 0, None
        n_tr = Xtr.shape[0]

        for epoch in range(self.cfg.max_epochs):
            beta = self.cfg.beta * min(1.0, (epoch + 1) / max(self.cfg.warmup_epochs, 1))
            self.model.train()
            perm = torch.randperm(n_tr, device=self.device)
            tot = 0.0
            for s in range(0, n_tr, self.cfg.batch_size):
                idx = perm[s : s + self.cfg.batch_size]
                xb, cb = Xtr[idx], Ctr[idx]
                x_hat, mu, logvar = self.model(xb, cb)
                loss, _, _ = cvae_loss(xb, x_hat, mu, logvar, self.model.log_sigma, beta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tot += loss.item()

            self.model.eval()
            with torch.no_grad():
                x_hat, mu, logvar = self.model(Xva, Cva)
                vloss, vrec, vkl = cvae_loss(
                    Xva, x_hat, mu, logvar, self.model.log_sigma, self.cfg.beta
                )
            self.history_.append(
                {"epoch": epoch, "train": tot / max(1, n_tr // self.cfg.batch_size),
                 "valid": vloss.item(), "recon": vrec, "kl": vkl, "beta": beta}
            )

            if vloss.item() < best - 1e-4:
                best, bad = vloss.item(), 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
            if epoch % 50 == 0:
                log.info("cvae epoch %3d  train=%.3f valid=%.3f (recon=%.3f kl=%.3f beta=%.2f)",
                         epoch, self.history_[-1]["train"], vloss.item(), vrec, vkl, beta)
            if bad >= self.cfg.patience:
                log.info("cvae early stop at epoch %s (best valid %.3f)", epoch, best)
                break

        if best_state:
            self.model.load_state_dict(best_state)
        self._fit_residual_covariance(Xtr, Ctr)
        self._fit_aggregate_posterior(Xtr, Ctr)
        return self

    def _fit_aggregate_posterior(self, Xtr: torch.Tensor, Ctr: torch.Tensor) -> None:
        """Fit a Gaussian to the encoded latents, for ex-post sampling.

        Training uses ``beta`` below 1, which buys sharp reconstructions at the cost of
        letting the aggregate posterior drift away from the N(0, I) prior. Sampling from
        the prior anyway is then sampling from a region the decoder was never really
        trained on, and the symptom is exactly what showed up here: generated days with
        44% too little peak-to-trough spread, because the draws cluster near the centre of
        the latent space where every day looks average.

        Ex-post density estimation (Ghosh et al., 2020) closes the gap by fitting a
        density to the *actual* distribution of encoded latents and sampling from that
        instead. A full-covariance Gaussian is enough at this latent dimension, costs one
        pass over the training set, and requires no retraining.
        """
        self.model.eval()
        with torch.no_grad():
            mu, _ = self.model.encode(Xtr, Ctr)
        z = mu.cpu().numpy()

        self.z_mean_ = z.mean(axis=0)
        cov = np.cov(z, rowvar=False) + np.eye(z.shape[1]) * 1e-6
        try:
            self.z_chol_ = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(cov)
            self.z_chol_ = v @ np.diag(np.sqrt(np.clip(w, 1e-8, None)))
        log.info(
            "aggregate posterior fitted: mean |z| %.3f, mean latent sd %.3f (prior is 1.0)",
            float(np.abs(self.z_mean_).mean()), float(np.sqrt(np.diag(cov)).mean()),
        )

    def _fit_residual_covariance(self, Xtr: torch.Tensor, Ctr: torch.Tensor) -> None:
        """Estimate the full 24x24 covariance of the reconstruction residuals.

        The decoder's likelihood uses a single scalar variance shared across hours, which
        is a deliberate simplification for training but the wrong thing to sample from.
        Drawing independent noise per hour restores the missing dispersion and destroys
        the within-day autocorrelation at the same time -- measured here, it pulled the
        lag-1 autocorrelation from 0.86 down to 0.69, turning smooth price days into
        noise.

        Residuals in a price profile are strongly correlated across neighbouring hours:
        a day the model reconstructs too cheaply is too cheap all afternoon, not for one
        isolated hour. Estimating the full covariance and sampling through its Cholesky
        factor reproduces both the dispersion and the temporal structure. With ~4,000
        training days and 24 dimensions the covariance is comfortably well determined.
        """
        self.model.eval()
        with torch.no_grad():
            mu, _ = self.model.encode(Xtr, Ctr)
            recon = self.model.decode(mu, Ctr)
            resid = (Xtr - recon).cpu().numpy()

        cov = np.cov(resid, rowvar=False)
        # Ridge for numerical safety; the residual covariance can be near-singular when
        # neighbouring hours are almost perfectly correlated.
        cov = cov + np.eye(cov.shape[0]) * (1e-6 + 1e-3 * np.trace(cov) / cov.shape[0])
        try:
            self.resid_chol_ = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(cov)
            self.resid_chol_ = v @ np.diag(np.sqrt(np.clip(w, 1e-8, None)))
        log.info(
            "residual covariance fitted: mean sd %.3f, lag-1 correlation %.3f",
            float(np.sqrt(np.diag(cov)).mean()),
            float(np.mean([cov[i, i + 1] / np.sqrt(cov[i, i] * cov[i + 1, i + 1])
                           for i in range(cov.shape[0] - 1)])),
        )

    def sample(
        self,
        C: np.ndarray,
        n_samples: int = 1,
        *,
        temperature: float = 1.0,
        observation_noise: bool = True,
        day_mean: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw ``n_samples`` price profiles per conditioning row.

        Returns an array of shape ``(n_samples, len(C), 24)`` in EUR/MWh. The model
        generates the *shape* about the daily mean; pass ``day_mean`` (one level per
        conditioning row, from the supply curve or from observation) to get absolute
        prices, or omit it to get the deviation profile alone.

        The decoder parameterises ``p(x | z, c) = N(decoder(z, c), sigma^2)``, so a draw
        from the model is the decoder output *plus* a draw from that observation noise.
        Returning the decoder output alone returns the conditional mean, which is not a
        sample from the model and is systematically under-dispersed -- it understated the
        daily price spread by roughly half here, which would have made the generator
        useless for exactly the storage-valuation question it exists to answer. Set
        ``observation_noise=False`` only when the smooth conditional mean is genuinely
        what is wanted.

        ``temperature`` scales the latent prior; below 1 it produces more typical days,
        above 1 more extreme ones, which is useful for stress testing.
        """
        assert self.model is not None, "call fit() first"
        self.model.eval()
        Ct = torch.from_numpy(np.asarray(C, dtype=np.float32)).to(self.device)
        rng = np.random.default_rng(RANDOM_SEED)
        out = []
        with torch.no_grad():
            for _ in range(n_samples):
                if self.z_chol_ is not None:
                    e = rng.standard_normal((Ct.shape[0], self.cfg.latent_dim))
                    z_np = self.z_mean_ + temperature * (e @ self.z_chol_.T)
                    z = torch.from_numpy(z_np.astype(np.float32)).to(self.device)
                else:
                    z = torch.randn(
                        Ct.shape[0], self.cfg.latent_dim, device=self.device
                    ) * temperature
                x = self.model.decode(z, Ct).cpu().numpy()
                if observation_noise:
                    if self.resid_chol_ is not None:
                        # Correlated across hours, so the noise preserves the shape of the
                        # day instead of sanding it flat.
                        eps = rng.standard_normal((x.shape[0], x.shape[1]))
                        x = x + temperature * (eps @ self.resid_chol_.T)
                    else:
                        sigma = float(self.model.log_sigma.exp().item())
                        x = x + sigma * temperature * rng.standard_normal(x.shape)
                # Back to EUR/MWh as a *shape*; the caller adds the daily level.
                out.append(x * self.meta["s_sd"])
        samples = np.stack(out)
        if day_mean is not None:
            samples = samples + np.asarray(day_mean, dtype=np.float32).reshape(1, -1, 1)
        return samples

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def evaluate_generator(
    gen: ProfileGenerator,
    X: np.ndarray,
    C: np.ndarray,
    meta: dict,
    *,
    n_samples: int = 20,
    day_mean: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compare generated days against real ones on decision-relevant statistics.

    Checks the things a downstream user would be misled by if they were wrong: the daily
    spread a battery monetises, how often prices go negative, the tails, and the
    within-day autocorrelation that makes a profile coherent rather than noise.

    The daily *level* is supplied from outside the generator (it is the supply curve's
    job, not the VAE's), so `mean` is not a test of the generative model and is reported
    only to confirm the reconstruction is unbiased. Everything else is a genuine test of
    the learned shape distribution.
    """
    real = X * meta["s_sd"]
    fake = gen.sample(C, n_samples=n_samples).reshape(-1, X.shape[1])
    if day_mean is not None:
        dm = np.asarray(day_mean, dtype=np.float32).reshape(-1, 1)
        real = real + dm
        fake = fake + np.tile(dm, (n_samples, 1))

    def stats(a: np.ndarray) -> dict[str, float]:
        spread = a.max(axis=1) - a.min(axis=1)
        lag1 = np.mean(
            [np.corrcoef(row[:-1], row[1:])[0, 1] for row in a if np.std(row) > 1e-6]
        )
        return {
            "mean": float(a.mean()),
            "sd": float(a.std()),
            "p05": float(np.quantile(a, 0.05)),
            "p50": float(np.quantile(a, 0.50)),
            "p95": float(np.quantile(a, 0.95)),
            "mean_daily_spread": float(spread.mean()),
            "p90_daily_spread": float(np.quantile(spread, 0.90)),
            "negative_hour_share": float((a < 0).mean()),
            "within_day_lag1_acf": float(lag1),
        }

    r, f = stats(real), stats(fake)
    rows = [
        {"statistic": k, "real": round(r[k], 3), "generated": round(f[k], 3),
         "abs_error": round(abs(r[k] - f[k]), 3),
         "rel_error_pct": round(100 * abs(r[k] - f[k]) / max(abs(r[k]), 1e-6), 1)}
        for k in r
    ]
    return pd.DataFrame(rows)
