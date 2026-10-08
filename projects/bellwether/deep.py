"""Sequence-to-sequence quantile network (compact TFT).

Cut-down Temporal Fusion Transformer, keeping the parts that earn their place at this
data size: variable selection (softmax weighting over inputs per timestep, which is also
readable afterwards -- wind dominates on stormy days, temperature on cold evenings),
gated residual blocks so the net can route around its own non-linearity where load really
is linear in degree-hours, GRU encoder/decoder, and attention back over the history.

Attention heads share one value projection. Standard multi-head attention maps get
over-interpreted constantly; sharing values means the averaged weights can honestly be
read as "which past hours did it look at".

Emits all quantiles for all horizons in one pass, trained on summed pinball loss, so it's
optimised for the thing it's scored on rather than fitted to MSE and widened afterwards.

Worth keeping alongside LightGBM because they fail differently -- the GBM is better on
sharp weather-driven non-linearities, this is better on trajectory shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from lowland.config import QUANTILES, RANDOM_SEED
from lowland.utils import get_logger, resolve_device, set_seed

log = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------


class GatedLinearUnit(nn.Module):
    """GLU: ``a * sigmoid(b)`` from a single projection split in two."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, 2 * d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GatedResidualNetwork(nn.Module):
    """GRN: ``LayerNorm(skip + GLU(W2 ELU(W1 x + W3 c)))``.

    The optional context ``c`` lets static information condition a time-varying
    transformation. The gate is the important part: it can shut the non-linear branch off
    entirely, so adding depth never costs accuracy on the parts of the mapping that are
    already linear.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1,
                 d_context: int | None = None) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc_context = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(x)
        if self.fc_context is not None and context is not None:
            h = h + self.fc_context(context)
        h = F.elu(h)
        h = self.dropout(self.fc2(h))
        return self.norm(self.skip(x) + self.glu(h))


class VariableSelectionNetwork(nn.Module):
    """Per-timestep softmax weighting over ``n_vars`` scalar inputs.

    Each variable is embedded independently, a GRN over the flattened stack produces
    selection logits, and the embeddings are combined by the resulting weights. Storing
    the weights makes the model self-explaining at no extra cost.
    """

    def __init__(self, n_vars: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_vars)])
        self.selection = GatedResidualNetwork(
            n_vars * d_model, d_model, n_vars, dropout=dropout
        )
        self.var_grns = nn.ModuleList(
            [GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout) for _ in range(n_vars)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, n_vars)
        embeds = [emb(x[..., i : i + 1]) for i, emb in enumerate(self.embeddings)]
        stacked = torch.stack(embeds, dim=-2)              # (B, T, n_vars, d)
        flat = stacked.flatten(start_dim=-2)               # (B, T, n_vars*d)
        weights = torch.softmax(self.selection(flat), dim=-1).unsqueeze(-1)  # (B,T,n_vars,1)
        processed = torch.stack(
            [grn(embeds[i]) for i, grn in enumerate(self.var_grns)], dim=-2
        )
        combined = (processed * weights).sum(dim=-2)       # (B, T, d)
        return combined, weights.squeeze(-1)


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with a single shared value projection.

    Sharing values across heads means the head outputs differ only in *where* they
    attend, so averaging the attention matrices across heads yields a weighting that can
    honestly be read as the model's temporal focus. Standard multi-head attention does not
    have this property, which is why its attention maps are so often over-interpreted.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, self.d_head * n_heads)
        self.k = nn.Linear(d_model, self.d_head * n_heads)
        self.v = nn.Linear(d_model, self.d_head)  # shared across heads
        self.out = nn.Linear(self.d_head, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        B, Tq, _ = q.shape
        Tk = k.shape[1]
        qh = self.q(q).view(B, Tq, self.n_heads, self.d_head).transpose(1, 2)
        kh = self.k(k).view(B, Tk, self.n_heads, self.d_head).transpose(1, 2)
        vh = self.v(v).unsqueeze(1)                        # (B, 1, Tk, d_head)

        scores = qh @ kh.transpose(-2, -1) / np.sqrt(self.d_head)
        attn = self.dropout(torch.softmax(scores, dim=-1))  # (B, H, Tq, Tk)
        ctx = (attn @ vh).mean(dim=1)                       # average over heads
        return self.out(ctx), attn.mean(dim=1)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


@dataclass
class DeepConfig:
    """Hyperparameters for :class:`TemporalQuantileNet`."""

    d_model: int = 64
    n_heads: int = 4
    dropout: float = 0.12
    lr: float = 1.2e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 60
    patience: int = 8
    grad_clip: float = 1.0
    quantiles: tuple[float, ...] = QUANTILES


class TemporalQuantileNet(nn.Module):
    """Encoder-decoder quantile network over past observations and known futures."""

    def __init__(self, n_past_vars: int, n_future_vars: int, horizon: int, cfg: DeepConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.horizon = horizon
        d = cfg.d_model

        self.vsn_past = VariableSelectionNetwork(n_past_vars, d, cfg.dropout)
        self.vsn_future = VariableSelectionNetwork(n_future_vars, d, cfg.dropout)

        self.encoder = nn.GRU(d, d, batch_first=True)
        self.decoder = nn.GRU(d, d, batch_first=True)

        self.post_lstm_gate = GatedLinearUnit(d, d)
        self.post_lstm_norm = nn.LayerNorm(d)

        self.attention = InterpretableMultiHeadAttention(d, cfg.n_heads, cfg.dropout)
        self.post_attn = GatedResidualNetwork(d, d, d, cfg.dropout)

        self.head = nn.Linear(d, len(cfg.quantiles))
        self.last_weights_: dict[str, torch.Tensor] = {}

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        past_emb, w_past = self.vsn_past(x_past)
        fut_emb, w_fut = self.vsn_future(x_future)

        enc_out, h_n = self.encoder(past_emb)
        dec_out, _ = self.decoder(fut_emb, h_n)

        # Gated skip around the recurrent stack.
        dec_out = self.post_lstm_norm(fut_emb + self.post_lstm_gate(dec_out))

        # Attend from each future step back over the encoded history.
        attn_out, attn_w = self.attention(dec_out, enc_out, enc_out)
        fused = self.post_attn(dec_out + attn_out)

        self.last_weights_ = {
            "past_selection": w_past.detach(),
            "future_selection": w_fut.detach(),
            "attention": attn_w.detach(),
        }
        return self.head(fused)  # (B, horizon, n_quantiles)


def pinball_loss_torch(
    y_true: torch.Tensor, y_pred: torch.Tensor, quantiles: tuple[float, ...]
) -> torch.Tensor:
    """Mean pinball loss over horizons and quantile levels.

    ``y_true`` is ``(B, H)``; ``y_pred`` is ``(B, H, Q)``. NaN targets are masked so that
    incomplete tails of the panel do not poison the gradient.
    """
    taus = torch.tensor(quantiles, device=y_pred.device, dtype=y_pred.dtype).view(1, 1, -1)
    diff = y_true.unsqueeze(-1) - y_pred
    loss = torch.maximum(taus * diff, (taus - 1.0) * diff)
    mask = torch.isfinite(y_true).unsqueeze(-1)
    loss = torch.where(mask, loss, torch.zeros_like(loss))
    denom = mask.sum().clamp(min=1) * len(quantiles)
    return loss.sum() / denom


# --------------------------------------------------------------------------------------
# Sequence dataset construction
# --------------------------------------------------------------------------------------

#: Variables fed to the encoder as observed history.
PAST_VARS = (
    "target",
    "temp_pop",
    "hdh",
    "wind_power_proxy_off",
    "wind_power_proxy_on",
    "solar_proxy",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)

#: Variables known in advance and fed to the decoder.
FUTURE_VARS = (
    "temp_pop",
    "hdh",
    "cdh",
    "wind_power_proxy_off",
    "wind_power_proxy_on",
    "solar_proxy",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_holiday",
)


def _cyclical(panel: pd.DataFrame) -> pd.DataFrame:
    """Sine/cosine encodings of local hour and weekday.

    A sequence model handles the wrap-around badly if hour is fed as 0-23: hour 23 and
    hour 0 are adjacent in reality but maximally distant as integers.
    """
    from lowland.config import TZ_LOCAL

    local = panel.index.tz_convert(TZ_LOCAL)
    out = pd.DataFrame(index=panel.index)
    out["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * local.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * local.dayofweek / 7)
    out["is_weekend"] = (local.dayofweek >= 5).astype(float)
    return out


@dataclass
class SequenceData:
    """Tensors plus the bookkeeping needed to score and plot them."""

    x_past: np.ndarray       # (N, L, n_past)
    x_future: np.ndarray     # (N, H, n_future)
    y: np.ndarray            # (N, H)
    origins: pd.DatetimeIndex
    past_vars: tuple[str, ...]
    future_vars: tuple[str, ...]
    mu: np.ndarray = field(default_factory=lambda: np.zeros(1))
    sigma: np.ndarray = field(default_factory=lambda: np.ones(1))
    y_mu: float = 0.0
    y_sigma: float = 1.0


def build_sequences(
    panel: pd.DataFrame,
    target: str,
    *,
    context: int = 168,
    horizon: int = 48,
    stride: int = 1,
) -> SequenceData:
    """Cut the panel into ``(past, known-future, target)`` windows.

    A window is emitted only when its *past* block is fully observed. The target block is
    allowed to contain gaps, which the masked loss handles -- discarding a whole 48-hour
    window because of one missing hour would throw away a lot of usable supervision.
    """
    from lowland.config import TZ_LOCAL  # noqa: F401  (used indirectly via _cyclical)

    df = panel.copy()
    cyc = _cyclical(df)
    df = pd.concat([df, cyc], axis=1)
    df["target"] = df[target].astype(float)
    if "is_holiday" not in df.columns:
        from lowland.features import _nl_holiday_flags

        df = pd.concat([df, _nl_holiday_flags(df.index)], axis=1)

    past_vars = tuple(v for v in PAST_VARS if v in df.columns)
    future_vars = tuple(v for v in FUTURE_VARS if v in df.columns)

    P = df[list(past_vars)].to_numpy(dtype=np.float32)
    Fu = df[list(future_vars)].to_numpy(dtype=np.float32)
    Y = df["target"].to_numpy(dtype=np.float32)
    n = len(df)

    starts = np.arange(0, n - context - horizon + 1, stride)
    past_ok = np.array([np.isfinite(P[s : s + context]).all() for s in starts])
    fut_ok = np.array([np.isfinite(Fu[s + context : s + context + horizon]).all() for s in starts])
    y_any = np.array([np.isfinite(Y[s + context : s + context + horizon]).any() for s in starts])
    keep = starts[past_ok & fut_ok & y_any]

    x_past = np.stack([P[s : s + context] for s in keep])
    x_future = np.stack([Fu[s + context : s + context + horizon] for s in keep])
    y = np.stack([Y[s + context : s + context + horizon] for s in keep])
    origins = df.index[keep + context - 1]

    log.info(
        "sequences: %s windows (from %s candidates), context=%s horizon=%s",
        len(keep), len(starts), context, horizon,
    )
    return SequenceData(x_past, x_future, y, origins, past_vars, future_vars)


def standardise(
    train: SequenceData, *others: SequenceData
) -> tuple[SequenceData, ...]:
    """Z-score using training statistics only, then apply to every split.

    Computing the statistics on the full sample would leak test-period information into
    training, which is a subtle but real form of look-ahead bias.
    """
    mu = train.x_past.reshape(-1, train.x_past.shape[-1]).mean(axis=0)
    sigma = train.x_past.reshape(-1, train.x_past.shape[-1]).std(axis=0) + 1e-6
    fut_mu = train.x_future.reshape(-1, train.x_future.shape[-1]).mean(axis=0)
    fut_sigma = train.x_future.reshape(-1, train.x_future.shape[-1]).std(axis=0) + 1e-6
    y_mu = float(np.nanmean(train.y))
    y_sigma = float(np.nanstd(train.y) + 1e-6)

    def apply(sd: SequenceData) -> SequenceData:
        return SequenceData(
            x_past=(sd.x_past - mu) / sigma,
            x_future=(sd.x_future - fut_mu) / fut_sigma,
            y=(sd.y - y_mu) / y_sigma,
            origins=sd.origins,
            past_vars=sd.past_vars,
            future_vars=sd.future_vars,
            mu=mu, sigma=sigma, y_mu=y_mu, y_sigma=y_sigma,
        )

    return tuple(apply(sd) for sd in (train, *others))


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


@dataclass
class DeepQuantileForecaster:
    """Training / inference wrapper around :class:`TemporalQuantileNet`."""

    cfg: DeepConfig = field(default_factory=DeepConfig)
    horizon: int = 48
    device: str = field(default_factory=lambda: resolve_device())
    model_: TemporalQuantileNet | None = None
    history_: list[dict[str, float]] = field(default_factory=list)
    y_mu_: float = 0.0
    y_sigma_: float = 1.0

    def fit(self, train: SequenceData, valid: SequenceData) -> DeepQuantileForecaster:
        set_seed(RANDOM_SEED)
        self.y_mu_, self.y_sigma_ = train.y_mu, train.y_sigma

        model = TemporalQuantileNet(
            n_past_vars=train.x_past.shape[-1],
            n_future_vars=train.x_future.shape[-1],
            horizon=self.horizon,
            cfg=self.cfg,
        ).to(self.device)
        self.model_ = model

        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        steps = max(1, len(train.y) // self.cfg.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.cfg.lr, total_steps=self.cfg.max_epochs * steps, pct_start=0.25
        )

        dl_tr = DataLoader(
            TensorDataset(
                torch.from_numpy(train.x_past), torch.from_numpy(train.x_future),
                torch.from_numpy(train.y),
            ),
            batch_size=self.cfg.batch_size, shuffle=True, drop_last=True,
        )
        dl_va = DataLoader(
            TensorDataset(
                torch.from_numpy(valid.x_past), torch.from_numpy(valid.x_future),
                torch.from_numpy(valid.y),
            ),
            batch_size=self.cfg.batch_size * 2, shuffle=False,
        )

        best = float("inf")
        best_state: dict | None = None
        bad_epochs = 0

        for epoch in range(self.cfg.max_epochs):
            model.train()
            tr_loss = 0.0
            for xp, xf, yy in dl_tr:
                xp, xf, yy = xp.to(self.device), xf.to(self.device), yy.to(self.device)
                opt.zero_grad(set_to_none=True)
                pred = model(xp, xf)
                loss = pinball_loss_torch(yy, pred, self.cfg.quantiles)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                opt.step()
                sched.step()
                tr_loss += loss.item()
            tr_loss /= max(1, len(dl_tr))

            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xp, xf, yy in dl_va:
                    xp, xf, yy = xp.to(self.device), xf.to(self.device), yy.to(self.device)
                    va_loss += pinball_loss_torch(yy, model(xp, xf), self.cfg.quantiles).item()
            va_loss /= max(1, len(dl_va))

            self.history_.append({"epoch": epoch, "train": tr_loss, "valid": va_loss})
            if va_loss < best - 1e-5:
                best, bad_epochs = va_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
            if epoch % 5 == 0 or bad_epochs >= self.cfg.patience:
                log.info("epoch %2d  train=%.4f  valid=%.4f  (best=%.4f)", epoch, tr_loss, va_loss, best)
            if bad_epochs >= self.cfg.patience:
                log.info("early stopping at epoch %s", epoch)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return self

    def predict_quantiles(self, data: SequenceData) -> np.ndarray:
        """Return ``(N, H, Q)`` quantile predictions on the original scale."""
        assert self.model_ is not None, "call fit() first"
        self.model_.eval()
        outs = []
        dl = DataLoader(
            TensorDataset(torch.from_numpy(data.x_past), torch.from_numpy(data.x_future)),
            batch_size=self.cfg.batch_size * 2, shuffle=False,
        )
        with torch.no_grad():
            for xp, xf in dl:
                p = self.model_(xp.to(self.device), xf.to(self.device))
                outs.append(p.cpu().numpy())
        pred = np.concatenate(outs, axis=0)
        pred = pred * self.y_sigma_ + self.y_mu_
        return np.sort(pred, axis=-1)

    def interpretation(self, data: SequenceData, n_batches: int = 4) -> dict[str, np.ndarray]:
        """Average variable-selection and attention weights over a sample of windows."""
        assert self.model_ is not None, "call fit() first"
        self.model_.eval()
        dl = DataLoader(
            TensorDataset(torch.from_numpy(data.x_past), torch.from_numpy(data.x_future)),
            batch_size=self.cfg.batch_size, shuffle=False,
        )
        past_w, fut_w, attn = [], [], []
        with torch.no_grad():
            for i, (xp, xf) in enumerate(dl):
                if i >= n_batches:
                    break
                self.model_(xp.to(self.device), xf.to(self.device))
                w = self.model_.last_weights_
                past_w.append(w["past_selection"].mean(dim=(0, 1)).cpu().numpy())
                fut_w.append(w["future_selection"].mean(dim=(0, 1)).cpu().numpy())
                attn.append(w["attention"].mean(dim=0).cpu().numpy())
        return {
            "past_selection": np.mean(past_w, axis=0),
            "future_selection": np.mean(fut_w, axis=0),
            "attention": np.mean(attn, axis=0),
            "past_vars": np.array(data.past_vars),
            "future_vars": np.array(data.future_vars),
        }

    def n_parameters(self) -> int:
        assert self.model_ is not None
        return sum(p.numel() for p in self.model_.parameters())
