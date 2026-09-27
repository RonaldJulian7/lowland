"""Rolling-origin backtest of every forecaster. Produces everything the README quotes.

Three things this is trying to establish: that the models beat *probabilistic* baselines
rather than a point-forecast strawman, that the intervals are calibrated before and after
conformal correction, and that the ranking between models is statistically
distinguishable (Diebold-Mariano on per-observation pinball losses, Newey-West corrected
for the autocorrelation a 24 h horizon necessarily induces).

Writes to artifacts/reports/bellwether_*.{csv,json,parquet}.

    python -m projects.bellwether.train --target residual_load --horizon 24
    python -m projects.bellwether.train --all
    python -m projects.bellwether.train --target price_da --no-deep --folds 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

# The linear quantile baseline is capped at a few hundred IRLS iterations by design; the
# resulting warning is expected and would otherwise drown the run log.
try:
    from statsmodels.tools.sm_exceptions import IterationLimitWarning

    warnings.filterwarnings("ignore", category=IterationLimitWarning)
except ImportError:  # pragma: no cover
    pass

from lowland.backtest import RollingOriginSplitter, calibration_split, summarise_folds
from lowland.conformal import (
    ConformalQuantileRegressor,
    MondrianConformalQuantileRegressor,
    hour_block_groups,
)
from lowland.config import MODELS_DIR, QUANTILES, REPORTS_DIR
from lowland.dataset import congestion_threshold, load_panel
from lowland.features import FeatureSpec, make_supervised, train_mask
from lowland.metrics import (
    diebold_mariano,
    evaluate_probabilistic,
    metrics_frame,
    pinball_loss,
    reliability_curve,
)
from lowland.utils import get_logger, set_seed
from projects.bellwether.baselines import (
    ClimatologyForecaster,
    QuantileARForecaster,
    SeasonalNaiveForecaster,
)
from projects.bellwether.congestion import risk_calibration
from projects.bellwether.gbm import LightGBMQuantileForecaster

log = get_logger("p1.train")


# --------------------------------------------------------------------------------------
# Per-fold fitting
# --------------------------------------------------------------------------------------


def _per_obs_pinball(y: np.ndarray, q: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    """Pinball loss per observation, averaged over quantile levels.

    The Diebold-Mariano test needs a *sequence* of loss differentials, not an aggregate,
    so the reduction over quantiles happens here but the reduction over time does not.
    """
    taus = np.asarray(quantiles).reshape(1, -1)
    d = y.reshape(-1, 1) - q
    return np.mean(np.maximum(taus * d, (taus - 1) * d), axis=1)


def run_fold(
    X: pd.DataFrame,
    y: pd.Series,
    origins: pd.DatetimeIndex,
    fold,
    *,
    quantiles: tuple[float, ...],
    calib_days: int,
    horizon: int,
    fit_deep: bool,
    panel: pd.DataFrame,
    target: str,
    deep_kwargs: dict | None = None,
) -> dict:
    """Fit every model on one fold and return predictions plus diagnostics."""
    fit_mask, calib_mask = calibration_split(
        fold.train_mask, origins, calib_days=calib_days, horizon=horizon
    )
    use_calib = calib_mask.sum() > 0

    X_fit, y_fit = X[fit_mask], y[fit_mask]
    X_cal, y_cal = X[calib_mask], y[calib_mask]
    X_te, y_te = X[fold.test_mask], y[fold.test_mask]
    te_index = X_te.index

    preds: dict[str, np.ndarray] = {}

    # ---- baselines -------------------------------------------------------------------
    for name, model in (
        ("Climatology", ClimatologyForecaster(quantiles)),
        ("SeasonalNaive", SeasonalNaiveForecaster(quantiles)),
        ("QuantileAR", QuantileARForecaster(quantiles)),
    ):
        t0 = time.time()
        model.fit(X_fit, y_fit)
        preds[name] = model.predict_quantiles(X_te)
        log.info("  %-14s fitted in %5.1fs", name, time.time() - t0)

    # ---- LightGBM --------------------------------------------------------------------
    t0 = time.time()
    gbm = LightGBMQuantileForecaster(quantiles=quantiles)
    gbm.fit(X_fit, y_fit)
    preds["LightGBM"] = gbm.predict_quantiles(X_te)
    log.info("  %-14s fitted in %5.1fs", "LightGBM", time.time() - t0)

    # ---- weather ablation -------------------------------------------------------------
    # Same model class, same hyperparameters, weather columns removed. Holding everything
    # but the feature set fixed is what makes the gap attributable to the meteorology
    # rather than to a change of estimator.
    from lowland.features import feature_groups

    groups = feature_groups(list(X.columns))
    weather_cols = set(
        groups.get("weather_temp", []) + groups.get("weather_wind", []) + groups.get("weather_solar", [])
    )
    no_wx = [c for c in X.columns if c not in weather_cols]
    if len(no_wx) < len(X.columns):
        t0 = time.time()
        gbm_nw = LightGBMQuantileForecaster(quantiles=quantiles)
        gbm_nw.fit(X_fit[no_wx], y_fit)
        preds["LightGBM-NoWeather"] = gbm_nw.predict_quantiles(X_te[no_wx])
        log.info("  %-14s fitted in %5.1fs", "LGBM-NoWx", time.time() - t0)

    # ---- conformal calibration of LightGBM --------------------------------------------
    if use_calib:
        q_cal = gbm.predict_quantiles(X_cal)
        cqr = ConformalQuantileRegressor(quantiles).fit(y_cal.to_numpy(), q_cal)
        preds["LightGBM+CQR"] = cqr.transform(preds["LightGBM"])

        mondrian = MondrianConformalQuantileRegressor(quantiles).fit(
            y_cal.to_numpy(), q_cal, hour_block_groups(X_cal.index)
        )
        preds["LightGBM+Mondrian"] = mondrian.transform(
            preds["LightGBM"], hour_block_groups(te_index)
        )

    # ---- deep sequence model ------------------------------------------------------------
    deep_info: dict = {}
    if fit_deep:
        from projects.bellwether.deep import (
            DeepConfig,
            DeepQuantileForecaster,
            build_sequences,
            standardise,
        )

        dk = deep_kwargs or {}
        context = dk.get("context", 168)
        seq_horizon = dk.get("seq_horizon", 48)
        stride = dk.get("stride", 2)
        train_days = dk.get("train_days", 1825)

        t0 = time.time()
        # Build at stride 1 so that *every* test hour gets a prediction and the sequence
        # model is scored on exactly the same sample as the tabular models. Thinning is
        # applied afterwards, to the training windows only: consecutive windows overlap by
        # 167 of 168 hours and are almost duplicates, so subsampling them costs little,
        # whereas subsampling the test set would make the comparison incommensurable.
        # The panel is sliced to just what this fold needs first, because a stride-1 build
        # over the full 11-year history would allocate about a gigabyte of overlapping
        # context windows.
        lo = fold.train_end - pd.Timedelta(days=train_days) - pd.Timedelta(hours=context + 1)
        hi = fold.test_end + pd.Timedelta(hours=seq_horizon + 1)
        panel_fold = panel.loc[(panel.index >= lo) & (panel.index <= hi)]
        seqs = build_sequences(
            panel_fold, target, context=context, horizon=seq_horizon, stride=1
        )
        # Split the windows on origin, matching the tabular folds exactly so the two
        # model families are scored on the same period.
        tr_sel = (seqs.origins <= fold.train_end) & (
            seqs.origins > fold.train_end - pd.Timedelta(days=train_days)
        )
        va_cut = fold.train_end - pd.Timedelta(days=dk.get("valid_days", 120))
        val_sel = tr_sel & (seqs.origins > va_cut)
        fit_sel = tr_sel & (seqs.origins <= va_cut - pd.Timedelta(hours=seq_horizon))
        te_sel = (seqs.origins > fold.test_start) & (seqs.origins <= fold.test_end)

        # Thin the fitting windows only; validation and test stay at full resolution.
        if stride > 1:
            thin = np.zeros(len(seqs.origins), dtype=bool)
            thin[np.where(fit_sel)[0][::stride]] = True
            fit_sel = thin

        def subset(sd, m):
            from projects.bellwether.deep import SequenceData

            return SequenceData(
                sd.x_past[m], sd.x_future[m], sd.y[m], sd.origins[m], sd.past_vars, sd.future_vars
            )

        cal_sel = (seqs.origins > va_cut) & (seqs.origins <= fold.train_end)

        if fit_sel.sum() > 1000 and te_sel.sum() > 10:
            # Standardise every split with the *fitting* split's statistics in a single
            # call. Standardising the calibration block on its own moments would feed the
            # network inputs on a different scale from the ones it was trained on, which
            # silently corrupts the conformal corrections derived from it.
            s_fit, s_val, s_te, s_cal = standardise(
                subset(seqs, fit_sel),
                subset(seqs, val_sel),
                subset(seqs, te_sel),
                subset(seqs, cal_sel),
            )
            cfg = DeepConfig(
                quantiles=quantiles,
                max_epochs=dk.get("max_epochs", 40),
                patience=dk.get("patience", 6),
                batch_size=dk.get("batch_size", 128),
            )
            deep = DeepQuantileForecaster(cfg=cfg, horizon=seq_horizon)
            deep.fit(s_fit, s_val)
            q_all = deep.predict_quantiles(s_te)  # (N, H, Q)

            # Align the sequence model's horizon-h slice onto the tabular test index.
            step = horizon - 1
            pred_index = s_te.origins + pd.Timedelta(hours=horizon)
            aligned = pd.DataFrame(q_all[:, step, :], index=pred_index)
            aligned = aligned.reindex(te_index)
            preds["DeepTFT"] = aligned.to_numpy()

            # Conformalise the deep model on the same calibration block.
            if use_calib:
                if cal_sel.sum() > 200:
                    q_cal_deep = deep.predict_quantiles(s_cal)[:, step, :]
                    cal_idx = s_cal.origins + pd.Timedelta(hours=horizon)
                    y_cal_deep = y.reindex(cal_idx).to_numpy()
                    ok = np.isfinite(y_cal_deep) & np.isfinite(q_cal_deep).all(axis=1)
                    if ok.sum() > 200:
                        cqr_d = ConformalQuantileRegressor(quantiles).fit(
                            y_cal_deep[ok], q_cal_deep[ok]
                        )
                        base = preds["DeepTFT"]
                        fin = np.isfinite(base).all(axis=1)
                        out = base.copy()
                        out[fin] = cqr_d.transform(base[fin])
                        preds["DeepTFT+CQR"] = out

            deep_info = {
                "n_parameters": deep.n_parameters(),
                "epochs_run": len(deep.history_),
                "best_valid": min(h["valid"] for h in deep.history_) if deep.history_ else None,
                "fit_seconds": round(time.time() - t0, 1),
                "n_train_windows": int(fit_sel.sum()),
                "device": deep.device,
            }
            log.info("  %-14s fitted in %5.1fs (%s params)", "DeepTFT",
                     deep_info["fit_seconds"], f"{deep_info['n_parameters']:,}")
        else:
            log.warning("  skipping deep model: insufficient windows in this fold")

    # ---- ensemble -----------------------------------------------------------------------
    members = [k for k in ("LightGBM+CQR", "DeepTFT+CQR") if k in preds]
    if len(members) == 2:
        stack = np.stack([preds[m] for m in members])
        # Vincentisation: averaging quantile *functions* rather than densities. It
        # preserves the shape of the component distributions instead of producing the
        # spuriously wide bimodal mixture that linear pooling gives.
        preds["Ensemble"] = np.nanmean(stack, axis=0)

    return {
        "fold": fold.fold_id,
        "index": te_index,
        "y_true": y_te.to_numpy(),
        "preds": preds,
        "gbm_importance": gbm.grouped_importance().to_dict(),
        "deep": deep_info,
        "n_fit": int(fit_mask.sum()),
        "n_calib": int(calib_mask.sum()),
        "n_test": int(fold.test_mask.sum()),
    }


# --------------------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------------------


def run_experiment(
    target: str = "residual_load",
    horizon: int = 24,
    *,
    n_folds: int = 6,
    test_days: int = 30,
    calib_days: int = 120,
    fit_deep: bool = True,
    deep_kwargs: dict | None = None,
    until: str | None = None,
    label: str = "",
) -> dict:
    """Run the full rolling-origin experiment for one target/horizon pair.

    ``until`` truncates the panel before splitting. The splitter always takes the *most
    recent* windows, so with the full panel every fold lands in whatever season the data
    happens to end in - here, summer. That is a poor test for a system about congestion,
    which is a winter problem: peak residual load, cold snaps, no solar. Cutting the panel
    at, say, 1 March puts the folds in November-February instead, so the same protocol can
    be re-run on the conditions that actually matter.
    """
    set_seed()
    panel = load_panel()
    if until:
        panel = panel.loc[panel.index < pd.Timestamp(until, tz="UTC")]
        log.info("panel truncated at %s -> %s rows", until, len(panel))
    spec = FeatureSpec(
        target=target,
        horizon=horizon,
        extra_lagged=("price_da",) if target != "price_da" else ("residual_load",),
    )
    X, y, origins = make_supervised(panel, spec)

    # Restrict to rows that are usable for *fitting and scoring*. The panel deliberately
    # extends into the future (weather present, target missing) so the apps can produce a
    # live forecast, but including those rows here would hand the splitter a final fold
    # whose test window lies entirely in the unobserved future and therefore scores
    # nothing. Forward inference is handled separately in ``forecast.py``.
    keep = train_mask(X, y, panel) & y.notna()
    X, y, origins = X[keep], y[keep], origins[keep]

    splitter = RollingOriginSplitter(
        n_folds=n_folds, test_days=test_days, horizon=horizon, expanding=True
    )
    folds = list(splitter.split(origins))
    if not folds:
        raise RuntimeError("no folds produced; check the panel length")
    log.info("target=%s horizon=%sh  folds=%s  features=%s", target, horizon, len(folds), X.shape[1])

    fold_results = []
    for fold in folds:
        log.info("fold %s: %s", fold.fold_id, fold)
        fold_results.append(
            run_fold(
                X, y, origins, fold,
                quantiles=QUANTILES, calib_days=calib_days, horizon=horizon,
                fit_deep=fit_deep, panel=panel, target=target, deep_kwargs=deep_kwargs,
            )
        )

    res = _aggregate(fold_results, panel, target, horizon, folds)
    res["label"] = label
    return res


def _aggregate(fold_results: list[dict], panel: pd.DataFrame, target: str, horizon: int, folds) -> dict:
    """Pool predictions across folds and compute the full metric suite."""
    model_names: list[str] = []
    for fr in fold_results:
        for k in fr["preds"]:
            if k not in model_names:
                model_names.append(k)

    pooled: dict[str, dict[str, np.ndarray]] = {}
    for name in model_names:
        ys, qs, idxs = [], [], []
        for fr in fold_results:
            if name not in fr["preds"]:
                continue
            q = fr["preds"][name]
            ok = np.isfinite(fr["y_true"]) & np.isfinite(q).all(axis=1)
            ys.append(fr["y_true"][ok])
            qs.append(q[ok])
            idxs.append(fr["index"][ok])
        if ys:
            pooled[name] = {
                "y": np.concatenate(ys),
                "q": np.concatenate(qs),
                "index": np.concatenate([i.to_numpy() for i in idxs]),
            }

    insample = panel[target].dropna().to_numpy()
    rows, curves = [], {}
    for name, d in pooled.items():
        m = evaluate_probabilistic(d["y"], d["q"], QUANTILES, y_insample=insample, label=name)
        m["label"] = name
        rows.append(m)
        curves[name] = reliability_curve(d["y"], d["q"], QUANTILES)

    table = metrics_frame(rows)

    # ---- significance testing -----------------------------------------------------------
    ref = "SeasonalNaive" if "SeasonalNaive" in pooled else model_names[0]
    best = table["crps"].idxmin()
    dm_rows = []
    for name in pooled:
        if name == ref:
            continue
        # Compare only on the intersection of scored timestamps.
        a, b = pooled[name], pooled[ref]
        common, ia, ib = np.intersect1d(a["index"], b["index"], return_indices=True)
        if common.size < 50:
            continue
        la = _per_obs_pinball(a["y"][ia], a["q"][ia], QUANTILES)
        lb = _per_obs_pinball(b["y"][ib], b["q"][ib], QUANTILES)
        stat, p = diebold_mariano(la, lb, h=horizon)
        dm_rows.append(
            {"model": name, "vs": ref, "dm_stat": stat, "p_value": p,
             "mean_loss_diff": float(np.mean(la - lb))}
        )
        if name != best and best in pooled:
            pass
    dm = pd.DataFrame(dm_rows)

    # Head-to-head between the two strongest models, which is the comparison a reader
    # actually cares about.
    dm_head = {}
    ranked = table["crps"].sort_values().index.tolist()
    if len(ranked) >= 2:
        a, b = pooled[ranked[0]], pooled[ranked[1]]
        common, ia, ib = np.intersect1d(a["index"], b["index"], return_indices=True)
        if common.size >= 50:
            stat, p = diebold_mariano(
                _per_obs_pinball(a["y"][ia], a["q"][ia], QUANTILES),
                _per_obs_pinball(b["y"][ib], b["q"][ib], QUANTILES),
                h=horizon,
            )
            dm_head = {"best": ranked[0], "runner_up": ranked[1], "dm_stat": stat, "p_value": p}

    # ---- congestion-risk calibration (residual load only) --------------------------------
    risk = {}
    if target == "residual_load":
        thr = congestion_threshold(panel)
        for name in ("LightGBM+CQR", "LightGBM", "DeepTFT+CQR", "Ensemble"):
            if name in pooled:
                rc = risk_calibration(pooled[name]["y"], pooled[name]["q"], QUANTILES, thr)
                risk[name] = {
                    "threshold_mw": thr,
                    "brier": rc.attrs.get("brier"),
                    "brier_skill_vs_climatology": rc.attrs.get("brier_skill"),
                    "bins": rc.to_dict(orient="records"),
                }

    # ---- importance ------------------------------------------------------------------------
    imp = pd.DataFrame([fr["gbm_importance"] for fr in fold_results]).mean().sort_values(ascending=False)

    return {
        "target": target,
        "horizon": horizon,
        "metrics": table,
        "reliability": curves,
        "dm_tests": dm,
        "dm_head_to_head": dm_head,
        "risk_calibration": risk,
        "grouped_importance": imp,
        "folds": summarise_folds(folds),
        "deep_info": [fr["deep"] for fr in fold_results if fr.get("deep")],
        "pooled": pooled,
    }


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def save_results(res: dict) -> None:
    """Write metrics, tests and predictions to ``artifacts/reports``."""
    suffix = f"_{res['label']}" if res.get("label") else ""
    tag = f"bellwether_{res['target']}_h{res['horizon']}{suffix}"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    res["metrics"].to_csv(REPORTS_DIR / f"{tag}_metrics.csv")
    res["dm_tests"].to_csv(REPORTS_DIR / f"{tag}_dm.csv", index=False)
    res["grouped_importance"].to_csv(REPORTS_DIR / f"{tag}_importance.csv")
    res["folds"].to_csv(REPORTS_DIR / f"{tag}_folds.csv", index=False)

    for name, curve in res["reliability"].items():
        curve.to_csv(REPORTS_DIR / f"{tag}_reliability_{name.replace('+', '_')}.csv", index=False)

    # Pooled predictions, so the app and the notebook never need to refit anything.
    frames = []
    for name, d in res["pooled"].items():
        f = pd.DataFrame(d["q"], columns=[f"q{q:.2f}" for q in QUANTILES])
        f.insert(0, "y_true", d["y"])
        f.insert(0, "timestamp", d["index"])
        f.insert(0, "model", name)
        frames.append(f)
    if frames:
        pd.concat(frames).to_parquet(REPORTS_DIR / f"{tag}_predictions.parquet", index=False)

    summary = {
        "target": res["target"],
        "horizon": res["horizon"],
        "best_by_crps": res["metrics"]["crps"].idxmin(),
        "metrics": json.loads(res["metrics"].to_json(orient="index")),
        "dm_head_to_head": res["dm_head_to_head"],
        "risk_calibration": res["risk_calibration"],
        "deep_info": res["deep_info"],
        "grouped_importance": res["grouped_importance"].to_dict(),
    }
    (REPORTS_DIR / f"{tag}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    log.info("results written to %s (%s*)", REPORTS_DIR, tag)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="residual_load")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument("--calib-days", type=int, default=120)
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--deep-epochs", type=int, default=40)
    ap.add_argument("--deep-stride", type=int, default=2)
    ap.add_argument("--all", action="store_true", help="run residual_load and price_da")
    ap.add_argument(
        "--until", default=None,
        help="truncate the panel at this date, to place the folds in a chosen season",
    )
    ap.add_argument("--label", default="", help="suffix for the artifact filenames")
    args = ap.parse_args()

    targets = ["residual_load", "price_da"] if args.all else [args.target]
    deep_kwargs = {"max_epochs": args.deep_epochs, "stride": args.deep_stride}

    for tgt in targets:
        log.info("=" * 78)
        log.info("EXPERIMENT  target=%s  horizon=%sh", tgt, args.horizon)
        log.info("=" * 78)
        res = run_experiment(
            target=tgt,
            horizon=args.horizon,
            n_folds=args.folds,
            test_days=args.test_days,
            calib_days=args.calib_days,
            fit_deep=not args.no_deep,
            deep_kwargs=deep_kwargs,
            until=args.until,
            label=args.label,
        )
        save_results(res)

        pd.set_option("display.width", 200)
        print(f"\n===== {tgt} @ h={args.horizon} =====")
        cols = [c for c in ("n", "mae", "rmse", "crps", "pinball", "picp90", "mpiw90", "winkler90")
                if c in res["metrics"].columns]
        print(res["metrics"][cols].to_string())
        if res["dm_head_to_head"]:
            print("\nHead-to-head:", res["dm_head_to_head"])
        print("\nGrouped feature importance:")
        print(res["grouped_importance"].round(4).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
