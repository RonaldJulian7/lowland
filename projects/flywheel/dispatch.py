"""Battery dispatch: MILP and model-predictive control.

Four controllers, forming a ladder so the value of a forecast is measurable rather than
asserted: perfect foresight (MILP on realised prices -- unachievable, but it bounds
everything else, and % of optimum compares across calm and volatile weeks in a way that
euros don't), deterministic MPC on the point forecast, stochastic MPC on scenarios drawn
from the predictive distribution, and a no-forecast threshold rule as the thing any of
this has to beat.

Two modelling choices worth flagging:

The on/off binaries stay. With strictly positive prices the relaxation is exact and much
faster, but NL prices are negative for several hundred hours a year, and there it
genuinely pays to burn energy through the round-trip loss -- so the relaxation breaks
exactly where a battery makes its money.

Degradation is priced (EUR/MWh of throughput) rather than constrained away, so the
optimiser cycles only when the spread justifies it. The cycle cap then rarely binds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

from lowland.config import DEFAULT_BATTERY, BatterySpec
from lowland.utils import get_logger

log = get_logger(__name__)


@dataclass
class DispatchResult:
    """Outcome of one dispatch run."""

    schedule: pd.DataFrame
    profit_eur: float
    energy_discharged_mwh: float
    equivalent_full_cycles: float
    congestion_penalty_eur: float
    solver_status: str
    solve_seconds: float = 0.0

    def summary(self) -> dict[str, float | str]:
        return {
            "profit_eur": round(self.profit_eur, 2),
            "energy_discharged_mwh": round(self.energy_discharged_mwh, 1),
            "equivalent_full_cycles": round(self.equivalent_full_cycles, 2),
            "congestion_penalty_eur": round(self.congestion_penalty_eur, 2),
            "status": self.solver_status,
        }


# --------------------------------------------------------------------------------------
# Core MILP
# --------------------------------------------------------------------------------------


def _build_milp(
    prices: np.ndarray,
    spec: BatterySpec,
    *,
    soc_init: float,
    dt_h: float = 1.0,
    congestion_weight: np.ndarray | None = None,
    congestion_lambda: float = 0.0,
    terminal_soc_value: float | None = None,
    cycle_limit: float | None = None,
    name: str = "dispatch",
) -> tuple[pulp.LpProblem, dict]:
    """Assemble the dispatch MILP over a horizon of ``len(prices)`` steps.

    ``terminal_soc_value`` prices the energy left in the battery at the end of the
    horizon. Without it, a rolling-horizon controller always empties the battery on the
    last step of every window -- a pure artefact of the truncation, and one of the most
    common bugs in MPC implementations. Valuing the residual energy at the mean price of
    the window removes it.
    """
    T = len(prices)
    prob = pulp.LpProblem(name, pulp.LpMaximize)

    c = [pulp.LpVariable(f"c_{t}", lowBound=0, upBound=spec.power_mw) for t in range(T)]
    d = [pulp.LpVariable(f"d_{t}", lowBound=0, upBound=spec.power_mw) for t in range(T)]
    soc = [pulp.LpVariable(f"soc_{t}", lowBound=spec.soc_min, upBound=spec.soc_max) for t in range(T)]
    on = [pulp.LpVariable(f"on_{t}", cat="Binary") for t in range(T)]

    for t in range(T):
        # Mutual exclusion: on_t = 1 permits discharge, 0 permits charge.
        prob += d[t] <= spec.power_mw * on[t]
        prob += c[t] <= spec.power_mw * (1 - on[t])

        prev = soc_init if t == 0 else soc[t - 1]
        prob += soc[t] == prev + spec.eta_charge * c[t] * dt_h - (d[t] / spec.eta_discharge) * dt_h

    if cycle_limit is not None:
        prob += pulp.lpSum(d) * dt_h <= cycle_limit * spec.energy_mwh

    revenue = pulp.lpSum(prices[t] * (d[t] - c[t]) * dt_h for t in range(T))
    degradation = pulp.lpSum(
        spec.degradation_cost_eur_per_mwh * (d[t] + c[t]) * dt_h for t in range(T)
    )

    objective = revenue - degradation

    if congestion_weight is not None and congestion_lambda > 0:
        # Charging during system stress makes congestion worse; discharging relieves it.
        # The signed term therefore rewards the battery for behaving as a grid asset, not
        # merely as an arbitrageur.
        congestion = pulp.lpSum(
            congestion_lambda * congestion_weight[t] * (c[t] - d[t]) * dt_h for t in range(T)
        )
        objective = objective - congestion

    if terminal_soc_value is not None:
        objective = objective + terminal_soc_value * soc[T - 1]

    prob += objective
    return prob, {"c": c, "d": d, "soc": soc, "on": on}


def _extract(
    vars_: dict, prices: np.ndarray, spec: BatterySpec, index: pd.DatetimeIndex,
    congestion_weight: np.ndarray | None, congestion_lambda: float, dt_h: float,
) -> tuple[pd.DataFrame, float, float]:
    c = np.array([v.value() or 0.0 for v in vars_["c"]])
    d = np.array([v.value() or 0.0 for v in vars_["d"]])
    soc = np.array([v.value() or 0.0 for v in vars_["soc"]])

    gross = prices * (d - c) * dt_h
    deg = spec.degradation_cost_eur_per_mwh * (d + c) * dt_h
    pen = (
        congestion_lambda * congestion_weight * (c - d) * dt_h
        if congestion_weight is not None and congestion_lambda > 0
        else np.zeros_like(c)
    )
    sched = pd.DataFrame(
        {
            "price": prices, "charge_mw": c, "discharge_mw": d, "soc_mwh": soc,
            "net_mw": d - c, "gross_eur": gross, "degradation_eur": deg,
            "congestion_eur": pen, "profit_eur": gross - deg,
        },
        index=index,
    )
    return sched, float(gross.sum() - deg.sum()), float(pen.sum())


def solve_perfect_foresight(
    prices: pd.Series,
    spec: BatterySpec = DEFAULT_BATTERY,
    *,
    congestion_weight: pd.Series | None = None,
    congestion_lambda: float = 0.0,
    dt_h: float = 1.0,
    daily_cycle_limit: bool = True,
    msg: bool = False,
) -> DispatchResult:
    """Optimal dispatch given the realised price path: the theoretical ceiling."""
    import time

    t0 = time.time()
    p = prices.to_numpy(dtype=float)
    cw = congestion_weight.to_numpy(dtype=float) if congestion_weight is not None else None
    cycles = spec.max_cycles_per_day * (len(p) * dt_h / 24.0) if daily_cycle_limit else None

    prob, v = _build_milp(
        p, spec, soc_init=spec.soc_init, dt_h=dt_h,
        congestion_weight=cw, congestion_lambda=congestion_lambda,
        cycle_limit=cycles, name="perfect_foresight",
    )
    prob.solve(pulp.PULP_CBC_CMD(msg=msg))
    status = pulp.LpStatus[prob.status]

    sched, profit, pen = _extract(v, p, spec, prices.index, cw, congestion_lambda, dt_h)
    discharged = float(sched["discharge_mw"].sum() * dt_h)
    return DispatchResult(
        schedule=sched, profit_eur=profit, energy_discharged_mwh=discharged,
        equivalent_full_cycles=discharged / spec.energy_mwh,
        congestion_penalty_eur=pen, solver_status=status,
        solve_seconds=round(time.time() - t0, 2),
    )


# --------------------------------------------------------------------------------------
# Model-predictive control
# --------------------------------------------------------------------------------------


def _rolling_mpc(
    realised: pd.Series,
    forecast_fn,
    spec: BatterySpec,
    *,
    horizon: int,
    commit: int,
    dt_h: float,
    congestion_weight: pd.Series | None,
    congestion_lambda: float,
    scenario_mode: bool,
    n_scenarios: int,
    msg: bool,
    label: str,
) -> DispatchResult:
    """Shared rolling-horizon loop for the deterministic and stochastic controllers.

    At each step the controller optimises over ``horizon`` hours using only forecast
    information, commits the first ``commit`` hours, and advances. Profit is always
    evaluated against the *realised* price, never the forecast -- scoring a controller on
    its own forecast is the classic way to manufacture an impressive backtest.
    """
    import time

    t0 = time.time()
    n = len(realised)
    prices_real = realised.to_numpy(dtype=float)
    cw_all = congestion_weight.to_numpy(dtype=float) if congestion_weight is not None else None

    soc = spec.soc_init
    rows = []

    for start in range(0, n, commit):
        end = min(start + horizon, n)
        if end <= start:
            break
        window = slice(start, end)
        fc = forecast_fn(start, end)  # (H,) or (S, H)
        cw = cw_all[window] if cw_all is not None else None
        H = end - start

        if scenario_mode:
            scenarios = np.atleast_2d(fc)[:n_scenarios]
            prob, v = _build_scenario_milp(
                scenarios, spec, soc_init=soc, dt_h=dt_h,
                congestion_weight=cw, congestion_lambda=congestion_lambda,
                terminal_soc_value=float(np.mean(scenarios)) * 0.85,
            )
        else:
            point = np.asarray(fc, dtype=float).reshape(-1)[:H]
            prob, v = _build_milp(
                point, spec, soc_init=soc, dt_h=dt_h,
                congestion_weight=cw, congestion_lambda=congestion_lambda,
                # Value leftover energy slightly below the window mean, so the controller
                # neither dumps nor hoards at the horizon boundary.
                terminal_soc_value=float(np.nanmean(point)) * 0.85,
                name=f"mpc_{start}",
            )

        prob.solve(pulp.PULP_CBC_CMD(msg=msg))

        n_commit = min(commit, H)
        for k in range(n_commit):
            c_k = v["c"][k].value() or 0.0
            d_k = v["d"][k].value() or 0.0
            soc = soc + spec.eta_charge * c_k * dt_h - (d_k / spec.eta_discharge) * dt_h
            soc = float(np.clip(soc, spec.soc_min, spec.soc_max))
            t = start + k
            price_t = prices_real[t]
            gross = price_t * (d_k - c_k) * dt_h
            deg = spec.degradation_cost_eur_per_mwh * (d_k + c_k) * dt_h
            pen = (
                congestion_lambda * cw_all[t] * (c_k - d_k) * dt_h
                if cw_all is not None and congestion_lambda > 0 else 0.0
            )
            rows.append(
                {
                    "timestamp": realised.index[t], "price": price_t,
                    "charge_mw": c_k, "discharge_mw": d_k, "soc_mwh": soc,
                    "net_mw": d_k - c_k, "gross_eur": gross, "degradation_eur": deg,
                    "congestion_eur": pen, "profit_eur": gross - deg,
                }
            )

    sched = pd.DataFrame(rows).set_index("timestamp")
    discharged = float(sched["discharge_mw"].sum() * dt_h)
    return DispatchResult(
        schedule=sched,
        profit_eur=float(sched["profit_eur"].sum()),
        energy_discharged_mwh=discharged,
        equivalent_full_cycles=discharged / spec.energy_mwh,
        congestion_penalty_eur=float(sched["congestion_eur"].sum()),
        solver_status=label,
        solve_seconds=round(time.time() - t0, 2),
    )


def _build_scenario_milp(
    scenarios: np.ndarray,
    spec: BatterySpec,
    *,
    soc_init: float,
    dt_h: float = 1.0,
    congestion_weight: np.ndarray | None = None,
    congestion_lambda: float = 0.0,
    terminal_soc_value: float | None = None,
    risk_aversion: float = 0.0,
) -> tuple[pulp.LpProblem, dict]:
    """Two-stage stochastic MILP with non-anticipativity on the first period.

    ``scenarios`` is ``(S, H)``. The first-period decision is a single variable shared by
    all scenarios -- the controller must commit before learning which scenario
    materialises -- while later periods are scenario-dependent recourse. This is what
    distinguishes genuine stochastic optimisation from optimising the mean path: the
    former hedges, the latter does not.

    ``risk_aversion`` blends expected profit with a CVaR term on the worst scenarios,
    implemented by the Rockafellar-Uryasev linearisation so the problem stays a MILP.
    """
    S, H = scenarios.shape
    prob = pulp.LpProblem("stochastic_dispatch", pulp.LpMaximize)

    # First-stage (here-and-now) decision.
    c0 = pulp.LpVariable("c0", lowBound=0, upBound=spec.power_mw)
    d0 = pulp.LpVariable("d0", lowBound=0, upBound=spec.power_mw)
    on0 = pulp.LpVariable("on0", cat="Binary")
    prob += d0 <= spec.power_mw * on0
    prob += c0 <= spec.power_mw * (1 - on0)

    soc_after_0 = soc_init + spec.eta_charge * c0 * dt_h - (d0 / spec.eta_discharge) * dt_h
    prob += soc_after_0 >= spec.soc_min
    prob += soc_after_0 <= spec.soc_max

    scen_profit = []
    all_c: list = [c0]
    all_d: list = [d0]

    for s in range(S):
        c = [c0] + [pulp.LpVariable(f"c_{s}_{t}", lowBound=0, upBound=spec.power_mw) for t in range(1, H)]
        d = [d0] + [pulp.LpVariable(f"d_{s}_{t}", lowBound=0, upBound=spec.power_mw) for t in range(1, H)]
        on = [on0] + [pulp.LpVariable(f"on_{s}_{t}", cat="Binary") for t in range(1, H)]
        soc = [pulp.LpVariable(f"soc_{s}_{t}", lowBound=spec.soc_min, upBound=spec.soc_max) for t in range(H)]

        for t in range(H):
            if t > 0:
                prob += d[t] <= spec.power_mw * on[t]
                prob += c[t] <= spec.power_mw * (1 - on[t])
            prev = soc_init if t == 0 else soc[t - 1]
            prob += soc[t] == prev + spec.eta_charge * c[t] * dt_h - (d[t] / spec.eta_discharge) * dt_h

        p = scenarios[s]
        rev = pulp.lpSum(p[t] * (d[t] - c[t]) * dt_h for t in range(H))
        deg = pulp.lpSum(spec.degradation_cost_eur_per_mwh * (d[t] + c[t]) * dt_h for t in range(H))
        expr = rev - deg
        if congestion_weight is not None and congestion_lambda > 0:
            expr = expr - pulp.lpSum(
                congestion_lambda * congestion_weight[t] * (c[t] - d[t]) * dt_h for t in range(H)
            )
        if terminal_soc_value is not None:
            expr = expr + terminal_soc_value * soc[H - 1]
        scen_profit.append(expr)
        if s == 0:
            all_c, all_d = c, d

    expected = pulp.lpSum(scen_profit) / S

    if risk_aversion > 0:
        # Rockafellar-Uryasev CVaR at 20%: maximise (1-k)*E[profit] + k*CVaR.
        beta = 0.2
        eta = pulp.LpVariable("cvar_eta")
        u = [pulp.LpVariable(f"cvar_u_{s}", lowBound=0) for s in range(S)]
        for s in range(S):
            prob += u[s] >= eta - scen_profit[s]
        cvar = eta - (1.0 / (beta * S)) * pulp.lpSum(u)
        prob += (1 - risk_aversion) * expected + risk_aversion * cvar
    else:
        prob += expected

    return prob, {"c": all_c, "d": all_d, "soc": [], "on": []}


def deterministic_mpc(
    realised: pd.Series,
    point_forecast: pd.Series,
    spec: BatterySpec = DEFAULT_BATTERY,
    *,
    horizon: int = 36,
    commit: int = 24,
    dt_h: float = 1.0,
    congestion_weight: pd.Series | None = None,
    congestion_lambda: float = 0.0,
    msg: bool = False,
) -> DispatchResult:
    """Rolling-horizon control using the median forecast only."""
    fc = point_forecast.to_numpy(dtype=float)

    def forecast_fn(start: int, end: int) -> np.ndarray:
        return fc[start:end]

    return _rolling_mpc(
        realised, forecast_fn, spec, horizon=horizon, commit=commit, dt_h=dt_h,
        congestion_weight=congestion_weight, congestion_lambda=congestion_lambda,
        scenario_mode=False, n_scenarios=1, msg=msg, label="deterministic_mpc",
    )


def stochastic_mpc(
    realised: pd.Series,
    scenario_fn,
    spec: BatterySpec = DEFAULT_BATTERY,
    *,
    horizon: int = 36,
    commit: int = 24,
    n_scenarios: int = 12,
    dt_h: float = 1.0,
    congestion_weight: pd.Series | None = None,
    congestion_lambda: float = 0.0,
    msg: bool = False,
) -> DispatchResult:
    """Rolling-horizon control over a scenario ensemble from the predictive distribution."""
    return _rolling_mpc(
        realised, scenario_fn, spec, horizon=horizon, commit=commit, dt_h=dt_h,
        congestion_weight=congestion_weight, congestion_lambda=congestion_lambda,
        scenario_mode=True, n_scenarios=n_scenarios, msg=msg, label="stochastic_mpc",
    )


# --------------------------------------------------------------------------------------
# Heuristic
# --------------------------------------------------------------------------------------


@dataclass
class ThresholdController:
    """Charge below a trailing low percentile, discharge above a trailing high one.

    The honest incumbent. It uses no forecast at all, only a rolling window of realised
    prices, and on calm weeks it is surprisingly hard to beat.
    """

    spec: BatterySpec = DEFAULT_BATTERY
    window: int = 168
    low_pct: float = 30.0
    high_pct: float = 70.0

    def run(self, prices: pd.Series, dt_h: float = 1.0) -> DispatchResult:
        p = prices.to_numpy(dtype=float)
        soc = self.spec.soc_init
        rows = []
        for t in range(len(p)):
            lo_i = max(0, t - self.window)
            hist = p[lo_i : t + 1]
            lo = np.nanpercentile(hist, self.low_pct) if hist.size > 24 else p[t]
            hi = np.nanpercentile(hist, self.high_pct) if hist.size > 24 else p[t]

            c = d = 0.0
            if p[t] <= lo and soc < self.spec.soc_max:
                c = min(self.spec.power_mw, (self.spec.soc_max - soc) / (self.spec.eta_charge * dt_h))
            elif p[t] >= hi and soc > self.spec.soc_min:
                d = min(self.spec.power_mw, (soc - self.spec.soc_min) * self.spec.eta_discharge / dt_h)

            soc = float(np.clip(
                soc + self.spec.eta_charge * c * dt_h - (d / self.spec.eta_discharge) * dt_h,
                self.spec.soc_min, self.spec.soc_max,
            ))
            gross = p[t] * (d - c) * dt_h
            deg = self.spec.degradation_cost_eur_per_mwh * (d + c) * dt_h
            rows.append(
                {"timestamp": prices.index[t], "price": p[t], "charge_mw": c, "discharge_mw": d,
                 "soc_mwh": soc, "net_mw": d - c, "gross_eur": gross, "degradation_eur": deg,
                 "congestion_eur": 0.0, "profit_eur": gross - deg}
            )

        sched = pd.DataFrame(rows).set_index("timestamp")
        discharged = float(sched["discharge_mw"].sum() * dt_h)
        return DispatchResult(
            schedule=sched, profit_eur=float(sched["profit_eur"].sum()),
            energy_discharged_mwh=discharged,
            equivalent_full_cycles=discharged / self.spec.energy_mwh,
            congestion_penalty_eur=0.0, solver_status="heuristic",
        )
