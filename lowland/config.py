"""Central configuration: paths, physical constants and Dutch power-system parameters.

All tunable constants live here so that experiments are reproducible from a single
source of truth and so the Streamlit apps and the training scripts cannot drift apart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

for _d in (RAW_DIR, CACHE_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Temporal conventions
# --------------------------------------------------------------------------------------

#: Everything is stored in UTC. Dutch calendar effects are derived by converting to
#: Europe/Amsterdam only at feature-construction time. Mixing the two is the single most
#: common source of silent bugs in European power-system modelling (DST creates a 23h and
#: a 25h day each year).
TZ_UTC = "UTC"
TZ_LOCAL = "Europe/Amsterdam"

#: Native resolution of the ENTSO-E-derived feeds republished by energy-charts.info.
NATIVE_FREQ = "15min"

#: Modelling resolution. Day-ahead markets clear hourly, so hourly keeps the target
#: aligned with the decision that actually matters.
MODEL_FREQ = "h"

#: First date with dependable NL coverage in the upstream feed.
HISTORY_START = "2015-01-01"

#: Forecast horizon in hours. 48h covers day-ahead gate closure (12:00 CET for delivery
#: the next day) with slack, which is the operationally relevant window.
HORIZON_H = 48

#: Length of the historical context window fed to the sequence models.
CONTEXT_H = 336  # 14 days


# --------------------------------------------------------------------------------------
# Geography: weather sampling points
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WeatherSite:
    """A point at which we sample reanalysis / forecast weather.

    Attributes
    ----------
    name: Short identifier used as a column prefix.
    lat, lon: WGS84 coordinates.
    kind: ``load`` (demand centre), ``wind_on``, ``wind_off`` or ``solar``.
    weight: Relative importance used when aggregating to a national index. For load
        sites this approximates population share; for wind sites it approximates
        installed-capacity share.
    """

    name: str
    lat: float
    lon: float
    kind: str
    weight: float


#: Demand centres, weighted by approximate share of Dutch population/industry.
LOAD_SITES: tuple[WeatherSite, ...] = (
    WeatherSite("amsterdam", 52.37, 4.89, "load", 0.22),
    WeatherSite("rotterdam", 51.92, 4.48, "load", 0.22),
    WeatherSite("denhaag", 52.08, 4.31, "load", 0.14),
    WeatherSite("utrecht", 52.09, 5.12, "load", 0.13),
    WeatherSite("eindhoven", 51.44, 5.48, "load", 0.15),
    WeatherSite("groningen", 53.22, 6.57, "load", 0.14),
)

#: Offshore wind clusters. Coordinates sit inside the operating/awarded wind farm zones
#: that dominate Dutch offshore capacity.
WIND_OFFSHORE_SITES: tuple[WeatherSite, ...] = (
    WeatherSite("borssele", 51.70, 3.05, "wind_off", 0.34),       # Borssele I-V
    WeatherSite("hollandsekust", 52.55, 4.05, "wind_off", 0.40),  # Hollandse Kust Zuid/Noord
    WeatherSite("gemini", 54.04, 5.96, "wind_off", 0.16),         # Gemini
    WeatherSite("egmond", 52.60, 4.42, "wind_off", 0.10),         # Egmond aan Zee / Prinses Amalia
)

#: Onshore wind clusters, concentrated in Flevoland, the northern provinces and the
#: south-western delta.
WIND_ONSHORE_SITES: tuple[WeatherSite, ...] = (
    WeatherSite("flevoland", 52.52, 5.60, "wind_on", 0.35),
    WeatherSite("eemshaven", 53.44, 6.83, "wind_on", 0.25),
    WeatherSite("zeeland", 51.55, 3.85, "wind_on", 0.20),
    WeatherSite("noordholland", 52.85, 4.80, "wind_on", 0.20),
)

ALL_SITES: tuple[WeatherSite, ...] = LOAD_SITES + WIND_OFFSHORE_SITES + WIND_ONSHORE_SITES


# --------------------------------------------------------------------------------------
# Weather variables
# --------------------------------------------------------------------------------------

#: Hourly variables requested from Open-Meteo. ``wind_speed_100m`` matters far more than
#: the 10 m standard because modern turbine hub heights are 100-150 m.
HOURLY_WEATHER_VARS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
)


# --------------------------------------------------------------------------------------
# Modelling targets and quantiles
# --------------------------------------------------------------------------------------

#: Quantile levels estimated by every probabilistic model. Symmetric around the median so
#: that central prediction intervals (50/80/90%) can be read off directly.
QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

#: Targets, in MW for load-like series and EUR/MWh for price.
TARGETS: tuple[str, ...] = ("residual_load", "load", "price_da")


# --------------------------------------------------------------------------------------
# Battery / flexibility asset defaults (project 2)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BatterySpec:
    """Grid-scale lithium-ion battery parameters.

    Defaults approximate a 2-hour utility-scale system of the kind now being connected in
    the Netherlands (e.g. the projects at Vlissingen and Lelystad), including the
    asymmetric round-trip efficiency and a marginal degradation cost that makes the
    optimiser trade cycles against spread.
    """

    power_mw: float = 25.0
    energy_mwh: float = 50.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    soc_min_frac: float = 0.05
    soc_max_frac: float = 0.95
    soc_init_frac: float = 0.50
    #: EUR per MWh of throughput, representing capacity fade. Sets the minimum spread
    #: worth cycling for.
    degradation_cost_eur_per_mwh: float = 3.0
    #: Maximum equivalent full cycles per day, a common warranty constraint.
    max_cycles_per_day: float = 1.5

    @property
    def soc_min(self) -> float:
        return self.soc_min_frac * self.energy_mwh

    @property
    def soc_max(self) -> float:
        return self.soc_max_frac * self.energy_mwh

    @property
    def soc_init(self) -> float:
        return self.soc_init_frac * self.energy_mwh

    @property
    def round_trip_efficiency(self) -> float:
        return self.eta_charge * self.eta_discharge


DEFAULT_BATTERY = BatterySpec()


# --------------------------------------------------------------------------------------
# Congestion analytics
# --------------------------------------------------------------------------------------

#: Residual load (MW) above which the Dutch system historically relies heavily on
#: dispatchable thermal plant and imports. Calibrated empirically in project 1 as a high
#: percentile of the historical distribution; this is the fallback if calibration data is
#: unavailable.
CONGESTION_THRESHOLD_MW_FALLBACK = 13_000.0

#: Percentile of the historical residual-load distribution used to define "system stress".
CONGESTION_PERCENTILE = 0.90


# --------------------------------------------------------------------------------------
# Emissions factors (tCO2 per MWh electrical output)
# --------------------------------------------------------------------------------------

#: Direct combustion intensities at typical Dutch plant efficiencies. Used by the
#: counterfactual simulator in project 3.
EMISSION_FACTORS_T_PER_MWH: dict[str, float] = {
    "Fossil gas": 0.37,
    "Fossil hard coal": 0.88,
    "Fossil oil": 0.65,
    "Waste": 0.30,
    "Biomass": 0.02,
    "Nuclear": 0.0,
    "Wind offshore": 0.0,
    "Wind onshore": 0.0,
    "Solar": 0.0,
    "Hydro Run-of-River": 0.0,
    "Others": 0.30,
}


# --------------------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------------------

RANDOM_SEED = 20260924

#: Honour CUDA when present but stay runnable on CPU-only machines.
TORCH_DEVICE = os.environ.get("NLEI_DEVICE", "auto")


@dataclass
class HttpConfig:
    """Shared HTTP behaviour for the ingestion clients."""

    timeout_s: float = 60.0
    max_retries: int = 8
    backoff_base_s: float = 4.0
    #: A descriptive agent is good API citizenship and helps upstream operators.
    user_agent: str = "lowland/0.1 (open research portfolio)"
    headers: dict[str, str] = field(default_factory=dict)


HTTP = HttpConfig()
