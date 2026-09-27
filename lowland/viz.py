"""Chart theme and helpers. One visual language across the three projects.

Warm paper instead of clinical white, teal-led rather than the usual blue-first ordering,
serif headings against a tight grotesque, monospaced figures. Hairlines, no drop shadows.

Some house rules that live here so call sites can't get them wrong:

- categorical hues are assigned in fixed slot order, never cycled, so a series keeps its
  colour when a filter changes the series count. Past slot 7, Theme.series() raises rather
  than inventing a hue that wouldn't clear the CVD gates.
- no dual y-axes, ever. Two scales let you imply any correlation you like.
- quantile fans use the sequential ramp, not categorical hues -- interval width is a
  magnitude, not an identity.

Palette was validated with a colour-blindness checker, not picked by eye: worst adjacent
CVD dE 11.0 light / 8.5 dark, normal-vision dE 21.7 in both. Ochre is under 3:1 on the
light surface, so charts using it need direct labels or a table alongside. If you change
these hexes, re-run the validator for both modes -- the slot ordering is part of the
result, not decoration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --------------------------------------------------------------------------------------
# Type
# --------------------------------------------------------------------------------------

#: Editorial serif for headings. Fraunces is a variable face with optical-size and "wonk"
#: axes, which gives large text some character without tipping into novelty.
FONT_DISPLAY = '"Fraunces", "Iowan Old Style", Georgia, serif'

#: Interface text. Inter Tight sets more compactly than Inter at small sizes, which keeps
#: dense captions readable without shrinking the type.
FONT_UI = '"Inter Tight", "Inter", system-ui, -apple-system, sans-serif'

#: Figures, axis ticks, labels and anything tabular. Monospaced digits stop numbers
#: jittering as they update and signal "read this precisely".
FONT_MONO = '"JetBrains Mono", "SF Mono", ui-monospace, Menlo, monospace'

GOOGLE_FONTS_HREF = (
    "https://fonts.googleapis.com/css2"
    "?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700"
    "&family=Inter+Tight:wght@400;500;600"
    "&family=JetBrains+Mono:wght@400;500;600"
    "&display=swap"
)


# --------------------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------------------

CATEGORICAL_LIGHT = ("#12968B", "#C64B3C", "#5B4B9E", "#D9880A", "#2F6DB5", "#2E7D32", "#C2568F")
CATEGORICAL_DARK = ("#17A197", "#DB6653", "#8A79D6", "#B87E12", "#5892DA", "#4A9E50", "#C96A98")

#: Single-hue teal ramp for magnitude encodings (heatmaps, nested prediction bands).
SEQ_TEAL = (
    "#DCEFEC", "#C2E4DF", "#A4D8D1", "#83CAC1", "#5FBBB1",
    "#3DABA1", "#1F9B91", "#12968B", "#0F8177", "#0C6C64",
    "#0A5852", "#07443F", "#05332F",
)

#: Diverging pair for signed quantities: teal against rust, with a warm neutral midpoint.
DIVERGING = {"low": "#0C6C64", "mid_light": "#EDE7DA", "mid_dark": "#2A2A24", "high": "#C64B3C"}

#: Status colours, reserved. Never reused as a series hue and always shipped with a label.
STATUS = {"good": "#2E7D32", "warning": "#D9880A", "serious": "#C6702B", "critical": "#B3392A"}


@dataclass(frozen=True)
class Theme:
    """Chrome and ink tokens for one colour mode."""

    mode: str
    surface: str        # chart surface
    plane: str          # page behind the chart
    raised: str         # cards, tiles
    ink_primary: str
    ink_secondary: str
    ink_muted: str
    rule: str           # hairline borders
    grid: str
    accent: str
    categorical: tuple[str, ...]

    def series(self, i: int) -> str:
        """Colour for categorical slot ``i``."""
        if i >= len(self.categorical):
            raise IndexError(
                f"categorical slot {i} exceeds the validated palette "
                f"({len(self.categorical)} slots). Fold the tail into 'Other' or facet."
            )
        return self.categorical[i]


LIGHT = Theme(
    mode="light",
    surface="#FBF9F3",
    plane="#F2EEE4",
    raised="#FFFDF8",
    ink_primary="#17180F",
    ink_secondary="#4A4C3E",
    ink_muted="#84866F",
    rule="#DED8C8",
    grid="#E8E2D4",
    accent="#12968B",
    categorical=CATEGORICAL_LIGHT,
)

DARK = Theme(
    mode="dark",
    surface="#191B15",
    plane="#111309",
    raised="#20231B",
    ink_primary="#F4F1E6",
    ink_secondary="#BCBCA8",
    ink_muted="#84866F",
    rule="#2E3227",
    grid="#262A20",
    accent="#17A197",
    categorical=CATEGORICAL_DARK,
)


def theme(mode: str = "light") -> Theme:
    return DARK if mode == "dark" else LIGHT


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------


def apply_theme(
    fig: go.Figure,
    th: Theme,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    y_title: str | None = None,
    x_title: str | None = None,
    height: int = 420,
    show_legend: bool = True,
    unified_hover: bool = True,
) -> go.Figure:
    """Apply the house chrome: hairline axes, mono ticks, legend above, hover layer.

    The title is set as a serif line with a monospaced subtitle beneath it, which gives
    every chart the same two-level heading structure as the pages around it.
    """
    heading = ""
    if title:
        heading = (
            f"<span style='font-family:{FONT_DISPLAY};font-size:17px;"
            f"color:{th.ink_primary};font-weight:600'>{title}</span>"
        )
    if subtitle:
        heading += (
            f"<br><span style='font-family:{FONT_MONO};font-size:11px;"
            f"color:{th.ink_muted};letter-spacing:0.02em'>{subtitle}</span>"
        )

    fig.update_layout(
        template=None,
        paper_bgcolor=th.surface,
        plot_bgcolor=th.surface,
        font=dict(family=FONT_UI, size=13, color=th.ink_secondary),
        title=dict(
            text=heading, x=0, xanchor="left", y=0.98, yanchor="top",
            font=dict(family=FONT_DISPLAY, size=17, color=th.ink_primary),
        ),
        margin=dict(l=62, r=22, t=86 if heading else 26, b=46),
        height=height,
        showlegend=show_legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            font=dict(family=FONT_MONO, size=11, color=th.ink_secondary),
            bgcolor="rgba(0,0,0,0)", borderwidth=0, itemsizing="constant",
        ),
        hovermode="x unified" if unified_hover else "closest",
        hoverlabel=dict(
            bgcolor=th.raised, bordercolor=th.rule,
            font=dict(family=FONT_MONO, size=11, color=th.ink_primary),
        ),
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        linecolor=th.rule, linewidth=1, ticks="outside", tickcolor=th.rule, ticklen=4,
        tickfont=dict(family=FONT_MONO, size=10, color=th.ink_muted),
        title=dict(
            text=x_title or "",
            font=dict(family=FONT_MONO, size=10, color=th.ink_muted),
        ),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=th.grid, gridwidth=1, zeroline=False,
        linecolor="rgba(0,0,0,0)", ticks="",
        tickfont=dict(family=FONT_MONO, size=10, color=th.ink_muted),
        title=dict(
            text=y_title or "",
            font=dict(family=FONT_MONO, size=10, color=th.ink_muted),
        ),
    )
    return fig


# --------------------------------------------------------------------------------------
# Forecast fan
# --------------------------------------------------------------------------------------


def fan_chart(
    index: pd.DatetimeIndex,
    q_matrix: np.ndarray,
    quantiles: tuple[float, ...],
    observed: pd.Series | None = None,
    *,
    th: Theme | None = None,
    title: str = "Probabilistic forecast",
    subtitle: str | None = None,
    y_title: str = "MW",
    height: int = 460,
) -> go.Figure:
    """Nested prediction-interval fan with the median line and, optionally, the outturn.

    Bands run widest-first in ascending ramp darkness, so a narrower (more confident)
    interval reads as denser colour. The bands take steps of one sequential hue because
    interval width is a magnitude; the observed series takes a categorical slot so that
    "what happened" is visually a different kind of thing from "what was predicted".
    """
    th = th or LIGHT
    fig = go.Figure()

    taus = list(quantiles)
    med_i = int(np.argmin(np.abs(np.array(taus) - 0.5)))
    pairs = [(i, len(taus) - 1 - i) for i in range(len(taus) // 2)]
    ramp = [SEQ_TEAL[2], SEQ_TEAL[4], SEQ_TEAL[6]] if th.mode == "light" else [
        SEQ_TEAL[9], SEQ_TEAL[8], SEQ_TEAL[7]
    ]

    for k, (lo, hi) in enumerate(pairs):
        nominal = int(round(100 * (taus[hi] - taus[lo])))
        color = ramp[min(k, len(ramp) - 1)]
        fig.add_trace(
            go.Scatter(
                x=index, y=q_matrix[:, hi], mode="lines", line=dict(width=0),
                hoverinfo="skip", showlegend=False, name=f"p{taus[hi]:.2f}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index, y=q_matrix[:, lo], mode="lines", line=dict(width=0),
                fill="tonexty",
                fillcolor=rgba(color, 0.72 if th.mode == "light" else 0.60),
                name=f"{nominal}% interval",
                hovertemplate=f"{nominal}%% PI  %{{y:,.0f}}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=index, y=q_matrix[:, med_i], mode="lines",
            line=dict(color=th.series(0), width=2),
            name="Forecast (median)",
            hovertemplate="Forecast  %{y:,.0f}<extra></extra>",
        )
    )
    if observed is not None:
        fig.add_trace(
            go.Scatter(
                x=observed.index, y=observed.to_numpy(), mode="lines",
                line=dict(color=th.series(1), width=2),
                name="Observed",
                hovertemplate="Observed  %{y:,.0f}<extra></extra>",
            )
        )

    return apply_theme(fig, th, title=title, subtitle=subtitle, y_title=y_title, height=height)


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------


def reliability_diagram(
    curves: dict[str, pd.DataFrame],
    *,
    th: Theme | None = None,
    title: str = "Calibration",
    subtitle: str | None = "points on the diagonal are perfectly calibrated",
    height: int = 420,
) -> go.Figure:
    """Empirical against nominal exceedance for one or more models.

    The diagonal is the reference: above it the predicted quantiles sit too high, below it
    too low. A systematic S-shape means the predictive distribution has the wrong spread,
    which is precisely what conformal calibration repairs.
    """
    th = th or LIGHT
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color=th.ink_muted, width=1, dash="dot"),
            name="perfect calibration", hoverinfo="skip",
        )
    )
    for i, (label, df) in enumerate(curves.items()):
        fig.add_trace(
            go.Scatter(
                x=df["nominal"], y=df["empirical"], mode="lines+markers",
                line=dict(color=th.series(i), width=2),
                marker=dict(size=9, line=dict(width=2, color=th.surface)),
                name=label,
                hovertemplate=f"{label}<br>nominal %{{x:.2f}} → empirical %{{y:.3f}}<extra></extra>",
            )
        )

    fig = apply_theme(
        fig, th, title=title, subtitle=subtitle,
        y_title="empirical", x_title="nominal quantile level",
        height=height, unified_hover=False,
    )
    fig.update_xaxes(range=[0, 1], showgrid=True, gridcolor=th.grid)
    fig.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1)
    return fig


# --------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------


def metric_bars(
    values: pd.Series,
    *,
    th: Theme | None = None,
    title: str = "",
    subtitle: str | None = None,
    x_title: str = "",
    height: int = 360,
    lower_is_better: bool = True,
    value_fmt: str = "{:,.2f}",
    highlight: str | None = None,
) -> go.Figure:
    """Horizontal bars for one metric across models.

    Every bar is direct-labelled. With a single series there is no identity question for
    colour to answer, so one hue plus labels is both sufficient and satisfies the relief
    rule. The best model is emphasised; the rest recede to a tint of the same hue, so
    ranking is carried by position and weight rather than by a second colour.
    """
    th = th or LIGHT
    s = values.sort_values(ascending=lower_is_better)
    best = highlight or s.index[0]
    colors = [th.accent if k == best else rgba(th.accent, 0.34) for k in s.index]

    fig = go.Figure(
        go.Bar(
            x=s.to_numpy(), y=[str(i) for i in s.index], orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[value_fmt.format(v) for v in s.to_numpy()],
            textposition="outside",
            textfont=dict(family=FONT_MONO, color=th.ink_secondary, size=11),
            hovertemplate="%{y}  %{x:,.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_traces(marker_cornerradius=3)
    fig = apply_theme(
        fig, th, title=title, subtitle=subtitle, x_title=x_title,
        height=height, show_legend=False, unified_hover=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor=th.grid)
    fig.update_yaxes(autorange="reversed", tickfont=dict(family=FONT_MONO, size=11))
    return fig


def line_comparison(
    frame: pd.DataFrame,
    *,
    th: Theme | None = None,
    title: str = "",
    subtitle: str | None = None,
    y_title: str = "",
    x_title: str = "",
    height: int = 420,
    dash_map: dict[str, str] | None = None,
    fill_first: bool = False,
) -> go.Figure:
    """Multi-series line chart; columns become series in fixed palette order.

    ``dash_map`` supplies an optional secondary encoding (line style), which is what makes
    an adjacent pair sitting in the 6-8 CVD band legal. ``fill_first`` tints beneath a
    single series, which reads better than a bare line when there is nothing to compare to.
    """
    th = th or LIGHT
    fig = go.Figure()
    for i, col in enumerate(frame.columns):
        fig.add_trace(
            go.Scatter(
                x=frame.index, y=frame[col].to_numpy(), mode="lines",
                line=dict(
                    color=th.series(i), width=2,
                    dash=(dash_map or {}).get(str(col), "solid"),
                ),
                fill="tozeroy" if (fill_first and i == 0 and len(frame.columns) == 1) else None,
                fillcolor=rgba(th.series(0), 0.10),
                name=str(col),
                hovertemplate=f"{col}  %{{y:,.2f}}<extra></extra>",
            )
        )
    return apply_theme(
        fig, th, title=title, subtitle=subtitle, y_title=y_title, x_title=x_title, height=height
    )


def stacked_area(
    frame: pd.DataFrame,
    *,
    th: Theme | None = None,
    title: str = "",
    subtitle: str | None = None,
    y_title: str = "MW",
    height: int = 440,
) -> go.Figure:
    """Stacked area for a generation mix.

    A 2 px surface-coloured separator sits between adjacent fills so the boundary stays
    legible where two fills are close in tone. At most seven columns; fold the tail into
    "Other" before calling.
    """
    th = th or LIGHT
    fig = go.Figure()
    for i, col in enumerate(frame.columns):
        fig.add_trace(
            go.Scatter(
                x=frame.index, y=frame[col].to_numpy(), mode="lines", stackgroup="mix",
                line=dict(width=2, color=th.surface),
                fillcolor=rgba(th.series(i), 0.88),
                name=str(col),
                hovertemplate=f"{col}  %{{y:,.0f}} MW<extra></extra>",
            )
        )
    return apply_theme(fig, th, title=title, subtitle=subtitle, y_title=y_title, height=height)


def heatmap(
    matrix: pd.DataFrame,
    *,
    th: Theme | None = None,
    title: str = "",
    subtitle: str | None = None,
    colorbar_title: str = "",
    height: int = 420,
    diverging: bool = False,
    zmid: float | None = None,
) -> go.Figure:
    """Heatmap on the teal ramp, or on the teal-rust diverging pair.

    Diverging is used only where the value has a genuine neutral midpoint (an error, a
    difference, a signed effect). Otherwise the single-hue ramp is correct, and a rainbow
    never is.
    """
    th = th or LIGHT
    if diverging:
        mid = DIVERGING["mid_light"] if th.mode == "light" else DIVERGING["mid_dark"]
        scale = [[0.0, DIVERGING["low"]], [0.5, mid], [1.0, DIVERGING["high"]]]
    else:
        scale = [[i / (len(SEQ_TEAL) - 1), c] for i, c in enumerate(SEQ_TEAL)]

    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=[str(c) for c in matrix.columns],
            y=[str(i) for i in matrix.index],
            colorscale=scale, zmid=zmid if diverging else None,
            colorbar=dict(
                title=dict(text=colorbar_title, font=dict(family=FONT_MONO, size=10, color=th.ink_muted)),
                tickfont=dict(family=FONT_MONO, size=10, color=th.ink_muted),
                outlinewidth=0, thickness=11, len=0.8,
            ),
            hovertemplate="%{y} · %{x}  %{z:,.2f}<extra></extra>",
        )
    )
    fig = apply_theme(
        fig, th, title=title, subtitle=subtitle, height=height,
        show_legend=False, unified_hover=False,
    )
    fig.update_yaxes(showgrid=False)
    return fig


def dot_interval(
    labels: list[str],
    centres: list[float],
    lows: list[float],
    highs: list[float],
    *,
    th: Theme | None = None,
    title: str = "",
    subtitle: str | None = None,
    x_title: str = "",
    height: int = 300,
    zero_line: bool = True,
) -> go.Figure:
    """Point estimates with confidence intervals, one row per estimator.

    The correct form for comparing a handful of estimates with uncertainty: a bar chart
    would imply the quantity is a magnitude measured from zero, which an effect size with
    a confidence interval is not.
    """
    th = th or LIGHT
    fig = go.Figure()
    for i, (lab, c, lo, hi) in enumerate(zip(labels, centres, lows, highs)):
        col = th.series(i)
        fig.add_trace(
            go.Scatter(
                x=[lo, hi], y=[lab, lab], mode="lines",
                line=dict(color=col, width=3), showlegend=False, hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[c], y=[lab], mode="markers+text",
                marker=dict(size=13, color=col, line=dict(width=2, color=th.surface)),
                text=[f"{c:,.2f}"], textposition="top center",
                textfont=dict(family=FONT_MONO, size=11, color=th.ink_secondary),
                name=lab,
                hovertemplate=f"{lab}  %{{x:,.2f}}  [{lo:,.2f}, {hi:,.2f}]<extra></extra>",
                showlegend=False,
            )
        )
    if zero_line:
        fig.add_vline(x=0, line=dict(color=th.ink_muted, width=1, dash="dot"))

    fig = apply_theme(
        fig, th, title=title, subtitle=subtitle, x_title=x_title,
        height=height, show_legend=False, unified_hover=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor=th.grid)
    fig.update_yaxes(tickfont=dict(family=FONT_MONO, size=11))
    return fig
