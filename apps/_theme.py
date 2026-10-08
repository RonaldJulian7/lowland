"""The Lowland app shell: typography, chrome and page furniture.

Streamlit has a strong default look - blue accents, rounded cards, system fonts, a header
bar - and a project that keeps it reads as a Streamlit app rather than as a piece of work.
Everything here exists to replace that with a deliberate one:

* an editorial serif (Fraunces) for headings, a tight grotesque (Inter Tight) for
  interface text, and a monospace (JetBrains Mono) for every figure and label;
* numbered section markers and lowercase monospaced eyebrows, so hierarchy is carried by
  weight, case and letterspacing rather than by size alone;
* hairline rules and square-ish corners instead of shadows and pills;
* warm paper and warm ink surfaces rather than neutral grey.

The type scale is deliberately narrow - four sizes for headings, two for body - because
a small scale used consistently reads as designed, and a large one rarely does.
"""

from __future__ import annotations

import streamlit as st

from lowland.viz import (
    DARK,
    FONT_DISPLAY,
    FONT_MONO,
    FONT_UI,
    GOOGLE_FONTS_HREF,
    LIGHT,
    STATUS,
    Theme,
    rgba,
)


def _css(th: Theme) -> str:
    """The full stylesheet for one colour mode."""
    is_light = th.mode == "light"
    return f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{GOOGLE_FONTS_HREF}" rel="stylesheet">
<style>
  :root {{
    --ink:        {th.ink_primary};
    --ink-2:      {th.ink_secondary};
    --ink-3:      {th.ink_muted};
    --paper:      {th.plane};
    --surface:    {th.surface};
    --raised:     {th.raised};
    --rule:       {th.rule};
    --accent:     {th.accent};
  }}

  /* ---- strip Streamlit chrome ------------------------------------------------ */
  /* The header is made transparent rather than hidden. Hiding it outright also hides
     the sidebar collapse/expand control that lives inside it, and once a visitor
     collapses the sidebar there is then no way to get it back. */
  #MainMenu, footer {{ display: none !important; }}
  [data-testid="stDecoration"] {{ display: none !important; }}
  [data-testid="stToolbar"] {{ display: none !important; }}
  header[data-testid="stHeader"] {{
    background: transparent !important;
    height: 2.5rem !important;
    box-shadow: none !important;
  }}
  [data-testid="stSidebarCollapseButton"],
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"] {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: var(--ink-2) !important;
  }}

  .stApp {{ background: var(--paper); }}
  .block-container {{
    padding: 3.2rem 3rem 5rem 3rem;
    max-width: 1460px;
  }}

  /* ---- type ------------------------------------------------------------------- */
  html, body, [class*="css"], .stMarkdown, p, li, label, span, div {{
    font-family: {FONT_UI};
    -webkit-font-smoothing: antialiased;
  }}
  .stMarkdown p, .stMarkdown li {{
    color: var(--ink-2);
    font-size: 14.5px;
    line-height: 1.62;
    letter-spacing: -0.004em;
  }}
  .stMarkdown strong {{ color: var(--ink); font-weight: 600; }}
  .stMarkdown a {{ color: var(--accent); text-decoration: none; border-bottom: 1px solid {rgba(th.accent, 0.35)}; }}

  h1, h2, h3, h4 {{
    font-family: {FONT_DISPLAY} !important;
    color: var(--ink) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
  }}
  h1 {{ font-size: 2.9rem !important; line-height: 1.04 !important; margin: 0 0 .35rem 0 !important; }}
  h2 {{ font-size: 1.55rem !important; line-height: 1.16 !important; margin: 2.4rem 0 .5rem 0 !important; }}
  h3 {{ font-size: 1.16rem !important; line-height: 1.22 !important; margin: 1.5rem 0 .4rem 0 !important; }}

  code, pre, .stCode {{ font-family: {FONT_MONO} !important; font-size: 12.5px !important; }}

  /* ---- sidebar ---------------------------------------------------------------- */
  [data-testid="stSidebar"] {{
    background: {th.raised if is_light else "#0C0E07"};
    border-right: 1px solid var(--rule);
  }}
  [data-testid="stSidebar"] .block-container {{ padding-top: 2.2rem; }}
  [data-testid="stSidebarNav"] {{ padding-top: .4rem; }}
  [data-testid="stSidebarNav"] a span {{
    font-family: {FONT_MONO} !important;
    font-size: 11.5px !important;
    letter-spacing: .04em;
    color: var(--ink-2) !important;
  }}
  [data-testid="stSidebarNav"] a[aria-current="page"] span {{
    color: var(--accent) !important;
    font-weight: 600 !important;
  }}

  /* ---- controls ---------------------------------------------------------------- */
  label, [data-testid="stWidgetLabel"] p {{
    font-family: {FONT_MONO} !important;
    font-size: 10.5px !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    color: var(--ink-3) !important;
    font-weight: 500 !important;
  }}
  [data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
    background: var(--surface) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 3px !important;
    font-family: {FONT_UI} !important;
    font-size: 13.5px !important;
    color: var(--ink) !important;
  }}
  [data-baseweb="tag"] {{
    background: {rgba(th.accent, 0.14)} !important;
    border-radius: 2px !important;
  }}
  [data-baseweb="tag"] span {{ color: var(--accent) !important; font-family: {FONT_MONO} !important; font-size: 11px !important; }}
  .stSlider [data-baseweb="slider"] div[role="slider"] {{ background: var(--accent) !important; }}
  .stRadio label p, .stCheckbox label p {{
    font-family: {FONT_UI} !important; text-transform: none !important;
    letter-spacing: 0 !important; font-size: 13px !important; color: var(--ink-2) !important;
  }}

  /* ---- expanders, tables ---------------------------------------------------------- */
  [data-testid="stExpander"] {{
    border: 1px solid var(--rule) !important;
    border-radius: 3px !important;
    background: var(--surface);
  }}
  [data-testid="stExpander"] summary p {{
    font-family: {FONT_MONO} !important;
    font-size: 11px !important;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--ink-3) !important;
  }}
  [data-testid="stDataFrame"] {{ border: 1px solid var(--rule) !important; border-radius: 3px; }}
  [data-testid="stDataFrame"] * {{ font-family: {FONT_MONO} !important; font-size: 11.5px !important; }}

  /* ---- alerts ------------------------------------------------------------------- */
  [data-testid="stAlert"] {{
    border-radius: 3px !important;
    border-left: 3px solid var(--accent) !important;
    background: {rgba(th.accent, 0.07)} !important;
  }}
  [data-testid="stAlert"] p {{ font-size: 13.5px !important; color: var(--ink-2) !important; }}

  /* ---- house components ------------------------------------------------------------ */
  .lw-eyebrow {{
    font-family: {FONT_MONO};
    font-size: 10.5px;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin: 0 0 .55rem 0;
    display: flex; align-items: center; gap: .6rem;
  }}
  .lw-eyebrow::after {{
    content: ""; flex: 1; height: 1px; background: var(--rule);
  }}
  .lw-lede {{
    font-family: {FONT_DISPLAY};
    font-size: 1.22rem;
    line-height: 1.48;
    font-weight: 400;
    color: var(--ink-2);
    max-width: 62ch;
    margin: .2rem 0 1.5rem 0;
  }}
  .lw-note {{
    font-family: {FONT_UI};
    font-size: 12.5px;
    line-height: 1.55;
    color: var(--ink-3);
    max-width: 78ch;
    margin: -.25rem 0 1.1rem 0;
  }}
  .lw-note b, .lw-note strong {{ color: var(--ink-2); font-weight: 600; }}

  .lw-tile {{
    background: var(--surface);
    border: 1px solid var(--rule);
    border-radius: 3px;
    padding: 15px 16px 16px 16px;
    height: 100%;
    position: relative;
  }}
  .lw-tile::before {{
    content: ""; position: absolute; top: -1px; left: -1px; width: 26px; height: 2px;
    background: var(--accent);
  }}
  .lw-tile-label {{
    font-family: {FONT_MONO}; font-size: 9.5px; letter-spacing: .14em;
    text-transform: uppercase; color: var(--ink-3);
  }}
  .lw-tile-value {{
    font-family: {FONT_MONO}; font-size: 26px; font-weight: 500; letter-spacing: -.02em;
    color: var(--ink); margin-top: 6px; line-height: 1.1;
  }}
  .lw-tile-value.is-text {{ font-family: {FONT_DISPLAY}; font-size: 25px; font-weight: 600; }}
  .lw-tile-delta {{ font-family: {FONT_MONO}; font-size: 11px; margin-top: 5px; }}
  .lw-tile-help {{
    font-family: {FONT_UI}; font-size: 11.5px; line-height: 1.45;
    color: var(--ink-3); margin-top: 9px;
  }}

  .lw-rule {{ height: 1px; background: var(--rule); margin: 2.6rem 0 1.6rem 0; }}

  /* ---- in-page nav ------------------------------------------------------------- */
  [data-testid="stPageLink"] a {{
    font-family: {FONT_MONO} !important;
    font-size: 11.5px !important;
    letter-spacing: .04em;
    padding: 4px 8px !important;
    border-radius: 2px !important;
    background: transparent !important;
  }}
  [data-testid="stPageLink"] a p {{
    font-family: {FONT_MONO} !important;
    font-size: 11.5px !important;
    letter-spacing: .04em;
    color: var(--ink-3) !important;
    margin: 0 !important;
  }}
  [data-testid="stPageLink"] a:hover p {{ color: var(--accent) !important; }}
  [data-testid="stPageLink"] a[aria-current="page"] p,
  [data-testid="stPageLink"] a.active p {{
    color: var(--accent) !important; font-weight: 600 !important;
  }}
  [data-testid="stPageLink"] svg {{ display: none !important; }}

  .lw-quote {{
    border-left: 2px solid var(--accent);
    padding: .2rem 0 .2rem 1.1rem;
    margin: 1.1rem 0 1.4rem 0;
    font-family: {FONT_DISPLAY};
    font-size: 1.02rem;
    line-height: 1.5;
    color: var(--ink-2);
    max-width: 68ch;
  }}

  .lw-card {{
    background: var(--surface); border: 1px solid var(--rule); border-radius: 3px;
    padding: 18px 20px; height: 100%;
  }}
  .lw-card h4 {{
    font-family: {FONT_DISPLAY} !important; font-size: 1.12rem !important;
    margin: .1rem 0 .45rem 0 !important;
  }}
  .lw-card p {{ font-size: 13.2px !important; line-height: 1.55 !important; color: var(--ink-2) !important; }}
  .lw-index {{
    font-family: {FONT_MONO}; font-size: 10.5px; letter-spacing: .14em;
    color: var(--accent); text-transform: uppercase;
  }}
</style>
"""


#: Pages for the in-page nav bar, as (candidate paths, label, url).
#
# st.page_link resolves paths against the entrypoint's own directory, which is `apps/`
# when the app is started as `streamlit run apps/Home.py` from the repo root - but not
# when it is started from inside `apps/`. Both spellings are tried, and a plain anchor is
# the last resort. The anchor works but costs a full page load, which resets the theme
# toggle, so it is genuinely a fallback rather than an equivalent.
NAV = (
    (("Home.py", "apps/Home.py"), "Home", "/"),
    (("pages/1_Bellwether.py", "apps/pages/1_Bellwether.py"), "Bellwether", "/Bellwether"),
    (("pages/2_Flywheel.py", "apps/pages/2_Flywheel.py"), "Flywheel", "/Flywheel"),
    (("pages/3_Vantage.py", "apps/pages/3_Vantage.py"), "Vantage", "/Vantage"),
)


def page_setup(title: str, *, wide: bool = True) -> Theme:
    """Configure the page, inject the stylesheet, draw the nav bar, return the theme.

    Navigation and the colour toggle are rendered in the main column, not only in the
    sidebar. Streamlit removes the sidebar from the page entirely when it is collapsed
    and only re-creates it when the pointer happens to land within about twenty pixels of
    the window edge - there is no visible affordance in the meantime. Anything the app
    genuinely needs therefore cannot live in the sidebar alone.
    """
    st.set_page_config(
        page_title=f"{title} · Lowland",
        layout="wide" if wide else "centered",
        initial_sidebar_state="expanded",
    )

    # A keyed widget means st.session_state["mode"] already holds the current value when
    # the script re-runs, so the stylesheet can be chosen before the toggle is drawn.
    mode = st.session_state.get("mode", "light")
    th = DARK if mode == "dark" else LIGHT
    # The stylesheet is flattened to a single line before injection. Streamlit renders it
    # through a Markdown parser, which mangles a readably-formatted <style> block in two
    # separate ways: any line indented four or more spaces becomes a code block, and a
    # blank line ends the HTML block, silently truncating the stylesheet at the first
    # paragraph break. Both were happening here -- the rules after the first blank line
    # simply never arrived, and earlier the tail leaked onto the page as visible text.
    # CSS is whitespace-insensitive, so collapsing it costs nothing and removes the whole
    # class of problem.
    flat = " ".join(line.strip() for line in _css(th).splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)

    _sidebar_mark(th)
    _topbar(th, title)
    return th


def _sidebar_mark(th: Theme) -> None:
    """Wordmark in the sidebar. Decorative only - nothing depends on it."""
    with st.sidebar:
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:1.45rem;font-weight:600;"
            f"letter-spacing:-.02em;margin:0 0 .15rem 0;color:{th.ink_primary}'>Lowland</div>"
            f"<div style='font-family:{FONT_MONO};font-size:9.5px;letter-spacing:.14em;"
            f"text-transform:uppercase;color:{th.ink_muted};margin-bottom:.6rem'>"
            f"Dutch power system studies</div>",
            unsafe_allow_html=True,
        )


def _topbar(th: Theme, active: str) -> None:
    """Navigation and the colour toggle, in the main column."""
    cols = st.columns([1.1, 1.3, 1.1, 1.1, 2.6, 1.5], gap="small")
    for (paths, label, href), col in zip(NAV, cols):
        with col:
            for candidate in paths:
                try:
                    st.page_link(candidate, label=label, use_container_width=True)
                    break
                except Exception:  # noqa: BLE001 - try the next spelling
                    continue
            else:
                is_active = label.startswith(active.lower()[:4])
                st.markdown(
                    f"<a href='{href}' target='_self' style='font-family:{FONT_MONO};"
                    f"font-size:11px;letter-spacing:.08em;text-decoration:none;"
                    f"color:{th.accent if is_active else th.ink_muted}'>{label}</a>",
                    unsafe_allow_html=True,
                )
    with cols[-1]:
        st.radio(
            "colour mode", ["light", "dark"], key="mode",
            horizontal=True, label_visibility="collapsed",
        )
    st.markdown(
        f"<div style='height:1px;background:{th.rule};margin:.1rem 0 1.6rem 0'></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------------------
# Page furniture
# --------------------------------------------------------------------------------------


def eyebrow(text: str) -> None:
    """A small monospaced kicker with a rule running to the right margin."""
    st.markdown(f"<div class='lw-eyebrow'>{text}</div>", unsafe_allow_html=True)


def lede(text: str) -> None:
    """The standfirst under a page title, set in the display serif."""
    st.markdown(f"<div class='lw-lede'>{text}</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    """Small explanatory text beneath a chart or section heading."""
    st.markdown(f"<div class='lw-note'>{text}</div>", unsafe_allow_html=True)


def rule() -> None:
    st.markdown("<div class='lw-rule'></div>", unsafe_allow_html=True)


def tile(
    label: str,
    value: str,
    *,
    delta: str | None = None,
    delta_state: str | None = None,
    help_text: str | None = None,
    text_value: bool = False,
) -> str:
    """One KPI as HTML.

    A bare number beats a one-bar chart: there is nothing to compare, so there is nothing
    for axes and marks to encode. Numeric values are set in the monospace so columns of
    them align; a value that is a *name* switches to the display serif instead, because
    monospaced words read as code rather than as a label.

    The delta carries a word, never colour alone.
    """
    colour = {
        "good": STATUS["good"], "bad": STATUS["critical"], "warn": STATUS["warning"],
    }.get(delta_state or "", "var(--ink-3)")
    delta_html = (
        f"<div class='lw-tile-delta' style='color:{colour}'>{delta}</div>" if delta else ""
    )
    help_html = f"<div class='lw-tile-help'>{help_text}</div>" if help_text else ""
    cls = "lw-tile-value is-text" if text_value else "lw-tile-value"
    return (
        f"<div class='lw-tile'><div class='lw-tile-label'>{label}</div>"
        f"<div class='{cls}'>{value}</div>{delta_html}{help_html}</div>"
    )


def tiles(items: list[dict]) -> None:
    """Render a row of KPI tiles."""
    cols = st.columns(len(items), gap="small")
    for col, item in zip(cols, items):
        with col:
            st.markdown(tile(**item), unsafe_allow_html=True)


def card(index: str, heading: str, body: str) -> str:
    """A bordered card with a monospaced index, for the three-up project summary."""
    return (
        f"<div class='lw-card'><div class='lw-index'>{index}</div>"
        f"<h4>{heading}</h4><p>{body}</p></div>"
    )
