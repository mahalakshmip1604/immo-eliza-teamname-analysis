#!/usr/bin/env python3
"""Single-insight Immo-Eliza report → rental yield by municipality (v2).

Same standalone rental-yield insight as ``yieldinsight.py``, but sourced from
the team's *cleaned* datasets and written into the ``reports/`` folder:

    in  : data/cleaned/cleaned_sale_properties.csv
          data/cleaned/cleaned_rent_properties.csv
    out : reports/yieldinsight.html

The cleaned files are de-duplicated / whitespace-trimmed / null-normalised but
still carry price & surface outliers, so this script keeps the market-bound
guard rails (and recomputes €/m²) to produce sane yields. It is fully
standalone: every figure is recomputed and the Plotly runtime is embedded, so
nothing external is needed to build or view the report.

Usage
-----
    python src/yieldinsight_v2.py
    # or override paths:
    IMMO_DATA=/path/to/cleaned IMMO_OUT=/path/to/out.html python src/yieldinsight_v2.py

Dependencies: pandas, numpy, plotly  (no scikit-learn / no network needed).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

# --------------------------------------------------------------------------- #
# Paths — read the cleaned datasets, write into reports/
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("IMMO_DATA", str(HERE.parent / "data" / "cleaned")))
OUT_HTML = Path(os.environ.get("IMMO_OUT", str(HERE.parent / "reports" / "yieldinsight.html")))
SALE_CSV = DATA_DIR / "cleaned_sale_properties.csv"
RENT_CSV = DATA_DIR / "cleaned_rent_properties.csv"

# --------------------------------------------------------------------------- #
# Cleaning thresholds — identical to the production model's guard rails
# --------------------------------------------------------------------------- #
SALE_PRICE_MIN, SALE_PRICE_MAX = 25_000, 15_000_000
RENT_PRICE_MIN, RENT_PRICE_MAX = 200, 25_000
SURFACE_MIN, SURFACE_MAX = 9, 3_000
SALE_PPSQM_MIN, SALE_PPSQM_MAX = 400, 18_000
RENT_PPSQM_MIN, RENT_PPSQM_MAX = 3, 70

# --------------------------------------------------------------------------- #
# Look & feel
# --------------------------------------------------------------------------- #
INK = "#eef3fb"
MUTED = "#9fb0c8"
GRID = "#1e2940"
BRAND = dict(blue="#5b8cff", cyan="#22d3ee", green="#22c98a",
             amber="#f5b53d", pink="#fb6f92", slate="#6f8099")
REGION_COLOR = {"Brussels": "#fb6f92", "Flanders": "#22d3ee", "Wallonia": "#f5b53d"}

PLOTLY_CFG = {"displayModeBar": False, "responsive": True}


# --------------------------------------------------------------------------- #
# Data loading / cleaning
# --------------------------------------------------------------------------- #
def load_clean(path: Path, kind: str) -> pd.DataFrame:
    """Load a listings CSV and apply the market-bound cleaning rules."""
    df = pd.read_csv(path, low_memory=False)
    for c in ["price", "livable_surface", "bedrooms", "bathrooms", "build_year",
              "latitude", "longitude", "garden", "terrace", "new_construction",
              "postal_code"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["price"] > 0]
    df = df[df["livable_surface"].between(SURFACE_MIN, SURFACE_MAX)]
    df["ppsqm"] = df["price"] / df["livable_surface"]      # recompute, authoritative
    if kind == "sale":
        df = df[df["price"].between(SALE_PRICE_MIN, SALE_PRICE_MAX)]
        df = df[df["ppsqm"].between(SALE_PPSQM_MIN, SALE_PPSQM_MAX)]
    else:
        df = df[df["price"].between(RENT_PRICE_MIN, RENT_PRICE_MAX)]
        df = df[df["ppsqm"].between(RENT_PPSQM_MIN, RENT_PPSQM_MAX)]
    return df.reset_index(drop=True)


def style(fig: go.Figure, height: int = 450, **kw) -> go.Figure:
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,-apple-system,Segoe UI,system-ui,sans-serif",
                  color="#cdd9ee", size=13),
        margin=dict(l=70, r=28, t=24, b=52), height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0e1626", bordercolor="#33415c",
                        font=dict(color=INK)),
    )
    layout.update(kw)                       # caller kwargs win (e.g. margin, axes)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#33415c")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#33415c")
    return fig


# --------------------------------------------------------------------------- #
# Per-area sale/rent yield table
# --------------------------------------------------------------------------- #
def area_yield_table(sale: pd.DataFrame, rent: pd.DataFrame, min_n: int = 15) -> pd.DataFrame:
    s = sale.dropna(subset=["locality"])
    r = rent.dropna(subset=["locality"])
    sg = s.groupby("locality").agg(
        sale_ppsqm=("ppsqm", "median"),
        n_sale=("ppsqm", "size"),
        region=("region", lambda x: x.mode().iat[0] if not x.mode().empty else None),
        lat=("latitude", "median"), lon=("longitude", "median"),
    )
    rg = r.groupby("locality").agg(rent_ppsqm=("ppsqm", "median"), n_rent=("ppsqm", "size"))
    m = sg.join(rg, how="inner")
    m = m[(m.n_sale >= min_n) & (m.n_rent >= min_n)].copy()
    m["yield"] = m["rent_ppsqm"] * 12 / m["sale_ppsqm"] * 100
    return m.reset_index().sort_values("yield", ascending=False)


# --------------------------------------------------------------------------- #
# The insight — rental yield by area (diverging bar)
# --------------------------------------------------------------------------- #
def insight_yield_bars(sale, rent, area):
    nat_yield = rent["ppsqm"].median() * 12 / sale["ppsqm"].median() * 100
    top = area.head(12)
    bot = area.tail(12)
    sub = pd.concat([bot, top]).sort_values("yield")          # ascending → best at top
    colors = [REGION_COLOR.get(r, BRAND["slate"]) for r in sub["region"]]

    fig = go.Figure(go.Bar(
        x=sub["yield"], y=sub["locality"], orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in sub["yield"]], textposition="outside",
        customdata=np.stack([sub["sale_ppsqm"], sub["rent_ppsqm"], sub["region"]], axis=-1),
        hovertemplate="<b>%{y}</b> (%{customdata[2]})<br>yield %{x:.1f}%"
                      "<br>buy €%{customdata[0]:.0f}/m² · rent €%{customdata[1]:.1f}/m²<extra></extra>",
    ))
    fig.add_vline(x=nat_yield, line_dash="dash", line_color="#9fb0c8",
                  annotation_text=f"national {nat_yield:.1f}%",
                  annotation_position="top", annotation_font_color=MUTED)
    # region legend proxies
    for reg, col in REGION_COLOR.items():
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=col, name=reg, orientation="h"))
    style(fig, height=560, xaxis_title="gross rental yield (%)", showlegend=True, barmode="overlay")

    best = area.iloc[0]
    worst = area.iloc[-1]
    return dict(
        n=3, audience="investor",
        title=f"{best['locality']} yields {best['yield']:.1f}% — over double the priciest Brussels communes",
        chart=fig,
        finding=(
            f"Across <b>{len(area)}</b> municipalities with ≥15 sale <i>and</i> ≥15 rent listings, gross "
            f"rental yield spans <b>{best['yield']:.1f}%</b> down to <b>{worst['yield']:.1f}%</b>. The high "
            "tail is Walloon industrial cities and outer-Brussels communes; the low tail is affluent "
            f"Brussels and its periphery. Gross payback ranges from ~{1200/best['yield']:.0f} months "
            f"({best['locality']}) to ~{1200/worst['yield']:.0f} months ({worst['locality']})."),
        chips=[(f"{best['locality']}", f"{best['yield']:.1f}%"),
               (f"{worst['locality']}", f"{worst['yield']:.1f}%"),
               ("national gross yield", f"{nat_yield:.1f}%"),
               ("areas matched", f"{len(area)}")],
        why=("<b>Sorted (diverging) horizontal bar</b> — one yield value per area reads best ranked; "
             "colouring by region exposes that the high-yield top is Wallonia / outer-Brussels and the "
             "low-yield bottom is affluent Brussels, with the dashed national line as the divider."),
        extra=None,
    )


# --------------------------------------------------------------------------- #
# Look & feel for the HTML page
# --------------------------------------------------------------------------- #
CSS = """
:root{--bg:#0b1220;--card:#111c30;--ink:#eef3fb;--muted:#9fb0c8;--line:#1e2940;
      --accent:#5b8cff;--accent2:#22d3ee;}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 600px at 80% -10%,#16243e 0%,var(--bg) 55%);
     color:var(--ink);font-family:Inter,-apple-system,Segoe UI,system-ui,sans-serif;line-height:1.55}
.wrap{max-width:1080px;margin:0 auto;padding:0 22px 80px}
header.hero{padding:64px 22px 30px;max-width:1080px;margin:0 auto}
.kicker{letter-spacing:.18em;text-transform:uppercase;font-size:12px;color:var(--accent2);font-weight:700}
h1{font-size:40px;line-height:1.08;margin:10px 0 8px;font-weight:800;
   background:linear-gradient(90deg,#eaf1ff,#8fb3ff);-webkit-background-clip:text;background-clip:text;color:transparent}
.lede{color:var(--muted);font-size:17px;max-width:760px}
.meta{display:flex;flex-wrap:wrap;gap:14px;margin-top:22px}
.meta .pill{background:#0e1830;border:1px solid var(--line);border-radius:12px;padding:10px 14px}
.meta .pill b{font-size:20px;display:block;color:var(--ink)}
.meta .pill span{font-size:12px;color:var(--muted)}
section.insight{background:linear-gradient(180deg,#121f36,#0f1830);border:1px solid var(--line);
   border-radius:18px;padding:26px 26px 12px;margin:26px 0;box-shadow:0 12px 40px rgba(0,0,0,.35)}
.ihead{display:flex;gap:16px;align-items:flex-start}
.badge{flex:none;width:46px;height:46px;border-radius:12px;display:grid;place-items:center;font-weight:800;
       font-size:20px;color:#06101f;background:linear-gradient(135deg,#7aa2ff,#22d3ee)}
.itext h2{margin:2px 0 6px;font-size:23px;font-weight:800}
.tag{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--accent2);
     border:1px solid #23406a;border-radius:999px;padding:2px 9px;margin-left:6px;vertical-align:middle}
.finding{color:#d6e0f2;font-size:15.5px;margin:6px 0 14px}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 14px}
.chip{background:#0c1730;border:1px solid var(--line);border-radius:10px;padding:7px 12px;font-size:13px;color:var(--muted)}
.chip b{color:var(--ink);font-size:15px;margin-right:5px}
.chart{margin:6px 0 4px}
.why{font-size:13.5px;color:var(--muted);border-top:1px dashed #25324c;padding:12px 0 14px;margin-top:6px}
.why b{color:#cfe0ff}
.extra{font-size:12.5px;color:#7d8fab;margin:-6px 0 12px}
footer{max-width:1080px;margin:40px auto 0;padding:24px 22px;color:var(--muted);font-size:13px;
       border-top:1px solid var(--line)}
footer code{background:#0e1830;border:1px solid var(--line);border-radius:6px;padding:1px 6px;color:#cfe0ff}
"""


# --------------------------------------------------------------------------- #
# HTML assembly — one insight only
# --------------------------------------------------------------------------- #
def render(section, n_sale, n_rent, n_areas):
    plotly_js = get_plotlyjs()
    chart_html = pio.to_html(section["chart"], full_html=False, include_plotlyjs=False,
                             div_id=f"chart{section['n']}", config=PLOTLY_CFG)
    chips = "".join(f'<span class="chip"><b>{v}</b>{k}</span>' for k, v in section["chips"])
    extra = f'<p class="extra">{section["extra"]}</p>' if section.get("extra") else ""
    tag = {"buyer": "for buyers", "investor": "for investors",
           "both": "buyers &amp; investors"}[section["audience"]]

    return f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Immo Eliza · Rental Yield by Municipality</title>
<style>{CSS}</style>
<script>{plotly_js}</script></head><body>
<header class="hero">
  <div class="kicker">Immo Eliza · Belgian residential market</div>
  <h1>{section['title']}</h1>
  <p class="lede">Gross rental yield ranked across Belgian municipalities, computed from the team's
     cleaned datasets ({n_sale:,} sale &amp; {n_rent:,} rental listings) with the model’s
     market-bound guard rails — for investors hunting the strongest cash-on-cash return.</p>
  <div class="meta">
    <div class="pill"><b>{n_sale:,}</b><span>clean sale listings</span></div>
    <div class="pill"><b>{n_rent:,}</b><span>clean rental listings</span></div>
    <div class="pill"><b>{n_areas}</b><span>municipalities matched</span></div>
    <div class="pill"><b>3</b><span>regions · whole of Belgium</span></div>
  </div></header><div class="wrap">
  <section class="insight" id="i{section['n']}">
    <div class="ihead">
      <div class="badge">%</div>
      <div class="itext"><h2>{section['title']}<span class="tag">{tag}</span></h2>
        <p class="finding">{section['finding']}</p></div>
    </div>
    <div class="chips">{chips}</div>
    {extra}
    <div class="chart">{chart_html}</div>
    <p class="why">{section['why']}</p>
  </section>
</div><footer>
  <b>Method.</b> Built from the team's cleaned datasets
  (<code>cleaned_sale_properties.csv</code>, <code>cleaned_rent_properties.csv</code>), with the
  production guard rails re-applied (sale price <code>€25k–€15M</code>, rent <code>€200–€25k</code>,
  living area <code>9–3000&nbsp;m²</code>, €/m² recomputed as price ÷ surface and bounded). Yields are
  gross (<code>median rent €/m² × 12 ÷ median sale €/m²</code>) over municipalities with ≥15 listings on
  each side. NaNs are dropped per metric. Chart is interactive — hover for the underlying values.
  Generated by <code>src/yieldinsight_v2.py</code>.
</footer></body></html>"""


def main():
    print(f"reading  {SALE_CSV}")
    print(f"reading  {RENT_CSV}")
    sale = load_clean(SALE_CSV, "sale")
    rent = load_clean(RENT_CSV, "rent")
    print(f"clean sale={len(sale):,}  clean rent={len(rent):,}")

    area = area_yield_table(sale, rent)
    print(f"matched yield areas (n>=15 both sides): {len(area)}")

    section = insight_yield_bars(sale, rent, area)
    html = render(section, len(sale), len(rent), len(area))
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    kb = OUT_HTML.stat().st_size / 1024
    print(f"\nwrote {OUT_HTML}  ({kb:,.0f} KB, 1 insight)")


if __name__ == "__main__":
    main()
