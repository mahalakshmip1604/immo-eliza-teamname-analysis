#!/usr/bin/env python3
"""Regional property-price evolution chart — static PNG render of
reports/price_evolution.ipynb.

Two-panel line chart (Houses and Flats) of the median sale price per region
(Flanders, Wallonia, Brussels): the last 5 years of real history (2020-2025,
Statbel) plus a 5-year projection (2026-2030) with a conservative-to-optimistic
scenario band.

Usage
-----
    python src/price_evolution.py
    # or override paths:
    IMMO_DATA=/path/to/data IMMO_IMAGES=/path/to/images python src/price_evolution.py

Reads (resolved relative to this file):
    ../data/property_prices_historical_by_municipality.csv
    ../data/property_prices_projected_by_municipality.csv
Writes:
    ../images/price_evolution_by_region.png

The input CSVs are produced by scripts/build_price_evolution.py — run that first
if they are missing.

Dependencies: pandas, matplotlib.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless: write PNG, never open a window
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("IMMO_DATA", str(HERE.parent / "data")))
OUT_DIR = Path(os.environ.get("IMMO_IMAGES", str(HERE.parent / "images")))
HIST_CSV = DATA_DIR / "property_prices_historical_by_municipality.csv"
PROJ_CSV = DATA_DIR / "property_prices_projected_by_municipality.csv"
OUT_PNG = OUT_DIR / "price_evolution_by_region.png"

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
REGIONS = ["Flanders", "Wallonia", "Brussels"]
REGION_COLOR = {"Flanders": "#1f77b4", "Wallonia": "#d62728", "Brussels": "#2ca02c"}
TYPES = {"houses_all": "Houses", "apartments": "Flats"}
HIST_START = 2020          # last 5 years of history
BASE_YEAR = 2025           # last actual / projection anchor
SCEN_CENTRAL = "long_run_base"
SCEN_LO, SCEN_HI = "long_run_conservative", "long_run_optimistic"


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def regional_hist(hist: pd.DataFrame, ptype: str) -> pd.DataFrame:
    """Transaction-weighted median price per region/year (a region's price is the
    weighted average of its municipal medians, weighted by transactions)."""
    d = hist[(hist.property_type == ptype) & (hist.year >= HIST_START)].copy()
    d["wp"] = d.median_price_eur * d.n_transactions
    g = (d.groupby(["region", "year"])
           .agg(wp=("wp", "sum"), w=("n_transactions", "sum"))
           .reset_index())
    g["price"] = g.wp / g.w
    return g.pivot(index="year", columns="region", values="price")


def cum_factor(proj: pd.DataFrame, ptype: str, scenario: str) -> pd.Series:
    """Cumulative growth factor per projection year (national, identical across
    municipalities, so the regional series scales by the same factor)."""
    s = proj[(proj.property_type == ptype) & (proj.scenario == scenario)]
    return s.groupby("year").cumulative_growth_pct.first() / 100 + 1


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build_figure(hist: pd.DataFrame, proj: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    for ax, (ptype, label) in zip(axes, TYPES.items()):
        h = regional_hist(hist, ptype)
        fac_c = cum_factor(proj, ptype, SCEN_CENTRAL)
        fac_lo = cum_factor(proj, ptype, SCEN_LO)
        fac_hi = cum_factor(proj, ptype, SCEN_HI)
        pyears = list(fac_c.index)                       # [2026..2030]

        for region in REGIONS:
            color = REGION_COLOR[region]
            ax.plot(h.index, h[region], color=color, lw=2.4, marker="o", ms=4)  # history

            base = h.loc[BASE_YEAR, region]
            px = [BASE_YEAR] + pyears                     # anchor projection at last actual
            pc = [base] + [base * fac_c[y] for y in pyears]
            plo = [base] + [base * fac_lo[y] for y in pyears]
            phi = [base] + [base * fac_hi[y] for y in pyears]
            ax.plot(px, pc, color=color, lw=2.4, ls="--", marker="o", ms=4)     # projection
            ax.fill_between(px, plo, phi, color=color, alpha=0.12, lw=0)        # scenario band
            ax.annotate(f"€{pc[-1] / 1000:,.0f}k", (px[-1], pc[-1]),
                        color=color, fontsize=9, fontweight="bold",
                        xytext=(5, 0), textcoords="offset points", va="center")

        ax.axvline(BASE_YEAR, color="0.5", ls=":", lw=1.2)
        ax.text(BASE_YEAR + 0.1, ax.get_ylim()[0], " projection →", color="0.4",
                fontsize=9, va="bottom")
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_xticks(range(HIST_START, max(pyears) + 1, 2))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"€{v / 1000:,.0f}k"))
        ax.grid(alpha=0.3)
        ax.margins(x=0.06)

    axes[0].set_ylabel("Median sale price")

    region_h = [plt.Line2D([], [], color=REGION_COLOR[r], lw=2.4, label=r) for r in REGIONS]
    style_h = [
        plt.Line2D([], [], color="0.3", lw=2.4, ls="-", label="Historical (Statbel)"),
        plt.Line2D([], [], color="0.3", lw=2.4, ls="--", label="Projected (base +3%/yr)"),
        plt.Rectangle((0, 0), 1, 1, color="0.5", alpha=0.18,
                      label="Scenario range (+2% to +4.3%/yr)"),
    ]
    fig.legend(handles=region_h + style_h, loc="upper center", ncol=6,
               frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=10)
    fig.suptitle("Belgian residential price evolution by region, 2020–2030",
                 fontsize=16, fontweight="bold", y=1.12)
    fig.tight_layout()
    return fig


def main():
    print(f"reading  {HIST_CSV}")
    print(f"reading  {PROJ_CSV}")
    hist = pd.read_csv(HIST_CSV)
    proj = pd.read_csv(PROJ_CSV)
    print(f"historical rows={len(hist):,}  projected rows={len(proj):,}")

    fig = build_figure(hist, proj)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT_PNG}  ({OUT_PNG.stat().st_size / 1024:,.0f} KB)")


if __name__ == "__main__":
    main()
