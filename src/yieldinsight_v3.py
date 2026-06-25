#!/usr/bin/env python3
"""Single-insight Immo-Eliza report → rental yield by municipality (v3, PNG).

Same standalone rental-yield insight as ``yieldinsight_v2.py`` — built from the
team's *cleaned* datasets with the production guard rails — but rendered as a
static **PNG** (matplotlib) into ``images/`` instead of an interactive HTML page:

    in  : data/cleaned/cleaned_sale_properties.csv
          data/cleaned/cleaned_rent_properties.csv
    out : images/yieldinsight_by_municipality.png

The chart is the diverging horizontal bar of the top & bottom 12 municipalities
by gross rental yield (= median rent €/m² x 12 / median sale €/m²), coloured by
region, with the national gross yield as a reference line.

Usage
-----
    python src/yieldinsight_v3.py
    # or override paths:
    IMMO_DATA=/path/to/cleaned IMMO_IMAGES=/path/to/images python src/yieldinsight_v3.py

Dependencies: pandas, numpy, matplotlib  (headless: no network, no browser).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless: write PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# Paths — read the cleaned datasets, write a PNG into reports/
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("IMMO_DATA", str(HERE.parent / "data" / "cleaned")))
OUT_DIR = Path(os.environ.get("IMMO_IMAGES", str(HERE.parent / "images")))
SALE_CSV = DATA_DIR / "cleaned_sale_properties.csv"
RENT_CSV = DATA_DIR / "cleaned_rent_properties.csv"
OUT_PNG = OUT_DIR / "yieldinsight_by_municipality.png"

# --------------------------------------------------------------------------- #
# Cleaning thresholds — identical to the production model's guard rails
# --------------------------------------------------------------------------- #
SALE_PRICE_MIN, SALE_PRICE_MAX = 25_000, 15_000_000
RENT_PRICE_MIN, RENT_PRICE_MAX = 200, 25_000
SURFACE_MIN, SURFACE_MAX = 9, 3_000
SALE_PPSQM_MIN, SALE_PPSQM_MAX = 400, 18_000
RENT_PPSQM_MIN, RENT_PPSQM_MAX = 3, 70

# --------------------------------------------------------------------------- #
# Look & feel — region palette mirrors yieldinsight_v2.py
# --------------------------------------------------------------------------- #
REGION_COLOR = {"Brussels": "#fb6f92", "Flanders": "#22a7c4", "Wallonia": "#f5b53d"}
SLATE = "#6f8099"
MIN_N = 15                  # min sale & rent listings before a municipality counts
TOP_BOTTOM = 12             # how many to show at each end


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


# --------------------------------------------------------------------------- #
# Per-area sale/rent yield table
# --------------------------------------------------------------------------- #
def area_yield_table(sale: pd.DataFrame, rent: pd.DataFrame, min_n: int = MIN_N) -> pd.DataFrame:
    s = sale.dropna(subset=["locality"])
    r = rent.dropna(subset=["locality"])
    sg = s.groupby("locality").agg(
        sale_ppsqm=("ppsqm", "median"),
        n_sale=("ppsqm", "size"),
        region=("region", lambda x: x.mode().iat[0] if not x.mode().empty else None),
    )
    rg = r.groupby("locality").agg(rent_ppsqm=("ppsqm", "median"), n_rent=("ppsqm", "size"))
    m = sg.join(rg, how="inner")
    m = m[(m.n_sale >= min_n) & (m.n_rent >= min_n)].copy()
    m["yield"] = m["rent_ppsqm"] * 12 / m["sale_ppsqm"] * 100
    return m.reset_index().sort_values("yield", ascending=False)


# --------------------------------------------------------------------------- #
# The insight — rental yield by area (diverging horizontal bar)
# --------------------------------------------------------------------------- #
def build_figure(sale: pd.DataFrame, rent: pd.DataFrame, area: pd.DataFrame) -> plt.Figure:
    nat_yield = rent["ppsqm"].median() * 12 / sale["ppsqm"].median() * 100
    sub = pd.concat([area.tail(TOP_BOTTOM), area.head(TOP_BOTTOM)]).sort_values("yield")
    colors = [REGION_COLOR.get(r, SLATE) for r in sub["region"]]
    y = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(y, sub["yield"], color=colors, edgecolor="white", lw=0.5)
    for yi, v in zip(y, sub["yield"]):
        ax.text(v + 0.06, yi, f"{v:.1f}%", va="center", fontsize=8, color="0.3")

    ax.axvline(nat_yield, ls="--", color="0.45", lw=1.3)
    ax.text(nat_yield, len(sub) - 0.4, f"  national {nat_yield:.1f}%",
            color="0.4", fontsize=9, va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(sub["locality"], fontsize=9)
    ax.set_xlim(0, sub["yield"].max() * 1.10)
    ax.set_xlabel("gross rental yield (%)")
    ax.grid(axis="x", alpha=0.3)
    ax.margins(y=0.01)

    best, worst = area.iloc[0], area.iloc[-1]
    ax.set_title(
        f"Gross rental yield by Belgian municipality — top & bottom {TOP_BOTTOM}\n"
        f"{best['locality']} {best['yield']:.1f}%  vs  {worst['locality']} {worst['yield']:.1f}%  "
        f"(≥{MIN_N} sale & rent listings each)",
        fontsize=13, fontweight="bold")

    handles = [Patch(color=c, label=r) for r, c in REGION_COLOR.items()]
    ax.legend(handles=handles, loc="lower right", frameon=False, title="Region")
    fig.tight_layout()
    return fig


def main():
    print(f"reading  {SALE_CSV}")
    print(f"reading  {RENT_CSV}")
    sale = load_clean(SALE_CSV, "sale")
    rent = load_clean(RENT_CSV, "rent")
    print(f"clean sale={len(sale):,}  clean rent={len(rent):,}")

    area = area_yield_table(sale, rent)
    print(f"matched yield areas (n>={MIN_N} both sides): {len(area)}")
    best, worst = area.iloc[0], area.iloc[-1]
    print(f"best  {best['locality']:<20} {best['yield']:.1f}%")
    print(f"worst {worst['locality']:<20} {worst['yield']:.1f}%")

    fig = build_figure(sale, rent, area)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT_PNG}  ({OUT_PNG.stat().st_size / 1024:,.0f} KB)")


if __name__ == "__main__":
    main()
