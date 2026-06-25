#!/usr/bin/env python3
"""City & regional rental-yield charts — static PNG render of
reports/cityregional.ipynb.

The notebook is an interactive Plotly report (pick a city from a dropdown).
This script renders the two static equivalents the deck needs:

  1. images/cityregional_overview.png
       Properties for sale per nearest city; bar height = count, bar colour =
       gross rental yield % (= median rent €/m² x 12 / median sale €/m²).
  2. images/cityregional_distance_bands_<page>.png  (one PNG per 6 cities)
       Every nearest city, yield broken down by distance band
       (0-1, 1-2.5, 2.5-5, 5-10, 10-15, 15-20, >20 km), 6 cities per page,
       ordered by listing volume.

Usage
-----
    python src/cityregional.py
    # or override paths:
    IMMO_DATA=/path/to/data IMMO_IMAGES=/path/to/images python src/cityregional.py

Reads (resolved relative to this file):
    ../data/cleaned/cleaned_sale_properties.csv
    ../data/cleaned/cleaned_rent_properties.csv
Writes:
    ../images/cityregional_overview.png
    ../images/cityregional_distance_bands_1.png ... _N.png  (all cities, 6 per page)

Dependencies: pandas, numpy, matplotlib.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless: write PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("IMMO_DATA", str(HERE.parent / "data")))
OUT_DIR = Path(os.environ.get("IMMO_IMAGES", str(HERE.parent / "images")))
SALE_CSV = DATA_DIR / "cleaned" / "cleaned_sale_properties.csv"
RENT_CSV = DATA_DIR / "cleaned" / "cleaned_rent_properties.csv"
OUT_OVERVIEW = OUT_DIR / "cityregional_overview.png"
# distance breakdown is paginated: cityregional_distance_bands_<page>.png
DISTANCE_STEM = "cityregional_distance_bands"

# --------------------------------------------------------------------------- #
# Cleaning thresholds — mirror the notebook / src/yieldinsight.py guard rails
# --------------------------------------------------------------------------- #
SALE_PRICE_MIN, SALE_PRICE_MAX = 25_000, 15_000_000
RENT_PRICE_MIN, RENT_PRICE_MAX = 200, 25_000
SURFACE_MIN, SURFACE_MAX = 9, 3_000
SALE_PPSQM_MIN, SALE_PPSQM_MAX = 400, 18_000
RENT_PPSQM_MIN, RENT_PPSQM_MAX = 3, 70

# --------------------------------------------------------------------------- #
# Analysis config
# --------------------------------------------------------------------------- #
DIST_BINS = [0, 1, 2.5, 5, 10, 15, 20, np.inf]
DIST_LABELS = ["0–1 km", "1–2.5 km", "2.5–5 km", "5–10 km",
               "10–15 km", "15–20 km", ">20 km"]
MIN_RENT_N = 5             # min comparable rent listings before we trust a yield
PAGE_SIZE = 6              # cities per distance-breakdown PNG (paginated over all cities)
CSCALE = "RdYlGn"          # red = low yield, green = high yield
NAN_COLOR = "#cccccc"      # bars with no trustworthy yield


# --------------------------------------------------------------------------- #
# Data loading / cleaning
# --------------------------------------------------------------------------- #
def load_clean(path: Path, kind: str) -> pd.DataFrame:
    """Load a listings CSV, apply guard rails, recompute the authoritative €/m²."""
    df = pd.read_csv(path, low_memory=False)
    for c in ["price", "livable_surface", "nearest_city_distance_km"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["price", "livable_surface", "nearest_city"])
    df = df[df["livable_surface"].between(SURFACE_MIN, SURFACE_MAX)]
    df["ppsqm"] = df["price"] / df["livable_surface"]
    if kind == "sale":
        df = df[df["price"].between(SALE_PRICE_MIN, SALE_PRICE_MAX)]
        df = df[df["ppsqm"].between(SALE_PPSQM_MIN, SALE_PPSQM_MAX)]
    else:
        df = df[df["price"].between(RENT_PRICE_MIN, RENT_PRICE_MAX)]
        df = df[df["ppsqm"].between(RENT_PPSQM_MIN, RENT_PPSQM_MAX)]
    return df.reset_index(drop=True)


def yield_pct(rent_ppsqm: float, sale_ppsqm: float) -> float:
    """Gross annual rental yield % = rent €/m² x 12 / sale €/m² x 100."""
    if not sale_ppsqm or np.isnan(sale_ppsqm) or rent_ppsqm is None or np.isnan(rent_ppsqm):
        return np.nan
    return rent_ppsqm * 12 / sale_ppsqm * 100


def city_table(sale: pd.DataFrame, rent: pd.DataFrame) -> pd.DataFrame:
    """Per-city: count for sale + gross yield vs comparable rent listings."""
    sg = sale.groupby("nearest_city").agg(n_sale=("ppsqm", "size"),
                                          sale_ppsqm=("ppsqm", "median"))
    rg = rent.groupby("nearest_city").agg(n_rent=("ppsqm", "size"),
                                          rent_ppsqm=("ppsqm", "median"))
    t = sg.join(rg, how="left")
    t["rent_ppsqm"] = t["rent_ppsqm"].where(t["n_rent"] >= MIN_RENT_N)
    t["yield"] = t["rent_ppsqm"] * 12 / t["sale_ppsqm"] * 100
    return t.sort_values("n_sale", ascending=False).reset_index()


def city_detail(sale: pd.DataFrame, rent: pd.DataFrame, city: str) -> pd.DataFrame:
    """Yield per distance band for one city; rent comparable falls back to the
    city-wide median when a band has too few rent listings."""
    s = sale[sale["nearest_city"] == city].copy()
    r = rent[rent["nearest_city"] == city].copy()
    s["band"] = pd.cut(s["nearest_city_distance_km"], DIST_BINS, labels=DIST_LABELS, right=False)
    r["band"] = pd.cut(r["nearest_city_distance_km"], DIST_BINS, labels=DIST_LABELS, right=False)

    city_rent_ppsqm = r["ppsqm"].median() if len(r) >= MIN_RENT_N else np.nan
    rows = []
    for lab in DIST_LABELS:
        sb = s[s["band"] == lab]
        rb = r[r["band"] == lab]
        n_sale = len(sb)
        sale_ppsqm = sb["ppsqm"].median() if n_sale else np.nan
        if len(rb) >= MIN_RENT_N:
            rent_ppsqm = rb["ppsqm"].median()
        else:
            rent_ppsqm = city_rent_ppsqm
        rows.append(dict(band=lab, n_sale=n_sale,
                         yield_pct=yield_pct(rent_ppsqm, sale_ppsqm)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Colour helpers
# --------------------------------------------------------------------------- #
def yield_norm(cities: pd.DataFrame):
    """Shared colour scale (5th-95th pct of valid yields), like the notebook."""
    valid = cities["yield"].dropna()
    cmin, cmax = float(np.floor(valid.quantile(0.05))), float(np.ceil(valid.quantile(0.95)))
    return Normalize(vmin=cmin, vmax=cmax), plt.get_cmap(CSCALE)


def bar_colors(values, norm, cmap):
    return [NAN_COLOR if pd.isna(v) else cmap(norm(v)) for v in values]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_overview(cities: pd.DataFrame, norm, cmap) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 6.5))
    colors = bar_colors(cities["yield"], norm, cmap)
    ax.bar(cities["nearest_city"], cities["n_sale"], color=colors, edgecolor="white", lw=0.4)

    for x, (n, y) in enumerate(zip(cities["n_sale"], cities["yield"])):
        if not pd.isna(y):
            ax.text(x, n, f"{y:.1f}%", ha="center", va="bottom",
                    fontsize=7, color="0.3", rotation=90)

    ax.set_title("Properties for sale per nearest city — colour = gross rental yield (%)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("# properties for sale")
    ax.set_xlabel("nearest city")
    ax.set_xticks(range(len(cities)))
    ax.set_xticklabels(cities["nearest_city"], rotation=-40, ha="left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.margins(x=0.01)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
    cbar.set_label("Gross rental yield (%)")
    fig.tight_layout()
    return fig


def fig_distance(details: dict[str, pd.DataFrame], norm, cmap, page_label: str) -> plt.Figure:
    cities = list(details)
    ncols = 3
    nrows = math.ceil(len(cities) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)

    for ax, city in zip(axes.flat, cities):
        d = details[city]
        colors = bar_colors(d["yield_pct"], norm, cmap)
        ax.bar(d["band"], d["n_sale"], color=colors, edgecolor="white", lw=0.4)
        for x, (n, y) in enumerate(zip(d["n_sale"], d["yield_pct"])):
            if not pd.isna(y):
                ax.text(x, n, f"{y:.1f}%", ha="center", va="bottom", fontsize=8, color="0.3")
        ax.set_title(city, fontsize=13, fontweight="bold")
        ax.set_ylabel("# for sale")
        ax.tick_params(axis="x", labelrotation=40, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes.flat[len(cities):]:          # hide unused panels
        ax.set_visible(False)

    fig.suptitle(f"Gross rental yield by distance band — {page_label} (colour = yield %)",
                 fontsize=15, fontweight="bold", y=1.0)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Gross rental yield (%)")
    return fig


def main():
    print(f"reading  {SALE_CSV}")
    print(f"reading  {RENT_CSV}")
    sale = load_clean(SALE_CSV, "sale")
    rent = load_clean(RENT_CSV, "rent")
    print(f"clean sale={len(sale):,}  clean rent={len(rent):,}  "
          f"cities={sale['nearest_city'].nunique()}")

    cities = city_table(sale, rent)
    norm, cmap = yield_norm(cities)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) overview (all cities, one PNG)
    fig = fig_overview(cities, norm, cmap)
    fig.savefig(OUT_OVERVIEW, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_OVERVIEW}  ({OUT_OVERVIEW.stat().st_size / 1024:,.0f} KB)")

    # 2) distance breakdown for ALL cities, paginated PAGE_SIZE per PNG
    city_list = cities["nearest_city"].tolist()          # ordered by listing volume
    total = len(city_list)
    pages = [city_list[i:i + PAGE_SIZE] for i in range(0, total, PAGE_SIZE)]
    for pi, page_cities in enumerate(pages, 1):
        details = {c: city_detail(sale, rent, c) for c in page_cities}
        lo = (pi - 1) * PAGE_SIZE + 1
        label = f"cities {lo}–{lo + len(page_cities) - 1} of {total} (by listing volume)"
        fig = fig_distance(details, norm, cmap, label)
        path = OUT_DIR / f"{DISTANCE_STEM}_{pi}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}  ({path.stat().st_size / 1024:,.0f} KB)")


if __name__ == "__main__":
    main()
