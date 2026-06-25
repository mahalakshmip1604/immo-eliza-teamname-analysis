#!/usr/bin/env python3
"""Export static PNGs of two charts from the Total-ROI dashboard (totalroi.html).

Rebuilds the exact data the interactive dashboard uses (via totalroi.build_cells)
and renders, for the dashboard's default view, two charts to images/:

  1. images/roi_map_by_municipality.png   — the "ROI map — every municipality"
       (scattergeo: each municipality coloured by total ROI)
  2. images/yield_vs_appreciation.png      — the "Yield vs appreciation" scatter
       (gross yield x  vs  annual appreciation y, colour = total ROI)

Both use the dashboard's default state — horizon 10Y, historical-trend appreciation,
all property types — and its light theme / RdYlGn ROI colour scale. The colour scale
max matches the dashboard (92nd percentile of municipal ROI).

Usage
-----
    python src/export_totalroi_charts.py
    # override the view:
    ROI_HORIZON=20 ROI_SCENARIO=base ROI_PTYPE=apartment python src/export_totalroi_charts.py

ROI_HORIZON in {1,5,10,20} · ROI_SCENARIO in {hist,cons,base,opt} · ROI_PTYPE in {all,house,apartment}

Dependencies: pandas, numpy, plotly, kaleido  (kaleido renders the PNGs offline).
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import totalroi as T  # reuse the dashboard's data pipeline + constants

OUT_DIR = Path(os.environ.get("IMMO_IMAGES", str(HERE.parent / "images")))
OUT_MAP = OUT_DIR / "roi_map_by_municipality.png"
OUT_SCATTER = OUT_DIR / "yield_vs_appreciation.png"

# --- default view (matches the dashboard's initial state), env-overridable ---
HORIZON = int(os.environ.get("ROI_HORIZON", "10"))
SCENARIO = os.environ.get("ROI_SCENARIO", "hist")        # hist | cons | base | opt
PTYPE = os.environ.get("ROI_PTYPE", "all")               # all | house | apartment

# dashboard look & feel
RDYLGN = [[0, "#d73027"], [0.25, "#fc8d59"], [0.5, "#fee08b"],
          [0.7, "#d9ef8b"], [0.85, "#91cf60"], [1, "#1a9850"]]
SCEN_LABEL = {"hist": "historical-trend", "cons": "+2%/yr", "base": "+3%/yr", "opt": "+4.3%/yr"}
PTYPE_LABEL = {"all": "houses & flats", "house": "houses", "apartment": "flats"}


def municipality_rows(cells):
    """Aggregate cells -> one row per municipality for the chosen view (same maths
    as the dashboard's aggregate() + roiOf())."""
    groups = defaultdict(list)
    for c in cells:
        if PTYPE == "all" or c["ptype"] == PTYPE:
            groups[c["refnis"]].append(c)
    rows = []
    for ref, cs in groups.items():
        n = sum(c["n_sale"] for c in cs)
        sale = sum(c["sale_ppsqm"] * c["n_sale"] for c in cs) / n
        rent = sum(c["rent_ppsqm"] * c["n_sale"] for c in cs) / n
        cagr = sum(c["cagr"] * c["n_sale"] for c in cs) / n
        yld = min(rent * 12 / sale * 100, T.YIELD_CAP)
        g = cagr if SCENARIO == "hist" else T.SCEN[SCENARIO]
        roi = yld * HORIZON + ((1 + g) ** HORIZON - 1) * 100
        rows.append(dict(municipality=cs[0]["municipality"], region=cs[0]["region"],
                         lat=cs[0]["lat"], lon=cs[0]["lon"], n_sale=n,
                         yield_pct=yld, appr_pct=g * 100, roi=roi))
    return rows


def base_layout(**extra):
    lay = dict(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
               font=dict(family="Inter,system-ui,sans-serif", color="#23314f", size=13))
    lay.update(extra)
    return lay


def hovertext(rows):
    return [f"<b>{r['municipality']}</b><br>total ROI {r['roi']:.0f}%<br>"
            f"yield {r['yield_pct']:.1f}% · appr {r['appr_pct']:.1f}%/yr<br>"
            f"{r['n_sale']:,} for sale" for r in rows]


def build_map(rows, cmax):
    fig = go.Figure(go.Scattergeo(
        lat=[r["lat"] for r in rows], lon=[r["lon"] for r in rows],
        text=[r["municipality"] for r in rows], hovertext=hovertext(rows), hoverinfo="text",
        marker=dict(
            size=[min(22, 6 + math.sqrt(r["n_sale"])) for r in rows],
            color=[r["roi"] for r in rows], colorscale=RDYLGN, cmin=0, cmax=cmax,
            line=dict(color="#7a8aa3", width=0.6),
            colorbar=dict(title=dict(text="total<br>ROI %", side="right"), thickness=14, len=0.8))))
    fig.update_layout(base_layout(
        height=820, margin=dict(l=0, r=0, t=54, b=0),
        title=dict(text=f"Total ROI by municipality — {HORIZON}Y, {SCEN_LABEL[SCENARIO]} "
                        f"appreciation ({PTYPE_LABEL[PTYPE]})", x=0.02, font=dict(size=17)),
        geo=dict(scope="europe", resolution=50, fitbounds="locations", bgcolor="rgba(0,0,0,0)",
                 showcountries=True, countrycolor="#c7d2e3", showland=True, landcolor="#eaf0f9",
                 showlakes=False, showframe=False, coastlinecolor="#c7d2e3")))
    return fig


def build_scatter(rows, cmax):
    my = float(np.median([r["yield_pct"] for r in rows]))
    mg = float(np.median([r["appr_pct"] for r in rows]))
    fig = go.Figure(go.Scatter(
        x=[r["yield_pct"] for r in rows], y=[r["appr_pct"] for r in rows],
        mode="markers", text=[r["municipality"] for r in rows],
        hovertext=hovertext(rows), hoverinfo="text",
        marker=dict(
            size=[min(34, 7 + math.sqrt(r["n_sale"])) for r in rows],
            color=[r["roi"] for r in rows], colorscale=RDYLGN, cmin=0, cmax=cmax,
            line=dict(color="#7a8aa3", width=0.6),
            colorbar=dict(title=dict(text="total<br>ROI %"), thickness=14, len=0.8))))
    fig.update_layout(base_layout(
        height=760, margin=dict(l=64, r=20, t=54, b=56),
        title=dict(text=f"Yield vs appreciation — Belgian municipalities "
                        f"(colour = {HORIZON}Y total ROI, {PTYPE_LABEL[PTYPE]})",
                   x=0.02, font=dict(size=17)),
        xaxis=dict(title="gross rental yield (%)", gridcolor="#e6ebf4", zeroline=False),
        yaxis=dict(title="annual appreciation (%/yr)", gridcolor="#e6ebf4", zeroline=False),
        shapes=[dict(type="line", x0=my, x1=my, yref="paper", y0=0, y1=1,
                     line=dict(color="#c7d2e3", dash="dot")),
                dict(type="line", yref="y", y0=mg, y1=mg, xref="paper", x0=0, x1=1,
                     line=dict(color="#c7d2e3", dash="dot"))],
        annotations=[dict(x=1, y=1, xref="paper", yref="paper", text="high yield + high growth",
                          showarrow=False, font=dict(size=12, color="#0a8f4f"),
                          xanchor="right", yanchor="top")]))
    return fig


def main():
    if SCENARIO not in ({"hist"} | set(T.SCEN)):
        sys.exit(f"ROI_SCENARIO must be one of hist/{'/'.join(T.SCEN)}")
    cells, _listings, _meta = T.build_cells()
    rows = municipality_rows(cells)
    cmax = float(max(10.0, np.percentile([r["roi"] for r in rows], 92)))  # matches dashboard
    print(f"view: {HORIZON}Y · {SCENARIO} · {PTYPE}  | municipalities: {len(rows)}  | colour max ROI {cmax:.0f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # landscape sizes so the charts read well both standalone and side-by-side in the deck
    for fig, path, w, h in [(build_map(rows, cmax), OUT_MAP, 1120, 760),
                            (build_scatter(rows, cmax), OUT_SCATTER, 1160, 760)]:
        fig.write_image(str(path), width=w, height=h, scale=2)
        print(f"wrote {path}  ({path.stat().st_size/1024:,.0f} KB)")


if __name__ == "__main__":
    main()
