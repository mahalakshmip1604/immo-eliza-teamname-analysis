#!/usr/bin/env python3
"""Total-ROI investor dashboard — combine rental yield with expected capital
appreciation, per Belgian municipality, into one interactive offline HTML report.

For every municipality (and house/flat split) it combines:
  * gross rental yield   = median rent EUR/m2 x 12 / median sale EUR/m2   (from the
    cleaned for-sale & to-rent listings, with the production guard rails)
  * capital appreciation = expected annual property-value growth, either each
    municipality's own historical trend (2015->2025 CAGR from the Statbel price
    series) or a uniform national scenario (+2% / +3% / +4.3% per year)

Total ROI over a horizon H (cumulative %, on the purchase price):
    ROI(H) = gross_yield * H        (rental income, gross)
           + ((1+g)^H - 1) * 100    (capital appreciation, g = annual rate)

The dashboard lets an investor explore where to invest for the best ROI over
1 / 5 / 10 / 20 years, with a systemic geographic drill-down
(Region -> Province -> Nearest city -> Municipality):
  * a sunburst that drills the hierarchy,
  * a ROI map of every municipality (colour = total ROI),
  * a ranking bar grouped at any level,
  * a yield-vs-appreciation scatter,
  * a per-municipality detail (ROI across all horizons, houses vs flats),
  * a reactive Top-5 "where & what to buy" conclusion.

Inputs (resolved relative to this file):
    ../data/cleaned/cleaned_sale_properties.csv
    ../data/cleaned/cleaned_rent_properties.csv
    ../data/property_prices_historical_by_municipality.csv
    ../data/nis_postal_crosswalk.csv
    ../data/external_raw/nis_postal.csv          (municipality centroids)
Output:
    ../reports/totalroi.html

Usage
-----
    python src/totalroi.py
    # or override paths:
    IMMO_DATA=/path/to/data IMMO_OUT=/path/to/out.html python src/totalroi.py

Dependencies: pandas, numpy, plotly  (offline: Plotly runtime is embedded, the
map uses Plotly's built-in geography so no tiles/network are needed to view).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("IMMO_DATA", str(HERE.parent / "data")))
OUT_HTML = Path(os.environ.get("IMMO_OUT", str(HERE.parent / "reports" / "totalroi.html")))
SALE_CSV = DATA_DIR / "cleaned" / "cleaned_sale_properties.csv"
RENT_CSV = DATA_DIR / "cleaned" / "cleaned_rent_properties.csv"
RAW_SALE_CSV = DATA_DIR / "raw" / "forsale.csv"   # carries the listing url (by property_id)
HIST_CSV = DATA_DIR / "property_prices_historical_by_municipality.csv"
XWALK_CSV = DATA_DIR / "nis_postal_crosswalk.csv"
CENTROID_CSV = DATA_DIR / "external_raw" / "nis_postal.csv"

# --------------------------------------------------------------------------- #
# Cleaning thresholds — identical to the production model's guard rails
# --------------------------------------------------------------------------- #
SALE_PRICE_MIN, SALE_PRICE_MAX = 25_000, 15_000_000
RENT_PRICE_MIN, RENT_PRICE_MAX = 200, 25_000
SURFACE_MIN, SURFACE_MAX = 9, 3_000
SALE_PPSQM_MIN, SALE_PPSQM_MAX = 400, 18_000
RENT_PPSQM_MIN, RENT_PPSQM_MAX = 3, 70

# --------------------------------------------------------------------------- #
# Analysis config
# --------------------------------------------------------------------------- #
MIN_SALE_N = 5             # min sale listings of a type in a municipality to include it
MIN_RENT_N = 5             # min rent comparables before falling back to province/region
CAGR_LO, CAGR_HI = 0.0, 0.06     # clip forward appreciation to a realistic 0-6%/yr
YIELD_CAP = 9.0            # clip gross yield to drop rent-fallback artifacts (rural tails)
TOP_LISTINGS = 50          # how many individual for-sale properties to recommend
SCEN = {"cons": 0.02, "base": 0.03, "opt": 0.043}   # uniform annual appreciation scenarios
BASE_G = 0.03              # last-resort appreciation when no historical trend exists
HORIZONS = [1, 5, 10, 20]

REGION_COLOR = {"Brussels": "#e24a76", "Flanders": "#1295b8", "Wallonia": "#e0901f"}


# --------------------------------------------------------------------------- #
# Data loading / cleaning
# --------------------------------------------------------------------------- #
def load_clean(path: Path, kind: str, pc2ref, ref2region, ref2prov) -> pd.DataFrame:
    """Load a listings CSV, apply guard rails, attach refnis/region/province/ptype."""
    df = pd.read_csv(path, low_memory=False)
    for c in ["price", "livable_surface", "postal_code"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["price", "livable_surface", "postal_code", "category"])
    df = df[df["category"].isin(["house", "apartment"])]
    df = df[df["livable_surface"].between(SURFACE_MIN, SURFACE_MAX)]
    df["ppsqm"] = df["price"] / df["livable_surface"]
    if kind == "sale":
        df = df[df["price"].between(SALE_PRICE_MIN, SALE_PRICE_MAX)]
        df = df[df["ppsqm"].between(SALE_PPSQM_MIN, SALE_PPSQM_MAX)]
    else:
        df = df[df["price"].between(RENT_PRICE_MIN, RENT_PRICE_MAX)]
        df = df[df["ppsqm"].between(RENT_PPSQM_MIN, RENT_PPSQM_MAX)]
    df["refnis"] = df["postal_code"].astype(int).map(pc2ref)
    df = df.dropna(subset=["refnis"])
    df["refnis"] = df["refnis"].astype(int)
    df["ptype"] = df["category"].map({"house": "house", "apartment": "apartment"})
    df["region"] = df["refnis"].map(ref2region)
    df["province"] = df["refnis"].map(ref2prov)
    return df.reset_index(drop=True)


def cagr_by_municipality(hist: pd.DataFrame, htype: str) -> dict[int, float]:
    """Annualised price growth ending 2025, starting from the earliest year >=2015
    (else earliest available), for one Statbel property type."""
    d = hist[hist.property_type == htype].dropna(subset=["median_price_eur"])
    out = {}
    for ref, g in d.groupby("refnis"):
        g = g.sort_values("year")
        end = g[g.year == 2025]
        if end.empty:
            continue
        p_end = float(end.median_price_eur.iloc[0])
        cand = g[g.year >= 2015]
        srow = (cand if not cand.empty else g).iloc[0]
        y0, p0, span = int(srow.year), float(srow.median_price_eur), 2025 - int(srow.year)
        if span >= 3 and p0 > 0:
            out[ref] = float(np.clip((p_end / p0) ** (1 / span) - 1, CAGR_LO, CAGR_HI))
    return out


def build_cells() -> tuple[list[dict], dict]:
    # --- geography lookups from the crosswalk ---
    xw = pd.read_csv(XWALK_CSV)
    pc2ref = (xw.sort_values("refnis").drop_duplicates("postal_code")
              .set_index("postal_code")["refnis"])
    geo = xw.drop_duplicates("refnis").set_index("refnis")
    ref2region, ref2prov, ref2name = geo["region"], geo["province"], geo["municipality_fr"]

    # --- municipality centroids (lat, lon) ---
    cg = pd.read_csv(CENTROID_CSV, sep=";", usecols=["refnis_code", "centroid"])
    latlon = cg["centroid"].str.split(",", expand=True)
    cg["lat"] = pd.to_numeric(latlon[0], errors="coerce")
    cg["lon"] = pd.to_numeric(latlon[1], errors="coerce")
    cg = cg.dropna(subset=["lat", "lon"]).groupby("refnis_code")[["lat", "lon"]].mean()
    ref2lat, ref2lon = cg["lat"], cg["lon"]

    # --- listings ---
    sale = load_clean(SALE_CSV, "sale", pc2ref, ref2region, ref2prov)
    rent = load_clean(RENT_CSV, "rent", pc2ref, ref2region, ref2prov)

    ref2city = (sale.dropna(subset=["nearest_city"]).groupby("refnis")["nearest_city"]
                .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "(n/a)"))

    sale_g = sale.groupby(["refnis", "ptype"]).agg(
        n_sale=("ppsqm", "size"), sale_ppsqm=("ppsqm", "median"), price=("price", "median"))
    rent_muni = rent.groupby(["refnis", "ptype"]).agg(
        n_rent=("ppsqm", "size"), rent_ppsqm=("ppsqm", "median"))
    rent_prov = rent.groupby(["province", "ptype"])["ppsqm"].median()
    rent_reg = rent.groupby(["region", "ptype"])["ppsqm"].median()

    # --- appreciation (historical CAGR) with fallback chain ---
    hist_df = pd.read_csv(HIST_CSV)
    cagr_apt = cagr_by_municipality(hist_df, "apartments")
    cagr_house = cagr_by_municipality(hist_df, "houses_all")
    reg_cagr_house = (pd.Series(cagr_house).rename_axis("refnis").reset_index(name="c")
                      .assign(region=lambda d: d.refnis.map(ref2region))
                      .groupby("region")["c"].median().to_dict())
    reg_cagr_apt = (pd.Series(cagr_apt).rename_axis("refnis").reset_index(name="c")
                    .assign(region=lambda d: d.refnis.map(ref2region))
                    .groupby("region")["c"].median().to_dict())

    def cagr_for(ref, ptype, region):
        if ptype == "apartment":
            v = cagr_apt.get(ref) or cagr_house.get(ref) or reg_cagr_apt.get(region) \
                or reg_cagr_house.get(region) or BASE_G
        else:
            v = cagr_house.get(ref) or reg_cagr_house.get(region) or BASE_G
        return float(np.clip(v, CAGR_LO, CAGR_HI))

    # --- assemble one cell per (municipality, property type) ---
    cells = []
    for (ref, pt), row in sale_g.iterrows():
        if row.n_sale < MIN_SALE_N or not np.isfinite(row.sale_ppsqm):
            continue
        region, province = ref2region.get(ref), ref2prov.get(ref)
        if pd.isna(region) or pd.isna(province):
            continue
        rm = rent_muni.loc[(ref, pt)] if (ref, pt) in rent_muni.index else None
        if rm is not None and rm.n_rent >= MIN_RENT_N:
            rent_ppsqm, n_rent, basis = float(rm.rent_ppsqm), int(rm.n_rent), "muni"
        elif (province, pt) in rent_prov.index:
            rent_ppsqm, n_rent, basis = float(rent_prov[(province, pt)]), \
                (int(rm.n_rent) if rm is not None else 0), "prov"
        elif (region, pt) in rent_reg.index:
            rent_ppsqm, n_rent, basis = float(rent_reg[(region, pt)]), \
                (int(rm.n_rent) if rm is not None else 0), "region"
        else:
            continue
        lat, lon = ref2lat.get(ref), ref2lon.get(ref)
        if pd.isna(lat) or pd.isna(lon):
            continue
        cells.append(dict(
            refnis=int(ref), municipality=str(ref2name.get(ref, ref)),
            region=str(region), province=str(province),
            nearest_city=str(ref2city.get(ref, "(n/a)")),
            ptype="apartment" if pt == "apartment" else "house",
            n_sale=int(row.n_sale), n_rent=int(n_rent),
            sale_ppsqm=round(float(row.sale_ppsqm)), rent_ppsqm=round(float(rent_ppsqm), 1),
            gross_yield=round(min(rent_ppsqm * 12 / float(row.sale_ppsqm) * 100, YIELD_CAP), 2),
            price=round(float(row.price)), cagr=round(cagr_for(ref, pt, region), 4),
            rent_basis=basis, lat=round(float(lat), 4), lon=round(float(lon), 4)))

    # --- individual for-sale listings in covered cells (for the Top-50 picks) ---
    covered = {(c["refnis"], c["ptype"]) for c in cells}
    ptidx = {"house": 0, "apartment": 1}
    url_map = {}
    try:
        raw = pd.read_csv(RAW_SALE_CSV, usecols=["property_id", "url"], low_memory=False)
        url_map = dict(zip(raw["property_id"].astype(str), raw["url"].astype(str)))
    except Exception as e:
        print(f"  (no listing urls: {e})")
    sale = sale.copy()
    sale["bedrooms"] = pd.to_numeric(sale.get("bedrooms"), errors="coerce")
    listings = []
    for r in sale.itertuples(index=False):
        if (int(r.refnis), r.ptype) not in covered:
            continue
        url = url_map.get(str(r.property_id), "")
        listings.append([
            int(r.refnis), ptidx[r.ptype], int(round(r.price)),
            int(round(r.livable_surface)), int(round(r.ppsqm)),
            int(r.bedrooms) if pd.notna(r.bedrooms) else -1,
            str(r.locality) if pd.notna(r.locality) else "",
            str(r.property_id), url if isinstance(url, str) and url.startswith("http") else ""])
    print(f"  listing urls matched: {sum(1 for L in listings if L[8]):,}/{len(listings):,}")

    meta = dict(
        regionColors=REGION_COLOR, horizons=HORIZONS, scen=SCEN, topListings=TOP_LISTINGS,
        scenLabel={"hist": "Historical trend (per municipality)",
                   "cons": "Conservative  +2%/yr", "base": "Base  +3%/yr",
                   "opt": "Optimistic  +4.3%/yr"},
        dims=["region", "province", "nearest_city", "municipality"],
        dimLabel={"region": "Region", "province": "Province",
                  "nearest_city": "Nearest city", "municipality": "Municipality"},
        nSale=int(len(sale)), nRent=int(len(rent)), minSaleN=MIN_SALE_N,
        yieldCap=YIELD_CAP)
    return cells, listings, meta


# --------------------------------------------------------------------------- #
# Python-side sanity: default Top-5 (10Y, historical trend, all types)
# --------------------------------------------------------------------------- #
def print_top5(cells):
    H = 10
    ranked = sorted(
        cells,
        key=lambda c: ((1 + c["cagr"]) ** H - 1) * 100 + c["gross_yield"] * H,
        reverse=True)[:5]
    print("\nTop 5 (10Y, historical-trend ROI):")
    for c in ranked:
        roi = ((1 + c["cagr"]) ** H - 1) * 100 + c["gross_yield"] * H
        print(f"  {c['municipality']:<22} {c['ptype']:<9} ROI {roi:5.0f}%  "
              f"(yield {c['gross_yield']:.1f}% · appr {c['cagr']*100:.1f}%/yr · {c['region']})")


# --------------------------------------------------------------------------- #
# HTML / CSS / JS
# --------------------------------------------------------------------------- #
CSS = """
:root{--bg:#f5f8fd;--card:#ffffff;--ink:#1a2540;--muted:#5a6a86;--line:#dde5f1;
      --accent:#2563eb;--accent2:#0e7fa6;}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 600px at 80% -10%,#e8f0ff 0%,var(--bg) 55%);
     color:var(--ink);font-family:Inter,-apple-system,Segoe UI,system-ui,sans-serif;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:0 22px 90px}
header.hero{padding:54px 22px 16px;max-width:1180px;margin:0 auto}
.kicker{letter-spacing:.18em;text-transform:uppercase;font-size:12px;color:var(--accent2);font-weight:700}
h1{font-size:36px;line-height:1.08;margin:10px 0 8px;font-weight:800;
   background:linear-gradient(90deg,#1e3a8a,#2563eb);-webkit-background-clip:text;background-clip:text;color:transparent}
.lede{color:var(--muted);font-size:16px;max-width:860px}
.meta{display:flex;flex-wrap:wrap;gap:12px;margin-top:18px}
.meta .pill{background:#fff;border:1px solid var(--line);border-radius:12px;padding:9px 13px;box-shadow:0 1px 3px rgba(20,40,80,.05)}
.meta .pill b{font-size:19px;display:block;color:var(--ink)}
.meta .pill span{font-size:12px;color:var(--muted)}
.controls{position:sticky;top:0;z-index:20;background:rgba(245,248,253,.94);backdrop-filter:blur(6px);
   border-bottom:1px solid var(--line);margin:18px 0 6px;padding:12px 0}
.controls .row{max-width:1180px;margin:0 auto;padding:0 22px;display:flex;flex-wrap:wrap;gap:18px;align-items:flex-end}
.ctl{display:flex;flex-direction:column;gap:5px}
.ctl label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:700}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fff}
.seg button{background:#fff;color:var(--muted);border:0;padding:7px 12px;font-size:13px;cursor:pointer;font-weight:600}
.seg button.active{background:linear-gradient(135deg,#2563eb,#0e7fa6);color:#fff}
select{background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:10px;padding:7px 10px;font-size:13px}
section.card{background:var(--card);border:1px solid var(--line);
   border-radius:16px;padding:18px 18px 8px;margin:18px 0;box-shadow:0 6px 24px rgba(20,40,80,.06)}
section.card h2{margin:0 0 2px;font-size:18px;font-weight:800}
section.card p.sub{margin:0 0 8px;color:var(--muted);font-size:13px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.concl{background:linear-gradient(180deg,#eef4ff,#f7faff);border:1px solid #cfe0f5}
.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:8px}
@media(max-width:980px){.cards{grid-template-columns:repeat(2,1fr)}}
.invest{background:#fff;border:1px solid var(--line);border-radius:12px;padding:13px;box-shadow:0 1px 3px rgba(20,40,80,.05)}
.invest .rank{font-size:11px;color:var(--accent2);font-weight:800;letter-spacing:.1em}
.invest .name{font-size:16px;font-weight:800;margin:3px 0 1px}
.invest .type{font-size:12px;color:var(--muted);text-transform:capitalize}
.invest .roi{font-size:26px;font-weight:800;margin:8px 0 2px;color:#0a8f4f}
.invest .bits{font-size:12px;color:var(--muted)}
.note{font-size:12.5px;color:var(--muted);border-top:1px dashed var(--line);padding-top:10px;margin-top:6px}
footer{max-width:1180px;margin:30px auto 0;padding:22px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--line)}
footer code{background:#eef2f9;border:1px solid var(--line);border-radius:6px;padding:1px 6px;color:#1e3a8a}
.t50wrap{max-height:580px;overflow:auto;border:1px solid var(--line);border-radius:10px}
table.t50{width:100%;border-collapse:collapse;font-size:12.5px;background:#fff}
table.t50 thead th{position:sticky;top:0;background:#eef2f9;color:var(--muted);text-align:right;
  padding:8px 10px;font-weight:700;border-bottom:1px solid var(--line);white-space:nowrap}
table.t50 thead th:nth-child(-n+4){text-align:left}
table.t50 td{padding:6px 10px;text-align:right;border-bottom:1px solid #eef2f7;white-space:nowrap}
table.t50 td:nth-child(-n+4){text-align:left}
table.t50 tbody tr:hover{background:#f3f7fe}
table.t50 .roi{color:#0a8f4f;font-weight:800}
table.t50 .good{color:#0a8f4f}
table.t50 .bad{color:#d6336c}
table.t50 .rg{color:var(--muted);font-size:11px}
table.t50 a{color:var(--accent);text-decoration:none;font-weight:600}
table.t50 a:hover{text-decoration:underline}
"""

JS = r"""
const CELLS = __CELLS__;
const LISTINGS = __LISTINGS__;
const META  = __META__;
const PT_NAME = ['house','apartment'];
const YCAP = META.yieldCap;              // gross-yield cap (drops rent-fallback artifacts)
const L_REF=0,L_PT=1,L_PRICE=2,L_SURF=3,L_PPSQM=4,L_BEDS=5,L_LOC=6,L_ID=7,L_URL=8;
const RDYLGN = [[0,'#d73027'],[0.25,'#fc8d59'],[0.5,'#fee08b'],[0.7,'#d9ef8b'],[0.85,'#91cf60'],[1,'#1a9850']];
const PCFG = {displayModeBar:false, responsive:true};

const state = {horizon:10, scenario:'hist', ptype:'all', level:'municipality',
               region:'all', hierarchy:'admin', selected:null};

const fmtPct = v => (v>=0?'':'') + v.toFixed(0) + '%';
const scenG  = c => state.scenario==='hist' ? c.cagr : META.scen[state.scenario];
const esc = s => String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
                  .replace(/>/g,'&gt;').replace(/"/g,'&quot;');   // safe HTML for innerHTML

function baseLayout(extra){
  return Object.assign({
    template:'plotly_white', paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'Inter,system-ui,sans-serif', color:'#23314f', size:12},
    margin:{l:60,r:18,t:14,b:46},
    hoverlabel:{bgcolor:'#ffffff', bordercolor:'#d7deeb', font:{color:'#1a2540'}}
  }, extra||{});
}

// ---- filtering + aggregation engine -------------------------------------
function activeCells(){
  return CELLS.filter(c =>
    (state.ptype==='all' || c.ptype===state.ptype) &&
    (state.region==='all' || c.region===state.region));
}
function aggregate(cells){
  let nS=0,nR=0,wSale=0,wRent=0,wCagr=0,wPrice=0;
  for(const c of cells){ nS+=c.n_sale; nR+=c.n_rent;
    wSale+=c.sale_ppsqm*c.n_sale; wRent+=c.rent_ppsqm*c.n_sale;
    wCagr+=c.cagr*c.n_sale; wPrice+=c.price*c.n_sale; }
  const sale=wSale/nS, rent=wRent/nS;
  return {n_sale:nS, n_rent:nR, sale_ppsqm:sale, rent_ppsqm:rent,
          gross_yield:Math.min(rent*12/sale*100, YCAP), cagr:wCagr/nS, price:wPrice/nS};
}
function roiOf(a, H){
  const g = state.scenario==='hist' ? a.cagr : META.scen[state.scenario];
  const cap = (Math.pow(1+g,H)-1)*100, rentInc = a.gross_yield*H;
  return {total:cap+rentInc, cap, rent:rentInc, g};
}
function groupBy(cells, dim){
  const m = new Map();                          // municipalities keyed by refnis (names can collide)
  for(const c of cells){ const gk = dim==='municipality' ? c.refnis : c[dim];
    if(!m.has(gk)) m.set(gk, []); m.get(gk).push(c); }
  const out=[];
  for(const [gk,cs] of m){ const a=aggregate(cs);
    out.push(Object.assign({
      key: dim==='municipality' ? cs[0].municipality : gk,
      refnis: dim==='municipality' ? gk : null,
      region: cs[0].region, cells: cs, roi: roiOf(a, state.horizon)}, a)); }
  return out;
}
const colorFor = g => META.regionColors[g.region] || '#6f8099';

// ---- 1. sunburst (systemic drill-down) ----------------------------------
function renderSunburst(){
  const dims = state.hierarchy==='admin'
      ? ['region','province','municipality'] : ['region','nearest_city','municipality'];
  const nodes = new Map();
  for(const c of activeCells()){
    let parent='', id='';
    dims.forEach((d,i)=>{ id = i===0? String(c[d]) : id+' / '+c[d];
      if(!nodes.has(id)) nodes.set(id,{label:String(c[d]), parent, cells:[]});
      nodes.get(id).cells.push(c); parent=id; });
  }
  const ids=[],labels=[],parents=[],values=[],colors=[],cd=[];
  for(const [id,n] of nodes){ const a=aggregate(n.cells), r=roiOf(a,state.horizon);
    ids.push(id); labels.push(n.label); parents.push(n.parent); values.push(a.n_sale);
    colors.push(r.total); cd.push([r.total, a.gross_yield, a.cagr*100, a.n_sale]); }
  Plotly.react('sun', [{
    type:'sunburst', ids, labels, parents, values, branchvalues:'total', maxdepth:2,
    marker:{colors, colorscale:RDYLGN, cmin:0, cmax:state.cmax, showscale:false,
            line:{color:'#ffffff', width:1}},
    customdata:cd, sort:false,
    hovertemplate:'<b>%{label}</b><br>total ROI %{customdata[0]:.0f}%<br>'+
      'yield %{customdata[1]:.1f}% · appr %{customdata[2]:.1f}%/yr<br>'+
      '%{customdata[3]:,} for sale<extra></extra>'
  }], baseLayout({margin:{l:6,r:6,t:6,b:6}, height:430}), PCFG);
}

// ---- 2. ROI map of every municipality -----------------------------------
function renderMap(){
  const g = groupBy(activeCells(), 'municipality');
  Plotly.react('map', [{
    type:'scattergeo', mode:'markers',
    lat:g.map(x=>x.cells[0].lat), lon:g.map(x=>x.cells[0].lon),
    text:g.map(x=>x.key),
    marker:{size:g.map(x=>Math.min(22,6+Math.sqrt(x.n_sale))), color:g.map(x=>x.roi.total),
            colorscale:RDYLGN, cmin:0, cmax:state.cmax, line:{color:'#7a8aa3',width:.6},
            colorbar:{title:{text:'ROI %',side:'right'}, thickness:12, len:.8, x:1.0}},
    customdata:g.map(x=>[x.roi.total, x.gross_yield, x.cagr*100, x.n_sale, x.refnis||x.cells[0].refnis]),
    hovertemplate:'<b>%{text}</b><br>total ROI %{customdata[0]:.0f}%<br>'+
      'yield %{customdata[1]:.1f}% · appr %{customdata[2]:.1f}%/yr<br>%{customdata[3]:,} for sale<extra></extra>'
  }], baseLayout({height:460, margin:{l:0,r:0,t:0,b:0},
      geo:{scope:'europe', resolution:50, fitbounds:'locations', bgcolor:'rgba(0,0,0,0)',
           showcountries:true, countrycolor:'#c7d2e3', showland:true, landcolor:'#eaf0f9',
           showlakes:false, showframe:false, coastlinecolor:'#c7d2e3'}}),
    {displayModeBar:false, responsive:true});
  bindClick('map', g);
}

// ---- 3. ranking bar (group at any level) --------------------------------
function renderRanking(){
  let g = groupBy(activeCells(), state.level).sort((a,b)=>b.roi.total-a.roi.total);
  const limited = state.level==='municipality' || state.level==='nearest_city';
  if(limited) g = g.slice(0,22);
  g.reverse();
  const xmax = Math.max(...g.map(x=>x.roi.total), 1);
  // NB: outside bar labels aren't measured by Plotly — keep range headroom (xmax*1.18)
  // and margin.r (88) coupled so the ROI % labels never clip.
  Plotly.react('rank', [{
    type:'bar', orientation:'h', x:g.map(x=>x.roi.total), y:g.map(x=>x.key),
    marker:{color:g.map(colorFor)},
    text:g.map(x=>x.roi.total.toFixed(0)+'%'), textposition:'outside', cliponaxis:false,
    textfont:{color:'#1a2540', size:11}, constraintext:'none',
    customdata:g.map(x=>[x.roi.rent, x.roi.cap, x.gross_yield, x.cagr*100, x.n_sale]),
    hovertemplate:'<b>%{y}</b><br>total ROI %{x:.0f}%<br>'+
      'rent %{customdata[0]:.0f}% + appreciation %{customdata[1]:.0f}%<br>'+
      'yield %{customdata[2]:.1f}% · appr %{customdata[3]:.1f}%/yr · %{customdata[4]:,} for sale<extra></extra>'
  }], baseLayout({height:Math.max(320, g.length*22+60),
      margin:{l:175,r:88,t:10,b:40},
      xaxis:{title:{text:state.horizon+'Y total ROI (%)'}, gridcolor:'#e6ebf4', range:[0, xmax*1.18]},
      yaxis:{automargin:true}}), PCFG);
  bindClick('rank', g);
}

// ---- 4. yield vs appreciation scatter -----------------------------------
function renderScatter(){
  const g = groupBy(activeCells(), 'municipality');
  const my = median(g.map(x=>x.gross_yield)), mg = median(g.map(x=>x.cagr*100));
  Plotly.react('scatter', [{
    type:'scatter', mode:'markers', x:g.map(x=>x.gross_yield), y:g.map(x=>x.cagr*100),
    text:g.map(x=>x.key),
    marker:{size:g.map(x=>Math.min(34,7+Math.sqrt(x.n_sale))), color:g.map(x=>x.roi.total),
            colorscale:RDYLGN, cmin:0, cmax:state.cmax, line:{color:'#7a8aa3',width:.6},
            colorbar:{title:{text:'ROI %'}, thickness:12, len:.8}},
    customdata:g.map(x=>[x.roi.total, x.n_sale, x.refnis||x.cells[0].refnis]),
    hovertemplate:'<b>%{text}</b><br>yield %{x:.1f}% · appr %{y:.1f}%/yr<br>'+
      'total ROI %{customdata[0]:.0f}% · %{customdata[1]:,} for sale<extra></extra>'
  }], baseLayout({height:430, margin:{l:54,r:18,t:10,b:46},
      xaxis:{title:{text:'gross rental yield (%)'}, gridcolor:'#e6ebf4', zeroline:false},
      yaxis:{title:{text:'annual appreciation (%/yr)'}, gridcolor:'#e6ebf4', zeroline:false},
      shapes:[{type:'line',x0:my,x1:my,yref:'paper',y0:0,y1:1,line:{color:'#c7d2e3',dash:'dot'}},
              {type:'line',yref:'y',y0:mg,y1:mg,xref:'paper',x0:0,x1:1,line:{color:'#c7d2e3',dash:'dot'}}],
      annotations:[{x:1,y:1,xref:'paper',yref:'paper',text:'high yield + high growth',
                    showarrow:false,font:{size:11,color:'#0a8f4f'},xanchor:'right',yanchor:'top'}]}),
    PCFG);
  bindClick('scatter', g);
}

// ---- 5. municipality detail (ROI across horizons, houses vs flats) ------
function renderDetail(){
  let ref = state.selected;
  if(ref==null){ const g=groupBy(activeCells(),'municipality').sort((a,b)=>b.roi.total-a.roi.total);
    ref = g.length? (g[0].refnis||g[0].cells[0].refnis):null; }
  const mine = CELLS.filter(c=>c.refnis===ref);
  const name = mine.length? mine[0].municipality : '—';
  const traces=[];
  for(const pt of ['house','apartment']){
    const c = mine.find(x=>x.ptype===pt); if(!c) continue;
    traces.push({type:'bar', name:pt==='house'?'Houses':'Flats',
      x:META.horizons.map(h=>h+'Y'),
      y:META.horizons.map(h=>roiOf(aggregate([c]),h).total),
      marker:{color: pt==='house'? '#2563eb':'#db2777'},
      customdata:META.horizons.map(h=>{const r=roiOf(aggregate([c]),h);return [r.rent,r.cap,c.gross_yield,c.cagr*100];}),
      hovertemplate:'<b>'+(pt==='house'?'House':'Flat')+' · %{x}</b><br>total ROI %{y:.0f}%<br>'+
        'rent %{customdata[0]:.0f}% + appr %{customdata[1]:.0f}%<br>'+
        'yield %{customdata[2]:.1f}% · appr %{customdata[3]:.1f}%/yr<extra></extra>'});
  }
  document.getElementById('detail-name').textContent = name;
  Plotly.react('detail', traces, baseLayout({height:330, barmode:'group',
      margin:{l:54,r:18,t:10,b:36}, legend:{orientation:'h',y:1.12,x:0},
      xaxis:{title:{text:'holding period'}}, yaxis:{title:{text:'cumulative ROI (%)'},gridcolor:'#e6ebf4'}}),
    PCFG);
}

// ---- 6. reactive Top-5 conclusion (place + property type) ---------------
function renderConclusion(){
  const H=state.horizon;
  const pool = activeCells().filter(c=>c.n_sale>=META.minSaleN)
     .map(c=>({c, r:roiOf(aggregate([c]),H)}))
     .sort((a,b)=>b.r.total-a.r.total).slice(0,5);
  const html = pool.map((o,i)=>{
    const c=o.c, r=o.r;
    return '<div class="invest"><div class="rank">#'+(i+1)+'</div>'+
      '<div class="name">'+esc(c.municipality)+'</div>'+
      '<div class="type">'+(c.ptype==='house'?'House':'Flat')+' · '+esc(c.region)+'</div>'+
      '<div class="roi">'+r.total.toFixed(0)+'%</div>'+
      '<div class="bits">rent '+r.rent.toFixed(0)+'% + appr '+r.cap.toFixed(0)+'%<br>'+
      'yield '+c.gross_yield.toFixed(1)+'% · '+(c.cagr*100).toFixed(1)+'%/yr · ~€'+
      (Math.round(c.price/1000))+'k</div></div>';
  }).join('');
  document.getElementById('top5').innerHTML = html;
  document.getElementById('concl-h').textContent = H;
  document.getElementById('concl-s').textContent = META.scenLabel[state.scenario];
}

// ---- 7. Top-50 individual properties to buy -----------------------------
function renderTop50(){
  const cellMap=new Map(); CELLS.forEach(c=>cellMap.set(c.refnis+'|'+c.ptype,c));
  const H=state.horizon, rows=[];
  for(const L of LISTINGS){
    const pt=PT_NAME[L[L_PT]];
    if(state.ptype!=='all' && state.ptype!==pt) continue;
    const c=cellMap.get(L[L_REF]+'|'+pt); if(!c) continue;
    if(state.region!=='all' && c.region!==state.region) continue;
    let yld=c.rent_ppsqm*12/L[L_PPSQM]*100; if(yld>YCAP) yld=YCAP;
    const g=state.scenario==='hist'?c.cagr:META.scen[state.scenario];
    const roi=yld*H+(Math.pow(1+g,H)-1)*100;
    rows.push({L,c,pt,yld,g,roi});
  }
  rows.sort((a,b)=>b.roi-a.roi);
  const top=rows.slice(0,META.topListings);
  let h='<table class="t50"><thead><tr><th>#</th><th>Municipality</th><th>Town</th><th>Type</th>'+
    '<th>Price</th><th>m²</th><th>Beds</th><th>€/m²</th><th>vs area</th><th>Yield</th><th>Appr/yr</th>'+
    '<th>ROI '+H+'Y</th><th>Listing</th></tr></thead><tbody>';
  top.forEach((o,i)=>{ const c=o.c,L=o.L;
    const rel=Math.round((L[L_PPSQM]/c.sale_ppsqm-1)*100);
    const link = L[L_URL] ? '<a href="'+esc(L[L_URL])+'" target="_blank" rel="noopener">open ↗</a>' : '–';
    h+='<tr><td title="id '+esc(L[L_ID])+'">'+(i+1)+'</td>'+
      '<td>'+esc(c.municipality)+' <span class="rg">'+esc(c.region)+'</span></td>'+
      '<td>'+esc(L[L_LOC]||'')+'</td><td>'+(o.pt==='house'?'House':'Flat')+'</td>'+
      '<td>€'+L[L_PRICE].toLocaleString()+'</td><td>'+L[L_SURF]+'</td>'+
      '<td>'+(L[L_BEDS]<0?'–':L[L_BEDS])+'</td><td>€'+L[L_PPSQM].toLocaleString()+'</td>'+
      '<td class="'+(rel<0?'good':'bad')+'">'+(rel>0?'+':'')+rel+'%</td>'+
      '<td>'+o.yld.toFixed(1)+'%</td><td>'+(o.g*100).toFixed(1)+'%</td>'+
      '<td class="roi">'+o.roi.toFixed(0)+'%</td><td>'+link+'</td></tr>';
  });
  document.getElementById('top50tbl').innerHTML = h+'</tbody></table>';
  document.getElementById('t50-h').textContent = H;
  document.getElementById('t50-n').textContent = top.length;
}

// ---- shared helpers -----------------------------------------------------
function pctl(arr,p){ const s=[...arr].sort((a,b)=>a-b); if(!s.length) return 0;
  const i=(s.length-1)*p, lo=Math.floor(i), hi=Math.ceil(i); return s[lo]+(s[hi]-s[lo])*(i-lo); }
function computeCmax(){ const g=groupBy(activeCells(),'municipality');
  state.cmax = Math.max(10, pctl(g.map(x=>x.roi.total),0.92)); }
function median(a){ const s=[...a].sort((x,y)=>x-y); const n=s.length;
  return n? (n%2? s[(n-1)/2] : (s[n/2-1]+s[n/2])/2) : 0; }
function bindClick(div, g){
  const el=document.getElementById(div); el._g=g;     // refresh group data each render
  if(el._bound) return; el._bound=true;               // attach the listener only once
  el.on('plotly_click', ev=>{
    const item=el._g[ev.points[0].pointNumber]; if(!item || !item.cells) return;
    const refs=new Set(item.cells.map(c=>c.refnis));  // drill only when the bar = one municipality
    if(refs.size!==1) return;
    state.selected=[...refs][0]; renderDetail();
    document.getElementById('detail').scrollIntoView({behavior:'smooth',block:'center'});
  });
}

function render(){ computeCmax(); renderSunburst(); renderMap(); renderRanking();
                   renderScatter(); renderDetail(); renderConclusion(); renderTop50(); }

// ---- controls -----------------------------------------------------------
function wireSeg(id, key, cast){
  document.querySelectorAll('#'+id+' button').forEach(b=>{
    b.onclick=()=>{ state[key]=cast?cast(b.dataset.v):b.dataset.v;
      document.querySelectorAll('#'+id+' button').forEach(x=>x.classList.toggle('active',x===b));
      render(); };
  });
}
wireSeg('horizon','horizon',Number);
wireSeg('ptype','ptype');
document.getElementById('scenario').onchange=e=>{state.scenario=e.target.value; render();};
document.getElementById('level').onchange   =e=>{state.level=e.target.value; render();};
document.getElementById('region').onchange  =e=>{state.region=e.target.value; render();};
document.getElementById('hierarchy').onchange=e=>{state.hierarchy=e.target.value; renderSunburst();};
render();
"""


def render_html(cells, listings, meta):
    body = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Immo Eliza · Total-ROI Investor Dashboard</title>
<style>__CSS__</style><script>__PLOTLY__</script></head><body>
<header class="hero">
  <div class="kicker">Immo Eliza · Belgian residential market</div>
  <h1>What properties should you buy for the highest ROI?</h1>
  <p class="lede">Total ROI = <b>rental income</b> (gross yield) + <b>property-value growth</b>. For every
     Belgian municipality we combine the gross rental yield (from the cleaned for-sale &amp; to-rent
     listings) with expected capital appreciation, then rank <b>where — and what — to buy</b> for the
     best total return over 1, 5, 10 and 20 years. Drill region → province → municipality, compare houses
     vs flats, then jump to the <b>Top-5 areas</b> and <b>Top-50 individual properties</b> below.</p>
  <div class="meta">
    <div class="pill"><b id="m-cells"></b><span>municipality × type cells</span></div>
    <div class="pill"><b>__NSALE__</b><span>clean sale listings</span></div>
    <div class="pill"><b>__NRENT__</b><span>clean rent listings</span></div>
    <div class="pill"><b>1·5·10·20Y</b><span>investment horizons</span></div>
  </div>
</header>

<div class="controls"><div class="row">
  <div class="ctl"><label>Horizon</label><div class="seg" id="horizon">
    <button data-v="1">1Y</button><button data-v="5">5Y</button>
    <button data-v="10" class="active">10Y</button><button data-v="20">20Y</button></div></div>
  <div class="ctl"><label>Property type</label><div class="seg" id="ptype">
    <button data-v="all" class="active">All</button><button data-v="house">Houses</button>
    <button data-v="apartment">Flats</button></div></div>
  <div class="ctl"><label>Appreciation scenario</label>
    <select id="scenario">
      <option value="hist" selected>Historical trend (per municipality)</option>
      <option value="cons">Conservative +2%/yr</option>
      <option value="base">Base +3%/yr</option>
      <option value="opt">Optimistic +4.3%/yr</option></select></div>
  <div class="ctl"><label>Region filter</label>
    <select id="region"><option value="all" selected>All Belgium</option>
      <option value="Flanders">Flanders</option><option value="Wallonia">Wallonia</option>
      <option value="Brussels">Brussels</option></select></div>
  <div class="ctl"><label>Rank / group by</label>
    <select id="level"><option value="region">Region</option><option value="province">Province</option>
      <option value="nearest_city">Nearest city</option>
      <option value="municipality" selected>Municipality</option></select></div>
</div></div>

<div class="wrap">
  <section class="card">
    <h2>Geographic drill-down — total ROI by hierarchy</h2>
    <p class="sub">Click a wedge to drill in (size = listings for sale, colour = total ROI).
      <span style="float:right"><select id="hierarchy">
        <option value="admin" selected>Region → Province → Municipality</option>
        <option value="city">Region → Nearest city → Municipality</option></select></span></p>
    <div id="sun"></div>
  </section>

  <section class="card">
    <h2>ROI map — every municipality</h2>
    <p class="sub">Each municipality coloured by total ROI for the selected horizon; size = listings for sale. Click to inspect.</p>
    <div id="map"></div>
  </section>

  <div class="grid2">
    <section class="card">
      <h2>Ranking — top by total ROI</h2>
      <p class="sub">Grouped at the “rank / group by” level. Click a bar to drill into a municipality.</p>
      <div id="rank"></div>
    </section>
    <section class="card">
      <h2>Yield vs appreciation</h2>
      <p class="sub">The two return engines. Top-right = high rent <i>and</i> high growth. Click a point.</p>
      <div id="scatter"></div>
    </section>
  </div>

  <section class="card">
    <h2>Drill-down — <span id="detail-name">—</span>: ROI by holding period</h2>
    <p class="sub">Cumulative total ROI at 1 / 5 / 10 / 20 years, houses vs flats (uses the selected scenario).</p>
    <div id="detail"></div>
  </section>

  <section class="card concl">
    <h2>Conclusion — top 5 places &amp; property types to invest (<span id="concl-h"></span>Y)</h2>
    <p class="sub">Best total ROI given the current filters · appreciation: <span id="concl-s"></span></p>
    <div class="cards" id="top5"></div>
    <p class="note">Total ROI = gross rental income (yield × years) + capital appreciation
      ((1+g)<sup>years</sup>−1). “Historical trend” sets g to each municipality’s own 2015→2025 price
      CAGR (a trend extrapolation, not a forecast); the uniform scenarios apply a single national rate
      to all. Yields are <b>gross</b> (before costs/taxes/vacancy); rent comparables fall back to
      province/region level where a municipality has few rent listings.</p>
  </section>

  <section class="card">
    <h2>Top <span id="t50-n">50</span> properties to buy now — highest expected <span id="t50-h"></span>Y ROI</h2>
    <p class="sub">Actual for-sale listings, ranked by expected total ROI = estimated gross rental yield
      (area rent €/m² ÷ this listing’s €/m²) + appreciation. “vs area” = this listing’s €/m² versus its
      municipality median (negative = below-market = a better deal). Reacts to every control above.</p>
    <div class="t50wrap"><div id="top50tbl"></div></div>
  </section>
</div>

<footer>
  <b>Method.</b> Built from the cleaned listings (<code>cleaned_sale_properties.csv</code>,
  <code>cleaned_rent_properties.csv</code>) joined by postal code → NIS to the Statbel municipal price
  series. Guard rails: sale €25k–€15M, rent €200–€25k, area 9–3000 m², €/m² recomputed and bounded.
  Only municipality×type cells with ≥__MINSALE__ sale listings are shown. Appreciation beyond published
  forecasts is scenario-based — see the conclusion note. Generated by <code>src/totalroi.py</code>.
</footer>

<script>__JS__</script>
<script>document.getElementById('m-cells').textContent = CELLS.length;</script>
</body></html>"""
    js = (JS.replace("__CELLS__", json.dumps(cells, separators=(",", ":")))
            .replace("__LISTINGS__", json.dumps(listings, separators=(",", ":")))
            .replace("__META__", json.dumps(meta, separators=(",", ":"))))
    return (body.replace("__CSS__", CSS)
                .replace("__PLOTLY__", get_plotlyjs())
                .replace("__NSALE__", f"{meta['nSale']:,}")
                .replace("__NRENT__", f"{meta['nRent']:,}")
                .replace("__MINSALE__", str(meta["minSaleN"]))
                .replace("__JS__", js))


def main():
    print("building cells…")
    cells, listings, meta = build_cells()
    print(f"cells (municipality × type): {len(cells)}  "
          f"| municipalities: {len({c['refnis'] for c in cells})}")
    print(f"  houses: {sum(c['ptype']=='house' for c in cells)}  "
          f"flats: {sum(c['ptype']=='apartment' for c in cells)}  "
          f"| recommendable listings: {len(listings):,}")
    print_top5(cells)

    html = render_html(cells, listings, meta)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nwrote {OUT_HTML}  ({OUT_HTML.stat().st_size/1024:,.0f} KB)")


if __name__ == "__main__":
    main()
