# immo-eliza-teamname-analysis

The real estate company Immo Eliza wants to establish itself as the biggest Belgian real estate services provider. To achieve that goal, they need to more accurately and faster than their competitors estimate the value of properties, to pick out those properties that are most valuable to them and their clients. Bringing the insights for this.

### Customer

Our customer is an investor who is looking to buy houses to rent out. Therefore the yield, how the buying price relates to the rent price is decisive.

### Data Cleaning

The cleaning pipeline ([data/cleaned/clean_data.py](data/cleaned/clean_data.py)) takes the two raw scrapes and produces the analysis-ready files:

| File | Raw rows | Cleaned rows | Rows removed |
| ---- | -------- | ------------ | ------------ |
| `data/raw/forsale.csv` → `data/cleaned/cleaned_sale_properties.csv` | 14,951 | 13,718 | 1,233 (8.2%) |
| `data/raw/torent.csv` → `data/cleaned/cleaned_rent_properties.csv` | 5,272 | 4,968 | 304 (5.8%) |

The 73-column schema is **identical** between raw and cleaned files — no columns were added or dropped at this stage. Cleaning operates on rows and individual values:

1. **Drop exact duplicate rows** (`drop_duplicates`). Removes records that are byte-for-byte identical across all 73 columns. (In this scrape there were none — the raw files had no fully-identical rows — but the step guards against it.)

2. **Trim whitespace** on every text column (`str.strip()`). Removes leading/trailing spaces so that values like `"Brussels "` and `"Brussels"` are treated as the same.

3. **Empty strings → `NaN`.** Cells containing only whitespace (`^\s*$`) are converted to proper missing values so they are counted as missing rather than as a distinct empty category.

4. **Normalise sentinel "missing" tokens → `NaN`.** The string literals `"None"`, `"N/A"`, `"NA"`, `"?"` and `"null"` are replaced with real `NaN`, so different spellings of "no value" collapse into one consistent missing marker.

5. **Drop near-duplicate listings** (`remove_custom_duplicates`). The same physical property is often scraped under several IDs/URLs. Rows are de-duplicated on the composite key **`(latitude, longitude, price, livable_surface)`**, keeping the first occurrence (sorted by `posting_date` descending so the most recent listing is retained). This is the step that does the heavy lifting: it removes **1,233 sale** and **304 rent** rows that share the same location, price and surface — i.e. the same dwelling listed more than once.

Steps 2–4 are defensive normalisations: in this particular raw extract the CSV parser already read blanks as `NaN` and no literal sentinel tokens were present, so they changed no cells here, but they make the pipeline robust to messier inputs. The measurable row reduction comes entirely from de-duplication (step 5).

### Visualizations and insights

### Questions and answers

The analysis below is based on the two cleaned datasets in [data/cleaned/](data/cleaned/): `cleaned_sale_properties.csv` (for-sale listings) and `cleaned_rent_properties.csv` (for-rent listings). Both share the same 73-column schema scraped from ImmoVlan.

**How many observations and features/columns do you have?**

| Dataset | Observations (rows) | Features (columns) |
| ------- | ------------------- | ------------------ |
| Sale    | 13,718              | 73                 |
| Rent    | 4,968               | 73                 |

Both files use the same 73 columns. After dropping pure-identifier/metadata columns (see below), roughly 60 columns carry analytical signal.

**What is the proportion of missing values per column?**

Missingness is very high and uneven. A large share of columns are mostly empty.

*Sale dataset* — fully or almost-empty columns: `co_ownership_charges` (100%), `demarcated_flooding_area` (99%), `maintenance_cost` (96%), `ground_depth` (93%), `balcony` (92%), `garden_orientation` (91%), `terrain_width` (89%), `heat_pump` (89%), `flood_g_score`/`flood_p_score` (86%), `indoor_parking` (86%), `solar_panels` (85%), `air_conditioning` (82%), `garden_surface` (80%), `kitchen_surface` (79%), `swimming_pool` (78%), `cadastral_income` (66%), `land_surface` (63%). Columns that are essentially complete (<5% missing): `price`, `livable_surface`, `region`, `province`, `locality`, `postal_code`, `latitude`, `longitude`, `bedrooms` (2.5%), `price_per_sqm` (4.9%).

*Rent dataset* — even sparser on some fields: `vat`, `cadastral_income`, `flooding_area_type`, `co_ownership_charges`, `demarcated_flooding_area` (≈100% missing), `flood_*` scores (99.8%), `land_surface` (90%), `garden_surface` (89%), `build_year` (57%). Complete fields again include `price`, `livable_surface`, location columns and coordinates.

The takeaway: only a core set of ~10–15 fields (price, surface, bedrooms/bathrooms, location) is reliably populated; most amenity/flags fields are sparse.

**Which variables would you delete and why?**

- **No information value:** `co_ownership_charges`, `demarcated_flooding_area` (≈100% empty in both files), and in the rent file `vat`, `cadastral_income`, `flooding_area_type`, `flood_g_score`, `flood_p_score` (all ~100% empty) — they contribute nothing.
- **Identifiers / scraping metadata, not predictive:** `property_id`, `url`, `html_path`, `source`, `scrape_date`, `posting_date`, `street`, `house_number`. These identify a record but should not feed a model.
- **Too sparse to impute reliably (>85% missing):** `maintenance_cost`, `ground_depth`, `terrain_width`, `balcony`, `garden_orientation`, `heat_pump`, `indoor_parking`, `solar_panels` — imputing 85–96% of values would inject more noise than signal.
- **Redundant:** `price_per_sqm` is `price` ÷ `livable_surface`, so it leaks the target and is collinear; keep it only for descriptive ranking, not modelling. `category` largely duplicates `property_type`.

**What variables are most subject to outliers?**

Using the 1.5×IQR rule on the sale data, the most outlier-prone variables are the surface/land and price fields, several with physically implausible extremes:

| Variable                       | % outliers | Max value (suspicious) | Median  |
| ------------------------------ | ---------- | ---------------------- | ------- |
| `garden_surface`             | 11.5%      | 150,000 m²            | 200     |
| `land_surface`               | 11.0%      | 178,367 m²            | 487     |
| `bedrooms`                   | 10.2%      | 100                    | 3       |
| `cadastral_income`           | 9.2%       | 190,501                | 850     |
| `terrace_surface`            | 8.0%       | 2,968 m²              | 15      |
| `livable_surface`            | 6.9%       | 9,762 m²              | 116     |
| `price`                      | 6.8%       | 13,000,000             | 345,500 |
| `primary_energy_consumption` | 3.9%       | 20,260,607             | 219     |
| `price_per_sqm`              | 3.0%       | 1,620,000              | 2,978   |

`price`, `land_surface`, `garden_surface`, `livable_surface` and `price_per_sqm` are the most affected. Some maxima (a 20M-kWh EPC, a 9,762 m² flat, €1.6M/m²) are clearly data errors and should be capped/winsorized or removed before modelling.

**How many qualitative and quantitative variables are there? Appropriate visuals and correlation measures?**

By dtype the sale file has **46 numeric (quantitative)** and **27 non-numeric (qualitative)** columns. Note that some numeric columns are really *binary qualitative flags* (e.g. `garden`, `terrace`, `swimming_pool`, `furnished`, `new_construction`) and some object columns are *ordinal* (`epc` A→G, `building_state`).

- **Quantitative data → ** histograms / KDE (distribution of `price`, `livable_surface`), box-plots (outliers), scatter plots (price vs surface), and a correlation heatmap.
- **Qualitative data → ** bar charts / count plots (listings per `province`, `property_type`), grouped/stacked bars, and box-plots of a numeric value *split by* category (e.g. price by `region`). Maps are ideal here given `latitude`/`longitude`.
- **Correlation measures:**
  - quantitative ↔ quantitative: **Pearson** (linear) or **Spearman** (monotonic, robust to the outliers above);
  - qualitative(ordinal) ↔ quantitative: **Spearman** or, qualitative↔quantitative in general, the **correlation ratio (η²)** / ANOVA F-test;
  - qualitative ↔ qualitative: **Cramér's V** (based on chi-square).

**What is the correlation between the variables and the price? Why are some more correlated than others?**

Pearson correlation with `price`:

| Sale (top)       | r    | Rent (top)       | r    |
| ---------------- | ---- | ---------------- | ---- |
| toilets          | 0.64 | livable_surface  | 0.66 |
| bathrooms        | 0.58 | bathrooms        | 0.65 |
| maintenance_cost | 0.55 | toilets          | 0.63 |
| bedrooms         | 0.52 | maintenance_cost | 0.56 |
| livable_surface  | 0.51 | bedrooms         | 0.42 |
| swimming_pool    | 0.40 | terrace_surface  | 0.41 |
| land_surface     | 0.35 | land_surface     | 0.39 |
| cadastral_income | 0.33 | garden_surface   | 0.29 |

The strongest drivers are all **size/capacity proxies** — number of bathrooms/toilets/bedrooms and livable & land surface — because price is fundamentally driven by how much property you get. `cadastral_income` correlates because it is itself a tax-assessed proxy of property value. By contrast, single amenity flags (`elevator` 0.00, `furnished` 0.01, `running_water` 0.01) and energy fields are weakly correlated: they are either nearly constant, mostly missing, or matter only marginally versus raw size and location. Interestingly `postal_code` shows a negative correlation (−0.24 sale, −0.29 rent) — an artefact of Belgium's geographic numbering (lower codes ≈ Brussels/Walloon-Brabant pricier areas), which really reflects a **location** effect, not the number itself.

**How are the variables themselves correlated to each other? Can you find groups of variables that are correlated together?**

Yes — clear clusters emerge:

1. **"Property size / capacity" cluster:** `bedrooms`, `bathrooms`, `toilets`, `showers`, `livable_surface` are strongly inter-correlated (e.g. bathrooms↔toilets 0.79, bedrooms↔bathrooms 0.74, bedrooms↔toilets 0.67, livable_surface↔toilets 0.63). Bigger homes have more of everything.
2. **"Land / outdoor" cluster:** `land_surface` ↔ `garden_surface` (0.55), with `facades` and `terrace_surface` loosely attached — detached homes on larger plots.
3. **"Value/assessment" link:** `cadastral_income` correlates with `terrace_surface` (0.35) and the size cluster, since it scales with property value.

Because the size variables move together (multicollinearity), a model does not need all of them — one or two (e.g. `livable_surface` + `bathrooms`) capture most of the shared signal.

**How are the number of properties distributed according to their surface?**

`livable_surface` (sale) is **right-skewed**: median 116 m², mean 150 m² (pulled up by a long tail), 25th–75th percentile 84–177 m². Count by surface band:

| Surface (m²) | # properties |
| ------------- | ------------ |
| 0–50         | 673          |
| 50–75        | 1,368        |
| 75–100       | 2,697        |
| 100–125      | 2,218        |
| 125–150      | 1,519        |
| 150–200      | 2,005        |
| 200–300      | 1,515        |
| 300–500      | 759          |
| 500+          | 288          |

The bulk of the market sits between **75 and 200 m²**, peaking in the 75–100 m² band, with a thin tail of very large properties.

**Which five variables do you consider the most important and why?**

For predicting price and serving the investor's yield use-case:

1. **`price`** — the target / one half of the yield ratio.
2. **`livable_surface`** — the single most universal size driver (complete data, strong correlation, used to derive €/m²).
3. **`locality` / `postal_code` (location)** — location explains the biggest price spread across Belgium (see municipality ranking below); essential context.
4. **`bedrooms` / `bathrooms`** — best-populated capacity proxies, strongly correlated with price and easy to interpret.
5. **`price_per_sqm`** — normalises price by size, the key comparison metric for spotting under/over-priced listings and computing rental yield against the rent dataset.

These five are both **highly correlated with price and reliably populated**, unlike most amenity flags.

**What are the least/most expensive municipalities in Belgium/Wallonia/Flanders?**

Restricting to municipalities with enough listings to be reliable (≥15–20). *Caveat:* a few `price_per_sqm` averages (e.g. Hoeselt, Knokke-Heist) are inflated by data-entry outliers and were excluded/flagged; the figures below use the cleaner metrics.

*Belgium — most affordable (median price):* Dour (€115k), Charleroi (€119k), Quaregnon (€130k), Marchienne-au-Pont (€130k), Gilly (€139k) — all in industrial Wallonia.

*Belgium — most expensive (median price):* Grez-Doiceau (€1.69M), Kraainem (€1.25M), Sint-Genesius-Rode (€1.25M), Sterrebeek (€999k), Lasne (€970k) — the affluent green belt around Brussels.

*Wallonia (avg €/m²):* cheapest — Dour (~€850), Marchienne-au-Pont (~€890), Colfontaine (~€1,100); dearest — Chaumont-Gistoux (~€4,700), Angleur (~€4,460), Waterloo (~€4,110).

*Flanders (avg €/m²):* cheapest — Menen (~€1,660), Beerse (~€2,010), Wielsbeke (~€2,170); dearest — the coastal Knokke-Heist area and outer-Brussels Lennik command the top prices (Knokke-Duinbergen ≈ €7,200/m² once outliers are removed).

*Brussels (avg €/m², 19 communes):* cheapest — Koekelberg (~€2,620), Ganshoren (~€2,630), Sint-Jans-Molenbeek (~€2,710); dearest — Sint-Joost-ten-Node (~€6,430), Oudergem (~€5,860), Sint-Lambrechts-Woluwe (~€5,370).

Overall pattern: cheapest property concentrates in the old Walloon industrial belt (Borinage/Charleroi), while the priciest sits in Brussels' wealthy south-eastern periphery and the coastal Knokke area.
