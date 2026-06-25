import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import pandas as pd

# ==========================================================
# 1. Pipeline Data Ingestion
# ==========================================================
base_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(
    base_dir,
    "..",
    "data",
    "cleaned",
    "cleaned_sale_properties.csv",
)

df = pd.read_csv(os.path.normpath(file_path))

# ==========================================================
# 2. Data Cleaning
# ==========================================================
df["epc"] = df["epc"].astype(str).str.strip().str.upper()

df = df.dropna(
    subset=[
        "property_id",
        "price_per_sqm",
        "livable_surface",
        "epc",
    ]
)

df = df[df["epc"].isin(["A++", "A+", "A", "B", "C", "D", "E", "F", "G"])]
df = df[df["livable_surface"] > 0]

# Remove extreme outliers
df = df[
    df["price_per_sqm"]
    <= df["price_per_sqm"].quantile(0.98)
]

# ==========================================================
# 3. EPC Tier Classification
# ==========================================================
def categorize_epc_tier(label):
    if label in ["A++", "A+", "A", "B"]:
        return "High Efficiency\n(A++, A+, A, B)"
    elif label in ["C", "D"]:
        return "Medium Efficiency\n(C, D)"
    else:
        return "Low Efficiency\n(E, F, G)"


df["epc_tier"] = df["epc"].apply(categorize_epc_tier)

tier_order = [
    "High Efficiency\n(A++, A+, A, B)",
    "Medium Efficiency\n(C, D)",
    "Low Efficiency\n(E, F, G)",
]

# ==========================================================
# 4. Aggregate Data
# ==========================================================
agg_df = (
    df.groupby("epc_tier", observed=False)
    .agg(
        property_count=("property_id", "nunique"),
        median_price_per_sqm=("price_per_sqm", "median"),
    )
    .reindex(tier_order)
    .reset_index()
)

print("\nAggregated Data:")
print(agg_df)

# ==========================================================
# 5. Create Figure
# ==========================================================
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=120)

fig.suptitle(
    "EPC Efficiency Impact on Sale Property Supply and Price per m²",
    fontsize=16,
    fontweight="bold",
)

# ==========================================================
# 6. Property Volume Bars
# ==========================================================
tier_colors = [
    "#63C26B",  # Green
    "#D8DD83",  # Yellow
    "#E08C74",  # Orange
]

bars = ax1.bar(
    agg_df["epc_tier"],
    agg_df["property_count"],
    color=tier_colors,
    edgecolor="gray",
    linewidth=1.2,
    width=0.50,
    label="Property Volume",
)

# --- ALIGNMENT & CLIPPING FIX FOR LEFT Y-AXIS ---
max_count = agg_df["property_count"].max()
ax1.set_ylim(0, max_count * 1.15)  # Adds 15% extra upper margin headspace

# Property Labels on top of bars
for bar in bars:
    height = bar.get_height()
    if pd.notna(height):
        ax1.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),           # Position safely right above the bar line
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
            color="#7F8C8D",         # Clean light gray tint
        )

# Left Axis
ax1.set_ylabel(
    "Number of Properties Available",
    fontsize=12,
    fontweight="bold",
)

ax1.set_xlabel(
    "EPC Efficiency Tier",
    fontsize=12,
    fontweight="bold",
)

ax1.tick_params(
    axis="both",
    labelsize=10,
)

ax1.grid(
    axis="y",
    linestyle=":",
    alpha=0.5,
)

# ==========================================================
# 7. Median Price per m² Line
# ==========================================================
ax2 = ax1.twinx()

ax2.plot(
    agg_df["epc_tier"],
    agg_df["median_price_per_sqm"],
    marker="o",
    markersize=9,
    linewidth=3,
    color="blue",
    label="Median Price per m²",
)

# Point Labels
for x, y in zip(
    agg_df["epc_tier"],
    agg_df["median_price_per_sqm"],
):
    if pd.notna(y):
        ax2.annotate(
            f"€{y/1000:.1f}K",
            xy=(x, y),
            xytext=(0, 10),          
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="blue",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
                pad=0.3,
            ),
        )

# Right Axis
ax2.set_ylabel(
    "Median Price per m² (€)",
    fontsize=12,
    fontweight="bold",
    color="blue",
)

ax2.tick_params(
    axis="y",
    labelsize=10,
    colors="blue",
)

# Fixed Step Scale (1.8K, 2.1K, 2.4K...)
max_price = agg_df["median_price_per_sqm"].max()
ymax = 3600 if max_price <= 3400 else (int(max_price) // 300) * 300 + 300

ax2.set_ylim(1800, ymax)

# Step interval set to 300 units (0.3K)
ax2.yaxis.set_major_locator(
    mtick.MultipleLocator(300)
)

# K format formatter
ax2.yaxis.set_major_formatter(
    mtick.FuncFormatter(
        lambda x, pos: f"€{x/1000:.1f}K"
    )
)

# ==========================================================
# 8. Legend (TOP RIGHT)
# ==========================================================
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax2.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    fontsize=10,
    framealpha=0.95,
)

# ==========================================================
# 9. Layout
# ==========================================================
plt.tight_layout(rect=[0, 0, 1, 0.94])

# ==========================================================
#  Save Figure
# ==========================================================
image_dir = os.path.join(
    base_dir,
    "..",
    "images",
)

os.makedirs(image_dir, exist_ok=True)

output_file = os.path.join(
    image_dir,
    "epc_combined_efficiency_price_per_sqm.png",
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

print(f"\nChart saved successfully:\n{output_file}")


