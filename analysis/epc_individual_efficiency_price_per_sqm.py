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

# Explicit sorting order for all individual EPC ratings
epc_order = ["A++", "A+", "A", "B", "C", "D", "E", "F", "G"]
df = df[df["epc"].isin(epc_order)]
df = df[df["livable_surface"] > 0]

# Remove extreme outliers
df = df[
    df["price_per_sqm"]
    <= df["price_per_sqm"].quantile(0.98)
]

# ==========================================================
# 3. Aggregate Data (By Individual EPC Letters)
# ==========================================================
agg_df = (
    df.groupby("epc", observed=False)
    .agg(
        property_count=("property_id", "nunique"),
        median_price_per_sqm=("price_per_sqm", "median"),
    )
    .reindex(epc_order)
    .reset_index()
)

print("\nAggregated Data:")
print(agg_df)

# ==========================================================
# 4. Create Figure
# ==========================================================
fig, ax1 = plt.subplots(figsize=(12, 6), dpi=120)

fig.suptitle(
    "EPC Efficiency Impact on Sale Property Supply and Price per m²",
    fontsize=16,
    fontweight="bold",
)

# ==========================================================
# 5. Individual Property Volume Bars
# ==========================================================
# Official European environmental standard scale color palette (Green to Red)
epc_colors = [
    "#006633",  # A++ (Dark Green)
    "#009933",  # A+  (Medium Green)
    "#33CC33",  # A   (Green)
    "#99FF33",  # B   (Light Green)
    "#FFFF33",  # C   (Yellow)
    "#FFCC00",  # D   (Amber / Yellow-Orange)
    "#FF9900",  # E   (Orange)
    "#FF3300",  # F   (Deep Orange)
    "#CC0000",  # G   (Red)
]

bars = ax1.bar(
    agg_df["epc"],
    agg_df["property_count"],
    color=epc_colors,
    edgecolor="gray",
    linewidth=1.0,
    width=0.60,
    label="Property Volume",
)

# Headroom adjustments to prevent label clipping on tallest bars
max_count = agg_df["property_count"].max()
ax1.set_ylim(0, max_count * 1.15)

# Property Labels on top of bars
for bar in bars:
    height = bar.get_height()
    if pd.notna(height) and height > 0:
        ax1.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),           
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
            color="#7F8C8D",
        )

# Left Axis Customization
ax1.set_ylabel(
    "Number of Properties Available",
    fontsize=12,
    fontweight="bold",
)

ax1.set_xlabel(
    "Individual EPC Rating",
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
# 6. Median Price per m² Trend Line (Restored)
# ==========================================================
ax2 = ax1.twinx()

ax2.plot(
    agg_df["epc"],
    agg_df["median_price_per_sqm"],
    marker="o",
    markersize=8,
    linewidth=2.5,
    color="blue",
    label="Median Price per m²",
)

# Point Labels (Positioned dynamically above markers)
for x, y in zip(
    agg_df["epc"],
    agg_df["median_price_per_sqm"],
):
    if pd.notna(y):
        ax2.annotate(
            f"€{y/1000:.1f}K",
            xy=(x, y),
            xytext=(0, 10),          
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="blue",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
                pad=0.2,
            ),
        )

# Right Axis Customization
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

# --- FIXED INTERVAL ADJUSTMENTS (1.8K, 2.2K, 2.6K...) ---
max_price = agg_df["median_price_per_sqm"].max()
ymax = 3800 if max_price <= 3400 else (int(max_price) // 400) * 400 + 400

ax2.set_ylim(1800, ymax)

# Step interval set to exactly 400 units (0.4K)
ax2.yaxis.set_major_locator(
    mtick.MultipleLocator(400)
)

# K format formatter
ax2.yaxis.set_major_formatter(
    mtick.FuncFormatter(
        lambda x, pos: f"€{x/1000:.1f}K"
    )
)

# ==========================================================
# 7. Legend
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
# 8. Layout & Export
# ==========================================================
plt.tight_layout(rect=[0, 0, 1, 0.94])

image_dir = os.path.join(base_dir, "..", "images")
os.makedirs(image_dir, exist_ok=True)

output_file = os.path.join(image_dir, "epc_individual_efficiency_price_per_sqm.png")
fig.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")

print(f"\nChart saved successfully:\n{output_file}")
plt.show()
