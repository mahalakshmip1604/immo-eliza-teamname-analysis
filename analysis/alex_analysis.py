import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import scipy.stats as stats
import warnings
import plotly.express as px
from scipy.stats import pearsonr
import io
from scipy.stats import pearsonr, ttest_ind, f_oneway
from sklearn.linear_model import LinearRegression
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
os.makedirs("results", exist_ok=True)


def preprocess_data(df):
    df["has_parking"] = df["has_parking"].map({True: 1, False: 0})
    df["garden"] = df["garden"].map({True: 1, False: 0})
    df["terrace"] = df["terrace"].map({True: 1, False: 0})

    # EPC cleaning
    df["epc"] = df["epc"].replace({
        "A++": "A"
    })

    epc_mapping = {
        "A": 7,
        "B": 6,
        "C": 5,
        "D": 4,
        "E": 3,
        "F": 2,
        "G": 1
    }

    df["epc_numeric"] = df["epc"].map(epc_mapping)

    return df

def compute_correlations(df, target="price"):

    results = []

    for col in df.columns:
        if col == target:
            continue

        try:
            temp = df[[col, target]].dropna()

            if len(temp) < 10:
                continue

            x = pd.to_numeric(temp[col], errors="coerce")
            y = temp[target]

            valid = x.notna() & y.notna()
            x = x[valid]
            y = y[valid]

            if x.nunique() < 2:
                continue

            r, p = pearsonr(x, y)

            results.append({
                "feature": col,
                "r": r,
                "p": p
            })

        except:
            continue

    return pd.DataFrame(results)


def filter_features(results_df, r_threshold=0.10, p_threshold=0.05):

    filtered = results_df[
        (results_df["r"].abs() >= r_threshold) &
        (results_df["p"] < p_threshold)
    ]

    return filtered["feature"].tolist()


def build_filtered_df(df, features):

    cols = features + ["price"]
    df_filtered = df[cols].copy()

    return df_filtered


def plot_filtered_heatmaps(df_house, df_apartment):

    # correlations
    corr1 = df_house.corr(numeric_only=True)
    corr1 = corr1.dropna(axis=0, how="all").dropna(axis=1, how="all")

    corr2 = df_apartment.corr(numeric_only=True)
    corr2 = corr2.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Houses (filtered)", "Apartments (filtered)"),
        horizontal_spacing=0.15              
    )

    # heatmap 1
    fig.add_trace(
        go.Heatmap(
            z=corr1.values,
            x=corr1.columns,
            y=corr1.index,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1
        ),
        row=1, col=1
    )

    # heatmap 2
    fig.add_trace(
        go.Heatmap(
            z=corr2.values,
            x=corr2.columns,
            y=corr2.index,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=600,
        width=1200,
        title="Correlation Heatmaps (Only Significant Features)"
    )

    fig.show()

    
    plt.savefig("images/plot_filtered_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close()

def regression_pipeline(df, target = "price", r_threshold = 0.15, p_threshold = 0.05, output_file = None):
    results = []
    output = []
    
    for col in df.columns:
        if col == target:
            continue

        try:
            temp = df[[col, target]].dropna()

            if len(temp) < 10:
                continue 

            x = pd.to_numeric(temp[col], errors= "coerce")
            y = temp[target]

            valid = x.notna() & y.notna()
            x = x[valid]
            y = y[valid]

            if len(x) < 2:
                continue
            corr, p_value = pearsonr(x, y)

            results.append({
                "feature":col,
                "correlation":corr,
                "p_value": p_value
            })
    
        except:
            continue
    results_df = pd.DataFrame(results)

    
    top_corr = results_df.sort_values(by="correlation", key=abs, ascending=False).head(10)

    text = "\nTop correlations:\n" + top_corr.to_string()
    print(text)
    output.append(text)

    selected = results_df[
        (results_df["correlation"].abs() >=r_threshold) &
        (results_df["p_value"] < p_threshold)
    ]
    features = selected["feature"].tolist()
    
    bad_features = ["postal_code"]
    features = [f for f in features if f not in bad_features]

    text = "\nSelected features: " + str(features)
    print(text)
    output.append(text)

    
    if len(features) == 0:
        print("No features selected — try lowering threshold")
        return None
    
    df = df[df[target] > 0]
    x = df[features]
    y = np.log(df[target])

    df_model = x.join(y.rename("price_log"))
    df_model = df_model.dropna(subset=["price_log"])

    x = df_model[features].fillna(0)
    y = df_model["price_log"]

    
    text = f"\nFinal dataset shape: {x.shape}"
    print(text)
    output.append(text)


    model = LinearRegression()
    model.fit(x,y)
    r2 = model.score(x, y)

    
    text = f"\nR² score: {r2}"
    print(text)
    output.append(text)

    
    n = x.shape[0]
    p = x.shape[1]
    
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    
    text = f"Adjusted R²: {adj_r2}"
    print(text)
    output.append(text)



  
    output.append("\nCoefficients:")
    print("\nCoefficients:")

    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(output))

    return model, x, y


def price_map():
        """Geographic scatter of price per m2 across Belgium."""
        geo = df.dropna(subset=["latitude", "longitude", "price_per_sqm"])
        lo, hi = geo["price_per_sqm"].quantile([0.01, 0.99])   
        geo = geo[(geo["price_per_sqm"] >= lo) & (geo["price_per_sqm"] <= hi)]

        
        plt.figure(figsize=(11, 11))
        sc = plt.scatter(geo["longitude"], geo["latitude"], c=geo["price_per_sqm"],
                         cmap="RdYlGn", s=6, alpha=0.6)
        plt.colorbar(sc, label="Price per m2 (EUR)")
        plt.title("Price per m2 map (Belgium)", fontsize=15, fontweight="bold")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
        plt.savefig("images/price_map.png", dpi=300, bbox_inches="tight")
        plt.close()



def box_plot_cities_with_values_unit(df, title):
    plt.figure(figsize=(12,6))
    top_cities = df["nearest_city"].value_counts().head(5).index
    
    epc_colors = [
    "#004E00", "#006400", "#228B22",
    "#9ACD32", "#FFF700", "#FFA500",
    "#FF4500", "#D2691E", "#8B4513"
]

    palette = {
         "apartment": epc_colors[2],
         "house": epc_colors[6]}
    df_top = df[df["nearest_city"].isin(top_cities)]
    ax = sns.boxplot(
        data =df_top,
        x="nearest_city",
        y = "price",
        hue= "category",
        palette = palette,
        showfliers=False 
    )
    cities_order = [t.get_text() for t in ax.get_xticklabels()]
    grouped = df_top.groupby(["nearest_city", "category"])["price"]
    
    for i, ((city, cat), values) in enumerate(grouped):
        q1 = values.quantile(0.25)
        median = values.quantile(0.5)
        q3 = values.quantile(0.75)

        x_pos = cities_order.index(city)

        if cat == "apartment":
            offset = -0.2
        else:
            offset = 0.2

        
        #ax.text(x_pos + offset, q1, f"{q1/1e6:.2f}M", ha="center", fontsize=7)
        ax.text(x_pos + offset, median, f"{median/1000:.0f}K EUR", ha="center", fontsize=8, color="white")
        #ax.text(x_pos + offset, q3, f"{q3/1e6:.2f}M", ha="center", fontsize=7)

                      
        ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{int(x)/1000000}M EUR')
    )
        """"
        #ax2 = ax.twinx()
        #counts = df_top["nearest_city"].value_counts().reindex(cities_order)
        #ax2.plot(
            counts.index,
            counts.values,
            color = "black",
            marker = "o",
            linestyle = "--"
        #)
        #ax2.set_ylabel("Number of properties")
        """


    
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylabel("Price (€/m²)")
    
    plt.savefig("images/box_plot_cities_with_values_unit.png", dpi=300, bbox_inches="tight")
    plt.close()


def box_plot_cities_with_values_sqm(df, title):
    plt.figure(figsize=(12,6))
    top_cities = df["nearest_city"].value_counts().head(5).index
    
    

    palette = {
         "apartment": epc_colors[2],
         "house": epc_colors[6]}
    df_top = df[df["nearest_city"].isin(top_cities)]
    ax = sns.boxplot(
        data =df_top,
        x="nearest_city",
        y = "price_per_sqm",
        hue= "category",
        palette = palette,
        showfliers=False 
    )
    cities_order = [t.get_text() for t in ax.get_xticklabels()]
    grouped = df_top.groupby(["nearest_city", "category"])["price_per_sqm"]
    
    for i, ((city, cat), values) in enumerate(grouped):
        q1 = values.quantile(0.25)
        median = values.quantile(0.5)
        q3 = values.quantile(0.75)

        x_pos = cities_order.index(city)

        if cat == "apartment":
            offset = -0.2
        else:
            offset = 0.2

        
        #ax.text(x_pos + offset, q1, f"{q1/1e6:.2f}M", ha="center", fontsize=7)
        ax.text(x_pos + offset, median, f"{median:.0f}EUR/m", ha="center", fontsize=8, color="white")
        #ax.text(x_pos + offset, q3, f"{q3/1e6:.2f}M", ha="center", fontsize=7)

                      
        ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{int(x)} EUR')
    )
        """"
        #ax2 = ax.twinx()
        #counts = df_top["nearest_city"].value_counts().reindex(cities_order)
        #ax2.plot(
            counts.index,
            counts.values,
            color = "black",
            marker = "o",
            linestyle = "--"
        #)
        #ax2.set_ylabel("Number of properties")
        """


    
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylabel("Price (€/m²)")
    plt.savefig("images/box_plot_cities_with_values_sqm.png", dpi=300, bbox_inches="tight")
    plt.close()


def stacked_bar_top_cities(df, title):
    
    top_cities = df["nearest_city"].value_counts().head(5).index

    counts = df.groupby(["nearest_city", "category"]).size().unstack()
    counts = counts.loc[top_cities]
    colors = [epc_colors[2], epc_colors[6]] 

    
    counts.plot(kind="bar", stacked=True, figsize=(10,6))
    ax = counts.plot(kind="bar", stacked=True, figsize=(10,6), color = colors)
    for p in ax.patches:

        height = p.get_height()
        bottom = p.get_y()

        if height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2,
                bottom + height / 2,     
                int(height),
                ha="center",
                va="center",
                fontsize=9,
                color="white"            
        )


    plt.title(title)
    plt.ylabel("Number of properties")
    plt.xticks(rotation=45)
    plt.savefig("images/stacked_bar.png", dpi=300)
    plt.close()



def run_analysis():

    print("Running analysis")

    #  LOAD 
    df = pd.read_csv("data/cleaned/cleaned_sale_properties.csv")

    #  PREPROCESS 
    df = preprocess_data(df)
    df["price_per_sqm"] = df["price"] / df["livable_surface"]

    #  SPLIT 
    df_apartment = df[df["category"].str.lower() == "apartment"]
    df_house = df[df["category"].str.lower() == "house"]

    #  REGRESSION (saved as txt) =====
    regression_pipeline(
        df_house,
        output_file="test_results_house.txt"
    )

    regression_pipeline(
        df_apartment,
        output_file="test_results_apartment.txt"
    )

    #  FILTERED DATA (for heatmap) 
    results_house = compute_correlations(df_house)
    results_apartment = compute_correlations(df_apartment)

    features_house = filter_features(results_house)
    features_apartment = filter_features(results_apartment)

    df_house_filtered = build_filtered_df(df_house, features_house)
    df_apartment_filtered = build_filtered_df(df_apartment, features_apartment)

    #  HEATMAP  
    plot_filtered_heatmaps(df_house_filtered, df_apartment_filtered)

    # STACKED BAR  
    stacked_bar_top_cities(df, "Distribution of properties by city")

    #  OTHER PLOTS 
    price_map(df)

    box_plot_cities_with_values_unit(df, "Price by city")
    box_plot_cities_with_values_sqm(df, "Price per sqm by city")

    print(" Done")

    




