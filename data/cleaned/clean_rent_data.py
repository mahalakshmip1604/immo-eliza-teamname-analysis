import pandas as pd
import numpy as np 

df = pd.read_csv("data/raw/torent.csv")

df["certain_parking_space"] = (
    (df["indoor_parking"] > 0) | (df["outdoor_parking"] > 0)
).astype("boolean")

# Keep NaN if both unknown
df.loc[
    df["indoor_parking"].isna() & df["outdoor_parking"].isna(),
    "certain_parking_space"
] = pd.NA


df["has_parking"] = (
    df["indoor_parking"].fillna(0) + df["outdoor_parking"].fillna(0)
) > 0


def remove_duplicates(df):
    return df.drop_duplicates()

def remove_custom_duplicates(df):
    if "positing_date" in df.columns:
        df = df.sort_values("posting_date", ascending = False)

    df = df.drop_duplicates(
        subset = ["latitude", "longitude", "price", "livable_surface"],
        keep = "first"
    )
    return df

def trim_whitespace(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df

#Replace empty strings with Nan
def empty_to_nan(df):
    df.replace(r"^\s*$", np.nan, regex=True, inplace = True)
    return df

def fix_wrong_values(df):
    df.replace(["None", "N/A", "NA", "?", "null"], np.nan, inplace = True)
    return df

columns_to_drop =["url","source", "scrape_date", "html_path", "vat", "house_number", "bedroom_surfaces", "living_room_surface", "kitchen_surface", "showers",
                  "currently_leased","heat_pump","demarcated_flooding_area", "ground_depth", "terrain_width", "co_ownership_charges"
                  ]



df = remove_duplicates(df)
df = trim_whitespace(df)
df = empty_to_nan(df)
df = fix_wrong_values(df)
df = remove_custom_duplicates(df)

df_new = df.drop(columns= columns_to_drop, errors = "ignore")




print("Shape after cleaning:", df_new.shape)
print("\nMissing values:\n", df_new.isna().sum().head())


df_new.to_csv("data/cleaned/cleaned_rent_properties.csv", index = False)

print(" Cleaned dataset saved!")

