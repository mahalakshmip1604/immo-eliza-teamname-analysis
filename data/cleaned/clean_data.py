import pandas as pd
import numpy as np 

df = pd.read_csv("data/raw/forsale.csv")

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



df = remove_duplicates(df)
df = trim_whitespace(df)
df = empty_to_nan(df)
df = fix_wrong_values(df)
df = remove_custom_duplicates(df)


print("Shape after cleaning:", df.shape)
print("\nMissing values:\n", df.isna().sum().head())


df.to_csv("cleaned_sale_properties.csv", index=False)

print(" Cleaned dataset saved!")



#duplicates = df.duplicated(
    #subset=["latitude", "longitude", "price", "livable_surface"])

#print(duplicates.sum())

