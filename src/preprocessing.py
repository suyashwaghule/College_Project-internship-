import pandas as pd

# Load raw data
df = pd.read_csv("data/raw/dstrIPC_2013.csv")

# Rename required columns
df = df.rename(columns={
    "STATE/UT": "state",
    "DISTRICT": "district",
    "YEAR": "year",
    "TOTAL IPC CRIMES": "total_crimes"
})

# Keep only relevant columns
df = df[["state", "district", "year", "total_crimes"]]

# Drop missing values (safety)
df = df.dropna()

# Convert data types
df["year"] = df["year"].astype(int)
df["total_crimes"] = df["total_crimes"].astype(int)

# Save processed data
df.to_csv("data/processed/clean_crime_data.csv", index=False)

print("✅ Clean data saved successfully!")
print(df.head())
