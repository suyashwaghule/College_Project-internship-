import pandas as pd
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_csv("data/processed/clean_crime_data.csv")

print(df.head())
print(df.shape)


# Aggregate district data to state-year level
state_year_df = (
    df.groupby(["state", "year"])["total_crimes"]
      .sum()
      .reset_index()
)

print(state_year_df.head())


# Choose one state to visualize
sample_state = "MAHARASHTRA"   # change if needed

state_data = state_year_df[state_year_df["state"] == sample_state]

plt.figure()
plt.plot(state_data["year"], state_data["total_crimes"])
plt.xlabel("Year")
plt.ylabel("Total IPC Crimes")
plt.title(f"Crime Trend in {sample_state}")
plt.show()
