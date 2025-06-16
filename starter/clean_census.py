import pandas as pd

# Load the original CSV
df = pd.read_csv("data/census.csv")

# Remove spaces from column names
df.columns = df.columns.str.replace(' ', '')

# Remove spaces from string data in all columns
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Save the cleaned DataFrame to a new CSV
df.to_csv("data/census_clean.csv", index=False)