import pandas as pd

# Q1. Pandas version
print(f"Pandas version: {pd.__version__}")

# Q2. Records count
df = pd.read_csv('laptops.csv')
row_count = len(df)
print(f"Number of records: {row_count}")

# Q3. Laptop brands

df.columns
unique_brands = df['Brand'].nunique()
print(f"Laptop brands: {unique_brands}")

# Q4. Missing values

missing_values = df.isnull().sum()

