import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
print(f"Missing values: {len(missing_values[missing_values > 0])}")


# Q5. Maximum final price

dell_laptops = df[df['Brand'].str.contains('Dell', case=False, na=False)]
print(f"Maximum final price: {dell_laptops['Final Price'].max()}")


# Q6. Median value of Screen

median_screen = df['Screen'].median()
most_frequent_val = df['Screen'].mode()[0]
df['Screen'].fillna(most_frequent_val, inplace=True)
new_median_screen = df['Screen'].median()
if median_screen != new_median_screen:
    print("Has median changed: Yes")
else:
    print("Has median changed: No")


# Q7. Sum of weights

import numpy as np


innjoo_laptops = df[df['Brand'].str.contains('Innjoo', case=False, na=False)]

subset_innjoo_values = innjoo_laptops[['RAM', 'Storage', 'Screen']]

subset_innjoo_values = subset_innjoo_values.apply(pd.to_numeric, errors='coerce')

X = subset_innjoo_values.to_numpy()

XTX = np.dot(X.T, X)

XTX_inv = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100])

w = np.dot(np.dot(XTX_inv, X.T), y)

sum_of_w = np.sum(w)
print(f"The sum of all the elements of the result: {"{:.2f}".format(np.sum(w))}")
