import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.read_csv('laptops.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

# EDA Look at the final_price variable. Does it have a long tail?
eda = df[['ram', 'storage', 'screen', 'final_price']]
plt.figure(figsize=(8, 6))
plt.hist(eda['final_price'])
plt.title('Distribution of Final Price')
plt.xlabel('Final Price')
plt.ylabel('Frequency')
plt.show()


# Q1. There's one column with missing values. What is it?
missing_values = eda.isnull().sum()

print(f"Missing values: {missing_values[missing_values > 0]}")


# Q2. What's the median (50% percentile) for variable 'ram'?
median_ram = eda['ram'].median()
print("Median RAM: ", median_ram)

# Shuffle the dataset (the filtered one you created above), use seed 42.
n = len(eda)
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)

# Split your data in train/val/test sets, with 60%/20%/20% distribution.

n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

print(n_train, n_test, n_val)

df_train = eda.iloc[idx[:n_train]]
df_test = eda.iloc[idx[n_train+n_val:]]
df_val = eda.iloc[idx[n_train:n_train+n_val]]

print(len(df_train), len(df_test), len(df_val))

df_train.reset_index()
df_test.reset_index()
df_val.reset_index()

y_train = np.log1p(df_train.final_price.values)
y_test = np.log1p(df_test.final_price.values)
y_val = np.log1p(df_val.final_price.values)

del df_train['final_price']
del y_test['final_price']
del y_val['final_price']

# Q3. Which option gives better RMSE?



# Q4. Which r gives the best RMSE?



# Q5. What's the value of std?



# Q6. What's the RMSE on the test dataset?