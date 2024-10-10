import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('laptops.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')


# EDA Look at the final_price variable. Does it have a long tail?
eda = df[['ram', 'storage', 'screen', 'final_price']]
plt.figure(figsize=(8, 6))
plt.hist(eda['final_price'])
plt.title('Distribution of Final Price')
plt.xlabel('Final Price')
plt.ylabel('Frequency')
# plt.show()
print("Long tail\n")


# Q1. There's one column with missing values. What is it?
print("Q1. There's one column with missing values. What is it?")
missing_values = eda.isnull().sum()

print(f"Missing values: {missing_values[missing_values > 0].to_string(index=True)}\n")


# Q2. What's the median (50% percentile) for variable 'ram'?
print("Q2. What's the median (50% percentile) for variable 'ram'?")
median_ram = eda['ram'].median()
print(f"Median RAM: {median_ram} \n")

# Shuffle the dataset (the filtered one you created above), use seed 42.
n = len(eda)
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)

# Split your data in train/val/test sets, with 60%/20%/20% distribution.
print("Split your data in train/val/test sets, with 60%/20%/20% distribution.")
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

print(n_train, n_test, n_val)

df_train = eda.iloc[idx[:n_train]].reset_index(drop=True)
df_test = eda.iloc[idx[n_train+n_val:]].reset_index(drop=True)
df_val = eda.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)

print(len(df_train), len(df_test), len(df_val), "\n")

y_train = df_train.final_price.values
y_test = df_test.final_price.values
y_val = df_val.final_price.values

del df_train['final_price']
del df_val['final_price']
del df_test['final_price']


# Q3. Which option gives better RMSE?
def prepare_X(df, mean=0):
    df_num = df.fillna(mean)
    X = df_num.values
    return X


def rmse(y, y_pred):
    mse = ((y - y_pred) ** 2).mean()
    return np.sqrt(mse)


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]


missing_values = df_train[df_train.isnull().any(axis=1)]
print("Q3. Which option gives better RMSE?")
print(missing_values)
indexes_of_missing_values = missing_values.index.tolist()

# Fill missing values with 0
X_train = df_train.fillna(0).values
w_0, w = train_linear_regression(X_train, y_train)
X_val = df_val.fillna(0).values
y_pred = w_0 + X_val.dot(w)
result_fill_zero = rmse(y_val, y_pred)
round(result_fill_zero, 2)
print("RMSE when replacing with zeros:", result_fill_zero)

# Fill missing values with mean
mean = df_train.screen.mean()
X_train = prepare_X(df_train, mean=mean)
w_0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(df_val, mean=mean)
y_pred = w_0 + X_val.dot(w)
result_fill_mean = rmse(y_val, y_pred)
round(result_fill_mean, 2)
print("RMSE when replacing with mean:", result_fill_mean)
if result_fill_zero < result_fill_mean:
    print("Filling with zeros is better\n")
elif result_fill_zero > result_fill_mean:
    print("Filling with mean is better\n")
else:
    print("Both options are equal\n")


# Q4. Which r gives the best RMSE?
print("Q4. Which r gives the best RMSE?")


def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)   
    return w[0], w[1:]


max_rmse = -float('inf')
max_r = None


for r in [0, 0.01, 0.1, 1, 10, 100]:

    X_train = prepare_X(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)

    rmse_val = rmse(y_val, y_pred)
    rmse_val = round(rmse_val, 2)

    print("r:", r, "w0:",  w0, "RMSE:", round(rmse_val, 2))

    if r == 0:
        max_rmse = rmse_val

    if rmse_val < max_rmse:
        max_rmse = rmse_val
        max_r = r

print(f"\nThe r corresponding to the best RMSE ({max_rmse}) is: {max_r}\n")

# Q5. What's the value of std?
print("Q5. What's the value of std?")


seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rmse_vals = []

for seed_val in seeds:
    n = len(eda)
    idx = np.arange(n)
    np.random.seed(seed_val)
    np.random.shuffle(idx)

    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    df_train = eda.iloc[idx[:n_train]].reset_index(drop=True)
    df_test = eda.iloc[idx[n_train+n_val:]].reset_index(drop=True)
    df_val = eda.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)

    y_train = df_train.final_price.values
    y_test = df_test.final_price.values
    y_val = df_val.final_price.values

    del df_train["final_price"]
    del df_val["final_price"]
    del df_test["final_price"]

    X_train = prepare_X(df_train)
    w0, w = train_linear_regression(X_train, y_train)

    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    rmse_vals.append(rmse(y_val, y_pred))

print(round(pd.DataFrame({"seed: ": seeds, "rmse: ": rmse_vals}), 2), "\n")
print("Standard deviation: ", round(np.std(rmse_vals), 3), "\n")


# Q6. What's the RMSE on the test dataset?
print("Q6. What's the RMSE on the test dataset?")

n = len(eda)
idx = np.arange(n)
np.random.seed(9)
np.random.shuffle(idx)

df_train = eda.iloc[idx[:n_train]].reset_index(drop=True)
df_test = eda.iloc[idx[n_train+n_val:]].reset_index(drop=True)
df_val = eda.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)

y_train = df_train.final_price.values
y_test = df_test.final_price.values
y_val = df_val.final_price.values

del df_train["final_price"]
del df_val["final_price"]
del df_test["final_price"]

df_train_val = pd.concat([df_train, df_val]).reset_index(
    drop=True).fillna(0)
y_train_val = np.concatenate([y_train, y_val])

df_test_v1 = df_test.fillna(0)
X_full_train = df_train_val.values
X_test1 = df_test_v1.values

w0, w = train_linear_regression_reg(X_full_train, y_train_val, 0.001)
y_test_pred1 = w0 + X_test1.dot(w)
rmse_val = rmse(y_test_pred1, y_test)
print("RMSE on the test dataset: ", rmse_val)
