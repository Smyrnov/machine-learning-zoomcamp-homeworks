import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
df = pd.read_csv('jamb_exam_results.csv')

# Data Preparation
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.drop(columns=['student_id'])
df = df.fillna(0)

# Split data into train, validation, and test sets
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)

y_train = df_train.jamb_score.values
y_val = df_val.jamb_score.values
y_test = df_test.jamb_score.values

# Encode categorical features
dv = DictVectorizer(sparse=False)
train_dicts = df_train.drop(columns=['jamb_score']).to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val.drop(columns=['jamb_score']).to_dict(orient='records')
X_val = dv.transform(val_dicts)

test_dicts = df_test.drop(columns=['jamb_score']).to_dict(orient='records')
X_test = dv.transform(test_dicts)

# Question 1: Decision Tree Regressor with max_depth=1
dt = DecisionTreeRegressor(max_depth=1, random_state=1)
dt.fit(X_train, y_train)
print(f"Feature used for splitting with max_depth=1: {dv.feature_names_[dt.tree_.feature[0]]}")

# Question 2: Random Forest Regressor with n_estimators=10
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE of Random Forest with n_estimators=10: {rmse:.2f}")

# Question 3: Experimenting with n_estimators
rmse_scores = []
for n in range(10, 201, 10):
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores.append((n, rmse))
    print(f"n_estimators={n}, RMSE={rmse:.3f}")

best_n = min(rmse_scores, key=lambda x: x[1])[0]
print(f"Best n_estimators after which RMSE stops improving: {best_n}")

# Question 4: Tuning max_depth
depth_rmse_scores = []
for depth in [10, 15, 20, 25]:
    rmse_list = []
    for n in range(10, 201, 10):
        rf = RandomForestRegressor(n_estimators=n, max_depth=depth, random_state=1, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_list.append(rmse)
    mean_rmse = np.mean(rmse_list)
    depth_rmse_scores.append((depth, mean_rmse))
    print(f"max_depth={depth}, Mean RMSE={mean_rmse:.3f}")

best_depth = min(depth_rmse_scores, key=lambda x: x[1])[0]
print(f"Best max_depth based on mean RMSE: {best_depth}")

# Question 5: Feature importance with Random Forest
rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
importance = rf.feature_importances_
important_feature = dv.feature_names_[np.argmax(importance)]
print(f"Most important feature: {important_feature}")

# Question 6: XGBoost tuning eta parameter
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

xgb_params_03 = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1
}

xgb_params_01 = xgb_params_03.copy()
xgb_params_01['eta'] = 0.1

# Train with eta=0.3
model_03 = xgb.train(xgb_params_03, dtrain, num_boost_round=100, evals=[(dval, 'val')], verbose_eval=False)
y_pred_03 = model_03.predict(dval)
rmse_03 = np.sqrt(mean_squared_error(y_val, y_pred_03))
print(f"RMSE with eta=0.3: {rmse_03:.3f}")

# Train with eta=0.1
model_01 = xgb.train(xgb_params_01, dtrain, num_boost_round=100, evals=[(dval, 'val')], verbose_eval=False)
y_pred_01 = model_01.predict(dval)
rmse_01 = np.sqrt(mean_squared_error(y_val, y_pred_01))
print(f"RMSE with eta=0.1: {rmse_01:.3f}")

best_eta = 0.1 if rmse_01 < rmse_03 else 0.3
print(f"Best eta based on RMSE: {best_eta}")
