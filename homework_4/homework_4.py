import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('bank-full.csv', delimiter=';')

# Select the necessary columns
selected_columns = [
    'age', 'job', 'marital', 'education', 'balance', 'housing', 'contact', 'day',
    'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'
]

df = df[selected_columns]
df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Split the data into train, validation, and test sets (60%/20%/20%)
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1) # 0.25 * 0.8 = 0.2

# Question 1: ROC AUC feature importance
numerical_columns = ['balance', 'day', 'duration', 'previous']
auc_scores = {}
for col in numerical_columns:
    auc = roc_auc_score(df_train['y'], df_train[col])
    if auc < 0.5:
        auc = roc_auc_score(df_train['y'], -df_train[col])
    auc_scores[col] = auc
print("AUC scores for numerical features:", auc_scores)
highest_auc_feature = max(auc_scores, key=auc_scores.get)
print(f"Numerical feature with the highest AUC: {highest_auc_feature}")

# Question 2: Train Logistic Regression model
dv = DictVectorizer(sparse=False)
train_dicts = df_train.drop(columns=['y']).to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)
y_train = df_train['y'].values

val_dicts = df_val.drop(columns=['y']).to_dict(orient='records')
X_val = dv.transform(val_dicts)
y_val = df_val['y'].values

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_pred)
print(f"AUC on validation set: {val_auc:.3f}")

# Question 3: Precision and Recall curves
precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred)
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
# plt.show()

# Find intersection of precision and recall
for i, (precision, recall) in enumerate(zip(precisions, recalls)):
    if np.isclose(precision, recall, atol=0.01):
        print(f"Precision and recall curves intersect at threshold: {thresholds[i]:.3f}")
        break

# Question 4: F1 score for all thresholds
f1_scores = []
for threshold in np.arange(0.0, 1.01, 0.01):
    y_pred_thresh = (y_pred >= threshold).astype(int)
    f1 = f1_score(y_val, y_pred_thresh)
    f1_scores.append(f1)
best_f1_threshold = np.argmax(f1_scores) / 100
print(f"Threshold with the highest F1 score: {best_f1_threshold:.2f}")

# Question 5: 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=1)
auc_scores = []
for train_index, val_index in kf.split(df_train_full):
    df_train_fold = df_train_full.iloc[train_index]
    df_val_fold = df_train_full.iloc[val_index]
    
    train_dicts = df_train_fold.drop(columns=['y']).to_dict(orient='records')
    X_train_fold = dv.fit_transform(train_dicts)
    y_train_fold = df_train_fold['y'].values
    
    val_dicts = df_val_fold.drop(columns=['y']).to_dict(orient='records')
    X_val_fold = dv.transform(val_dicts)
    y_val_fold = df_val_fold['y'].values
    
    model.fit(X_train_fold, y_train_fold)
    y_pred_fold = model.predict_proba(X_val_fold)[:, 1]
    auc_scores.append(roc_auc_score(y_val_fold, y_pred_fold))

std_auc = np.std(auc_scores)
print(f"Standard deviation of AUC scores across folds: {std_auc:.4f}")

# Question 6: Hyperparameter tuning for C
C_values = [0.000001, 0.001, 1]
best_mean_score = -1
best_C = None
for C in C_values:
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    scores = []
    for train_index, val_index in kf.split(df_train_full):
        df_train_fold = df_train_full.iloc[train_index]
        df_val_fold = df_train_full.iloc[val_index]
        
        train_dicts = df_train_fold.drop(columns=['y']).to_dict(orient='records')
        X_train_fold = dv.fit_transform(train_dicts)
        y_train_fold = df_train_fold['y'].values
        
        val_dicts = df_val_fold.drop(columns=['y']).to_dict(orient='records')
        X_val_fold = dv.transform(val_dicts)
        y_val_fold = df_val_fold['y'].values
        
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict_proba(X_val_fold)[:, 1]
        scores.append(roc_auc_score(y_val_fold, y_pred_fold))
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"C={C}: Mean AUC={mean_score:.3f}, Std={std_score:.3f}")
    
    if mean_score > best_mean_score or (mean_score == best_mean_score and std_score < std_auc):
        best_mean_score = mean_score
        best_C = C

print(f"Best C: {best_C} with mean AUC: {best_mean_score:.3f}")
