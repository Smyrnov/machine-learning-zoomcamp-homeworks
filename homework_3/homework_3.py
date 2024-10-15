import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None


df = pd.read_csv('bank-full.csv', delimiter=';')
# For the rest of the homework, you'll need to use only these columns

# Data preparation
df_subset = df[[
    "age", "job", "marital", "education", "balance", "housing", "contact",
    "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"
]]
missing_values = df_subset.isnull().sum()
print(missing_values, "\n")


# Question 1
# What is the most frequent observation (mode) for the column education?
print("Question 1 \n")
print(f"The most frequent observation (mode) for the column education: {df_subset['education'].mode()[0]}\n")


# Question 2
# Create the correlation matrix for the numerical features of your dataset. In a correlation matrix, you compute the correlation coefficient between every pair of features.
# What are the two features that have the biggest correlation?
print("Question 2 \n")

numerical_features = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
correlation_matrix = df_subset[numerical_features].corr()

print("correlation matrix \n", correlation_matrix, "\n")


# Now we want to encode the y variable. Let's replace the values yes/no with 1/0.

df_subset['y'] = df_subset['y'].apply(lambda x: 1 if x == 'yes' else 0)

X = df_subset.drop(columns=['y'])
y = df_subset['y']
df_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  
df_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  


# Question 3 Calculate the mutual information score between y and other categorical variables in the dataset. Use the training set only.
# Round the scores to 2 decimals using round(score, 2).
# Which of these variables has the biggest mutual information score?
print("Question 3 \n")


categorical_features = ['job', 'marital', 'education', 'housing', 'contact', 'month', 'poutcome']


def calculate_mutual_info(df, target):
    mi_scores = {}
    for feature in categorical_features:
        mi_score = mutual_info_score(df[feature], target)
        mi_scores[feature] = round(mi_score, 2)
    return mi_scores


mi_scores = calculate_mutual_info(df_train, y_train)

print("Variable with the biggest mutual information score: ", max(mi_scores, key=mi_scores.get), "\n")


# Question 4 Calculate the accuracy on the validation dataset and round it to 2 decimal digits.
print("Question 4 \n")

dv = DictVectorizer(sparse=False)
train_dicts = df_train[numerical_features + categorical_features].to_dict(orient='records')
dv.fit(train_dicts)
X_train = dv.transform(train_dicts)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)
val_dict = df_val[numerical_features + categorical_features].to_dict(orient='records')
X_val = dv.transform(val_dict)
y_pred = model.predict_proba(X_val)[:, 1]
y_res = (y_pred >= 0.5)
print("Accuracy: ", round(accuracy_score(y_val, y_res), 2), "\n")
original_accuracy = accuracy_score(y_val, y_res)


# Question 5 Which of following feature has the smallest difference?
print("Question 5 \n")

def calculate_accuracy_excluding_feature(exclude_feature):
    features_to_use = [f for f in numerical_features + categorical_features if f != exclude_feature]
    dv = DictVectorizer(sparse=False)
    train_dicts = df_train[features_to_use].to_dict(orient='records')
    dv.fit(train_dicts)
    X_train = dv.transform(train_dicts)
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    val_dict = df_val[features_to_use].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    y_pred = model.predict_proba(X_val)[:, 1]
    y_res = (y_pred >= 0.5)
    accuracy = accuracy_score(y_val, y_res)

    return accuracy


task5_res_dict = {}

for feature in categorical_features:
    task5_res_dict[feature] = abs(original_accuracy - calculate_accuracy_excluding_feature(feature))
print("All results: ", task5_res_dict, "\n")
print("Original Accuracy: ", original_accuracy)
task5_res_dict.pop('housing')
print("All results: ", task5_res_dict, "\n")
min_key = min(task5_res_dict, key=task5_res_dict.get)
print("Smallest difference:", min_key, task5_res_dict[min_key])


# Question 6. Which of these C leads to the best accuracy on the validation set?
print("Question 6 \n")

C_values = [0.01, 0.1, 1, 10, 100]

# Dictionary to store the accuracy for each C value
accuracy_results = {}

# Iterate over each C value, train the model, and calculate accuracy
for C_value in C_values:
    # Initialize and fit the logistic regression model with regularization
    model = LogisticRegression(solver='liblinear', C=C_value, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = model.predict(X_val)

    # Calculate accuracy
    val_accuracy = round(accuracy_score(y_val, y_pred), 3)
    accuracy_results[C_value] = val_accuracy
print("All results: ", accuracy_results, "\n")
print("best accuracy on the validation set: \n")
max_key = max(accuracy_results, key=accuracy_results.get)
print(max_key, accuracy_results[max_key])
