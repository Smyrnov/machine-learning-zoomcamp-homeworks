import pandas as pd

df = pd.read_csv('bank-full.csv', delimiter=';')
# For the rest of the homework, you'll need to use only these columns

# Data preparation
df_subset = df[[
    "age", "job", "marital", "education", "balance", "housing", "contact",
    "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"
]]
missing_values = df_subset.isnull().sum()
print(missing_values,"\n")

# Question 1
# What is the most frequent observation (mode) for the column education?

print(f"The most frequent observation (mode) for the column education: {df_subset['education'].mode()[0]}\n")
