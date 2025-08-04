# Task 1: Data Cleaning & Preprocessing (Titanic Dataset Example)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Dataset
df = sns.load_dataset('titanic')  # or use pd.read_csv('titanic.csv')

# 2. Explore Basic Info
print("First 5 rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# 3. Handle Missing Values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = df.drop(['deck'], axis=1)

print("\nMissing Values After Imputation:")
print(df.isnull().sum())

# 4. Encode Categorical Features
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])  # male=1, female=0
df = pd.get_dummies(df, columns=['embarked'], prefix='embarked')

print("\nData Types After Encoding:")
print(df.dtypes)

# 5. Normalize/Standardize Numerical Features
num_cols = ['age', 'fare', 'sibsp', 'parch']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nNumerical Features After Standardization:")
print(df[num_cols].head())

# 6. Visualize and Remove Outliers
for col in num_cols:
    plt.figure(figsize=(4,1))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.show()

# Example: Remove 'fare' outliers using IQR
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['fare'] >= Q1 - 1.5 * IQR) & (df['fare'] <= Q3 + 1.5 * IQR)]

# 7. Final Output Preview
print("\nCleaned Data Sample:")
print(df.head())

print("\nCleaned Data Description:")
print(df.describe())
