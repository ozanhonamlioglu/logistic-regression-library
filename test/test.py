import sys
import os

# Adds the parent directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ******** DATA CLEANING ********
# 1. Load data and drop Date
df = pd.read_csv('./weatherAUS.csv')
df = df.drop(columns=['Date'])

# Clean the data sheet
# 2. Drop rows where target is missing and handle NA values
df = df.dropna(subset=['RainTomorrow'])
numeric_cols = df.select_dtypes(include=['float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
  df[col] = df[col].fillna(df[col].mode()[0])

# Encode the columns and their data
# 3. Encoding categorical variables
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})
df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})

le = LabelEncoder()
for col in ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']:
  df[col] = le.fit_transform(df[col])
  
from sklearn.model_selection import train_test_split
X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# ******** TESTING ********
import numpy as np
from logistic_regression import LogisticRegression
from sklearn.metrics import accuracy_score

logistic = LogisticRegression(epoch=2000)
logistic.train(X_train.to_numpy(), y_train.to_numpy())

# print(logistic.weights)
predictions = logistic.predict(X_test)
print(f"Accuracy score: {accuracy_score(y_test, predictions)}")

comparison_df = pd.DataFrame({
  "Actual": y_test[:10].values,
  "Prediction": predictions[:10].flatten()
})

print(comparison_df)