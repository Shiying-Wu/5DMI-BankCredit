# Data handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# Warning control
import warnings
warnings.filterwarnings("ignore")

# Decision tree
from sklearn.tree import DecisionTreeClassifier, plot_tree

#Neural network
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense

#-----------------  DATA PREPROCESSING  -------------------#

# Load dataset
df = pd.read_csv("./credit copy 1.csv")

# Unique value summary
print(" Unique values per column:")
print(df.nunique())

# Missing value summary
#missing_values = df.isnull().sum().sort_values(ascending=False)
#missing_values_df = pd.DataFrame({
#    "Missing Value": missing_values.values,
#    "Percentage Missing": (missing_values / len(df)) * 100
#}, index=missing_values.index)
#print("\n Missing value summary:")
#print(missing_values_df)

# Impute numerical column 
df["duration"].fillna(df["duration"].median(), inplace=True)
df["residence"].fillna(df["residence"].median(), inplace=True)
df["cards"].fillna(df["cards"].median(), inplace=True)

# Impute categorical columns with mode
categorical_cols = df.select_dtypes("object").columns[df.select_dtypes("object").isnull().any()].tolist()
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# One-hot encode categorical columns
# .columns.tolist() extracts just the column names from the filtered DataFrame and Converts the column names from an Index object to a regular Python list
categorical_all = df.select_dtypes(include="object").columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_all, drop_first=True)

#output preprocessed dataset 

df_encoded.to_csv("credit_preprocessed.csv",index=False)

# Review resulting DataFrame 
print("\n Encoded DataFrame Info:")
print(df_encoded.info())

print("\n Sample of Encoded DataFrame:")
print(df_encoded.head())

print("\n Statistical Summary:")
print(df_encoded.describe(include='all'))
