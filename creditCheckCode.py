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
#print("\n Encoded DataFrame Info:")
#print(df_encoded.info())

#print("\n Sample of Encoded DataFrame:")
#print(df_encoded.head())

#print("\n Statistical Summary:")
#print(df_encoded.describe(include='all'))


#-----------------  DECISION TREE MODELING  -------------------#

# Split features and target
X = df_encoded.drop("Default", axis=1)
y = df_encoded["Default"]

# Optional: shuffle
X = X.sample(frac=1, random_state=42)
y = y.sample(frac=1, random_state=42)
assert all(X.index == y.index)

# Train/test split
split_index = round(0.8 * len(df_encoded))
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

# Train decision tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))

#-----------------  COST MATRIX EVALUATION -------------------#

# Mapping: 1 = Good customer (positive class), 0 = Bad customer (negative class)
# Cost Matrix: FN = 10, FP = 1
cm = confusion_matrix(y_test, y_pred)
FP = cm[0][1]
FN = cm[1][0]
cost = FP * 1 + FN * 10
print("Cost Matrix Score:", cost)

#-----------------  10-FOLD CROSS-VALIDATION -------------------#

from sklearn.model_selection import StratifiedKFold

# Convert bool columns to int for consistency
X = X.astype({col: int for col in X.select_dtypes('bool').columns})

# Set up stratified 10-fold CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Store metrics
cv_accuracies = []
cv_roc_aucs = []
cv_costs = []

for train_index, test_index in skf.split(X, y):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

    model_cv = DecisionTreeClassifier(random_state=42)
    model_cv.fit(X_train_cv, y_train_cv)
    y_pred_cv = model_cv.predict(X_test_cv)

    acc = accuracy_score(y_test_cv, y_pred_cv)
    roc = roc_auc_score(y_test_cv, y_pred_cv)
    cm_cv = confusion_matrix(y_test_cv, y_pred_cv)
    FP_cv = cm_cv[0][1]
    FN_cv = cm_cv[1][0]
    cost_cv = FP_cv * 1 + FN_cv * 10

    cv_accuracies.append(acc)
    cv_roc_aucs.append(roc)
    cv_costs.append(cost_cv)

print("\n--- 10-Fold Cross-Validation Results ---")
print("Average Accuracy:", np.mean(cv_accuracies))
print("Average ROC AUC:", np.mean(cv_roc_aucs))
print("Average Cost Matrix Score:", np.mean(cv_costs))

