# Data handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# Warning control
import warnings
warnings.filterwarnings("ignore")

# Modeling
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

#-----------------  SMART PREPROCESSING -------------------#

# Load original dataset
df = pd.read_csv("credit copy 1.csv")

# Step 1: Flag and impute missing numerical values
for col in ["duration", "residence", "cards"]:
    df[col + "_missing"] = df[col].isnull().astype(int)
    df[col].fillna(df[col].median(), inplace=True)

# Step 2: Impute missing categorical values with 'Missing'
categorical_cols = df.select_dtypes("object").columns[df.select_dtypes("object").isnull().any()].tolist()
for col in categorical_cols:
    df[col] = df[col].fillna("Missing")

# Step 3: Label encode all categorical columns
for col in df.select_dtypes("object").columns:
    df[col] = df[col].astype("category").cat.codes

# Step 4: Feature engineering
df["debt_duration_ratio"] = df["amount"] / (df["duration"] + 1)
df["age_group"] = pd.cut(df["age"], bins=[18, 30, 45, 60, 75], labels=[0, 1, 2, 3])

# Drop any remaining NaNs (from age_group binning)
df = df.dropna()

# Step 5: Save processed dataset
df.to_csv("credit_processed_v2.csv", index=False)

#-----------------  MODELING -------------------#

# Mapping: 1 = Good customer (positive class), 0 = Bad customer (negative class)
X = df.drop("Default", axis=1)
y = df["Default"]

# Shuffle
X = X.sample(frac=1, random_state=42)
y = y.sample(frac=1, random_state=42)
assert all(X.index == y.index)

# Train/test split
split_index = round(0.8 * len(df))
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

# Train decision tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))

# Cost Matrix Evaluation
cm = confusion_matrix(y_test, y_pred)
FP = cm[0][1]
FN = cm[1][0]
cost = FP * 1 + FN * 10
print("Cost Matrix Score:", cost)

#-----------------  10-FOLD CROSS-VALIDATION -------------------#

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
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
