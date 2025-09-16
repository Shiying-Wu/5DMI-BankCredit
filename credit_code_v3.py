# credit_code_v3.py

# 📦 Imports
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ------------------ 📁 Load Original Data ------------------ #
df = pd.read_csv("credit copy 1.csv")

# ------------------ 🧠 Preprocessing (v3 Strategy) ------------------ #

# 1️⃣ Missing Flag + Imputation (Feature Creation + Aggregation)
for col in ["duration", "residence", "cards"]:
    df[col + "_missing"] = df[col].isnull().astype(int)
    df[col].fillna(df[col].median(), inplace=True)

# 2️⃣ Fill missing categorical values with 'Missing'
cat_cols_with_na = df.select_dtypes("object").columns[df.select_dtypes("object").isnull().any()].tolist()
for col in cat_cols_with_na:
    df[col] = df[col].fillna("Missing")

# 3️⃣ Label Encoding (Attribute Transformation)
for col in df.select_dtypes("object").columns:
    df[col] = df[col].astype("category").cat.codes

# 4️⃣ Discretization (age binning)
df["age_group"] = pd.cut(df["age"], bins=[18, 30, 45, 60, 75], labels=[0, 1, 2, 3])

# 5️⃣ Feature Creation
df["debt_duration_ratio"] = df["amount"] / (df["duration"] + 1)

# 6️⃣ Drop any remaining NaNs (from binning)
df = df.dropna()

# 7️⃣ Save processed dataset
df.to_csv("credit_processed_v3.csv", index=False)

# ------------------ 🤖 Modeling ------------------ #

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

# ------------------ 🔁 10-Fold Cross-Validation ------------------ #

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
