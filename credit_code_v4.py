import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency

# ------------------ Load Raw Data ------------------ #
df_raw = pd.read_csv("credit copy 1.csv")
df = df_raw.copy()

# ------------------ Preprocessing ------------------ #
for col in ["duration", "residence", "cards"]:
    df[col + "_missing"] = df[col].isnull().astype(int)
    df[col] = df[col].fillna(df[col].median())

cat_cols_with_na = df.select_dtypes("object").columns[df.select_dtypes("object").isnull().any()].tolist()
for col in cat_cols_with_na:
    df[col] = df[col].fillna("Missing")

for col in df.select_dtypes("object").columns:
    df[col] = df[col].astype("category").cat.codes

df["age_group"] = pd.cut(df["age"], bins=[18, 30, 45, 60, 75], labels=[0, 1, 2, 3])
df["debt_duration_ratio"] = df["amount"] / (df["duration"] + 1)
df = df.dropna()

# ------------------ Mutual Information (All Variables) ------------------ #
X_mi = df.drop(columns=["Default"])
y_mi = df["Default"]
mi_scores = mutual_info_classif(X_mi, y_mi, discrete_features="auto")
mi_table = pd.DataFrame({"Feature": X_mi.columns, "MI Score": mi_scores})
mi_table = mi_table.sort_values(by="MI Score", ascending=False)

print("\n🔍 Mutual Information Scores (All Variables):")
print(mi_table)

# ------------------ Cramér’s V for Raw Categorical ------------------ #
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

cramer_results = []
for col in df_raw.select_dtypes("object").columns:
    score = cramers_v(df_raw[col].fillna("Missing"), df_raw["Default"])
    cramer_results.append((col, score))

cramer_table = pd.DataFrame(cramer_results, columns=["Feature", "Cramér's V"])
cramer_table = cramer_table.sort_values(by="Cramér's V", ascending=False)

print("\n🔍 Cramér’s V Scores (Categorical Variables):")
print(cramer_table)

# ------------------ Groupby Summary for Key Features ------------------ #
# Function to show descriptive statistics of a feature grouped by Default
def groupby_summary(df, col, label):
    print(f"\n📊 Groupby Summary for {label} '{col}'")
    print(df.groupby("Default")[col].describe())

# Compare raw vs processed features
groupby_summary(df_raw, "amount", "Raw")                  # Loan amount
groupby_summary(df, "debt_duration_ratio", "Processed")   # Financial pressure
groupby_summary(df_raw, "age", "Raw")                     # Age
groupby_summary(df, "age_group", "Processed")             # Age bins


# ------------------ Visualization ------------------ #
plt.figure(figsize=(10, 6))
sns.barplot(data=mi_table, x="MI Score", y="Feature", palette="viridis")
plt.title("Mutual Information Scores (All Variables)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=cramer_table, x="Cramér's V", y="Feature", palette="magma")
plt.title("Cramér’s V Scores (Categorical Variables)")
plt.tight_layout()
plt.show()

# ------------------ Save Processed Dataset ------------------ #
df.to_csv("credit_processed_v4.csv", index=False)



