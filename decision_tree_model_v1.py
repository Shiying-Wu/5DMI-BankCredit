# ------------------ Import Libraries ------------------ #
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Load Processed Credit Data ------------------ #
df = pd.read_csv("credit_processed_v4.csv")
print(df.head())
print("Shape of dataset:", df.shape)

# ------------------ Feature Selection ------------------ #
# Based on diagnostic analysis (MI, Cramér’s V, groupby)
selected_features = [
    "checkingstatus1", "savings", "history", "duration", "purpose",
    "property", "status", "foreign", "job", "otherplans", "employ",
    "age_group", "debt_duration_ratio", "amount", "age",
    "residence_missing", "duration_missing", "cards_missing"
]

features = df[selected_features]
classLabels = df["Default"]  # 0 = good customer, 1 = bad customer

# ------------------ Shuffle Features and Labels ------------------ #
features = features.sample(frac=1, random_state=42)
classLabels = classLabels.sample(frac=1, random_state=42)

# Ensure indices are aligned
assert all(features.index == classLabels.index)

# ------------------ Split 80% Train / 20% Test ------------------ #
split_index = round(0.8 * len(df))
trainFeatures = features.iloc[:split_index]
trainClassLabels = classLabels.iloc[:split_index]
testFeatures = features.iloc[split_index:]
testClassLabels = classLabels.iloc[split_index:]

# ------------------ Train Decision Tree ------------------ #
treeLearner = DecisionTreeClassifier(random_state=42)
classifier = treeLearner.fit(trainFeatures, trainClassLabels)

# ------------------ Make Predictions ------------------ #
predictions = classifier.predict(testFeatures)
probabilities = classifier.predict_proba(testFeatures)[:, 1]

# ------------------ Evaluation ------------------ #
conf_matrix = confusion_matrix(testClassLabels, predictions)
accuracy = accuracy_score(testClassLabels, predictions)
roc_auc = roc_auc_score(testClassLabels, probabilities)

# ------------------ Cost Matrix Evaluation ------------------ #
# Cost: FP × 10 + FN × 1
tn, fp, fn, tp = conf_matrix.ravel()
cost = fp * 10 + fn * 1

# ------------------ Print Results ------------------ #
print("\n📊 Confusion Matrix:")
print(conf_matrix)

print(f"\n✅ Accuracy: {accuracy:.4f}")
print(f"📈 ROC AUC Score: {roc_auc:.4f}")
print(f"💰 Cost (FP×10 + FN×1): {cost}")

# ------------------ Visualize Confusion Matrix ------------------ #
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
