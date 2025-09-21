# ------------------ Import Libraries ------------------ #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ------------------ Load Credit Risk Dataset ------------------ #
df = pd.read_csv("credit_processed_v4.csv")
print(df.head())
print("Shape of dataset:", df.shape)

# ------------------ Feature Selection ------------------ #
selected_features = [
    "checkingstatus1", "savings", "history", "duration", "purpose",
    "property", "status", "foreign", "job", "otherplans", "employ",
    "age_group", "debt_duration_ratio", "amount", "age",
    "residence_missing", "duration_missing", "cards_missing"
]

features = df[selected_features]
labels = df["Default"]

# ------------------ Train/Test Split ------------------ #
X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=42)

# ------------------ Standardize Features ------------------ #
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ Initialize and Train Optimized MLPClassifier ------------------ #
mlp = MLPClassifier(
    hidden_layer_sizes=(10, 10),       # Two hidden layers with 10 neurons each
    solver='adam',                     # Fast, adaptive optimizer
    learning_rate_init=0.005,          # Slightly higher learning rate
    max_iter=1500,                     # Increased iterations for convergence
    random_state=0,
    verbose=False
)
mlp.fit(X_train, y_train)

# ------------------ Predict and Evaluate ------------------ #
predictions = mlp.predict(X_test)
conf_matrix = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
tn, fp, fn, tp = conf_matrix.ravel()
cost = fp * 10 + fn * 1

# ------------------ Output Results ------------------ #
print("\n✅ Accuracy =", accuracy)
print("📊 Confusion Matrix:\n", conf_matrix)
print("📋 Classification Report:\n", classification_report(y_test, predictions))
print(f"💰 Cost (FP×10 + FN×1): {cost}")

# ------------------ Plot Loss Curve ------------------ #
plt.figure(dpi=125)
plt.plot(mlp.loss_curve_)
plt.title("Loss Curve (Optimized MLPClassifier)")
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.grid(True)
plt.show()
