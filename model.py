import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = "lung_cancer_dataset.csv"  # Update the path if needed
df = pd.read_csv(file_path)

# Convert target variable to binary (YES -> 1, NO -> 0)
df['PULMONARY_DISEASE'] = df['PULMONARY_DISEASE'].map({'YES': 1, 'NO': 0})

# Define features and target
X = df.drop(columns=['PULMONARY_DISEASE'])
y = df['PULMONARY_DISEASE']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_rep)

# Save the trained model and scaler for later use
joblib.dump(model, "lung_cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and Scaler saved successfully!")

# ===========================
# VISUALIZATIONS
# ===========================

# 1. Age Distribution for Lung Cancer vs Non-Patients
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="AGE", hue="PULMONARY_DISEASE", kde=True, bins=30, palette="coolwarm")
plt.title("Age Distribution for Lung Cancer Patients vs Non-Patients")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend(["No Lung Cancer", "Lung Cancer"])
plt.show()

# 2. Correlation Heatmap of Features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()