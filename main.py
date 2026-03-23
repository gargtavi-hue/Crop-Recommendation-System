import pandas as pd

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

print(data.head())
print(data.info())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split features and target
X = data.drop("label", axis=1)
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()

# Train
model.fit(X_train, y_train)

import pickle

# Save the model
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Model Accuracy: {accuracy*100:.2f}%")

print("\n🌿 Welcome to Crop Recommendation System 🌿")

# Take user input

print("\n📋 INPUT SUMMARY")
print("="*30)
N = float(input("Nitrogen: "))
P = float(input("Phosphorus: "))
K = float(input("Potassium: "))
temperature = float(input("Temperature: "))
humidity = float(input("Humidity: "))
ph = float(input("pH: "))
rainfall = float(input("Rainfall: "))
print("="*30)

# Make prediction

input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
prediction = model.predict(input_data)

print("\n" + "="*40)
print("🌾 CROP RECOMMENDATION RESULT")
print("="*40)
print(f"✅ Recommended Crop: {prediction[0].upper()}")
print("="*40)

if prediction[0] == "rice":
    print("💡 Suggestion: Ensure high water availability for best yield.")
elif prediction[0] == "wheat":
    print("💡 Suggestion: Suitable for moderate temperature and low rainfall.")
elif prediction[0] == "cotton":
    print("💡 Suggestion: Requires warm climate and moderate rainfall.")
else:
    print("💡 Suggestion: Ensure proper soil and climate conditions.")

import matplotlib.pyplot as plt

# Get feature importance from model
importance = model.feature_importances_
features = X.columns

# Plot
plt.figure()
plt.bar(features, importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()