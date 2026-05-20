import pandas as pd

from sklearn.linear_model import LinearRegression

import joblib


# Training data
hours = [[1], [2], [3], [4], [5]]

marks = [20, 40, 60, 80, 100]


# Create model
model = LinearRegression()


# Train model
model.fit(hours, marks)


# Predict
prediction = model.predict([[6]])

print("Predicted Marks:", prediction)


# Save model
joblib.dump(model, "model.pkl")

print("Model Saved")