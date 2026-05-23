from fastapi import FastAPI

import joblib


# Load trained model
model = joblib.load("model.pkl")


# Create FastAPI app
app = FastAPI()


# Home route
@app.get("/")
def home():
    return {"message": "ML API Running"}


# Prediction route
@app.get("/predict/{hours}")
def predict(hours: int):

    prediction = model.predict([[hours]])

    return {
        "hours_studied": hours,
        "predicted_marks": prediction[0]
    }