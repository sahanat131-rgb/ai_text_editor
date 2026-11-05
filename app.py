from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="AI Text Detector")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "AI Text Detector API is running. Go to /docs to try it."}

@app.post("/predict")
def predict(input: TextInput):
    X = vectorizer.transform([input.text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()
    return {
        "prediction": "AI-generated" if pred == 1 else "Human-written",
        "confidence": round(float(prob), 4)
    }