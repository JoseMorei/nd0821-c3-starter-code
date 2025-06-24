from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import os
import pandas as pd
from ml.data import process_data


app = FastAPI()

model = None
encoder = None
lb = None

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Define input schema with field aliasing for hyphenated names
class InferenceInput(BaseModel):
    age: int
    workclass: str = Field(..., alias="workclass")
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 35,
                "workclass": "Private",
                "fnlgt": 284582,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 45,
                "native-country": "United-States"
            }
        }

# starter/main.py

@app.on_event("startup")
def load_model():
    global model, encoder, lb
    # Use absolute path to the model folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "model")

    model_path = os.path.join(model_dir, "model.pkl")
    encoder_path = os.path.join(model_dir, "encoder.pkl")
    lb_path = os.path.join(model_dir, "label_binarizer.pkl")

    print(f"Loading model from: {model_path}")
    print("Exists:", os.path.exists(model_path))

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(encoder_path):
        encoder = joblib.load(encoder_path)
    if os.path.exists(lb_path):
        lb = joblib.load(lb_path)

@app.get("/")
def root():
    return {"message": "Welcome to the Income Prediction API!"}

@app.post("/predict")
def predict(input_data: InferenceInput):
    global model, encoder, lb

    if model is None or encoder is None or lb is None:
        return {"error": "Model or encoders not loaded"}

    input_df = pd.DataFrame([input_data.dict(by_alias=True)])

    # Preprocess using training-time encoder
    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}

__all__ = ["app", "InferenceInput"]
