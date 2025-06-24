from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import os
import pandas as pd

app = FastAPI()

model = None

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

@app.on_event("startup")
def load_model():
    global model
    model_path = os.path.join("starter", "model", "model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = joblib.load(model_path)
#            model = pickle.load(f)

@app.get("/")
def root():
    return {"message": "Welcome to the Income Prediction API!"}

@app.post("/predict")
def predict(input_data: InferenceInput):
    global model
    if model is None:
        return {"error": "Model not loaded"}

    input_df = pd.DataFrame([input_data.dict(by_alias=True)])
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}

__all__ = ["app", "InferenceInput"]
