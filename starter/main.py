# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Create the app
app = FastAPI()

# Define what input should looks like
class PersonInput(BaseModel):
    age: int
    workclass: str = Field(..., alias="workclass")
    fnlwgt: int
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
                "age": 39,
                "workclass": "State-gov",
                "fnlwgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }
# root path
@app.get("/")
def hello():
    return {"message": "Hello Udacity"}

# prediction path
@app.post("/predict")
def make_prediction(input_data: PersonInput):
    data = pd.DataFrame([input_data.model_dump(by_alias=True)])
    
    # Use model to predict
    prediction = model.predict(data)[0]
    return {"prediction": int(prediction)}
