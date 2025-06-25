from fastapi.testclient import TestClient
from main import app, InferenceInput 

client = TestClient(app)

# GET test
def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["message"] == "Welcome to the Income Prediction API!"

# POST test - over 50K 
def test_predict_over_50k():
    input_data = InferenceInput(
        age=52,
        workclass="Self-emp-not-inc",
        fnlgt=209642,
        education="HS-grad",
        education_num=9,
        marital_status="Married-civ-spouse",
        occupation="Exec-managerial",
        relationship="Husband",
        race="White",
        sex="Male",
        capital_gain=0,
        capital_loss=0,
        hours_per_week=45,
        native_country="United-States"
    )
    res = client.post("/predict", json=input_data.dict(by_alias=True))
    assert res.status_code == 200
    
# # POST test - under 50K 
def test_predict_under_50k():
    input_data = InferenceInput(
        age= 23,
        workclass= "Private",
        fnlgt= 190709,
        education= "Some-college",
        education_num= 10,
        marital_status= "Never-married",
        occupation= "Other-service",
        relationship= "Own-child",
        race= "Black",
        sex= "Female",
        capital_gain=0,
        capital_loss=0,
        hours_per_week= 20,
        native_country= "United-States"
    )
    res = client.post("/predict", json=input_data.dict(by_alias=True))
    assert res.status_code == 200
