from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# GET test
def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "Hello" in res.json()["message"]

# POST test - over 50K 
def test_predict_over_50k():
    input_data = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlwgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
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
    res = client.post("/predict", json=input_data)
    assert res.status_code == 200
    assert res.json()["prediction"] in [0, 1]
    
# POST test - under 50K 
def test_predict_under_50k():
    input_data = {
        "age": 23,
        "workclass": "Private",
        "fnlwgt": 190709,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    }
    res = client.post("/predict", json=input_data)
    assert res.status_code == 200
    assert res.json()["prediction"] in [0, 1]
