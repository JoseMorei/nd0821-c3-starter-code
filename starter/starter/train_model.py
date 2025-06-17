# This script trains on the clean data and saves the model. 

#  Add the necessary imports for the starter code.
import pandas as pd
#from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
#from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import os

from data import process_data
from model import train_model, inference

# Add code to load in the data.
data = pd.read_csv("data/census_clean.csv")

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

train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process test data using the encoder and label binarizer from training
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)
# Train the model
model = train_model(X_train, y_train)

# Save final model, encoder, and label binarizer from last fold
def save_model(model, encoder, lb, path="model"):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, "model.pkl"))
    joblib.dump(encoder, os.path.join(path, "encoder.pkl"))
    joblib.dump(lb, os.path.join(path, "label_binarizer.pkl"))

save_model(model, encoder, lb)