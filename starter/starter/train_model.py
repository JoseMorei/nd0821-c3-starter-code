# This script trains on the clean data and saves the model. 
#  Add the necessary imports for the starter code.
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from slice_metrics import evaluate_slices

# Add code to load in the data.
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/census.csv"))
data = pd.read_csv(data_path)


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

#Prints overall model performance (Precision, Recall, F1)
def print_model_performance(model, X_test, y_test):
    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)
    print("=== Overall Model Performance ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

# Call slice evaluation
evaluate_slices(
    model=model,
    data=test,
    label_col="salary",
    categorical_features=cat_features,
    encoder=encoder,
    lb=lb,
    output_path="slice_output.txt"
)

# Print overall model performance
print_model_performance(model, X_test, y_test)