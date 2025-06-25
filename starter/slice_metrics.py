#  Evaluate model performance on slices of data for each categorical feature.
from ml.model import inference, compute_model_metrics
from ml.data import process_data

def evaluate_slices(model, data, label_col, categorical_features, encoder, lb, output_path="slice_output.txt"):
    with open(output_path, "w") as f:
        for col in categorical_features:
            for cls in data[col].unique():
                df_slice = data[data[col] == cls]
                if df_slice.empty:
                    continue
                X_slice, y_slice, _, _ = process_data(
                    df_slice,
                    categorical_features=categorical_features,
                    label=label_col,
                    training=False,
                    encoder=encoder,
                    lb=lb,
                )
                preds = inference(model, X_slice)
                precision, recall, f1 = compute_model_metrics(y_slice, preds)
                f.write(f"{col} - {cls} | Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}\n")