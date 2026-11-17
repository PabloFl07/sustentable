
"""
Example: Save and load a Scikit-learn Linear Regression model
using ONNX and JSON.

Generated files:
    models/model.onnx
    models/model_metadata.json
"""

import json
import platform
from pathlib import Path
from examples.linear_regression import train_model, DATA_FILE, EXPERIMENTS, data_preparation, model_formula, evaluate_model, TARGET_COLUMN
from sustentable.core.io.readers import get_reader
from pandas import DataFrame

import numpy as np
import onnx
import onnxruntime as ort
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

ONNX_PATH = OUTPUT_DIR / "model.onnx"
JSON_PATH = OUTPUT_DIR / "model_metadata.json"

data = get_reader(DATA_FILE).read()

for new_values in EXPERIMENTS:

    features = list(new_values)
    INPUT_COLUMNS = features

    X_train, X_test, y_train, y_test = data_preparation(data, features)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_train, y_train, "train")
    evaluate_model(model, X_test, y_test, "test")

    new_input = DataFrame([new_values])



# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation metrics
metrics = {
    "train": {
        "r2": float(r2_score(y_train, y_train_pred)),
        "mse": float(mean_squared_error(y_train, y_train_pred))
    },
    "test": {
        "r2": float(r2_score(y_test, y_test_pred)),
        "mse": float(mean_squared_error(y_test, y_test_pred))
    }
}


# --------------------------------------------------
# 4. Convert model to ONNX
# --------------------------------------------------

# ONNX input must use float32
initial_types = [
    ("input", FloatTensorType([None, len(INPUT_COLUMNS)]))
]

onnx_model = convert_sklearn(
    model,
    initial_types=initial_types
)

print("\nONNX input:")
for input_info in onnx_model.graph.input:
    print(input_info.name, input_info.type)

# Validate ONNX model
onnx.checker.check_model(onnx_model)

# Save ONNX model
with open(ONNX_PATH, "wb") as file:
    file.write(onnx_model.SerializeToString())

print(f"ONNX model saved: {ONNX_PATH}")


# --------------------------------------------------
# 5. Generate model metadata
# --------------------------------------------------

# Build formula from trained coefficients
formula = (model_formula(model))

metadata = {
    "name": "Linear Regression - Energy Consumption",
    "format": "ONNX",
    "version": "1.0",

    "model_type": "LinearRegression",

    "formula": formula,

    "input_columns": INPUT_COLUMNS,
    "output_columns": TARGET_COLUMN,

    "parameters": {
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_)
    },

    "metrics": metrics,

    "description": (
        "Linear regression model that predicts energy "
        "consumption based on temperature and humidity."
    ),

    "training": {
        "samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "random_state": 42
    },

    "dependencies": {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__
    }
}

# Save metadata as JSON
with open(JSON_PATH, "w", encoding="utf-8") as file:
    json.dump(
        metadata,
        file,
        indent=4,
        ensure_ascii=False
    )

print(f"JSON metadata saved: {JSON_PATH}")


# --------------------------------------------------
# 6. Load ONNX model
# --------------------------------------------------

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("\nONNX model loaded successfully.")


# --------------------------------------------------
# 7. Load JSON metadata
# --------------------------------------------------

with open(JSON_PATH, "r", encoding="utf-8") as file:
    loaded_metadata = json.load(file)

print("JSON metadata loaded successfully.")

print("\nModel formula:")
print(loaded_metadata["formula"])

print("\nInput columns:")
print(loaded_metadata["input_columns"])

print("\nOutput columns:")
print(loaded_metadata["output_columns"])


# --------------------------------------------------
# 8. Make predictions using ONNX Runtime
# --------------------------------------------------

X_test_onnx = np.asarray(X_test, dtype=np.float32)

# Inspect expected input
input_info = session.get_inputs()[0]

print("\nONNX input information:")
print("Name:", input_info.name)
print("Type:", input_info.type)
print("Shape:", input_info.shape)

# Run inference
onnx_predictions = session.run(
    None,
    {input_info.name: X_test_onnx}
)[0].reshape(-1)

# Compare with Scikit-learn
sklearn_predictions = model.predict(X_test)

print("\nPrediction comparison:")
for i in range(min(5, len(sklearn_predictions))):
    print(
        f"Scikit-learn: {sklearn_predictions[i]:.4f} | "
        f"ONNX: {onnx_predictions[i]:.4f}"
    )

np.testing.assert_allclose(
    sklearn_predictions,
    onnx_predictions,
    rtol=1e-4,
    atol=1e-4
)

print("\nONNX and Scikit-learn predictions match!")


# --------------------------------------------------
# 9. Evaluate ONNX predictions
# --------------------------------------------------

onnx_r2 = r2_score(y_test, onnx_predictions)
onnx_mse = mean_squared_error(y_test, onnx_predictions)

print("\nONNX test metrics:")
print(f"R2:  {onnx_r2:.4f}")
print(f"MSE: {onnx_mse:.4f}")

print("\nAll operations completed successfully.")