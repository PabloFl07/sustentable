
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

INPUT_COLUMNS = ["temperatura", "humedad"]
OUTPUT_COLUMNS = ["consumo"]


# --------------------------------------------------
# 2. Generate example dataset
# --------------------------------------------------

# Reproducible synthetic dataset
rng = np.random.default_rng(42)

X = rng.uniform(10, 40, size=(200, 2))

# Linear relationship:
# consumo = 2.5 * temperatura + 1.8 * humedad + 0.7
y = (
    2.5 * X[:, 0]
    + 1.8 * X[:, 1]
    + 0.7
    + rng.normal(0, 2, size=200)
)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 3. Train Scikit-learn model
# --------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

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
formula = (
    f"consumo = "
    f"{model.coef_[0]:.6f} * temperatura + "
    f"{model.coef_[1]:.6f} * humedad + "
    f"{model.intercept_:.6f}"
)

metadata = {
    "name": "Linear Regression - Energy Consumption",
    "format": "ONNX",
    "version": "1.0",

    "model_type": "LinearRegression",

    "formula": formula,

    "input_columns": INPUT_COLUMNS,
    "output_columns": OUTPUT_COLUMNS,

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

# ONNX expects float32 input
X_test_onnx = X_test.astype(np.float32)

onnx_predictions = session.run(
    [output_name],
    {input_name: X_test_onnx}
)[0].reshape(-1)

# Compare with original Scikit-learn predictions
sklearn_predictions = model.predict(X_test)

print("\nPrediction comparison:")
for i in range(5):
    print(
        f"Scikit-learn: {sklearn_predictions[i]:.4f} | "
        f"ONNX: {onnx_predictions[i]:.4f}"
    )

# Verify that both implementations produce equivalent results
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