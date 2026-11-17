import json, joblib
import numpy as np
from sklearn.linear_model import LinearRegression


## SAVING USING JSON

data = {
    "type": "LinearRegression",
    "coefficients": [2.5, 1.8],
    "intercept": 0.7,
    "formula": "y = 2.5*x1 + 1.8*x2 + 0.7",
    "input_columns": ["temperature", "humidity"],
    "output_columns": ["consumption"],
    "metrics": {
        "train": {"r2": 0.95},
        "test": {"r2": 0.91}
    },
    "description": "Energy consumption prediction model."
}

with open("model.json", "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)


## LOADING USING JSON

with open("model.json", encoding="utf-8") as file:
    data = json.load(file)

model = LinearRegression()

model.coef_ = np.array(data["coefficients"])
model.intercept_ = data["intercept"]
model.n_features_in_ = len(data["coefficients"])
model.feature_names_in_ = np.array(data["input_columns"])


## SAVING USING JOBLIB

data = {
    "model": model,
    "formula": "y = a*x + b",
    "input_columns": ["temperature", "humidity"],
    "output_columns": ["consumption"],
    "metrics": {
        "train": {"r2": 0.95},
        "test": {"r2": 0.91}
    },
    "description": "Energy consumption prediction model."
}

joblib.dump(data, "model.joblib")

## LOADING USING JOBLIB

data = joblib.load("model.joblib")

model = data["model"]
formula = data["formula"]
metrics = data["metrics"]