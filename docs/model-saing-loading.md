Claro. Aquí tienes el documento completo en inglés, en Markdown puro, listo para copiar y guardar como `model_persistence_comparison.md`.

markdown

````markdown

# Comparison of Model Persistence Formats: ONNX, Joblib, and JSON

This document compares ONNX, Joblib, and JSON as methods for saving and loading machine learning models in the Sustentable project. The comparison evaluates ease of use, file size and readability, security, compatibility across Python and library versions, and the ability to store metadata required by the application.

The comparison focuses primarily on models trained with Python and scikit-learn. It also considers that the three formats do not serve exactly the same purpose: Joblib serializes Python objects, ONNX represents models for execution through a compatible runtime, and JSON stores structured data but does not directly serialize a scikit-learn model.

## Comparison

| Criterion | ONNX | Joblib | JSON |
| --- | --- | --- | --- |
| Ease of use | Requires converting the model to ONNX, usually with `skl2onnx`, and using `onnxruntime` to load and execute it. Some additional configuration is required. | Very easy for scikit-learn models. Save with `joblib.dump()` and load with `joblib.load()`. | Very easy. Uses Python's `json.dump()` and `json.load()`. However, the model must be reconstructed from its parameters. |
| File size | Generally compact, although it depends on the model architecture, operators, and stored parameters. | Variable. It supports compression, especially for large NumPy arrays. | Can be considerably larger because numbers and structures are stored as text. Not ideal for large matrices. |
| Readability | Not designed for human readability. It is a binary format based on a structured representation of the computational graph. It can be inspected using ONNX tools. | Not directly readable. It is a binary serialization format for Python objects. | Highly readable. Files contain text-based structures such as objects, lists, strings, and numbers. They can be opened with any text editor. |
| Security when loading unknown files | Does not use Python's `pickle` object deserialization mechanism. Therefore, it avoids that specific risk. However, models should still be validated and up-to-date runtime versions should be used. | Unsafe. It uses serialization mechanisms based on `pickle`, so loading a malicious file can execute arbitrary code. | Does not execute Python code when using Python's standard `json` parser. It is the safest of the three against code execution during loading. Input data must still be validated. |
| Compatibility across Python and library versions | High. Once converted, the model can run in other languages and environments that support ONNX Runtime, without requiring the original Python or scikit-learn versions. Compatibility depends on the operators and ONNX format and runtime versions. | More limited. It may depend on the Python, Joblib, NumPy, scikit-learn, and other library versions used to train the model. Compatibility across different scikit-learn versions is not guaranteed. | High for data. JSON is independent of Python and machine learning libraries. However, model reconstruction depends on the implementation used by the application. |
| Storing additional information | Supports metadata through properties such as `metadata_props`. Formula, columns, metrics, and descriptions can be included. | Allows storing a dictionary or structure containing the model and all its metadata in a single file. | Particularly suitable for storing metadata: formula, input and output columns, metrics, description, and model parameters. |
| Model recovery | The model is executed through ONNX Runtime. There is no need to reconstruct the original scikit-learn estimator. | The serialized Python object is recovered directly, preserving its structure and behavior, provided that dependencies are compatible. | Does not directly recover a scikit-learn object. The estimator must be reconstructed using Python code and its stored parameters. |
| Dependencies | `onnx`, `onnxruntime`, and, for converting scikit-learn models, `skl2onnx`. | `joblib` and the dependencies used by the model, such as scikit-learn and NumPy. | Python's standard `json` library. Reconstructing the model requires the corresponding libraries. |

## Security When Loading Models

### Why can Pickle and Joblib execute arbitrary code?

Python uses the `pickle` module to convert Python objects into a byte stream and recover them later.

The problem is that `pickle` does not simply store an object's data. During deserialization, it can import and execute functions to reconstruct the object. Therefore, a manipulated file can contain instructions that execute automatically when it is loaded.

The official Python documentation explicitly warns that `pickle` is not secure and that files from untrusted sources should never be loaded. It recommends using safer formats, such as JSON, when processing unknown data.

Reference: [Python Documentation: pickle](https://docs.python.org/3/library/pickle.html)

Joblib has the same problem because it uses the `pickle` serialization protocol. Its documentation also warns that loading a Joblib file from an untrusted source is equivalent to executing untrusted code.

Reference: [Joblib: Persistence](https://joblib.readthedocs.io/en/latest/persistence.html)

### Proof of concept: code execution when loading a Pickle file

The following harmless example demonstrates this behavior. An object is created whose reconstruction executes `print()`. There is no need to explicitly call this method: the instruction runs when the file is loaded.

```python
import pickle

class TestObject:
    def __reduce__(self):
        return (print, ("Code executed during loading!",))

# Save the object
obj = TestObject()

with open("test.pkl", "wb") as file:
    pickle.dump(obj, file)

# Load the object
with open("test.pkl", "rb") as file:
    loaded_obj = pickle.load(file)
```

**Expected console output:**

```text
Code executed during loading!
```

Calling `pickle.load()` causes `print()` to execute as part of the object reconstruction process.

### Checking the same risk with Joblib

Since Joblib uses the `pickle` protocol internally, the same test can be performed:

```python
import joblib

class TestObject:
    def __reduce__(self):
        return (print, ("Code executed during loading with Joblib!",))

# Save the object
obj = TestObject()
joblib.dump(obj, "test.joblib")

# Load the object
loaded_obj = joblib.load("test.joblib")
```

**Expected output:**

```text
Code executed during loading with Joblib!
```

This demonstrates that Joblib inherits Pickle's security risk. It should not be used to load models from untrusted sources.

Reference: [Joblib: Persistence](https://joblib.readthedocs.io/en/latest/persistence.html)

### What happens with ONNX and JSON?

| Format | Can it execute arbitrary Python code when loading a file? | Security limitations |
| --- | --- | --- |
| ONNX | It does not use Python's Pickle object deserialization mechanism. | A model can represent computational operations and potentially cause excessive memory or resource consumption. Running it in an isolated environment (sandbox) is recommended. |
| Joblib | Yes, potentially. | A manipulated file can execute code during deserialization. |
| JSON | No, when using Python's standard `json` parser. | It can contain incorrect, excessively large, or unexpected data. Its structure and values must be validated before use. |

ONNX avoids the specific risk of code execution through `pickle`, but it should not be considered completely risk-free. Scikit-learn recommends running ONNX models in isolated environments to reduce potential attacks involving memory and resource consumption.

JSON, on the other hand, only represents data. It does not automatically reconstruct Python objects or execute functions stored in the file.

References:
- [Scikit-learn: Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
- [Python Documentation: pickle](https://docs.python.org/3/library/pickle.html)

## Compatibility and Metadata Storage

### ONNX

ONNX allows additional information to be stored through `metadata_props`, a collection of key-value pairs.

For example, the following information can be included:

- The formula or mathematical description of the model.
- Input columns.
- Output columns.
- Training and test metrics.
- Model description.
- Model version and training process information.

This information is stored as text, so complex structures must be converted into a format such as JSON before being added to ONNX metadata.

An important advantage is that the model can be used without installing scikit-learn or having access to the original Python object. ONNX Runtime allows models to be executed across different languages and platforms. Compatibility depends on the format versions, supported operators, and runtime.

References:
- [ONNX: Intermediate Representation](https://onnx.ai/onnx/repo-docs/IR.html)
- [ONNX Runtime: Compatibility](https://onnxruntime.ai/docs/reference/compatibility.html)

### Joblib

Joblib allows a dictionary containing the model and all additional information to be stored:

```python
import joblib

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
```

Loading the information is straightforward:

```python
data = joblib.load("model.joblib")

model = data["model"]
formula = data["formula"]
metrics = data["metrics"]
```

Its main advantage is that it preserves the original object and allows all information to be stored in a single file. Its main disadvantages are security risks and dependencies on Python and library versions.

Scikit-learn states that backward compatibility is not guaranteed when loading models saved with a different scikit-learn version. Therefore, it is recommended to preserve the dependency versions used during training.

Reference: [Scikit-learn: Model Persistence](https://scikit-learn.org/stable/model_persistence.html)

### JSON

JSON is suitable for storing model parameters and metadata, but it cannot directly serialize a scikit-learn estimator.

For example, the coefficients of a linear regression model could be stored as follows:

```python
import json

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
```

To use the model, it must be reconstructed:

```python
import json
import numpy as np
from sklearn.linear_model import LinearRegression

with open("model.json", encoding="utf-8") as file:
    data = json.load(file)

model = LinearRegression()

model.coef_ = np.array(data["coefficients"])
model.intercept_ = data["intercept"]
model.n_features_in_ = len(data["coefficients"])
model.feature_names_in_ = np.array(data["input_columns"])
```

This example reconstructs the parameters of a simple linear regression model. It is not a general solution for every model type: each estimator requires its attributes to be saved and restored correctly. There may also be dependencies on transformations, preprocessors, and other pipeline components.

JSON's main advantage is that its data is readable, independent of Python, and easy to validate. Its main disadvantage is that model reconstruction must be implemented and maintained in the application code.

## Decision

For the Sustentable project, the recommendation is to use **ONNX to store and execute the model, accompanied by a JSON file to store its metadata**.

This combination provides a good balance between security, compatibility, and maintainability.

The reasons are:

1. **Security:** ONNX and JSON do not use Python's Pickle object deserialization mechanism. This avoids the specific risk of arbitrary code execution associated with Joblib and Pickle.
2. **Compatibility:** ONNX allows models to run without depending on the original Python or scikit-learn versions. JSON allows metadata to be retrieved across different programming languages.
3. **Maintainability:** JSON makes it easy to inspect and modify model information without loading the model itself.
4. **Portability:** ONNX allows models to be deployed in environments different from the one in which they were trained.
5. **Additional information:** JSON makes it straightforward to store formulas, columns, metrics, and descriptions required by Sustentable.

Joblib remains a valid alternative during development and for storing models that are only loaded from controlled and trusted sources. However, it is not recommended as the primary format for distributing models.

Scikit-learn's official documentation also recommends ONNX when the application only needs to execute predictions and does not need to recover the original Python object.

Reference: [Scikit-learn: Model Persistence](https://scikit-learn.org/stable/model_persistence.html)

### Proposed File Structure

```text
models/
├── model.onnx
└── model_metadata.json
```

- `model.onnx`: contains the structure and parameters required to make predictions.
- `model_metadata.json`: contains the formula, input and output columns, training and test metrics, description, and versions used to generate the model.

Example metadata:

```json
{
    "name": "Energy consumption prediction model",
    "format": "ONNX",
    "version": "1.0",
    "formula": "y = 2.5*x1 + 1.8*x2 + 0.7",
    "input_columns": [
        "temperature",
        "humidity"
    ],
    "output_columns": [
        "consumption"
    ],
    "metrics": {
        "train": {
            "r2": 0.95
        },
        "test": {
            "r2": 0.91
        }
    },
    "description": "Model for predicting energy consumption.",
    "dependencies": {
        "python": "3.x",
        "scikit-learn": "version used",
        "onnx": "version used",
        "onnxruntime": "version used"
    }
}
```

The JSON file can also record the model's origin, training date, and a reference to the dataset used.

## References

Official documentation consulted:

1. **Python – pickle: Python object serialization.** Security warnings and risks of executing code during deserialization.  
   https://docs.python.org/3/library/pickle.html

2. **Scikit-learn – Model persistence.** Comparison of persistence formats, compatibility across versions, and security recommendations.  
   https://scikit-learn.org/stable/model_persistence.html

3. **Joblib – Persistence.** Explanation of the risk of code execution when loading Joblib files.  
   https://joblib.readthedocs.io/en/latest/persistence.html

4. **ONNX – Intermediate Representation.** Model structure and metadata storage through `metadata_props`.  
   https://onnx.ai/onnx/repo-docs/IR.html

5. **ONNX Runtime – Compatibility.** Compatibility between ONNX Runtime versions and ONNX operators.  
   https://onnxruntime.ai/docs/reference/compatibility.html

## Conclusion

ONNX is the best option among the three formats for distributing and executing Sustentable models, while JSON is particularly useful for storing the additional information required by the application.

Joblib stands out for its ease of use, but its security risks and compatibility limitations make it less suitable for model distribution.
````