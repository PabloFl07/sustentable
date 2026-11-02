# Linear regression libraries for Sustentable

This note compares scikit-learn 1.9.1 and statsmodels 0.15.0 for building the linear regression models in Sustentable. Both are mature, open-source Python libraries that can fit simple and multiple linear regressions. NumPy (`numpy.polyfit`, `numpy.linalg.lstsq`) was not considered, because it only solves the least-squares problem and leaves splitting, prediction, and metrics to the project.

## Comparison

| Criterion | scikit-learn 1.9.1 | statsmodels 0.15.0 |
| --- | --- | --- |
| Ease of use and documentation | Every model follows the same estimator API: `fit()`, `predict()`, and `score()`. The user guide and examples are written for newcomers to machine learning. Large community. | Oriented to statistical analysis. `OLS(y, X).fit()` or the formula API (`ols("y ~ x1 + x2", data)`). The documentation assumes some background in statistics. |
| Simple and multiple regression | `LinearRegression` takes a `DataFrame` with one or several columns. The code is the same in both cases. | `OLS` also takes one or several columns. The intercept must be added with `add_constant()`, unless the formula API is used. |
| Coefficients and intercept | `model.coef_`, `model.intercept_`, and `model.feature_names_in_` give everything needed to write the formula with the column names. | `results.params` is a `Series` indexed by column name, with the intercept as `const`. `summary()` also reports standard errors, p-values, and confidence intervals. |
| Error metrics | `sklearn.metrics` has `mean_squared_error`, `r2_score`, `mean_absolute_error`, and others, and they work on any set of predictions. | `results.rsquared` and `results.mse_resid` describe the training fit. Metrics on new data need `statsmodels.tools.eval_measures` or manual code. |
| Train and test split | `train_test_split()` is part of the library. | Not included. It needs scikit-learn or manual code. |
| `uv add` | `uv add scikit-learn`. Wheels for Python 3.14. Brings NumPy, SciPy, joblib, and threadpoolctl. | `uv add statsmodels` resolves on Python 3.14. Brings NumPy, SciPy, pandas, and patsy. |

## Decision

The team will build the regression models with scikit-learn. It covers the whole workflow the application needs, which is splitting the data, training, getting the formula, evaluating on training and test sets, and predicting, in one library with one consistent API. statsmodels gives a richer statistical report, but the application does not show p-values or confidence intervals, and it would still need scikit-learn or extra code for the split and the test metrics.

scikit-learn also lets the project reuse the same workflow and structure for simple and multiple regression. A simple regression is a `LinearRegression` trained on a single column, so the functions that prepare the data, train the model, write its formula, and evaluate it are identical in both modes. Only the list of input columns changes. This keeps the code smaller and means that every feature added to the application, such as saving a model or showing its description, works for both modes at once. Choosing scikit-learn now also leaves the door open to other models with the same API, and to model persistence with `joblib`, which it already depends on.

The project adds it with `uv add scikit-learn`. statsmodels stays out of `pyproject.toml`.

## Proof of concept

`examples/linear_regression2.py` trains and evaluates two models on the California housing dataset in `data/housing(1).csv`, with `median_house_value` as the output:

- A simple regression with `median_income`.
- A multiple regression with `median_income`, `housing_median_age`, and `total_rooms`.

Each experiment is a dictionary that maps the input columns to the values of a new prediction, and both go through the same steps:

1. Load the file with the data import module (`get_reader(path).read()`).
2. Select the input and output columns and split them into training (80%) and test (20%) sets with `train_test_split()` and a fixed `random_state`, so the results can be repeated.
3. Train a `LinearRegression` on the training set.
4. Write the formula from `coef_`, `feature_names_in_`, and `intercept_`.
5. Compute the mean squared error (MSE) and R² on the training and test sets.
6. Predict the output for a new input.

Run it with `uv run python examples/linear_regression2.py`. `src/sustentable` does not import it.

### Results

```
median_house_value = 41933.8494 * median_income + 44459.7292

median_house_value = 42653.3268 * median_income + 1975.7439 * housing_median_age + 3.8510 * total_rooms + -25029.5119
```

| Model | Train MSE | Train R² | Test MSE | Test R² | Prediction |
| --- | --- | --- | --- | --- | --- |
| Simple | 6,991,447,170 | 0.4770 | 7,091,157,771 | 0.4589 | 254,129 for `median_income = 5.0` |
| Multiple | 6,453,767,350 | 0.5172 | 6,563,178,660 | 0.4992 | 255,212 for `median_income = 5.0`, `housing_median_age = 30.0`, `total_rooms = 2000.0` |

### Interpretation

- **Coefficients.** Each coefficient is the change in the output when its input grows by one unit and the other inputs stay the same. In the simple model, one more unit of `median_income` adds about 41,934 to `median_house_value`. In the multiple model, each extra year of `housing_median_age` adds about 1,976, and each extra room about 3.85. The coefficients do not say which input matters most, because the inputs use very different scales.
- **Intercept.** The predicted output when every input is zero. It fixes the line in place, but it has no practical meaning here: no district has zero income and zero rooms, which is why the multiple model can have a negative intercept.
- **MSE.** The average of the squared prediction errors, in squared output units, so it is large for house values. Its square root is easier to read: about 84,200 for the simple model and 81,000 for the multiple model on the test set.
- **R².** The share of the variation in the output that the model explains, from 0 to 1 on the training set (it can be negative on new data). Income alone explains about 46% of the house value on the test set, and the two extra inputs bring it to about 50%.
- **Training vs test.** The test metrics are only slightly worse than the training ones, so the models are not overfitting: they generalise to data they did not see. The test set is the one that estimates how the model will behave with new data.

### Difficulties

- `total_bedrooms` has 207 empty values, and `LinearRegression` fails when an input has missing values. The prototype leaves that column out. The application will need a preprocessing step (story #26) to drop or fill missing values before training.
- `ocean_proximity` is a text column and cannot be used as an input without encoding it first, so it was also left out.
