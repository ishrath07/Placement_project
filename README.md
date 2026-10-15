# Placement Classification

A small machine-learning project that explores whether a student is placed based on their CGPA and IQ. The workflow is documented in the Jupyter notebook and uses logistic regression for binary classification.

## Project files

- `end_to_end_ml_pipeline.ipynb` — data exploration, preprocessing, plotting, model training, and evaluation.
- `modified_placement_data.csv` — dataset with `cgpa`, `iq`, and `placement` columns.
- `placement_model.pkl` — serialized logistic regression model.
- `requirements.txt` — pinned Python package dependencies.

## Dataset

The dataset has two input features:

- `cgpa` — CGPA value; the source data uses a comma as the decimal separator (for example, `"6,8"`).
- `iq` — IQ value.

The target is `placement`, with `Yes` and `No` categories. In the notebook, CGPA is converted to a numeric value, missing CGPA values are filled with the median, and the target is mapped to `1` (`Yes`) and `0` (`No`).

## Setup

Use Python 3 and create/activate a virtual environment if desired. From the project directory, install the declared dependencies:

```powershell
python -m pip install -r requirements.txt
```

The notebook also uses `mlxtend` to plot classifier decision regions. It is not currently listed in `requirements.txt`, so install it separately:

```powershell
python -m pip install mlxtend
```

## Run the notebook

Start Jupyter:

```powershell
python -m jupyter notebook
```

Open `end_to_end_ml_pipeline.ipynb` and run the cells from top to bottom. The notebook expects `modified_placement_data.csv` to be in the current working directory.

## Workflow

1. Load the CSV with pandas and inspect the data.
2. Convert CGPA strings from comma decimals to numeric values and fill missing CGPA values with the median.
3. Explore the relationship between CGPA, IQ, and placement with a scatter plot.
4. Map the target labels to numeric classes and split the data into training and test sets.
5. Train a logistic regression classifier and evaluate its predictions.
6. Plot decision regions and save the trained model as `placement_model.pkl`.

The notebook uses a 90/10 train/test split with `random_state=42`.

## Notes

- The notebook creates scaled feature arrays, but the current logistic regression training and prediction cells use the original, unscaled `X_train` and `X_test`. To use scaling in the model, fit the scaler on the training data, transform both splits with that scaler, and train/predict using the transformed arrays.
- The decision-region visualization is intended for the two-feature dataset (`cgpa` and `iq`).
- Only load `placement_model.pkl` if you trust its source. Pickle files can execute code when loaded.
