# LightGBM Shift Predictor

## Environment Setup

### Create virtual environment (recommended)
```sh
pyenv install 3.8.6
pyenv virtualenv 3.8.6 lightgbm
pyenv activate lightgbm
```

### Install packages
```sh
pip install -r requirements.txt
```

# `lgb_tune.py` Overview
`lgb_tune.py` can find optimal hyperparameters and optionally generate predictions.

To only find hyperparameters:
1. Set up environment.
2. Configure "General Configuration" and "Tuning Configuration" described below. Set `MAKE_PREDICTIONS` to `False`.
3. Run `lgb_tune.py`. Outputs: `HISTORY_FILE` and `FIGURES_FOLDER`.

To find hyperparmeters and generate predictions:
1. Set up environment.
2. Configure all globals described below. Set `MAKE_PREDICTIONS` to `True`.
3. Run `lgb_tune.py`. Outputs: `HISTORY_FILE`, `FIGURES_FOLDER`, `FINAL_HISTORY_FILE`, `FINAL_FIGURES_FOLDER`, `TRAIN_PREDICTIONS_CSV`, and `VAL_PREDICTIONS_CSV`.

# `lgb_predict.py` Overview

`lgb_predict.py` is a lightweight isolation of the prediction portion of `lgb_tune.py`.

1. Use `lgb_tune.py` to find optimal hyperparameters.
2. Set `PARAMS` to desired hyperparameters.
3. Set configuration variables. These variables are a subset of those in `lgb_tune.py` and described below.
4. Run `lgb_predict.py`. Outputs: `FINAL_HISTORY_FILE`, `FINAL_FIGURES_FOLDER`, `TRAIN_PREDICTIONS_CSV`, and `VAL_PREDICTIONS_CSV`.



# `lgb_tune.py` Configuration

## General Configuration

`THROW_ON_WARNING`: Whether to exit immediately if a configuration problem is detected. (Recommended: `True`)

`COLUMNS`: Dict. Each key is a column to pull from data files, and the value is the corresponding type. Each column is cased with `pandas.Series.astype`. Additionally, columns of type `"category"` are flagged for LightGBM.

`FILL_NA`: Value to fill NaNs with

`EARLY_STOPPING_ROUNDS`: After how many rounds of no improvement in validation loss to stop training and revert back to the best iteration.

## Tuning Configuration
`TRAIN_FILE`, `VAL_FILE`: Paths to train/validation data files to read from for tuning

`ROWS`: Number of rows to truncate to. Unless debugging, should always be set to `None` so full data files are used.

`HISTORY_FILE`: Path to model tuning history file to write to

`FIGURES_FOLDER`: Folder in which to put feature importance plots generated while tuning

`PARAM_AXES`: Dict. Each key is a LGBM parameter to tune, and each value is list of options to pick from for that parameter. When tuning:
1. The params will be initialized to `INIT_PARAMS`
2. For each param axis, each value will be tried with the current params, and the best value (based on mse validation loss) will be kept.

`INIT_PARAMS`: Parameters for LightGBM training. Those also defined in PARAM_AXES above will be overriden as the model is tuned.

## Prediction Configuration
`MAKE_PREDICTIONS`: Whether to use the best hyperparameters found from tuning to train a model and generate predictions on another set of data fiels. Typically, the purpose of this is to tune on a subsample to save time, then make predictions on the full dataset.

`FINAL_TRAIN_FILE`, `FINAL_VAL_FILE`: Paths to train/validation data files to read from for final training and prediction.

`FINAL_ROWS`: Number of rows to truncate final train/val sets to. Unless debugging, should always be set to None so full data files are used.

`TRAIN_PREDICTIONS_CSV`: Where to output predictions for `FINAL_TRAIN_FILE`

`VAL_PREDICTIONS_CSV`: Where to output predictions for `FINAL_VAL_FILE`

`FINAL_HISTORY_FILE`: Path to final training log CSV

`FINAL_FIGURES_FOLDER`: Folder in which to put feature importance plots generated during final training
