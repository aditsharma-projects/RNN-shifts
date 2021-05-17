import pandas as pd
import tensorflow as tf
import numpy as np
from flaml import AutoML

print("====================")

# %%
# CONFIGURATION

ROWS = 10 ** 4

TRAIN_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/training_set.csv"
VAL_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/validation_set.csv"
TEST_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/testing_set.csv"

types = {
    # "prov_id": np.int32,
    # "employee_id": np.int32,
    "job_title": np.int32,
    "pay_type": np.int32,
    "hours": np.double,
    "day_of_week": np.int32,
}

for i in range(1, 31):
    types[f"hours_l{i}"] = np.double
    types[f"employees_l{i}"] = np.int32

for i in range(7):
    types[f"week_perc{i}"] = np.double

cats = ['job_title', 'prov_id', 'pay_type', 'day_of_week']

# %%
# LOAD DATA
# train = pd.read_csv(TRAIN_FILE, parse_dates=["date"], nrows=ROWS)
# val = pd.read_csv(VAL_FILE, parse_dates=["date"], nrows=ROWS //2)
# test = pd.read_csv(TEST_FILE, parse_dates=["date"], nrows=ROWS //2)
train = pd.read_csv(TRAIN_FILE, parse_dates=["date"]) #, nrows=ROWS)
val = pd.read_csv(VAL_FILE, parse_dates=["date"]) #, nrows=ROWS //2)
test = pd.read_csv(TEST_FILE, parse_dates=["date"]) #, nrows=ROWS //1000)

for df in [train, val, test]:
    df.dropna(inplace=True) # Drop all rows with N/A values
    for col, t in types.items(): # Cast remaining rows to appropriate type
        df[col] = df[col].astype(t)

train_inputs, train_labels = train.drop(['employee_id','date','hours', 'week'], axis=1).filter(types.keys()), train.filter(['hours'])
val_inputs, val_labels = val.drop(['employee_id','date','hours', 'week'], axis=1).filter(types.keys()), val.filter(['hours'])
test_inputs, test_labels = test.drop(['employee_id','date','hours', 'week'], axis=1).filter(types.keys()), test.filter(['hours'])

automl = AutoML()
automl.fit(train_inputs, train_labels['hours'], X_val = val_inputs, y_val=val_labels['hours'], task="regression", metric="mse")

print("automl:")
print(automl)

print("automl.best_loss:")
print(automl.best_loss)

print("manually calculated MSE:")
mse =  sum((automl.predict(test_inputs) - test_labels['hours']) ** 2) / len(test_labels)
print(mse)