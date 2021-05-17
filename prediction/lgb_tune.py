import pandas as pd
import numpy as np
import lightgbm as lgb
import datetime
now = datetime.datetime.now

# %%
#### CONFIGURATION ####

ROWS = 10 ** 3

TRAIN_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/training_set.csv"
VAL_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/validation_set.csv"
TEST_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/testing_set.csv"

COLUMNS = {
    "job_title": np.int32,
    "pay_type": np.int32,
    "hours": np.double,
    "day_of_week": np.int32,
    "week_perc0": np.double,
    "week_perc1": np.double,
    "week_perc2": np.double,
    "week_perc3": np.double,
    "week_perc4": np.double,
    "week_perc5": np.double,
    "week_perc6": np.double,
    "hours_l1": np.double,
    "hours_l2": np.double,
    "hours_l3": np.double,
    "hours_l4": np.double,
    "hours_l5": np.double,
    "hours_l6": np.double,
    "hours_l7": np.double,
    "hours_l14": np.double,
    "hours_l21": np.double,
    "hours_l28": np.double,
    "hours_l29": np.double,
}

CATEGORICALS = ['job_title', 'pay_type', 'day_of_week']

PARAM_AXES = {
    "num_leaves": [30, 50, 100, 200, 300, 500, 1000],
}

INIT_PARAM = {
    "num_leaves": 50,
    'learning_rate': 0.1,
    'metric': 'mse',
}

# %%
#### UTILITY ####

def print_header(s):
    print("=" * 60)
    print("    " + s)
    print("=" * 60)
    print()

def print_kv(key, value):
    print(key.ljust(20), value)
    print()

def print_dict(name, dict_obj):
    print(name)
    for k, v in dict_obj.items():
        print(("    " + k).ljust(20), v)
    print()

class Timer(object):
    def __init__(self, msg, one_line=True):
        if one_line:
            print(msg, '', end='')
        else:
            print(msg)
        self.one_line = one_line
        self.msg = msg
        self.start_time = now()
    
    def done(self):
        if self.start_time is None:
            print("Error: time stopped but not running")
            return
        duration = now() - self.start_time
        if self.one_line:
            print(f"Took {duration}.")
        else:
            print(f"Finished \"{self.msg}\" in {duration}.")
        self.start_time = None

# %%
#### PRINT CONFIGURATION ####

print(now())
print()

print_header("Configuration")

if ROWS is not None:
    print(f"WARNING: Truncating to {ROWS} rows. Set ROWS=None for meaningful results.\n")

print_kv("TRAIN_FILE", TRAIN_FILE)
print_kv("VAL_FILE", VAL_FILE)
print_kv("TEST_FILE", TRAIN_FILE)
print_dict("COLUMNS", COLUMNS)
print_kv("CATEGORICALS", CATEGORICALS)
print_dict("PARAM_AXES", PARAM_AXES)


# %%
#### LOAD DATA ####

print_header("Data setup")

timer_load = Timer("Loading...")
if ROWS is not None:
    train = pd.read_csv(TRAIN_FILE, parse_dates=["date"], nrows=ROWS)
    val = pd.read_csv(VAL_FILE, parse_dates=["date"], nrows=ROWS//2)
    test = pd.read_csv(TEST_FILE, parse_dates=["date"], nrows=ROWS//2)
else:
    train = pd.read_csv(TRAIN_FILE, parse_dates=["date"])
    val = pd.read_csv(VAL_FILE, parse_dates=["date"])
    test = pd.read_csv(TEST_FILE, parse_dates=["date"])
timer_load.done()

timer_dropna = Timer("Dropping N/A values...")
for df in [train, val, test]:
    df.dropna(inplace=True) # Drop all rows with N/A values
    for col, t in COLUMNS.items(): # Cast remaining rows to appropriate type
        df[col] = df[col].astype(t)
timer_dropna.done()

timer_split = Timer("Splitting into test/train...")
train_inputs, train_labels = train.drop(['hours'], axis=1).filter(COLUMNS.keys()), train.filter(['hours'])
val_inputs, val_labels = val.drop(['hours'], axis=1).filter(COLUMNS.keys()), val.filter(['hours'])
test_inputs, test_labels = test.drop(['hours'], axis=1).filter(COLUMNS.keys()), test.filter(['hours'])
timer_split.done()

param = INIT_PARAM.copy()
for name in PARAM_AXES:
    for value in PARAM_AXES[name]:
        param[name] = value
        print_dict(param)
        train_data = lgb.Dataset(train_inputs, label=train_labels, categorical_feature=CATEGORICALS)
        val_data = lgb.Dataset(val_inputs, label=val_labels, categorical_feature=CATEGORICALS)
        test_data = lgb.Dataset(test_inputs, label=test_labels, categorical_feature=CATEGORICALS)
        evals_result = {}
        bst = lgb.train(param, train_data, valid_sets=[val_data], evals_result=evals_result, early_stopping_rounds=5)
        # TODO: choose best value and save
        break