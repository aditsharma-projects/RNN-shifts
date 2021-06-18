import pandas as pd
import numpy as np
import lightgbm as lgb
import datetime
import getpass
import os
import sys

# %%
#### CONFIGURATION ####

FINAL_ROWS = None

THROW_ON_WARNING = False

# Paths for final training/eval on full dataset
FINAL_TRAIN_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/training_set_80perc.csv"
FINAL_VAL_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/testing_set_20perc.csv"
FINAL_HISTORY_FILE = "/users/facsupport/rtjoa/final.csv"
FINAL_FIGURES_FOLDER = "figures_final" # Avoid using ".." in path
TRAIN_PREDICTIONS_CSV = "train_80perc_predictions.csv"
VAL_PREDICTIONS_CSV = "val_80perc_predictions.csv"

# Value to fill NA values with
FILL_NA = None

# Columns to pull from data files and their corresponding types
COLUMNS = {
    "job_title": 'category',
    "pay_type": 'category',
    "hours": np.double,
    "day_of_week": 'category',
    "perc_hours_today_before": np.double,
    "perc_hours_yesterday_before": np.double,
    "perc_hours_tomorrow_before": np.double,
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
}

# Parameters for LightGBM training. Those also defined in PARAM_AXES above will
# be overriden as the model is tuned.
PARAMS = {
    # Tuned params
    "num_leaves": 4000,
    'learning_rate': 0.01,
    'min_data_in_leaf': 20,
    # Other params
    'metric': 'mse',
    'verbose': -1,
    'force_row_wise': True,
    'deterministic': True,
    'num_threads': 25,
}

# After how many rounds of no improvement in validation loss to stop training,
# then revert back to the best iteration.
EARLY_STOPPING_ROUNDS = 5

# %%
#### UTILITY ####

user = getpass.getuser()

# Force print flushing
def print(*objects, sep=' ', end='\n', file=sys.stdout, flush=True):
    __builtins__.print(*objects, sep=sep, end=end, file=file, flush=flush)

# Print string s prominently
def print_header(s):
    print("=" * 60)
    print("    " + s)
    print("=" * 60)
    print()

# Print a key and value, formatted consistently
def print_kv(key, value):
    print(key.ljust(20), value)
    print()

# Print the name and contents of a dictionary
def print_dict(name, dict_obj):
    print(name)
    for k, v in dict_obj.items():
        print(("    " + k).ljust(20), v)
    print()

# Append dict to CSV, creating new columns as needed
def log_model_info(model_info, path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"History csv not found at {path}. Creating new file.")
        df = pd.DataFrame()

    new_df = pd.DataFrame({k: [v] for k, v in model_info.items()})
    df = pd.concat([df, new_df], axis=0)
    df.to_csv(path, index=False)

# Timer class to record how long something takes
# >>> timer_foo = Timer("Beginning foo...")
# >>> foo()
# >>> timer_foo.done()
# Can access timer_foo.start_time, timer_foo.duration, timer_foo.end_time
class Timer(object):
    def __init__(self, msg, one_line=True):
        if one_line:
            print(msg, '', end='')
        else:
            print(msg)
        self.one_line = one_line
        self.msg = msg
        self.start_time = datetime.datetime.now()
        self.is_done = False
    
    def done(self, msg=None):
        if self.is_done:
            print("Error: time stopped but not running")
            return
        self.is_done = True

        self.end_time = datetime.datetime.now()
        self.duration = self.end_time - self.start_time
        if msg is None:
            if self.one_line:
                msg = "done in %s."
            else:
                msg = "Done in %s."
        print(msg % self.duration)

# %%
#### CONFIGURATION VALIDATION ####

def warn(msg):
    if THROW_ON_WARNING:
        raise ValueError(f"{msg}\nTo ignore this error, set THROW_ON_WARNING=False.")
    else:
        print(f"WARNING: {msg}\n")

if not os.path.isdir(FINAL_FIGURES_FOLDER):
    try:
        os.makedirs(FINAL_FIGURES_FOLDER)
    except Exception as e:
        warn(
            "WARNING: Could not create FINAL_FIGURES_FOLDER directory."
            + f" Feature importance plot for final eval will not be saved.\n{e}"
        )

if FINAL_ROWS is not None:
    warn(f"Truncating to {FINAL_ROWS} rows. Set FINAL_ROWS=None for meaningful results.")

# %%
#### PRINT CONFIGURATION ####

print(datetime.datetime.now())
print()

print_header("Configuration")

print_kv("FINAL_TRAIN_FILE", FINAL_TRAIN_FILE)
print_kv("FINAL_VAL_FILE", FINAL_VAL_FILE)
print_kv("FINAL_FIGURES_FOLDER", FINAL_FIGURES_FOLDER)
print_dict("COLUMNS", COLUMNS)
print_dict("PARAMS", PARAMS)

CATEGORICALS = [col for col, col_type in COLUMNS.items() if col_type == "category"]
print(f"Set CATEGORICALS to {CATEGORICALS}")
print()

# %%
#### LOAD FINAL DATA ####

cols_list = list(COLUMNS.keys())

print_header("Final data setup")

timer_load = Timer("Loading...")
if FINAL_ROWS is None:
    train = pd.read_csv(FINAL_TRAIN_FILE, usecols=cols_list)
    val = pd.read_csv(FINAL_VAL_FILE, usecols=cols_list)
else:
    train = pd.read_csv(FINAL_TRAIN_FILE, usecols=cols_list, nrows=FINAL_ROWS)
    val = pd.read_csv(FINAL_VAL_FILE, usecols=cols_list, nrows=FINAL_ROWS)
timer_load.done()

timer_cast = Timer("Casting values...")
for df in [train, val]:
    for col, t in COLUMNS.items(): # Cast columns to appropriate type
        if FILL_NA is not None:
            df[col].fillna(FILL_NA)
        df[col] = df[col].astype(t)
timer_cast.done()

timer_split = Timer("Splitting into inputs and labels...")
train_inputs, train_labels = train.drop(['hours'], axis=1).filter(COLUMNS.keys()), train.filter(['hours'])
val_inputs, val_labels = val.drop(['hours'], axis=1).filter(COLUMNS.keys()), val.filter(['hours'])
timer_split.done()
print()

# %%
#### FINAL EVAL ####

params = PARAMS

# Create datasets
train_data = lgb.Dataset(train_inputs, label=train_labels, categorical_feature=CATEGORICALS)
val_data = lgb.Dataset(val_inputs, label=val_labels, categorical_feature=CATEGORICALS)

# Short descriptor, made up of params being tuned
desc = ", ".join(f"{k}:{v}" for k, v in sorted(params.items()))

# Train, if we have not already tested this config
timer_train = Timer(f"Final training ({desc})...")        
evals_result = {}
bst = lgb.train(params, train_data,
    valid_sets=[val_data],
    valid_names=['val_data'],
    evals_result=evals_result,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    num_boost_round=10**5,
    categorical_feature=CATEGORICALS,
    verbose_eval=False,
)
train_predict = bst.predict(train_inputs)
val_predict = bst.predict(val_inputs)
training_loss = ((train_predict - train_labels['hours'])**2).mean()
val_loss = ((val_predict - val_labels['hours'])**2).mean()
num_trees = bst.num_trees()
chosen_iter, total_iters = bst.current_iteration(), len(evals_result['val_data']['l2'])
timer_train.done(f"chose iter {chosen_iter}/{total_iters} ({num_trees} trees) in %s.")
print(f"Training loss: {training_loss:.5f} | Val loss: {val_loss:.5f}")
print()

# Save model info and performance
model_info = params.copy()
model_info['training_loss'] = training_loss
model_info['val_loss'] = val_loss
model_info['num_trees'] = num_trees
model_info['time_start'] = timer_train.start_time
model_info['time_duration'] = timer_train.duration
model_info['iterations'] = bst.current_iteration()
model_info['final_rows'] = FINAL_ROWS
model_info['fill_na'] = FILL_NA
model_info['columns_used'] = ','.join(COLUMNS.keys())
model_info['user'] = user
log_model_info(model_info, FINAL_HISTORY_FILE)

# Save feature importance figures
if os.path.isdir(FINAL_FIGURES_FOLDER):
    fig_path = os.path.join(FINAL_FIGURES_FOLDER, f"{desc}.png")
    lgb.plot_importance(bst).get_figure().savefig(fig_path)

timer_save_train_pred = Timer("Saving training predictions...")
train_predict_df = train.filter(['hours'])
train_predict_df.insert(1, "predicted_hours", train_predict, True)
train_predict_df.to_csv(TRAIN_PREDICTIONS_CSV)
timer_save_train_pred.done()

timer_save_val_pred = Timer("Saving val predictions...")
val_predict_df = val.filter(['hours'])
val_predict_df.insert(1, "predicted_hours", val_predict, True)
val_predict_df.to_csv(VAL_PREDICTIONS_CSV)
timer_save_val_pred.done()

print()
print("Finished!")