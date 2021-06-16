import pandas as pd
import numpy as np
import lightgbm as lgb
import datetime
import getpass
import os
import sys

# %%
#### CONFIGURATION ####

# Number of rows to truncate to. Unless debugging, should always be set to None
# so full data files are used.
ROWS = None

# Paths to tuning data files
TRAIN_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/training_set_30.csv"
VAL_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/crossvalidation_set_30.csv"

# Path to CSV for model tuning history
HISTORY_FILE = "/users/facsupport/rtjoa/lgb_model_30.csv"

# Path to directory for feature importance plots. Avoid using ".." in path
FIGURES_FOLDER = "figures_30"

# Whether to train/eval on full dataset
MAKE_PREDICTIONS = False

# Whether to exit if a configuration problem is detected
THROW_ON_WARNING = True

# Paths for final training/eval on full dataset
FINAL_TRAIN_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/training_set_80perc.csv"
FINAL_VAL_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/testing_set_20perc.csv"
FINAL_HISTORY_FILE = "/users/facsupport/rtjoa/final.csv"
FINAL_FIGURES_FOLDER = "figures_final" # Avoid using ".." in path
TRAIN_PREDICTIONS_CSV = "train_80perc_predictions.csv"
VAL_PREDICTIONS_CSV = "val_80perc_predictions.csv"

# Value to fill NA values with
FILL_NA = None

# Ensure we don't save truncated output to same place
if ROWS is not None:
    HISTORY_FILE += ".temp.csv"
    FIGURES_FOLDER += "_temp"

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

# List of columns to treat as categorical variables.
# Set to None to autogenerate based on COLUMNS.
CATEGORICALS = None

# LightGBM parameters to tune, and options to pick from for each
PARAM_AXES = {
    "num_leaves": [50, 100, 200, 500, 1000, 2000, 4000, 6000],
    "learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    "min_data_in_leaf": [1, 5, 15, 18, 20, 22, 25],
}

# Parameters for LightGBM training. Those also defined in PARAM_AXES above will
# be overriden as the model is tuned.
INIT_PARAMS = {
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

if ROWS is not None:
    warn(f"Truncating to {ROWS} rows. Set ROWS=None for meaningful results.")

if not os.path.isdir(FIGURES_FOLDER):
    try:
        os.makedirs(FIGURES_FOLDER)
    except Exception as e:
        warn(
            "Could not create FIGURES_FOLDER directory."
            + f" Feature importance plots will not be saved.\n{e}"
        )

if not os.path.isdir(FINAL_FIGURES_FOLDER):
    try:
        os.makedirs(FINAL_FIGURES_FOLDER)
    except Exception as e:
        warn(
            "WARNING: Could not create FINAL_FIGURES_FOLDER directory."
            + f" Feature importance plot for final eval will not be saved.\n{e}"
        )

# %%
#### PRINT CONFIGURATION ####

print(datetime.datetime.now())
print()

print_header("Configuration")

print_kv("TRAIN_FILE", TRAIN_FILE)
print_kv("VAL_FILE", VAL_FILE)
print_kv("TEST_FILE", TRAIN_FILE)
print_kv("FIGURES_FOLDER", FIGURES_FOLDER)
print_kv("FINAL_TRAIN_FILE", FINAL_TRAIN_FILE)
print_kv("FINAL_VAL_FILE", FINAL_VAL_FILE)
print_kv("FINAL_FIGURES_FOLDER", FINAL_FIGURES_FOLDER)
print_dict("COLUMNS", COLUMNS)
print_kv("CATEGORICALS", CATEGORICALS)
print_dict("PARAM_AXES", PARAM_AXES)
print_dict("INIT_PARAMS", INIT_PARAMS)

if CATEGORICALS is None:
    CATEGORICALS = [col for col, col_type in COLUMNS.items() if col_type == "category"]
    print(f"Set CATEGORICALS to {CATEGORICALS}")
    print()

# %%
#### LOAD DATA ####

cols_list = list(COLUMNS.keys())

print_header("Data setup")

timer_load = Timer("Loading...")
if ROWS is not None:
    train = pd.read_csv(TRAIN_FILE, nrows=ROWS, usecols=cols_list)
    val = pd.read_csv(VAL_FILE, nrows=ROWS//2, usecols=cols_list)
else:
    train = pd.read_csv(TRAIN_FILE, usecols=cols_list)
    val = pd.read_csv(VAL_FILE, usecols=cols_list)
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
#### TRAIN AND TUNE ####

params = INIT_PARAMS.copy()
best_params = {}

desc_to_val_loss = {}

# Tune one param at a time
for name in PARAM_AXES:
    print_header(f"Tuning {name}")
    
    best_value = None
    best_loss = float('inf') # best loss for tests along this axis

    # For each option given for that parameter, train a model
    for value in PARAM_AXES[name]:
        # Update param as we move along its axis
        params[name] = value

        # Recreate datasets
        train_data = lgb.Dataset(train_inputs, label=train_labels, categorical_feature=CATEGORICALS)
        val_data = lgb.Dataset(val_inputs, label=val_labels, categorical_feature=CATEGORICALS)
        
        # Short descriptor, made up of params being tuned
        desc = ", ".join(f"{k}:{v}" for k, v in sorted(params.items()) if k in PARAM_AXES)
        
        # Train, if we have not already tested this config
        if desc not in desc_to_val_loss:
            timer_train = Timer(f"Training ({desc})...")        
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
            training_loss = ((bst.predict(train_inputs) - train_labels['hours'])**2).mean()
            val_loss = ((bst.predict(val_inputs) - val_labels['hours'])**2).mean()
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
            model_info['truncate_rows'] = ROWS
            model_info['fill_na'] = FILL_NA
            model_info['columns_used'] = ','.join(COLUMNS.keys())
            model_info['user'] = user
            log_model_info(model_info, HISTORY_FILE)

            # Save feature importance figures
            if os.path.isdir(FIGURES_FOLDER):
                fig_path = os.path.join(FIGURES_FOLDER, f"{timer_train.start_time} {desc}.png")
                lgb.plot_importance(bst).get_figure().savefig(fig_path)

            desc_to_val_loss[desc] = val_loss
        else:
            print(f"Already trained this config ({desc})")
        # Track ideal value for this param
        if desc_to_val_loss[desc] < best_loss:
            best_value = value
            best_loss = desc_to_val_loss[desc]
            best_params = params.copy() # Optimal value of this axis is the same as overal optimal value thus far
            
    print(f"Choosing {best_value} for {name} (val loss: {best_loss}).")
    print()
    params[name] = best_value

# %%
#### LOAD FINAL DATA ####

if MAKE_PREDICTIONS:
    print_header("Final data setup")

    timer_load = Timer("Loading...")
    train = pd.read_csv(FINAL_TRAIN_FILE, usecols=cols_list)
    val = pd.read_csv(FINAL_VAL_FILE, usecols=cols_list)
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

    params = best_params

    # Create datasets
    train_data = lgb.Dataset(train_inputs, label=train_labels, categorical_feature=CATEGORICALS)
    val_data = lgb.Dataset(val_inputs, label=val_labels, categorical_feature=CATEGORICALS)

    # Short descriptor, made up of params being tuned
    desc = ", ".join(f"{k}:{v}" for k, v in sorted(params.items()) if k in PARAM_AXES)

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
    model_info['truncate_rows'] = ROWS
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