import pandas as pd
import numpy as np
import lightgbm as lgb
import datetime
import getpass
import os

# %%
#### CONFIGURATION ####

# Number of rows to truncate to. Unless debugging, should always be set to None
# so full data files are used.
ROWS = None

# Paths to data files
TRAIN_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/training_set.csv"
VAL_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/crossvalidation_set.csv"
TEST_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/holdout_set.csv"

# Path to CSV for model tuning history
HISTORY_FILE = "/users/facsupport/rtjoa/model_history.csv"

# Path to directory for feature importance plots
FIGURES_FOLDER = "figures"

# Columns to pull from data files and their corresponding types
COLUMNS = {
    "prov_id": "category",
    "job_title": "category",
    "pay_type": "category",
    "hours": np.double,
    "day_of_week": "category",
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
}

# List of columsn to treat as categorical variables
CATEGORICALS = ['prov_id', 'job_title', 'pay_type', 'day_of_week']

# LightGBM parameters to tune, and options to pick from for each
PARAM_AXES = {
    "learning_rate": [0.003, 0.005, 0.01, 0.03],
    "num_leaves": [2000, 4000, 8000],
    "min_data_in_leaf": [1, 10, 20, 50, 100],
}

# Parameters for LightGBM training. Those also defined in PARAM_AXES above will
# be overriden as the model is tuned.
INIT_PARAMS = {
    # Tuned params
    "num_leaves": 4000,
    'learning_rate': 0.1,
    'min_data_in_leaf': 20,
    # Other params
    'metric': 'mse',
    'verbose': -1,
    'force_row_wise': True,
    'deterministic': True,
    'num_threads': 50,
}

# After how many rounds of no improvement in validation loss to stop training,
# then revert back to the best iteration.
EARLY_STOPPING_ROUNDS = 5

# %%
#### UTILITY ####

user = getpass.getuser()

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
        df = pd.read_csv(HISTORY_FILE)
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
#### PRINT CONFIGURATION ####

print(datetime.datetime.now())
print()

print_header("Configuration")

if ROWS is not None:
    print(f"WARNING: Truncating to {ROWS} rows. Set ROWS=None for meaningful results.\n")

if not os.path.isdir(FIGURES_FOLDER):
    print(f"WARNING: FIGURES_FOLDER is not a valid directory. Feature importance plots will not be saved.\n")

print_kv("TRAIN_FILE", TRAIN_FILE)
print_kv("VAL_FILE", VAL_FILE)
print_kv("TEST_FILE", TRAIN_FILE)
print_dict("COLUMNS", COLUMNS)
print_kv("CATEGORICALS", CATEGORICALS)
print_dict("PARAM_AXES", PARAM_AXES)
print_dict("INIT_PARAMS", INIT_PARAMS)

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
    for col, t in COLUMNS.items(): # Cast rows to appropriate type
        df[col] = df[col].astype(t)
    df.dropna(inplace=True) # Drop all rows with N/A values
timer_dropna.done()

timer_split = Timer("Splitting into inputs and labels...")
train_inputs, train_labels = train.drop(['hours'], axis=1).filter(COLUMNS.keys()), train.filter(['hours'])
val_inputs, val_labels = val.drop(['hours'], axis=1).filter(COLUMNS.keys()), val.filter(['hours'])
test_inputs, test_labels = test.drop(['hours'], axis=1).filter(COLUMNS.keys()), test.filter(['hours'])
timer_split.done()
print()

# %%
#### TRAIN AND TUNE ####

params = INIT_PARAMS.copy()

# Tune one param at a time
for name in PARAM_AXES:
    print_header(f"Tuning {name}")
    
    best_value = None
    best_loss = float('inf')

    # For each option given for that parameter, train a model
    for value in PARAM_AXES[name]:
        # Update param as we move along its axis
        params[name] = value

        # Recreate datasets
        train_data = lgb.Dataset(train_inputs, label=train_labels, categorical_feature=CATEGORICALS)
        val_data = lgb.Dataset(val_inputs, label=val_labels, categorical_feature=CATEGORICALS)
        
        # Short descriptor, made up of params being tuned
        desc = ", ".join(f"{k}:{v}" for k, v in params.items() if k in PARAM_AXES)
        
        # Train!
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
        model_info['columns_used'] = ','.join(COLUMNS.keys())
        model_info['user'] = user
        log_model_info(model_info, HISTORY_FILE)

        # Save feature importance figures
        if os.path.isdir(FIGURES_FOLDER):
            fig_path = os.path.join(FIGURES_FOLDER, f"{desc}.png")
            lgb.plot_importance(bst).get_figure().savefig(fig_path)

        # Track ideal value for this param
        if val_loss < best_loss:
            best_value = value
            best_loss = val_loss
            
    print(f"Choosing {best_value} for {name} (val loss: {best_loss}).")
    print()
    params[name] = best_value