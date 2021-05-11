import pandas as pd
import tensorflow as tf
import numpy as np

# %%
# CONFIGURATION

ROWS = 10 ** 5

TRAIN_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/training_set.csv"
VAL_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/validation_set.csv"
TEST_FILE = "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/testing_set.csv"

types = {
    "prov_id": np.int32,
    "employee_id": np.int32,
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
train = pd.read_csv(TRAIN_FILE, parse_dates=["date"], nrows=ROWS)
val = pd.read_csv(VAL_FILE, parse_dates=["date"], nrows=ROWS //2)
test = pd.read_csv(TEST_FILE, parse_dates=["date"], nrows=ROWS //1000)

stds = {}

for df in [train, val, test]:
    df.dropna(inplace=True) # Drop all rows with N/A values
    for col, t in types.items(): # Cast remaining rows to appropriate type
        df[col] = df[col].astype(t)

for col_name in ["hours"] + [f"hours_l{i}" for i in range(1, 31)]:
    mean = train[col_name].mean()
    std = train[col_name].std()
    stds[col_name] = std
    for df in [train, val, test]:
        df[col_name] = (df[col_name] - mean) / std

# todo - choose columns based on class constants
train_inputs, train_labels = train.drop(['employee_id','date','hours', 'week'], axis=1).filter(types.keys()), train.filter(['hours'])
val_inputs, val_labels = val.drop(['employee_id','date','hours', 'week'], axis=1).filter(types.keys()), val.filter(['hours'])
test_inputs, test_labels = test.drop(['employee_id','date','hours', 'week'], axis=1).filter(types.keys()), test.filter(['hours'])

# %%
# DEFINE MODELS

dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=64),
    tf.keras.layers.Dense(units=1),
])

# %%
# TRAIN MODELS

print('\nTraining dense model.\n')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=2,
                                                mode='min')
dense.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(clipvalue=0.5),
            metrics=[tf.metrics.MeanAbsoluteError()])

history = dense.fit(train_inputs, train_labels, epochs=20,
            validation_data=(val_inputs, val_labels),
            callbacks=[early_stopping],
            verbose=1)

print('\nTraining LSTM.\n')
lstm.compile(loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(clipvalue=0.5),
            metrics=[tf.metrics.MeanAbsoluteError()])

def lstm_reshape(df):
    df = df.to_numpy()
    return df.reshape((df.shape[0], 1, df.shape[1]))

history = lstm.fit(lstm_reshape(train_inputs), lstm_reshape(train_labels), epochs=20,
            validation_data=(lstm_reshape(val_inputs), lstm_reshape(val_labels)),
            callbacks=[early_stopping],
            verbose=1)

print('\nHours STD: ' + str(stds['hours']))