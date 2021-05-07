# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Perform imports relative to parent directory
import sys
sys.path.insert(0, '..')
from preprocessing.window_generator import WindowGenerator
from preprocessing.data_prep import initial_preprocess
# %%
# ========================================
#                 SETTINGS
# ========================================

# Data input file
RAW_DATA_PATH = '/users/facsupport/asharma/Data/pbj_full.csv'
PREPROCESSED_DIR = '/users/facsupport/asharma/Data/prep/'
ROWS_TO_READ = 1000

# Weights to split data set
TRAINING_WEIGHT = 0.7
VALIDATION_WEIGHT = 0.2
TEST_WEIGHT = 0.1

# For model training
MAX_EPOCHS = 20
VERBOSE_TRAINING = 1

# Window parameters
INPUT_WIDTH = 7
GAP_WIDTH = 0
LABEL_WIDTH = 1

# %%
# ========================================
#               PREPROCESSING
# ========================================

# Preprocess data
df, info = initial_preprocess(
    RAW_DATA_PATH, PREPROCESSED_DIR,
    nrows=ROWS_TO_READ,
    fill_missing_shifts=True,
    normalize=True
)

# Split data into training/validation/test sets
n = len(df)
weights_sum = TRAINING_WEIGHT + VALIDATION_WEIGHT + TEST_WEIGHT
split1 = int(TRAINING_WEIGHT / weights_sum * n)
split2 = int((TRAINING_WEIGHT + VALIDATION_WEIGHT) / weights_sum * n)
train_df = df[:split1]
val_df = df[split1:split2]
test_df = df[split2:]

# Create window generator
window = WindowGenerator(
    train_df, val_df, test_df,
    INPUT_WIDTH, LABEL_WIDTH, GAP_WIDTH,
    label_columns=['hours'])

# %%
# ========================================
#                 MODELS
# ========================================

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(64, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

# %%
# ========================================
#                 TRAINING
# ========================================

def compile_and_fit(model, window, patience=3, verbose=0):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping],
                      verbose=verbose)
    return history

print()
print("Training dense model.")
history = compile_and_fit(dense, window, verbose=VERBOSE_TRAINING)

print()
print("Training LSTM model.")
history = compile_and_fit(lstm_model, window, verbose=VERBOSE_TRAINING)


# %%
# ========================================
#                 EVALUATION
# ========================================

val_performance = {}
performance = {}

print()
print("Evaluating dense model.")
val_performance['Dense'] = dense.evaluate(window.val, verbose=VERBOSE_TRAINING)
performance['Dense'] = dense.evaluate(window.test, verbose=0)

print()
print("Evaluating LSTM model.")
val_performance['LSTM'] = lstm_model.evaluate(window.val, verbose=VERBOSE_TRAINING)
performance['LSTM'] = lstm_model.evaluate(window.test, verbose=0)

print()
print("Overall validation performance:")
for model_name, (loss, mea) in val_performance.items():
    print("%s %.4f loss, %.4f mean abs error (%.4f hours)" % ((model_name + ":").ljust(17), loss, mea, mea * std['hours']) )