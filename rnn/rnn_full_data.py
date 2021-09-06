import pandas as pd
import tensorflow as tf
import numpy as np
from models import RNN
import os.path
import json

SET = "B"
EXPERIMENTAL = False

LAGGED_DAYS = 60
BUFFER_SIZE = 10000
BATCH_SIZE = 256


if SET == "A":
  steps = 555919099/BATCH_SIZE
  PATH = '/export/storage_adgandhi/PBJhours_ML/tf_data/full_header.csv'
elif SET == "B":
  steps = 557000804/BATCH_SIZE
  PATH = '/export/storage_adgandhi/PBJhours_ML/tf_data/fullB.csv'


LOG_PATH = '/users/facsupport/asharma/RNN-shifts/output/rnn_autotuning_history.csv'
checkpoint_path = "./"+SET+"_checkpoints/cp-{epoch:04d}.ckpt"
if EXPERIMENTAL:
  checkpoint_path = "./"+SET+"_experimental/cp-{epoch:04d}.ckpt"
CHECKPOINT_DIR = os.path.dirname(checkpoint_path)

logs = pd.read_csv(LOG_PATH)
logs = logs.loc[logs['Embedding Dimension']==0]
LSTM_UNITS = logs.iloc[0]["LSTM Units"]
SHAPE = json.loads(logs.iloc[0]["FF model shape"])
if EXPERIMENTAL:
  LSTM_UNITS = 128
  SHAPE = [8,1]


dataset = tf.data.experimental.make_csv_dataset(
    PATH,
    batch_size=BATCH_SIZE,
    label_name='hours'
)

def pack(features, label):
  lst = list(features.values())
  return tf.stack([tf.cast(i,tf.float32) for i in lst], axis=-1), label

dataset = dataset.map(pack)

INITIAL_LEARNING_RATE = 0.000005
def decay(epoch):
  if epoch < 3:
    return INITIAL_LEARNING_RATE
  elif epoch >= 3 and epoch < 7:
    return INITIAL_LEARNING_RATE/10
  else:
    return INITIAL_LEARNING_RATE/100

model = RNN(LAGGED_DAYS,LSTM_UNITS,SHAPE,0,1)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()])
if os.path.isdir(CHECKPOINT_DIR):
    model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_DIR))
    print(f"SUCCESSFULLY RESTORED TRAINING CHECKPOINT FOR MODEL {SET}")
callbacks = [tf.keras.callbacks.LearningRateScheduler(decay),
            tf.keras.callbacks.ModelCheckpoint(
                            filepath=checkpoint_path,
                            save_weights_only=True,
                            save_freq = 1000,
            )
]
history = model.fit(dataset, epochs=10, callbacks=callbacks,steps_per_epoch=steps)

