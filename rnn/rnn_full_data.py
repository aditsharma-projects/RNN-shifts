import pandas as pd
import tensorflow as tf
import numpy as np
from models import RNN, RNN_Conditioned
import os.path
import json

SET = "A"

LAGGED_DAYS = 60
BUFFER_SIZE = 10000
BATCH_SIZE = 128


if SET == "A":
  PATH = '/export/storage_adgandhi/PBJhours_ML/tf_data/50A.csv'
  steps = len(pd.read_csv(PATH,usecols=['hours']))/BATCH_SIZE
elif SET == "B":
  PATH = '/export/storage_adgandhi/PBJhours_ML/tf_data/50B.csv'
  steps = len(pd.read_csv(PATH,usecols=['hours']))/BATCH_SIZE


LOG_PATH = '/export/storage_adgandhi/PBJhours_ML/Data/Logs/rnn_autotuning_history.csv'
checkpoint_path = "./"+SET+"_checkpoints/cp-{epoch:04d}.ckpt"
CHECKPOINT_DIR = os.path.dirname(checkpoint_path)

logs = pd.read_csv(LOG_PATH)
logs = logs.loc[logs['Embedding Dimension']==0]
LSTM_UNITS = logs.iloc[0]["LSTM Units"]
SHAPE = json.loads(logs.iloc[0]["FF model shape"])
LSTM_TYPE = logs.iloc[0]["LSTM type"]


include_fields = ['hours','avg_employees_7days','0','1','2','3','4','5','6']
for i in range(1,LAGGED_DAYS+1):
    include_fields.insert(i,f"L{i}_hours")
for i in range(1,LAGGED_DAYS+1):
    include_fields.insert(i+LAGGED_DAYS,f"mask_{i}")

dataset = tf.data.experimental.make_csv_dataset(
PATH,
batch_size=BATCH_SIZE,
label_name='hours',
select_columns=include_fields
)

def pack(features, label):
  lst = list(features.values())
  return tf.stack([tf.cast(i,tf.float32) for i in lst], axis=-1), label

dataset = dataset.map(pack)

INITIAL_LEARNING_RATE = 0.0001
def decay(epoch):
  if epoch < 2:
    return INITIAL_LEARNING_RATE
  elif epoch < 5:
    return INITIAL_LEARNING_RATE/2
  elif epoch < 8:
    return INITIAL_LEARNING_RATE/10
  elif epoch < 11:
    return INITIAL_LEARNING_RATE/100
  else:
    return INITIAL_LEARNING_RATE/200

#Bug???
if LSTM_TYPE == "Conditioned":
  print("CONDITIONED... SUCCESS!")
  model = RNN_Conditioned(LAGGED_DAYS,LSTM_UNITS,SHAPE,0,1)
else: model = RNN(LAGGED_DAYS,LSTM_UNITS,SHAPE,0,1)

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

