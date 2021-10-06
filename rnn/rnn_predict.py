import pandas as pd
import tensorflow as tf
import numpy as np
from models import RNN
import os.path
import json
import time

SET = "A"
NET = "B"

LAGGED_DAYS = 60
BUFFER_SIZE = 10000
BATCH_SIZE = 2048*8
#BATCH_SIZE = 8

if SET == NET:
  print("Warning, running predictions on train set")

if SET == "A":
  PATH = '/export/storage_adgandhi/PBJhours_ML/tf_data/50A.csv'
  #steps = int(len(pd.read_csv(PATH,usecols=['hours']))/BATCH_SIZE)
  steps = 10000000
elif SET == "B":
  PATH = '/export/storage_adgandhi/PBJhours_ML/tf_data/50B.csv'
  steps = int(len(pd.read_csv(PATH,usecols=['hours']))/BATCH_SIZE)


LOG_PATH = '/export/storage_adgandhi/PBJhours_ML/Data/Logs/rnn_autotuning_history.csv'
checkpoint_path = "./"+NET+"_checkpoints/cp-{epoch:04d}.ckpt"
CHECKPOINT_DIR = os.path.dirname(checkpoint_path)

logs = pd.read_csv(LOG_PATH)
logs = logs.loc[logs['Embedding Dimension']==0]
LSTM_UNITS = logs.iloc[0]["LSTM Units"]
SHAPE = json.loads(logs.iloc[0]["FF model shape"])

d_types = ['float32']+['string']*5+['float32']*128
dataset = tf.data.experimental.make_csv_dataset(
    PATH,
    batch_size=BATCH_SIZE,
    label_name='hours',
    column_defaults=d_types
)

def pack(features, label):
  #description = [tf.strings.as_string(features.pop('employee_id',None)),tf.strings.as_string(features.pop('prov_id',None)),tf.strings.as_string(features.pop('date',None)),
  #               tf.strings.as_string(features.pop('job',None)),tf.strings.as_string(features.pop('pay_type',None))]
  #description = [tf.strings.as_string(features.pop('employee_id',None)),tf.strings.as_string(features.pop('prov_id',None)),features.pop('date',None),features.pop('job',None),features.pop('pay_type',None)]
  description = [features.pop('employee_id',None),features.pop('prov_id',None),features.pop('date',None),features.pop('job',None),features.pop('pay_type',None)]
  lst = list(features.values())
  return tf.stack(description,axis=1), tf.stack([tf.cast(i,tf.float32) for i in lst], axis=-1), label

dataset = dataset.map(pack)

model = RNN(LAGGED_DAYS,LSTM_UNITS,SHAPE,0,1)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()])

model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_DIR))

outList = []
start_time = time.time()
for i, (descriptions,features,labels) in enumerate(dataset):
    if i%1000 == 10:
        print(f"Completed {i} batches. Last {1000} batches took {time.time()-start_time} seconds. {100*i/steps} % done")
        start_time = time.time()
    predictions = model(features)
    outList.append(tf.concat([descriptions,tf.strings.as_string(predictions),tf.strings.as_string(tf.expand_dims(labels,axis=1))],axis=1))
    if i==steps:
      break



final_tensor = tf.concat(outList,axis=0)
final_tensor = final_tensor.numpy()
frame = pd.DataFrame(final_tensor,columns=['employee_id','prov_id','date','job_title','pay_type','prediction','hours'])
print(frame)


frame[['employee_id','prov_id','date','job_title','pay_type']] = frame[['employee_id','prov_id','date','job_title','pay_type']].astype('|S')
for col in ['employee_id','prov_id','date','job_title','pay_type']:
  frame[col] = frame[col].str.decode('utf-8')

frame[['prediction','hours']] = frame[['prediction','hours']].apply(pd.to_numeric)
frame.to_csv('/export/storage_adgandhi/PBJhours_ML/Data/Predictions/RNN_60days/Latest_Data/'+SET+'pred.csv',index=False)
print(f"MSE: {((frame.prediction-frame.hours)**2).mean()}")