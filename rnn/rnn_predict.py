import pandas as pd
import tensorflow as tf
import numpy as np
from models import RNN
import os.path
import json
import time

SET = "B"
NET = "A"

LAGGED_DAYS = 60
BUFFER_SIZE = 10000
BATCH_SIZE = 2048*8
#BATCH_SIZE = 8


if SET == "A":
  steps = int(555919099/BATCH_SIZE)
  PATH = '/export/storage_adgandhi/PBJhours_ML/tf_data/50A.csv'
elif SET == "B":
  steps = int(557000804/BATCH_SIZE)
  PATH = '/export/storage_adgandhi/PBJhours_ML/tf_data/50B.csv'


LOG_PATH = '/users/facsupport/asharma/RNN-shifts/output/rnn_autotuning_history.csv'
checkpoint_path = "./"+NET+"_checkpoints/cp-{epoch:04d}.ckpt"
CHECKPOINT_DIR = os.path.dirname(checkpoint_path)

logs = pd.read_csv(LOG_PATH)
logs = logs.loc[logs['Embedding Dimension']==0]
LSTM_UNITS = logs.iloc[0]["LSTM Units"]
SHAPE = json.loads(logs.iloc[0]["FF model shape"])


dataset = tf.data.experimental.make_csv_dataset(
    PATH,
    batch_size=BATCH_SIZE,
    label_name='hours'
)

def pack(features, label):
  description = [tf.strings.as_string(features.pop('employee_id',None)),tf.strings.as_string(features['prov_id']),features.pop('date',None),
                 tf.strings.as_string(features.pop('job_title',None)),tf.strings.as_string(features.pop('pay_type',None))]
  lst = list(features.values())
  return tf.stack(description,axis=1), tf.stack([tf.cast(i,tf.float32) for i in lst], axis=-1), label

dataset = dataset.map(pack)

model = RNN(LAGGED_DAYS,LSTM_UNITS,SHAPE,0,1)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()])

model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_DIR))

outList = []
i=0
start_time = time.time()
for descriptions,features,labels in dataset:
    i+=1
    if i%1000 == 10:
        print(f"Completed {i} batches. Last {1000} batches took {time.time()-start_time} seconds")
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
frame.to_csv('/export/storage_adgandhi/PBJhours_ML/Data/Predictions/RNN_60days/'+SET+'pred.csv',index=False)
#np.savetxt('/export/storage_adgandhi/PBJhours_ML/Data/Predictions/'+SET+'pred.csv',final_tensor.numpy(),header="employee_id,prov_id,date,prediction,hours",fmt='%s',delimiter',')