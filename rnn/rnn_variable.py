import pandas as pd
import tensorflow as tf
import numpy as np

INIT_PERIOD = 30

train = pd.read_csv("/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/VariableLengths/train10_sample_sequences.csv").dropna()
train_coords = pd.read_csv("/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/VariableLengths/train10_sample_coords.csv").dropna()
#val = pd.read_csv("/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/VariableLengths/val5_sample_sequences.csv")
#val_coords = pd.read_csv("/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/VariableLengths/val5_sample_coords.csv")
def generate_sequences(offset,dataset):
    if dataset==0:
        frame = train
        coords = train_coords
    elif dataset==1:
        frame = val
        coords = val_coords
    for i in range(len(coords)):
        start = int(coords.iloc[i]['first_index'])
        end = int(coords.iloc[i]['last_index'])
        if end-start <= INIT_PERIOD:
            continue
        if offset==0:
            series = tf.stack([frame[start:end+1]['hours'].astype('float32'),
                            frame[start:end+1]['avg_employees_7days'].astype('float32'),
                            frame[start:end+1]['Lemployees'].astype('float32')],axis=1)
            series = tf.concat([series,tf.one_hot(frame[start:end+1]['day_of_week'].astype('float32'),7)],axis=1)
        elif offset==1:
            series = frame[start:end+1]['hours'].astype('float32')
        if len(series)>1:
            yield series[offset:len(series)-1+offset]

    

train_feautures = tf.data.Dataset.from_generator(
            generate_sequences, args=[0,0], output_types=tf.float32, output_shapes=[None,10]
)
train_labels = tf.data.Dataset.from_generator(
            generate_sequences, args=[1,0], output_types=tf.float32, output_shapes=[None]
)

train_dataset = tf.data.Dataset.zip((train_feautures,train_labels))

train_dataset = train_dataset.apply(
    tf.data.experimental.bucket_by_sequence_length(
        element_length_func=lambda _,elem: tf.shape(elem),
        bucket_boundaries=list(range(1,720)),
        bucket_batch_sizes=[16]*720,
        drop_remainder=False)
)


#RNN class, defines attributes of a model
class RNN(tf.keras.Model):
    #define all components of model
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm = tf.keras.layers.LSTM(64,return_sequences=True,dtype='float32')
        self.dense = tf.keras.layers.Dense(8)
        self.out = tf.keras.layers.Dense(1)
       
    
    #define model architecture
    def call(self, inputs, training=False):
        #time_series = tf.expand_dims(inputs,2)  
        x = self.lstm(inputs)
        x = self.dense(x)
        return self.out(x)

loss_object = tf.keras.losses.MeanSquaredError()
def get_loss(predictions,labels):
    return loss_object(predictions[:,INIT_PERIOD:],labels[:,INIT_PERIOD:])

model = RNN()
model.compile(loss=get_loss,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()])
            
history = model.fit(train_dataset, epochs=10)



