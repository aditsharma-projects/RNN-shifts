import pandas as pd
import tensorflow as tf

frame = pd.read_csv("/export/storage_adgandhi/PBJhours_ML/fake_data/full.csv")
coords = pd.read_csv("/export/storage_adgandhi/PBJhours_ML/fake_data/coords.csv")

def generate_sequences(offset):
    for i in range(len(coords)):
        start = coords.iloc[i]['start']
        end = coords.iloc[i]['end']
        if end-start == 1:
            continue
        series = frame[start:end+1]['hours']
        if len(series)>1:
            yield series[offset:len(series)-1+offset]

    

feautures = tf.data.Dataset.from_generator(
            generate_sequences, args=[0], output_types=tf.float32, output_shapes=[None]
)
labels = tf.data.Dataset.from_generator(
            generate_sequences, args=[1], output_types=tf.float32, output_shapes=[None]
)

dataset = tf.data.Dataset.zip((feautures,labels))

dataset_bucketed = dataset.apply(
    tf.data.experimental.bucket_by_sequence_length(
        element_length_func=lambda elem,_: tf.shape(elem),
        bucket_boundaries=list(range(1,90)),
        bucket_batch_sizes=[64]*90,
        drop_remainder=True)
)


#RNN class, defines attributes of a model
class RNN(tf.keras.Model):
    #define all components of model
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm = tf.keras.layers.LSTM(32,return_sequences=True,dtype='float32')
        self.out = tf.keras.layers.Dense(1)
       
    
    #define model architecture
    def call(self, inputs, training=False):
                
        time_series = tf.expand_dims(inputs,2)  
        x = self.lstm(time_series)
        return self.out(x)

model = RNN()
model.compile(loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()])
            
history = model.fit(dataset_bucketed, epochs=5)



