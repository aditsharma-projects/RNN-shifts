import tensorflow as tf

LAGGED_DAYS=60
embedded_idx = 0
recurrence_start_idx = 1
mask_start_idx = LAGGED_DAYS + 1
descriptors_start_idx = 2 * LAGGED_DAYS + 1

#RNN class, defines attributes of a model
class RNN(tf.keras.Model):
    #define all components of model
    def __init__(self,recurrence_length,lstm_units,dense_shape,embed_dim,vocab_size):
        super(RNN, self).__init__()
        self.embed_dim = embed_dim
        if embed_dim == 0:
            embed_dim += 1
        self.embeddings = tf.keras.layers.Embedding(vocab_size,embed_dim) 
        self.recurrence_length = int(recurrence_length)
        self.lstm = tf.keras.layers.LSTM(lstm_units)
        self.dense_layers = []
        for width in dense_shape:
            if width == 1:
                break
            self.dense_layers.append(tf.keras.layers.Dense(width,activation=tf.nn.relu))
            self.dense_layers.append(tf.keras.layers.BatchNormalization())
        self.out = tf.keras.layers.Dense(1)
       
    
    #define model architecture
    def call(self, inputs, training=False):
        series = tf.reverse(inputs[:,recurrence_start_idx:recurrence_start_idx+self.recurrence_length],[1])
        mask = tf.reverse(inputs[:,mask_start_idx:mask_start_idx+self.recurrence_length],[1])
        
        time_series = tf.concat([tf.expand_dims(series,2),tf.expand_dims(mask,2)],2)
        additional_inputs = inputs[:,descriptors_start_idx:]
        
        x = self.lstm(time_series)
        if self.embed_dim != 0:    
            embedding_vectors = self.embeddings(inputs[:,embedded_idx])
            x = tf.concat([x,additional_inputs,embedding_vectors],1)
        else:
            x = tf.concat([x,additional_inputs],1)
        for layer in self.dense_layers:
            x = layer(x)
        
        return self.out(x)
