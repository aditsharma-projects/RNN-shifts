from numpy import genfromtxt
import tensorflow as tf
import numpy as np
data = genfromtxt('/users/facsupport/asharma/Data/batches/8/one.csv',delimiter=',')

train_set = data[0:int(2*len(data)/3)]
test_set = data[int(2*len(data)/3):]

def make_traindata(set):
    examples = set[:,:7]
    labels = set[:,7:8]
    means = set[:,8:9]
    stds = set[:,9:10]

    return tf.data.Dataset.from_tensor_slices((examples,labels))

train_dataset = make_traindata(train_set)

BATCH_SIZE = 64

train_dataset = train_dataset.batch(BATCH_SIZE)

tf.keras.backend.set_floatx('float64')

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(1)
])

loss_object = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(2e-6, beta_1=0.5)

for example, label in train_dataset:
    example = np.expand_dims(example,2)
    label = np.expand_dims(label,2)
    with tf.GradientTape() as tape:
        prediction = model(example)
        loss = loss_object(prediction,label)*BATCH_SIZE
        if loss.dtype == tf.int64:
            continue
        print("LOSS: "+str(loss))
        gradients = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))


def testModel(testSet):
    for entry in testSet:
        seq = entry[:7]
        seq = np.array([[seq]])
        seq = np.transpose(seq, axes=[0, 2, 1])

        label = entry[7:8]
        mean = entry[8:9]
        std = entry[9:10]
        
        prediction = model(seq)
        prediction = (prediction*std)+mean
        label = (label*std)+mean
        print("Prediction: "+str(prediction)+" True Value: "+str(label))

        
