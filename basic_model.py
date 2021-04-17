from numpy import genfromtxt
import tensorflow as tf
import numpy as np
data = genfromtxt('/users/facsupport/asharma/Data/batches/31/one.csv',delimiter=',')

train_set = data[0:int(2*len(data)/3)]
test_set = data[int(2*len(data)/3):]

def make_traindata(set):
    examples = set[:,:30]
    labels = set[:,30:31]
    #means = set[:,8:9]
    #stds = set[:,9:10]

    return tf.data.Dataset.from_tensor_slices((examples,labels))

train_dataset = make_traindata(train_set)

BATCH_SIZE = 32

train_dataset = train_dataset.batch(BATCH_SIZE)

tf.keras.backend.set_floatx('float64')

#model = tf.keras.Sequential([
    #tf.keras.layers.LSTM(64),
    #tf.keras.layers.Dense(32),
    #tf.keras.layers.Dense(1)
#])

loss_object = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(2e-6, beta_1=0.5)


def testModel(testSet, model):
    for entry in testSet:
        seq = entry[:30]
        seq = np.array([[seq]])
        seq = np.transpose(seq, axes=[0, 2, 1])

        label = entry[30:31]
        mean = entry[31:32]
        std = entry[32:33]

        prediction = model(seq)
        prediction = (prediction*std)+mean
        label = (label*std)+mean
        print("Prediction: "+str(prediction)+" True Value: "+str(label))



def train_epoch(model):
 i = 0
 nanCount = 0
 for example, label in train_dataset:
     example = np.expand_dims(example,2)
     label = np.expand_dims(label,2)
     with tf.GradientTape() as tape:
         prediction = model(example)
         loss = loss_object(prediction,label)*BATCH_SIZE
         if loss.dtype == tf.int64:
             continue

         
         gradients = tape.gradient(loss,model.trainable_variables)
         if np.isnan(gradients[0].numpy())[0][0]:
            nanCount+=1 
            continue
         i += 1
         if i%1000 == 1:    
            print("LOSS: "+str(loss))
         optimizer.apply_gradients(zip(gradients,model.trainable_variables))
 print("NAN Count: "+str(nanCount))
print()
