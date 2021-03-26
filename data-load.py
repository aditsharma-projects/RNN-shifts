import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from operator import itemgetter
import matplotlib.pyplot as plt

def dateIndex(date):    #indexing years starting at 2000, works for sure as long as records stop before 2100
    year = int(date[2:6]) - 2000
    month = int(date[7:9])
    day = int(date[10:12])

    num = year*365
    num += year/4
    months = [31,28,31,30,31,30,31,31,30,31,30,31]
    for i in range(0,month-1):
        num += months[i]
    num += day

    return int(num)


data = genfromtxt('/Users/asharma/Downloads/pbj_sample.csv',delimiter=',', skip_header = 1, dtype="i8,f8,i8,S10,S30,i8,S12")
print("LOADED DATA")
list = []
for x in range(0,len(data)):
    nextTuple = (data[x][0],data[x][1],data[x][2],dateIndex(str(data[x][3])),data[x][4],data[x][5],data[x][6])
    list.append(nextTuple)

listSorted = sorted(list,key=itemgetter(2))


def gen_train_example(i,j): #listSorted[i] to listSorted[j-1] inclusive are the same employee
    dataPoints = sorted(listSorted[i:j],key=itemgetter(3))
    min = dataPoints[0][3]
    max = dataPoints[len(dataPoints)-1][3]
    output = []
    currIndex = 0
    for ind in range(min,max+1):
        if dataPoints[currIndex][3] == ind:
            output.append(dataPoints[currIndex][1])
            currIndex += 1
        else:
            output.append(0)

    return output

def pad_train_example():        #pad/cap to 730 days
    return

dataList = []

currID = listSorted[0][2]
lastStart = 0
lastEnd = 0
for i in range(0,len(listSorted)):
    lastEnd = i
    if listSorted[i][2] != currID:
        currID = listSorted[i][2]
        sample = gen_train_example(lastStart,lastEnd)
        dataList.append(sample)
        lastStart = i
    #elif i == len(listSorted)-1:
        #sample = gen_train_example(lastStart, lastEnd)
        #target = gen_train_example(lastStart + 1, lastEnd+1)
        #dataList.append((sample, target))

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64,return_sequences=True, return_state=False),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1)
])

loss_object = tf.keras.losses.MeanSquaredError()

def get_loss(predictions,labels):
    return loss_object(predictions,labels) #divide by length of sequence

optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

def train_step(sequence):
    with tf.GradientTape() as tape:
        input = np.array([[sequence[:-1]]])
        input = np.transpose(input, axes=[0, 2, 1])

        target = np.array(([[sequence[1:]]]))
        target = np.transpose(target, axes=[0, 2, 1])

        prediction = model(input)

        loss = get_loss(prediction,target)
        if loss.dtype == tf.int64:
            return prediction, target, loss

        gradients = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients,
                                                model.trainable_variables))

        return prediction,target,loss


def visualize(prediction,target):
    prediction = tf.reshape(prediction,[prediction.shape[1]])
    target = tf.reshape(target,[target.shape[1]])
    plt.plot(prediction,'ro')
    plt.plot(target,'bs')
    plt.show()

for epoch in range(0,10):
    for i in range(0,len(dataList)):
        if len(dataList[i]) == 1:
            continue
        prediction,target,loss = train_step(dataList[i])
        if i%1000 == 0:
            visualize(prediction,target)


checkpoint_dir = './training_checkpointsProject'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model)
checkpoint.save(file_prefix=checkpoint_prefix)