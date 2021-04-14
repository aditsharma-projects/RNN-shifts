import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from operator import itemgetter
import matplotlib.pyplot as plt

ONLYGENERATE8 = True

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

ROWS = 100000000
data = genfromtxt('/users/facsupport/asharma/Data/pbj_full.csv',delimiter=',', skip_header = 1+ROWS, dtype="i8,f8,i8,S10,S30,i8,S12",max_rows=ROWS)
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
    if max-min+1!=8 and ONLYGENERATE8:
        return (False,None,None)
    output = []
    currIndex = 0
    for ind in range(min,max+1):
        if dataPoints[currIndex][3] == ind:
            output.append(dataPoints[currIndex][1])
            currIndex += 1
        else:
            output.append(0)

    return (True,output,max-min+1)

def findksequence():        
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
        if sample[0]:
            seq = sample[1]
            dataList.append(sample[1:])
        lastStart = i
    if i%1000000 == 0:
        print(i)

#print(dataList)
print(type(dataList))    
print(len(dataList))
dataList = sorted(dataList,key=itemgetter(1))
#print(len(dataList)
for i in range(0,len(dataList)):
    print(dataList[i][1])
for i in range(0,len(dataList)):
    print(dataList[i])


dataFinal = np.ndarray(shape=(len(dataList),10))
for i in range(len(dataList)):
    seq = dataList[i][0]
    mean = np.mean(seq)
    std = np.std(seq)
    list = np.append((seq-mean)/std,mean)
    dataFinal[i] = np.append(list,std)


np.savetxt("/users/facsupport/asharma/Data/batches/8/two.csv",dataFinal,delimiter=",")
