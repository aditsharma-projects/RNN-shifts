import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from operator import itemgetter
import matplotlib.pyplot as plt

ONLYGENERATENUM = 31
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

def gen_list():
    data = genfromtxt('/users/facsupport/asharma/Data/pbj_full.csv',delimiter=',', skip_header = 1,dtype="i8,f8,i8,S10,S30,i8,S12",max_rows=ROWS)
    print("LOADED DATA")
    list = []
    for x in range(0,len(data)):
        nextTuple = (data[x][0],data[x][1],data[x][2],dateIndex(str(data[x][3])),str(data[x][4]),data[x][5],data[x][6])
        list.append(nextTuple)
        if x%1000000 == 0:
            print(x)
            
    return list

#0-27 numerical encoding of job description
def OH_Job(job_description):
    unique_jobs = ["b'Administrator'", "b'Respiratory Therapist'", "b'Other Activities Staff'", "b'Physical Therapy Aide'", "b'Nurse Practitioner'", "b'Speech/Language Pathologist'", "b'Other Social Worker'", "b'Nurse Aide in Training'", "b'Registered Nurse Director of N'", "b'Dietitian'", "b'Other Physician'", "b'Registered Nurse with Administ'", "b'Certified Nurse Aide'", "b'Occupational Therapist'", "b'Physical Therapist'", "b'Therapeutic Recreation Special'", "b'Pharmacist'", "b'Housekeeping Service Worker'", "b'Medical Director'", "b'Qualified Activities Professio'", "b'Mental Health Service Worker'", "b'Occupational Therapy Assistant'", "b'Registered Nurse'", "b'Feeding Assistant'", "b'Qualified Social Worker'", "b'Physical Therapy Assistant'", "b'Licensed Practical/Vocational '", "b'Other Service Worker'"]

    for i in range(28):
        if unique_jobs[i] == job_description:
            return i
    return -1

#returns start and end indices of a "rich" subsequence
def get_rich_sequence(seq,prefix,threshold):
    if len(seq) == ONLYGENERATENUM:
        return 0,ONLYGENERATENUM
    
    for x in range(len(seq)-ONLYGENERATENUM):
        num_entries = prefix[x+ONLYGENERATENUM] - prefix[x] + 1
        if num_entries > threshold:
            return x, x+ONLYGENERATENUM
    return -1, -1
    
    

def gen_train_example(i,j,listSorted): #listSorted[i] to listSorted[j-1] inclusive are the same employee
    dataPoints = sorted(listSorted[i:j],key=itemgetter(3))
    min = dataPoints[0][3]
    max = dataPoints[len(dataPoints)-1][3]
    if max-min+1 < ONLYGENERATENUM or max-min+1 > 365:
        return (False,None)
    output = []
    currIndex = 0
    prefix = [0]
    i = 0
    for ind in range(min,max+1):
        if dataPoints[currIndex][3] == ind:
            output.append(dataPoints[currIndex][1])
            currIndex += 1
            prefix.append(prefix[i]+1)
        else:
            output.append(0)
            prefix.append(prefix[i])
           
        i += 1
    
    prefix = prefix[1:] #drop the first dummy entry
    x,y = get_rich_sequence(output,prefix,10) #atleast 10 nonzero entries
    if x==-1:
        return (False,None)
    
    seq = np.array(output)
    seq = seq[x:y]
    
    #mean = np.mean(seq)
    #std = np.std(seq)
    #seq = (seq-mean)/std
    #seq = np.append(seq,[mean,std])
    
    
    return (True,np.append(seq,OH_Job(dataPoints[0][4])))


def generate_dataList():
    listSorted = sorted(gen_list(),key=itemgetter(2))
    
    dataList = []
    currID = listSorted[0][2]
    lastStart = 0
    lastEnd = 0
    for i in range(0,len(listSorted)):
        lastEnd = i
        if listSorted[i][2] != currID:
            currID = listSorted[i][2]
            sample = gen_train_example(lastStart,lastEnd,listSorted)
            if sample[0]:
                record = sample[1]
                dataList.append(record)
            lastStart = i
        if i%1000000 == 0:
            print(i)
     
    return dataList

dataList = generate_dataList()

print(len(dataList))



def save_Data(dataList):
    np.savetxt("/users/facsupport/asharma/Data/batches/31/OH/BIGSET.csv",dataList,delimiter=",")
    
    
save_Data(dataList)
