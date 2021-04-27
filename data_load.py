import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from operator import itemgetter
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import path
from data_prep import labels_list_to_dict
from data_prep import label_mapping_dict
labels_list_logfile = '/mnt/staff/rtjoa/shifts/RNN-shifts/labels.txt'
labels_map = labels_list_to_dict(labels_list_logfile)
map_abbrev = label_mapping_dict("labelsToAbbrev.txt")

ONLYGENERATENUM = 90
def dateIndex(date):    #indexing years starting at 2000, works for sure as long as records stop before 2100
    date = str(date)
    year = int(date[7:11])
    
    names = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    days = [31,28,31,30,31,30,31,31,30,31,30,31]
    index = 0
    for i in range(12):
        if names[i] == date[4:7]:
            break
        index += days[i]
    index += int(date[2:4])
    
    return (index,year)
ROWS = 100000
###########################
frame1 = None
frame2 = None
currQ = -1
currY = -1
def load_csvs(quarter,year):
    fileNurse = "/export/storage_adgandhi/PBJhours_ML/Data/Raw/PBJ_Public/Nurse/"+"PBJ_Daily_Nurse_Staffing_CY_"+ str(year) + "_Q" + str(quarter) + ".csv"
    if not path.exists(fileNurse):
        fileNurse = "/export/storage_adgandhi/PBJhours_ML/Data/Raw/PBJ_Public/Nurse/"+"PBJ_Daily_Nurse_Staffing_CY_"+ str(year) + "Q" + str(quarter) + ".csv"
    
    fileNonNurse = "/export/storage_adgandhi/PBJhours_ML/Data/Raw/PBJ_Public/Non_Nurse/"+"PBJ_Daily_Non-Nurse_Staffing_CY_"+ str(year) + "_Q" + str(quarter) + ".csv"
    
    if not path.exists(fileNonNurse):
        fileNonNurse = "/export/storage_adgandhi/PBJhours_ML/Data/Raw/PBJ_Public/Non_Nurse/"+"PBJ_Daily_Non-Nurse_Staffing_CY_"+ str(year) + "Q" + str(quarter) + ".csv"
        
    global frame1
    global frame2
    global currQ
    global currY
    frame1 = pd.read_csv(fileNurse)
    frame2 = pd.read_csv(fileNonNurse)
    frame1.columns = frame1.columns.str.upper()
    frame2.columns = frame2.columns.str.upper()
    currQ = quarter
    currY = year
    return 


def get_facility_data(quarter,year,job,provNum,pay):
    provNum = labels_map['prov_id_label'][provNum]
    if provNum.isdigit():
        provNum = int(provNum)
    else:
        return np.zeros(90).tolist()
    job = labels_map['job_title_label'][job]
    if job != 'Diagnostic X-ray Service Worker':
        job = map_abbrev[job]
    job = job.upper()
    if currQ == -1 or year != currY or quarter != currQ:
        load_csvs(quarter,year)
    
    #get subset for provider id
    mask1 = (frame1["PROVNUM"] == provNum)
    mask2 = (frame2["PROVNUM"] == provNum)
    
    #print("PROVNUM " +str(provNum))
    #print(type(provNum))
    
    
    #extract the right column depending on worker type
    #print(job)
    if job in frame1:
        seq = frame1[mask1][job].tolist()
        #print(seq)
        return seq[0:90]
    elif job in frame2:
        seq = frame2[mask2][job].tolist()
        #print(seq)
        return seq[0:90]
    else:
        #print("NOT FOUND")
        return np.zeros(90).tolist()
    return

def one_quarter(seq,startDate):
    index, year = dateIndex(startDate)
    quarter = int(index/90)+1
    output = []
    for day in range((quarter-1)*90,quarter*90+1):
        if day < index:
            output.append(0)
        elif day >= index + len(seq):
            output.append(0)
        else:
            output.append(seq[day-index])
    
    return (output,quarter,year)

def helper_index(inTuple):
    op = itemgetter(2)
    dateString = op(inTuple)
    dateTuple = dateIndex(dateString)
    return dateTuple[0]+(dateTuple[1] - 2015)*365

def get_entry(i,j,listSorted):
    dataPoints = sorted(listSorted[i:j],key=helper_index) #Issue, can't currently sort the date string
    min = dateIndex(dataPoints[0][2])
    max = dateIndex(dataPoints[len(dataPoints)-1][2])
    edge = 1
    while max[1] != min[1]:
        max = dateIndex(dataPoints[len(dataPoints)-2*edge][2]) #makes things easier if only spanning one year
    #want to generate (0 padded) shifts for a single quarter
    shifts = []
    currIndex = 0
    prefix = [0]
    for ind in range(min[0],max[0]+1):
        if dateIndex(dataPoints[currIndex][2])[0] == ind:
            shifts.append(dataPoints[currIndex][0])
            currIndex += 1
        else:
            shifts.append(0)
    
    startDate = dataPoints[0][2]
    jobTitle = dataPoints[0][3]
    providerId = dataPoints[0][4]
    payType = dataPoints[0][5]
    shifts,quarter,year = one_quarter(shifts,startDate)
    #print(quarter)
    if quarter == 5:
        return None
    if len(shifts)==0:
        return None
    
    facSequence = get_facility_data(quarter,year,jobTitle,providerId,payType)
    
    extraDescriptors = [jobTitle,providerId,payType]
    
    #print(len(shifts))
    #print(len(facSequence))
    
    
    return shifts+facSequence+extraDescriptors

def process_Data(data, ind):
    for row in data:
        row = tuple(row)
    data = data.tolist()
    data = sorted(data,key=itemgetter(1))
    
    dataList = []
    currID = data[0][1]
    lastStart = 0
    lastEnd = 0
    for i in range(0,len(data)):
        lastEnd = i
        if data[i][1] != currID:
            currID = data[i][1]
            sample = get_entry(lastStart,lastEnd,data)
            if sample != None:
                sample = np.array(sample)
                dataList.append(sample)
                #print(sample.shape)
            
            lastStart = i
        if i%10000 == 0:
            print(i)
    #print(len(dataList))
    #print(dataList[0].shape)
    dataList = np.array(dataList)
    #print(dataList.shape)
    np.savetxt("/users/facsupport/asharma/Data/Preprocessed/"+str(ind)+".csv",np.array(dataList),delimiter=",")
    
    return

offset = 0
data = genfromtxt('/export/storage_adgandhi/PBJ_data_prep/pbj_full.csv',delimiter=',',skip_header=1,dtype="f8,i8,S9,i8,i8,i8",max_rows=ROWS)

while data.shape[0] != 0:
    process_Data(data,offset)
    offset += 1
    data = genfromtxt('/export/storage_adgandhi/PBJ_data_prep/pbj_full.csv',delimiter=',',skip_header=1+offset*ROWS,dtype="f8,i8,S9,i8,i8,i8",max_rows=ROWS)
    print("Processed and Saved "+str(offset*ROWS)+" data entries")



############################

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

#dataList = generate_dataList()

#print(len(dataList))



def save_Data(dataList):
    np.savetxt("/users/facsupport/asharma/Data/batches/31/OH/BIGSET.csv",dataList,delimiter=",")
    
    
#save_Data(dataList)
