import numpy as np
from numpy import genfromtxt
from operator import itemgetter
import pandas as pd
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

def date_to_quarter(dateTuple):
    quarters = [90,91,92,92]
    if dateTuple[1] % 4 == 0 and dateTuple[1] % 100 != 0:
        quarters[0] += 1
    total = 0
    for i in range(4):
        total += quarters[i]
        if total > dateTuple[0]:
            return (i+1,total-quarters[i],total)
    return -1

ROWS = 100000
###########################
frame1 = None
frame2 = None
currQ = -1
currY = -1
#Loads the 2 csv's corresponding to quarter and year into pandas data frame objects
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

#Checks that current csv's match quarter/year, then extracts the correct column based on job id
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

#returns (zero-padded) exactly one quarter of shift history for any input
def one_quarter(seq,startDate):
    index, year = dateIndex(startDate)
    quarter,start,end = date_to_quarter((index,year))
    output = []
    for day in range(start,end):
        if day < index:
            output.append(0)
        elif day >= index + len(seq):
            output.append(0)
        else:
            output.append(seq[day-index])
    
    return (output[0:90],quarter,year)

#used to sort by date
def helper_index(inTuple):
    op = itemgetter(2)
    dateString = op(inTuple)
    dateTuple = dateIndex(dateString)
    return dateTuple[0]+(dateTuple[1] - 2015)*365

#returns 183 length sequence corresponding to a given worker
def get_entry(i,j,listSorted):
    dataPoints = sorted(listSorted[i:j],key=helper_index) 
    min = dateIndex(dataPoints[0][2])
    max = dateIndex(dataPoints[len(dataPoints)-1][2])
    edge = 1
    while max[1] != min[1]:
        max = dateIndex(dataPoints[len(dataPoints)-2*edge][2]) #makes things easier if only spanning one year
        edge += 1
    #want to generate (0 padded) shifts for a single quarter
    shifts = []
    currIndex = 0
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
    if quarter == -1:
        return None
    if len(shifts)==0:
        return None
    
    facSequence = get_facility_data(quarter,year,jobTitle,providerId,payType)
    if len(facSequence) != 90:
        return None
    
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

offset = 8
data = genfromtxt('/export/storage_adgandhi/PBJ_data_prep/pbj_full.csv',delimiter=',',skip_header=1,dtype="f8,i8,S9,i8,i8,i8",max_rows=ROWS)

while data.shape[0] != 0:
    process_Data(data,offset)
    offset += 1
    data = genfromtxt('/export/storage_adgandhi/PBJ_data_prep/pbj_full.csv',delimiter=',',skip_header=1+offset*ROWS,dtype="f8,i8,S9,i8,i8,i8",max_rows=ROWS)
    print("Processed and Saved "+str(offset*ROWS)+" data entries")



############################


#returns start and end indices of a "rich" subsequence
def get_rich_sequence(seq,prefix,threshold):
    if len(seq) == ONLYGENERATENUM:
        return 0,ONLYGENERATENUM
    
    for x in range(len(seq)-ONLYGENERATENUM):
        num_entries = prefix[x+ONLYGENERATENUM] - prefix[x] + 1
        if num_entries > threshold:
            return x, x+ONLYGENERATENUM
    return -1, -1
    
