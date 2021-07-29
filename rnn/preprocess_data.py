import pandas as pd
import time
import datetime
import tensorflow as tf
import numpy as np
import sys
import multiprocessing as mp
import getpass
from random import randrange

SET = "A"

PATH = f'/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/full_5050{SET}_60.csv'
SAVE = f'/export/storage_adgandhi/PBJhours_ML/tf_data/50{SET}.csv'

LAGGED_DAYS = 60

#Inserts a mask equal to LAGGED_DAYS to distinguish nan values from 0
def mask_nan(frame):
    mask = frame.isna()
    #Input frame is arranged as follows: 'hours','prov_id',recurrance block,other block
    for i in range(1,LAGGED_DAYS+1):
        frame.insert(1+LAGGED_DAYS+i+4,f"mask_{i}",mask[f"hours_l{i}"].astype(int).astype('float16'))
    frame = frame.fillna(0)
    return frame

def data_gen(frame):
    for i in range(len(frame)):
        yield frame[i]

#Loads and preprocesses data from training_set.csv and crossvalidation_set.csv
def get_data():
    nrows = None
    include_fields = ['hours','employee_id','prov_id','date','job_title','pay_type','day_of_week','avg_employees','perc_hours_today_before',
                      'perc_hours_yesterday_before', 'perc_hours_tomorrow_before']
    recurrance_block = []
    for i in range(1,LAGGED_DAYS+1):
        ##Inserts recurrence block starting at index 2
        include_fields.insert(i+5,f"hours_l{i}")
        recurrance_block.append(f"hours_l{i}")

    A = pd.read_csv(PATH,nrows=nrows,usecols=include_fields,engine='c',dtype={c:np.float16 for c in recurrance_block})
    print("Loaded")

    
    #Reorder columns to the order specified in include_fields [hours,prov_id,recurrence block,other block]
    A = A.reindex(columns=include_fields)
    print("Reindexed")

    #Mask nan values
    A = mask_nan(A)
    print("Masked")

    ###Temp Block
    one_hot = pd.get_dummies(A.day_of_week)
    A = A.drop(['day_of_week'],axis=1)
    A = pd.concat([A,one_hot],axis=1)
    A.to_csv(SAVE,index=False,header=True)
    ####
    print("FINSIHED TEMP BLOCK")
    return 0
        

trainSet = get_data()
print("DONE")
