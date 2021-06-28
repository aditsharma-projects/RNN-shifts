import pandas as pd
import time
import datetime
import tensorflow as tf
import numpy as np
import sys
import multiprocessing as mp
import getpass
from random import randrange

# %%
#### CONFIGURATION - DATA & FILES ####

# Number of rows to truncate to. Unless debugging, should always be set to None
# so full data files are used.
ROWS = None

# File inputs/outputs
LOG_PATH = '/users/facsupport/asharma/RNN-shifts/output/rnn_autotuning_history.csv'
TRAIN_PATH = '/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/training_set_60.csv'
VAL_PATH = '/export/storage_adgandhi/PBJhours_ML/Data/Intermediate/train_test_validation/crossvalidation_set_60.csv'

# Ensure we don't save truncated output to same place
if ROWS is not None:
    LOG_PATH = 'temp_' + LOG_PATH

# %%
#### CONFIGURATION - MODEL STRUCTURES ####

# Hyperparameter choices
SHAPES = [[1],[1,1],[1,1,1],[1,1,1,1],[4,1],[8,4,1],[16,8,4,1],[8,1],[16,8,1],[4,2,1,1]]
SHAPE_SCALES = [2,4,6,8]
EMBED_SIZES = [0,5,10,20,50,100]
LSTM_UNITS = [8,16,32,64,128]

# Number of days of lagged shifts to feed directly to RNN
LAGGED_DAYS = 60

#### CONFIGURATION - TRAINING SETTINGS ####

# Number of epochs to train an individual model
EPOCHS = 5

# After three epochs, train 0.1 * this rate; after seven, 0.01 * this rate
INITIAL_LEARNING_RATE = 1e-3

# Number of rounds without improvement after which to stop. Each "round"
# consists of a pool of multiple models.
EARLY_STOPPING_ROUNDS = 1

BUFFER_SIZE = 10000
BATCH_SIZE = 128

# %%
####

# Base dict of values to log for each run.
base_model_info = {
 'Recurrence length':-1,
 'LSTM Units':-1,
 'Embedding Dimension':-1,
 'FF model shape':[],
 'Initial Learning Rate':INITIAL_LEARNING_RATE,
 'Regularization':"Batch Normalization",
 'Metric':"mse",
 'Training loss':-1,
 'Val loss':-1,
 'time_start':-1,
 'time_duration':-1,
 'Epochs':EPOCHS,
 'Columns':['day_of_week', 'avg_employees', 'perc_hours_today_before', 'perc_hours_yesterday_before', 'perc_hours_tomorrow_before'],
 'LSTM type':"Unconditioned",
 'user':getpass.getuser(),
 'coordinates':[]
}

# Key indices of input columns
# The embedded column (default: prov_id) takes up index 0
# The recurrence columns (default: hours_l1, hoursl2, etc.) take up indices 1 through LAGGED_DAYS
# The recurrence mask columns take up indices LAGGED_DAYS + 1 through LAGGED_DAYS * 2
# The descriptor columns and one-hot encoded columns follow
embedded_idx = 0
recurrence_start_idx = 1
mask_start_idx = LAGGED_DAYS + 1
descriptors_start_idx = 2 * LAGGED_DAYS + 1

# Override print to flush by default
def print(*objects, sep=' ', end='\n', file=sys.stdout, flush=True):
    __builtins__.print(*objects, sep=sep, end=end, file=file, flush=flush)

# Logs hyperparameter specifications and other attributes of each run into a csv file
def log_model_info(model_info, path):
    try:
        print(path)
        df = pd.read_csv(path)
        print('done')
    except FileNotFoundError:
        #print(f"History csv not found at {path}. Creating new file.")
        df = pd.DataFrame()
    
    new_df = pd.DataFrame({k: [v] for k, v in model_info.items()})
    df = pd.concat([df, new_df], axis=0)
    df = df.sort_values(by='Val loss')
    df.to_csv(path, index=False)
          
    
# Expand categorical variables enumerated in labels to one-hot encodings
# Takes in pandas dataframe and returns tf tensor 
# Column ordering is preservered, with the convereted categorical columns dropped from the frame
# and their one-hot encodings concatenated to the end of the converted tensor
def expand_one_hot(labels,dataset):
    outList = []
    for label in labels:  
        col = dataset[label]
        # Generate a dict for all unique values (Don't waste space encoding non important job id's)
        map = {}
        index = 0
        for element in col.unique():
            map[element] = index
            index += 1
        col = col.map(map)
        tensor = tf.one_hot(col,len(col.unique()))
        outList.append(tensor)
        dataset = dataset.drop(columns=[label])
    
    outList.insert(0,dataset)
    output = tf.concat(outList,1)
    return output

#Inserts a mask equal to LAGGED_DAYS to distinguish nan values from 0
def mask_nan(frame):
    mask = frame.isna()
    #Input frame is arranged as follows: 'hours','prov_id',recurrance block,other block
    for i in range(1,LAGGED_DAYS+1):
        frame.insert(1+LAGGED_DAYS+i,f"mask_{i}",mask[f"hours_l{i}"].astype(int).astype('float32'))
    frame = frame.fillna(0)
    return frame


#Loads and preprocesses data from training_set.csv and crossvalidation_set.csv
def get_data():
    nrows = None
    include_fields = ['hours','prov_id','day_of_week','avg_employees','perc_hours_today_before',
                      'perc_hours_yesterday_before', 'perc_hours_tomorrow_before']
    for i in range(1,LAGGED_DAYS+1):
        ##Inserts recurrence block starting at index 2
        include_fields.insert(i+1,f"hours_l{i}")
    
    train = pd.read_csv(TRAIN_PATH,nrows=nrows,usecols=include_fields)
    val = pd.read_csv(VAL_PATH,nrows=nrows,usecols=include_fields)
    print("Loaded")
    #Reorder columns to the order specified in include_fields [hours,prov_id,recurrence block,other block]
    train = train.reindex(columns=include_fields)
    val = val.reindex(columns=include_fields)
    
    #Mask nan values
    train = mask_nan(train)
    val = mask_nan(val)
      
    #Remove providers that appear in val set but not train
    train_providers = train['prov_id'].unique()
    val_providers = val['prov_id'].unique()
    for value in val_providers:
        if value not in train_providers:
            mask = (val['prov_id']!=value)
            val = val[mask]

    # Remap prov_id's between 0 - # providers
    provider_map = {}
    index = 0
    for element in train['prov_id'].unique():
        provider_map[element]=index
        index +=1
    train['prov_id'] = train['prov_id'].map(provider_map)
    val['prov_id'] = val['prov_id'].map(provider_map)

    #Separate predictors and labels
    train_inputs, train_labels = train.drop(['hours'], axis=1), train.filter(['hours'])
    val_inputs, val_labels = val.drop(['hours'], axis=1), val.filter(['hours'])

    vocab_size = len(train_inputs['prov_id'].unique())

    #expand categoricals to one-hot encodings
    train_inputs = expand_one_hot(['day_of_week'],train_inputs)
    val_inputs = expand_one_hot(['day_of_week'],val_inputs) 
    
    BUFFER_SIZE = 10000
    BATCH_SIZE = 128
    trainSet = tf.data.Dataset.from_tensor_slices((train_inputs,train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)
    valSet = tf.data.Dataset.from_tensor_slices((val_inputs,val_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)
    return trainSet,valSet,vocab_size
        

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

class RNN_Conditioned(tf.keras.Model):
    #define all components of model
    def __init__(self,recurrence_length,lstm_units,dense_shape,embed_dim,vocab_size):
        super(RNN_Conditioned, self).__init__()
        self.embed_dim = embed_dim
        self.recurrence_length = int(recurrence_length)
        self.units = lstm_units
        self.transform = tf.keras.layers.Dense(lstm_units-embed_dim)
        if embed_dim == 0:
            embed_dim += 1
        self.embeddings = tf.keras.layers.Embedding(vocab_size,embed_dim) 
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
        transformed_inputs = self.transform(additional_inputs)
        if self.embed_dim != 0:    
            embedding_vectors = self.embeddings(inputs[:,embedded_idx])
            transformed_inputs = tf.concat([transformed_inputs,embedding_vectors],1)
        c_0 = tf.convert_to_tensor(np.random.random([128, self.units]).astype(np.float32))
       #h_0 = tf.convert_to_tensor(np.random.random([128, self.units]).astype(np.float32))
        h_0 = transformed_inputs
        x = self.lstm(time_series, initial_state=[h_0,c_0])
           
        for layer in self.dense_layers:
            x = layer(x)
        
        return self.out(x)

# Callback function to decrease learning rate
def decay(epoch):
  if epoch < 3:
    return INITIAL_LEARNING_RATE
  elif epoch >= 3 and epoch < 7:
    return INITIAL_LEARNING_RATE/10
  else:
    return INITIAL_LEARNING_RATE/100

# helper function for lines 228 and 229
def hash_coordinates(coords):
    return int(coords[1]) + int(coords[4])*10 + int(coords[7])*100 + int(coords[10])*1000

# Worker function for multiprocessing Process. Trains an RNN with the specified recurrence length
def train_and_test_models(time_offset,recurrence_length,lstm_units,dense_shape,embed_dim,lstm_type,coordinates):
    #Check if we've already tested this run
    try:
        log_file = pd.read_csv(LOG_PATH)
        log_file['coordinates'] = log_file['coordinates'].apply(hash_coordinates)
        if hash_coordinates(str(coordinates)) in log_file['coordinates'].unique():
            matching_trials = log_file.loc[log_file['coordinates']==hash_coordinates(str(coordinates))] 
            if LAGGED_DAYS in matching_trials['Recurrence length'].unique():
                return matching_trials['Val loss'].iloc[0] 
    except FileNotFoundError:
         garbage = 0

    #restrict each process to 50 cores
    tf.config.threading.set_intra_op_parallelism_threads(50)
    tf.config.threading.set_inter_op_parallelism_threads(50)

    #To prevent all launched threads from loading dataframes simultaneously
    time.sleep(time_offset)
    start_time = time.time()
    trainSet,valSet,vocab = get_data()     
    start_date = datetime.datetime.now()
    with tf.device('/cpu:0'):
        if lstm_type == "Unconditioned":
            model = RNN(recurrence_length,lstm_units,dense_shape,embed_dim,vocab)
        else:
            model = RNN_Conditioned(recurrence_length,lstm_units,dense_shape,embed_dim,vocab)
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()])
        
        callbacks = [tf.keras.callbacks.LearningRateScheduler(decay)]
        history = model.fit(trainSet, epochs=EPOCHS, callbacks=callbacks,verbose=0)
        valLoss, metric = model.evaluate(valSet)
   
    time_taken = str(datetime.timedelta(seconds=(time.time()-start_time)))
    
    #Store values of run into dict
    param_dict = base_model_info.copy()
    train_loss = history.history['loss'][EPOCHS-1]
    param_dict['Recurrence length'] = recurrence_length
    param_dict['LSTM Units'] = lstm_units
    param_dict['Embedding Dimension'] = embed_dim
    param_dict['FF model shape'] = dense_shape
    param_dict['Training loss'] = train_loss
    param_dict['Val loss'] = valLoss
    param_dict['time_start'] = start_date
    param_dict['time_duration'] = time_taken
    param_dict['LSTM type'] = lstm_type
    param_dict['coordinates'] = coordinates
    log_model_info(param_dict,LOG_PATH)
    #print("COMPLETED WORK")
    return valLoss

#takes in list x and multiplies all elements of x by mult
def list_helper(x,mult):
    outList = [y*mult for y in x]
    outList = outList + [1]
    return outList

#Returns permutation corresponding to startCoords + its 4 greater neighbors
def gen_perm(startCoords,units,shape_ratios,embeddings,multiplier):
    out = []
    coord_list = []
    fields_list = [units,shape_ratios,embeddings,multiplier]
    for i in range(-1,4):
        time_delay = randrange(5)*600
        currCoords = startCoords.copy()
        if i >= 4:  #This case implements a way to search over multiple shape ratios
            currCoords[1] += (i-3) # in one search iteration
            i=1
        if i < 4 and i >= 0:
            currCoords[i] += 1 
        if currCoords[i] >= len(fields_list[i]):
            continue
        if embeddings[currCoords[2]] >= units[currCoords[0]]:
            continue
        shapes = [list_helper(x,multiplier[currCoords[3]]) for x in shape_ratios] #Apply list_helper to each list in shape_ratios
        out.append((time_delay,LAGGED_DAYS,units[currCoords[0]],shapes[currCoords[1]],
                    embeddings[currCoords[2]],"Unconditioned",currCoords))
        out.append((time_delay,LAGGED_DAYS,units[currCoords[0]],shapes[currCoords[1]],
                    embeddings[currCoords[2]],"Conditioned",currCoords))
        coord_list.append(currCoords)
        coord_list.append(currCoords)
            
    return out,coord_list

#interleaves permuations corresponding to all start coordinates provided in starts
def coordList_wrapper(starts,lstm_units,shape_ratios,embed_sizes,mult):
    work_list = []
    coords_list = []
    for startCoords in starts:
        work_perm, coords_perm = gen_perm(startCoords,lstm_units,shape_ratios,embed_sizes,mult)
        work_list.append(work_perm)
        coords_list.append(coords_perm)

    outList_work = [val for tup in zip(*work_list) for val in tup]
    outList_coords = [val for tup in zip(*coords_list) for val in tup]
    return outList_work,outList_coords
#Implements a greedy search: Compute delta_val_loss for startCoords and its 4 upper adjacent neighbors
#If atleast one of the neighbors gives an improvement, choose the neighbor with the best improvement
#and repeat witht the neighbor as the new staertCoords
def autotune(starts,shape_ratios,embed_sizes,lstm_units,mult):
    best_loss = 1000               
    improving = True              
    while(improving):
        improving = False        
        work_list, coords_list = coordList_wrapper(starts,lstm_units,shape_ratios,embed_sizes,mult)
        with mp.Pool(processes=6) as pool:           
            results = pool.starmap(train_and_test_models,work_list)

            # Process losses of results
            for loss, coords in zip(results, coords_list):
                if loss < best_loss:
                    best_loss = loss
                    starts= [coords]
                    improving = True
                
    return best_loss
            
if __name__ == '__main__':
    starts = [[2,0,0,0],[2,0,2,0],[3,0,0,0],[3,0,2,0],[3,0,0,2]]
    optimum = autotune(starts,SHAPES,EMBED_SIZES,LSTM_UNITS,SHAPE_SCALES)
    print(f"Best Validation Loss: {optimum}")
