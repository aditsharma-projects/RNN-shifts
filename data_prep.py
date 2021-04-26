import pandas as pd
import numpy as np
import pickle

# Generates a dictionary from Stata log file in which 'labels list' is run
# For some column and numerically encoded val, the decoded string is:
# `labels_map[column][encoded_val]` (if labels_map is this function's output)
def labels_list_to_dict(file):
    f = open(file, "r")
    started = False
    labels_map = {} # labels_map[column][encoded_val] = decoded_string
    col = None
    for line in f:
        line = line.strip()
        if line == '. label list': # labels list command started
            started = True
            continue
        elif not started: # wait for command start
            continue
        
        if line == '': # labels list command ended
            break
        
        if line[-1] == ':': # start of column
            col = line[:-1]
            labels_map[col] = {}
        elif col is None:
            print(line)
            raise Exception("Error parsing input")
        else:
            tokens = line.split(' ')
            encoded_val = int(tokens[0])
            decoded_string = ' '.join(tokens[1:])
            labels_map[col][encoded_val] = decoded_string
    return labels_map

#Generates a dictionary that maps label names as they appear in labels.txt
#to the abbreviations as they appear in the pbj facility csv files
def label_mapping_dict(file):
    f = open(file)
    output = {}
    for line in f:
        line = line.strip()
        LHS,RHS = line.split(":")
        LHS = LHS.strip()
        RHS = RHS.strip()
        _, label = LHS.split("-")
        _, abbrev = RHS.split("-")
        output[label] = abbrev
       
    return output

# Generate deterministic file name from configuration
def generate_file_names(preprocessed_dir, nrows, fill_missing_shifts, normalize):
    name = preprocessed_dir + 'pbj'
    if nrows is not None:
        name += f"_nrows_{nrows}"
    if fill_missing_shifts:
        name += "_zeros"
    if normalize:
        name += "_norm"
    return name + '.csv', name + '.pickle'

# Print s if conditional is truthy
def print_if(s, conditional):
    if conditional:
        print(s)

# Pad insert rows between each employee's start and end days with 0 hours
def do_fill_missing_shifts(df, verbose):
    print_if("Filling missing shifts...", verbose)

    # Partition by employee id
    grouped = df.groupby('employee_id')

    partitions = []
    for name, group in grouped:
        # Sort by date
        group = group.sort_values(by='date')
        day = min(group['date'])
        for i in range(len(group)):
            row_copy = {key:value for key, value in group.iloc[0].items()}
            row_copy['hours'] = 0
            # Catch date up to current index by filling in with cached rows
            while day < group.iloc[i]['date']:
                row_copy['date'] = day
                group = group.append({key:value for key, value in row_copy.items()}, ignore_index=True)
                day += pd.DateOffset(1)

            # Account for current index's day
            day += pd.DateOffset(1)

        # Sort by date with new rows
        group = group.sort_values(by='date')
        partitions.append(group)

    return pd.concat(partitions)

def initial_preprocess(raw_path, preprocessed_dir, nrows=None, fill_missing_shifts=False, verbose=True, normalize=False):
    data_file, info_file = generate_file_names(preprocessed_dir, nrows, fill_missing_shifts, normalize)

    try:
        print_if(f"Loading preprocessed data from '{data_file}'...", verbose)
        df = pd.read_csv(data_file)

        print_if(f"Loading related info from '{info_file}'...", verbose)
        with open(info_file, 'rb') as f:
            info = pickle.load(f)
        return df, info
    except FileNotFoundError:
        print_if("Failed.", verbose)
    
    print_if("Loading data...", verbose)
    df = pd.read_csv(raw_path, nrows=nrows, dtype={'hours':'float64'}, parse_dates = ['date'])
    info = {}
    if fill_missing_shifts:
        df = do_fill_missing_shifts(df, verbose)
    
    if normalize:
        # Normalize features
        means = {}
        stds = {}
        #df = df.filter(['hours', 'date_int'])
        for col in ['hours']:
            means[col] = df[col].mean()
            stds[col] = df[col].std()
            df[col] = (df[col] - means[col]) / stds[col]
        info['means'] = means
        info['stds'] = stds

    print_if("Saving preprocessed data...", verbose)
    df.to_csv(data_file, index=False)

    with open(info_file, 'wb') as handle:
        pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print_if("Preprocessing finished.", verbose)
    return df, info