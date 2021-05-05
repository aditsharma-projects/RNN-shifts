import pandas as pd
import numpy as np
from datetime import datetime

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

# Generates a dictionary that maps label names as they appear in labels.txt
# to the abbreviations as they appear in the pbj facility csv files
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
def generate_file_names(preprocessed_dir, nrows, fill_missing_shifts, normalize, prev_shifts):
    name = preprocessed_dir + 'pbj'
    if nrows is not None:
        name += f"_nrows_{nrows}"
    if fill_missing_shifts:
        name += "_zeros"
    if normalize:
        name += "_norm"
    if prev_shifts:
        name += f"_prev_shifts_{prev_shifts}"
    return name + '.csv', name + '.info.csv'

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

def do_add_prev_shifts(df, prev_shifts, verbose):
    print_if("Adding previous shifts...", verbose)

    for t in range(prev_shifts + 1):
        df.insert(t, f"t_{t}", -1)
    
    for i in range(prev_shifts, len(df)):
        for t in range(prev_shifts+1):
            if df.iloc[i-t].employee_id == df.iloc[i].employee_id:
                df.iloc[i, df.columns.get_loc(f"t_{t}")] = df["hours"].iloc[i-t]

    # Drop rows that do not have complete prev shift data
    df = df[df[f"t_{prev_shifts}"] != -1]
    return df

def initial_preprocess(raw_path, preprocessed_dir, nrows=None, fill_missing_shifts=False, verbose=True, normalize=False, prev_shifts=0, force_reload = False):
    if prev_shifts < 0:
        raise ValueError()

    data_file, info_file = generate_file_names(preprocessed_dir, nrows, fill_missing_shifts, normalize, prev_shifts)

    if not force_reload:
        try:
            print_if(f"Loading preprocessed data from '{data_file}'...", verbose)
            df = pd.read_csv(data_file)

            print_if(f"Loading related info from '{info_file}'...", verbose)
            info_df = pd.read_csv(info_file)
            info = {col:info_df.iloc[0][col] for col in info_df.columns}
            return df, info
        except FileNotFoundError:
            print_if("Failed.", verbose)
    
    print_if("Loading data...", verbose)
    df = pd.read_csv(raw_path, nrows=nrows, dtype={'hours':'float64'}, parse_dates = ['date'])
    info = {
        "nrows": nrows,
        "fill_missing_shifts": fill_missing_shifts,
        "normalize": normalize,
        "prev_shifts": prev_shifts,
        "creation_date": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    }

    if fill_missing_shifts:
        df = do_fill_missing_shifts(df, verbose)
    
    if normalize:
        # Normalize features
        for col in ['hours']:
            mean = df[col].mean()
            std = df[col].std()
            info[f"means.{col}"] = mean
            info[f"stds.{col}"] = std
            df[col] = (df[col] - mean) / std
    
    if prev_shifts:
        df = do_add_prev_shifts(df, prev_shifts, verbose)

    print_if("Saving preprocessed data...", verbose)
    df.to_csv(data_file, index=False)

    info_cols = []
    info_row = []
    for k, v in info.items():
        info_cols.append(k)
        info_row.append(v)
    info_df = pd.DataFrame([info_row], columns = info_cols)
    info_df.to_csv(info_file, index=False)
    
    print_if("Preprocessing finished.", verbose)
    return df, info