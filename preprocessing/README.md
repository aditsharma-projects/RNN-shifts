# Data Preprocessing Pipeline

## `Python Dependencies`
```sh
conda install python=3.8
conda install pandas
conda install numpy
```

## `dta_to_csv.do`
Converts the shift data Stata file to a CSV, where numbers are used for encoded variables, and a list of labels with which the encoded variable strings can be recovered.

Each row in the shift data files (Stata or CSV) corresponds to one day of work for a single employee.

**IN:** `/export/storage_adgandhi/PBJhours_ML/Data/Raw/PBJ/pbj_full.dta`

**OUT:** `/export/storage_adgandhi/PBJ_data_prep/pbj_full.csv` and `labels.txt`

## `data_prep.py`
Preprocesses a csv of raw shift data (produced by `dta_to_csv.do`) based on several parameters. Returns a tuple containing the processed dataframe and a dictionary with info about how/when preprocessing took place.
```py
initial_preprocess(
    raw_path, preprocessed_dir,
    nrows=None, fill_missing_shifts=False, normalize=False, prev_shifts=0,
    day_of_week=False, force_reload=False, verbose=True, fac_data=True):
```
### raw_path: ***str***
CSV file to read from.

### preprocessed_dir: ***str***
Prefix for filenames of saved preprocessed files. If they already exist and can be loaded successfully, the preprocessing computations can be skipped (unless `force_reload` is set to `True`).

### nrows: ***int***
Number of rows from the head of CSV to read.

### fill_missing_shifts: ***bool***
If true: The data will be partitioned by employee then sorted by timestamp. For every day between the employee's start and end date, if there is no data for it, a row for a zero-hour shift will be inserted in order (other values are copied from an existing row of this employee).

### normalize: ***bool***
Whether to normalize hours worked on a given day. The mean and standard deviation of normalized columns are stored in the info object.

### prev_shifts: ***int***
If nonzero: How many days of shifts before the current day to include in each row. For example, the column `t_7` would refer to how many hours the employee worked 7 days ago. Rows for which there is no data `prev_shifts` days ago are dropped.

### day_of_week: ***bool***
Whether to add a column for the day of the week, where Monday=0, Sunday=6.

### force_reload: ***bool***
If true, raw data will be loaded and preprocessing will be performed even if preprocessed data/info files with the same parameters already exist.

### verbose: ***bool***
Whether to print out progress messages.

### fac_data: ***bool***
Whether to generate columns for the following facility level characteristics: 'nresid', 'multifac', 'profit', 'avg_dailycensus', 'sd_dailycensus'.
Can optionally set this flag to false for quicker dataset generation

### Example usage:
```py
from data_prep import initial_preprocess

RAW_DATA_PATH = '/export/storage_adgandhi/PBJ_data_prep/pbj_full.csv'
PREPROCESSED_DIR = '/export/storage_adgandhi/PBJ_data_prep/prepped/'

df, info = initial_preprocess(
    RAW_DATA_PATH, PREPROCESSED_DIR,
    nrows=None,
    fill_missing_shifts=True,
    normalize=True,
    day_of_week=True,
    prev_shifts=30,
    fac_data=True
)
```
