from data_prep import initial_preprocess

RAW_DATA_PATH = '/export/storage_adgandhi/PBJ_data_prep/pbj_full.csv'
PREPROCESSED_DIR = '/export/storage_adgandhi/PBJ_data_prep/prepped/'

df, info = initial_preprocess(
    RAW_DATA_PATH, PREPROCESSED_DIR,
    nrows=140801548, # 10%
    fill_missing_shifts=True,
    normalize=True,
    day_of_week=True,
    prev_shifts=30
)