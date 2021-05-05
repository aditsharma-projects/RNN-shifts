# ========================================
#               LOAD DATA
# ========================================
import pandas as pd
import lightgbm as lgb
from data_prep import initial_preprocess

ROWS_TO_READ = 1000
RAW_DATA_PATH = '/export/storage_adgandhi/PBJ_data_prep/pbj_full.csv'
PREPROCESSED_DIR = '/export/storage_adgandhi/PBJ_data_prep/prepped/'

df, info = initial_preprocess(
    RAW_DATA_PATH, PREPROCESSED_DIR,
    nrows=ROWS_TO_READ,
    fill_missing_shifts=True,
    normalize=True,
    day_of_week=True,
    prev_shifts=30
)

# Keep "interesting" jobs (TODO - use strings instead)
df = df[df['job_title'].isin([33,34,35,11,12,3,5,16,17])]
df['date'] = pd.to_datetime(df['date']).astype(int)

# %%
# ========================================
#          SPLIT & PREP DATAFRAME
# ========================================

inputs = df.drop(['t_0', 'hours'], axis=1)
labels = df.filter(['t_0'])

# Weights to split data set
TRAINING_WEIGHT = 0.7
VALIDATION_WEIGHT = 0.2
TEST_WEIGHT = 0.1

n = len(df)
weights_sum = TRAINING_WEIGHT + VALIDATION_WEIGHT + TEST_WEIGHT
split1 = int(TRAINING_WEIGHT / weights_sum * n)
split2 = int((TRAINING_WEIGHT + VALIDATION_WEIGHT) / weights_sum * n)

train_inputs, train_labels = inputs[:split1], labels[:split1]
val_inputs, val_labels = inputs[split1:split2], labels[split1:split2]
test_inputs, test_labels = inputs[split2:], labels[split2:]

print(train_inputs)
print(train_labels)

# %%
# ========================================
#          DATAFRAME TO LGB DS
# ========================================

cats = ['job_title', 'prov_id', 'pay_type', 'day_of_week']

train_data = lgb.Dataset(train_inputs, label=train_labels, categorical_feature=cats)
val_data = lgb.Dataset(val_inputs, label=val_labels, categorical_feature=cats)
test_data = lgb.Dataset(test_inputs, label=test_labels, categorical_feature=cats)
print(train_data)

# %%
# ========================================
#            TRAIN WITH LGB
# ========================================
param = {
  'num_leaves': 100,
  'learning_rate': 0.1,
  'metric': 'mse',
}
evals_result = {}
bst = lgb.train(param, train_data, valid_sets=[val_data], evals_result=evals_result)