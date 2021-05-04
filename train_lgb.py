# ========================================
#               LOAD DATA
# ========================================
import pandas as pd
import lightgbm as lgb

df = pd.read_csv("/users/facsupport/asharma/Data/Preprocessed/tmp/ONE.csv")
# Keep "interesting" jobs (TODO - use strings instead)
df = df[df['jobTitle'].isin([33,34,35,11,12,3,5,16,17])]

#%%
# ========================================
#          SPLIT & PREP DATAFRAME
# ========================================

# Drop unnamed column (TODO - export csv without this in the first place)
del df['Unnamed: 0']

# Drop 90th day of facility data
del df['f_89']

inputs = df.drop(['t_89'], axis=1)
labels = df.filter(['t_89'])

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

#%%
# ========================================
#          DATAFRAME TO LGB DS
# ========================================

cats = ['jobTitle', 'providerId', 'payType', 'dayOfWeek']

train_data = lgb.Dataset(train_inputs, label=train_labels, categorical_feature=cats)
val_data = lgb.Dataset(val_inputs, label=val_labels, categorical_feature=cats)
test_data = lgb.Dataset(test_inputs, label=test_labels, categorical_feature=cats)

#%%
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