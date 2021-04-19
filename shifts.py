import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# ========== SETTINGS ==========

# Data input file
FILE = '/users/facsupport/asharma/Data/pbj_full.csv'
ROWS_TO_READ = 10000

PREPROCESSED_FILE = 'data/preprocessed.csv'
FORCE_RELOAD_DATA = True

# Weights to split data set
TRAINING_WEIGHT = 0.7
VALIDATION_WEIGHT = 0.2
TEST_WEIGHT = 0.1

# For model training
MAX_EPOCHS = 20

VERBOSE_TRAINING = 1

tf.get_logger().setLevel('INFO') # todo: investigate warnings


# ## Loading and Preprocessing

df = None

if PREPROCESSED_FILE and not FORCE_RELOAD_DATA:
    try:
        print("Loading preprocessed data...")
        df = pd.read_csv(PREPROCESSED_FILE)
    except FileNotFoundError:
        print("Failed.")        

if df is None:
    print("Loading data...")
    raw_df = pd.read_csv(FILE, nrows = ROWS_TO_READ, dtype={'hours':'float64'}, parse_dates = ['date'])
    
    print("Preprocessing data (this may take a few minutes)...")

    # Partition by employee id
    grouped = raw_df.groupby('employee_id')

    partitions = []
    for name, group in grouped:
        # Sort by date
        group = group.sort_values(by='date')

        # Pad empty rows with 0 hours

        day = min(group['date'])
        last = None
        i = 0
        orig_length = len(group)
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

        group['date_int'] = group['date'].astype(np.int64)

        # Sort by date with new rows
        group = group.sort_values(by='date')

        partitions.append(group)

    df = pd.concat(partitions)
    
    print("Saving preprocessed data...")
    df.to_csv(PREPROCESSED_FILE, index=False)

# We do "destructive" preprocessing (can't recover std, mean from normalized data) in local memory

# Normalize features
means = {}
stds = {}
#df = df.filter(['hours', 'date_int'])
for col in ['date_int', 'hours']:
    means[col] = df[col].mean()
    stds[col] = df[col].std()
    df[col] = (df[col] - means[col]) / stds[col]

# Split data into training/validation/test sets
n = len(df)
weights_sum = TRAINING_WEIGHT + VALIDATION_WEIGHT + TEST_WEIGHT
split1 = int(TRAINING_WEIGHT / weights_sum * n)
split2 = int((TRAINING_WEIGHT + VALIDATION_WEIGHT) / weights_sum * n)
train_df = df[:split1]
val_df = df[split1:split2]
test_df = df[split2:]

class WindowGenerator():
    def __init__(self, input_width, label_width, shift, train_df = train_df, val_df = val_df, test_df = test_df, label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
          
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
          
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
          
        self.total_window_size = input_width + shift
          
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
          
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        # if self.label_columns is not None:
        #     labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns],axis=-1)
        
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels

    def plot(self, model=None, plot_col='hours', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)
        
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            
            if label_col_index is None:
                continue
          
            plt.scatter(self.label_indices, labels[n, :, label_col_index], edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)
          
            if n == 0:
                plt.legend()
        
        plt.xlabel('x')

    @tf.autograph.experimental.do_not_convert
    def make_dataset(self, data):
        concat_ds = None
        grouped = data.groupby('employee_id')
        for name, group in grouped:
            group = group.filter(['hours'])
            data = np.array(group, dtype=np.float32)
            if len(data) == 1:
                continue
            ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=False,
                batch_size=32,)
    
            ds = ds.map(self.split_window)
            
            if concat_ds is None:
                concat_ds = ds
            else:
                concat_ds = concat_ds.concatenate(ds)
                
        return concat_ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


# ## Training and Evaluation

val_performance = {}
performance = {}
column_indices = {name: i for i, name in enumerate(df.columns)}

# Windows
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['hours'])

wide_window = WindowGenerator(
    input_width=7, label_width=1, shift=1,
    label_columns=['hours'])

LABEL_WIDTH = 24
CONV_WIDTH = 3
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['hours'])

wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['hours'])

def compile_and_fit(model, window, patience=3, verbose=0):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping],
                      verbose=verbose)
    return history

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(64, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])




dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

print()
print("Training dense model.")
history = compile_and_fit(dense, wide_window, verbose=VERBOSE_TRAINING)

print("Evaluating dense model.")
val_performance['Dense'] = dense.evaluate(wide_window.val, verbose=VERBOSE_TRAINING)
performance['Dense'] = dense.evaluate(wide_window.test, verbose=0)





print("Training LSTM model.")
history = compile_and_fit(lstm_model, wide_window, verbose=VERBOSE_TRAINING)

print("Evaluating LSTM model.")
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val, verbose=VERBOSE_TRAINING)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)











if False: # manual window creation
    print("Training LSTM model.")
    inputs_map = {}
    labels_map = {}

    for data, data_name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        grouped = data.groupby('employee_id')

        inputs_size = 7
        labels_size = 1
        total_window_size = inputs_size + labels_size

        inputs = []
        labels = []

        for name, group in grouped:
            group = group.filter(['name', 'hours'])
            data = np.array(group, dtype=np.float32)
            for i in range(0, len(data) - total_window_size + 1):
                inputs.append(data[i:i+inputs_size])
                labels.append(data[i+inputs_size : i+total_window_size])

        inputs = np.array(inputs)
        labels = np.array(labels)

        lstm_model.compile(loss=tf.losses.MeanSquaredError(),
                            optimizer=tf.optimizers.Adam(),
                            metrics=[tf.metrics.MeanAbsoluteError()])
        
        inputs_map[data_name] = labels
        labels_map[data_name] = labels


    history=lstm_model.fit(inputs_map['train'], labels_map['train'])