# Data Preprocessing 

## make_variable_length_sequences.do

This code creates the variable length sequences. It's copied over from the `PBJ-Prediction/PBJMLcode` repo. Note that you must set the globals for `pbjmlcode` and `pbjmldir` in your `profile.do`. Ashvin's are:
```
global pbjmlcode "/mnt/faculty/adgandhi/RNN-shifts"
global pbjmldir "/export/storage_adgandhi/PBJhours_ML"
```

## rectangularize_sequences.do

This code creates a rectangularized version of the variable-length sequences. The first input to the do file is used as the number of days lagged-values to include.  Thus, `rectangularize_sequences.do` will include 60 days of lagged values.