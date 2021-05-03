# Basic Workflow design

### Overview of code design
General idea is to take the mixed data(each entry corresponds to a different employee) in pbj_full.csv and produce a more ordered set of data(where each entry correspond to all possible information about a particular employee). From this processed version of the data we can easily generate different permutations of input variables as well as large amounts of training samples per employee (using the sliding window code in window_generator.py you could, for example create ~64+ different sequences of length 7 just from one employee) without much computational effort. 

All of the data preprocessing code is in data_load.py, and running the file will begin producing csv's (data_load.py reads in 100,000 lines from pbj_full.csv at a time)

### Pipeline stages
1) Raw csv (each row is a random entry corresponding to one employee's hours on one day) "/export/storage_adgandhi/PBJ_data_prep/pbj_full.csv"
2) Preprocessed csv (each row contains all of an employee's shifts for one quarter + additional variables) "/users/facsupport/asharma/Data/Preprocessed/tmp/ONE.csv"

      <ins>Preprocessing steps (included are functions in data_load.py to pay attention to</ins>
      
        1. Sort raw csv by employee id to get contiguous "blocks" of data for each employee -- process_Data()
        2. Sort each "block" of employee data by date -- get_entry()
        3. Impute 0's between the start and end date to produce a sequence of shifts -- get_entry()
        4. Line up the generated sequence with a particular year and quarter. For example, if a particular employee's shift data is as follows: 
        Jan 2: 8 hours  ... March 30: 7.5 hours April 1: 8 hours April 2: 8 hours
        we will only consider the sequence of shifts in quarter 1 (Jan 1 - March 31) and impute missing entries as 0 to get a 90 day sequence of the shifts 
        for that particular quarter. So the sequence would be [0,8,...,7.5] -- one_quarter()
        5. Using the computed year and quarter, load the corresponding facility data and extract the data entries corresponding to the current employee's facility id
           and job id. Append this to the existing sequence (which is now length 180) -- get_facility_data()
        6. Append additional descriptors to the sequence -- get_entry()
    For each employee we generate a 191 length sequence of data and save to a csv
        
3) Load preprocessed csv into data frame, splice the data however you want (by selecting the column labels of interest), construct tf Dataset object, feed into model 
