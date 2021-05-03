# Basic Workflow design
1) Raw csv (each row is a random entry corresponding to one employee's hours on one day)
2) Preprocessed csv (each row contains all of an employee's shifts for one quarter + additional variables)
    a) Preprocessing steps
        i) Sort raw csv by employee id to get contiguous "blocks" of data for each employee
        ii) Sort each "block" of employee data by date
        iii) Impute 0's between the start and end date to produce a sequence of shifts
        iv) Line up the generated sequence with a particular year and quarter. For example, if a particular employee's shift data is as follows: 
        Jan 2: 8 hours  ... March 30: 7.5 hours April 1: 8 hours April 2: 8 hours
        we will only consider the sequence of shifts in quarter 1 (Jan 1 - March 31) and impute missing entries as 0 to get a 90 day sequence of the shifts 
        for that particular quarter. So the sequence would be [0,8,...,7.5]
        v) Using the computed year and quarter, load the corresponding facility data and extract the data entries corresponding to the current employee's facility id
           and job id. Append this to the existing sequence (which is now length 180)
        vi) Append additional descriptors to the sequence
    For each employee we generate a 191 length sequence of data and save to a csv
        
3) Load preprocessed csv into data frame, splice the data however you want (by selecting the column labels of interest), construct tf Dataset object, feed into model 
