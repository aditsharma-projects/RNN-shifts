# Notebook Overview
Testing effect of recurrence length on prediction performance
Inputs
The input variables for the RNN notebook are the following:

1. Time series data -- Variable
2. Additional predictors -- day_of_week, avg_employees, perc_hours_today_before,
   perc_hours_yesterday_before, perc_hours_tomorrow_before

##Model Architecture
The architecture used in this notebook combines an RNN with a feed forward neural network. The RNN layer recieves n days of lagged shift data and makes a prediction for the shift on the n+1th day. This prediction is then concatenated with the additional predictors and fed into a traditional neural network(various shapes, see log files for all tested archectures) to generate a better prediction--the idea being that the RNN (through it's long and short term memory) learns patterns over time and the FF network adjusts these patterns based on additional information.

![Unconditioned RNN diagram](README_assets/Unconditioned_RNN_Diagram.jpeg)

