import delimited "$pbjmldir/Data/Predictions/RNN_60days/Apred.csv", clear
rename date date_string 
gen date = date(date_string, "DMY")
format date %td
drop date_string
tempfile tf_A
save `tf_A'


import delimited "$pbjmldir/Data/Predictions/RNN_60days/Bpred.csv", clear
rename date date_string 
gen date = date(date_string, "DMY")
format date %td
drop date_string
tempfile tf_B
save `tf_B'


clear
use in 1 using "$pbjmldir/Data/Raw/PBJ/pbj_full.dta", clear
drop if _n>0
append using `tf_A'
append using `tf_B'

qui compress
save "$pbjmldir/Data/Predictions/RNN_60days/predictions.dta", replace
