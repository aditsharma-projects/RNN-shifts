do "$pbjmlcode/make_globals.do"

if "`1'" != "" local num_lags = `1'
else local num_lags = 60

capture log close
clear all
clear frames
set more off, permanently

if "$test"=="1" local _test = "_test"
display "Testing global is $test and _test =`_test'"

log using "$logdir/rectangularize_sequences_`num_lags'`_test'.log", replace






*******************************************************************
*Save 10% sample for tuning.
*******************************************************************
use if randnum<.1 using "$exportdir/VariableLengths/full_sequences`_test'.dta", clear
drop randnum
forvalues dd=1/`num_lags'{
    display "Generating lag `dd'"
    gen L`dd'_hours = L`dd'.hours
}
qui compress
export delimited "$exportdir/Length`num_lags'/train10_sample_sequences`_test'.csv", replace
*******************************************************************




*******************************************************************
*Save 5% sample for tuning.
*******************************************************************
use if (randnum>=.1) & (randnum<.15) using "$exportdir/VariableLengths/full_sequences`_test'.dta", clear
drop randnum
forvalues dd=1/`num_lags'{
    display "Generating lag `dd'"
    gen L`dd'_hours = L`dd'.hours
}
qui compress
export delimited "$exportdir/Length`num_lags'/val5_sample_sequences`_test'.csv", replace
*******************************************************************



/*
*******************************************************************
*Save 50% A sample
*******************************************************************
use if randnum<.5 using "$exportdir/VariableLengths/full_sequences`_test'.dta", clear
drop randnum
forvalues dd=1/`num_lags'{
    display "Generating lag `dd'"
    gen L`dd'_hours = L`dd'.hours
}
qui compress
export delimited "$exportdir/Length`num_lags'/full5050A_sequences`_test'.csv", replace
*******************************************************************


*******************************************************************
*Save 50% B sample
*******************************************************************
use if randnum >=.5 using "$exportdir/VariableLengths/full_sequences`_test'.dta", clear
drop randnum
forvalues dd=1/`num_lags'{
    display "Generating lag `dd'"
    gen L`dd'_hours = L`dd'.hours
}
qui compressexport delimited "$exportdir/Length`num_lags'/full5050B_sequences`_test'.csv", replace
*******************************************************************

*/