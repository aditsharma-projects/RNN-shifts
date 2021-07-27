do "$pbjmlcode/make_globals.do"

capture log close
clear all
clear frames
set more off, permanently

if "$test"=="1" local _test = "_test"
display "Testing global is $test and _test =`_test'"

log using "$logdir/make_variable_length_sequences`_test'.log", replace

set seed 2021
use if (hours != 0) & (year(date)<=2019) using "$pbjmldir/Data/Raw/PBJ/pbj_full`_test'.dta", clear

* keeping all the nurses
keep if inlist(job_title, ///
	"Certified Nurse Aide":job_title_label, ///
	"Nurse Aide in Training":job_title_label, ///
	"Medication Aide/Technician":job_title_label, ///
	"Licensed Practical/Vocational Nurse":job_title_label, ///
	"Licensed Practical/Vocational Nurse with Administrative Duties":job_title_label ///
	"Registered Nurse":job_title_label, ///
	"Registered Nurse Director of Nursing":job_title_label, ///
	"Registered Nurse with Administrative Duties":job_title_label)

*Group by RN, LPN, CNA
gen job_bin = .
replace job_bin = 1 if inlist(job_title, ///
	"Certified Nurse Aide":job_title_label, ///
	"Nurse Aide in Training":job_title_label, ///
	"Medication Aide/Technician":job_title_label)

replace job_bin = 2 if inlist(job_title, ///
	"Licensed Practical/Vocational Nurse":job_title_label, ///
	"Licensed Practical/Vocational Nurse with Administrative Duties":job_title_label)

replace job_bin = 3 if inlist(job_title, ///
	"Registered Nurse":job_title_label, ///
	"Registered Nurse Director of Nursing":job_title_label, ///
	"Registered Nurse with Administrative Duties":job_title_label)

lab def job_enc 1 "CNA" 2 "LPN" 3 "RN" 
lab values job_bin job_enc
drop job_title // drop the more disaggregated job definition
rename job_bin job
assert inlist(job, "CNA":job_enc, "LPN":job_enc, "RN":job_enc)



* Assign dominant job and payroll type for the day.
gsort prov_id employee_id date hours
gcollapse (sum) hours (lastnm) job pay_type, ///
	by(prov_id employee_id date) labelformat(#sourcelabel#)

display "Possibly could add variables for dominant job and pay-type for each individual."


* Don't tsfill if two shifts are 15 days apart
bysort prov_id employee_id (date): gen spell_start = _n==1 
bysort prov_id employee_id (date): replace spell_start = (date - date[_n-1] > 14) if _n!=1
bysort prov_id employee_id (date): gen work_spell = sum(spell_start)

* filling the missing characteristics values of filled values
gegen spell_id = group(prov_id employee_id work_spell)
drop spell_start work_spell
qui compress
tsset spell_id date
tsfill

*Fill in missing values for prov_id employee_id job pay_type
foreach vv in prov_id employee_id job pay_type{
    bysort spell_id (`vv'): replace `vv' = `vv'[1] if missing(`vv')
}
replace hours = 0 if missing(hours) 
assert !missing(prov_id,employee_id,job,pay_type, hours)


* Set panel ID to be individual-level. No longer need spell-ID
drop spell_id
gegen long panel_id = group(prov_id employee_id)
qui compress panel_id
tsset panel_id date 


*Average of employees worked any hours over last lagged 7 days.
gegen employees = count(hours != 0), by(prov_id date job) //Since 0s are imputed
gen avg_employees_7days = round(  ( cond(!missing(L.employees),L.employees,0) + cond(!missing(L2.employees),L2.employees,0) ///
    + cond(!missing(L3.employees),L3.employees,0) + cond(!missing(L4.employees),L4.employees,0) ///
    + cond(!missing(L5.employees),L5.employees,0) + cond(!missing(L6.employees),L6.employees,0) ///
    + cond(!missing(L7.employees),L7.employees,0) ) ///
    / ( !missing(L.employees) + !missing(L2.employees) + !missing(L3.employees) + !missing(L4.employees) ///
    + !missing(L5.employees) + !missing(L6.employees) + !missing(L7.employees) ) , .01)
gen Lemployees = L.employees
drop employees

*Generate week. Will use to merge in variables later.
gen week = wofd(date)
format week %tw



************************************************************************************
*** cumulative hours of work each week till this week
************************************************************************************
frame put panel_id hours week date, into(week_cum_hours)
frame change week_cum_hours

forvalues dd=0/6{
	gen hours_dow`dd' = hours if dow(date)==`dd'
}

drop date
gcollapse (sum) hours*, by(panel_id week)
qui compress

* total cumulative hours up to last week
bysort panel_id (week): gen cum_hours_last_week = sum(hours)  - hours
forvalues dd=0/6{
	bysort panel_id (week): gen pct_hours_dow`dd' = round((sum(hours_dow`dd')  - hours_dow`dd')/cum_hours_last_week,.01)
	qui compress pct_hours_dow`dd'
}
drop hours* cum_hours_last_week

tempfile weekly_hours
save `weekly_hours'
***********************************************************************************


frame change default 
frame drop week_cum_hours
merge m:1 panel_id week using `weekly_hours', nogen
drop week


gen pct_hours_yesterday = .
gen pct_hours_today = .
gen pct_hours_tomorrow = .
gen day_of_week = dow(date)
forvalues dd=0/6{
	local yesterday = mod(`dd'-1,7) 
	local today = mod(`dd',7) 
	local tomorrow = mod(`dd'+1,7) 
	replace pct_hours_yesterday = pct_hours_dow`yesterday' if day_of_week==`dd'
	replace pct_hours_today = pct_hours_dow`today' if day_of_week==`dd'
	replace pct_hours_tomorrow = pct_hours_dow`tomorrow' if day_of_week==`dd'

	display "Yesterday: `yesterday'. Today: `today'. Tomorrow: `tomorrow'"
}

rename pct_hours_dow0 pct_hours_sunday
rename pct_hours_dow1 pct_hours_monday
rename pct_hours_dow2 pct_hours_tuesday
rename pct_hours_dow3 pct_hours_wednesday
rename pct_hours_dow4 pct_hours_thursday
rename pct_hours_dow5 pct_hours_friday
rename pct_hours_dow6 pct_hours_saturday


bysort panel_id: gen sequence_length = _N
bysort panel_id (date): gen date_index = _n-1 //Uses 0-indexing to make this easy for Python.


*Used for sampling.
gen randnum=runiform()
bysort panel_id (date): replace randnum = randnum[1] //forces randnum to be same within individual


qui compress
save "$exportdir/VariableLengths/full_sequences`_test'.dta", replace


*******************************************************************
*Generate coordinates for full sequences.
*******************************************************************
frame put panel_id, into(fr_coordinates)
frame change fr_coordinates
gen rownum=_n-1
gcollapse (first) first_index=rownum (last) last_index=rownum, by(panel_id)
qui compress
save "$exportdir/VariableLengths/full_coordinates`_test'.dta", replace
frames reset
********************************************************************




*******************************************************************
*Save 10% sample for training.
*******************************************************************
use if randnum<.1 using "$exportdir/VariableLengths/full_sequences`_test'.dta", clear
drop randnum
export delimited "$exportdir/VariableLengths/train10_sample_sequences`_test'.csv", replace
frame put panel_id, into(fr_coordinates)
frame change fr_coordinates
gen rownum=_n-1
gcollapse (first) first_index=rownum (last) last_index=rownum, by(panel_id)
qui compress
export delimited "$exportdir/VariableLengths/train10_sample_coords`_test'.csv", replace
frames reset
*******************************************************************




*******************************************************************
*Save 5% sample for training.
*******************************************************************
use if (randnum>=.1) & (randnum<.15) using "$exportdir/VariableLengths/full_sequences`_test'.dta", clear
drop randnum
export delimited "$exportdir/VariableLengths/val5_sample_sequences`_test'.csv", replace
frame put panel_id, into(fr_coordinates)
frame change fr_coordinates
gen rownum=_n-1
gcollapse (first) first_index=rownum (last) last_index=rownum, by(panel_id)
qui compress
export delimited "$exportdir/VariableLengths/val5_sample_coords`_test'.csv", replace
frames reset
*******************************************************************





*******************************************************************
*Save 50% A sample
*******************************************************************
use if randnum<.5 using "$exportdir/VariableLengths/full_sequences`_test'.dta", clear
drop randnum
export delimited "$exportdir/VariableLengths/full5050A_sequences`_test'.csv", replace
frame put panel_id, into(fr_coordinates)
frame change fr_coordinates
gen rownum=_n-1
gcollapse (first) first_index=rownum (last) last_index=rownum, by(panel_id)
qui compress
export delimited "$exportdir/VariableLengths/full5050A_coords`_test'.csv", replace
frames reset
*******************************************************************


*******************************************************************
*Save 50% B sample
*******************************************************************
use if randnum >=.5 using "$exportdir/VariableLengths/full_sequences`_test'.dta", clear
drop randnum
export delimited "$exportdir/VariableLengths/full5050B_sequences`_test'.csv", replace
frame put panel_id, into(fr_coordinates)
frame change fr_coordinates
gen rownum=_n-1
gcollapse (first) first_index=rownum (last) last_index=rownum, by(panel_id)
qui compress
export delimited "$exportdir/VariableLengths/full5050B_coords`_test'.csv", replace
frames reset
*******************************************************************