use "/export/storage_adgandhi/PBJhours_ML/Data/Raw/PBJ/pbj_full.dta"
log using labels.txt, replace t
label list
log close
label drop _all
outsheet using "/export/storage_adgandhi/PBJ_data_prep/pbj_full.csv" , comma
