global cleandir "$pbjmldir/Data/Clean"
global intermediatedir "$pbjmldir/Data/Intermediate"
global tempdir "$pbjmldir/Data/Temp"
global exportdir "/export/storage_adgandhi/PBJhours_ML/Data/Intermediate"
global logdir "$pbjmldir/logs"

* code path 
global cleancode "$pbjmlcode/Clean"
global analysiscode "$pbjmlcode/Analysis"

local required_ados "gtools todate winsor2 vallabsave distinct rangestat unique distinct ftools randomtag randomize" //add the required ados here//
foreach x in `required_ados' {
	capture findfile `x'.ado
	if _rc==601 {
		ssc install `x'
		if "`x'"=="gtools"{
			gtools, upgrade
		}
	}
	else{
		display "`x' already installed."
	}
}