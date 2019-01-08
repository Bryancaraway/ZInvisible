# quickPlots.sh
# Caleb J. Smith
# October 11, 2018

# Intended for quick use and multiple plots (not on condor)

# Outline

# Compile MakePlots
# Remove existing plots
# Save root files for different MC
# hadd the results
# Make plots of results (containing both MC)


# Compile MakePlots

echo " - Compile MakePlots"
make -j8

retVal=$?
if [ $retVal -ne 0 ]; then
    echo "ERROR: There was an error when compiling MakePlots"
    exit $retVal;
fi

# Remove existing plots

echo " - Remove existing plots"
rm plots/*

# Save root files for different MC

echo " - Save root files for different MC"
n_events=1000
#sample1=GJets_HT-400To600
sample1=DYJetsToLL_HT_400to600
sample2=ZJetsToNuNu_HT_400to600
output1=""$sample1".root"
output2=""$sample2".root"

echo " - Running makePlots to create $output1"
./makePlots -D $sample1 -E $n_events -I $output1 | grep -v LHAPDF
echo " - Running makePlots to create $output2"
./makePlots -D $sample2 -E $n_events -I $output2 | grep -v LHAPDF

# hadd the results

# IMPORTANT: you need quotes around each variable (due to underscores)
# otherwise bash interprets the underscores as part of the variable names
result=""$sample1"_and_"$sample2".root"
echo " - hadd the results to create $result"
# Use -f to overwrite target file if it already exists
hadd -f $result $output1 $output2

# Make plots of results (containing both MC)

echo " - Make plots of results (containing both MC)"
./makePlots -f -I $result | grep -v LHAPDF


