
mkdir files
mkdir NN_output 
mkdir files/val 
mkdir files/test 
mkdir files/train 
mkdir files/train_overSample 
mkdir files/skim 
mkdir files/skim_kinemFit 
mkdir files/root  
mkdir pdf 

voms-proxy-init --rfc --voms cms

xrdcp -v root://cmseos.fnal.gov//store/user/bcaraway/skimAnaSamples/result_2016.root root/.
xrdcp -v root://cmseos.fnal.gov//store/user/bcaraway/skimAnaSamples/result_2017.root root/.
