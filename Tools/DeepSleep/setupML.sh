

mkdir NN_ouput val test train train_overSample skim skim_kinemFit root  pdf 

voms-proxy-init --rfc --voms cms

xrdcp -v root://cmseos.fnal.gov//store/user/bcaraway/skimAnaSamples/result_2016.root root/.
xrdcp -v root://cmseos.fnal.gov//store/user/bcaraway/skimAnaSamples/result_2017.root root/.
