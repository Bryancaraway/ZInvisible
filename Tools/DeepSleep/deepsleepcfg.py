################
## Config for ##
## Deep Sleep ##
################

# Overhead #
file_path = '/uscms/home/bcaraway/nobackup/analysis/CMSSW_10_2_9/src/ZInvisible/Tools/DeepSleep/root/'
files     = ['result_2016','result_2017','result_2018_AB','result_2018_CD']
tree_dir  = 'Training'
MCsamples = ['TTZ','DY','TTX','DiBoson','TTBarLep']
skim_dir  = '/uscms/home/bcaraway/nobackup/analysis/CMSSW_10_2_9/src/ZInvisible/Tools/DeepSleep/skim/'
# Train Overhead #
train_dir = '/uscms/home/bcaraway/nobackup/analysis/CMSSW_10_2_9/src/ZInvisible/Tools/DeepSleep/train/'
test_dir  = '/uscms/home/bcaraway/nobackup/analysis/CMSSW_10_2_9/src/ZInvisible/Tools/DeepSleep/test/'
val_dir   = '/uscms/home/bcaraway/nobackup/analysis/CMSSW_10_2_9/src/ZInvisible/Tools/DeepSleep/val/'
###################
# Input Variables #
ak4vars = ['Jet_btagCSVV2_drLeptonCleaned','Jet_btagDeepB_drLeptonCleaned','Jet_qgl_drLeptonCleaned']
ak4lvec = ['JetTLV_drLeptonCleaned']
ak8vars = ['FatJet_tau1_drLeptonCleaned','FatJet_tau2_drLeptonCleaned','FatJet_tau3_drLeptonCleaned','FatJet_tau4_drLeptonCleaned','FatJet_deepTag_WvsQCD_drLeptonCleaned','FatJet_deepTag_TvsQCD_drLeptonCleaned','FatJet_deepTag_ZvsQCD_drLeptonCleaned','FatJet_msoftdrop_drLeptonCleaned','FatJet_mass_drLeptonCleaned']
ak8lvec = ['FatJetTLV_drLeptonCleaned']
genvars = ['passGenCuts','isZToLL']
valvars = ['nResolvedTops_drLeptonCleaned','nMergedTops_drLeptonCleaned','nBottoms_drLeptonCleaned','nJets30_drLeptonCleaned','bestRecoZpt_drLeptonCleaned','passElecZinvSelOnZMassPeak','passMuZinvSelOnZMassPeak','genWeight']
label   = ['isTAllHad']
# Derived Varialbes #
ak4comb = 'true'
ak8comb = 'true'
# Model Hyperparams #
epochs        = 100
alpha         = 0.0001
batch_size    = 32768
hiddenl       = 3 
NNoutputDir   = '/uscms/home/bcaraway/nobackup/analysis/CMSSW_10_2_9/src/ZInvisible/Tools/DeepSleep/NN_ouput/'
NNoutputName  = 'first_try.h5' 
