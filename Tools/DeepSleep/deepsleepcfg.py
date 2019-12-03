################
## Config for ##
## Deep Sleep ##
################

master_file_path  = './files/'
# Overhead #
file_path         = master_file_path+'root/'
files             = ['result_2016','result_2017','result_2018_AB','result_2018_CD']
tree_dir          = 'Training'
MCsamples         = ['TTZ','DY','TTX','DiBoson','TTBarLep']
skim_dir          = master_file_path+'skim/'
skim_kinemFit_dir = master_file_path+'skim_kinemFit/'
# Train Overhead #
train_dir      = master_file_path+'train/'
train_over_dir = master_file_path+'train_overSample/'
test_dir       = master_file_path+'test/'
val_dir        = master_file_path+'val/'
###################
# Input Variables #
ak4vars = ['Jet_btagCSVV2_drLeptonCleaned','Jet_btagDeepB_drLeptonCleaned','Jet_qgl_drLeptonCleaned']
ak4lvec = ['JetTLV_drLeptonCleaned']
ak8vars = ['FatJet_tau1_drLeptonCleaned','FatJet_tau2_drLeptonCleaned','FatJet_tau3_drLeptonCleaned','FatJet_tau4_drLeptonCleaned','FatJet_deepTag_WvsQCD_drLeptonCleaned','FatJet_deepTag_TvsQCD_drLeptonCleaned','FatJet_deepTag_ZvsQCD_drLeptonCleaned','FatJet_msoftdrop_drLeptonCleaned','FatJet_mass_drLeptonCleaned']
ak8lvec = ['FatJetTLV_drLeptonCleaned']
genvars = ['passGenCuts','isZToLL']
valvars = ['nResolvedTops_drLeptonCleaned','nMergedTops_drLeptonCleaned','nBottoms_drLeptonCleaned','nJets30_drLeptonCleaned','bestRecoZPt','passElecZinvSelOnZMassPeak','passMuZinvSelOnZMassPeak','genWeight','weight']
label   = ['isTAllHad']
# Derived Varialbes #
ak4comb = 'true'
ak8comb = 'true'
# Model Hyperparams #
epochs        = 300
alpha         = 0.0001
batch_size    = int(32768/4)
hiddenl       = 3 
NNoutputDir   = './NN_ouput/'
NNoutputName  = 'sixth_try.h5' 
NNmodel1Name  = 'sixth_model.h5'
#NNOUTPUTNAME  = 'fifth_try.h5' 
#NNmodel1Name  = 'fifth_model.h5'
useWeights    = True
plotHist      = True
