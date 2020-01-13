################
## Config for ##
## Deep Sleep ##
################
import operator
#
master_file_path  = './files/'
# Overhead #
file_path         = master_file_path+'root/'
files             = ['result_2016','result_2017','result_2018_AB','result_2018_CD', 'result_2018']
tree_dir          = 'Training'
MCsamples         = ['TTZ','DY','TTX','DiBoson','TTBarLep']
skim_dir          = master_file_path+'skim/'
skim_kinemFit_dir = master_file_path+'skim_kinemFit/'
# Kinematic Fit sub cfg args
kinemFitCfg    = (['result_2017'], 
                  ['DY','TTZ'],
                  #['DY'],#['TTZ','DY','TTX', 'DiBoson', 'TTBarLep'], 
                  skim_kinemFit_dir)
kinemFitCut    = (operator.ge, 5)
kinemFitoverlap= 0
kinemFitMaxJets= 14
# Train Overhead #
train_dir      = master_file_path+'train/'
train_over_dir = master_file_path+'train_overSample/'
test_dir       = master_file_path+'test/'
val_dir        = master_file_path+'val/'
###################
# Input Variables #
LC = '_drLeptonCleaned'
#
ak4vars = ['Jet_btagCSVV2'+LC,'Jet_btagDeepB'+LC,'Jet_qgl'+LC]
ak4lvec = {'TLV'      :['JetTLV'+LC],
           'TLVarsLC' :['Jet_pt'+LC, 'Jet_eta'+LC, 'Jet_phi'+LC, 'Jet_E'+LC],
           'TLVars'   :['Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_E']}
#
ak8vars = ['FatJet_tau1'+LC,'FatJet_tau2'+LC,'FatJet_tau3'+LC,'FatJet_tau4'+LC,'FatJet_deepTag_WvsQCD'+LC,'FatJet_deepTag_TvsQCD'+LC,'FatJet_deepTag_ZvsQCD'+LC,'FatJet_msoftdrop'+LC,'FatJet_mass'+LC]
ak8lvec = {'TLV'      :['FatJetTLV'+LC],
           'TLVarsLC' :['FatJet_pt'+LC, 'FatJet_eta'+LC, 'FatJet_phi'+LC, 'FatJet_E'+LC],
           'TLVars'   :['FatJet_pt', 'FatJet_eta', 'FatJet_phi', 'FatJet_E']}
#
genpvars   = ['GenPart_pt', 'GenPart_eta', 'GenPart_phi', 'GenPart_E', 'GenPart_status', 'GenPart_pdgID', 'GenPart_genPartIdxMother']
genLevCuts = ['passGenCuts','isZToLL']
valvars    = ['nResolvedTops'+LC,'nMergedTops'+LC,'nBottoms'+LC,'nJets30'+LC,'bestRecoZPt','passElecZinvSelOnZMassPeak','passMuZinvSelOnZMassPeak','genWeight','weight']
valRCvars  = ['ResolvedTopCandidate_discriminator', 'ResolvedTopCandidate_j1Idx', 'ResolvedTopCandidate_j2Idx', 'ResolvedTopCandidate_j3Idx']
label      = ['isTAllHad']
# Derived Varialbes #
ak4comb = 'true'
ak8comb = 'true'
# Model Hyperparams #
epochs        = 300
alpha         = 0.001
batch_size    = int(32768/4)
hiddenl       = 3 
NNoutputDir   = './NN_ouput/'
NNoutputName  = 'sixth_try.h5' 
NNmodel1Name  = 'sixth_model.h5'
#NNOUTPUTNAME  = 'fifth_try.h5' 
#NNmodel1Name  = 'fifth_model.h5'
useWeights    = True
plotHist      = True

##### CNN backend prep #####
skim_cnn_dir  = master_file_path+'skim_cnn/'
cnn_data_dir  = (master_file_path+'train_cnn/', master_file_path+'test_cnn/', master_file_path+'val_cnn/')
cnnCut        = (operator.ge, 5)
cnnMaxJets    = 10
cnnProcessCfg = (files, MCsamples, skim_cnn_dir)
cnn_vars = ['btagCSVV2','btagDeepB', 'qgl', 'pt', 'eta', 'phi', 'E']
#
cnn_alpha      = 0.001
cnn_batch_size = 512
cnn_epochs     = 30
CNNoutputDir   = './CNN_output/'
CNNoutputName  = 'first_try.h5'
CNNmodelName   = 'first_model.h5'
