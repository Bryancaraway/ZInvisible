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
# Kinematic Fit sub cfg args
kinemFitCfg    = (['result_2017'], 
                  ['TTZ','DY'], 
                  skim_kinemFit_dir)
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
ak4lvec = {'TLV'    :['JetTLV'+LC],
           'VarsLC' :['Jet_pt'+LC, 'Jet_eta'+LC, 'Jet_phi'+LC, 'Jet_E'+LC],
           'TLVars' :['Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_E']}
#
ak8vars = ['FatJet_tau1'+LC,'FatJet_tau2'+LC,'FatJet_tau3'+LC,'FatJet_tau4'+LC,'FatJet_deepTag_WvsQCD'+LC,'FatJet_deepTag_TvsQCD'+LC,'FatJet_deepTag_ZvsQCD'+LC,'FatJet_msoftdrop'+LC,'FatJet_mass'+LC]
ak8lvec = {'TLV'    :['FatJetTLV'+LC],
           'VarsLC' :['FatJet_pt'+LC, 'FatJet_eta'+LC, 'FatJet_phi'+LC, 'FatJet_E'+LC],
           'TLVars' :['FatJet_pt', 'FatJet_eta', 'FatJet_phi', 'FatJet_E']}
#
genpvars   = ['GenPart_pt', 'GenPart_eta', 'GenPart_phi', 'GenPart_E', 'GenPart_status', 'GenPart_pdgID', 'GenPart_genPartIdxMother']
genLevCuts = ['passGenCuts','isZToLL']
valvars    = ['nResolvedTops'+LC,'nMergedTops'+LC,'nBottoms'+LC,'nJets30'+LC,'bestRecoZPt','passElecZinvSelOnZMassPeak','passMuZinvSelOnZMassPeak','genWeight','weight']
label      = ['isTAllHad']
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
