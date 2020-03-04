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
skim_Zinv_dir     = master_file_path+'skim_Zinv/'
skim_ZHbb_dir     = master_file_path+'skim_ZHbb/'
##############
sample_maxJets  = {'DiLep':{'DY':14, 'TTZ':14, 'TTX':14, 'TTBarLep':11, 'DiBoson':11, 'TriBoson':10},
                   'ZInv':{'WJets':14, 'ZJets':13, 'DiBoson':11, 'TriBoson':11, 'TTX':14, 'QCD':13, 
                           'TTBarHad':13, 'TTBarLep':14, 'TTZ':13 }}
# Kinematic Fit sub cfg args
kinemFitCfg    = (['result_2017'], 
                  [ 'TTX', 'DiBoson', 'TTBarLep', 'TriBoson', 'DY','TTZ'],
                  #['DY'],#['TTZ','DY','TTX', 'DiBoson', 'TTBarLep'], 
                  skim_kinemFit_dir)
kinemFitCut    = (operator.ge, 5)
kinemFitoverlap= 1
kinemFitMaxJets= 14
##### TTZ, Z to MET CONFIG #####
ZinvFitCfg    = (['result_2017'],
                 ['WJets','ZJets','DiBoson','TriBoson','TTX','QCD','TTBarHad','TTBarLep','TTZ'],
                 skim_Zinv_dir)
ZinvFitCut     = (operator.ge, 5)
ZinvFitoverlap = 0
ZinvFitMaxJets = 14
##### TTZ, Z to MET CONFIG #####
ZHbbFitCfg    = (['result_2017'],
                 #['WJets','ZJets','DY','DiBoson','TriBoson','TTX','QCD','TTBarHad','TTBarLep','TTZ/H'],
                 [ 'TTZH', 'QCD',  'TTX',  'DY',  'TTBarLep',  'WJets', 'TTBarHad',  'DiBoson',  'TriBoson',  'ZJets'],
                 skim_ZHbb_dir)
ZHbbFitCut    = (operator.ge, 0)
ZHbbFitoverlap = 0
ZHbbFitMaxJets = 14
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
ak8vars = ['FatJet_tau1'+LC,'FatJet_tau2'+LC,'FatJet_tau3'+LC,'FatJet_tau4'+LC,
           'FatJet_deepTag_WvsQCD'+LC,'FatJet_deepTag_TvsQCD'+LC,'FatJet_deepTag_ZvsQCD'+LC,
           'FatJet_msoftdrop'+LC,'FatJet_mass'+LC,'FatJet_btagDeepB'+LC,'FatJet_btagHbb'+LC,
           'FatJet_subJetIdx1'+LC,'FatJet_subJetIdx2'+LC,'SubJet_pt', 'SubJet_btagDeepB']
ak8lvec = {'TLV'      :['FatJetTLV'+LC],
           'TLVarsLC' :['FatJet_pt'+LC, 'FatJet_eta'+LC, 'FatJet_phi'+LC, 'FatJet_E'+LC],
           'TLVars'   :['FatJet_pt', 'FatJet_eta', 'FatJet_phi', 'FatJet_E']}
#
genpvars   = ['GenPart_pt', 'GenPart_eta', 'GenPart_phi', 'GenPart_E', 'GenPart_status', 'GenPart_pdgId', 'GenPart_genPartIdxMother']
genLevCuts = ['passGenCuts','isZToLL']
valvars    = ['nResolvedTops'+LC,'nMergedTops'+LC,'nBottoms'+LC,'nSoftBottoms'+LC,'nJets30'+LC,
              'bestRecoZPt', 'bestRecoZEta', 'bestRecoZPhi', 'bestRecoZM',
              'MET_phi', 'MET_pt', 'Lep_pt', 'Lep_eta', 'Lep_phi', 'Lep_E',
              'passElecZinvSelOnZMassPeak','passMuZinvSelOnZMassPeak','genWeight','weight']
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

##### DNN backend for Z/H -> bb #####
dnn_ZH_dir  = (master_file_path+'train_ZH/', master_file_path+'test_ZH/', master_file_path+'val_ZH/')
aux_ZH_dir  = master_file_path+'aux_ZH/'
# only event level variables
dnn_ZH_vars = [
    'max_lb_dr','min_lb_dr','max_lb_invm','min_lb_invm', 'n_H_sj_btag', 'nJets30', 'H_score', 'best_rt_score',
    'n_qnonHbb', 'n_nonHbb', 'H_M','Hl_dr', 'Hl_invm_sd', 'n_H_sj', 'n_b_Hbb', 'H_sj_bestb', 'H_sj_worstb',
    'H_eta','H_bbscore','b1_outH_score', 'b2_outH_score', 'best_Wb_invM_sd', 'Hb_invM1_sd', 'Hb_invM2_sd',
    'H_Wscore', 'H_Tscore', 'MET_pt', 'nhbbFatJets', 'nFatJets', 'nJets', 'nonHbb_b1_dr', 'nonHbb_b2_dr', 
    'n_q_Hbb', 'weight', 'genWeight']
#
dnn_ZH_alpha      = 0.0001
dnn_ZH_batch_size = 1024
dnn_ZH_epochs     = 120 # 120
DNNoutputDir      = './DNN_ZH_output/'
DNNoutputName     = 'best_case7_noweight.h5'
DNNmodelName      = 'best_case7_noweight_model.h5' 
DNNuseWeights     = False
