#                      #  
##                    ##
########################                               
### Process TTZ,     ###
### Z to MET, data   ###                               
# for score computation#                               
########################                               
### written by:      ###                               
### Bryan Caraway    ###                               
########################                               
##                    ##                                 
#                      #

##
import sys
import os
import pickle
#
import deepsleepcfg as cfg
import processData  as prD 
import kinematicFit as kFit
#
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
##
def genDataAna(files_, samples_, outDir_, overlap_ = cfg.ZinvFitoverlap) : 
    df = kFit.retrieveData(files_, samples_, outDir_+'overlap'+str(overlap_)+'/')
    for file_ in files_:
        for sample in samples_:
            if not os.path.exists(outDir_+file_+'_'+sample+'_gen.pkl') : continue
            
            with open(outDir_+file_+'_'+sample+'_gen.pkl', 'rb') as handle:
                gen_ = pickle.load(handle)
            df[sample+file_.strip('result')]['gen'] = gen_
    #
    ## Analyze TTBarLep_2017, TTZ_2017 ##
    #
    tt_ids = df['TTBarLep_2017']['gen']['GenPart_pdgId']
    tt_mom = df['TTBarLep_2017']['gen']['GenPart_genPartIdxMother']
    tt_pt  = df['TTBarLep_2017']['gen']['GenPart_pt']
    tt_eta = df['TTBarLep_2017']['gen']['GenPart_eta']
    tt_phi = df['TTBarLep_2017']['gen']['GenPart_phi']
    tt_E   = df['TTBarLep_2017']['gen']['GenPart_E']

    tt_cut = (
        ( ( (abs(tt_ids) <= 16) & (abs(tt_ids) >= 11) ) & (abs(tt_ids[tt_mom[tt_mom]]) == 6) ) |
        (   (abs(tt_ids) == 5)  & (abs(tt_ids[tt_mom]) == 6)                                 ) 
    )
    mt_cut = (
        ( ((tt_ids == 11) | (tt_ids == 13) | (tt_ids == 15) | (tt_ids == -12) | (tt_ids == -14) | (tt_ids == -16))    & (tt_ids[tt_mom[tt_mom]] == -6) ) |
        ( (tt_ids == -5) & (tt_ids[tt_mom] == -6) )
    )
    pt_cut = (
        ( ((tt_ids == -11) | (tt_ids == -13) | (tt_ids == -15) | (tt_ids == 12) | (tt_ids == 14) | (tt_ids == 16))    & (tt_ids[tt_mom[tt_mom]] == 6) ) |
        ( (tt_ids == 5) & (tt_ids[tt_mom] == 6) )
    )
    mb_inv = ((mt_cut.sum() == 3) & ( tt_ids == -5))
    pb_inv = ((pt_cut.sum() == 3) & ( tt_ids == 5))
    #
    base_cuts = ((df['TTBarLep_2017']['val']['nBottoms'] >= 0)     &
                 (df['TTBarLep_2017']['df']['TopMinPt'] > 50)     &
                 (df['TTBarLep_2017']['df']['TopMaxEta'] <= 2.4)  &
                 #(df['TTBarLep_2017']['df']['Minmtb'] >= 200)      &                                                                                                                     
                 (df['TTBarLep_2017']['df']['Cmtb_l3'] >= 175)         &                                                                                                                    
                 (df['TTBarLep_2017']['val']['MET_pt'] >= 250)     &
                 #(df['TTBarLep_2017']['val']['nResolvedTops'] > 0 ))                                                                                                                     
                 (df['TTBarLep_2017']['df']['TopMaxDiffM'] <= 55)) #&                                                                                                                     
                 #(df['TTBarLep_2017']['val']['MET_pt'] <= 100))#     & 
    #
    mb_id,  pb_id  = tt_ids[mb_inv], tt_ids[pb_inv]
    mb_pt,  pb_pt  = tt_pt[mb_inv],  tt_pt[pb_inv]
    plt.figure()
    plt.hist([mb_pt[base_cuts].flatten(),pb_pt[base_cuts].flatten()], histtype= 'step')
    plt.show()

    mb_eta, pb_eta = tt_eta[mb_inv], tt_eta[pb_inv]
    mb_phi, pb_phi = tt_phi[mb_inv], tt_phi[pb_inv]
    mb_E,   pb_E   = tt_E[mb_inv],   tt_E[pb_inv]


if __name__ == '__main__':   

    files_samples_outDir = cfg.ZinvFitCfg
    #
    #prD.getData(         *files_samples_outDir, *cfg.ZinvFitCut, cfg.ZinvFitMaxJets, treeDir_ = cfg.tree_dir+'_Inv', getGenData_ = True)
    #prD.interpData(      *files_samples_outDir, cfg.ZinvFitMaxJets)  
    #
    #kFit.evaluateScore(  *files_samples_outDir, cfg.ZinvFitoverlap)
    #kFit.AnalyzeScore(   *files_samples_outDir, cfg.ZinvFitoverlap) 
    genDataAna( *files_samples_outDir, cfg.ZinvFitoverlap)
