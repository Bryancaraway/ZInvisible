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
    df = kFit.retrieveData(files_, ['TTBarLep', 'TTZ'], outDir_+'overlap'+str(overlap_)+'/')
    for file_ in files_:
        for sample in ['TTBarLep', 'TTZ']:
            if not os.path.exists(outDir_+file_+'_'+sample+'_gen.pkl') : continue
            
            with open(outDir_+file_+'_'+sample+'_gen.pkl', 'rb') as handle:
                gen_ = pickle.load(handle)
            df[sample+file_.strip('result')]['gen'] = gen_
    #
    ## Analyze TTBarLep_2017, TTZ_2017 ##
    #
    sample = 'TTBarLep_2017'
    #sample = 'TTZ_2017'
    Q_cut = 1.5
    #
    tt_ids = df[sample]['gen']['GenPart_pdgId']
    tt_mom = df[sample]['gen']['GenPart_genPartIdxMother']
    tt_pt  = df[sample]['gen']['GenPart_pt']
    tt_eta = df[sample]['gen']['GenPart_eta']
    tt_phi = df[sample]['gen']['GenPart_phi']
    tt_E   = df[sample]['gen']['GenPart_E']

    mt_cut = (
        ( ((tt_ids == 11) | (tt_ids == 13) | (tt_ids == 15) | (tt_ids == -12) | (tt_ids == -14) | (tt_ids == -16))    & (tt_ids[tt_mom[tt_mom]] == -6) ) |
        ( (tt_ids == -5) & (tt_ids[tt_mom] == -6) )
    )
    pt_cut = (
        ( ((tt_ids == -11) | (tt_ids == -13) | (tt_ids == -15) | (tt_ids == 12) | (tt_ids == 14) | (tt_ids == 16))    & (tt_ids[tt_mom[tt_mom]] == 6) ) |
        ( (tt_ids == 5) & (tt_ids[tt_mom] == 6) )
    )
    mb_inv = ((mt_cut.sum() == 3) & ( tt_ids == -5) & (tt_ids[tt_mom] == -6))
    pb_inv = ((pt_cut.sum() == 3) & ( tt_ids == 5)  & (tt_ids[tt_mom] ==  6))
    #
    base_cuts = ((df[sample]['val']['nBottoms'] >= 0)     &
                 (df[sample]['df']['TopMinPt'] > 50)     &
                 (df[sample]['df']['TopMaxEta'] <= 2.4)  &
                 #(df[sample]['df']['Minmtb'] >= 200)      &                                                                                                                     
                 (df[sample]['df']['Cmtb_l3'] >= 200)     &                                                         
                 (df[sample]['val']['MET_pt'] >= 250)     &
                 #(df[sample]['val']['nResolvedTops'] > 0 ))                                                                                                                     
                 (df[sample]['df']['Q'] >= Q_cut)          &
                 (df[sample]['df']['TopMaxDiffM'] <= 55)) #&                                                                                                                     
                 #(df[sample]['val']['MET_pt'] <= 100))#     & 
    #
    just_mb = ((mb_inv) & (pt_cut.sum() != 3))
    just_pb = ((pb_inv) & (mt_cut.sum() != 3))
    #
    w_mb = df[sample]['val']['weight'][((mt_cut.sum() == 3) & (pt_cut.sum() != 3))][base_cuts] * np.sign(df[sample]['val']['genWeight'][((mt_cut.sum() == 3) & (pt_cut.sum() != 3))][base_cuts]) * (137/41.9)
    w_pb = df[sample]['val']['weight'][((mt_cut.sum() != 3) & (pt_cut.sum() == 3))][base_cuts] * np.sign(df[sample]['val']['genWeight'][((mt_cut.sum() != 3) & (pt_cut.sum() == 3))][base_cuts]) * (137/41.9)
    #
    mb_pt,  pb_pt  = tt_pt[just_mb][base_cuts].flatten(),  tt_pt[just_pb][base_cuts].flatten()
    #
    from matplotlib.ticker import AutoMinorLocator
    def plot_result(x_,bins_,range_,label_):
        fig, ax = plt.subplots()
        ax.hist(x_,
                range=range_,
                bins=bins_, stacked=True, histtype='step',
                weights = [w_mb,w_pb])
        plt.xlabel(label_)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True)
    #
    mb_eta, pb_eta = tt_eta[just_mb][base_cuts].flatten(), tt_eta[just_pb][base_cuts].flatten()
    mb_phi, pb_phi = tt_phi[just_mb][base_cuts].flatten(), tt_phi[just_pb][base_cuts].flatten()
    mb_E,   pb_E   = tt_E[just_mb][base_cuts].flatten(), tt_E[just_pb][base_cuts].flatten()
    #
    plot_result([mb_pt,pb_pt],   15, (0,350),    'b_pt')
    plot_result([mb_eta,pb_eta], 25, (-5,5),     'b_eta')
    plot_result([mb_phi,pb_phi], 15, (-3.5,3.5), 'b_phi')
    plot_result([mb_E,pb_E],     15, (0,350),    'b_E')
    #plt.show()
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages('money_pdf/genb'+sample+'Q'+str(Q_cut)+'ge200_overlap'+str(overlap_)+'.pdf')
    for fig_ in range(1, plt.gcf().number+1):
        pdf.savefig( fig_ )
    pdf.close()
    #

if __name__ == '__main__':   

    files_samples_outDir = cfg.ZinvFitCfg
    #
    #prD.getData(         *files_samples_outDir, *cfg.ZinvFitCut, cfg.ZinvFitMaxJets, treeDir_ = cfg.tree_dir+'_Inv', getGenData_ = True)
    #prD.interpData(      *files_samples_outDir, cfg.ZinvFitMaxJets)  
    #
    #kFit.evaluateScore(  *files_samples_outDir, cfg.ZinvFitoverlap)
    kFit.AnalyzeScore(   *files_samples_outDir, cfg.ZinvFitoverlap) 
    #genDataAna( *files_samples_outDir, cfg.ZinvFitoverlap)
    
