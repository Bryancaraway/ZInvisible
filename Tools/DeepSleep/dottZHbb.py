#                      #  
##                    ##
########################                               
### Process TTZ/H,   ###
### Z/H to bb, data  ###                               
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
import math
#
import deepsleepcfg as cfg
import processData  as prD 
import kinematicFit as kFit
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)
##
def fille(one_d_array):
    return pd.DataFrame.from_records(one_d_array).values.flatten()
    #
def deltaR(eta1,phi1,eta2,phi2):
    deta = eta1-eta2
    dphi = phi1-phi2
    #
    dphi[((dphi > math.pi)   & (dphi != np.nan))] = dphi[((dphi > math.pi)   & (dphi != np.nan))] - 2*math.pi
    dphi[((dphi <= -math.pi) & (dphi != np.nan))] = dphi[((dphi <= -math.pi) & (dphi != np.nan))] + 2*math.pi
    #
    delta_r = np.sqrt(np.add(np.power(deta,2),np.power(dphi,2)))
    return delta_r
def invM(pt1,eta1,phi1,pt2,eta2,phi2):
    pt1pt2 = pt1*pt2
    cosheta1eta2 = np.cosh(eta1-eta2)
    cosphi1phi2  = np.cos(phi1-phi2)
    #
    invm2 = 2*pt1pt2*(cosheta1eta2-cosphi1phi2)
    return np.sqrt(invm2)
    #
def ZHbbAna(files_, samples_, outDir_, overlap_ = cfg.ZHbbFitoverlap):
    df = kFit.retrieveData(files_, samples_, outDir_, getak8_=True)
    for i_, key_ in enumerate(df.keys()):
        b_wp = .4941
        fj_pt   = df[key_]['ak8']['pt']
        fj_phi  = df[key_]['ak8']['phi']
        fj_eta  = df[key_]['ak8']['eta']
        fj_E    = df[key_]['ak8']['E']
        sd_M    = df[key_]['ak8']['msoftdrop']
        bb_tag  = df[key_]['ak8']['btagDeepB']
        hbb_tag = df[key_]['ak8']['btagHbb']
        #
        lep_pt   = df[key_]['val']['Lep_pt']
        lep_eta  = df[key_]['val']['Lep_eta']
        lep_phi  = df[key_]['val']['Lep_phi']
        lep_E    = df[key_]['val']['Lep_E']
        #
        ak8_bbcut =  ((fj_pt > 300)  & (bb_tag >= 0.9))
        ak8_hbbcut = ((fj_pt > 300) & (hbb_tag >= 0.5))
        #
        tmp_ = df[key_]['df']
        b_disc = []
        pt_    = []
        phi_   = []
        eta_   = []
        E_     = []
        for key_str in tmp_.keys():
            if   ('btagDeepB_' in key_str):
                b_disc.append(key_str)
            elif ('pt_' in key_str):
                pt_.append(key_str)
            elif ('eta_' in key_str):
                eta_.append(key_str)
            elif ('phi_' in key_str):
                phi_.append(key_str)
            elif ('E_' in key_str):
                E_.append(key_str)
        maxb= 8
        b_cut = (tmp_[b_disc].to_numpy() < b_wp)
        b_disc= tmp_[b_disc].to_numpy()
        b_pt  = tmp_[pt_].to_numpy()
        b_phi = tmp_[phi_].to_numpy()
        b_eta = tmp_[eta_].to_numpy()
        b_E   = tmp_[E_].to_numpy()
        #
        b_disc[b_cut] = np.nan
        b_pt[b_cut]   = np.nan
        b_phi[b_cut]  = np.nan
        b_eta[b_cut]  = np.nan
        b_E[b_cut]    = np.nan
        
        #
        df[key_]['ak8']['nbbFatJets']  = bb_tag[ak8_bbcut].counts
        df[key_]['ak8']['nhbbFatJets'] = hbb_tag[ak8_hbbcut].counts
        H_pt   = fj_pt[ak8_hbbcut]
        df[key_]['ak8']['H_pt'] = fille(H_pt[:,0:1] )
        df[key_]['ak8']['H_score'] = fille(hbb_tag[ak8_hbbcut][:,0:1])
        H_eta  = fj_eta[ak8_hbbcut]
        H_phi  = fj_phi[ak8_hbbcut]
        H_M    = sd_M[ak8_hbbcut]
        df[key_]['ak8']['H_M'] = fille(H_M[:,0:1])
        #
        fjbb_pt  = fj_pt[ak8_bbcut]
        df[key_]['ak8']['fjbb_pt'] = fille(fjbb_pt[:,0:1] )
        df[key_]['ak8']['fjbb_score'] = fille(bb_tag[ak8_bbcut][:,0:1])
        fjbb_eta = fj_eta[ak8_bbcut]
        fjbb_phi = fj_phi[ak8_bbcut]
        fjbb_M   = sd_M[ak8_hbbcut]
        df[key_]['ak8']['fjbb_M'] = fille(fjbb_M[:,0:1])
        #
        #H_eta[H_eta.counts == 0] = np.nan
        Hb_dr  = []
        Hl_dr  = []
        fjb_dr = []
        fjl_dr = []
        for i_ in range(maxb):
            Hb_dr.append(deltaR(
                H_eta[:,0:1],H_phi[:,0:1],
                b_eta[:,i_],b_phi[:,i_]))
            fjb_dr.append(deltaR(
                fjbb_eta[:,0:1],fjbb_phi[:,0:1],
                b_eta[:,i_],b_phi[:,i_]))
        Hl_dr = fille(deltaR(
            H_eta[:,0:1],H_phi[:,0:1],
            lep_eta,lep_phi))
        Hl_invm = fille(invM(
            H_pt[:,0:1],H_eta[:,0:1],H_phi[:,0:1],
            lep_pt,lep_eta,lep_phi))
        fjl_dr = fille(deltaR(
            fjbb_eta[:,0:1],fjbb_phi[:,0:1],
            lep_eta,lep_phi))
        fjl_invm = fille(invM(
            fjbb_pt[:,0:1],fjbb_eta[:,0:1],fjbb_phi[:,0:1],
            lep_pt,lep_eta,lep_phi))
        Hb_dr  = np.array(Hb_dr).T
        fjb_dr = np.array(fjb_dr).T
        n_nonHbb = np.count_nonzero(Hb_dr > .8, axis=1)
        n_nonfjbb = np.count_nonzero(fjb_dr > .8, axis=1)
        
        #print(pd.Series(Hl_dr).values.tolist())
        df[key_]['ak8']['n_nonHbb'] = n_nonHbb
        df[key_]['ak8']['n_nonfjbb'] = n_nonfjbb
        df[key_]['ak8']['Hl_dr']  = Hl_dr
        df[key_]['ak8']['Hl_invm']  = Hl_invm
        df[key_]['ak8']['fjl_dr'] = fjl_dr
        df[key_]['ak8']['fjl_invm']  = fjl_invm

#    StackedHisto(df, 'n_nonHbb', (0,4),     'nb_nonHbb',  4) 
#    StackedHisto(df, 'Hl_dr',    (0,5),     'Hl_dr',  20)
#    StackedHisto(df, 'Hl_invm',  (0,300),   'Hl_invm',  20)
#    StackedHisto(df, 'H_pt',     (200,600), 'H_pt',  20)
#    StackedHisto(df, 'H_M',     (0,300),    'H_M',  20)
#    StackedHisto(df, 'H_score', (.4,1.2),   'Hbb_score',  20)
    StackedHisto(df, 'n_nonfjbb', (0,4),     'nb_nonfjbb',  4) 
    StackedHisto(df, 'fjl_dr',    (0,5),     'fjl_dr',  20)
    StackedHisto(df, 'fjl_invm',  (0,300),   'fjl_invm',  20)
    StackedHisto(df, 'fj_pt',     (200,600), 'fj_pt',  20)
    StackedHisto(df, 'fj_M',     (0,300),    'fj_M',  20)
    StackedHisto(df, 'fj_score', (.4,1.2),   'fjbb_score',  20)

    #StackedHisto(df, 'nbbFatJets',  (0,4), 'nbbFatJets',  4)
    #StackedHisto(df, 'nhbbFatJets', (0,4), 'nhbbFatJets', 4)

        
def StackedHisto(df_, kinem_, range_, xlabel_, n_bins=20):
    from matplotlib import rc
    #
    fontsize = 12    
    rc("savefig", dpi=250)
    #rc("figure", figsize=(3.375, 3.375*(6./8.)), dpi=250)                                                            
    #rc("text.latex", preamble=r"\usepackage{amsmath}")                                                             
    #rc('font',**{'family':'serif','serif':['Times']})
    #rc("hatch", linewidth=0.0) 
    #
    h        = []
    w        = []
    integral = []
    labels   = []
    colors   = []
    #

    bins = np.arange(range_[0],range_[-1]+(int((range_[-1]+1)/n_bins)) , (range_[-1]+1)/n_bins)
    #                                                                                                
    for i_, key_ in enumerate(df_.keys()):
        if (kinem_ in df_[key_]['df'].keys()):
            kinem     = df_[key_]['df'][kinem_]
        elif (kinem_ in df_[key_]['val'].keys()):
            kinem     = df_[key_]['val'][kinem_]
        elif (kinem_ in df_[key_]['valRC'].keys()):
            kinem     = df_[key_]['valRC'][kinem_]
        try:
            if (kinem_ in df_[key_]['ak8'].keys()):
                kinem = df_[key_]['ak8'][kinem_]
        except:
            pass
        base_cuts = (
            #(( (df_[key_]['ak8']['nbbFatJets']  == 1) & (df_[key_]['ak8']['nhbbFatJets'] == 0) ) |
            # ( (df_[key_]['ak8']['nbbFatJets']  == 0) & (df_[key_]['ak8']['nhbbFatJets'] == 1) )) &
            (df_[key_]['ak8']['n_nonfjbb'] >= 2) &
            #(df_[key_]['ak8']['nbbFatJets'] == 1) &
            (df_[key_]['val']['MET_pt'] >= 20))# &
                      #(df_[key_]['val']['nResolvedTops'] == 1))
        ########
        h.append( np.clip(kinem[base_cuts], bins[0], bins[-1]))
        w.append( df_[key_]['val']['weight'][base_cuts] * np.sign(df_[key_]['val']['genWeight'][base_cuts]) * (137/41.9))
        n_, bins_, _ = plt.hist(h[i_], weights=w[i_])
        integral.append( sum(n_[:]))
        la_label, color = kFit.getLaLabel(key_)
        labels.append( la_label + ' ({0:3.1f})'.format(integral[i_]))
        colors.append( color)
        plt.close('all')
    #          
    fig, ax = plt.subplots()
    fig.subplots_adjust(
        top=0.88,
        bottom=0.11,
        left=0.11,
        right=0.88,
        hspace=0.2,
        wspace=0.2
    )
    #fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.92)
    n_, bins_, patches_ = ax.hist(h,
                                  bins=bins, stacked=True,# fill=True,
                                  #range=range_,
                                  histtype='stepfilled',
                                  #linewidth=0,
                                  weights= w,
                                  color  = colors,
                                  label= labels)
    #
    #ax.grid(True)
    from matplotlib.ticker import AutoMinorLocator
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.text(0.12,0.89, r"$\bf{CMS}$ $Simulation$",   fontsize = fontsize)
    fig.text(0.64,0.89, r'137 fb$^{-1}$ (13 TeV)', fontsize = fontsize)
    plt.xlabel(xlabel_, fontsize = fontsize)
    plt.ylabel('Events / '+str(int((range_[-1]+1)/n_bins))+' GeV', fontsize = fontsize)
    plt.xlim(range_)
    plt.yscale('log')
    #plt.setp(patches_, linewidth=0)
    plt.legend()
    plt.savefig('money_pdf/moneyplot'+xlabel_+'_.pdf', dpi = 300)
    plt.show()
    plt.close(fig)
    #    
# 
if __name__ == '__main__':   

    files_samples_outDir = cfg.ZHbbFitCfg
    #
    #prD.getData(         *files_samples_outDir, *cfg.ZHbbFitCut, cfg.ZHbbFitMaxJets, treeDir_ = cfg.tree_dir+'_bb', getak8var_=True)
    #prD.interpData(      *files_samples_outDir, cfg.ZHbbFitMaxJets)  
    #
    #kFit.evaluateScore(  *files_samples_outDir, cfg.ZHbbFitoverlap, getak8_=True)
    #kFit.AnalyzeScore(   *files_samples_outDir, cfg.ZHbbFitoverlap) 
    ZHbbAna(*files_samples_outDir, cfg.ZHbbFitoverlap)
    
