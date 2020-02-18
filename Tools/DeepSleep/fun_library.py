#                      #  
##                    ##
########################                               
### Library for      ###
### TTZ/H utility    ###
### functions        ###
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

def fill1e(one_d_array):
    return pd.DataFrame.from_records(one_d_array).values.flatten()
def fillne(n_d_array):
    return pd.DataFrame.from_records(n_d_array).values
    #
def sortbyscore(vars_, score_, cut_):
    ret_vars_ = []
    ind_=np.argsort(fillne(-score_[cut_]),axis=1)
    for var_ in vars_:
        temp_ = var_[:][cut_]
        ret_vars_.append(np.take_along_axis(fillne(temp_),ind_, axis=1))
    return ret_vars_
    #
def deltaR(eta1,phi1,eta2,phi2):
    try:
        deta = np.subtract(eta1,eta2.T).T
        dphi = np.subtract(phi1,phi2.T).T
    except (AttributeError) :
        deta = eta1 - eta2
        dphi = phi1 - phi2
    #
    dphi[((dphi > math.pi)   & (dphi != np.nan))] = dphi[((dphi > math.pi)   & (dphi != np.nan))] - 2*math.pi
    dphi[((dphi <= -math.pi) & (dphi != np.nan))] = dphi[((dphi <= -math.pi) & (dphi != np.nan))] + 2*math.pi
    #
    delta_r = np.sqrt(np.add(np.power(deta,2),np.power(dphi,2)))
    return delta_r
    #
def invM(pt1,eta1,phi1,pt2,eta2,phi2):
    pt1pt2 = pt1*pt2
    cosheta1eta2 = np.cosh(eta1-eta2)
    cosphi1phi2  = np.cos(phi1-phi2)
    #
    invm2 = 2*pt1pt2*(cosheta1eta2-cosphi1phi2)
    return np.sqrt(invm2)
    #
def deltaPhi(phi1, phi2):
    dphi = phi1-phi2
    dphi = pd.concat([dphi.loc[dphi.isna()],
                      dphi.loc[dphi > math.pi] - 2*math.pi,
                      dphi.loc[dphi <= -math.pi] + 2*math.pi,
                      dphi.loc[(dphi <= math.pi) & (dphi > -math.pi)]]).sort_index()
    return dphi

def calc_mtb(ptb, phib, m_pt, m_phi):
    mtb2 =  2*m_pt*ptb*(1 - np.cos(deltaPhi(phib, m_phi)))
    return np.sqrt(mtb2)
    #
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

    bins = np.arange(range_[0],range_[-1]+((range_[-1])/n_bins) , (range_[-1])/n_bins)
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
            #(df_[key_]['ak8']['n_nonfjbb'] >= 2) &
            (df_[key_]['ak8']['n_nonHbb'] >= 2)    &
            (df_[key_]['ak8']['best_rt_score'] >= .5)    &
            #(df_[key_]['val']['matchedGen'] == True)   &
            #(df_[key_]['ak8']['n_b_Hbb'] >= 1)     &
            #(df_[key_]['ak8']['n_jnonHbb'] >= 1)     &
            (df_[key_]['ak8']['nhbbFatJets'] > 0)  &
            (df_[key_]['ak8']['H_M']         > 50) &  
            (df_[key_]['ak8']['H_M']         < 250)& 
            #(df_[key_]['ak8']['H_Wscore']     < .80)&
            #(((df_[key_]['ak8']['best_Wb_invM']<= 175)&(df_[key_]['ak8']['H_Wscore']<.85))|(df_[key_]['ak8']['best_Wb_invM']> 175))&
            #(df_[key_]['ak8']['best_Wb_invM']> 200)&
            #(df_[key_]['ak8']['Hb_invM1']    > 175)&
            #(df_[key_]['ak8']['H_score']     > .75)&
            #(df_[key_]['ak8']['nbbFatJets'] == 1) &
            #(df_[key_]['val']['nResolvedTops'] == 1) &
            (df_[key_]['val']['MET_pt']      >= 20))# &

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
    plt.ylabel('Events / '+str((range_[-1])/n_bins)+' GeV', fontsize = fontsize)
    plt.xlim(range_)
    plt.yscale('log')
    #plt.setp(patches_, linewidth=0)
    plt.legend(framealpha = 0.2)
    plt.savefig('money_pdf/moneyplot'+xlabel_+'_.pdf', dpi = 300)
    plt.show()
    plt.close(fig)
    #    
# 
