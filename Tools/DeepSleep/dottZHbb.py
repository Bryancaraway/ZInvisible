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
def fill1e(one_d_array):
    return pd.DataFrame.from_records(one_d_array).values.flatten()
def fillne(n_d_array):
    return pd.DataFrame.from_records(n_d_array).values
    #
def sortbyscore(vars_, score_):
    ind_=np.argsort(fillne(-score_),axis=1)
    for i in range(len(vars_)):
        vars_[i] = np.take_along_axis(fillne(vars_[i]),ind_, axis=1)
    return vars_
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
def ZHbbAna(files_, samples_, outDir_, overlap_ = cfg.ZHbbFitoverlap):
    df = kFit.retrieveData(files_, samples_, outDir_, getak8_=True)
    for i_, key_ in enumerate(df.keys()):
        print(key_)
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
        #########
        hbb_tag, H_pt, H_eta, H_phi, H_M = sortbyscore([hbb_tag[fj_pt>300],
                                                        fj_pt  [fj_pt>300],
                                                        fj_eta [fj_pt>300],
                                                        fj_phi [fj_pt>300],
                                                        sd_M   [fj_pt>300]],  
                                                       hbb_tag [fj_pt>300])
        # take the best tagged H/Z -> bb
        df[key_]['ak8']['H_score'] = hbb_tag[:,0]
        df[key_]['ak8']['H_pt']    = H_pt[:,0]
        df[key_]['ak8']['H_eta']   = H_eta[:,0]
        df[key_]['ak8']['H_phi']   = H_phi[:,0]
        df[key_]['ak8']['H_M']     = H_M[:,0]
        #########
        fjbb_pt  = fj_pt[ak8_bbcut]
        df[key_]['ak8']['fjbb_pt'] = fill1e(fjbb_pt[:,0:1] )
        df[key_]['ak8']['fjbb_score'] = fill1e(bb_tag[ak8_bbcut][:,0:1])
        fjbb_eta = fj_eta[ak8_bbcut]
        fjbb_phi = fj_phi[ak8_bbcut]
        fjbb_M   = sd_M[ak8_hbbcut]
        df[key_]['ak8']['fjbb_M'] = fill1e(fjbb_M[:,0:1])
        #
        #H_eta[H_eta.counts == 0] = np.nan
        Hb_dr  = []
        Hl_dr  = []
        fjb_dr = []
        fjl_dr = []
        for i_ in range(maxb):
            Hb_dr.append(deltaR(
                H_eta[:,0],H_phi[:,0],
                b_eta[:,i_],b_phi[:,i_]))
            fjb_dr.append(deltaR(
                fjbb_eta[:,0:1],fjbb_phi[:,0:1],
                b_eta[:,i_],b_phi[:,i_]))
        Hl_dr = deltaR(
            H_eta[:,0],H_phi[:,0],
            lep_eta,lep_phi)
        Hl_invm = invM(
            H_pt[:,0],H_eta[:,0],H_phi[:,0],
            lep_pt,lep_eta,lep_phi)
        fjl_dr = fill1e(deltaR(
            fjbb_eta[:,0:1],fjbb_phi[:,0:1],
            lep_eta,lep_phi))
        fjl_invm = fill1e(invM(
            fjbb_pt[:,0:1],fjbb_eta[:,0:1],fjbb_phi[:,0:1],
            lep_pt,lep_eta,lep_phi))
        Hb_dr  = np.array(Hb_dr).T
        fjb_dr = np.array(fjb_dr).T
        n_nonHbb = np.count_nonzero(Hb_dr > .8, axis=1)
        n_b_Hbb  = np.count_nonzero(Hb_dr < .8, axis=1)
        n_nonfjbb = np.count_nonzero(fjb_dr > .8, axis=1)
        
        #print(pd.Series(Hl_dr).values.tolist())
        df[key_]['ak8']['n_nonHbb'] = n_nonHbb
        df[key_]['ak8']['n_b_Hbb'] = n_b_Hbb
        df[key_]['ak8']['n_nonfjbb'] = n_nonfjbb
        df[key_]['ak8']['Hl_dr']  = Hl_dr
        df[key_]['ak8']['Hl_invm']  = Hl_invm
        df[key_]['ak8']['fjl_dr'] = fjl_dr
        df[key_]['ak8']['fjl_invm']  = fjl_invm

    for key_ in df.keys():
        sample_, year_ = key_.split('_')
        with open(outDir_+'result_'+year_+'_'+sample_+'_ak8.pkl'   ,'wb') as handle:
            pickle.dump(df[key_]['ak8'], handle, protocol=pickle.HIGHEST_PROTOCOL)

    StackedHisto(df, 'n_nonHbb', (0,4),     'nb_nonHZbb',  4) 
    StackedHisto(df, 'n_b_Hbb', (0,4),      'nb_HZbb',  4) 
    StackedHisto(df, 'Hl_dr',    (0,5),     'HZl_dr',  20)
    StackedHisto(df, 'Hl_invm',  (0,300),   'HZl_invm',  20)
    StackedHisto(df, 'H_pt',     (200,600), 'HZ_pt',  20)
    StackedHisto(df, 'H_M',     (0,300),    'HZ_M',  100)
    StackedHisto(df, 'H_score', (-1,1),     'HZbb_score',  20)
    StackedHisto(df, 'MET_pt',  (20,500),   'MET',         40)
#    StackedHisto(df, 'n_nonfjbb', (0,4),     'nb_nonfjbb',  4) 
#    StackedHisto(df, 'fjl_dr',    (0,5),     'fjl_dr',  20)
#    StackedHisto(df, 'fjl_invm',  (0,300),   'fjl_invm',  20)
#    StackedHisto(df, 'fj_pt',     (200,600), 'fj_pt',  20)
#    StackedHisto(df, 'fj_M',     (0,300),    'fj_M',  20)
#    StackedHisto(df, 'fj_score', (.4,1.2),   'fjbb_score',  20)

    #StackedHisto(df, 'nbbFatJets',  (0,4), 'nbbFatJets',  4)
    #StackedHisto(df, 'nhbbFatJets', (0,4), 'nhbbFatJets', 4)
def GenAna_ttbar(files_, samples_, outDir_, overlap_ = cfg.ZHbbFitoverlap):
    df = kFit.retrieveData(files_, ['TTBarLep'], outDir_, getgen_=True, getak8_=True)
    print(df.keys())
    df = df['TTBarLep_2017']
    w  = df['val']['weight'].values * np.sign(df['val']['genWeight'].values) * (137/41.9)
    #
    gen_df = df['gen']
    fat_df = df['ak8']
    met    = df['val']['MET_pt']
    base_cuts = (
        (fat_df['n_nonHbb']   >=  2) &   
        (fat_df['nhbbFatJets']>   0) & 
        (fat_df['H_M']        >  50) & 
        (fat_df['H_M']        < 140) &
        (met                  >  20)
    )
    w = w[base_cuts]
    #
    ZH_eta = fat_df['H_eta'][base_cuts]
    ZH_phi = fat_df['H_phi'][base_cuts]
    ZH_M   = fat_df['H_M'][base_cuts]

    #
    gen_ids = gen_df['GenPart_pdgId'][base_cuts]
    gen_mom = gen_df['GenPart_genPartIdxMother'][base_cuts]
    gen_pt  = gen_df['GenPart_pt'][base_cuts]
    gen_eta = gen_df['GenPart_eta'][base_cuts]
    gen_phi = gen_df['GenPart_phi'][base_cuts]
    gen_E   = gen_df['GenPart_E'][base_cuts]
    #
    get_bq_fromtop = (
        ((abs(gen_ids) == 5) & (abs(gen_ids[gen_mom]) == 6)) | # if bottom and mom is top
        ((abs(gen_ids) < 5) & (abs(gen_ids[gen_mom]) == 24) & (abs(gen_ids[gen_mom[gen_mom]]) == 6))   # get quarks from W whos parrent is a top
    )
    base = ((get_bq_fromtop) & (get_bq_fromtop.sum() ==4))
    case_1_base_pt = ((base) & 
                      (gen_ids[gen_mom] > 0) & (gen_ids[gen_mom] != 22))
    case_1_base_mt = ((base) & 
                      (gen_ids[gen_mom] < 0) & (gen_ids[gen_mom] != -22))
    case_1_base    = (
        ((case_1_base_pt) & (case_1_base_pt.sum() == 3)) |
        ((case_1_base_mt) & (case_1_base_mt.sum() == 3)) 
    )
    case_4_base = ((base & (abs(gen_ids) == 5)) & ((base & (abs(gen_ids) == 5)).sum() == 2))
    # 
    case_123_dr = deltaR(ZH_eta,ZH_phi,gen_eta[case_1_base],gen_phi[case_1_base])
    case_4_dr   = deltaR(ZH_eta,ZH_phi,gen_eta[case_4_base],gen_phi[case_4_base])
    #
    case_1  = (((case_123_dr < 0.8) & 
                (((abs(gen_ids[case_1_base][case_123_dr<0.8]) < 6).sum() == 2) & ((abs(gen_ids[case_1_base][case_123_dr<0.8]) < 5).sum() == 2))
            ).sum() == 2)
    case_2  = (((case_123_dr < 0.8) & 
                (((abs(gen_ids[case_1_base][case_123_dr<0.8]) < 6).sum() == 2) & ((abs(gen_ids[case_1_base][case_123_dr<0.8]) < 5).sum() == 1))
            ).sum() == 2)
    case_3  = (((case_123_dr < 0.8) & 
                (((abs(gen_ids[case_1_base][case_123_dr<0.8]) < 6).sum() == 3) & ((abs(gen_ids[case_1_base][case_123_dr<0.8]) < 5).sum() == 2))
            ).sum() == 3)
    case_4  = (((case_4_dr < 0.8) & ((abs(gen_ids[case_4_base][case_4_dr<0.8]) == 5).sum() == 2)).sum() == 2)
    #case_4  = ((case_4_dr < 0.8).sum() == 2)
    case_5  = ((case_1 == False) & (case_2 == False) & (case_3 == False) & (case_4 == False))
    #
    #print(((case_2 == case_4) & (case_2 == True)).sum()) ########
    #print(((case_3 == case_4) & (case_3 == True)).sum()) ########
    #print(gen_ids[((case_2 == case_4) & (case_2 == True))]) ########
    #print(*gen_ids[((case_3 == case_4) & (case_3 == True))][0]) ########
    #print(*gen_ids[gen_mom][((case_3 == case_4) & (case_3 == True))][0]) ########
    plt.hist(ZH_M[case_1],weights=w[case_1],bins = 50)    
    plt.show()
    plt.hist(ZH_M[case_2],weights=w[case_2],bins = 50)    
    plt.show()
    plt.hist(ZH_M[case_3],weights=w[case_3],bins = 20)    
    plt.show()
    plt.hist(ZH_M[case_4],weights=w[case_4],bins = 20)    
    plt.show()
    plt.hist(ZH_M[case_5],weights=w[case_5],bins = 50)    
    plt.show()
    
def GenAna(files_, samples_, outDir_, overlap_ = cfg.ZHbbFitoverlap):
    df = kFit.retrieveData(files_, ['TTZH'], outDir_, getgen_=True, getak8_=True)
    df = df['TTZH_2017']
    w  = df['val']['weight'].values * np.sign(df['val']['genWeight'].values) * (137/41.9)
    #
    gen_df = df['gen']
    fat_df = df['ak8']

    #
    zh_ids = gen_df['GenPart_pdgId']
    zh_mom = gen_df['GenPart_genPartIdxMother']
    zh_pt  = gen_df['GenPart_pt']
    zh_eta = gen_df['GenPart_eta']
    zh_phi = gen_df['GenPart_phi']
    zh_E   = gen_df['GenPart_E']
    # higgs pdgid = 25
    # Z     pdgid = 23
    isHbb = (zh_ids == 25)
    #
    isbb_fromZ = (((zh_ids == -5) | (zh_ids == 5)) & 
                 (zh_ids[zh_mom] == 23))
    isZbb  = ((zh_ids == 23) & (isbb_fromZ.sum() > 0))
    isZHbb = (isHbb | isZbb)
    #
    zh_pt  = fill1e(zh_pt[isZHbb]).flatten()
    zh_eta = fill1e(zh_eta[isZHbb]).flatten()
    zh_phi = fill1e(zh_phi[isZHbb]).flatten()
    zh_E   = fill1e(zh_E[isZHbb]).flatten()    
    #
    fj_pt   = fillne(fat_df['pt'])
    fj_phi  = fillne(fat_df['phi'])
    fj_eta  = fillne(fat_df['eta'])
    fj_E    = fillne(fat_df['E'])
    sd_M    = fillne(fat_df['msoftdrop'])
    hbb_tag = fillne(fat_df['btagHbb'])
    #
    gen_dr = deltaR(zh_eta,zh_phi,fj_eta,fj_phi)
    gen_dr_match = np.nanmin(gen_dr,axis=1)
    gen_match    = ((gen_dr_match < 0.8) & (zh_pt >= 300) & (zh_eta <= 2.4) & (zh_eta >= -2.4))
    #
    def matchkinem(kinem_):
        ind_=np.argsort(gen_dr,axis=1)
        return np.take_along_axis(kinem_,ind_,axis=1)[:,0]
    #
    reco_zh_pt    = matchkinem(fj_pt)[gen_match]
    reco_zh_M     = matchkinem(sd_M)[gen_match]
    reco_zh_score = matchkinem(hbb_tag)[gen_match]
    reco_zh_w     = w[gen_match]
    #
    #isZ = (isZ.counts == 1)[gen_match]
    gen_matchZ = (gen_match & (isZbb.sum() == 1))
    reco_z_pt    = matchkinem(fj_pt)[gen_matchZ]
    reco_z_M     = matchkinem(sd_M)[gen_matchZ]
    reco_z_score = matchkinem(hbb_tag)[gen_matchZ]
    reco_z_w     = w[gen_matchZ]
    #
    gen_matchH = (gen_match & (isHbb.sum() == 1))
    reco_h_pt    = matchkinem(fj_pt)[gen_matchH]
    reco_h_M     = matchkinem(sd_M)[gen_matchH]
    reco_h_score = matchkinem(hbb_tag)[gen_matchH]
    reco_h_w     = w[gen_matchH]
    #
    from matplotlib.ticker import AutoMinorLocator
    def simpleplot(x_,l_,w_):
        for x,l in zip(x_,l_):
            fig, ax = plt.subplots()
            ax.hist(x, bins=50, weights=w_)
            plt.xlabel(l)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.grid(True)
            #plt.show()
        fig, ax = plt.subplots()
        ax.hist2d(x=x_[2], y=x_[1], 
                  range= ((-1,1),(0,300)),
                  cmin = 0.01,
                  bins=50, weights=w_)
        plt.xlabel(l_[2])
        plt.ylabel(l_[1])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True)
        #plt.show()
        #
    simpleplot([reco_zh_pt,reco_zh_M,reco_zh_score],['zh_pt','zh_M','zh_score'],reco_zh_w)
    simpleplot([reco_z_pt, reco_z_M, reco_z_score],  ['z_pt', 'z_M', 'z_score'],reco_z_w)
    simpleplot([reco_h_pt, reco_h_M, reco_h_score],  ['h_pt', 'h_M', 'h_score'],reco_h_w)
    #
    import matplotlib.backends.backend_pdf as matpdf
    pdf = matpdf.PdfPages('money_pdf/money_genZH.pdf')
    for fig_ in range(1, plt.gcf().number+1):
        pdf.savefig( fig_ )
    pdf.close()

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
            (df_[key_]['ak8']['n_b_Hbb'] >= 1)     &
            (df_[key_]['ak8']['nhbbFatJets'] > 0)  &
            (df_[key_]['ak8']['H_M']         > 50) &  
            (df_[key_]['ak8']['H_M']         < 200)&  
            #(df_[key_]['ak8']['nbbFatJets'] == 1) &
            (df_[key_]['val']['MET_pt']      >= 20))# &
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
    plt.ylabel('Events / '+str((range_[-1])/n_bins)+' GeV', fontsize = fontsize)
    plt.xlim(range_)
    plt.yscale('log')
    #plt.setp(patches_, linewidth=0)
    plt.legend()
    plt.savefig('money_pdf/moneyplot'+xlabel_+'_.pdf', dpi = 300)
    plt.show()
    plt.close(fig)
    #    
# 
def fixttH_weight(files_, samples_, outDir_, overlap_ = cfg.ZHbbFitoverlap):
    df_ = kFit.retrieveData(files_, ['TTZH'], outDir_, getgen_=True)
    df = df_['TTZH_2017']
    w  = df['val']['weight'].values
    #
    gen_df = df['gen']
    zh_ids = gen_df['GenPart_pdgId']
    isHbb = ((zh_ids == 25).sum() > 0)
    #

    fixed_w = np.where(isHbb,w*(.2934/.153),w)
    df_['TTZH_2017']['val']['weight'] = fixed_w
    df_['TTZH_2017']['val'].to_pickle(outDir_+'result_2017_TTZH_val.pkl')
    
if __name__ == '__main__':   

    files_samples_outDir = cfg.ZHbbFitCfg
    #
    #prD.getData(         *files_samples_outDir, *cfg.ZHbbFitCut, cfg.ZHbbFitMaxJets, treeDir_ = cfg.tree_dir+'_bb', getGenData_ = True, getak8var_=True)
    #prD.interpData(      *files_samples_outDir, cfg.ZHbbFitMaxJets)  
    #fixttH_weight(*files_samples_outDir, cfg.ZHbbFitoverlap)
    #
    #kFit.evaluateScore(  *files_samples_outDir, cfg.ZHbbFitoverlap, getak8_=True)
    #kFit.AnalyzeScore(   *files_samples_outDir, cfg.ZHbbFitoverlap) 
    #ZHbbAna(*files_samples_outDir, cfg.ZHbbFitoverlap)
    ########
    #GenAna(*files_samples_outDir, cfg.ZHbbFitoverlap)
    GenAna_ttbar(*files_samples_outDir, cfg.ZHbbFitoverlap)
