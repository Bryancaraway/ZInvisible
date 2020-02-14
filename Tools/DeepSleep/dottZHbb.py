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
import fun_library as lib
from fun_library import fill1e, fillne, deltaR, deltaPhi, invM, calc_mtb
np.random.seed(0)
##

def ZHbbAna(files_, samples_, outDir_, overlap_ = cfg.ZHbbFitoverlap):
    df = kFit.retrieveData(files_, samples_, outDir_, getak8_=True)
    for i_, key_ in enumerate(df.keys()):
        print(key_)
        fj_pt   = df[key_]['ak8']['pt']
        fj_phi  = df[key_]['ak8']['phi']
        fj_eta  = df[key_]['ak8']['eta']
        fj_E    = df[key_]['ak8']['E']
        sd_M    = df[key_]['ak8']['msoftdrop']
        bb_tag  = df[key_]['ak8']['btagDeepB']
        hbb_tag = df[key_]['ak8']['btagHbb']
        w_tag   = df[key_]['ak8']['deepTag_WvsQCD']
        w_tag   = df[key_]['ak8']['deepTag_TvsQCD']
        #
        subj_pt    = df[key_]['ak8']['Subpt']
        subj_btag  = df[key_]['ak8']['SubbtagDeepB']
        fj_subjId1 = df[key_]['ak8']['subJetIdx1']
        fj_subjId2 = df[key_]['ak8']['subJetIdx2']
        #
        fj_sj1_pt   = subj_pt  [fj_subjId1[fj_subjId1 != -1]]
        fj_sj2_pt   = subj_pt  [fj_subjId2[fj_subjId2 != -1]]
        fj_sj1_btag = subj_btag[fj_subjId1[fj_subjId1 != -1]]
        fj_sj2_btag = subj_btag[fj_subjId2[fj_subjId2 != -1]]
        #
        lep_pt   = df[key_]['val']['Lep_pt']
        lep_eta  = df[key_]['val']['Lep_eta']
        lep_phi  = df[key_]['val']['Lep_phi']
        lep_E    = df[key_]['val']['Lep_E']
        #
        met_pt   = df[key_]['val']['MET_pt']
        met_phi  = df[key_]['val']['MET_phi']
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
        b_wp = .4941
        b_cut = (tmp_[b_disc].to_numpy() < b_wp)
        b_disc= tmp_[b_disc].to_numpy()
        b_pt  = tmp_[pt_].to_numpy()
        b_phi = tmp_[phi_].to_numpy()
        b_eta = tmp_[eta_].to_numpy()
        b_E   = tmp_[E_].to_numpy()
        #
        j_pt  = tmp_[pt_].to_numpy()
        j_phi = tmp_[phi_].to_numpy()
        j_eta = tmp_[eta_].to_numpy()
        j_E   = tmp_[E_].to_numpy()
        #
        b_disc[b_cut] = np.nan
        b_pt[b_cut]   = np.nan
        b_phi[b_cut]  = np.nan
        b_eta[b_cut]  = np.nan 
        b_E[b_cut]    = np.nan
        #
        j_pt [b_cut == False] = np.nan
        j_phi[b_cut == False] = np.nan
        j_eta[b_cut == False] = np.nan
        j_E  [b_cut == False] = np.nan
        #
        df[key_]['ak8']['nbbFatJets']  = bb_tag[ak8_bbcut].counts
        df[key_]['ak8']['nhbbFatJets'] = hbb_tag[ak8_hbbcut].counts
        df[key_]['ak8']['n_nonHZ_W'] = w_tag[(w_tag >= 0.8)].counts
        #########
        hz_kinem_cut = ((fj_pt>300) & (sd_M > 50) & (sd_M < 250) & (hbb_tag >= 0.5))
        H_hbbtag, H_pt, H_eta, H_phi, H_M, H_wtag, H_bbtag = lib.sortbyscore([hbb_tag    ,
                                                                             fj_pt      ,
                                                                             fj_eta     ,
                                                                             fj_phi     ,
                                                                             sd_M       ,
                                                                             w_tag      , # 0.8 Med Wp
                                                                             bb_tag    ],
                                                                            hbb_tag     ,
                                                                            hz_kinem_cut)
        hz_kinem_sj1_cut = ((fj_pt[fj_subjId1 != -1]>300) & (sd_M[fj_subjId1 != -1] > 50) & (sd_M[fj_subjId1 != -1] < 250) & (hbb_tag[fj_subjId1 != -1] >= 0.5))
        H_sj1_pt, H_sj1_btag, H_sj1_hbb = lib.sortbyscore([fj_sj1_pt  ,
                                                           fj_sj1_btag,
                                                           hbb_tag[fj_subjId1 != -1]],
                                                          hbb_tag[fj_subjId1 != -1]  ,
                                                          hz_kinem_sj1_cut)
        hz_kinem_sj2_cut = ((fj_pt[fj_subjId2 != -1]>300) & (sd_M[fj_subjId2 != -1] > 50) & (sd_M[fj_subjId2 != -1] < 250) & (hbb_tag[fj_subjId2 != -1] >= 0.5))
        H_sj2_pt, H_sj2_btag, H_sj2_hbb = lib.sortbyscore([fj_sj2_pt  ,
                                                           fj_sj2_btag,
                                                           hbb_tag[fj_subjId2 != -1]],
                                                          hbb_tag[fj_subjId2 != -1]  ,
                                                          hz_kinem_sj2_cut)
        
        # take the best tagged H/Z -> bb
        df[key_]['ak8']['H_score'] = H_hbbtag[:,0]
        df[key_]['ak8']['H_pt']    = H_pt[:,0]
        df[key_]['ak8']['H_eta']   = H_eta[:,0]
        df[key_]['ak8']['H_phi']   = H_phi[:,0]
        df[key_]['ak8']['H_M']     = H_M[:,0]
        df[key_]['ak8']['H_Wscore']= H_wtag[:,0]
        df[key_]['ak8']['H_bbscore']=H_bbtag[:,0]
        #
        H_sj_b12 = np.column_stack([H_sj1_btag[:,0], H_sj2_btag[:,0]])
        df[key_]['ak8']['n_H_sj_btag'] = np.sum(H_sj_b12 >= b_wp, axis=1)
        df[key_]['ak8']['H_sj_bestb']  = np.nanmax(H_sj_b12, axis=1)
        df[key_]['ak8']['H_sj_worstb'] = np.nanmin(H_sj_b12, axis=1)
        #
        #df[key_]['ak8']['H2_score'] = H_hbbtag[:,1]
        #df[key_]['ak8']['H2_pt']    = H_pt   [:,1]
        #df[key_]['ak8']['H2_eta']   = H_eta  [:,1]
        #df[key_]['ak8']['H2_phi']   = H_phi  [:,1]
        #df[key_]['ak8']['H2_M']     = H_M    [:,1]
        #df[key_]['ak8']['H2_Wscore']= H_wtag [:,1]
        #df[key_]['ak8']['H2_bbscore']=H_bbtag[:,1]
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
        #
        Hj_dr = deltaR(
            H_eta[:,0],H_phi[:,0],
            j_eta,j_phi)
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
        n_jnonHbb= np.count_nonzero(Hj_dr > .8, axis=1)
        n_b_Hbb  = np.count_nonzero(Hb_dr < .8, axis=1)
        n_j_Hbb  = np.count_nonzero(Hj_dr < .8, axis=1)
        n_nonfjbb = np.count_nonzero(fjb_dr > .8, axis=1)
        #
        ind = np.argsort(np.where(Hb_dr > 0.8, Hb_dr, np.nan),axis=1)
        b_pt_dr  = np.take_along_axis(b_pt,ind,axis=1)
        b_eta_dr = np.take_along_axis(b_eta,ind,axis=1)
        b_phi_dr = np.take_along_axis(b_phi,ind,axis=1)
        b_disc_dr= np.take_along_axis(b_disc,ind,axis=1) 
        Hb_invM1 = invM(
            H_pt[:,0],H_eta[:,0],H_phi[:,0],
            b_pt_dr[:,0],b_eta_dr[:,0],b_phi_dr[:,0])
        Hb_invM2 = invM(
            H_pt[:,0],H_eta[:,0],H_phi[:,0],
            b_pt_dr[:,1],b_eta_dr[:,1],b_phi_dr[:,1])
        mtb1 = calc_mtb(b_pt_dr[:,0],b_phi_dr[:,0],met_pt,met_phi)
        mtb2 = calc_mtb(b_pt_dr[:,1],b_phi_dr[:,1],met_pt,met_phi)
        best_Wb_invM = np.where(((mtb2 > mtb1) & (mtb2 != np.nan)), Hb_invM2, Hb_invM1) 
        #
        #print(pd.Series(Hl_dr).values.tolist())
        df[key_]['ak8']['b1_outH_score'] = b_disc_dr[:,0]
        df[key_]['ak8']['b2_outH_score'] = b_disc_dr[:,1]
        df[key_]['ak8']['mtb1_outH']  = mtb1
        df[key_]['ak8']['mtb2_outH']  = mtb2
        df[key_]['ak8']['best_Wb_invM'] = best_Wb_invM
        df[key_]['ak8']['Hb_invM1']  = Hb_invM1
        df[key_]['ak8']['Hb_invM2']  = Hb_invM2
        df[key_]['ak8']['nonHbbj1_pt'] = -np.sort(-np.where(Hj_dr > 0.8, j_pt, np.nan),axis = 1 )[:,0]
        df[key_]['ak8']['nonHbbj2_pt'] = -np.sort(-np.where(Hj_dr > 0.8, j_pt, np.nan),axis = 1 )[:,1]
        df[key_]['ak8']['nonHbbj1_dr'] = np.sort(np.where(Hj_dr > 0.8, Hj_dr, np.nan),axis = 1 )[:,0]
        df[key_]['ak8']['nonHbbj2_dr'] = np.sort(np.where(Hj_dr > 0.8, Hj_dr, np.nan),axis = 1 )[:,1]
        df[key_]['ak8']['nonHbb_b1_dr'] = np.sort(np.where(Hb_dr > 0.8, Hb_dr, np.nan),axis = 1 )[:,0]
        df[key_]['ak8']['nonHbb_b2_dr'] = np.sort(np.where(Hb_dr > 0.8, Hb_dr, np.nan),axis = 1 )[:,1]
        df[key_]['ak8']['n_nonHbb'] = n_nonHbb
        df[key_]['ak8']['n_b_Hbb'] = n_b_Hbb
        df[key_]['ak8']['n_jnonHbb'] = n_jnonHbb
        df[key_]['ak8']['n_j_Hbb'] = n_j_Hbb
        df[key_]['ak8']['n_nonfjbb'] = n_nonfjbb
        df[key_]['ak8']['Hl_dr']  = Hl_dr
        df[key_]['ak8']['Hl_invm']  = Hl_invm
        df[key_]['ak8']['fjl_dr'] = fjl_dr
        df[key_]['ak8']['fjl_invm']  = fjl_invm

    for key_ in df.keys():
        sample_, year_ = key_.split('_')
        with open(outDir_+'result_'+year_+'_'+sample_+'_ak8.pkl'   ,'wb') as handle:
            pickle.dump(df[key_]['ak8'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def plotAna(files_, samples_, outDir_, overlap_ = cfg.ZHbbFitoverlap):
    df = kFit.retrieveData(files_, samples_, outDir_, getgen_=False, getak8_=True)
    #
    from fun_library import StackedHisto
    #StackedHisto(df, 'nJets30',         (0,12),     'nJets30', 12)
    #StackedHisto(df, 'n_jnonHbb', (0,6),     'nq_nonHZbb',  6) 
    StackedHisto(df, 'n_H_sj_btag', (0,6),     'n_H_sj_btag',  6) 
    StackedHisto(df, 'n_b_Hbb', (0,6),      'nb_HZbb',  6) 
    StackedHisto(df, 'H_sj_bestb', (0,1), 'H_sj_bestb', 20)
    StackedHisto(df, 'H_sj_worstb', (0,1), 'H_sj_worstb', 20)
    StackedHisto(df, 'nMergedTops', (0,5),'nMergedTops',5)
    StackedHisto(df, 'n_nonHZ_W', (0,4),       'n_nonHZ_W', 4)
    StackedHisto(df, 'H_eta',     (-3.2,3.2),    'HZ_eta',  50)
    StackedHisto(df, 'H_bbscore', (0,1), 'H_bbscore', 20)
    StackedHisto(df, 'b1_outH_score', (0,1), 'b1_outH_score', 20)
    StackedHisto(df, 'b2_outH_score', (0,1), 'b2_outH_score', 20)
    StackedHisto(df, 'best_Wb_invM',  (0,500), 'best_Wb_invM',  40)
    StackedHisto(df, 'Hb_invM1',  (0,500), 'Hb_invM1',  40)
    StackedHisto(df, 'Hb_invM2',  (0,500), 'Hb_invM2',  40)
    #StackedHisto(df, 'H2_M',     (0,300),    'HZ2_M',  40)
    StackedHisto(df, 'H_pt',     (200,600), 'HZ_pt',  20)
    #StackedHisto(df, 'H2_pt',     (200,600), 'HZ2_pt',  20)
    #StackedHisto(df, 'H2_score', (-1,1),     'HZbb2_score',  20)
    StackedHisto(df, 'H_Wscore', (0,1),     'H_Wscore',  20)
    StackedHisto(df, 'H_M',     (0,300),    'HZ_M',  100)
    StackedHisto(df, 'MET_pt',  (20,500),   'MET',         40)
    StackedHisto(df, 'H_score', (.4,1),     'HZbb_score',  20)
    StackedHisto(df, 'mtb1_outH',  (0,500), 'mtb1_outH',  40)
    #StackedHisto(df, 'mtb2_outH',  (0,500), 'mtb2_outH',  40)
    StackedHisto(df, 'nhbbFatJets', (0,6),      'nhbbFatJets',  6) 
    
    #StackedHisto(df, 'H2_Wscore', (0,1),     'H2_Wscore',  20)
    StackedHisto(df, 'nFatJets',      (0,5),     'nFatJets', 5)
    StackedHisto(df, 'nJets',         (0,12),     'nJets', 12)
    StackedHisto(df, 'nonHbbj1_pt',     (0,600), 'nonHbbq1_pt',  60)
    StackedHisto(df, 'nonHbbj2_pt',     (0,600), 'nonHbbq2_pt',  60)
    StackedHisto(df, 'nonHbbj1_dr',     (0,5), 'nonHbbq1_dr',  20)
    StackedHisto(df, 'nonHbbj2_dr',     (0,5), 'nonHbbq2_dr',  20)
    StackedHisto(df, 'nonHbb_b1_dr',    (0,5), 'nonHbb_b1_dr', 20)
    StackedHisto(df, 'nonHbb_b2_dr',    (0,5), 'nonHbb_b2_dr', 20)
    StackedHisto(df, 'nResolvedTops', (0,5),'nResolvedTops',5)
    StackedHisto(df, 'n_nonHbb', (0,6),     'nb_nonHZbb',  6) 
    StackedHisto(df, 'n_j_Hbb', (0,6),      'nq_HZbb',  6) 
    #StackedHisto(df, 'Hl_dr',    (0,5),     'HZl_dr',  20)
    #StackedHisto(df, 'Hl_invm',  (0,300),   'HZl_invm',  20)


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
        (fat_df['n_b_Hbb']    >=  1) &
        (fat_df['nhbbFatJets']>   0) & 
        (fat_df['H_M']        >  50) & 
        (fat_df['H_M']        < 180) &
        #(fat_df['best_Wb_invM']> 200) &
        (fat_df['H_Wscore'] < .90) &
        (met                  >  20)
    )
    w = w[base_cuts]
    #
    ZH_eta   = fat_df['H_eta'][base_cuts]
    ZH_phi   = fat_df['H_phi'][base_cuts]
    ZH_M     = fat_df['H_M'][base_cuts]
    ZH_score = fat_df['H_score'][base_cuts] 
    ZH_Wscore= fat_df['H_Wscore'][base_cuts]
    best_Wb_invM = fat_df['best_Wb_invM'][base_cuts]
    #
    fig, ax = plt.subplots()
    ax.hist2d(x=ZH_Wscore, y=best_Wb_invM, 
              range= ((0,1),(0,300)),
              cmin = 0.01,
              bins=50, weights=w)
    plt.show()
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
                      (gen_ids[gen_mom] > 0) & (gen_ids[gen_mom] != 21))
    case_1_base_mt = ((base) & 
                      (gen_ids[gen_mom] < 0) & (gen_ids[gen_mom] != -21))
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
    case_5  = ((case_1 == False) & (case_2 == False) & (case_3 == False) & (case_4 == False))
    #
    #print(((case_2 == case_4) & (case_2 == True)).sum()) ########
    #print(((case_3 == case_4) & (case_3 == True)).sum()) ########
    #print(gen_ids[((case_2 == case_4) & (case_2 == True))]) ########
    #print(*gen_ids[((case_3 == case_4) & (case_3 == True))][0]) ########
    #print(*gen_ids[gen_mom][((case_3 == case_4) & (case_3 == True))][0]) ########
    cases = [case_1,case_2,case_3,case_4,case_5]
    des_dict = {
        "Case_1" : 'Case_1 (qq from W within fatjet)',
        "Case_2" : 'Case_2 (b+q from W, same top within fatjet)',
        "Case_3" : 'Case_3 (b+qq from W, same top within fatjet)',
        "Case_4" : 'Case_4 (bb from ttbar within fatjet)',
        "Case_5" : 'Case_5 (Else)'
        }
    for case, des_key in zip(cases,des_dict):
        n,b,_ =plt.hist(ZH_M[case],weights=w[case],bins = 50)    
        plt.title(des_dict[des_key]+' ({0:3.1f})'.format(sum(n[:])))
        plt.xlabel('reco_HZ_M (softdrop)')
        plt.show()

    for case, des_key in zip(cases,des_dict):
        n,b,_ =plt.hist(ZH_score[case],weights=w[case],bins = 50)    
        plt.title(des_dict[des_key]+' ({0:3.1f})'.format(sum(n[:])))
        plt.xlabel('reco_HZ_score')
        plt.show()
    for case, des_key in zip(cases,des_dict):
        n,b,_ =plt.hist(ZH_Wscore[case],weights=w[case],bins = 50)    
        plt.title(des_dict[des_key]+' ({0:3.1f})'.format(sum(n[:])))
        plt.xlabel('reco_HZ_Wscore')
        plt.show()
    for case, des_key in zip(cases,des_dict):
        n,b,_ =plt.hist(best_Wb_invM[case],weights=w[case],bins = 50)    
        plt.title(des_dict[des_key]+' ({0:3.1f})'.format(sum(n[:])))
        plt.xlabel('reco_Wb_invM')
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
    fj_w_tag= fillne(fat_df['deepTag_WvsQCD'])
    fj_b_tag= fillne(fat_df['btagDeepB'])
    hbb_tag = fillne(fat_df['btagHbb'])
    best_wb = fat_df['best_Wb_invM']
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
    reco_zh_Wscore= matchkinem(fj_w_tag)[gen_match]
    reco_zh_bscore= matchkinem(fj_b_tag)[gen_match]
    reco_zh_wb_invM  = best_wb[gen_match]
    reco_zh_w     = w[gen_match]
    #
    #isZ = (isZ.counts == 1)[gen_match]
    gen_matchZ = (gen_match & (isZbb.sum() == 1))
    reco_z_pt    = matchkinem(fj_pt)[gen_matchZ]
    reco_z_M     = matchkinem(sd_M)[gen_matchZ]
    reco_z_score = matchkinem(hbb_tag)[gen_matchZ]
    reco_z_Wscore= matchkinem(fj_w_tag)[gen_matchZ]
    reco_z_bscore= matchkinem(fj_b_tag)[gen_matchZ]
    reco_z_wb_invM= best_wb[gen_matchZ]
    reco_z_w     = w[gen_matchZ]
    #
    gen_matchH = (gen_match & (isHbb.sum() == 1))
    reco_h_pt    = matchkinem(fj_pt)[gen_matchH]
    reco_h_M     = matchkinem(sd_M)[gen_matchH]
    reco_h_score = matchkinem(hbb_tag)[gen_matchH]
    reco_h_Wscore= matchkinem(fj_w_tag)[gen_matchH]
    reco_h_bscore= matchkinem(fj_b_tag)[gen_matchH]
    reco_h_wb_invM  = best_wb[gen_matchH]
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
    simpleplot([reco_zh_pt,reco_zh_M,reco_zh_score,reco_zh_Wscore,reco_zh_bscore,reco_zh_wb_invM],['zh_pt','zh_M','zh_score','reco_zh_Wscore','reco_zh_bscore','reco_zh_wb_invM'],reco_zh_w)
    simpleplot([reco_z_pt, reco_z_M, reco_z_score,reco_z_Wscore,reco_z_bscore,reco_z_wb_invM],  ['z_pt', 'z_M', 'z_score','reco_z_Wscore','reco_z_bscore','reco_z_wb_invM'],reco_z_w)
    simpleplot([reco_h_pt, reco_h_M, reco_h_score,reco_h_Wscore,reco_h_bscore,reco_h_wb_invM],  ['h_pt', 'h_M', 'h_score','reco_h_Wscore','reco_h_bscore','reco_h_wb_invM'],reco_h_w)
    #
    import matplotlib.backends.backend_pdf as matpdf
    pdf = matpdf.PdfPages('money_pdf/money_genZH.pdf')
    for fig_ in range(1, plt.gcf().number+1):
        pdf.savefig( fig_ )
    pdf.close()


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
    ZHbbAna(*files_samples_outDir, cfg.ZHbbFitoverlap)
    ########
    #GenAna(*files_samples_outDir, cfg.ZHbbFitoverlap)
    #GenAna_ttbar(*files_samples_outDir, cfg.ZHbbFitoverlap)
    plotAna(*files_samples_outDir, cfg.ZHbbFitoverlap)
