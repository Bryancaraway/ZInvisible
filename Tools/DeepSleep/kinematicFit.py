########################%
### Process data     ###
# for score computation#
########################
### written by:      ###
### Bryan Caraway    ###
########################
##
#
import uproot
import sys
import os
import pickle
import deepsleepcfg as cfg
import processData  as prD 
#
import numpy as np
np.random.seed(0)
import pandas as pd
from itertools import combinations
##
import math
def deltaR(eta1, phi1, eta2, phi2):
    deta = eta1-eta2
    dphi = phi1-phi2
    dphi = pd.concat([dphi.loc[dphi.isna()],
                      dphi.loc[dphi > math.pi] - 2*math.pi,
                      dphi.loc[dphi <= -math.pi] + 2*math.pi,
                      dphi.loc[(dphi <= math.pi) & (dphi > -math.pi)]]).sort_index()
    delta_r = np.sqrt(np.add(np.power(deta,2),np.power(dphi,2)))
    return delta_r
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
#########################################################

def computeChi2(files_, samples_, outDir_, njets_):
    T_mass = 173.2
    W_mass = 80.4
    files = files_
    for file_ in files:
        for sample in samples_:
            if not os.path.exists(outDir_+file_+'_'+sample+'.pkl') : continue
            df = pd.DataFrame()
            df = pd.read_pickle(outDir_+file_+'_'+sample+'.pkl')
            ##
            chi2_df = pd.DataFrame()
            #
            invTM_combs = list(combinations(range(1,njets_+1),3))
            for i, combi in enumerate(invTM_combs):
                #
                combt1 = str(combi[0])+'.'+str(combi[1])+'.'+str(combi[2])
                for combj in invTM_combs[i+1:]:
                    
                    combt2 = str(combj[0])+'.'+str(combj[1])+'.'+str(combj[2])
                    # >= 2 --> only have one repetitous jet in calculation 
                    if (len(combi+combj)-len(set(combi+combj)) >= 1): continue 
                    print(combt1,combt2)
                    W1_combs = list(combinations(combi,2))
                    W2_combs = list(combinations(combj,2))
                    for combwi  in W1_combs:
                        combw1 = str(combwi[0])+'.'+str(combwi[1])
                        for combwj in W2_combs:
                            combw2 = str(combwj[0])+'.'+ str(combwj[1])
                            #
                            massT1 =  df['InvTM_'+combt1] 
                            massT2 =  df['InvTM_'+combt2]
                            massW1 =  df['InvWM_'+combw1]
                            massW2 =  df['InvWM_'+combw2]
                            sigmaT =  pow(25,2)
                            sigmaW =  pow(17.5,2)
                            chi2 =  (np.power((massT1 - T_mass),2)/sigmaT + np.power((massT2 - T_mass),2)/sigmaT + 
                                     np.power((massW1 - W_mass),2)/sigmaW + np.power((massW2 - W_mass),2)/sigmaW)
                            chi2_df['Chi2_'+combt1+'_'+combw1+'_'+combt2+'_'+combw2] = chi2
                            #
                        #
                    #
                #
            #
            df['Chi2']      = chi2_df.min(axis=1)
            df['Chi2_Comb'] = chi2_df.idxmin(axis=1)
            print(df['Chi2'])
            df.to_pickle(outDir_+file_+'_'+sample+'.pkl')
            del df, chi2_df
            #
        #
    #
#
def retrieveData(files_, samples_, outDir_):
    df = {}
    files = files_
    for file_ in files:
        for sample in samples_:
            if not os.path.exists(outDir_+file_+'_'+sample+'.pkl') : continue
            df_ = pd.DataFrame()
            val_df_ = pd.DataFrame()
            df_ = pd.read_pickle(outDir_+file_+'_'+sample+'.pkl')
            val_df_ = pd.read_pickle(outDir_+file_+'_'+sample+'_val.pkl')
            with open(outDir_+file_+'_'+sample+'_valRC.pkl', 'rb') as handle:
                valRCdict_ = pickle.load(handle)
            df[sample+file_.strip('result')] = {'df':df_, 'val':val_df_, 'valRC':valRCdict_}
            #
        #
    #
    return df
#
def evaluateScore(files_, samples_, outDir_, overlap_= cfg.kinemFitoverlap):
    
    df = retrieveData(files_, samples_, outDir_)
    #
    def calcQscore(df_, overlap_ = overlap_):
        for key_ in df.keys():
            Q_dict = {'Q':[],'Q_comb':[]}
            #
            rtopd_  = df_[key_]['valRC']['ResolvedTopCandidate_discriminator']
            rtidx_  = df_[key_]['valRC']['ResolvedTopCandidate_j1j2j3Idx']
            #
            for idx_, (rtopd, rtidx) in enumerate(zip(rtopd_,rtidx_)):
                q_score = []
                q_index = []
                tcombs_ = list(combinations(rtidx,2))
                for i_ in tcombs_:
                    if ('0.0.0' in i_): continue
                    # Dont consider combos with repeating jets
                    full_comb = i_[0].split('.')+i_[1].split('.')
                    ####
                    if ( (len(full_comb) - len(set(full_comb))) > overlap_) : 
                        continue
                    #### Dont count if overlap is from likely b
                    if ( ((len(full_comb) - len(set(full_comb))) == overlap_) and (overlap_ > 0) ) : 
                        jetid_ = np.array(full_comb[0:3])[np.in1d(full_comb[0:3],full_comb[3:])].item()
                        bscore_ = df_[key_]['df']['btagDeepB_'+str(jetid_)].iloc[idx_]
                        if ( bscore_ >= .4941) :
                            continue
                    #
                    q_score.append(rtopd[np.in1d(rtidx,i_[0])].item() + rtopd[np.in1d(rtidx,i_[1])].item())
                    q_index.append(i_[0]+'_'+i_[1])
                    #
                #  print(temp_df)
                if (len(q_score) > 0) :
                    Q_dict['Q'].append(     np.nanmax(q_score))
                    if (len(np.array(q_index)[np.in1d(q_score,np.nanmax(q_score))]) == 1 ) :
                        Q_dict['Q_comb'].append(np.array(q_index)[np.in1d(q_score,np.nanmax(q_score))].item())
                    else:
                        Q_dict['Q_comb'].append(np.array(q_index)[np.in1d(q_score,np.nanmax(q_score))][0].item())
                else:
                    Q_dict['Q'].append(0)
                    Q_dict['Q_comb'].append('0.0.0_0.0.0')
                #
                del q_score, q_index
                #
            #
            df[key_]['df']['Q']      = Q_dict['Q']
            df[key_]['df']['Q_comb'] = Q_dict['Q_comb']
            #
        #
    #
    def calcCmtb(df_,maxb,wp):
        b_wp = .4941
        if   wp == 'm' :
            b_wp = .4941
        elif wp == 'l' :
            b_wp = .1522
        elif wp == 'n' :
            b_wp = 0.0
        #
        for key_ in df_.keys():
            tmp_ = df_[key_]['df']
            b_disc = []
            pt_    = []
            eta_   = []
            phi_   = []
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
            #
            ind_=np.argsort(-tmp_[b_disc].values,axis=1)
            #
            b_disc = np.take_along_axis(tmp_[b_disc].values,ind_, axis=1)[:,:maxb]
            pt_    = np.take_along_axis(tmp_[pt_].values,   ind_, axis=1)[:,:maxb]
            phi_   = np.take_along_axis(tmp_[phi_].values,  ind_, axis=1)[:,:maxb]
            pt_[ b_disc < b_wp] = np.nan # 0.4941 (Medium), 0.1522 (Loose)
            phi_[b_disc < b_wp] = np.nan # 0.4941 (Medium), 0.1522 (Loose)
            metpt_ =df_[key_]['val']['MET_pt']
            metphi_=df_[key_]['val']['MET_phi']
            #
            mtb_   = []
            for i_ in range(maxb):
                mtb_.append(calc_mtb(pt_[:,i_],phi_[:,i_],metpt_,metphi_))
            Cmtb_ = np.nanmin(mtb_,axis=0)
            df_[key_]['df']['Cmtb_'+wp+str(maxb)] = Cmtb_
    #
    def calcTopTLV(df_):
        import uproot_methods
        import awkward
        from uproot_methods import TLorentzVector as TLV
        #
        for key_ in df_.keys():
            def getTLV(comb_,idx_):
                tmp_ = df_[key_]['df'].loc[idx_]
                
                comb_ = comb_.split('.')
                j1_ = TLV.from_ptetaphi(tmp_['pt_'+str(comb_[0])], tmp_['eta_'+str(comb_[0])], tmp_['phi_'+str(comb_[0])], tmp_['E_'+str(comb_[0])]) 
                j2_ = TLV.from_ptetaphi(tmp_['pt_'+str(comb_[1])], tmp_['eta_'+str(comb_[1])], tmp_['phi_'+str(comb_[1])], tmp_['E_'+str(comb_[1])]) 
                j3_ = TLV.from_ptetaphi(tmp_['pt_'+str(comb_[2])], tmp_['eta_'+str(comb_[2])], tmp_['phi_'+str(comb_[2])], tmp_['E_'+str(comb_[2])]) 
                comb_tlv = j1_ + j2_ + j3_
                #
                b_scores = [tmp_['btagDeepB_'+str(comb_[0])],tmp_['btagDeepB_'+str(comb_[1])],tmp_['btagDeepB_'+str(comb_[2])]]
                best_b  = np.nanmax(b_scores)
                if (best_b >= .4941 ):
                    if (len(set(b_scores)) == len(b_scores)):
                        b_jet = np.array([j1_, j2_, j3_])[np.in1d(b_scores,best_b)].item()
                    else:
                        b_jet = np.array([j1_, j2_, j3_])[np.in1d(b_scores,best_b)][0] 
                    return [[comb_tlv.pt, comb_tlv.eta, comb_tlv.phi, comb_tlv.mass],[b_jet.pt, b_jet.eta, b_jet.phi, b_jet.mass]]
                else:
                    return [[comb_tlv.pt, comb_tlv.eta, comb_tlv.phi, comb_tlv.mass],[np.nan,np.nan,np.nan,np.nan]]
                #
            #
            top1_    = []
            top2_    = []
            b1_      = []
            b2_      = []
            q_scores_ = df_[key_]['df']['Q']
            q_combs_  = df_[key_]['df']['Q_comb']
            for i_, (q_score, q_comb) in enumerate(zip(q_scores_,q_combs_)):
                top1_comb = q_comb.split('_')[0]
                top2_comb = q_comb.split('_')[1]
                if ('0' in top1_comb or '0' in top2_comb ):
                    top1_.append([np.nan,np.nan,np.nan,np.nan])
                    top2_.append([np.nan,np.nan,np.nan,np.nan])
                    b1_.append(  [np.nan,np.nan,np.nan,np.nan])
                    b2_.append(  [np.nan,np.nan,np.nan,np.nan])
                    
                else:
                    ret = getTLV(top1_comb,i_)
                    top1_.append(ret[0])
                    b1_.append(  ret[1])
                    ret = getTLV(top2_comb,i_)
                    top2_.append(ret[0])
                    b2_.append(  ret[1])
                    #
            #
            top1_ = np.array(top1_)
            top2_ = np.array(top2_) 
            top1_ = uproot_methods.TLorentzVectorArray.from_ptetaphim(top1_[:,[0]], top1_[:,[1]], top1_[:,[2]], top1_[:,[3]])
            top2_ = uproot_methods.TLorentzVectorArray.from_ptetaphim(top2_[:,[0]], top2_[:,[1]], top2_[:,[2]], top2_[:,[3]])
            #
            b1_ = np.array(b1_)
            b2_ = np.array(b2_) 
            b1_ = uproot_methods.TLorentzVectorArray.from_ptetaphim(b1_[:,[0]], b1_[:,[1]], b1_[:,[2]], b1_[:,[3]])
            b2_ = uproot_methods.TLorentzVectorArray.from_ptetaphim(b2_[:,[0]], b2_[:,[1]], b2_[:,[2]], b2_[:,[3]])
            #
            TopPt_    = np.concatenate((top1_.pt,                       top2_.pt),                        axis=1) 
            TopEta_   = np.concatenate((np.absolute(top1_.eta),         np.absolute(top2_.eta)),          axis=1)
            TopDiffM_ = np.concatenate((np.absolute(top1_.mass - 173.0),np.absolute(top2_.mass - 173.0)), axis=1) 
            #
            bMt_      = np.concatenate((b1_.mt,                         b2_.mt),                          axis=1)
            #
            METdPhiTop_ = np.array((deltaPhi(df_[key_]['val']['MET_phi'], top1_.phi.ravel()),
                                    deltaPhi(df_[key_]['val']['MET_phi'], top2_.phi.ravel()))).T
            
            mtb_      =  np.array((calc_mtb(b1_.pt.ravel(), b1_.phi.ravel(), df_[key_]['val']['MET_pt'], df_[key_]['val']['MET_phi']),
                                   calc_mtb(b2_.pt.ravel(), b2_.phi.ravel(), df_[key_]['val']['MET_pt'], df_[key_]['val']['MET_phi']))).T
            #
            df_[key_]['df']['TopMaxPt'] = np.nanmax(TopPt_, axis=1)
            df_[key_]['df']['TopMinPt'] = np.nanmin(TopPt_, axis=1)
            df_[key_]['df']['TopMaxEta'] = np.nanmax(TopEta_, axis=1)
            df_[key_]['df']['TopMinEta'] = np.nanmin(TopEta_, axis=1)
            df_[key_]['df']['TopMaxDiffM'] = np.nanmax(TopDiffM_, axis=1)
            df_[key_]['df']['TopMinDiffM'] = np.nanmin(TopDiffM_, axis=1)
            df_[key_]['df']['tt_dR']   = top1_.delta_r(top2_)
            df_[key_]['df']['tt_dPhi'] = top1_.delta_phi(top2_)
            df_[key_]['df']['tt_Pt']   = (top1_ + top2_).pt
            #
            df_[key_]['df']['MaxbMt']  = np.nanmax(bMt_, axis=1)
            df_[key_]['df']['MinbMt']  = np.nanmin(bMt_, axis=1)
            df_[key_]['df']['Maxmtb']  = np.nanmax(mtb_, axis=1)
            df_[key_]['df']['Minmtb']  = np.nanmin(mtb_, axis=1)
            #
            df_[key_]['df']['METtMaxDPhi'] = np.nanmax(METdPhiTop_, axis=1)
            df_[key_]['df']['METtMinDPhi'] = np.nanmin(METdPhiTop_, axis=1)
            df_[key_]['df']['METttDPhi']   = deltaPhi(df_[key_]['val']['MET_phi'], (top1_ + top2_).phi.ravel())
            
            if __name__ == '__main__' :
                ZdRTop_   = np.array((deltaR(df_[key_]['val']['bestRecoZEta'], df_[key_]['val']['bestRecoZPhi'], top1_.eta.ravel(), top1_.phi.ravel()),
                                      deltaR(df_[key_]['val']['bestRecoZEta'], df_[key_]['val']['bestRecoZPhi'], top2_.eta.ravel(), top2_.phi.ravel()))).T
                ZdPhiTop_ = np.array((deltaPhi(df_[key_]['val']['bestRecoZPhi'], top1_.phi.ravel()),
                                      deltaPhi(df_[key_]['val']['bestRecoZPhi'], top2_.phi.ravel()))).T
                ZDiffM_ = np.absolute(df_[key_]['val']['bestRecoZM'] - 91.2)
                METdPhiZ_   = deltaPhi(df_[key_]['val']['MET_phi'], df_[key_]['val']['bestRecoZPhi'])
                #
                df_[key_]['df']['ZtMaxDR'] = np.nanmax(ZdRTop_, axis=1)
                df_[key_]['df']['ZtMinDR'] = np.nanmin(ZdRTop_, axis=1)
                df_[key_]['df']['ZtMaxDPhi'] = np.nanmax(ZdPhiTop_, axis=1)
                df_[key_]['df']['ZtMinDPhi'] = np.nanmin(ZdPhiTop_, axis=1)
                df_[key_]['df']['ZttDPhi']   = deltaPhi(df_[key_]['val']['bestRecoZPhi'], (top1_ + top2_).phi.ravel())
                df_[key_]['df']['ZDiffM'] = ZDiffM_
                df_[key_]['df']['METZDPhi'] = METdPhiZ_
            
            
        #
    #
    def getCombRTDisc(df_):
        for key_ in df_.keys():
            top1_  = [] 
            top2_  = []
            #
            combo_ = df_[key_]['df']['Chi2_Comb']
            rtopd_ = df_[key_]['valRC']['ResolvedTopCandidate_discriminator']
            rtidx_ = df_[key_]['valRC']['ResolvedTopCandidate_j1j2j3Idx']
            print(combo_)
            print(key_, len(rtopd_), len(rtidx_))
            #
            for combo, rtopd, rtidx in zip(combo_,rtopd_,rtidx_):
                top1_comb = [str(combo[5:8])]
                top2_comb = [str(combo[12:15])]
                print(top1_comb, top2_comb)
                print(rtidx)
                print(rtopd)
                print('===========')
                print(rtopd[np.in1d(rtidx,top1_comb)].size)
                if (rtopd[np.in1d(rtidx,top1_comb)].size == 0) : 
                    top1_.append(None)
                else: top1_.append(rtopd[np.in1d(rtidx,top1_comb)].item())
                if (rtopd[np.in1d(rtidx,top2_comb)].size == 0) : 
                    top2_.append(None)
                else: top2_.append(rtopd[np.in1d(rtidx,top2_comb)].item())
                        	    
            df_[key_]['df']['Top1_disc'] = top1_
            df_[key_]['df']['Top2_disc'] = top2_
    #####################
    #getCombRTDisc(df)
    # df, max b's, working point: m = medium, l = loose (no tight)
    calcCmtb(df,2,'m') 
    calcCmtb(df,2,'l') 
    calcCmtb(df,3,'m') 
    calcCmtb(df,3,'l') 
    calcCmtb(df,4,'m') 
    calcCmtb(df,4,'l') 
    calcCmtb(df,5,'m') 
    calcCmtb(df,5,'l') 
    calcCmtb(df,6,'m') 
    calcCmtb(df,6,'l') 
    calcCmtb(df,7,'m') 
    calcCmtb(df,7,'l')
    calcCmtb(df,5,'n') 
    calcCmtb(df,6,'n') 
    calcCmtb(df,7,'n') 
    #
    calcQscore(df)
    calcTopTLV(df)
    #
    for key_ in df.keys():
        sample_, year_ = key_.split('_')
        df[key_]['df'].to_pickle( outDir_+'overlap'+str(overlap_)+'/'+'result_'+year_+'_'+sample_+'.pkl')
        df[key_]['val'].to_pickle(outDir_+'overlap'+str(overlap_)+'/'+'result_'+year_+'_'+sample_+'_val.pkl')
        with open(                outDir_+'overlap'+str(overlap_)+'/'+'result_'+year_+'_'+sample_+'_valRC.pkl', 
                                  'wb') as handle:
            pickle.dump(df[key_]['valRC'], handle,  protocol=pickle.HIGHEST_PROTOCOL)
    #
#
def AnalyzeScore(files_, samples_, outDir_, overlap_ = cfg.kinemFitoverlap):
    import matplotlib.pyplot as plt
    import matplotlib        as mpl
    df = retrieveData(files_, samples_, outDir_+'overlap'+str(overlap_)+'/')    

    def plotScore(df_, cut_, score_, range_, xlabel_, int_range = None, norm_=True, n_bins=20):
        plt.figure()
        for key_ in df_.keys():
            cut    = (df_[key_]['val']['bestRecoZPt'] > cut_)
            weight = df_[key_]['val']['weight'][cut] * np.sign(df_[key_]['val']['genWeight'][cut])
            n_, bins_, _ = plt.hist(x = df_[key_]['df'][score_][cut], bins=n_bins, 
                                    range=range_, 
                                    histtype= 'step', 
                                    weights= weight, density=norm_, label= key_)
            #
            #Calculate integral in interesting region
            if (int_range) :
                integral = sum(n_[int(int_range[0]/(range_[1]/n_bins)):int(int_range[1]/(range_[1]/n_bins))])
                print('{0} with Zpt cut > {1}: {2} score integral between {3} = {4:4.3f}'.format(key_, cut_, score_, int_range, integral))
            #
        print()
        #
        plt.title('Z pt > '+str(cut_))
        plt.grid(True)
        #plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel(xlabel_)
        plt.legend()
        #plt.show()
        plt.close
        #
    #        
    def getLaLabel(str_):
        la_str = ''
        col_str= ''
        if   ('TTZ' in str_):
            la_str = r't$\mathregular{\bar{t}}$Z'
            col_str = 'tab:blue'
        elif ('DY' in str_):
            la_str = 'Drell-Yan'
            col_str = 'tab:orange'
        elif ('DiBoson' in str_):
            la_str = 'VV'
            col_str = 'tab:olive'
        elif ('TriBoson' in str_):
            la_str = 'VVV'
            col_str = 'tab:pink'
        elif ('TTX' in str_):
            la_str = r't($\mathregular{\bar{t}}$)X'
            col_str = 'tab:red'
        elif ('TTBarLep' in str_):
            la_str = r't$\mathregular{\bar{t}}$'
            col_str = 'tab:green'
        elif ('TTBarHad' in str_):
            la_str = r't$\mathregular{\bar{t}}$'
            col_str = 'tab:brown'
        elif ('WJets' in str_):
            la_str = r'W$+$jets'
            col_str = 'tab:cyan'
        elif ('ZJets' in str_):
            la_str = r'Z$+$jets'
            col_str = 'tab:orange'
        elif ('QCD' in str_):
            la_str = r'QCD'
            col_str = 'tab:purple'
        return la_str, col_str
        
    def QScoreVsKinem(df_, cut_, kinem_, range_, xlabel_, n_bins=20, norm_=True, add_cuts_= None):
        from matplotlib import rc
        rc("legend", fontsize=10, scatterpoints=1, numpoints=1, borderpad=0.3, labelspacing=0.2,
           handlelength=0.7, handletextpad=0.25, handleheight=0.7, columnspacing=0.6)
        cut_range = np.linspace(cut_,2,5, endpoint=False)
        fig, axs = plt.subplots(3,2, figsize=(16,10)) 
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.92, hspace=0.4, wspace=0.25)
        hist_integral = [None,None,None,None,None,None,None,None,None]
        for i_,ax in zip(cut_range,axs.flat):
            for idx_, key_ in enumerate(df_.keys()):
                kinem = []
                if (kinem_ in df_[key_]['df'].keys()):
                    kinem     = df_[key_]['df'][kinem_]
                elif (kinem_ in df_[key_]['val'].keys()):
                    kinem     = df_[key_]['val'][kinem_]
                elif (kinem_ in df_[key_]['valRC'].keys()):
                    kinem     = df_[key_]['valRC'][kinem_]
                else:
                    continue
                #
                cut = (df_[key_]['df']['Q'] >= i_)
                if (add_cuts_):
                    if __name__ == '__main__':
                    	if (overlap_ == 1) :
                    	    base_cuts = ((df_[key_]['val']['nBottoms'] > 0)     & 
                    	                 (df_[key_]['df']['TopMinPt'] > 50)     & 
                    	                 (df_[key_]['df']['TopMaxEta'] <= 2.4)  &
                    	                 (df_[key_]['df']['TopMaxDiffM'] <= 55) &
                    	                 (df_[key_]['val']['MET_pt'] <= 75)     & 
                    	                 (df_[key_]['df']['ZDiffM'] <= 6))     
                    	
                    	elif (overlap_ == 0) :
                    	    base_cuts = ((df_[key_]['val']['nBottoms'] >= 2)     &
                    	                 (df_[key_]['df']['TopMinPt'] > 50)     &
                    	                 (df_[key_]['df']['TopMaxEta'] <= 2.4)  &
                    	                 (df_[key_]['df']['Minmtb'] >= 175)      &
                    	                 (df_[key_]['val']['MET_pt'] >= 250)     & 
                    	                 (df_[key_]['df']['TopMaxDiffM'] <= 55)) #&
                    	                 #(df_[key_]['val']['MET_pt'] <= 100))#     & 
                    	                 #(df_[key_]['df']['ZDiffM'] <= 7.5))     
                    else:
                    	if (overlap_ == 1) :
                    	    base_cuts = ((df_[key_]['val']['nBottoms'] == 2)     & 
                    	                 (df_[key_]['df']['TopMinPt'] > 50)     & 
                    	                 (df_[key_]['df']['TopMaxEta'] <= 2.4)  &
                    	                 (df_[key_]['df']['TopMaxDiffM'] <= 55) &
                    	                 (df_[key_]['val']['MET_pt'] >= 250)     & 
                                         (df_[key_]['df']['Minmtb'] >= 175))
                    	
                    	elif (overlap_ == 0) :
                    	    base_cuts = ((df_[key_]['val']['nBottoms'] >= 0)     &
                                         (df_[key_]['df']['TopMinPt'] > 50)     &
                    	                 (df_[key_]['df']['TopMaxEta'] <= 2.4)  &
                    	                 #(df_[key_]['df']['Minmtb'] >= 200)      &
                                         #(df_[key_]['df']['Cmtb'] >= 200)         &
                    	                 (df_[key_]['val']['MET_pt'] >= 250)     & 
                                         #(df_[key_]['val']['nResolvedTops'] > 0 ))
                    	                 (df_[key_]['df']['TopMaxDiffM'] <= 55)) #&
                    	                 #(df_[key_]['val']['MET_pt'] <= 100))#     & 
                    #
                    cut = (cut) & (base_cuts)
                #
                weight = df_[key_]['val']['weight'][cut] * np.sign(df_[key_]['val']['genWeight'][cut])
                alpha_  = 1.0
                label, color = getLaLabel(key_)
                n_, bins_, _= ax.hist(x = kinem[cut], bins=n_bins, 
                                      range=range_, 
                                      histtype= 'step',
                                      alpha   = alpha_,
                                      color   = color,
                                      weights= weight, density=norm_, label= label)
                alpha_ -= .1
                handles, labels = ax.get_legend_handles_labels()
                #### Calculate integral for both plots and add them to legend label string
                hist_integral[idx_] = sum(n_[:])
                for j_, (l, h) in enumerate(zip(labels, hist_integral)):
                    labels[j_] = l + '({0:3.2f})'.format(h)
                ax.set_title(' (Q > '+str(i_)+')')
                ax.legend(handles, labels, loc='best', ncol=2)
                ax.grid(True)
                #
            #
        fig_title = xlabel_+'(Overlap='+str(overlap_)+')'
        if (add_cuts_):
            fig_title += '[base_cuts]'
        fig.suptitle(fig_title)
        #
        #       
    def StackedHisto(df_, kinem_, range_, xlabel_, n_bins=20):
        #
        from matplotlib import rc
        fontsize = 12
        rc("legend", fontsize=fontsize, scatterpoints=1, numpoints=1, borderpad=0.3, labelspacing=0.2,
           handlelength=0.7, handletextpad=0.25, handleheight=0.7, columnspacing=0.6)
        rc("savefig", dpi=250)
        #rc("figure", figsize=(3.375, 3.375*(6./8.)), dpi=250)
        #rc("text", usetex=True)
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
        bins = np.arange(range_[0],range_[-1]+(int((range_[-1]+1)/n_bins)) , int((range_[-1]+1)/n_bins))
        #
        for i_, key_ in enumerate(df.keys()):
            if (kinem_ in df_[key_]['df'].keys()):
                kinem     = df_[key_]['df'][kinem_]
            elif (kinem_ in df_[key_]['val'].keys()):
                kinem     = df_[key_]['val'][kinem_]
            elif (kinem_ in df_[key_]['valRC'].keys()):
                kinem     = df_[key_]['valRC'][kinem_]
            ###########
            if __name__ == '__main__':
            	if (overlap_ == 1) :
            	    base_cuts = ((df_[key_]['val']['nBottoms'] > 0)     & 
            	                 #(df_[key_]['df']['TopMinPt'] > 50)     & 
            	                 (df_[key_]['df']['TopMaxEta'] <= 2.4)  &
            	                 #(df_[key_]['df']['TopMaxDiffM'] <= 55) &   # 55
                                 #(df_[key_]['df']['ZDiffM'] <= 7.5)       &  # 5
            	                 (df_[key_]['val']['bestRecoZPt'] >= 300) &
            	                 #(df_[key_]['val']['MET_pt'] <= 200)   & 
            	                 (df_[key_]['df']['Q'] >= 1.5))             # 1.8
            	elif (overlap_ == 0) :
            	    base_cuts = ((df_[key_]['val']['nBottoms'] > 0)     &
            	                 (df_[key_]['df']['TopMinPt'] > 50)     &
                                 (df_[key_]['df']['TopMaxEta'] <= 2.4)  &
            	                 (df_[key_]['df']['TopMaxDiffM'] <= 55) &
            	                 (df_[key_]['df']['ZDiffM'] <= 7.5)     & # 7.5
            	                 (df_[key_]['val']['bestRecoZPt'] >= 300) & 
            	                 #(df_[key_]['val']['MET_pt'] <= 100)     & 
            	                 (df_[key_]['df']['Q'] >= 1.5))
            else:
                if (overlap_ == 1) :
                    base_cuts = ((df_[key_]['val']['nBottoms'] == 2)      &
                                 #(df_[key_]['df']['Minmtb'] >= 200)       & # 175
                                 #(df_[key_]['df']['Cmtb'] >= 175)         &
                                 (df_[key_]['val']['nResolvedTops'] >= 1) &
                                 (df_[key_]['df']['Q'] >= 1.9)           &
            	                 (df_[key_]['df']['TopMinPt'] > 50)     &
                                 (df_[key_]['df']['TopMaxEta'] <= 2.4)  &
            	                 (df_[key_]['df']['TopMaxDiffM'] <= 55) &
                                 (df_[key_]['val']['MET_pt'] >= 250))
                if (overlap_ == 0) :
                    base_cuts = ((df_[key_]['val']['nBottoms'] >= 1)      &
                                 #(df_[key_]['df']['Minmtb'] >= 200)       &
                                 (df_[key_]['df']['Cmtb_l3'] >= 180)         &
                                 #(df_[key_]['val']['nResolvedTops'] == 2) &
                                 (df_[key_]['df']['Q'] >= 1.89)           &
            	                 (df_[key_]['df']['TopMinPt'] > 50)     &
                                 (df_[key_]['df']['TopMaxEta'] <= 2.4)  &
            	                 (df_[key_]['df']['TopMaxDiffM'] <= 55) &
                                 (df_[key_]['val']['MET_pt'] >= 250))
            ###########
            h.append( np.clip(kinem[base_cuts], bins[0], bins[-1]))
            w.append( df_[key_]['val']['weight'][base_cuts] * np.sign(df_[key_]['val']['genWeight'][base_cuts]) * (137/41.9))
            n_, bins_, _ = plt.hist(h[i_], weights=w[i_])
            integral.append( sum(n_[:]))
            la_label, color = getLaLabel(key_)
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
        #plt.setp(patches_, linewidth=0)
        plt.legend()
        if __name__ == '__main__':
            plt.savefig('money_pdf/moneyplot_overlap'+str(overlap_)+'.pdf', dpi = 300)
        else:
            plt.savefig('money_pdf/moneyplot_overlapInv'+str(overlap_)+'.pdf', dpi = 300)
        plt.show()
        plt.close(fig)
        #
    #
    #############################
    if __name__ == '__main__':
        StackedHisto(df, 'bestRecoZPt', (0,500), r'Z($\ell\ell$) $P_T$ (GeV)', 20)
    else:
        StackedHisto(df, 'MET_pt', (0,500), r'$E^{miss}_{T}$ (GeV)', 20)
    #exit()
    #
    add_cuts = True
    #
    QScoreVsKinem(df, 1.5, 'Cmtb_m2',     (180,500),   'Cmtb_m2',          20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'Cmtb_m3',     (180,500),   'Cmtb_m3',          20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'Cmtb_m4',     (180,500),   'Cmtb_m4',          20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'Cmtb_m5',     (180,500),   'Cmtb_m5',          20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'Cmtb_m6',     (180,500),   'Cmtb_m6',          20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'Cmtb_m7',     (180,500),   'Cmtb_m7',          20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'Cmtb_l2',     (180,500),   'Cmtb_l2',          20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'Cmtb_l3',     (180,500),   'Cmtb_l3',          20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'Cmtb_l4',     (180,500),   'Cmtb_l4',          20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'Cmtb_l5',     (180,500),   'Cmtb_l5',          20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'Cmtb_l6',     (180,500),   'Cmtb_l6',          20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'Cmtb_l7',     (180,500),   'Cmtb_l7',          20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'Cmtb_n5',     (180,500),   'Cmtb_n5',          20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'Cmtb_n6',     (180,500),   'Cmtb_n6',          20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'Cmtb_n7',     (180,500),   'Cmtb_n7',          20, False, add_cuts)
    #
    QScoreVsKinem(df, 1.5, 'Q',          (1.5,2), 'Q',             20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'TopMaxPt',    (0,500), 'TopMaxPt',     20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'TopMinPt',    (0,500), 'TopMinPt',     20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'TopMaxEta',   (0,3), 'TopMaxEta',      10, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'TopMinEta',   (0,3), 'TopMinEta',      10, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'TopMaxDiffM', (0,100), 'TopMaxDiffM',  20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'TopMinDiffM', (0,50), 'TopMinDiffM',   10, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'tt_dR',       (0,5),   'tt_dR',        10, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'tt_dPhi',       (0,3.2), 'tt_dPhi',    10, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'tt_Pt',       (0,500), 'tt_Pt',        20, False, add_cuts)
    #QScoreVsKinem(df, 1.5, 'Cmtb',     (0,500),   'Cmtb',          20, False, add_cuts)
    #

    #
    QScoreVsKinem(df, 1.5, 'Maxmtb',     (0,500),   'Maxmtb',      20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'Minmtb',     (0,500),   'Minmtb',      20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'MaxbMt',     (0,500),   'MaxbMt',      20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'MinbMt',     (0,500),   'MinbMt',      20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'MET_pt',    (0,500),   'MET_pt',       20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'MET_phi',   (-3.2,3.2),'MET_phi',      20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'METtMaxDPhi', (0,3.2),  'METtMaxDPhi', 20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'METtMinDPhi', (0,3.2),  'METtMinDPhi', 20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'METttDPhi',   (0,3.2),  'METttDPhi',   20, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'nJets30',     (0,10),  'nJets30',      11, False, add_cuts)
    QScoreVsKinem(df, 1.5, 'nBottoms',    (0,5),   'nBottoms',     6,  False, add_cuts)
    QScoreVsKinem(df, 1.5, 'nResolvedTops', (0,5), 'nResolvedTops',6,  False, add_cuts)
    QScoreVsKinem(df, 1.5, 'nMergedTops', (0,5),   'nMergedTops',  6,  False, add_cuts)
    #
    if __name__ == '__main__':
            QScoreVsKinem(df, 1.5, 'bestRecoZPt', (0,500), 'bestRecoZPt',  20, False, add_cuts)
            QScoreVsKinem(df, 1.5, 'bestRecoZEta', (0,3),'bestRecoZEta',   10, False, add_cuts)
            QScoreVsKinem(df, 1.5, 'bestRecoZM', (70,110), 'bestRecoZM',   20, False, add_cuts)
            QScoreVsKinem(df, 1.5, 'ZDiffM',      (0,20), 'ZDiffM',        20, False, add_cuts)
            QScoreVsKinem(df, 1.5, 'ZtMaxDR',     (0,5),   'ZtMaxdR',      10, False, add_cuts)
            QScoreVsKinem(df, 1.5, 'ZtMinDR',     (0,5),   'ZtMindR',      10, False, add_cuts)
            QScoreVsKinem(df, 1.5, 'ZtMaxDPhi',   (0,3.2),  'ZtMaxDPhi',   20, False, add_cuts)
            QScoreVsKinem(df, 1.5, 'ZtMinDPhi',   (0,3.2),  'ZtMinDPhi',   20, False, add_cuts)
            QScoreVsKinem(df, 1.5, 'ZttDPhi',   (0,3.2),  'ZttDPhi',       20, False, add_cuts)
            QScoreVsKinem(df, 1.5, 'METZDPhi',    (0,3.2),  'METZDPhi',    20, False, add_cuts)
    #
    import matplotlib.backends.backend_pdf
    
    if __name__ == '__main__':
        pdf = matplotlib.backends.backend_pdf.PdfPages('money_pdf/QvsKinem_overlap'+str(overlap_)+'.pdf')
    else:
        pdf = matplotlib.backends.backend_pdf.PdfPages('money_pdf/QvsKinem_overlapInv'+str(overlap_)+'.pdf')
    for fig_ in range(1, plt.gcf().number+1):
        pdf.savefig( fig_ )
    pdf.close()
    #
    exit()
    #############################
    for i_ in range(15,19):
        x_lower = i_/10
        Q_args = ('Q', (0,2), 'Q_score', (x_lower,2), False)
        #plotScore(df, 0,   *Q_args)
        #plotScore(df, 100, *Q_args)
        #plotScore(df, 200, *Q_args)
        plotScore(df, 300, *Q_args)
    #plotScore(df, 0,  'Chi2', (0,20), '${\chi^2}$)\
    #plotScore(df, 100,'Chi2', (0,20), '${\chi^2}$)
    #plotScore(df, 200,'Chi2', (0,20), '${\chi^2}$)
    #plotScore(df, 300,'Chi2', (0,20), '${\chi^2}$)
    def plotChi2Comb(df_):
        fig, axs = plt.subplots(1, len(df_.keys()), figsize=(16,10))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.3, wspace=0.25)
        for key,ax in zip(df_.keys(), axs):
            combs = pd.Series(df[key]['df']['Chi2_Comb']).value_counts()
            ax.pie(combs.values/combs.values.sum(), labels=combs.index, 
                   autopct='%1.1f%%', textprops={'size': 'xx-small'})
            ax.set_title(key)
            #

        plt.show(fig)
        plt.close('all')
        #
    #
    #plotChi2Comb(df)
    #
#
# ================================================================================= #
if __name__ == '__main__':
    files_samples_outDir = cfg.kinemFitCfg
    #
    #prD.getData(   *files_samples_outDir, *cfg.kinemFitCut, cfg.kinemFitMaxJets)
    #prD.interpData(*files_samples_outDir, cfg.kinemFitMaxJets)
    #
    ####computeChi2(   *files_samples_outDir, cfg.kinemFitMaxJets)
    #evaluateScore(  *files_samples_outDir)
    AnalyzeScore(   *files_samples_outDir)
