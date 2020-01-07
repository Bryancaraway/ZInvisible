########################
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
def evaluateScore(files_, samples_, outDir_):

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
    def calcQscore(df_, overlap_ = cfg.kinemFitoverlap):
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
                        if ( bscore_ > .90) :
                            continue
                    #
                    q_score.append(rtopd[np.in1d(rtidx,i_[0])].item() + rtopd[np.in1d(rtidx,i_[1])].item())
                    q_index.append(i_[0]+'_'+i_[1])
                    #
                #  print(temp_df)
                if (len(q_score) > 0) :
                    Q_dict['Q'].append(     max(q_score))
                    if (len(np.array(q_index)[np.in1d(q_score,max(q_score))]) == 1 ) :
                        Q_dict['Q_comb'].append(np.array(q_index)[np.in1d(q_score,max(q_score))].item())
                    else:
                        Q_dict['Q_comb'].append(np.array(q_index)[np.in1d(q_score,max(q_score))][0].item())
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
                return [comb_tlv.pt, comb_tlv.eta, comb_tlv.phi, comb_tlv.mass]
                #
            #
            top1_    = []
            top2_    = []
            q_scores_ = df_[key_]['df']['Q']
            q_combs_  = df_[key_]['df']['Q_comb']
            for i_, (q_score, q_comb) in enumerate(zip(q_scores_,q_combs_)):
                top1_comb = q_comb.split('_')[0]
                top2_comb = q_comb.split('_')[1]
                if ('0' in top1_comb or '0' in top2_comb ):
                    #top1_.append([0,0,0,0])
                    #top2_.append([0,0,0,0])
                    top1_.append([np.nan,np.nan,np.nan,np.nan])
                    top2_.append([np.nan,np.nan,np.nan,np.nan])
                else:
                    top1_.append(getTLV(top1_comb,i_))
                    top2_.append(getTLV(top2_comb,i_))
                #
            #
            top1_ = np.array(top1_)
            top2_ = np.array(top2_) 
            top1_ = uproot_methods.TLorentzVectorArray.from_ptetaphim(top1_[:,[0]], top1_[:,[1]], top1_[:,[2]], top1_[:,[3]])
            top2_ = uproot_methods.TLorentzVectorArray.from_ptetaphim(top2_[:,[0]], top2_[:,[1]], top2_[:,[2]], top2_[:,[3]])
            df_[key_]['df']['Top_1'] = top1_
            df_[key_]['df']['Top_2'] = top2_
            #
            TopPt_    = np.concatenate((top1_.pt,                       top2_.pt),                        axis=1) 
            TopEta_   = np.concatenate((np.absolute(top1_.eta),         np.absolute(top2_.eta)),          axis=1)
            TopDiffM_ = np.concatenate((np.absolute(top1_.mass - 173.0),np.absolute(top2_.mass - 173.0)), axis=1) 
            df_[key_]['df']['TopMaxPt'] = np.amax(TopPt_, axis=1)
            df_[key_]['df']['TopMinPt'] = np.amin(TopPt_, axis=1)
            df_[key_]['df']['TopMaxEta'] = np.amax(TopEta_, axis=1)
            df_[key_]['df']['TopMinEta'] = np.amin(TopEta_, axis=1)
            df_[key_]['df']['TopMaxDiffM'] = np.amax(TopDiffM_, axis=1)
            df_[key_]['df']['TopMinDiffM'] = np.amin(TopDiffM_, axis=1)
            df_[key_]['df']['Top_dR'] = top1_.delta_r(top2_)
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
    calcQscore(df)
    calcTopTLV(df)
    #####################
    import matplotlib.pyplot as plt
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
    def QScoreVsKinem(df_, cut_, kinem_, range_, xlabel_, n_bins=20, norm_=True):
        cut_range = np.linspace(cut_,2,5, endpoint=False)
        fig, axs = plt.subplots(1,len(cut_range), figsize=(16,10)) 
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.95, hspace=0.4, wspace=0.35)
        for i_,ax in zip(cut_range,axs):
            for key_ in df_.keys():
                if (kinem_ in df_[key_]['df'].keys()):
                    kinem     = df_[key_]['df'][kinem_]
                elif (kinem_ in df_[key_]['val'].keys()):
                    kinem     = df_[key_]['val'][kinem_]
                elif (kinem_ in df_[key_]['valRC'].keys()):
                    kinem     = df_[key_]['valRC'][kinem_]
            #
         
                cut = (df_[key_]['df']['Q'] > i_)
                weight = df_[key_]['val']['weight'][cut] * np.sign(df_[key_]['val']['genWeight'][cut])
                alpha_  = 1.0
                ax.hist(x = kinem[cut], bins=n_bins, 
                        range=range_, 
                        histtype= 'step',
                        alpha   = alpha_,
                        weights= weight, density=norm_, label= key_+' (Q > '+str(i_)+')')
                alpha_ -= .1
                ax.legend()
                ax.grid(True)
                #
            #
        fig.suptitle(xlabel_)
        plt.show()
        plt.close()
        #
    #       
    #############################
    QScoreVsKinem(df, 1.5, 'TopMaxPt',    (0,500), 'TopMaxPt',    20, False)
    QScoreVsKinem(df, 1.5, 'TopMinPt',    (0,500), 'TopMinPt',    20, False)
    QScoreVsKinem(df, 1.5, 'TopMaxEta',   (0,3), 'TopMaxEta',    10, False)
    QScoreVsKinem(df, 1.5, 'TopMinEta',   (0,3), 'TopMinEta',    10, False)
    QScoreVsKinem(df, 1.5, 'TopMaxDiffM', (0,100), 'TopMaxDiffM', 20, False)
    QScoreVsKinem(df, 1.5, 'TopMinDiffM', (0,50), 'TopMinDiffM',  10, False)
    QScoreVsKinem(df, 1.5, 'Top_dR',      (0,5),   'Top_dR',      10, False)
    QScoreVsKinem(df, 1.5, 'bestRecoZPt', (0,500), 'bestRecoZPt', 20, False)
    QScoreVsKinem(df, 1.5, 'nJets30',     (0,10),  'nJets30',     11, False)
    QScoreVsKinem(df, 1.5, 'nBottoms',    (0,5),   'nBottoms',    6,  False)
    QScoreVsKinem(df, 1.5, 'nMergedTops', (0,5),   'nMergedTops', 6,  False)
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
    #computeChi2(   *files_samples_outDir, cfg.kinemFitMaxJets)
    evaluateScore(  *files_samples_outDir)

