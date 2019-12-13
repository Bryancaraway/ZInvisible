########################
### Process data     ###
### for training     ###
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
    import matplotlib.pyplot as plt
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
                #print(rtidx)
                for i_ in tcombs_:
                    if ('0.0.0' in i_): continue
                    # Dont consider combos with repeating jets
                    #print(i_[0], i_[1])
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
                    #print(rtopd[np.in1d(rtidx,i_[0])].item())
                    #print(rtopd[np.in1d(rtidx,i_[1])].item())
                    #temp_df[i_[0]+'_'+i_[1]] = sum(rtopd[np.in1d(rtidx,i_[0])], rtopd[np.in1d(rtidx,i_[1])])
                    q_score.append(rtopd[np.in1d(rtidx,i_[0])].item() + rtopd[np.in1d(rtidx,i_[1])].item())
                    q_index.append(i_[0]+'_'+i_[1])
                    #
                #  print(temp_df)
                if (len(q_score) > 0) :
                    #print(q_score)
                    #print(q_index)
                    #print(max(q_score))
                    #print(np.array(q_index)[np.in1d(q_score,max(q_score))])
                    Q_dict['Q'].append(     max(q_score))
                    if (len(np.array(q_index)[np.in1d(q_score,max(q_score))]) == 1 ) :
                        Q_dict['Q_comb'].append(np.array(q_index)[np.in1d(q_score,max(q_score))].item())
                    else:
                        Q_dict['Q_comb'].append(np.array(q_index)[np.in1d(q_score,max(q_score))][0].item())
                        #print(np.array(q_index)[np.in1d(q_score,max(q_score))])
                        #print(max(q_score))
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
            
    def getCombRTDisc(df_):
        for key_ in df_.keys():
            top1_  = []
            top2_  = []
            #
            combo_ = df_[key_]['df']['Chi2_Comb']
            rtopd_ = df_[key_]['valRC']['ResolvedTopCandidate_discriminator']
            rtidx_ = df_[key_]['valRC']['ResolvedTopCandidate_j1j2j3Idx']
            #print(rtidx_)
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
    #####################
    def plotScore(df_, cut_, score_, range_, xlabel_, int_range = None, norm_=True, n_bins=20):
        plt.figure()
        for key in df_.keys():
            cut    = (df_[key]['val']['bestRecoZPt'] > cut_)
            weight = df_[key]['val']['weight'][cut] * np.sign(df_[key]['val']['genWeight'][cut])
            n_, bins_, _ = plt.hist(x = df_[key]['df'][score_][cut], bins=n_bins, 
                                    range=range_, 
                                    histtype= 'step', 
                                    weights= weight, density=norm_, label= key)
            #
            #Calculate integral in interesting region
            if (int_range) :
                integral = sum(n_[int(int_range[0]/(range_[1]/n_bins)):int(int_range[1]/(range_[1]/n_bins))])
                print('{0} with Zpt cut > {1}: {2} score integral between {3} = {4:4.3f}'.format(key, cut_, score_, int_range, integral))
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
    def plotChi2Comb(df_):
        fig, axs = plt.subplots(1, len(df_.keys()), figsize=(16,10))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.3, wspace=0.25)
        for key,ax in zip(df_.keys(), axs):
            combs = pd.Series(df[key]['df']['Chi2_Comb']).value_counts()
            ax.pie(combs.values/combs.values.sum(), labels=combs.index, 
                   autopct='%1.1f%%', textprops={'size': 'xx-small'})
            ax.set_title(key)
            #
        plt.show()
        plt.close()
        #
    #
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
    #plotChi2Comb(df)
    #
#
if __name__ == '__main__':
    files_samples_outDir = cfg.kinemFitCfg
    #
    prD.getData(   *files_samples_outDir, *cfg.kinemFitCut, cfg.kinemFitMaxJets)
    #prD.interpData(*files_samples_outDir, cfg.kinemFitMaxJets)
    #
    #computeChi2(   *files_samples_outDir, cfg.kinemFitMaxJets)
    #evaluateScore(  *files_samples_outDir)
