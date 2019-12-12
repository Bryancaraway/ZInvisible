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
                if i == 5 : exit()
                #
                combt1 = str(combi[0])+str(combi[1])+str(combi[2])
                for combj in invTM_combs[i+1:]:
                    
                    combt2 = str(combj[0])+str(combj[1])+str(combj[2])
                    for _ in combt1: combt2=combt2.replace(_,"")
                    # CODE BREAKS HERE
                    print(combt1,combt2)
                    if (len(combt2.strip(combt1)) != 3) : continue
                    W1_combs = list(combinations(combt1,2))
                    W2_combs = list(combinations(combt2,2))
                    for combwi  in W1_combs:
                        combw1 = str(combwi[0])+str(combwi[1])
                        for combwj in W2_combs:
                            combw2 = str(combwj[0])+str(combwj[1])
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
            df.to_pickle(outDir_+file_+'_'+sample+'.pkl')
            del df, chi2_df
            #
        #
    #
#
def evaluateChi2(files_, samples_, outDir_):
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
    #
    getCombRTDisc(df)
    for key in df.keys():
        cut    = (df[key]['val']['bestRecoZPt'] > 300)
        print(key, len(df[key]['df'].dropna()), len(df[key]['df']['Top1_disc'].dropna()), len(df[key]['df']['Top2_disc'].dropna()))
        print(key, len(df[key]['df'][cut].dropna()), len(df[key]['df']['Top1_disc'][cut].dropna()), len(df[key]['df']['Top2_disc'][cut].dropna()))
    exit()
    #
    def plotChi2(df_,cut_):
        plt.figure()
        for key in df_.keys():
            cut    = (df_[key]['val']['bestRecoZPt'] > cut_)
            weight = df_[key]['val']['weight'][cut] * np.sign(df_[key]['val']['genWeight'][cut])
            plt.hist(x = df_[key]['df']['Chi2'][cut], bins=20, 
                     range=(0,20), 
                     histtype= 'step', 
                     weights= weight, density=True, label= key)
            #
        #
        plt.title('Z pt > '+str(cut_))
        plt.grid(True)
        #plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('${\chi^2}$')
        plt.legend()
        plt.show()
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
    plotChi2(df, 0)
    plotChi2(df, 100)
    plotChi2(df, 200)
    plotChi2(df, 300)
    plotChi2Comb(df)
    #
#
if __name__ == '__main__':
    files_samples_outDir = cfg.kinemFitCfg
    #
    #prD.getData(   *files_samples_outDir, *cfg.kinemFitCut, cfg.kinemFitMaxJets)
    #prD.interpData(*files_samples_outDir, cfg.kinemFitMaxJets)
    #
    computeChi2(   *files_samples_outDir, cfg.kinemFitMaxJets)
    #evaluateChi2(  *files_samples_outDir)
