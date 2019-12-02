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
import deepsleepcfg as cfg
#
import numpy as np
np.random.seed(0)
import pandas as pd
from itertools import combinations
##

def getData():
    files = ['result_2017']     
    for file_ in files:
        if not os.path.exists(cfg.file_path+file_+'.root') : continue
        with uproot.open(cfg.file_path+file_+'.root') as f_:
            print('Opening File:\t{}'.format(file_))
            samples = ['TTZ','DY']
            for sample in samples:
                print(sample)
                t = f_.get(cfg.tree_dir+'/'+sample)
                ak4vars = {}
                ak4lvec = {}
                #ak8vars = {}
                #ak8lvec = {}
                selvar  = {'nJets':t.array('nJets30_drLeptonCleaned')} ##### temporary, only train on 6 ak4 jet events
                valvars  = {}
                label   = {}
                def defineKeys(dict_,keys):
                    for key in keys:
                        if ('_drLeptonCleaned' in key) : 
                            dict_[key.strip('_drLeptonCleaned')] = t.array(key)[(selvar['nJets'] >= 3)]
                        else : 
                            dict_[key] = t.array(key)[(selvar['nJets'] >= 3)]
                #
                defineKeys(ak4lvec,cfg.ak4lvec)
                defineKeys(valvars,cfg.valvars)
                defineKeys(label,  cfg.label)
                # Extract LVec info
                def extractLVecInfo(lvecdict):
                    keys = list(lvecdict.keys())
                    for key in keys:
                        lvecdict['Pt']  = lvecdict[key].pt
                        lvecdict['Eta'] = lvecdict[key].eta
                        lvecdict['Phi'] = lvecdict[key].phi
                        lvecdict['E']   = lvecdict[key].E
                        
                    del lvecdict[key]
                #
                extractLVecInfo(ak4lvec)
                # Cuts for initial round of training #
                # Ak4 Jet Pt > 30, Ak4 Jet Eta < 2.6 #
                # after which nJet = 6               #
                ak4_cuts = ((ak4lvec['Pt'] > 30) & (abs(ak4lvec['Eta']) < 2.6) )
                #
                def applyAK4Cuts(dict_, cuts_):
                    for key in dict_.keys():
                        try : 
                            dict_[key] = dict_[key][cuts_] ## bool switch might work better with try! statement
                        except:
                            pass
                        dict_[key]  = dict_[key][(cuts_).sum() == 6]
                #
                applyAK4Cuts(ak4vars, ak4_cuts)
                applyAK4Cuts(ak4lvec, ak4_cuts)
                applyAK4Cuts(valvars, ak4_cuts)
                applyAK4Cuts(label,   ak4_cuts)
                del ak4_cuts
                # Add to dataframe #
                def addToDF(dict_, df_):
                    for key in dict_.keys():
                        df_temp = pd.DataFrame.from_dict(dict_)
                        key_list = []
                        try:
                            nVarPerKey = len(df_temp[key][0])
                            for i in range(0,nVarPerKey):
                                key_list.append(key+'_'+str(i+1))
                            df_temp = pd.DataFrame(df_temp[key].values.tolist(), columns = key_list )
                            df_ = pd.concat([df_,df_temp],axis=1)
                        except:
                            df_ = pd.concat([df_,df_temp[key]],axis=1)

                    return df_
                #
                dfs     = pd.DataFrame()
                val_dfs = pd.DataFrame()
                #
                dfs     = addToDF(ak4lvec, dfs)
                del ak4lvec
                val_dfs = addToDF(valvars, val_dfs)
                del valvars
                dfs     = addToDF(label,   dfs)
                del label
                dfs     = dfs.dropna()
                val_dfs = val_dfs.dropna()
                #reduce memory usage of DF by converting float64 to float32
                def reduceDF(df_):
                    for key in df_.keys():
                        if str(df_[key].dtype) == 'float64': 
                            df_[key] = df_[key].astype('float32')
                    return df_
                #
                dfs     = reduceDF(dfs)
                val_dfs = reduceDF(val_dfs)
                print(dfs)
                print(val_dfs)
                dfs.to_pickle(    cfg.skim_test_dir+file_+'_'    +sample+'.pkl')
                val_dfs.to_pickle(cfg.skim_test_dir+file_+'_'+sample+'_val.pkl')
                del dfs
                del val_dfs
                #
            #
        #
    #
#
def interpData():
    files = ['result_2017']
    for file_ in files:
        samples = ['TTZ','DY']
        for sample in samples:
            if not os.path.exists(cfg.skim_test_dir+file_+'_'+sample+'.pkl') : continue
            df = pd.DataFrame()
            df = pd.read_pickle(cfg.skim_test_dir+file_+'_'+sample+'.pkl')
            #
            def computeCombs(df_):
                # DO THE CALCS BY HAND SO THAT IS IS DONE IN PARALLEL
                invTM_combs = list(combinations(range(1,6+1),3))                
                invWM_combs = list(combinations(range(1,6+1),2))
                for comb in invWM_combs:
                    pt1pt2  = 2 * df_['Pt_'+str(comb[0])] * df_['Pt_'+str(comb[1])]  
                    cosheta = np.cosh(df_['Eta_'+str(comb[0])] - df_['Eta_'+str(comb[1])])
                    cosphi  = np.cos( df_['Phi_'+str(comb[0])] - df_['Phi_'+str(comb[1])])
                    df['InvWM_'+str(comb[0])+str(comb[1])] = np.sqrt(pt1pt2 * (cosheta - cosphi))
                #
                for comb in invTM_combs:
                    E_sum2  = np.power(df_['E_'+str(comb[0])] + df_['E_'+str(comb[1])] + df_['E_'+str(comb[2])],2)
                    p_xmag2 = np.power((df_['Pt_'+str(comb[0])]*np.cos(df_['Phi_'+str(comb[0])]))+(df_['Pt_'+str(comb[1])]*np.cos(df_['Phi_'+str(comb[1])]))+(df_['Pt_'+str(comb[2])]*np.cos(df_['Phi_'+str(comb[2])])),2)
                    p_ymag2 = np.power((df_['Pt_'+str(comb[0])]*np.sin(df_['Phi_'+str(comb[0])]))+(df_['Pt_'+str(comb[1])]*np.sin(df_['Phi_'+str(comb[1])]))+(df_['Pt_'+str(comb[2])]*np.sin(df_['Phi_'+str(comb[2])])),2)
                    p_zmag2 = np.power((df_['Pt_'+str(comb[0])]*np.sinh(df_['Eta_'+str(comb[0])]))+(df_['Pt_'+str(comb[1])]*np.sinh(df_['Eta_'+str(comb[1])]))+(df_['Pt_'+str(comb[2])]*np.sinh(df_['Eta_'+str(comb[2])])),2)
                    p_mag2 = p_xmag2 + p_ymag2 + p_zmag2
                    del  p_xmag2,p_ymag2,p_zmag2
                    df_['InvTM_'+str(comb[0])+str(comb[1])+str(comb[2])] = np.sqrt(E_sum2 - p_mag2)
                return df_
                #
            #
            df = computeCombs(df)
            df.to_pickle(cfg.skim_test_dir+file_+'_'+sample+'.pkl')
            del df
            #
        #
    #
#
def computeChi2():
    T_mass = 173.2
    W_mass = 80.4
    files = ['result_2017']
    for file_ in files:
        samples = ['TTZ','DY']
        for sample in samples:
            if not os.path.exists(cfg.skim_test_dir+file_+'_'+sample+'.pkl') : continue
            df = pd.DataFrame()
            df = pd.read_pickle(cfg.skim_test_dir+file_+'_'+sample+'.pkl')
            chi2_df = pd.DataFrame()
            #
            invTM_combs = list(combinations(range(1,6+1),3))
            for i, combi in enumerate(invTM_combs):
                combt1 = str(combi[0])+str(combi[1])+str(combi[2])
                for combj in invTM_combs[i+1:]:
                    combt2 = str(combj[0])+str(combj[1])+str(combj[2])
                    for _ in combt1: combt2=combt2.replace(_,"")
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
                            chi2 =  (np.power((massT1 - T_mass),2)/(pow(40,2)) + np.power((massT2 - T_mass),2)/(pow(40,2)) + 
                                     np.power((massW1 - W_mass),2)/(pow(30,2)) + np.power((massW2 - W_mass),2)/(pow(30,2)))
                            chi2_df['Chi2_'+combt1+'_'+combw1+'_'+combt2+'_'+combw2] = chi2
                            #
                        #
                    #
                #
            #
            df['Chi2']      = chi2_df.min(axis=1)
            df['Chi2_Comb'] = chi2_df.idxmin(axis=1)
            df.to_pickle(cfg.skim_test_dir+file_+'_'+sample+'.pkl')
            del df, chi2_df
            #
        #
    #
#
def evaluateChi2():
    import matplotlib.pyplot as plt
    df = {}
    files = ['result_2017']
    for file_ in files:
        samples = ['TTZ','DY']
        for sample in samples:
            if not os.path.exists(cfg.skim_test_dir+file_+'_'+sample+'.pkl') : continue
            df_ = pd.DataFrame()
            val_df_ = pd.DataFrame()
            df_ = pd.read_pickle(cfg.skim_test_dir+file_+'_'+sample+'.pkl')
            val_df_ = pd.read_pickle(cfg.skim_test_dir+file_+'_'+sample+'_val.pkl')
            df[sample+file_.strip('result')] = {'df':df_, 'val':val_df_}
            #
        #
    #
    def plotChi2(df_,cut_):
        plt.figure()
        for key in df_.keys():
            cut    = (df_[key]['val']['bestRecoZPt'] > cut_)
            weight = df_[key]['val']['weight'][cut] * np.sign(df_[key]['val']['genWeight'][cut])
            plt.hist(x = df_[key]['df']['Chi2'][cut], bins=20, range=(0,2000), histtype= 'step', 
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
    #getData()
    #interpData()
    #computeChi2()
    evaluateChi2()
