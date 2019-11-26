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

def getFiles():
    return cfg.files

def getData():
    files = getFiles()
    for file_ in files:
        if not os.path.exists(cfg.file_path+file_+'.root') : continue
        with uproot.open(cfg.file_path+file_+'.root') as f_:
            print('Opening File:\t{}'.format(file_))
            for sample in cfg.MCsamples:
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
                defineKeys(ak4vars,cfg.ak4vars)
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
                ak4_cuts = ((ak4lvec['Pt'] > 30) & (abs(ak4lvec['Eta']) < 2.6) 
                             & (abs(ak4vars['Jet_btagCSVV2']) <= 1) & (abs(ak4vars['Jet_btagDeepB']) <= 1) & (abs(ak4vars['Jet_qg']) <= 1))
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
                dfs     = addToDF(ak4vars, dfs)
                del ak4vars
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
                dfs.to_pickle(    cfg.skim_dir+file_+'_'    +sample+'.pkl')
                val_dfs.to_pickle(cfg.skim_dir+file_+'_'+sample+'_val.pkl')
                del dfs
                del val_dfs
                #
            #
        #
    #
#   
def interpData():
    files = getFiles()
    for file_ in files:
        for sample in cfg.MCsamples:
            if not os.path.exists(cfg.skim_dir+file_+'_'+sample+'.pkl') : continue
            df = pd.DataFrame()
            df = pd.read_pickle(cfg.skim_dir+file_+'_'+sample+'.pkl')
            #
            def computeCombs(df_):
                # DO THE CALCS BY HAND SO THAT IS IS DONE IN PARALLEL
                dr_combs   = list(combinations(range(1,6+1),2))
                invM_combs = list(combinations(range(1,6+1),3))                
                for comb in dr_combs:
                    deta = df_['Eta_'+str(comb[0])] - df_['Eta_'+str(comb[1])]
                    dphi = df_['Phi_'+str(comb[0])] - df_['Phi_'+str(comb[1])]
                    df_['dR_'+str(comb[0])+str(comb[1])] = np.sqrt(np.power(deta,2)+np.power(dphi,2))
                #
                for comb in invM_combs:
                    E_sum2  = np.power(df_['E_'+str(comb[0])] + df_['E_'+str(comb[1])] + df_['E_'+str(comb[2])],2)
                    p_xmag2 = np.power((df_['Pt_'+str(comb[0])]*np.cos(df_['Phi_'+str(comb[0])]))+(df_['Pt_'+str(comb[1])]*np.cos(df_['Phi_'+str(comb[1])]))+(df_['Pt_'+str(comb[2])]*np.cos(df_['Phi_'+str(comb[2])])),2)
                    p_ymag2 = np.power((df_['Pt_'+str(comb[0])]*np.sin(df_['Phi_'+str(comb[0])]))+(df_['Pt_'+str(comb[1])]*np.sin(df_['Phi_'+str(comb[1])]))+(df_['Pt_'+str(comb[2])]*np.sin(df_['Phi_'+str(comb[2])])),2)
                    p_zmag2 = np.power((df_['Pt_'+str(comb[0])]*np.sinh(df_['Eta_'+str(comb[0])]))+(df_['Pt_'+str(comb[1])]*np.sinh(df_['Eta_'+str(comb[1])]))+(df_['Pt_'+str(comb[2])]*np.sinh(df_['Eta_'+str(comb[2])])),2)
                    p_mag2 = p_xmag2 + p_ymag2 + p_zmag2
                    del  p_xmag2,p_ymag2,p_zmag2
                    df_['InvM_'+str(comb[0])+str(comb[1])+str(comb[2])] = np.sqrt(E_sum2 - p_mag2)
                return df_
                #
            #
            df = computeCombs(df)
            df.to_pickle(cfg.skim_dir+file_+'_'+sample+'.pkl')
            del df
            #
        #
    #
#
def preProcess():
    df = pd.DataFrame()
    files = getFiles()
    for file_ in files:
        for sample in cfg.MCsamples:
            if not os.path.exists(cfg.skim_dir+file_+'_'+sample+'.pkl') : continue
            df = pd.concat([df,pd.read_pickle(cfg.skim_dir+file_+'_'+sample+'.pkl')], axis=1)
    #
    ##### Seperate DF diffinitively #######
    trainX = df.sample(frac=0.70,random_state=1)      ## make sure these lengths make sense 
    testX  = df.drop(trainX.index).copy()             ## between drops: 
    print(('Test:\t{}\n').format(len(testX)))
    valX   = trainX.sample(frac=0.30,random_state=1)  ## need to fix!!!!!!!!!!!!!!!
    print(('Val:\t{}\n').format(len(valX)))
    trainX = trainX.drop(valX.index).copy()          ##
    print(('Train:\t{}\n').format(len(trainX)))
    print(('Total:\t{}\n').format(len(trainX)+len(valX)+len(testX)))
    del df
    #
    #### Get Labels for val,train,test ####
    for label in cfg.label:
        trainY = trainX[label].copy()
        del trainX[label]
        valY = valX[label].copy()
        del valX[label]
        testY = testX[label].copy()
        del testX[label]
    #
    trainX = resetIndex(trainX)
    trainY = resetIndex(trainY)
    #
    valX = resetIndex(valX)
    valY = resetIndex(valY)
    #
    testX = resetIndex(testX)
    testY = resetIndex(testY)
    ### Store ###
    trainX.to_pickle(cfg.train_dir+'X.pkl')
    trainY.to_pickle(cfg.train_dir+'Y.pkl')
    #
    valX.to_pickle(cfg.val_dir+'X.pkl')
    valY.to_pickle(cfg.val_dir+'Y.pkl')
    #
    testX.to_pickle(cfg.test_dir+'X.pkl')
    testY.to_pickle(cfg.test_dir+'Y.pkl')
    #
    del testX, testY, trainX, trainY, valX, valY
#
def doOverSampling():
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import recall_score
    #
    sm = SMOTE(random_state=10, ratio=0.35)
    trainX = pd.read_pickle(cfg.train_dir+'X.pkl')
    trainY = pd.read_pickle(cfg.train_dir+'Y.pkl')
    #
    trainX_ndarray, trainY_ndarray = sm.fit_sample(trainX, trainY)
    #
    trainX = pd.DataFrame(trainX_ndarray, columns = list(trainX.keys()))
    print(trainX)

    trainY = pd.Series(trainY_ndarray, name = trainY.name)
    print(trainY)
    trainX.to_pickle(cfg.train_over_dir+'X.pkl')
    trainY.to_pickle(cfg.train_over_dir+'Y.pkl')
    #
    del trainX, trainY
#
def resetIndex(df_):
    return df_.reset_index(drop=True).copy()

if __name__ == '__main__':
    #getData()
    #interpData()
    #preProcess()
    doOverSampling()
