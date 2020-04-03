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
import math
import pickle
import operator
import deepsleepcfg as cfg
#
import numpy as np
np.random.seed(0)
import pandas as pd
from itertools import combinations
##

def getData(files_ = cfg.files, samples_ = cfg.MCsamples, outDir_ = cfg.skim_dir, blOps_ = operator.eq, njets_ = 6, maxJets_ = 6 , ZptCut_ = 0, treeDir_ = cfg.tree_dir, getGenData_ = False, getak8var_ = False):
    files = files_
    for file_ in files:
        if not os.path.exists(cfg.file_path+file_+'.root') : continue
        with uproot.open(cfg.file_path+file_+'.root') as f_:
            print('Opening File:\t{}'.format(file_))
            t_ = f_.get(treeDir_)
            for sample in samples_:
            #for sample in samples_:
                print(sample)
                t = t_.get(sample)
                getGenData = getGenData_
                if ((sample != 'TTZ') and (sample != 'TTBarLep') and (sample != 'TTZH')): getGenData = False
                #t = t_.get(sample)
                ak4vars = {}
                ak4lvec = {}
                genData = {}
                ak8vars = {}
                ak8lvec = {}
                #selvar  = {'nJets':t.array('nJets30_drLeptonCleaned')} ##### temporary, only train on 6 ak4 jet events
                selvar   = {'nJets':t.array('Jet_pt_drLeptonCleaned').counts}
                valRCvars = {}
                valvars  = {}
                label   = {}
                def defineKeys(dict_,keys_):
                    for key_ in keys_:
                        key = key_ 
                        if ('_drLeptonCleaned' in key) : 
                            key = key.replace('_drLeptonCleaned','')
                        if   ('Jet_' in key and 'FatJet_' not in key) :
                            key = key.replace('Jet_', '')
                        elif ('FatJet_' in key) :
                            key = key.replace('FatJet_', '')
                        dict_[key] = t.array(key_)[((selvar['nJets'] >= (njets_-1)) & (selvar['nJets'] <= maxJets_))]
                        #
                    #
                #
                # Extract LVec info
                def extractLVecInfo(lvecdict):
                    keys = list(lvecdict.keys())
                    for key in keys:
                        lvecdict['pt']  = lvecdict[key].pt
                        lvecdict['eta'] = lvecdict[key].eta
                        lvecdict['phi'] = lvecdict[key].phi
                        lvecdict['E']   = lvecdict[key].E
                        
                    del lvecdict[key]
                #
                #try:
                #    defineKeys(ak4lvec,cfg.ak4lvec['TLV'])
                #    extractLVecInfo(ak4lvec)
                #except:
                defineKeys(ak4lvec,   cfg.ak4lvec['TLVarsLC'])
                defineKeys(valRCvars, cfg.ak4lvec['TLVars'])
                defineKeys(valRCvars, cfg.valRCvars)
                #
                defineKeys(ak4vars,cfg.ak4vars)
                defineKeys(valvars,cfg.valvars)
                defineKeys(label,  cfg.label)
                #
                if (getak8var_):
                    defineKeys(ak8vars,cfg.ak8vars)
                    defineKeys(ak8lvec,cfg.ak8lvec['TLVarsLC'])
                #
                if (getGenData):
                    defineKeys(genData,cfg.genpvars)
                #
                del selvar
                # Cuts for initial round of training #
                # Ak4 Jet Pt > 30, Ak4 Jet Eta < 2.6 #
                # after which nJet cut, check cfg    #
                ak4_cuts = ((ak4lvec['pt'] > 20) & (abs(ak4lvec['eta']) < 2.6) 
                            & (abs(ak4vars['btagCSVV2']) <= 1) & (abs(ak4vars['btagDeepB']) <= 1) & (abs(ak4vars['qgl']) <= 1))
                zptcut = (valvars['bestRecoZPt'] >= ZptCut_)
                #
                def applyAK4Cuts(dict_, cuts_, zptcut_, isak4=False):
                    for key in dict_.keys():
                        if isak4 : 
                            dict_[key] = dict_[key][cuts_] ## bool switch might work better with try! statement
                        dict_[key]  = dict_[key][(blOps_((cuts_).sum(), njets_)) & (cuts_.sum() <= maxJets_) & (zptcut_)]
                #
                applyAK4Cuts(ak4vars,   ak4_cuts, zptcut, isak4=True)
                applyAK4Cuts(ak4lvec,   ak4_cuts, zptcut, isak4=True)
                applyAK4Cuts(valRCvars, ak4_cuts, zptcut)
                applyAK4Cuts(valvars,   ak4_cuts, zptcut)
                applyAK4Cuts(label,     ak4_cuts, zptcut)
                #
                if (getak8var_):
                    applyAK4Cuts(ak8vars, ak4_cuts, zptcut)
                    applyAK4Cuts(ak8lvec, ak4_cuts, zptcut)
                #
                if (getGenData):
                    applyAK4Cuts(genData, ak4_cuts, zptcut)
                #
                del ak4_cuts, zptcut
                #
                sample_maxJets    = max(ak4lvec['pt'].counts)
                if (getak8var_):
                    sample_maxFatJets = max(ak8lvec['pt'].counts)
                valvars['nJets'] = ak4lvec['pt'].counts
                if (getak8var_):
                    valvars['nFatJets'] = ak8lvec['pt'].counts
                ##
                ##
                def CleanRTCJetIdx(RC_, LC_, RC_j1, RC_j2, RC_j3):
                    RC_j1j2j3 = []
                    def tryLVar(i,j,k,idx_,rcvar_,lcvar_):
                        RC_j1j2j3_ = []
                        inter, rc_ind, lc_ind = np.intersect1d(rcvar_,lcvar_, return_indices=True)
                        if len(rcvar_) != len(set(rcvar_)): 
                            raise ValueError('At index {}'.format(idx_))
                        rc_ind = np.sort(rc_ind)
                        lc_ind = np.sort(lc_ind)
                        for idx2_, (i_,j_,k_) in enumerate(zip(i, j, k)):
                            string_ = ''
                            if (len(np.where(rc_ind == i_)[0]) == 0 or len(np.where(rc_ind == j_)[0]) == 0 or len(np.where(rc_ind == k_)[0]) == 0):
                                string_ = '0.0.0'
                            else:
                                string_ = str(np.where(rc_ind == i_)[0].item()+1)+'.'+str(np.where(rc_ind == j_)[0].item()+1)+'.'+str(np.where(rc_ind == k_)[0].item()+1)
                            RC_j1j2j3_.append(string_)
                            #
                        return RC_j1j2j3_
                            
                    for idx1_, (j1_, j2_, j3_) in enumerate(zip(RC_j1, RC_j2, RC_j3)):
                        var_ = ['pt', 'eta', 'phi', 'E']
                        for v_ in var_: 
                            try:
                                RC_j1j2j3.append(tryLVar(j1_,j2_,j3_,idx1_,
                                                         RC_[v_][idx1_],LC_[v_][idx1_]))
                                break
                            except ValueError:
                                if (v_ == var_[-1]) :
                                    print('Matched value in {0}. Worst case scinario (throwing away event at index {1})'.format(var_,idx1_))
                                    RC_j1j2j3.append([])
                                    continue
                                    #raise ValueError('Matched value in {}. Worst case scinario (throw away event?)'.format(var_))
                                else : continue
                            #
                        #
                    #
                    return RC_j1j2j3
                ###########
                valRCvars['ResolvedTopCandidate_j1j2j3Idx'] = CleanRTCJetIdx(
                    valRCvars, ak4lvec, 
                    valRCvars['ResolvedTopCandidate_j1Idx'], 
                    valRCvars['ResolvedTopCandidate_j2Idx'], 
                    valRCvars['ResolvedTopCandidate_j3Idx'])
                #
                del valRCvars['ResolvedTopCandidate_j1Idx'] 
                del valRCvars['ResolvedTopCandidate_j2Idx'] 
                del valRCvars['ResolvedTopCandidate_j3Idx']
                del valRCvars['pt'], valRCvars['eta'], valRCvars['phi'], valRCvars['E']
                ###########
                # Add to dataframe #
                def addToDF(dict_, df_, n_colpervar=None):
                    for key in dict_.keys():
                        df_temp = pd.DataFrame.from_dict(dict_)
                        key_list = []
                        if n_colpervar:
                            nVarPerKey = n_colpervar #len(df_temp[key][0])
                            #nVarPerKey = maxJets_
                            for i in range(0,nVarPerKey):
                                key_list.append(key+'_'+str(i+1))
                            df_temp = pd.DataFrame(df_temp[key].values.tolist(), columns = key_list )
                            df_ = pd.concat([df_,df_temp],axis=1)
                        else:
                            df_ = pd.concat([df_,df_temp[key]],axis=1)

                    return df_
                #
                dfs       = pd.DataFrame()
                val_dfs   = pd.DataFrame()
                #
                dfs     = addToDF(ak4vars, dfs, sample_maxJets)
                del ak4vars
                dfs     = addToDF(ak4lvec, dfs, sample_maxJets)
                del ak4lvec
                val_dfs = addToDF(valvars, val_dfs)
                del valvars
                dfs     = addToDF(label,   dfs)
                del label
                #
                #if (getak8var_):
                #    ak8_dfs   = pd.DataFrame()
                #    ak8_dfs = addToDF(ak8vars, ak8_dfs, sample_maxFatJets)
                #    del ak8vars
                #    ak8_dfs = addToDF(ak8lvec, ak8_dfs, sample_maxFatJets)
                #    del ak8lvec
                #
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
                dfs.to_pickle(      outDir_+file_+'_'+sample+'.pkl')
                val_dfs.to_pickle(  outDir_+file_+'_'+sample+'_val.pkl')
                with open(outDir_+file_+'_'+sample+'_valRC.pkl', 'wb') as handle:
                    pickle.dump(valRCvars, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if (getGenData) :
                    print(genData)
                    with open(outDir_+file_+'_'+sample+'_gen.pkl'   ,'wb') as handle:
                        pickle.dump(genData, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if (getak8var_):
                    #ak8_dfs.to_pickle(  outDir_+file_+'_'+sample+'_ak8.pkl')
                    #del ak8_dfs
                    with open(outDir_+file_+'_'+sample+'_ak8.pkl'   ,'wb') as handle:
                        ak8_dict = {**ak8vars, **ak8lvec}
                        pickle.dump(ak8_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                del dfs
                del val_dfs
                del valRCvars
                del genData
                #
            #
        #
    #
#   
def interpData(files_ = cfg.files, samples_ = cfg.MCsamples, outDir_ = cfg.skim_dir, njets_ = 6):
    files = files_
    for file_ in files:
        for sample in samples_:
            if not os.path.exists(outDir_+file_+'_'+sample+'.pkl') : continue
            df = pd.DataFrame()
            df = pd.read_pickle(outDir_+file_+'_'+sample+'.pkl')
            #
            sample_maxJets = max(pd.read_pickle(outDir_+file_+'_'+sample+'_val.pkl')['nJets'])
            print(sample, sample_maxJets)
            #
            def computeCombs(df_):
                # DO THE CALCS BY HAND SO THAT IS IS DONE IN PARALLEL
                dr_combs   = list(combinations(range(1,sample_maxJets+1),2))
                invTM_combs = list(combinations(range(1,sample_maxJets+1),3))                
               
                for comb in dr_combs:
                    deta = df_['eta_'+str(comb[0])] - df_['eta_'+str(comb[1])]
                    dphi = df_['phi_'+str(comb[0])] - df_['phi_'+str(comb[1])]
                    dphi = pd.concat([dphi.loc[dphi > math.pi] - 2*math.pi, 
                                      dphi.loc[dphi <= -math.pi] + 2*math.pi, 
                                      dphi.loc[(dphi <= math.pi) & (dphi > -math.pi)]]).sort_index()
                    df_['dR_'+str(comb[0])+'.'+str(comb[1])] = np.sqrt(np.power(deta,2)+np.power(dphi,2))
                    del deta, dphi
                    #
                    pt1pt2   = 2 * df_['pt_'+str(comb[0])] * df_['pt_'+str(comb[1])]  
                    cosheta  = np.cosh(df_['eta_'+str(comb[0])] - df_['eta_'+str(comb[1])])
                    cosphi   = np.cos( df_['phi_'+str(comb[0])] - df_['phi_'+str(comb[1])])
                    df['InvWM_'+str(comb[0])+'.'+str(comb[1])] = np.sqrt(pt1pt2 * (cosheta - cosphi))
                #
                for comb in invTM_combs:
                    E_sum2  = np.power(df_['E_'+str(comb[0])] + df_['E_'+str(comb[1])] + df_['E_'+str(comb[2])],2)
                    p_xmag2 = np.power((df_['pt_'+str(comb[0])]*np.cos(df_['phi_'+str(comb[0])]))+(df_['pt_'+str(comb[1])]*np.cos(df_['phi_'+str(comb[1])]))+(df_['pt_'+str(comb[2])]*np.cos(df_['phi_'+str(comb[2])])),2)
                    p_ymag2 = np.power((df_['pt_'+str(comb[0])]*np.sin(df_['phi_'+str(comb[0])]))+(df_['pt_'+str(comb[1])]*np.sin(df_['phi_'+str(comb[1])]))+(df_['pt_'+str(comb[2])]*np.sin(df_['phi_'+str(comb[2])])),2)
                    p_zmag2 = np.power((df_['pt_'+str(comb[0])]*np.sinh(df_['eta_'+str(comb[0])]))+(df_['pt_'+str(comb[1])]*np.sinh(df_['eta_'+str(comb[1])]))+(df_['pt_'+str(comb[2])]*np.sinh(df_['eta_'+str(comb[2])])),2)
                    p_mag2 = p_xmag2 + p_ymag2 + p_zmag2
                    del  p_xmag2,p_ymag2,p_zmag2
                    df_['InvTM_'+str(comb[0])+'.'+str(comb[1])+'.'+str(comb[2])] = np.sqrt(E_sum2 - p_mag2)
                return df_
                #
            #
            df = computeCombs(df)
            print(df)
            df.to_pickle(outDir_+file_+'_'+sample+'.pkl')
            del df
            #
        #
    #
#
def preProcess(files_ = cfg.files, samples_ = cfg.MCsamples, outDir_ = cfg.skim_dir,
               trainDir_ = cfg.train_dir, testDir_ = cfg.test_dir, valDir_ = cfg.val_dir):
    df     = pd.DataFrame()
    df_val = pd.DataFrame()
    df_aux = pd.DataFrame()
    files = files_
    for file_ in files:
        for sample in samples_:
            if not os.path.exists(outDir_+file_+'_'+sample+'.pkl') : continue
            df     = pd.concat([df,pd.read_pickle(outDir_+file_+'_'+sample+'.pkl')],     ignore_index = True, sort=False)
            df_val = pd.concat([df_val,pd.read_pickle(outDir_+file_+'_'+sample+'_val.pkl')], ignore_index = True, sort=False)
            #
            temp_ = pd.concat([df, df_val], axis=1, sort = False)
            temp_['Sample'] = sample
            temp_['Year']   = file_.strip('result_')
            df_aux          = pd.concat([df_aux,temp_], ignore_index = True, sort=False)
            
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
    df_val = df_val.drop(columns=['genWeight' ,'weight']).copy()
    #
    trainX_val = df_val.iloc[trainX.index]
    testX_val = df_val.iloc[testX.index]
    valX_val = df_val.iloc[valX.index]
    #
    train_aux = df_aux.iloc[trainX.index]
    test_aux = df_aux.iloc[testX.index]
    val_aux = df_aux.iloc[valX.index]
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
    trainX_val = resetIndex(trainX_val)
    trainY = resetIndex(trainY)
    train_aux = resetIndex(train_aux)
    #
    valX = resetIndex(valX)
    valX_val = resetIndex(valX_val)
    valY = resetIndex(valY)
    val_aux = resetIndex(val_aux)
    #
    testX = resetIndex(testX)
    testX_val = resetIndex(testX_val)
    testY = resetIndex(testY)
    test_aux = resetIndex(test_aux)
    print(test_aux)
    ### Store ###
    trainX.to_pickle(trainDir_+'X.pkl')
    trainX_val.to_pickle(trainDir_+'X_val.pkl')
    trainY.to_pickle(trainDir_+'Y.pkl')
    train_aux.to_pickle(trainDir_+'_aux.pkl')
    #
    valX.to_pickle(valDir_+'X.pkl')
    valX_val.to_pickle(valDir_+'X_val.pkl')
    valY.to_pickle(valDir_+'Y.pkl')
    val_aux.to_pickle(valDir_+'_aux.pkl')
    #
    testX.to_pickle(testDir_+'X.pkl')
    testX_val.to_pickle(testDir_+'X_val.pkl')
    testY.to_pickle(testDir_+'Y.pkl')
    test_aux.to_pickle(testDir_+'_aux.pkl')
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
    print(trainX)
    print(trainY)
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
    getData()
    interpData()
    #preProcess()
    #doOverSampling()