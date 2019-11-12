########################
### Process data     ###
### for training     ###
########################
### written by:      ###
### Bryan Caraway    ###
########################
##
import ROOT
from ROOT import TLorentzVector
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
                defineKeys(label,cfg.label)
                # Extract LVec info
                def extractLVecInfo(lvecdict):
                    for key in lvecdict.keys():
                        lvecdict['Pt']  = lvecdict[key].pt
                        lvecdict['Eta'] = lvecdict[key].eta
                        lvecdict['Phi'] = lvecdict[key].phi
                        print(len(lvecdict['Pt']))
                        lvecdict['E']   = lvecdict[key].E
                        
                    del lvecdict[key]
                #
                extractLVecInfo(ak4lvec)
                # Cuts for initial round of training #
                # Ak4 Jet Pt > 30, Ak4 Jet Eta < 2.6 #
                # after which nJet = 6               #
                ak4_cuts = ((ak4lvec['Pt'] > 30) & (abs(ak4lvec['Eta']) < 2.6) 
                             & (abs(ak4vars['Jet_btagCSVV2']) <= 1) & (abs(ak4vars['Jet_btagDeepB']) <= 1) & (abs(ak4vars['Jet_qg']) <= 1))
                def applyAK4Cuts(dict_,isJetVec, cuts_):
                    for key in dict_.keys():
                        if (isJetVec) : dict_[key] = dict_[key][cuts_] ## bool switch might work better with try! statement
                        dict_[key]  = dict_[key][(cuts_).sum() == 6]
                #
                applyAK4Cuts(ak4vars, True,  ak4_cuts)
                applyAK4Cuts(ak4lvec, True,  ak4_cuts)
                applyAK4Cuts(label,   False, ak4_cuts)
                del ak4_cuts
                # Add to dataframe #
                def addToDF(dict_,isJetVec,dfs):
                    for key in dict_.keys():
                        #print(key)
                        df_temp = pd.DataFrame.from_dict(dict_)
                        key_list = []
                        nKeys = 0
                        if (isJetVec) :
                            nKeys = len(df_temp[key][0])
                            for i in xrange(nKeys):
                                key_list.append(key+'_'+str(i+1))
                            df_temp = pd.DataFrame(df_temp[key].values.tolist(), columns = key_list )
                        dfs = pd.concat([dfs,df_temp],axis=1)
                    return dfs
                #
                dfs = pd.DataFrame()
                dfs = addToDF(ak4vars,True, dfs)
                del ak4vars
                dfs = addToDF(ak4lvec,True, dfs)
                del ak4lvec
                dfs = addToDF(label,  False,dfs)
                del label
                dfs = dfs.dropna()
                #reduce memory usage of DF by converting float64 to float32
                def reduceDF(df_):
                    for key in df_.keys():
                        if str(df_[key].dtype) == 'float64': 
                            df_[key] = df_[key].astype('float32')
                #
                reduceDF(dfs)
                #df_ = pd.concat([df_,dfs], ignore_index=True)
                dfs.to_pickle(cfg.skim_dir+file_+'_'+sample+'.pkl')
                del dfs
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
                dr_combs   = list(combinations(xrange(1,6+1),2))
                invM_combs = list(combinations(xrange(1,6+1),3))                
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
            df = pd.concat([df,pd.read_pickle(cfg.skim_dir+file_+'_'+sample+'.pkl')], ignore_index = True)
    #
    ##### Seperate DF diffinitively #######
    trainX = df.sample(frac=0.70,random_state=1)
    testX  = df.drop(trainX.index).copy()
    valX   = trainX.sample(frac=0.30,random_state=1)
    trainDF = trainX.drop(valX.index).copy()
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
    def resetIndex(df_):
        return df_.reset_index(drop=True).copy()
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

class ProcessData:
    # parse data from input files getten from config
    def __init__(self, period, variables):
        self.preiod = period
        self.vars   = variables
        self.file   = None
        
    
if __name__ == '__main__':
    preProcess()
