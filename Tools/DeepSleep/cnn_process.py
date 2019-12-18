########################
### Process data     ###
### for CNN training ###
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
def featureEng(files_, samples_, outDir_):
    def calcQscore(df_, df_valRC_, overlap_ = cfg.kinemFitoverlap):
        Q_dict = {'Q':[],'Q_comb':[]}
        #
        rtopd_  = df_valRC['ResolvedTopCandidate_discriminator']
        rtidx_  = df_valRC['ResolvedTopCandidate_j1j2j3Idx']
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
                    bscore_ = df_['btagDeepB_'+str(jetid_)].iloc[idx_]
                    if ( bscore_ > .90) :
                        continue
                        #
                q_score.append(rtopd[np.in1d(rtidx,i_[0])].item() + rtopd[np.in1d(rtidx,i_[1])].item())
                q_index.append(i_[0]+'_'+i_[1])
                #
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
        return Q_dict['Q']
        #
    #
    #######
    df       = pd.DataFrame()
    df_val   = pd.DataFrame()
    df_valRC = pd.DataFrame()
    files = files_
    for file_ in files:
        for sample in samples_:
            if not os.path.exists(outDir_+file_+'_'+sample+'.pkl') : continue
            df       = pd.read_pickle(outDir_+file_+'_'+sample+'.pkl')
            df_val   = pd.read_pickle(outDir_+file_+'_'+sample+'_val.pkl')
            df_valRC = pd.read_pickle(outDir_+file_+'_'+sample+'_valRC.pkl')
            # 
            df_val['Q'] = calcQscore(df, df_valRC)
            df_val.to_pickle(outDir_+file_+'_'+sample+'_val.pkl')
            #
        #
    #
#
                             

def reshapeInput(trainDir_, testDir_, valDir_, maxJets_):
    #
    trainX = pd.read_pickle(trainDir_+'X.pkl') 
    testX  = pd.read_pickle(testDir_ +'X.pkl') 
    valX   = pd.read_pickle(valDir_  +'X.pkl') 
    #
    def ReshapeForCNN(df_):
        x_ = None
        for i_ in range(1,maxJets_+1):
            vars_ = [s + '_'+str(i_) for s in cfg.cnn_vars]
            if i_ == 1 : 
                x_ = df_[vars_].values
            else:
                x_ = np.dstack((x_, df_[vars_].values))
            #
        #
        x_ = np.swapaxes(x_,1,2)
        print('steps: {}'.format(x_.shape[1]))
        print('features: {}'.format(x_.shape[2]))
        return x_
        #
    #
    trainCNNX = ReshapeForCNN(trainX)
    testCNNX = ReshapeForCNN(testX)
    valCNNX = ReshapeForCNN(valX)
    #
    with open(trainDir_+'X.pkl', 'wb') as handle:
        pickle.dump(trainCNNX, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(testDir_+'X.pkl', 'wb') as handle:
        pickle.dump(testCNNX, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(valDir_+'X.pkl', 'wb') as handle:
        pickle.dump(valCNNX, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
#

if __name__ == '__main__':
    files_samples_outDir = cfg.cnnProcessCfg
    #    
    #prD.getData(   *files_samples_outDir, *cfg.cnnCut, cfg.cnnMaxJets, ZptCut_ = 200)
    ##prD.interpData(*files_samples_outDir, cfg.cnnMaxJets)
    #featureEng(*files_samples_outDir)
    prD.preProcess(*files_samples_outDir, *cfg.cnn_data_dir)
    #
    reshapeInput(*cfg.cnn_data_dir, cfg.cnnMaxJets)
