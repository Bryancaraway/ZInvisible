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
        print(x_.shape)
        x_ = np.swapaxes(x_,1,2)
        print(x_.shape)
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
    #prD.getData(   *files_samples_outDir, *cfg.cnnCut, cfg.cnnMaxJets)
    #prD.interpData(*files_samples_outDir, cfg.cnnMaxJets)
    prD.preProcess(*files_samples_outDir, *cfg.cnn_data_dir)
    #
    reshapeInput(*cfg.cnn_data_dir, cfg.cnnMaxJets)
