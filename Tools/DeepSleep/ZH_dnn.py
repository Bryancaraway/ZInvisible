#                      #  
##                    ##
########################                               
### Classify TTZ/H,  ###
### Z/H to bb, from  ###                               
### TTbar using DNN  ###                               
########################                               
### written by:      ###                               
### Bryan Caraway    ###                               
########################                               
##                    ##                                 
#                      #

##
import sys
import os
import pickle
import math
#
import deepsleepcfg as cfg
import processData  as prD 
import kinematicFit as kFit
import dottZHbb     as Ana
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fun_library as lib
from fun_library import fill1e, fillne, deltaR, deltaPhi, invM, calc_mtb
np.random.seed(0)
#
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
#tf.compat.v1.set_random_seed(2)
print(tf.__version__)
from tensorflow.python.keras import backend as K
from keras import backend as k
from tensorflow.python.ops import math_ops
from keras.models import Sequential
from keras.layers import Dense
#
from sklearn import metrics 
##
def preProcess_DNN(files_, samples_, outDir_):
    df = kFit.retrieveData(files_, ['TTZH', 'TTBarLep'], outDir_, getak8_ = True)
    #
    dnn_df   = pd.DataFrame()
    train_df = pd.DataFrame()
    test_df  = pd.DataFrame()
    val_df   = pd.DataFrame()
    #
    for key_ in df.keys():
        temp_df_ = pd.DataFrame()
        base_cuts = (
            (df[key_]['ak8']['n_nonHbb'] >= 2)   & 
            (df[key_]['ak8']['nhbbFatJets'] > 0) &
            (df[key_]['ak8']['H_M']         > 50)&
            (df[key_]['ak8']['H_M']         < 200))
        if ('TTZH' in key_):
            base_cuts = base_cuts & (df[key_]['val']['matchedGen_ZHbb'] == True)
        #print(df[key_]['val']['weight'][base_cuts])
        for var_ in cfg.dnn_ZH_vars:
            key_str = ''
            if (  var_ in df[key_]['df'].keys() ) :
                key_str = 'df'
            elif (var_ in df[key_]['val'].keys()) :
                key_str = 'val'
            elif (var_ in df[key_]['ak8'].keys()) :
                key_str = 'ak8'
            else:
                print(var_+' NOT FOUND IN '+key_+'!!!, FIX NOW AND RERUN!!!')
                exit()
            try:
                temp_df_[var_] = df[key_][key_str][var_][base_cuts].values
            except (AttributeError):
                temp_df_[var_] = df[key_][key_str][var_][base_cuts]
        temp_df_['Signal'] = key_.split('_')[0] == 'TTZH'
        temp_df_['weight'] = temp_df_['weight']*np.sign(temp_df_['genWeight'])*(137/41.9)
        print(key_)
        print(temp_df_['weight'].sum())
        del temp_df_['genWeight']
        dnn_df = pd.concat([dnn_df,temp_df_], axis=0, ignore_index = True)
        #
    ## Split into 50/30/20 (Train, Validation, Test)
    train_df = dnn_df.sample(frac = 0.50, random_state = 5) 
    dnn_df   = dnn_df.drop(train_df.index).copy()
    val_df   = dnn_df.sample(frac = 0.60, random_state = 4)
    dnn_df   = dnn_df.drop(val_df.index).copy()
    test_df  = dnn_df.sample(frac = 1.0)
    #
    train_dir , test_dir, val_dir = cfg.dnn_ZH_dir
    train_df.to_pickle(train_dir+'train.pkl')
    test_df .to_pickle(test_dir+ 'test.pkl')
    val_df  .to_pickle(val_dir+  'val.pkl')
    #
def train_DNN(train_dir, test_dir, val_dir):
    train_df = pd.read_pickle(train_dir+'train.pkl')
    test_df  = pd.read_pickle(test_dir+ 'test.pkl')
    val_df   = pd.read_pickle(val_dir+  'val.pkl')
    #
    # Split into X, Y, weight
    #
    def splitXYW(df_):
        y_ = df_['Signal']
        w_ = df_['weight']
        x_ = df_.drop(columns=['Signal','weight'])
        return resetIndex(x_), resetIndex(y_), resetIndex(w_)
    trainX, trainY, trainW = splitXYW(train_df)
    testX,  testY,  testW  = splitXYW(test_df)
    valX,   valY,   valW   = splitXYW(val_df)
    #
    model = Build_Model(len(trainX.keys()),1,trainX.mean().values,trainX.std().values)
    print(model.summary())
    #
    from keras.callbacks import LearningRateScheduler
    def scheduler():
        if epoch < cfg.epochs*.20:
            return 0.01
        if epoch < cfg.epochs*.70:
            return 0.005
        return 0.001
    #
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr]
    history =model.fit(trainX,
                       trainY.values.astype('int'),
                       #sample_weight   = trainW.values,
                       epochs          = cfg.dnn_ZH_epochs,
                       batch_size      = cfg.dnn_ZH_batch_size,
                       validation_data = (valX,valY.values.astype('int')),#,valW), 
                       #callbacks  = cbks,
                       verbose         = 1)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    import train as Tr
    Tr.plot_history(hist)
    #
    model.save_weights(cfg.DNNoutputDir+cfg.DNNoutputName)
    model.save(cfg.DNNoutputDir+cfg.DNNmodelName)
    #
    loss ,acc, auc = model.evaluate(testX.values,testY.values)#, sample_weight = testW.values)
    print("Test Set Acc: {0:.4f}\nTest Set AUC: {1:.4f}".format(acc,auc))
    X = pd.concat([testX,trainX,valX])
    Y = pd.concat([testY,trainY,valY])
    W = pd.concat([testW,trainW,valW])
    pred = model.predict(X.values).flatten()
    #
    fpr, tpr, thresholds = metrics.roc_curve(Y.astype('int'), pred)
    Tr.plot_roc(fpr,tpr)
    plt.hist(
        [pred[Y == True],pred[Y == False]] , 
        histtype='step',
        weights = [W[Y == True], W[Y == False]], 
        bins = 10,
        range = (0,1),
        label=['True','False'])
    plt.legend()
    plt.yscale('log')
    plt.show()
    
#
def resetIndex(df_):
    return df_.reset_index(drop=True).copy()
#
#
#
def Build_Model(inputs_,outputs_,mean_,std_):
    #
    main_input = keras.layers.Input(shape=[inputs_], name='input')
    layer      = keras.layers.Lambda(lambda x: (x - K.constant(mean_)) / K.constant(std_), name='normalizeData')(main_input)
    #
    layer      = keras.layers.Dropout(0.0)(layer)
    layer      = keras.layers.Dense(128, activation='relu')(layer)
    #layer      = keras.layers.Dropout(0.3)(layer)
    layer      = keras.layers.Dense(64, activation='relu')(layer)
    #layer      = keras.layers.Dense(32, activation='relu')(layer)
    layer      = keras.layers.Dropout(0.7)(layer)
    #layer      = keras.layers.Dense(66, activation='relu')(layer)
    #layer      = keras.layers.BatchNormalization()(layer)
    #
    output     = keras.layers.Dense(1, activation='sigmoid', name='output')(layer)
    #
    model      = keras.models.Model(inputs=main_input, outputs=output, name='model')
    optimizer  = keras.optimizers.Adam(learning_rate=cfg.dnn_ZH_alpha)# decay=cfg.dnn_ZH_alpha*1e-3, momentum=0.9, nesterov=True)
    #
    if (cfg.DNNuseWeights and os.path.exists(cfg.DNNoutputDir+cfg.DNNoutputName)): 
        model.load_weights(cfg.DNNoutputDir+cfg.DNNoutputName)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy',tf.keras.metrics.AUC()])
    ##
    return model

if __name__ == '__main__':   
    files_samples_outDir = cfg.ZHbbFitCfg
    ##
    #Ana.ZHbbAna(*files_samples_outDir, cfg.ZHbbFitoverlap)
    ##
    ##
    #preProcess_DNN(*files_samples_outDir)    
    train_DNN(     *cfg.dnn_ZH_dir)
