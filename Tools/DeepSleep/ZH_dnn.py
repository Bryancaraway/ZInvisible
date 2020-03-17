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
import re
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
from sklearn import metrics 
##
def preProcess_DNN(files_, samples_, outDir_):
    df = kFit.retrieveData(files_, ['TTZH', 'TTBarLep'], outDir_, getak8_ = True)
    #
    dnn_df       = pd.DataFrame()
    train_df     = pd.DataFrame()
    test_df      = pd.DataFrame()
    val_df       = pd.DataFrame()
    aux_bkg_df   = pd.DataFrame()
    case_names    = ''
    #
    for key_ in df.keys():
        temp_df_ = pd.DataFrame()
        base_cuts = (
            (df[key_]['ak8']['n_nonHbb'] >= 2)    & 
            (df[key_]['ak8']['nhbbFatJets'] > 0)  &
            (df[key_]['ak8']['H_M']         > 50) &
            (df[key_]['ak8']['H_M']         < 200))
        if ('TTZH' in key_):
            base_cuts = base_cuts & (df[key_]['val']['matchedGen_ZHbb'] == True)
        if ('TTBarLep' in key_ ):
            case_names = re.findall(r'case_\d',''.join(df[key_]['val'].keys()))
            aux_bkg_df = df[key_]['val'][case_names]
            if (len(sys.argv) > 1):
                base_cuts = base_cuts & (aux_bkg_df[sys.argv[1]])
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
    #
    #  Create Aux df containing the background cases
    sig = dnn_df[dnn_df['Signal'] == True]
    bkg = dnn_df[dnn_df['Signal'] == False]
    bkg = pd.concat([bkg.reset_index(drop=True),aux_bkg_df[base_cuts].reset_index(drop=True)], axis=1)
    for name_ in case_names:
        sig[name_] = True
    pd.concat([sig,bkg],axis=0,ignore_index=True).to_pickle(cfg.aux_ZH_dir+'aux.pkl')
    ## Calculate DNN weighting
    dnn_df['DNNweight'] = pd.concat([dnn_df['weight'][dnn_df['Signal'] == True] * (dnn_df['weight'][dnn_df['Signal'] == False].sum() / dnn_df['weight'][dnn_df['Signal'] == True].sum()),
                                     dnn_df['weight'][dnn_df['Signal'] == False]])
    #
    import seaborn as sns
    plt.figure(figsize=(16,9))
    sns.set(font_scale=0.5)
    sns.heatmap(dnn_df.corr(), annot=True, fmt= '1.2f',annot_kws={"size": 6}, cmap=plt.cm.Reds, cbar=False, square= False)
    plt.title(sys.argv[1] if len(sys.argv) > 1 else 'General')
    plt.show()
    plt.close()
    ## Split into 50/30/20 (Train, Validation, Test)
    train_df = dnn_df.sample(frac = 0.80, random_state = 5) 
    dnn_df   = dnn_df.drop(train_df.index).copy()
    val_df   = train_df.sample(frac = 0.20, random_state = 4)
    train_df   = train_df.drop(val_df.index).copy()
    test_df  = dnn_df.sample(frac = 1.0)
    #
    train_dir , test_dir, val_dir = cfg.dnn_ZH_dir
    train_df.to_pickle(train_dir+'train.pkl')
    test_df .to_pickle(test_dir+ 'test.pkl')
    val_df  .to_pickle(val_dir+  'val.pkl')
    #
def train_DNN(train_dir, test_dir, val_dir):
    #
    train_df = pd.read_pickle(train_dir+'train.pkl')
    test_df  = pd.read_pickle(test_dir+ 'test.pkl')
    val_df   = pd.read_pickle(val_dir+  'val.pkl')
    #
    # Split into X, Y, weight
    #
    def splitXYW(df_):
        y_ = df_['Signal']
        w_ = df_[re.findall(r'\w*weight', ' '.join(df_.keys()))]
        x_ = df_.drop(columns=[y_.name, *w_.columns])
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
                       #sample_weight   = trainW['DNNweight'].values,
                       epochs          = cfg.dnn_ZH_epochs,
                       batch_size      = cfg.dnn_ZH_batch_size,
                       validation_data = (valX,valY.values.astype('int')),#valW['DNNweight'].values), 
                       #callbacks  = cbks,
                       verbose         = 1)
    import train as Tr
    if (cfg.dnn_ZH_epochs > 0):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        Tr.plot_history(hist)
        #
        model.save_weights(cfg.DNNoutputDir+cfg.DNNoutputName)
        model.save(cfg.DNNoutputDir+cfg.DNNmodelName)
    #
    loss ,acc, auc = model.evaluate(testX.values,testY.values)#, sample_weight = testW['DNNweight'].values)
    print("Test Set Acc: {0:.4f}\nTest Set AUC: {1:.4f}".format(acc,auc))
    X = pd.concat([testX,trainX,valX], ignore_index=True)
    Y = pd.concat([testY,trainY,valY], ignore_index=True)
    W = pd.concat([testW['weight'],trainW['weight'],valW['weight']], ignore_index=True)
    pred = model.predict(X.values).flatten()
    #
    df = pd.concat([X,Y,pd.Series(pred,name='Pred')], axis=1)
    import seaborn as sns
    plt.figure(figsize=(16,9))
    sns.set(font_scale=0.5)
    sns.heatmap(df.corr(), annot=True, fmt= '1.2f',annot_kws={"size": 6}, cmap=plt.cm.Reds, cbar=False, square= False)
    plt.title(sys.argv[1] if len(sys.argv) > 1 else 'General')
    #plt.show()
    plt.close()
    #
    fpr, tpr, thresholds = metrics.roc_curve(Y.astype('int'), pred)
    Tr.plot_roc(fpr,tpr)
    plt.hist(
        [pred[Y == True],pred[Y == False]] , 
        histtype='step',
        weights = [W[Y == True], W[Y == False]], 
        bins = 40,
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
def focal_loss(y_true, y_pred):
    gamma = cfg.fl_gamma
    alpha = cfg.fl_alpha
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
#
def Build_Model(inputs_,outputs_,mean_,std_):
    #
    if  ( int(cfg.skim_ZHbb_dir.split('_')[-1][:3]) == 200 ):
        main_input = keras.layers.Input(shape=[inputs_], name='input')

        layer      = keras.layers.Lambda(lambda x: (x - K.constant(mean_)) / K.constant(std_), name='normalizeData')(main_input)
        #
        #layer      = keras.layers.BatchNormalization()(layer) 
        #layer      = keras.layers.Dropout(0.0)(layer)
        layer      = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l1(0.00))(layer)
        layer      = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l1(0.00))(layer)
        layer      = keras.layers.Dropout(0.5)(layer) 
        #
        output     = keras.layers.Dense(1, activation='sigmoid', name='output')(layer)
        #
    elif( int(cfg.skim_ZHbb_dir.split('_')[-1][:3]) == 300 ):
        main_input = keras.layers.Input(shape=[inputs_], name='input')
        layer      = keras.layers.Lambda(lambda x: (x - K.constant(mean_)) / K.constant(std_), name='normalizeData')(main_input)
        #
        layer      = keras.layers.Dropout(0.0)(layer)
        layer      = keras.layers.Dense(128, activation='relu')(layer)
        layer      = keras.layers.Dense(64, activation='relu')(layer)
        layer      = keras.layers.Dropout(0.7)(layer) # .7
        #layer      = keras.layers.BatchNormalization()(layer)
        #
        output     = keras.layers.Dense(1, activation='sigmoid', name='output')(layer)
        #
    model      = keras.models.Model(inputs=main_input, outputs=output, name='model')
    optimizer  = keras.optimizers.Adam(learning_rate=cfg.dnn_ZH_alpha)# decay=cfg.dnn_ZH_alpha*1e-3, momentum=0.9, nesterov=True)
    #
    if (cfg.DNNuseWeights and os.path.exists(cfg.DNNoutputDir+cfg.DNNoutputName)): 
        model.load_weights(cfg.DNNoutputDir+cfg.DNNoutputName)
        #
    #from focal_loss import BinaryFocalLoss
    model.compile(loss=[focal_loss], optimizer=optimizer, metrics=['accuracy',tf.keras.metrics.AUC()])
    ##
    return model
def corr_study(train_dir, test_dir, val_dir):
    df = pd.read_pickle(cfg.aux_ZH_dir+'aux.pkl')
    case_df = df.filter(regex='case_\d', axis=1)
    df = df.drop(columns=case_df.keys())
    df = df.drop(columns='weight')
    #
    corr_df = pd.DataFrame()
    corr_df = pd.concat([df.corr()['Signal'].rename('General')]+list(df[case_df[key_] == True].corr()['Signal'].rename(key_) for key_ in case_df.keys()), axis=1)
    #
    import seaborn as sns
    from matplotlib.colors import LogNorm
    def plot_heatmap(score_df,title, norm_=False):
        plt.figure(figsize=(12,8))
        sns.set(font_scale=0.75)
        sns.heatmap(score_df, annot=True, fmt= '1.3f',annot_kws={"size": 6}, 
                    cmap=plt.cm.Reds, cbar=False, square= False, norm=(LogNorm() if norm_ else None))
        plt.title(title)
        plt.show()
        plt.close()
    #
    from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
    #from statsmodels.stats.multicomp import pairwise_tukeyhsd
    df_target = df['Signal']
    df = df.drop(columns=df_target.name)
    def plot_norm():
        for key_ in df.keys():
            n_,edges_,_ = plt.hist([df[key_][df_target == True], df[key_][df_target == False]],density=True, histtype='step', label=['True','False'])
            bin_centers = (edges_[1:]+edges_[:-1])/2
            bin_width   = edges_[1]-edges_[0]
            plt.errorbar(bin_centers,n_[0],yerr=np.sqrt(n_[0]/(sum(df_target == True) *bin_width)), color='tab:blue',   fmt='.')
            plt.errorbar(bin_centers,n_[1],yerr=np.sqrt(n_[1]/(sum(df_target == False)*bin_width)), color='tab:orange', fmt='.')
            plt.legend()
            plt.title(key_)
            plt.show()
            plt.close()
            #
    #plot_norm()
    #
    chi2_score, chi_2_p_value = chi2(abs(df),df_target)
    f_score, f_p_value = f_classif(df,df_target)
    mut_info_score = mutual_info_classif(df,df_target)
    #
    chi2_df   = pd.DataFrame(
        [chi2(abs(df),df_target)[0]]+list(chi2(abs(df[case_df[key_] == True]),df_target[case_df[key_] == True])[0] for key_ in case_df.keys()), 
        columns=df.keys(), index=['General']+list(case_df.keys())).transpose()
    chi2_p_df = pd.DataFrame(
        [chi2(abs(df),df_target)[1]]+list(chi2(abs(df[case_df[key_] == True]),df_target[case_df[key_] == True])[1] for key_ in case_df.keys()), 
        columns=df.keys(), index=['General']+list(case_df.keys())).transpose()
    f_classif_df   = pd.DataFrame(
        [f_classif(abs(df),df_target)[0]]+list(f_classif(abs(df[case_df[key_] == True]),df_target[case_df[key_] == True])[0] for key_ in case_df.keys()), 
        columns=df.keys(), index=['General']+list(case_df.keys())).transpose()
    f_classif_p_df = pd.DataFrame(
        [f_classif(abs(df),df_target)[1]]+list(f_classif(abs(df[case_df[key_] == True]),df_target[case_df[key_] == True])[1] for key_ in case_df.keys()), 
        columns=df.keys(), index=['General']+list(case_df.keys())).transpose()
    mut_info_classif_df   = pd.DataFrame(
        [mutual_info_classif(abs(df),df_target)]+list(mutual_info_classif(abs(df[case_df[key_] == True]),df_target[case_df[key_] == True]) for key_ in case_df.keys()), 
        columns=df.keys(), index=['General']+list(case_df.keys())).transpose()
    #
    plot_heatmap(corr_df,             'Pearson Corr',       False)
    plot_heatmap(chi2_df,             'Chi2',               True)
    plot_heatmap(chi2_p_df,           'Chi2_p_score',       False)
    plot_heatmap(f_classif_df,        'F_test',             True)
    plot_heatmap(f_classif_p_df,      'F_test_p_score',     False)
    plot_heatmap(mut_info_classif_df, 'mutual_information', False)
    #
    def Plot_score (df_,score_, title):
        plt.barh(np.arange(len(score_)),score_,tick_label=df_.keys())
        plt.title(title)
        plt.show()
        plt.close()
    #Plot_score(df, chi2_score,     'Chi2 score'       )
    #Plot_score(df, chi_2_p_value,  'Chi2, p-value'    )
    #Plot_score(df, f_score,        'F score'          )
    #Plot_score(df, f_p_value,      'F, p-value'       )
    #Plot_score(df, mut_info_score, 'Mutual info score')
    #
def pred_all_samples(files_, samples_, outDir_):
    df = kFit.retrieveData(files_, samples_, outDir_, getak8_ = True)
    train_dir, test_dir, val_dir = cfg.dnn_ZH_dir
    train_df = pd.read_pickle(train_dir+'train.pkl')
    trainX = resetIndex(train_df.drop(columns=[ 'Signal',*re.findall(r'\w*weight', ' '.join(train_df.keys()))]))
    nn_model = Build_Model(len(trainX.keys()),1,trainX.mean().values,trainX.std().values)
    nn_model.load_weights(str(cfg.DNNoutputDir+cfg.DNNoutputName))
    for key_ in df.keys():
        temp_df_ = pd.DataFrame()
        base_cuts = (
            (df[key_]['ak8']['n_nonHbb'] >= 2)   & 
            (df[key_]['ak8']['nhbbFatJets'] > 0) &
            (df[key_]['ak8']['H_M']         > 50)&
            (df[key_]['ak8']['H_M']         < 200))
        for var_ in cfg.dnn_ZH_vars:
            if 'weight' in var_ or 'Weight' in var_:
                continue
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
            #
        pred = nn_model.predict(temp_df_.values).flatten()

        df[key_]['val']['NN'] = -1.
        df[key_]['val']['NN'][base_cuts] = pred
        #
        year   = key_[key_.index('201'):key_.index('201')+4]
        sample = key_.split('_201')[0]
        df[key_]['val'].to_pickle(outDir_+'result_'+year+'_'+sample+'_val.pkl')
            
if __name__ == '__main__':   
    files_samples_outDir = cfg.ZHbbFitCfg
    ##
    #Ana.ZHbbAna(*files_samples_outDir, cfg.ZHbbFitoverlap)
    ##
    ##
    #preProcess_DNN(*files_samples_outDir)    
    #corr_study(    *cfg.dnn_ZH_dir)
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
    #train_DNN(     *cfg.dnn_ZH_dir)
    pred_all_samples(*files_samples_outDir)

