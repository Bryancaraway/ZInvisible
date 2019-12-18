from __future__ import division
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
tf.compat.v1.set_random_seed(2)
print(tf.__version__)
#
import pandas as pd
import numpy as np
np.random.seed(0)
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics

#
import deepsleepcfg as cfg
#
########################################
# Note: This model must have a 3D input;
#       you should have a shape of:
#(nSamples, nSequences/Sample, nFeatures)
#######
# Additionally: nSequences/Sample does
#               not have to be constant
#               however, nFeatures does
########################################
def Build_1DCNN_Model(inShape, input_2, outputs_, meanAK4_,stdAK4_, mean_, std_) : 
    # 
    input_1 = layers.Input(shape=inShape, name='cnn_input') # (11,7,1)
    input_2 = layers.Input(shape=[input_2], name='dnn_input')
    mask    = layers.Masking(mask_value=-10)(input_1)
    norm_1  = layers.Lambda(lambda x: (x - K.constant(meanAK4_)) / K.constant(stdAK4_))(mask)
    norm_2  = layers.Lambda(lambda x: (x - K.constant(mean_)) / K.constant(std_))(input_2)
    dnn_1   = layers.Dense(16, activation='relu')(norm_2)
    #
    cnn_1 = layers.Conv1D(16, 3, strides=1, activation= 'relu')(norm_1)
    cnn_1 = layers.AveragePooling1D(pool_size=2)(cnn_1)
    cnn_1 = layers.Flatten()(cnn_1)
    #
    cnn_2 = layers.Conv1D(16, 1, strides=1, activation= 'relu')(input_1)
    cnn_2 = layers.Flatten()(cnn_2)
    #
    layer = layers.concatenate([cnn_1,cnn_2, dnn_1], axis = -1)
    #
    layer = layers.Dense(200, activation='relu')(layer)
    layer = layers.Dropout(0.4)(layer)
    layer = layers.Dense(32, activation='relu')(layer)
    #
    output = layers.Dense(outputs_, activation='sigmoid',name='Output')(layer)
    #
    model      = keras.models.Model(inputs=[input_1, input_2], outputs=[output], name='model')
    optimizer  = keras.optimizers.SGD(lr=cfg.cnn_alpha, decay=cfg.cnn_alpha*1e-3, momentum=0.9, nesterov=True)
    #
    if (cfg.useWeights) : 
        try:
            model.load_weights(cfg.CNNoutputDir+cfg.CNNoutputName)
        except:
            pass
    #
    from focal_loss import BinaryFocalLoss
    model.compile(loss=BinaryFocalLoss(gamma=1.5), optimizer=optimizer, metrics=['accuracy',tf.keras.metrics.AUC()])
    ##
    return model
#
def doTraining(trainDir_, testDir_, valDir_) :
    ## prepare first input
    with open(trainDir_+'X.pkl', 'rb') as handle: 
        trainX_1 = pickle.load(handle) 
    trainY = pd.read_pickle(trainDir_+'Y.pkl')   
    ## prepare second input
    trainX_2 = pd.read_pickle(trainDir_+'X_val.pkl')
    #
    #trueX = trainX_1[trainY == True]
    #falseX = trainX_1[trainY == False][np.random.choice(len(trainX_1[trainY == False]), size = len(trueX))]
    #trainX_1 = np.concatenate((trueX , falseX))
    #trainY = np.concatenate((np.zeros(len(trueX)) , np.ones(len(falseX))))
    #
    mean_1 = np.nanmean(trainX_1, axis=(0,1))
    std_1  = np.nanstd(trainX_1, axis=(0,1))
    mean_2 = trainX_2.mean().values
    std_2  = trainX_2.std().values
    #
    trainX_1 = np.nan_to_num(trainX_1, nan=-10)
    #
    with open(valDir_+'X.pkl', 'rb') as handle: 
        valX_1 = pickle.load(handle) 
    valY = pd.read_pickle(valDir_+'Y.pkl')
    valX_2 = pd.read_pickle(valDir_+'X_val.pkl')
    ##
    model = Build_1DCNN_Model(trainX_1[0].shape, len(trainX_2.keys()), len(cfg.label), mean_1, std_1, mean_2, std_2)
    print(model.summary())
    #
    history =model.fit([trainX_1,trainX_2],
                       trainY,
                       epochs          = cfg.cnn_epochs,
                       batch_size      = cfg.cnn_batch_size,
                       validation_data = ([valX_1,valX_2],valY), 
                       #callbacks  = cbks,
                       verbose         = 1)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    #plot_history(hist)
    #
    #model.save_weights(cfg.CNNoutputDir+cfg.CNNoutputName)
    model.save_weights(cfg.CNNoutputDir+cfg.CNNoutputName)
    model.save(cfg.CNNoutputDir+cfg.CNNmodelName)
    #
    with open(testDir_+'X.pkl', 'rb') as handle: 
        testX_1 = pickle.load(handle) 
    testY = pd.read_pickle(testDir_+'Y.pkl')
    testX_2 = pd.read_pickle(testDir_+'X_val.pkl')
    test_aux = pd.read_pickle(testDir_+'_aux.pkl')
    #
    pred = model.predict([testX_1, testX_2]).flatten()
    test_aux['Pred'] = pred
    test_aux.to_pickle(testDir_+'_aux.pkl')
    #
    loss, acc, auc = model.evaluate([testX_1,testX_2],testY)
    print("Test Set Acc: {0:.4f}\nTest Set AUC: {1:.4f}".format(acc,auc))
    loss, acc, auc = model.evaluate([testX_1[testY == True],testX_2[testY == True]],testY[testY == True])
    print("True Test Set Acc: {0:.4f}\nTest Set AUC: {1:.4f}".format(acc,auc))
    loss, acc, auc = model.evaluate([testX_1[testY == False],testX_2[testY == False]],testY[testY == False])
    print("False Test Set Acc: {0:.4f}\nTest Set AUC: {1:.4f}".format(acc,auc))
    #
    fpr, tpr, thresholds = metrics.roc_curve(testY.values.astype('int'), pred)
    plot_roc(fpr,tpr)
#
def validateTraining(trainDir_, testDir_, valDir_):
    #
    with open(testDir_+'X.pkl', 'rb') as handle: 
        testX_1 = pickle.load(handle) 
    testY    = pd.read_pickle(testDir_+'Y.pkl')
    testX_2  = pd.read_pickle(testDir_+'X_val.pkl')
    test_aux = pd.read_pickle(testDir_+'_aux.pkl')
    #
    def plotScore(df_, cut_, score_, range_, xlabel_, int_range = None, norm_=True, n_bins=20):
        plt.figure()
        for key in ['TTZ', 'DY']:
            cut    = ((df_['Sample'] == key) & (df_['bestRecoZPt'] > cut_))
            weight = df_['weight'][cut] * np.sign(df_['genWeight'][cut])
            n_, bins_, _ = plt.hist(x = score_[cut], bins=n_bins, 
                                    range=range_, 
                                    histtype= 'step', 
                                    weights= weight*(1604785/481435), density=norm_, label= key)
            #
            #Calculate integral in interesting region
            if (int_range) :
                print(sum(n_[10:]))
                print(int(int_range[0]/(range_[1]/n_bins)))
                integral = sum(n_[int(int_range[0]/(range_[1]/n_bins)):])
                print('{0} with Zpt cut > {1}: {2} score integral between {3} = {4:4.3f}'.format(key, cut_, xlabel_, int_range, integral))
            #
        print()
        #
        plt.title('Z pt > '+str(cut_))
        plt.grid(True)
        #plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(xlabel_)
        plt.legend()
        plt.show()
        plt.close
        #
    #

    Q_args = (test_aux['Pred'], (0,1), 'NN_score', (.50,1), False)
    #plotScore(df, 0,   *Q_args)                                                                     
    #plotScore(df, 100, *Q_args)
    plotScore(test_aux, 200, *Q_args)
    plotScore(test_aux, 250, *Q_args)
    plotScore(test_aux, 300, *Q_args)
        
def plot_history(hist):
    fig, [loss, acc] = plt.subplots(1, 2, figsize=(12, 6))
    loss.set_xlabel('Epoch')
    loss.set_ylabel('Loss')
    loss.grid(True)
    loss.plot(hist['epoch'], hist['loss'],
              label='Train Loss')
    loss.plot(hist['epoch'], hist['val_loss'],
              label = 'Val Loss')
    #loss.set_yscale('log')                                                                                                                                                                            
    loss.legend
    
    acc.set_xlabel('Epoch')
    acc.set_ylabel('Acc')
    acc.grid(True)
    acc.plot(hist['epoch'], hist['acc'],
                     label='Train Acc')
    acc.plot(hist['epoch'], hist['val_acc'],
                     label = 'Val Acc')
    #acc.set_yscale('log')                                                                                                                                                                     
    acc.legend()
    plt.show()
    plt.close()
    plt.clf()
#
def plot_roc(fpr_,tpr_):
    plt.plot([0,1],[0,1], 'k--')
    plt.plot(fpr_,tpr_, 
             label='ROC curve (area = {:.4f})'.format(metrics.auc(fpr_,tpr_)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    plt.close()
    plt.clf()


if __name__ == '__main__':
    doTraining(      *cfg.cnn_data_dir)
    validateTraining(*cfg.cnn_data_dir)
    
