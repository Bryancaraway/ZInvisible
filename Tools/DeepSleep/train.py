from __future__ import division
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
tf.compat.v1.set_random_seed(2)
print(tf.__version__)
#
import pandas as pd
import numpy as np
np.random.seed(0)
#
from tensorflow.python.keras import backend as K
from keras import backend as k
from tensorflow.python.ops import math_ops
from keras.models import Sequential
from keras.layers import Dense
#
import matplotlib.pyplot as plt
from sklearn import metrics
#
import deepsleepcfg as cfg

def AllocateVRam():
    tf_cfg = tf.ConfigProto()
    tf_cfg.gpu_options.allow_growth = True
    tf_cfg.gpu_options.per_process_gpu_memory_fraction = 0.5
    k.tensorflow_backend.set_session(tf.Session(config=tf_cfg))
#
def Build_Model(inputs_,outputs_,mean_,std_):
    #
    main_input = layers.Input(shape=[inputs_], name='input')
    layer = keras.layers.Lambda(lambda x: (x - K.constant(mean_)) / K.constant(std_), name='normalizeData')(main_input)
    #
    layer = keras.layers.Dropout(0.0)(layer)
    layer = layers.Dense(128, activation='relu')(layer)
    layer = layers.BatchNormalization()(layer)
    layer = keras.layers.Dropout(0.0)(layer)
    layer = layers.Dense(128, activation='relu')(layer)
    layer = layers.BatchNormalization()(layer)
    layer = keras.layers.Dropout(0.0)(layer)
    layer = layers.Dense(64, activation='relu')(layer)
    layer = layers.BatchNormalization()(layer)
    layer = keras.layers.Dropout(0.0)(layer)
    layer = layers.Dense(64, activation='relu')(layer)
    layer = layers.BatchNormalization()(layer)
    layer = keras.layers.Dropout(0.0)(layer)
    layer = layers.Dense(64, activation='relu')(layer)
    layer = layers.BatchNormalization()(layer)
    layer = keras.layers.Dropout(0.0)(layer)
    layer = layers.Dense(32, activation='relu')(layer)
    layer = layers.BatchNormalization()(layer)
    layer = keras.layers.Dropout(0.0)(layer)
    #layer = layers.Dense(32, activation='relu')(layer)
    #layer = layers.BatchNormalization()(layer)
    #layer = keras.layers.Dropout(0.5)(layer)
    #layer = layers.Dense(128, activxation='relu')(layer)
    #layer = layers.Dense(128, activation='relu')(layer)
    #layer = layers.Dense(128, activation='relu')(layer)
    #layer = layers.Dense(128, activation='relu')(layer)
    #layer = layers.Dense(128, activation='relu')(layer)
    #
    if (outputs_ == 1) : output = layers.Dense(outputs_, activation='sigmoid',name='Output')(layer)
    else               : output = layers.Dense(outputs_, activation='softmax',name='Output')(layer)
    #
    model      = keras.models.Model(inputs=main_input, outputs=output, name='model')
    optimizer  = keras.optimizers.SGD(lr=cfg.alpha, decay=cfg.alpha*1e-3, momentum=0.9, nesterov=True)
    #
    if (cfg.useWeights and os.path.exists(cfg.NNoutputDir+cfg.NNoutputName)): 
        model.load_weights(cfg.NNoutputDir+cfg.NNoutputName)
    #try:
    #    model = tf.keras.utils.multi_gpu_model(model, gpus=4)
    #except:
    #    pass
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy',tf.keras.metrics.AUC()])
    ##
    return model

#
def doTraining():
    trainX = pd.read_pickle(cfg.train_over_dir+'X.pkl')
    trainY = pd.read_pickle(cfg.train_over_dir+'Y.pkl')
    print(trainX)
    print(trainY)
    valX = pd.read_pickle(cfg.val_dir+'X.pkl')
    valY = pd.read_pickle(cfg.val_dir+'Y.pkl')
    #
    model = Build_Model(len(trainX.keys()),len(cfg.label),trainX.mean().values,trainX.std().values)
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
    if (cfg.plotHist) :
        history =model.fit(trainX,
                           trainY,
                           epochs          = cfg.epochs,
                           batch_size      = cfg.batch_size,
                           validation_data = (valX,valY), 
                           #callbacks  = cbks,
                           verbose         = 1)
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        plot_history(hist)
        #
        model.save_weights(cfg.NNoutputDir+cfg.NNoutputName)
        model.save(cfg.NNoutputDir+cfg.NNmodel1Name)
    #
    testX = pd.read_pickle(cfg.test_dir+'X.pkl')
    testY = pd.read_pickle(cfg.test_dir+'Y.pkl')
    loss, acc, auc = model.evaluate(testX,testY)
    print("Test Set Acc: {0:.4f}\nTest Set AUC: {1:.4f}".format(acc,auc))
    

    #
    pred = model.predict(testX).flatten()
    
    #
    fpr, tpr, thresholds = metrics.roc_curve(testY.values.astype('int'), pred)
    plot_roc(fpr,tpr)
#

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
    doTraining()
