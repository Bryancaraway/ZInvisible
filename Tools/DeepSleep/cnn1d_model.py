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
import pickle
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
def Build_1DCNN_Model(inShape, outputs_):#meanAK4_,stdAK4_) : 
    # 
    input_1 = layers.Input(shape=inShape, name='cnn_input') # (11,7,1)
    #mask    = layers.Masking(mask_value=-1)(input_1)
    #
    #norm = layers.BatchNormalization()(input_1)
    CNN = layers.Conv1D(64, 3, strides=1, activation= 'relu')(input_1)
    #CNN = layers.MaxPooling1D(pool_size=2)(CNN)
    CNN = layers.Conv1D(64, 2, strides=1, activation= 'relu')(CNN)
    CNN = layers.Conv1D(64, 1, strides=1, activation= 'relu')(CNN)
    CNN = layers.MaxPooling1D(pool_size=2)(CNN)
    CNN = layers.Flatten()(CNN)
    #
    layer = layers.Dense(32, activation='relu')(CNN)
    layer = layers.Dense(32, activation='relu')(layer)
    layer = layers.Dense(32, activation='relu')(layer)
    #
    output = layers.Dense(outputs_, activation='sigmoid',name='Output')(layer)
    #
    model      = keras.models.Model(inputs=input_1, outputs=output, name='model')
    optimizer  = keras.optimizers.SGD(lr=cfg.cnn_alpha, decay=cfg.cnn_alpha*1e-3, momentum=0.9, nesterov=True)
    #
    #if (cfg.useWeights) : model.load_weights(cfg.NNoutputDir+cfg.NNoutputName)
    #try:
    #    model = tf.keras.utils.multi_gpu_model(model, gpus=4)
    #except:
    #    pass
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy',tf.keras.metrics.AUC()])
    ##
    return model
#
def doTraining(trainDir_, testDir_, valDir_) :
    with open(trainDir_+'X.pkl', 'rb') as handle: 
        trainX = pickle.load(handle) 
    trainY = pd.read_pickle(trainDir_+'Y.pkl')
    #
    trueX = trainX[trainY == True]
    falseX = trainX[np.random.choice(len(trainX[trainY == False]), size = len(trueX))]
    trainX = np.concatenate((trueX , falseX))
    trainY = np.concatenate((np.zeros(len(trueX)) , np.ones(len(falseX))))
    #
    trainX = np.nan_to_num(trainX, nan=-1)
    #
    with open(valDir_+'X.pkl', 'rb') as handle: 
        valX = pickle.load(handle) 
    valY = pd.read_pickle(valDir_+'Y.pkl')
    #
    model = Build_1DCNN_Model(trainX[0].shape,len(cfg.label))#trainX.mean().values,trainX.std().values)
    print(model.summary())
    #
    history =model.fit(trainX,
                           trainY,
                           epochs          = cfg.cnn_epochs,
                           batch_size      = cfg.cnn_batch_size,
                           validation_data = (valX,valY), 
                           #callbacks  = cbks,
                           verbose         = 1)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    #
    with open(testDir_+'X.pkl', 'rb') as handle: 
        testX = pickle.load(handle) 
    testY = pd.read_pickle(testDir_+'Y.pkl')
    #
    loss, acc, auc = model.evaluate(testX,testY)
    print("Test Set Acc: {0:.4f}\nTest Set AUC: {1:.4f}".format(acc,auc))
    #
    #pred = model.predict(testX).flatten()


if __name__ == '__main__':
    doTraining(*cfg.cnn_data_dir)
