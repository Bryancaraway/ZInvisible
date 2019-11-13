from __future__ import division
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow import set_random_seed
set_random_seed(2)
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
    layer = layers.Dense(128, activation='relu')(layer)
    layer = layers.Dense(64, activation='relu')(layer)
    layer = layers.Dense(64, activation='relu')(layer)
    layer = layers.Dense(64, activation='relu')(layer)
    layer = layers.Dense(32, activation='relu')(layer)
    layer = layers.Dense(32, activation='relu')(layer)
    layer = layers.Dense(32, activation='relu')(layer)
    layer = layers.Dense(32, activation='relu')(layer)
    layer = layers.Dense(32, activation='relu')(layer)
    layer = layers.Dense(32, activation='relu')(layer)
    #
    if (outputs_ == 1) : output = layers.Dense(outputs_, activation='sigmoid',name='Output')(layer)
    else               : output = layers.Dense(outputs_, activation='softmax',name='Output')(layer)
    #
    model      = keras.models.Model(inputs=main_input, outputs=output, name='model')
    optimizer  = keras.optimizers.SGD(lr=cfg.alpha, decay=cfg.alpha*1e-3, momentum=0.9, nesterov=True)
    #
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    ##
    return model

#
def doTraining():
    trainX = pd.read_pickle(cfg.train_dir+'X.pkl')
    trainY = pd.read_pickle(cfg.train_dir+'Y.pkl')
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
    history =model.fit(trainX,
                       trainY,
                       epochs          = cfg.epochs,
                       batch_size      = cfg.batch_size,
                       validation_data = (valX,valY), 
                       #callbacks  = cbks,                                                                                                                                                                                               
                       verbose         = 1)
    #
    model.save_weights(cfg.NNoutputDir+cfg.NNoutputName)
    #
    testX = pd.read_pickle(cfg.test_dir+'X.pkl')
    testY = pd.read_pickle(cfg.test_dir+'Y.pkl')
    loss, acc = model.evaluate(testX,testY)
    print("Test Set Acc: {}".format(acc))

if __name__ == '__main__':
    doTraining()
