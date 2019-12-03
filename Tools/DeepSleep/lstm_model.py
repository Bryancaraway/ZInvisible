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
def Build_LSTM_Model(outputs_,meanAK4_,meanAK8_,stdAK4_,stdAK8_) :
    #
    lstm_input1 = layers.Input(shape=(None,), name='Ak4_inputs')
    lstm_input2 = layers.Input(shape=(None,), name='Ak8_inputs')
    #
    lambda1 = layers.Lambda(lambda x: (x - K.constant(meanAK4_)) / K.constant(stdAK4_))(lstm_input1)
    lambda2 = layers.Lambda(lambda x: (x - K.constant(meanAK8_)) / K.constant(stdAK8_))(lstm_input2)
    #
    lstm1 = layers.LSTM(32, return_sequences=True)(lambda1)
    lstm2 = layers.LSTM(32, return_sequences=True)(lambda2)
    lstm1 = layers.LSTM(32)(lstm1)
    lstm2 = layers.LSTM(32)(lstm2)
    #
    concat = layers.concatenate([lstm1,lstm2],axis=-1)
    layer  = layers.Dense(32, activation='relu')(layer)
    layer  = layers.Dense(32, activation='relu')(layer)
    layer  = layers.Dense(32, activation='relu')(layer)
    #
    output = layers.Dense(outputs_, activation='sigmoid',name='Output')(layer)
    #
    model      = keras.models.Model(inputs=[lstm_input1,lstm_input1], outputs=output, name='model')
    optimizer  = keras.optimizers.SGD(lr=cfg.alpha, decay=cfg.alpha*1e-3, momentum=0.9, nesterov=True)
    #
    if (cfg.useWeights) : model.load_weights(cfg.NNoutputDir+cfg.NNoutputName)
    #try:
    #    model = tf.keras.utils.multi_gpu_model(model, gpus=4)
    #except:
    #    pass
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy',tf.keras.metrics.AUC()])
    ##
    return model
#
