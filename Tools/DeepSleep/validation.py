import os
import tensorflow as tf
from tensorflow import keras
tf.compat.v1.set_random_seed(2)
print(tf.__version__)
#
import pandas as pd
import numpy as np
np.random.seed(0)
#
from keras.models import Sequential, load_model
#
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import math
#
import deepsleepcfg as cfg
#

def validateModel():
    sampleDictX = {} # Dictionary of pandas DFs, with sample and year key
    sampleDictY = {}
    valDict     = {}
    #
    dfX   = pd.DataFrame()
    listY = []
    predY = []
    #
    print(cfg.NNoutputDir+cfg.NNmodel1Name)
    model = tf.keras.models.load_model(str(cfg.NNoutputDir+cfg.NNmodel1Name))
    #
    fig, axs = plt.subplots(2,len(cfg.MCsamples), figsize=(16, 10))
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.95, hspace=0.4, wspace=0.35)
    #
    for i,file_ in enumerate(cfg.files):
        for j,sample_ in enumerate(cfg.MCsamples):
            sample_file = file_+'_'+sample_
            if os.path.exists(cfg.skim_dir+sample_file+'.pkl') : 
                sampleDictX[sample_file], sampleDictY[sample_file] = testSample(sample_file)
                valDict[sample_file]                               = valSample(sample_file)
                #
                dfX   = pd.concat([dfX,sampleDictX[sample_file]], ignore_index = True)
                listY.extend(sampleDictY[sample_file])
                predY.extend(model.predict(sampleDictX[sample_file]).flatten())
                #plotYieldvsZpt(model, sampleDictX[sample_file], sampleDictY[sample_file], valDict[sample_file], sample_file)
                try:
                    loss, acc, auc = model.evaluate(sampleDictX[sample_file][sampleDictY[sample_file]==True],  sampleDictY[sample_file][sampleDictY[sample_file]==True],  verbose=0)
                    print("\nTRUE  SET: {0:} Set Acc:\t\t{1:.4f}\n{0:} Set AUC:\t\t\t{2:.4f}".format(sample_file,acc,auc))
                    loss, acc, auc = model.evaluate(sampleDictX[sample_file][sampleDictY[sample_file]==False], sampleDictY[sample_file][sampleDictY[sample_file]==False], verbose=0)
                    print(  "FALSE SET: {0:} Set Acc:\t\t{1:.4f}\n{0:} Set AUC:\t\t\t{2:.4f}".format(sample_file,acc,auc))
                    #plotvsZpt(model, sampleDictX[sample_file], sampleDictY[sample_file], valDict[sample_file]['bestRecoZPt'], sample_file)
                    #
                    try:
                        TZpt = (sampleDictY[sample_file]==True)  & (valDict[sample_file]['bestRecoZPt'] > 250)
                        FZpt = (sampleDictY[sample_file]==False) & (valDict[sample_file]['bestRecoZPt'] > 250)
                        loss, acc, auc = model.evaluate(sampleDictX[sample_file][TZpt], sampleDictY[sample_file][TZpt], verbose=0)
                        print("\nZpt > 250, TRUE SET:  {0:} Set Acc:\t{1:.4f}\n{0:} Set AUC:\t\t\t{2:.4f}".format(sample_file,acc,auc))
                        loss, acc, auc = model.evaluate(sampleDictX[sample_file][FZpt], sampleDictY[sample_file][FZpt], verbose=0)
                        print(  "Zpt > 250, FALSE SET: {0:} Set Acc:\t{1:.4f}\n{0:} Set AUC:\t\t\t{2:.4f}".format(sample_file,acc,auc))
                        #
                    except:
                        pass
                except:
                    pass
                #
                loss, acc, auc = model.evaluate(sampleDictX[sample_file], sampleDictY[sample_file], verbose=0)  
                print("\nSET: {0:} Set Acc:\t\t{1:.4f}\n{0:} Set AUC:\t\t\t{2:.4f}\n".format(sample_file,acc,auc))  
                print("====\t====\t====\t====\n")
                plotModelPred(axs[i,j],model, sampleDictX[sample_file], sampleDictY[sample_file],sample_file)
            #
        #
    #
    plt.show()
    fig.savefig('pdf/SamplevsPred'+'.pdf')
    plt.close()
    plt.clf()
    #
    plotPredvsVar(dfX,np.array(listY),np.array(predY))
#
def resetIndex(df_) :
    return df_.reset_index(drop=True).copy()
    #
#
def testSample(sampleFile_):
    x_ = pd.read_pickle(cfg.skim_dir+sampleFile_+'.pkl')
    for label in cfg.label:
        y_ = x_[label].copy()
        del x_[label]
        #    
    x_ = resetIndex(x_)
    y_ = resetIndex(y_)
    
    return x_, y_
    #
#
def valSample(sampleFile_):
    x_ = pd.read_pickle(cfg.skim_dir+sampleFile_+'_val.pkl')
    x_ = resetIndex(x_)
    return x_
    #
#
def plotModelPred(ax,model_,x_,y_,sampleFile_) :
    if (len(x_[y_==True]) > 0) :
        ax.hist(model_.predict(x_[y_==True]).ravel(),  
                 histtype='step', color = 'b', label='true')
    if (len(x_[y_==False]) > 0) :
        ax.hist(model_.predict(x_[y_==False]).ravel(), 
                 histtype='step', color = 'r', label='false')
    ax.set_title(sampleFile_)
    ax.grid(True)
    ax.legend(loc='upper center')
    #
#
def plotvsZpt(model_,x_,y_,val_,sampleFile_):
    sns.regplot(x=val_[y_ == True],  y=model_.predict(x_[y_ == True]).ravel(),  x_bins=20, fit_reg=False, label='True',  scatter_kws={'s':10}, dropna=False)
    sns.regplot(x=val_[y_ == False],  y=model_.predict(x_[y_ == False]).ravel(), x_bins=20, fit_reg=False, label='False',  scatter_kws={'s':10}, dropna=False)
    plt.legend()
    plt.grid(True)
    plt.savefig('pdf/predvsZpt_'+sampleFile_+'.pdf')
    plt.show()
    plt.clf()
    #
#
def plotYieldvsZpt(model_,x_,y_,val_,sampleFile_):
    plt.figure()
    try:
        w1 = val_['genWeight'][ ((y_ == True) & (model_.predict(x_).ravel() >= .5)) ]
        plt.hist(x=val_['bestRecoZPt'],  
                 bins=2, range=(0,600), 
                 weights=(w1/abs(w1)) * 0.0029644,
                 label='Gen_True')
    #
    except:
        pass
    try:
        w2 = val_['genWeight'][ ((y_ == False) & (model_.predict(x_).ravel() >= .5)) ]
        plt.hist(x=val_['bestRecoZPt'][ ((y_ == False) & (model_.predict(x_).ravel() >= .5)) ], 
                 bins=2, range=(0,600), 
                 weights=(w2/abs(w2)) * 0.0029644,
                 label='Gen_False')
    except:
        pass
    plt.title(sampleFile_)
    plt.legend()
    plt.grid(True)
    plt.savefig('pdf/expectedYield'+sampleFile_+'.pdf')
    plt.show()
    plt.clf()
    #
#
def plotPredvsVar(x_,y_,yhat_):
    for i in range(0,math.ceil(len(x_.keys())/6)-1):
        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        fig.subplots_adjust(left=0.06, right=0.97, bottom=0.05, top=0.95, hspace=0.3, wspace=0.35)
        for j, ax in enumerate(axs.flat) :
            x = x_[x_.keys()[i*(6)+j]]
            #print(len(y),len(yhat_),len(y_))
            #print(i,j)
            sns.regplot(x=x[y_ == True],  y=yhat_[y_ == True], x_bins=20,  fit_reg=False, label='True',  scatter_kws={'s':10}, dropna=False, ax=ax)
            sns.regplot(x=x[y_ == False], y=yhat_[y_ == False], x_bins=20, fit_reg=False, label='False', scatter_kws={'s':10}, dropna=False, ax=ax)
            #ax.set_title(str(x_.keys()[i*(9)+j]))
            ax.legend()
            ax.grid(True)
            #
        #
        print('HERE')
        plt.savefig('pdf/PredvsVar_'+str(i)+'_'+str(j)+'.pdf')
        plt.show()
        plt.clf()
        del fig
            
                 
if __name__ == '__main__':
    validateModel()
