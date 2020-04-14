#     OA                 #  
##                    ##
########################                               
### TTZ/H, Z/H to bb ###
### build datacard   ###                               
### for HCT          ###                               
########################                               
### written by:      ###                               
### Bryan Caraway    ###                               
########################                               
##                    ##                                 
#                      #

import sys
import os
import pickle
import math
import operator as OP
#
import deepsleepcfg as cfg
#import processData  as prD 
import kinematicFit as kFit
#
import uproot
import uproot_methods
from ROOT import TFile, TDirectory, TH1F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fun_library as lib
from fun_library import fill1e, fillne, deltaR, deltaPhi, invM, calc_mtb, getZHbbBaseCuts
np.random.seed(1)
##

def sepSigBKgdf(df_,sig_):
    sig_df = pd.DataFrame()
    bkg_df = pd.DataFrame()
    for key_ in df_.keys():
        def get_weight(cut):
            if len(cut) > 0:
                weight = (df_[key_]['val']['weight']*np.sign(df_[key_]['val']['genWeight']) * (137/41.9))[cut]
            else:
                weight = (df_[key_]['val']['weight']*np.sign(df_[key_]['val']['genWeight']) * (137/41.9))
            return weight
        def add_toDF(df, cut=[], name=key_.split('_201')[0], isSig=False):
            temp_df = pd.DataFrame({ 
                'HZ_pt':   df_[key_]['ak8']['H_pt'],
                'M_sd':    df_[key_]['ak8']['H_M'], 
                'NN':      df_[key_]['val']['NN'],
                'Weight':get_weight(cut), 'Name':name}) 
            if isSig:
                temp_df['HZ_genpt'] = df_[key_]['val']['genZHpt']
            if len(cut) > 0:
                df = df.append(pd.DataFrame(temp_df[cut]), ignore_index=True)
            else:
                df = df.append(pd.DataFrame(temp_df), ignore_index=True)
            return(df)
        #
        if (sig_ in key_):
            ttZbb   = (df_[key_]['val']['Zbb'] == True)
            ttHbb   = (df_[key_]['val']['Hbb'] == True)
            ttZqq   = (df_[key_]['val']['Zqq'] == True)
            #
            sig_df = add_toDF(sig_df,ttZbb,'ttZbb', isSig=True)
            sig_df = add_toDF(sig_df,ttHbb,'ttHbb', isSig=True)
            bkg_df = add_toDF(bkg_df,ttZqq,'ttZqq')
        else:
            bkg_df = add_toDF(bkg_df)
    #
    return sig_df, bkg_df
#


def makeDataCard(files_, samples_, outDir_):
    df = kFit.retrieveData(files_, samples_, outDir_, getak8_ = True)
    sig = 'TTZH'
    suf = '_2017'
    # Organize Sig and Bkg DataFrame
    sig_df, bkg_df = sepSigBKgdf(df,sig)
    pdata_df = pd.concat([sig_df,bkg_df], ignore_index=True, sort=False)
    del df
    #
    # Seperate NN histogram by H/Z pt bin
    nn_bins = np.quantile(sig_df['NN'][sig_df['NN'] > -1],np.arange(0.,1.1,.1))
    pt_bins = [200,300] # [200,300,400]
    pt_bin_dict = {'200':'lowpt', '300':'highpt'}
    def binSigBkg(df_):
        hist_vals  = []
        for i,pt_bin in enumerate(pt_bins):
            if i == len(pt_bins)-1:
                pt_cut = df_['HZ_pt'] >= pt_bin
            else:
                pt_cut = (df_['HZ_pt'] >= pt_bins[i]) & (df_['HZ_pt'] < pt_bins[i+1])
            #
            temp_vals,_ = np.histogram(df_['NN'][pt_cut], bins=50, range=(0,1), weights=df_['Weight'][pt_cut])
            #
            hist_vals.append(temp_vals)
        return np.array(hist_vals)
    #
    #sig_vals = binSigBkg(sig_df)
    #bkg_vals  = binSigBkg(bkg_df)
    #data_vals = sig_vals+bkg_vals
    #
    # Create a root file to store the shapes for shape fit
    dc_dir = 'Higgs-Combine-Tool'
    root_file = TFile(dc_dir+'/input.root', 'recreate')
    for i_,pt_bin in enumerate(pt_bins):
        dif_prefix = 'HZpt_'
        if i_ == len(pt_bins)-1:
            pt_cut_str = dif_prefix+'ge'+str(pt_bin)
            pt_cut = ((OP.ge,pt_bin),(OP.lt,np.inf))
        else:
            pt_cut_str = dif_prefix+str(pt_bins[i_])+'to'+str(pt_bins[i_+1])
            pt_cut = ((OP.ge,pt_bins[i_]),(OP.lt,pt_bins[i_+1]))
        ## need to add histograms to root file with format $channel/$process_, $channel/$process_$process_systematic
        ## as well as a .txt file in datacard format
        dir_ = root_file.mkdir(pt_cut_str)
        dir_.cd()
        #
        dc_txt = open(dc_dir+'/'+pt_cut_str+'.txt', 'w')
        dc_txt.writelines(['max\t1\n',
                           'jmax\t'+str(len(set(sig_df['Name']))+len(set(bkg_df['Name']))-1)+'\n', 
                           'kmax\t*\n',100*'-'+'\n'])
        dc_txt.writelines(['shapes * * '+str(root_file.GetName())+' $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC\n', 
                           100*'-'+'\n',
                           'bin\t\t\t\t\t'+pt_cut_str+'\n',
                           'observation\t\t\t\t'+str(-1)+'\n',
                           100*'-'+'\n'])
        bin_line      = ['bin\t\t\t\t\t']
        process_line1 = ['process\t\t\t\t\t']
        process_line2 = ['process\t\t\t\t\t']
        rate_line     = ['rate\t\t\t\t\t']
        #
        def fillHistDC(df_,issig=False,ispData=False):
            bins_min_max = (10,nn_bins)
            if ispData:
                h_data = TH1F('data_obs','data_obs',*bins_min_max)
                pt_cut_apl = (pt_cut[0][0](df_['HZ_pt'],pt_cut[0][1])) & (pt_cut[1][0](df_['HZ_pt'],pt_cut[1][1]))
                [h_data.Fill(entry,w) for entry, w in zip(df_['NN'][pt_cut_apl], df_['Weight'][pt_cut_apl])]
                root_file.Write()
                del h_data
                return
            #        
            h_arr = []
            for j_,sample_ in enumerate(set(df_['Name'])):
                sample_df = df_[df_['Name'] == sample_]
                pt_cut_apl = (pt_cut[0][0](sample_df['HZ_pt'],pt_cut[0][1])) & (pt_cut[1][0](sample_df['HZ_pt'],pt_cut[1][1]))
                #
                if issig:
                    sample_ = sample_+'_'+pt_bin_dict[str(pt_bin)]
                    pt_cut_apl = (pt_cut[0][0](sample_df['HZ_genpt'],pt_cut[0][1])) & (pt_cut[1][0](sample_df['HZ_genpt'],pt_cut[1][1]))
                h_tuple = (sample_,sample_,*bins_min_max)
                h_temp  = TH1F(*h_tuple)
                [h_temp.Fill(entry, w) for entry, w in zip(sample_df['NN'][pt_cut_apl], sample_df['Weight'][pt_cut_apl])]
                #
                nonlocal bin_line
                nonlocal process_line1
                nonlocal process_line2
                nonlocal rate_line
                bin_line          = bin_line +  [pt_cut_str+'\t']
                process_line1     = process_line1 + ['{:14}\t'.format(sample_)]
                if issig:
                    process_line2 = process_line2 + [str(j_+1-len(set(df_['Name'])))+'\t\t']
                else:
                    process_line2 = process_line2 + [str(j_+1)+'\t\t']
        
                rate_line         = rate_line + ['{0}\t\t'.format(-1)] # changed from h_temp.Integral()
                #
                h_arr.append(h_temp)
                del h_temp
                #
            #
            root_file.Write()
            del h_arr
        #
        fillHistDC(sig_df,issig=True)
        fillHistDC(bkg_df)
        fillHistDC(pdata_df,ispData=True)
        #
        dc_txt.writelines([''.join(bin_line)+'\n',''.join(process_line1)+'\n',''.join(process_line2)+'\n',''.join(rate_line)+'\n',100*'-'+'\n'])
        dc_txt.close()
        #
    #

    root_file.Close()
    
if __name__ == '__main__':
    files_samples_outDir = cfg.ZHbbFitCfg
    #
    #
    makeDataCard(*files_samples_outDir)
