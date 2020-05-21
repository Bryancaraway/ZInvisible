########################
### Get data         ###
### for analysis     ###
########################
### written by:      ###
### Bryan Caraway    ###
########################
##
#
import uproot
import sys
import os
import math
import pickle
import operator
import re
import time
from collections import defaultdict
import concurrent.futures
import deepsleepcfg as cfg
#
import numpy as np
np.random.seed(0)
import pandas as pd
from itertools import combinations
##


class getData : 
    ''' 
    callable object designed to get relevent data from root file
    and transform into pandas dataframes 
    '''
    # Allowed class variables
    files      = cfg.files
    samples    = cfg.MCsamples
    outDir     = cfg.skim_dir
    njets      = 2
    maxAk4Jets  = 6
    treeDir    = cfg.tree_dir
    getGenData = False
    getak8var  = False
    
    

    def __init__ (self, kwargs):
        [setattr(self,k,v) for k,v in kwargs.items() if k in getData.__dict__.keys()]
        self.get_root_data()

    def get_root_data(self):
        # open root file
        for file_ in self.files: 
            if not os.path.exists(cfg.file_path+file_+'.root') : continue
            with uproot.open(cfg.file_path+file_+'.root') as f_:
                print(f'Opening File:\t{file_}')
                tree_dir = f_.get(self.treeDir)
                for sample in self.samples:
                    start = time.perf_counter()
                    t = tree_dir.get(sample)
                    DF_Container.set_current_tree_mask(t)
                    # 
                    ak4_df   = DF_Container(cfg.ak4lvec['TLVarsLC']+cfg.ak4vars,'ak4',   'AK4_Variables')
                    ak8_df   = DF_Container(cfg.ak8lvec['TLVarsLC']+cfg.ak8vars,'ak8',   'AK8_Variables')
                    val_df   = DF_Container(cfg.valvars+cfg.sysvars,            'other', 'Event_Variables')
                    gen_df   = DF_Container(cfg.genpvars,                       'other', 'GenPart_Variables')
                    #
                    RC_ak4    = DF_Container(cfg.ak4lvec['TLVars'], 'other',   'RC_AK4LVec' )
                    RC_vars   = DF_Container(cfg.valRCvars,         'other', 'RC_TopInfo' )
                    #
                    idx = pd.IndexSlice
                    finish = time.perf_counter()
                    print(f'\nTime to finish {sample}: {finish-start:.1f}\n')
                    exit()
                #
    #
class DF_Container(): 
    '''
    container to dynamically handle root variables
    and save to .pkl files for further analysis
    container type must be ak4, ak8, or other
    '''
    current_tree = None
    allowed_types = ['ak4', 'ak8', 'other']

    # pre-selection cuts need to be applied when getting data from tree to increase speed
    # object cuts: ak4jet_pt >= 30, |ak4jet_eta| <= 2.4 ... |ak8jet_eta| <= 2.0
    # event cuts:  n_ak4jets >= 4, n_ak4bottoms >= 2, n_ak8jets(pt>=200) >= 1
    ak4_mask   = None
    ak8_mask   = None
    event_mask = None
    
    def __init__(self, variables, var_type, name):
        self.variables = variables
        self.var_type  = var_type
        self.name      = name
        # handle df depending on type 
        self.df = self.extract_and_cut()

    def extract_and_cut(self):
        idx = pd.IndexSlice
        tree_to_df = self.current_tree.pandas.df
        type_indexer = defaultdict(
            None,{'ak4':  lambda v: self.apply_event_mask(tree_to_df(v)[self.ak4_mask]),
                  'ak8':  lambda v: pd.concat([self.apply_event_mask(tree_to_df(v[:-2])[self.ak8_mask]),
                                              self.apply_event_mask(tree_to_df(v[-2:]))], axis='columns'),
                  'other':lambda v: self.apply_event_mask(tree_to_df(v))})
        try:
            df = type_indexer[self.var_type](self.variables)
        except KeyError:
            raise KeyError(f"Name '{self.var_type}' is not defined, Required to be: {self.allowed_types}")
        return df
        
    def apply_event_mask(self,df):
        return df[self.event_mask[df.index.get_level_values('entry')].values]

    @classmethod
    def set_current_tree_mask(cls,tree):
        cls.current_tree = tree
        #
        jet_pt_eta    = tree.pandas.df(cfg.ak4lvec['TLVarsLC'][:2])
        fatjet_pt_eta = tree.pandas.df(cfg.ak8lvec['TLVarsLC'][:2])
        jet_pt_key, jet_eta_key = list(jet_pt_eta.columns)
        fatjet_pt_key, fatjet_eta_key = list(fatjet_pt_eta.columns)
        #
        cls.ak4_mask = ((jet_pt_eta[jet_pt_key] >= 30) & (abs(jet_pt_eta[jet_eta_key]) <= 2.4))
        cls.ak8_mask = (abs(fatjet_pt_eta[fatjet_eta_key]) <= 2.0)
        #
        jetpt_df, fatjetpt_df = jet_pt_eta[jet_pt_key], fatjet_pt_eta[fatjet_pt_key]
        del jet_pt_eta, fatjet_pt_eta
        #        
        cls.event_mask = ((jetpt_df[cls.ak4_mask].count(level='entry') >= getData.njets) & 
                          (jetpt_df[cls.ak4_mask].count(level='entry') <= getData.maxAk4Jets) &
                          (fatjetpt_df[fatjetpt_df >= 200][cls.ak8_mask].count(level='entry') >= 1))
        del jetpt_df, fatjetpt_df


##

if __name__ == '__main__':
    #
    getData_cfg = {'files': ['result_2017'], 'samples': cfg.ZHbbFitCfg[1], 'outDir': cfg.skim_ZHbb_dir,
                   'njets':cfg.ZHbbFitCut[1], 'maxJets':cfg.ZHbbFitMaxJets, 
                   'treeDir':cfg.tree_dir+'_bb', 'getGenData':True, 'getak8var':True}
    #
    getData(getData_cfg)

             
