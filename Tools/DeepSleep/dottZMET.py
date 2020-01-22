#                      #  
##                    ##
########################                               
### Process TTZ,     ###
### Z to MET, data   ###                               
# for score computation#                               
########################                               
### written by:      ###                               
### Bryan Caraway    ###                               
########################                               
##                    ##                                 
#                      #

##
import sys
import os
#
import deepsleepcfg as cfg
import processData  as prD 
import kinematicFit as kFit
#
import numpy as np
np.random.seed(0)
##

if __name__ == '__main__':   

    files_samples_outDir = cfg.ZinvFitCfg
    #
    #prD.getData(         *files_samples_outDir, *cfg.ZinvFitCut, cfg.ZinvFitMaxJets, treeDir_ = cfg.tree_dir+'_Inv')
    #prD.interpData(      *files_samples_outDir, cfg.ZinvFitMaxJets)  
    #
    #kFit.evaluateScore(  *files_samples_outDir, cfg.ZinvFitoverlap)
    kFit.AnalyzeScore(   *files_samples_outDir, cfg.ZinvFitoverlap) 
