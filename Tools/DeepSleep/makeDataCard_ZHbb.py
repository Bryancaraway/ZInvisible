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
from fun_library import fill1e, fillne, deltaR, deltaPhi, invM, calc_mtb, getZHbbBaseCuts, weighted_quantile
np.random.seed(1)
##
pt_bins = [200,300,400] # [200,300,400]
#pt_bin_dict = {'200':'lopt', '300':'hipt'}


def sepSigBKgdf(df_,sig_):
    sig_df = pd.DataFrame()
    bkg_df = pd.DataFrame()
    for key_ in df_.keys():
        def get_weight():
                weight = (df_[key_]['val']['weight']*np.sign(df_[key_]['val']['genWeight']) * (137/41.9))
            return weight
        def add_toDF(df, cut=None, name=key_.split('_201')[0], isSig=False):
            if cut == None:
                cut = []
            temp_df = pd.DataFrame({ 
                'HZ_pt':   df_[key_]['ak8']['H_pt'],
                'M_sd':    df_[key_]['ak8']['H_M'], 
                'NN':      df_[key_]['val']['NN'],
                'Weight':  get_weight(),
                'Name':name}) 
            # add systematics to df
            for syst in cfg.sysvars:
                temp_df[sys] = df_[key_]['val'][syst]
            if isSig:
                temp_df['HZ_genpt'] = df_[key_]['val']['genZHpt']
            if len(cut) > 0:
                df = df.append(pd.DataFrame(temp_df[cut]), ignore_index=True)
            else:
                df = df.append(pd.DataFrame(temp_df), ignore_index=True)
            return(df)
        #
        if (sig_ in key_):
            bin2         = ((df_[key_]['val']['genZHpt'] >= 200) & (df_[key_]['val']['genZHpt'] < 300))
            bin3         = ((df_[key_]['val']['genZHpt'] >= 300) & (df_[key_]['val']['genZHpt'] < 400))
            bin4         = ((df_[key_]['val']['genZHpt'] >= 400) & (df_[key_]['val']['genZHpt'] < np.inf))
            bin1         = ((~bin2) & (~bin3) & (~bin4))
            gen_bin_dict = { 'bin1':bin1,'bin2':bin2,'bin2':bin3,'bin3':bin4, }
            #
            ttZbb   = (df_[key_]['val']['Zbb'] == True)
            ttHbb   = (df_[key_]['val']['Hbb'] == True)
            ttZqq   = (df_[key_]['val']['Zqq'] == True)
            #
            for bin, bin_cut in gen_bin_dict.items():
                for sig, sig_cut in {'ttZ':ttZbb, 'ttH':ttHbb}.items():
                    sig_df = add_toDF(sig_df,cut=sig_cut & bin_cut, name=sig+bin, isSig=True)
            #sig_df = add_toDF(sig_df,cut=ttZbb & lo,       name='ttZgenlopt', isSig=True)
            #sig_df = add_toDF(sig_df,cut=ttZbb & hi,       name='ttZgenhipt', isSig=True)
            #sig_df = add_toDF(sig_df,cut=ttHbb & lo,       name='ttHgenlopt', isSig=True)
            #sig_df = add_toDF(sig_df,cut=ttHbb & hi,       name='ttHgenhipt', isSig=True)
            #sig_df = add_toDF(sig_df,cut=ttZbb & ~lo & ~hi,name='ttZelse',    isSig=True)
            #sig_df = add_toDF(sig_df,cut=ttHbb & ~lo & ~hi,name='ttHelse',    isSig=True)
            #
            bkg_df = add_toDF(bkg_df,cut=ttZqq,name='ttZqq')
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
    # Create a root file to store the shapes for shape fit
    dc_dir = 'Higgs-Combine-Tool'
    root_file = TFile(dc_dir+'/input.root', 'recreate')
    def setupDC(dc_name, channel):
        _txt = open(dc_name, 'w')
        _txt.writelines(['max\t1\n',
                         'jmax\t'+str(len(set(sig_df['Name']))+len(set(bkg_df['Name']))-1)+'\n', 
                         'kmax\t*\n',100*'-'+'\n'])
        _txt.writelines(['shapes * * '+str(root_file.GetName())+' $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC\n', 
                         100*'-'+'\n',
                         '{0:21}{1:12}\n'.format('bin', channel),#'bin\t\t\t\t\t'+channel+'\n',
                         '{0:21}{1:12}\n'.format('observation', str(-1)),#'observation\t\t\t\t'+str(-1)+'\n',
                         100*'-'+'\n'])
        b_line      = ['{0:21}'.format('bin')]    #['bin\t\t\t\t\t']
        p_line1     = ['{0:21}'.format('process')]#['process\t\t\t\t\t']
        p_line2     = ['{0:21}'.format('process')]#['process\t\t\t\t\t']
        r_line      = ['{0:21}'.format('rate')]   #['rate\t\t\t\t\t']
        return _txt, b_line, p_line1, p_line2, r_line
    #
    def main_body_DC(_txt,b_line,p_line1,p_line2,r_line):
        _txt.writelines([''.join(b_line)+'\n',''.join(p_line1)+'\n',''.join(p_line2)+'\n',''.join(r_line)+'\n',100*'-'+'\n'])
    #
    for i_,pt_bin in enumerate(pt_bins):
        dif_prefix = 'HZpt_'
        if i_ == len(pt_bins)-1:
            pt_cut_str = dif_prefix+'ge'+str(pt_bin)
            pt_cut = ((OP.ge,pt_bin),(OP.lt,np.inf))
        else:
            pt_cut_str = dif_prefix+str(pt_bins[i_])+'to'+str(pt_bins[i_+1])
            pt_cut = ((OP.ge,pt_bins[i_]),(OP.lt,pt_bins[i_+1]))
        # Seperate NN histogram by H/Z pt bin
        quant_cut = ((sig_df['NN'] > -1) &  # only consider quantile contruction for events that qualify for NN discrimination
                     (pt_cut[0][0](sig_df['HZ_genpt'],pt_cut[0][1])) & 
                     (pt_cut[1][0](sig_df['HZ_genpt'],pt_cut[1][1])))   
        quantiles = np.arange(0.,1.1,.1) # break up bins to contain 10% of total MC (might need to change later to account for weights)
        nn_bins   = weighted_quantile(sig_df['NN']  [quant_cut],quantiles,sig_df['Weight'][quant_cut])
        #mass_bins = weighted_quantile(sig_df['M_sd'][quant_cut],quantiles,sig_df['Weight'][quant_cut]) 
        mass_bins = [50,80,105,145,200]
        bins_dict = {'NN': nn_bins, 'M_sd': mass_bins}
        ## need to add histograms to root file with format $channel/$process_, $channel/$process_$process_systematic
        ## as well as a .txt file in datacard format
        channel_name = pt_cut_str
        print(10*'='+pt_cut_str+(10*'='))
        dir_ = root_file.mkdir(channel_name)
        dir_.cd()
        #
        dc_txt, bin_line, process_line1, process_line2, rate_line = setupDC(dc_name=dc_dir+'/'+channel_name+'.txt', channel=channel_name)
        #
        def fillHistDC(df_, issig=False,ispData=False):
            bins_min_max = ((len(bins_dict['NN'])-1)*(len(bins_dict['M_sd'])-1),0,(len(bins_dict['NN'])-1)*(len(bins_dict['M_sd'])-1))
            def getflat2dHist(df_):
                pt_cut_apl = (pt_cut[0][0](df_['HZ_pt'],pt_cut[0][1])) & (pt_cut[1][0](df_['HZ_pt'],pt_cut[1][1]))
                
                #
                #if issig == True:
                #    hist_cut = pt_cut_apl & (df_['NN'] >= nn_bins[-3] ) 
                #    plt.hist(x=df_['M_sd'][hist_cut], bins=bins_dict['M_sd'], weights=df_['Weight'][hist_cut])
                #    plt.xlabel('M_sd (NN top 20% quantile)')
                #    plt.title(channel_name+'_'+df_['Name'].iloc[0]+'_NN>='+'{0:2.3f}'.format(nn_bins[-3]))
                #    plt.show()
                #
                h_bins, _x, _y = np.histogram2d(x=df_['NN'][pt_cut_apl], y=df_['M_sd'][pt_cut_apl], # fill 2d histogram of mass (y) vs NN score (x) 
                                                bins=[bins_dict['NN'],bins_dict['M_sd']],
                                                weights=df_['Weight'][pt_cut_apl]) 
                #print(np.sum(h_bins.flatten()),'\n')
                return np.where(h_bins.flatten() < 0, 0.0, h_bins.flatten())
            #
            if ispData:
                h_data = TH1F('data_obs','data_obs',*bins_min_max)
                #[h_data.Fill(entry,w) for entry, w in zip(df_[kinem][pt_cut_apl], df_['Weight'][pt_cut_apl])]
                h_bins = getflat2dHist(df_)
                [h_data.SetBinContent(i+1,entry) for i,entry in enumerate(h_bins)]
                h_data.Write()
                del h_data
                return
            #        
            for j_,sample_ in enumerate(set(df_['Name'])):
                sample_df = df_[df_['Name'] == sample_]
                #
                #if issig:
                #    sample_ = sample_+'_'+pt_bin_dict[str(pt_bin)]
                h_tuple = (sample_,sample_,*bins_min_max)
                h_temp  = TH1F(*h_tuple)
                #[h_temp.Fill(entry, w) for entry, w in zip(sample_df[kinem][pt_cut_apl], sample_df['Weight'][pt_cut_apl])]
                #print(sample_)
                h_bins = getflat2dHist(sample_df)
                [h_temp.SetBinContent(i+1,entry) for i,entry in enumerate(h_bins)]
                #
                nonlocal channel_name
                nonlocal bin_line
                nonlocal process_line1
                nonlocal process_line2
                nonlocal rate_line
                bin_line          = bin_line +      ['{0:14}'.format(channel_name)]
                process_line1     = process_line1 + ['{0:14}'.format(sample_)]
                if issig:
                    process_line2 = process_line2 + ['{0:14}'.format(str(j_+1-len(set(df_['Name']))))]#[str(j_+1-len(set(df_['Name'])))+'\t\t']
                else:
                    process_line2 = process_line2 + ['{0:14}'.format(str(j_+1))]#[str(j_+1)+'\t\t']
        
                rate_line         = rate_line +     ['{0:14}'.format(str(-1))] # changed from h_temp.Integral()
                #
                h_temp.Write()
                del h_temp
                #
            #
        #
        fillHistDC(sig_df,  issig=True)
        fillHistDC(bkg_df)
        fillHistDC(pdata_df,ispData=True)
        #
        main_body_DC(dc_txt, bin_line, process_line1, process_line2, rate_line)
        #
        # Define systematics here
        # and setup systmatics class attributes
        Systematic.set_processline(process_line1)
        Systematic.set_current_dc(dc_txt)
        ShapeSystematic.set_ptcut_bindict(pt_cut,bin_dict)
        ShapeSystematic.set_dataframe(pdata_df)
        #
        tt_sys = Systematic('dummy_tt_sys','lnN', channel_name, ['TTBarLep','TTBarHad'], 1.05, 'Testing logNorm systematic')
        #
        sig_sys = ShapeSystematic('dummy_ttH_sys','shape', channel_name, ['ttHelse','ttHgenlopt','ttHgenhipt'], 1, 1.008, .993, 'Testing shape systematic')
        sig_sys.scaleHists(root_file)
        del sig_sys
        dc_txt.close()
    #
    root_file.Close()

class Systematic: # Class to handle Datacard systematics 
    ''' 
    Syntax: Systematic(Systematic Name,Systematic Type, Channel, Affected Processes, Value, Additional Information)
    --- Possible Processes ---
    (*OLD*) Signal: ttZelse, ttHelse, ttZgenlopt, ttHgenlopt, ttZgenhipt, ttHgenhipt
    (*NEW*) Signal: ttZbin1, ttZbin2, ttZbin3, ttZbin4, ttHbin1, ttHbin2, ttHbin3, ttHbin4,
    Bkg:    TTBarLep, ttZqq, TTBarHad, QCD, WJets, TTX, DiBoson, ZJets, TriBoson
    '''
    dc_root_dir = 'Higgs-Combine-Tool/'
    datacard     = None
    process_line = None

    def __init__(self, name, stype, channel, process_ids, value, info=None):
        self.name     = name
        self.stype    = stype
        self.ids      = process_ids
        self.channel  = channel
        self.value    = value
        self.info     = '' if info == None else info
        #
        if self.datacard != None: 
            self.datacard.write(self.get_DC_line()) # write to datacard file upon instance creation

    @property
    def line(self):
        return '{0:14} {1:6}'.format(self.name,self.stype)

    @classmethod
    def set_current_dc(cls,datacard):
        cls.datacard = datacard

    @classmethod
    def set_processline(cls,process_line):
        cls.process_line = process_line
        
    def get_DC_line(self):
        _line = self.line
        for p in self.process_line[1:]:
            # reformat process to exclude \t 
            _process = p.replace('\t', '').replace(' ','' )
            if _process in self.ids: 
                _line +='{0:14}'.format(str(self.value))
            else :
                _line +='{0:14}'.format('-')
        _line += '\t'+self.info+'\n'
        return _line


class ShapeSystematic(Systematic): # Class to handle Datacard shape systematics
    
    '''
    Create additional histograms with systematic variations and writes them to root file
    Supports MCStats
    Inheirits from the Systematics Class
    Need to set cut_op and bin_dict to make effective use of this class
    '''

    bin_dict = None
    cut_op   = None
    df       = None

    def __init__(self, name, stype, channel, process_ids, value, up=None, down=None, info=None):
        super().__init__(name, stype, channel, process_ids, value, info)
        self.up     = up
        self.down   = down
        #
        if   'mcstat' in name:
            self.makeMCStatHist()
        elif    type(up).__name__   == 'int' or type(up).__name__ == 'float' or \
                type(down).__name__ == 'int' or type(down).__name__ == 'float':
            self.scaleHists()
        elif    type(up).__name__   == 'str' or type(down).__name__ == 'str':
            self.makeUpDownHist()


    @property
    def bins_min_max(self):
        if self.bin_dict == None:
            return None
        _n = (len(self.bins_dict['NN'])-1)*(len(self.bins_dict['M_sd'])-1)
        return (_n,0,_n)

    @classmethod
    def set_ptcut_bindict(cls,_cut,_dict):
        cls.bin_dict=_dict
        cls.cut_op=_cut
        
    @classmethod
    def set_dataframe(cls,df_):
        cls.df = df_

    def FillandWrite(self, bin_content, add_str=None):
        add_str = '' if add_str == None else add_str
        h_tuple = (process+'_'+self.name+add_str, process+'_'+self.name+add_str, *self.bins_min_max)
        h_temp = TH1F(*h_tuple)
        [h_temp.SetBinCOntent(i+1,entry) for i,entry in enumerate(bin_content)]
        h_temp.Write()
        del h_temp

    def pt_cut(self,df_):
        if cut_op == None:
            return None
        _min = self.cut_op[0][0](df_['HZ_pt'],self.cut_op[0][1])
        _max = self.cut_op[1][0](df_['HZ_pt'],self.cut_op[1][1])
        return ((_min) & (_max))
        

    def makeScaleHist(self, rfile): # return renamed, scaled hists
        for process in self.ids:
            df_ = self.df[self.df['Name'] == process].copy()
            pt_cut = self.pt_cut(df_)
            h_bins, *_ =  np.histogram2d(x=df_['NN'][pt_cut], y=df_['M_sd'][pt_cut],
                                             bins=[self.bins_dict['NN'],self.bins_dict['M_sd']],
                                             weights=df_['Weight'][pt_cut])
            for ud_str, ud_bins in zip(['Up','Down'],[h_bins.flatten()*self.up,h_bins.flatten()*self.down]):
                temp_bins = np.where(ud_bins < 0, 0.0, ud_bins)
                self.FillandWrite(temp_bins, ud_str)
            #hist_ = rfile.Get(self.channel+'/'+process)
            #hist_up   = hist_.Clone()#.Scale(self.up)
            #hist_down = hist_.Clone()#.Scale(self.down)
            #del hist_
            #up   = 1 if self.up   == None else self.up
            #down = 1 if self.down == None else self.down
            #hist_up.Scale(  up)
            #hist_down.Scale(down)
            #hist_up.Write(process+'_'+self.name+'Up')
            #hist_down.Write(process+'_'+self.name+'Down')
            #del hist_up, hist_down
            
    def makeUpDownHist(self):
        for process in self.ids:
            df_ = self.df[self.df['Name'] == process].copy() 
            pt_cut = self.pt_cut(df_)
            for ud_str, ud in zip(['Up','Down'],[self.up,self.down]):
                ud_bins, *_ =  np.histogram2d(x=df_['NN'][pt_cut], y=df_['M_sd'][pt_cut],
                                             bins=[self.bins_dict['NN'],self.bins_dict['M_sd']],
                                             weights=df_['Weight'][pt_cut]*df_[ud][pt_cut])
                ud_bins = np.where(ud_bins.flatten() < 0, 0.0, ud_bins.flatten())                           
                self.FillandWrite(ud_bins, ud_str)

    def makeMCStatHist(self):
        for process in self.ids:
            df_ = self.df[self.df['Name'] == process].copy()
            pt_cut = self.pt_cut(df_)
            mcstat_bins, *_ =  np.histogram2d(x=df_['NN'][pt_cut], y=df_['M_sd'][pt_cut],
                                              bins=[self.bins_dict['NN'],self.bins_dict['M_sd']])
            mcstat_bins = np.where(mcstat_bins.flatten() < 0, 0.0, np.sqrt(mcstat_bins.flatten()))
            #
            h_bins, *_ =  np.histogram2d(x=df_['NN'][pt_cut], y=df_['M_sd'][pt_cut],
                                             bins=[self.bins_dict['NN'],self.bins_dict['M_sd']],
                                             weights=df_['Weight'][pt_cut])
            for ud_str, ud_bins in zip(['Up','Down'],[np.add(h_bins.flatten(),mcstat_bins),np.add(h_bins.flatten(),-1*mcstat_bins)]):
                temp_bins = np.where(ud_bins < 0, 0.0, ud_bins)
                self.FillandWrite(temp_bins, ud_str)
                
if __name__ == '__main__':
    files_samples_outDir = cfg.ZHbbFitCfg
    #
    #
    makeDataCard(*files_samples_outDir)
