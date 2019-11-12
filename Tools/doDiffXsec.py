########################## 
### Code to create     ###
### differential x-sec ###
### measurement        ###
##########################
### written by:        ###
### Bryan Caraway      ###
##########################

### Root lib ###
import ROOT
from   ROOT import TCanvas, TProfile, TH1F, TFile
from   ROOT import gROOT
#
import uproot
import sys
import os
import numpy as np
from glob import glob
#

# global config
inputDir       = 'root/'
inputRootFiles = glob('./root/result_*.root')
outRootFile    = 'root/diffXSec.root' 
#
variableName   = 'bestRecoZPt'
baselineName   = 'genWeight'
enrichedRegion = 'nbg1_nRtg0'
# Luminosity in pb^-1
lumiDict       = {'2016'   : 35922.0,
                  '2017'   : 41856.0,
                  '2018_AB': 20757.0,
                  '2018_CD': 38148.0,
                  '2018'   : 58905.0}
#
signal         =       ['t#bar{t}Z_pass']
bkgDict        = {'C': ['t#bar{t}Z_fail'],
                  'E': ['DY',
                        'VVV',
                        'Diboson',
                        't(#bar{t})X',
                        't#bar{t}']}
pseudoData     = {'G': signal + bkgDict['C'] + bkgDict['E']}
#
def main():
    
    diffXSec = TH1F('xSec', 'xSec', 20, 0, 4)
    #
    G     = TH1F('G','G',1,300,1000)
    E     = TH1F('E','E',1,300,1000)
    C     = TH1F('C','C',1,300,1000)
    AandB = TH1F('A+B','A+B',1,300,1000)
    A     = TH1F('A','A',1,300,1000)
    Lumi  = lumiDict['2016'] + lumiDict['2017'] + lumiDict['2018']
    ## Loop through Root files and calculate quantities A+B, G-E-C
    def fillAandB(fileName):
        TDir = fileName.Get(baselineName)
        hists = TDir.GetListOfKeys()
        targetHist = [hist for hist in hists 
                      if (signal[0] in hist.GetName() and 'noGenW' not in hist.GetName())]
        targetHist = targetHist[0].ReadObj()
        tempHist   = [hist.ReadObj() for hist in hists
                      if ('t#bar{t}Z_fail' in hist.GetName() and 'noGenW' not in hist.GetName())]
        AandB.SetBinContent(1, AandB.GetBinContent(1) + targetHist.Integral())
        print(targetHist.Integral()+tempHist[0].Integral())
    def fillACEG(fileName):
        TDir = fileName.Get(variableName)
        hists = TDir.GetListOfKeys()
        targetA = [hist.ReadObj() for hist in hists 
                   if (signal[0] in hist.GetName() and enrichedRegion in hist.GetName())]
        targetC = [hist.ReadObj() for hist in hists
                   if (bkgDict['C'][0] in hist.GetName() and enrichedRegion in hist.GetName())]
        #
        targetEtot = 0
        for bkg in bkgDict['E']:
            targetE = [hist.ReadObj() for hist in hists
                      if (bkg in hist.GetName() and enrichedRegion in hist.GetName())]
            targetEtot += targetE[0].Integral() + targetE[1].Integral()
        E.SetBinContent(1, E.GetBinContent(1) + targetEtot)
        #
        targetA = targetA[0].Integral() + targetA[1].Integral()
        targetC = targetC[0].Integral() + targetC[1].Integral()
        A.SetBinContent(1, A.GetBinContent(1) + targetA)
        C.SetBinContent(1, C.GetBinContent(1) + targetC)
        #
        #print(targetEtot,targetA,targetC)
        G.SetBinContent(1, G.GetBinContent(1) + 
                        targetA + targetC + targetEtot)
        #print(E.Integral(), A.Integral(), C.Integral())
        #print(G.Integral())

    for inputRootFile in inputRootFiles:
        f = TFile(inputRootFile,'READ')
        fillAandB(f)
        fillACEG(f)

    for _ in xrange(0,10000):
        diffXsecMeas = 0
        G_clone = G.Clone()
        G_clone.Scale(np.random.poisson(G.GetBinContent(1))/(G.GetBinContent(1)))
        
        diffXsecMeas = (G_clone.Integral() - E.Integral() - C.Integral())*(AandB.Integral()/A.Integral())*(1/Lumi)*1000
        diffXSec.Fill(diffXsecMeas)
        #diffXSec.SetBinContent(1, G.Integral() - E.Integral() - C.Integral())
        #diffXSec.Scale(AandB.Integral()/A.Integral())
        #diffXSec.Scale((1/Lumi*1000))
    from time import sleep
    diffXSec.GetXaxis().SetTitle('#sigma(Z P_{t}>300) fb')
    diffXSec.GetYaxis().SetTitle('Frequency')
    diffXSec.Draw()
    sleep(30)
    
    
if __name__ == "__main__":
    main()
