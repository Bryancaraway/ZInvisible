import ROOT
import csv
import sys
####################################
### This code is designed to add ###
### overflow to the last bin of  ###
### each histogram AND to output ###
### a .csv file with the ratio   ###
### of signal/bkg and yield      ###
####################################
def AssembleSelections(b,r,m):
    selCuts = []
    for i in b:
        for j in r:
            for k in m:
                selCuts.append(i+"_"+j+"_"+k)
    return selCuts

def doHistManip(hist):
    underflow = 0
    overflow  = len(hist)-1
    last_bin  = len(hist)-2
    hist.SetBinContent(len(hist)-2, hist.GetBinContent(overflow)+hist.GetBinContent(last_bin)) ## add overflow to last bin
    
#####CONFIG#####
inputRootFile  = "result"
outputCSVFile  = "tableContent" 

zPtCut   = 300.0
sampleTypes     = ["elec","mu"]
bselectionCuts  = ["nb0", "nbg0","nbg1"]
rTselectionCuts = ["nRt0","nRtg0","nRtg1"]
mTselectionCuts = ["nMt0","nMtg0","nMtg1"]
selectionCuts = AssembleSelections(bselectionCuts,rTselectionCuts,mTselectionCuts)

histName          = "bestRecoZPt" 
histNameExclusion = "NoZMassCut"
histSignal        = "tt#bar{t}Z_ll"
################
csvF = open(outputCSVFile+".csv","w+")
bkgSum = 0.0
csvDict = []
f = ROOT.TFile.Open(inputRootFile+".root","UPDATE")

print("PostProcessing Root File: "+inputRootFile+".root")
ROOT.TH1.AddDirectory(False)
for i in f.GetListOfKeys(): # Loop through directories
    cat = i.ReadObj()
    print cat
    if isinstance(cat, ROOT.TDirectoryFile):
        for sampleType in sampleTypes:
            for sel in selectionCuts: # Loop through Selections            
                csvDict = {"type": sampleType, "sel":sel, "bkg":0.0, "sig":0.0} 
                for j in cat.GetListOfKeys(): # Loop though hists
                    hist = j.ReadObj()
                    if isinstance(hist, ROOT.TH1):## do hist manip here ##
                        #doHistManip(hist)
                        if ((histName in hist.GetName()) and (histNameExclusion not in hist.GetName())):
                            if ((sel in hist.GetName()) and (sampleType in hist.GetName())):
                                h = hist.Clone()
                                for bin in range(1, h.GetNbinsX()+1):
                                    if (h.GetXaxis().GetBinCenter(bin) < zPtCut): 
                                        h.SetBinContent(bin,0.0)
                                if (histSignal in h.GetName()):
                                    csvDict["sig"]  = h.Integral()
                                else :
                                    csvDict["bkg"] += h.Integral()
                                del h

                if (csvDict["bkg"] != 0 and csvDict["sig"] != 0 ):
                    csvF.write(str(csvDict["type"])+","+str(csvDict["sel"])+","+str(csvDict["sig"]/csvDict["bkg"])+","+str(csvDict["sig"])+"\n")
                if (csvDict["bkg"] == 0 and csvDict["sig"] != 0 ):
                    csvF.write(str(csvDict["type"])+","+str(csvDict["sel"])+","+"NAN"+","+str(csvDict["sig"])+"\n")
                print ".",
print("PostProcessComplete!!!")        
#f.Write()
f.Close()
print("Writing to CSV File: "+outputCSVFile+".csv")
csvF.close()


