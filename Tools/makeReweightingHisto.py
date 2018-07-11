#!/cvmfs/cms.cern.ch/slc6_amd64_gcc491/cms/cmssw/CMSSW_7_4_8/external/slc6_amd64_gcc491/bin/python

from optparse import OptionParser

import array, sys
# Check if we want the help, if not import ROOT (otherwise ROOT overwrites the help)
if '-h' not in sys.argv and '--help' not in sys.argv:
    from ROOT import TH1D, TMath, TFile, TCanvas, TF1, TH2D

from os import environ
sys.path = [environ["CMSSW_BASE"] + "/src/SusyAnaTools/Tools/condor/",] + sys.path
from samples import SampleCollection

from math import sqrt
import math

from ctypes import cdll
from ctypes import c_float

class ScaleFactors:
    def __init__(self):
        self.lib = cdll.LoadLibrary('obj/ScaleFactors.so')
        print(dir(self.lib))
        self.lib.python_sf_norm0b.restype = c_float
        self.lib.python_sf_norm0b_err.restype = c_float

    def getRnorm(self):
        return float(self.lib.python_sf_norm0b())

    def getRnormErr(self):
        return float(self.lib.python_sf_norm0b_err())

############################
##  Some utilities first  ##
############################
if '-h' not in sys.argv and '--help' not in sys.argv:
    c = TCanvas("c1", "c1", 800, 800)

def rebin1D(h, bins):
    """Rebin histo h to bins and recompute the errors."""
    new_h = TH1D("%s_rebin_"%(h.GetName()), "%s_rebin_"%(h.GetName()),
                 len(bins)-1, array.array('d', bins))
    new_h.Sumw2()
    for i in xrange(h.GetNbinsX()):
        bin_to_fill = new_h.FindBin(h.GetBinCenter(i+1))
        new_h.SetBinContent(bin_to_fill, h.GetBinContent(i+1) + new_h.GetBinContent(bin_to_fill))
        new_h.SetBinError(bin_to_fill, TMath.Sqrt(h.GetBinError(i+1)*h.GetBinError(i+1) + new_h.GetBinError(bin_to_fill)*new_h.GetBinError(bin_to_fill) ) )
    return new_h

def rebin2D(h, binsx, binsy):
    """Rebin histo h to binsx and binsy and recompute the errors."""
    new_h = TH2D("%s_rebin_"%(h.GetName()), "%s_rebin_"%(h.GetName()),
                 len(binsx)-1, array.array('d', binsx), len(binsy)-1, array.array('d', binsy))
    new_h.Sumw2()
    for i in xrange(h.GetNbinsX()):
        bin_to_fill_x = new_h.GetXaxis().FindBin(h.GetXaxis().GetBinCenter(i+1))
        for j in xrange(h.GetNbinsY()):
            bin_to_fill_y = new_h.GetYaxis().FindBin(h.GetYaxis().GetBinCenter(j+1))
            new_h.SetBinContent(bin_to_fill_x, bin_to_fill_y, h.GetBinContent(i+1,j+1) + new_h.GetBinContent(bin_to_fill_x, bin_to_fill_y))
            new_h.SetBinError(bin_to_fill_x, bin_to_fill_y,
                              TMath.Sqrt(h.GetBinError(i+1, j+1)*h.GetBinError(i+1,j+1) + new_h.GetBinError(bin_to_fill_x, bin_to_fill_y)*new_h.GetBinError(bin_to_fill_x, bin_to_fill_y) ) )
    return new_h

def add(hs):
    """Add all histograms in list hs and return output."""
    hNew = hs[0].Clone()
    if len(hs) > 1:
        for h in hs[1:]:
            hNew.Add(h)
    return hNew

def makeRatio(h1, h2s, bins=None, newname="test"):
    """Make the data/stack ratio for the new bins."""
    # Add stack together
    h2 = add(h2s)
    if bins != None:
        # Rebin the histograms
        if type(bins) is list:
            h1 = rebin1D(h1, bins)
            h2 = rebin1D(h2, bins)
        elif type(bins) is tuple:
            h1 = rebin2D(h1, bins[0], bins[1])
            h2 = rebin2D(h2, bins[0], bins[1])
    # Make the ratio
    h1.Sumw2()
    h1.Divide(h2)
    h1.SetName(newname)
    h1.SetTitle(newname)
    return h1

def reweight(h, hsf):
    """Reweight histogram h according to histogram hsf: h_i * hsf_i, for each bin i"""
    new_h = h.Clone()
    for i in xrange(new_h.GetNbinsX()):
        # Get the scale factor
        sfbin = hsf.FindBin(new_h.GetBinCenter(i+1))
        sf = hsf.GetBinContent(sfbin)
        sf_e = hsf.GetBinError(sfbin)
        if sf == 0: continue
        # Get the old bin info
        bc_old = new_h.GetBinContent(i+1)
        be_old = new_h.GetBinError(i+1)
        if bc_old == 0: continue
        # Get the new bin content
        bc = bc_old*sf
        # Get the new bin error, just use basic error propagation without correlations: [E(AB)/(AB)]^2 = [E(A)/A]^2 + [E(B)/B]^2
        be = bc * TMath.Sqrt( (be_old/bc_old)*(be_old/bc_old) + (sf_e/sf)*(sf_e/sf) )
        # Write the new info
        new_h.SetBinContent(i+1,bc)
        new_h.SetBinError(i+1,be)
    return new_h

def subtract(h, hlist):
    """Subtract all histograms in hlist from h"""
    new_h = h.Clone()
    new_h.Sumw2()
    for hl in hlist:
        new_h.Add(hl,-1.)
        #print hl.GetName(), new_h.Integral()
    return new_h

##############################
##  Main scale factor code  ##
##############################

##################
## Njet weights ##
##################
def njetWeights(filename):
    # Get the file
    f = TFile.Open(filename)

    # Prepare writing to a file
    fout = TFile.Open("dataMCweights.root","RECREATE")
    fout.cd()

    # How to rebin
    bins = [0,1,2,3,4,5,6,7,20]
    #bins_TT = [0,1,2,3,4,5,6,20]

    # Run over the relevant histograms
    cuts_GJets = ["LooseLepVeto"] 
    #cuts_DY = ["muZinv", "muZinv_0b", "muZinv_g1b"]
    #cuts_TT = ["elmuZinv", "elmuZinv_0b", "elmuZinv_g1b"]
    #selection = "loose0_mt2_MET"#"loose0_mt2_MET"#"blnotag"#"loose0"#"loose0_mt2"
    # histo names
    #hname1 = "cntNJetsPt30Eta24Zinv/DataMC_SingleMuon_nj_%(cut)s_%(selection)scntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvDatadata"
    hname1 = "nJets/dataMC_Photon_%(cut)snJetsnJetsDatadata",

    hnames2 = ["nJets/dataMC_Photon_%(cut)snJetsnJetsQCDstack",
               "nJets/dataMC_Photon_%(cut)snJetsnJetsTTGJets",
               "nJets/dataMC_Photon_%(cut)snJetsnJetsW(l#nu)+jetsstack",
               "nJets/dataMC_Photon_%(cut)snJetsnJetst#bar{t}stack",
               "nJets/dataMC_Photon_%(cut)snJetsnJetsDibosonstack",
               "nJets/dataMC_Photon_%(cut)snJetsnJetstWstack",
               "nJets/dataMC_Photon_%(cut)snJetsnJetsRarestack",
               "nJets/dataMC_Photon_%(cut)snJetsnJetst#bat{t}Zstack",
               ]

    # dictionary to keep track of all the scale factors
    SFs = {}

    # Now get GJets reweighting:
    # Procedure: 1. Apply TTbar reweighting to TTbar MC
    #            2. Subtract non-DY MC from data
    #            3. Make ratio of subtracted data and DY
    for cut in cuts_GJets:
        hname1_GJets = hname1 % {"cut":cut}
        hnames2_GJets = [elem % {"cut":cut} for elem in hnames2]
        # Get all histos
        h1 = f.Get(hname1_GJets)
        h2s = [f.Get(hname2_GJets) for hname2_GJets in hnames2_GJets]

        # apply weights to ttbar
        #h2s[1] = reweight(h2s[1], SFs["TT_elmuZinv"])#"TT_%s"%(cut.replace("mu","elmu"))])#weight applied to Njet for the Rnorm or DY correction
        #h2s[1].Scale( SFs["TT_%s"%(cut.replace("mu","elmu"))].GetBinContent(1) )

        # subtract relevant histograms from data
        data_subtracted = subtract(h1, h2s[1:])

        newname = "DataMC_nj_%s"%(cut)
        newh = makeRatio(data_subtracted, [h2s[0]], bins, newname)

        SFs["GJets_%s"%(cut)] = newh.Clone()
        newh.Write()

    #calculate Rnorm
    normWeightwithReweight(f, SFs)

    # Close files
    fout.Close()
    f.Close()
##For new ttbar SF normalization##
includeBasettBar = """#ifndef SCALEFACTORSTTBAR_H
#define SCALEFACTORSTTBAR_H

class ScaleFactorsttBar
{
public:
    static float sf_norm0b()
    {
        return %0.3f;
    }
    static float sfunc_norm0b()
    {
        return %0.3f;
    }
};

#endif
"""

sourceBasettBar = """#include "ScaleFactorsttBar.h"

extern "C"
{
    static float python_sf_norm0b()
    {
        return ScaleFactors::sf_norm0b();
    }
    static float python_sf_norm0b_err()
    {
        return ScaleFactors::sfunc_norm0b();
    }
}
"""


##################
## norm weights ##
##################

includeBase = """#ifndef SCALEFACTORS_H
#define SCALEFACTORS_H

class ScaleFactors
{
public:
    static float sf_norm0b()
    {
	return %0.3f;
    }
    static float sfunc_norm0b()
    {
	return %0.3f;
    }
};

#endif
"""

sourceBase = """#include "ScaleFactors.h"

extern "C" 
{
    static float python_sf_norm0b()
    {
        return ScaleFactors::sf_norm0b();
    }
    static float python_sf_norm0b_err()
    {
        return ScaleFactors::sfunc_norm0b();
    }
}
"""

def normWeightwithReweight(f, SFs):
    # Run over the relevant histograms
    cuts_DY = ["muZinv"]
    selections = ["g1b_blnotag","0b_blnotag"]
    selection2 = "blnotag"
    # histo names
    hname1 = "cntNJetsPt30Eta24Zinv/DataMC_SingleMuon_nj_%(cut)s_%(selection)scntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvDatadata"
    hnames2 = ["cntNJetsPt30Eta24Zinv/DataMC_SingleMuon_nj_%(cut)s_%(selection)scntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvDYstack",
               #"cntNJetsPt30Eta24Zinv/DataMCw_SingleMuon_nj_%(cut)s_%(selection)scntCSVSZinvcntCSVSZinvDY HT<100stack",
               "cntNJetsPt30Eta24Zinv/DataMC_SingleMuon_nj_%(cut)s_%(selection)scntNJetsPt30Eta24ZinvcntNJetsPt30Eta24Zinvt#bar{t}stack",
               "cntNJetsPt30Eta24Zinv/DataMC_SingleMuon_nj_%(cut)s_%(selection)scntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvSingle topstack",
               "cntNJetsPt30Eta24Zinv/DataMC_SingleMuon_nj_%(cut)s_%(selection)scntNJetsPt30Eta24ZinvcntNJetsPt30Eta24Zinvt#bar{t}Zstack",
               "cntNJetsPt30Eta24Zinv/DataMC_SingleMuon_nj_%(cut)s_%(selection)scntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvDibosonstack",
               "cntNJetsPt30Eta24Zinv/DataMC_SingleMuon_nj_%(cut)s_%(selection)scntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvRarestack"
               ]

    # Procedure: 1. Grab the njet reweighted MC
    #            2. Subtract non-DY MC from data
    #            3. Make ratio of subtracted data and DY
    for cut in cuts_DY:
        hname1_DY_0b = hname1 % {"cut":cut, "selection":selections[1]}
        hnames2_DY_0b = [elem % {"cut":cut, "selection":selections[1]} for elem in hnames2]
        hname1_DY_g1b = hname1 % {"cut":cut, "selection":selections[0]}
        hnames2_DY_g1b = [elem % {"cut":cut, "selection":selections[0]} for elem in hnames2]
        # Get all histos
        h1_0b = f.Get(hname1_DY_0b)
        print h1_0b.Integral()
        h2s_0b = [f.Get(sel) for sel in hnames2_DY_0b]
        h1_g1b = f.Get(hname1_DY_g1b)
        print h1_g1b.Integral()
        h2s_g1b = [f.Get(sel) for sel in hnames2_DY_g1b]

        #h1test = f.Get(hname1%{"cut":cut, "selection":selection2})
        #print h1test.Integral()

        # apply weights to DY
        h2s_0b[0]  = reweight(h2s_0b[0],  SFs["DY_muZinv_0b"])
        h2s_g1b[0] = reweight(h2s_g1b[0], SFs["DY_muZinv_g1b"])

        # apply weights to ttbar
        h2s_0b[1].Scale(SFs["TT_elmuZinv_0b"].GetBinContent(1))#.Scale(0.92)
        h2s_g1b[1].Scale(SFs["TT_elmuZinv_g1b"].GetBinContent(1))#.Scale(0.92)

        # Combine histograms
        h1 = h1_g1b.Clone()
        h1.Add(h1_0b)
        h2s = []
        for i in xrange(len(h2s_0b)):
            h2s.append(h2s_g1b[i].Clone())
            h2s[i].Add(h2s_0b[i])

        # subtract relevant histograms from data
        data_subtracted = subtract(h1, h2s[1:])
        #print data_subtracted.Integral()
        newname = "DataMC_nj_%s_%s"%(cut,selection2)
        newh = makeRatio(data_subtracted, [h2s[0]], newname=newname, bins=[0,20])
        #newh = makeRatio(h1, h2s, newname=newname)

        print "Data/MC normalization scale factor in region %s_%s: %.3f +- %.3f" % (cut, selection2, newh.GetBinContent(1), newh.GetBinError(1))

        includeFile = open("ScaleFactors.h", "w")
        includeFile.write(includeBase%(newh.GetBinContent(1), newh.GetBinError(1)))
        includeFile.close()

        sourceFile = open("ScaleFactors.cc", "w") 
        sourceFile.write(sourceBase)
        sourceFile.close()



##################
## norm weights ##
##################
def normWeight(filename):
    # Get the file
    f = TFile.Open(filename)
    # Run over the relevant histograms
    cuts_DY = ["muZinv"]
    selections = ["bl","0b_blnotag"]
    selection2 = "blnotag"
    # histo names
    hname1 = "cntCSVSZinv/DataMC_SingleMuon_nb_%(cut)s_%(selection)scntCSVSZinvcntCSVSZinvDatadata"
    hnames2 = ["cntCSVSZinv/DataMCw_SingleMuon_nb_%(cut)s_%(selection)scntCSVSZinvcntCSVSZinvDYstack",
               #"cntCSVSZinv/DataMCw_SingleMuon_nb_%(cut)s_%(selection)scntCSVSZinvcntCSVSZinvDY HT<100stack",
               "cntCSVSZinv/DataMCw_SingleMuon_nb_%(cut)s_%(selection)scntCSVSZinvcntCSVSZinvt#bar{t}stack",
               "cntCSVSZinv/DataMCw_SingleMuon_nb_%(cut)s_%(selection)scntCSVSZinvcntCSVSZinvSingle topstack",
               "cntCSVSZinv/DataMCw_SingleMuon_nb_%(cut)s_%(selection)scntCSVSZinvcntCSVSZinvt#bar{t}Zstack",
               "cntCSVSZinv/DataMCw_SingleMuon_nb_%(cut)s_%(selection)scntCSVSZinvcntCSVSZinvDibosonstack",
               "cntCSVSZinv/DataMCw_SingleMuon_nb_%(cut)s_%(selection)scntCSVSZinvcntCSVSZinvRarestack"
               ]

    # Procedure: 1. Grab the njet reweighted MC
    #            2. Subtract non-DY MC from data
    #            3. Make ratio of subtracted data and DY
    for cut in cuts_DY:
        #hname1_DY = [hname1 % {"cut":cut, "selection":selection} for selection in selections]
        #hnames2_DY = [[elem % {"cut":cut, "selection":selection} for selection in selections] for elem in hnames2]
        hname1_DY = hname1 % {"cut":cut, "selection":selection2}
        hnames2_DY = [elem % {"cut":cut, "selection":selection2} for elem in hnames2]
        # Get all histos
        #h1 = add([f.Get(hname1a_DY) for hname1a_DY in hname1_DY])
        #h2s = [add([f.Get(sel) for sel in hname2a_DY]) for hname2a_DY in hnames2_DY]
        h1 = f.Get(hname1_DY)
        h2s = [f.Get(sel) for sel in hnames2_DY]
        # subtract relevant histograms from data
        data_subtracted = subtract(h1, h2s[1:])
        #print data_subtracted.Integral()
        newname = "DataMC_nb_%s_%s"%(cut,selection2)
        newh = makeRatio(data_subtracted, [h2s[0]], newname=newname, bins=[0,6])
        #newh = makeRatio(h1, h2s, newname=newname)

        print "Data/MC normalization scale factor in region %s_%s: %.3f +- %.3f" % (cuts_DY[0], selection2, newh.GetBinContent(1), newh.GetBinError(1))

    f.Close()

def shapeSyst(filename):
    # Get the file
    f = TFile.Open(filename)
    fout = TFile.Open("syst_shape.root", "RECREATE")
    # Run over the relevant histograms
    # histo names
    hnameData = "%(var)s/DataMCw_SingleMuon_%(name)s_muZinv_loose0_mt2_MET%(var)s%(var)sDatadata"
    hnames2 =  ["%(var)s/DataMCw_SingleMuon_%(name)s_muZinv_loose0_mt2_MET%(var)s%(var)sDYstack",
                #"%(var)s/DataMCw_SingleMuon_%(name)s_muZinv_loose0_mt2%(var)s%(var)sDY HT<100stack",
                "%(var)s/DataMCw_SingleMuon_%(name)s_muZinv_loose0_mt2_MET%(var)s%(var)st#bar{t}stack",
                "%(var)s/DataMCw_SingleMuon_%(name)s_muZinv_loose0_mt2_MET%(var)s%(var)sSingle topstack",
                "%(var)s/DataMCw_SingleMuon_%(name)s_muZinv_loose0_mt2_MET%(var)s%(var)st#bar{t}Zstack",
                "%(var)s/DataMCw_SingleMuon_%(name)s_muZinv_loose0_mt2_MET%(var)s%(var)sDibosonstack",
                "%(var)s/DataMCw_SingleMuon_%(name)s_muZinv_loose0_mt2_MET%(var)s%(var)sRarestack"]

    varList = [#["met", "cleanMetPt",             [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 1500], "MET" ],
               #["mt2", "best_had_brJet_MT2Zinv", [0, 50, 100, 150, 200, 250, 300, 350, 400, 1500],      "M_{T2}" ],
               #["met", "cleanMetPt",             [0, 100, 200, 275, 350, 450, 1500], "MET" ],
               ["met", "cleanMetPt",             [0, 100, 200, 350, 500, 650, 1500], "MET" ],
               ["mt2", "best_had_brJet_MT2Zinv", [0, 100, 200, 350, 450, 1500],      "M_{T2}" ],
               ["nt",  "nTopCandSortedCntZinv",  [0, 1, 2, 8 ],                                         "N(t)" ],
               ["nb",  "cntCSVSZinv",            [0, 1, 2, 8 ],                                      "N(b)" ],
               ["ht",  "HTZinv",                 [0, 100, 200, 350, 500, 650,800,1200, 1500],   "HT"],
               ["nj",  "cntNJetsPt30Eta24Zinv",            [4,5,6,7,8,15 ],                                      "N(jets)" ]]
    for var in varList:
        # Procedure: 1. Grab the njet reweighted MC
        #            2. Subtract non-DY MC from data
        #            3. Make ratio of subtracted data and DY
        # Get all histos
        hData = f.Get(hnameData%{"name":var[0], "var":var[1]})
        h2s   = [f.Get(hname2%{"name":var[0], "var":var[1]}) for hname2 in hnames2]

        # subtract relevant histograms from data
        data_subtracted = subtract(hData, h2s[1:])

        newname = "ShapeRatio_%s"%var[0]
        newh = makeRatio(data_subtracted, [h2s[0]], newname=newname, bins=var[2])

        for i in xrange(1, newh.GetNbinsX() + 1):
            print newh.GetBinContent(i)

        fit = TF1("fit_%s"%var[0], "pol1")
        newh.SetLineWidth(2)
        newh.GetXaxis().SetTitle(var[3])
        newh.GetYaxis().SetTitle("Data/MC")
        newh.GetYaxis().SetTitleOffset(1.15)
        newh.SetStats(0)
        newh.SetTitle("")
        #if not "nt" in var[0] and not "nb" in var[0]:
        #    newh.Fit(fit, "")
        #else:
        #    newh.Draw()
        newh.Draw()
        #
        #    fit.Draw("same")
        #newh.DrawCopy()
        c.Print("shapeSyst_%s.png"%var[0])
        c.Print("shapeSyst_%s.pdf"%var[0])

        fout.cd()
        newh.Write()

    f.Close()
    fout.Close()

def corrCheck():
    # Get the file
    f = TFile.Open("/uscms/home/pastika/nobackup/zinv/dev/CMSSW_7_4_8/src/ZInvisible/Tools/Corrolation_MET-MT2.root")
    fout = TFile.Open("correlations_ratio.root", "RECREATE")

    # Make 1D and 2D ratios for both smearing options

    # Run over the relevant histograms
    # histo names
    types = ["Gaus", "Logi"]
    variables = ["MET", "MT2", "MT2vMET"]
    basename = "h%(var)s_%(typ)s"

    bins_var = {"MET": [0, 100, 200, 300, 600, 1500],
                "MT2": [0, 100, 200, 300, 400, 1500],
                "MT2vMET": ([0, 100, 200, 300, 600, 1500],[0, 100, 200, 300, 400, 1500])}
    for typ in types:
        # Procedure: 1. Grab the nominal and varied
        #            2. Make ratio
        for var in variables:
            # Get all histos
            hnom = f.Get("h%(var)s_nom"%locals())
            hsmeared   = f.Get(basename%locals())
            hnom_rebin = hnom.Clone(hnom.GetName()+"_rebin")
            hsmeared_rebin = hsmeared.Clone(hsmeared.GetName()+"_rebin")
            if type(bins_var[var]) is list:
                hnom_rebin = rebin1D(hnom_rebin, bins_var[var])
                hsmeared_rebin = rebin1D(hsmeared, bins_var[var])
            elif type(bins_var[var]) is tuple:
                hnom_rebin = rebin2D(hnom_rebin, bins_var[var][0], bins_var[var][1])
                hsmeared_rebin = rebin2D(hsmeared, bins_var[var][0], bins_var[var][1])

            newname = "Ratio_%s_%s"%(var,typ)
            newh = makeRatio(hsmeared, [hnom], newname=newname, bins=bins_var[var])

            for i in xrange(1, newh.GetNbinsX() + 1):
                print newh.GetBinContent(i)

            newh.SetLineWidth(2)
            newh.GetXaxis().SetTitle(var)
            newh.GetYaxis().SetTitle("Smeared/Nom")
            newh.GetYaxis().SetTitleOffset(1.15)
            newh.SetStats(0)
            newh.SetTitle("")

            fout.cd()
            newh.Write()
            hsmeared_rebin.Write()
            hnom_rebin.Write()

    f.Close()
    fout.Close()


def systHarvest(filename):
    # Get the file
    #f = TFile.Open(filename)
    #fout = TFile.Open("syst_shape.root", "RECREATE")
    # Run over the relevant histograms
    # histo names
    NSB = 84

    # Get shape central value uncertainty
    f = TFile("systematics.root")

    hShape_MET_Nom1 = f.Get("nSearchBin/systWgtTest_cleanMetPtnSearchBinnSearchBinNominalsingle")
    hShape_MET_Var = f.Get("nSearchBin/systWgtMET_cleanMetPtnSearchBinnSearchBinvariedsingle")
    hShape_MT2_Nom = f.Get("nSearchBin/systWgtMT2_best_had_brJet_MT2ZinvnSearchBinnSearchBinNominalsingle")
    hShape_MT2_Var = f.Get("nSearchBin/systWgtMT2_best_had_brJet_MT2ZinvnSearchBinnSearchBinvariedsingle")
    hShape_NT_Nom  = f.Get("nSearchBin/systWgtNT_nTopCandSortedCntZinvnSearchBinnSearchBinNominalsingle")
    hShape_NT_Var  = f.Get("nSearchBin/systWgtNT_nTopCandSortedCntZinvnSearchBinnSearchBinvariedsingle")
    hShape_NB_Nom  = f.Get("nSearchBin/systWgtNB_cntCSVSZinvnSearchBinnSearchBinNominalsingle")
    hShape_NB_Var  = f.Get("nSearchBin/systWgtNB_cntCSVSZinvnSearchBinnSearchBinvariedsingle")
    #When we move to the spacing than it is done!
    #hShape_MET_Nom1 = f.Get("nSearchBin/systWgtTest_cleanMetPt__nSearchBin______nSearchBin__Nominal__single")
    #hShape_MET_Var = f.Get("nSearchBin/systWgtMET_cleanMetPt__nSearchBin______nSearchBin__varied__single")
    #hShape_MT2_Nom = f.Get("nSearchBin/systWgtMT2_best_had_brJet_MT2Zinv__nSearchBin______nSearchBin__Nominal__single")
    #hShape_MT2_Var = f.Get("nSearchBin/systWgtMT2_best_had_brJet_MT2Zinv__nSearchBin______nSearchBin__varied__single")
    #hShape_NT_Nom  = f.Get("nSearchBin/systWgtNT_nTopCandSortedCntZinv__nSearchBin______nSearchBin__Nominal__single")
    #hShape_NT_Var  = f.Get("nSearchBin/systWgtNT_nTopCandSortedCntZinv__nSearchBin______nSearchBin__varied__single")
    #hShape_NB_Nom  = f.Get("nSearchBin/systWgtNB_cntCSVSZinv__nSearchBin______nSearchBin__Nominal__single")
    #hShape_NB_Var  = f.Get("nSearchBin/systWgtNB_cntCSVSZinv__nSearchBin______nSearchBin__varied__single")

    print hShape_MET_Var.Clone("nSearchBin/"+hShape_MET_Nom1.GetName()+"_ratio").GetName(), f
    hShape_MET_ratio = hShape_MET_Var.Clone("nSearchBin/"+hShape_MET_Nom1.GetName()+"_ratio")
    print hShape_MET_Var.GetName()
    hShape_MET_ratio.Divide(hShape_MET_Nom1)

    hShape_MT2_ratio = hShape_MT2_Var.Clone(hShape_MT2_Nom.GetName()+"_ratio")
    hShape_MT2_ratio.Divide(hShape_MT2_Nom)

    hShape_NT_ratio = hShape_NT_Var.Clone(hShape_NT_Nom.GetName()+"_ratio")
    hShape_NT_ratio.Divide(hShape_NT_Nom)

    hShape_NB_ratio = hShape_NB_Var.Clone(hShape_NB_Nom.GetName()+"_ratio")
    hShape_NB_ratio.Divide(hShape_NB_Nom)

    hShape_final = hShape_MET_ratio.Clone("shape_central")

    for i in xrange(1, hShape_MET_ratio.GetNbinsX() + 1):
        uncertMET = hShape_MET_ratio.GetBinContent(i) - 1 if hShape_MET_ratio.GetBinContent(i) > 0 else 0
        uncertMT2 = hShape_MT2_ratio.GetBinContent(i) - 1 if hShape_MT2_ratio.GetBinContent(i) > 0 else 0
        uncertNT = hShape_NT_ratio.GetBinContent(i) - 1 if hShape_NT_ratio.GetBinContent(i) > 0 else 0
        uncertNB = hShape_NB_ratio.GetBinContent(i) - 1 if hShape_NB_ratio.GetBinContent(i) > 0 else 0
        hShape_final.SetBinContent(i, sqrt(uncertMET**2 + uncertMT2**2 + uncertNT**2 + uncertNB**2))

    fout = TFile("syst_all.root", "RECREATE")
    hShape_final.Write()


    # Get correlation study info
    hCorr_METGaus_Nom = f.Get("nSearchBin/CorrMETGaus_cleanMetPtnSearchBinnSearchBinNominalsingle")
    hCorr_METGaus_Var = f.Get("nSearchBin/CorrMETGaus_cleanMetPtnSearchBinnSearchBinvariedsingle")
    hCorr_MT2Gaus_Nom = f.Get("nSearchBin/CorrMT2Gaus_best_had_brJet_MT2ZinvnSearchBinnSearchBinNominalsingle")
    hCorr_MT2Gaus_Var = f.Get("nSearchBin/CorrMT2Gaus_best_had_brJet_MT2ZinvnSearchBinnSearchBinvariedsingle")
    hCorr_MT2vMETGaus_Nom = f.Get("nSearchBin/CorrMT2vMETGaus_cleanMetPt_best_had_brJet_MT2ZinvnSearchBinnSearchBinNominalsingle")
    hCorr_MT2vMETGaus_Var = f.Get("nSearchBin/CorrMT2vMETGaus_cleanMetPt_best_had_brJet_MT2ZinvnSearchBinnSearchBinvariedsingle")
    hCorr_METLogi_Nom = f.Get("nSearchBin/CorrMETLogi_cleanMetPtnSearchBinnSearchBinNominalsingle")
    hCorr_METLogi_Var = f.Get("nSearchBin/CorrMETLogi_cleanMetPtnSearchBinnSearchBinvariedsingle")
    hCorr_MT2Logi_Nom = f.Get("nSearchBin/CorrMT2Logi_best_had_brJet_MT2ZinvnSearchBinnSearchBinNominalsingle")
    hCorr_MT2Logi_Var = f.Get("nSearchBin/CorrMT2Logi_best_had_brJet_MT2ZinvnSearchBinnSearchBinvariedsingle")
    hCorr_MT2vMETLogi_Nom = f.Get("nSearchBin/CorrMT2vMETLogi_cleanMetPt_best_had_brJet_MT2ZinvnSearchBinnSearchBinNominalsingle")
    hCorr_MT2vMETLogi_Var = f.Get("nSearchBin/CorrMT2vMETLogi_cleanMetPt_best_had_brJet_MT2ZinvnSearchBinnSearchBinvariedsingle")

    #hCorr_METGaus_Nom = f.Get("nSearchBin/CorrMETGaus_cleanMetPt__nSearchBin______nSearchBin__Nominal__single")
    #hCorr_METGaus_Var = f.Get("nSearchBin/CorrMETGaus_cleanMetPt__nSearchBin______nSearchBin__varied__single")
    #hCorr_MT2Gaus_Nom = f.Get("nSearchBin/CorrMT2Gaus_best_had_brJet_MT2Zinv__nSearchBin______nSearchBin__Nominal__single")
    #hCorr_MT2Gaus_Var = f.Get("nSearchBin/CorrMT2Gaus_best_had_brJet_MT2Zinv__nSearchBin______nSearchBin__varied__single")
    #hCorr_MT2vMETGaus_Nom = f.Get("nSearchBin/CorrMT2vMETGaus_cleanMetPt_best_had_brJet_MT2Zinv__nSearchBin______nSearchBin__Nominal__single")
    #hCorr_MT2vMETGaus_Var = f.Get("nSearchBin/CorrMT2vMETGaus_cleanMetPt_best_had_brJet_MT2Zinv__nSearchBin______nSearchBin__varied__single")
    #hCorr_METLogi_Nom = f.Get("nSearchBin/CorrMETLogi_cleanMetPt__nSearchBin______nSearchBin__Nominal__single")
    #hCorr_METLogi_Var = f.Get("nSearchBin/CorrMETLogi_cleanMetPt__nSearchBin______nSearchBin__varied__single")
    #hCorr_MT2Logi_Nom = f.Get("nSearchBin/CorrMT2Logi_best_had_brJet_MT2Zinv__nSearchBin______nSearchBin__Nominal__single")
    #hCorr_MT2Logi_Var = f.Get("nSearchBin/CorrMT2Logi_best_had_brJet_MT2Zinv__nSearchBin______nSearchBin__varied__single")
    #hCorr_MT2vMETLogi_Nom = f.Get("nSearchBin/CorrMT2vMETLogi_cleanMetPt_best_had_brJet_MT2Zinv__nSearchBin______nSearchBin__Nominal__single")
    #hCorr_MT2vMETLogi_Var = f.Get("nSearchBin/CorrMT2vMETLogi_cleanMetPt_best_had_brJet_MT2Zinv__nSearchBin______nSearchBin__varied__single")

    hCorr_METGaus_ratio = hCorr_METGaus_Var.Clone(hCorr_METGaus_Nom.GetName()+"_ratio")
    hCorr_METGaus_ratio.Divide(hCorr_METGaus_Nom)
    hCorr_METGaus_ratio.Write()

    hCorr_MT2Gaus_ratio = hCorr_MT2Gaus_Var.Clone(hCorr_MT2Gaus_Nom.GetName()+"_ratio")
    hCorr_MT2Gaus_ratio.Divide(hCorr_MT2Gaus_Nom)
    hCorr_MT2Gaus_ratio.Write()

    hCorr_MT2vMETGaus_ratio = hCorr_MT2vMETGaus_Var.Clone(hCorr_MT2vMETGaus_Nom.GetName()+"_ratio")
    hCorr_MT2vMETGaus_ratio.Divide(hCorr_MT2vMETGaus_Nom)

    hCorr_METLogi_ratio = hCorr_METLogi_Var.Clone(hCorr_METLogi_Nom.GetName()+"_ratio")
    hCorr_METLogi_ratio.Divide(hCorr_METLogi_Nom)
    hCorr_METLogi_ratio.Write()

    hCorr_MT2Logi_ratio = hCorr_MT2Logi_Var.Clone(hCorr_MT2Logi_Nom.GetName()+"_ratio")
    hCorr_MT2Logi_ratio.Divide(hCorr_MT2Logi_Nom)
    hCorr_MT2Logi_ratio.Write()

    hCorr_MT2vMETLogi_ratio = hCorr_MT2vMETLogi_Var.Clone(hCorr_MT2vMETLogi_Nom.GetName()+"_ratio")
    hCorr_MT2vMETLogi_ratio.Divide(hCorr_MT2vMETLogi_Nom)

    hCorr_Gaus_final = hCorr_METGaus_ratio.Clone("Corr_1D_Gauss")
    hCorr_Logi_final = hCorr_METLogi_ratio.Clone("Corr_1D_Logistic")
    for i in xrange(1, hCorr_METGaus_ratio.GetNbinsX() + 1):
        uncertCorrMET = hCorr_METGaus_ratio.GetBinContent(i) - 1 if hCorr_METGaus_ratio.GetBinContent(i) > 0 else 0
        uncertCorrMT2 = hCorr_MT2Gaus_ratio.GetBinContent(i) - 1 if hCorr_MT2Gaus_ratio.GetBinContent(i) > 0 else 0
        hCorr_Gaus_final.SetBinContent(i, sqrt(uncertCorrMET**2 + uncertCorrMT2**2))
    for i in xrange(1, hCorr_METLogi_ratio.GetNbinsX() + 1):
        uncertCorrMET = hCorr_METLogi_ratio.GetBinContent(i) - 1 if hCorr_METLogi_ratio.GetBinContent(i) > 0 else 0
        uncertCorrMT2 = hCorr_MT2Logi_ratio.GetBinContent(i) - 1 if hCorr_MT2Logi_ratio.GetBinContent(i) > 0 else 0
        hCorr_Logi_final.SetBinContent(i, sqrt(uncertCorrMET**2 + uncertCorrMT2**2))
    for i in xrange(1, hCorr_MT2vMETGaus_ratio.GetNbinsX()+1):
        hCorr_MT2vMETGaus_ratio.SetBinContent(i,abs(hCorr_MT2vMETGaus_ratio.GetBinContent(i)-1) if hCorr_MT2vMETGaus_ratio.GetBinContent(i)>0 else 0)
        hCorr_MT2vMETLogi_ratio.SetBinContent(i,abs(hCorr_MT2vMETLogi_ratio.GetBinContent(i)-1) if hCorr_MT2vMETLogi_ratio.GetBinContent(i)>0 else 0)

    hCorr_Gaus_final.Write()
    hCorr_Logi_final.Write()
    hCorr_MT2vMETGaus_ratio.Write("Corr_2D_Gauss")
    hCorr_MT2vMETLogi_ratio.Write("Corr_2D_Logistic")

    # Pull
    #hPull = hCorr_MT2vMETLogi_ratio.Clone("Pull_Logi")
    #hPull.Add(hCorr_Logi_final,-1)
    #for i in xrange(1, hPull.GetNbinsX()+1):
    #    sf = sqrt(hCorr_Logi_final.GetBinError(i)**2 + hCorr_MT2vMETLogi_ratio.GetBinError(i)**2)
    #    print sf
    #    hPull.SetBinContent(i, hPull.GetBinContent(i)/sf)
    #hPull.Write()

    # Get shape stats uncertainty
    f2 = TFile("syst_nJetWgt.root")
    hShapeStat = f2.Get("syst68Max").Clone("shape_stat")

    hAvgWgt = f2.Get("avgWgt").Clone("avgWgt")
    hNEff   = f2.Get("neff").Clone("neff")

    fout.cd()
    hShapeStat.Write()

    # Get PDF and scale uncertainties
    f3 = TFile("syst_scalePDF.root")
    hScaleUp = f3.Get("nSearchBin_ratio_scale_up").Clone("scale_up")
    hScaleDn = f3.Get("nSearchBin_ratio_scale_down").Clone("scale_down")
    hPDFUp = f3.Get("nSearchBin_ratio_pdf_up").Clone("pdf_up")
    hPDFDn = f3.Get("nSearchBin_ratio_pdf_down").Clone("pdf_down")

    fout.cd()
    hScaleUp.Write()
    hScaleDn.Write()
    hPDFUp.Write()
    hPDFDn.Write()

    # Get central, MC stats, closure, trigger, MEU, and JEU
    #f4 = TFile("/uscms/home/nstrobbe/nobackup/HadronicStop/DataTest/CMSSW_7_4_8/src/ZInvisible/Tools/condor/dataplots_muon_Jan24.root")
    f4 = TFile(filename)
    hMC = f4.Get("nSearchBin/NJetWgt_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nusingle")
    hMCstats = hMC.Clone("MC_stats")
    for i in xrange(1, hMCstats.GetNbinsX() + 1):
        if hMC.GetBinContent(i) > 0.00000001:
            hMCstats.SetBinContent(i, hMC.GetBinError(i) / hMC.GetBinContent(i))
        else:
            hMCstats.SetBinContent(i, 0.0)
    fout.cd()
    hMCstats.Write()

    hClosureZ = f4.Get("nSearchBin/nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nusingle")
    hClosureDY = f4.Get("nSearchBin/nSearchBinnSearchBinnSearchBinDY#rightarrow#mu#mu no #mu, Z eff+accsingle")
    hClosureRatio = hClosureZ.Clone("MC_closure")
    hClosureRatio.Add(hClosureDY, -1)
    hClosureRatio.Divide(hClosureZ)
    for i in xrange(1, hClosureRatio.GetNbinsX() + 1):
        hClosureRatio.SetBinContent(i, abs(hClosureRatio.GetBinContent(i)))
    fout.cd()
    hClosureRatio.Write()

    hJECNom = f4.Get("nSearchBin/syst_JESUncert_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu Njet+norm weightsingle")
    hJECUp = f4.Get("nSearchBinJEUUp/syst_JESUncert_nSearchBinnSearchBinJEUUpnSearchBinJEUUpZ#rightarrow#nu#nu JEC Upsingle")
    hJECDn = f4.Get("nSearchBinJEUDn/syst_JESUncert_nSearchBinnSearchBinJEUDnnSearchBinJEUDnZ#rightarrow#nu#nu JEC Downsingle")

    hMECNom = f4.Get("nSearchBin/syst_MESUncert_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu Njet+norm weightsingle")
    hMECUp = f4.Get("nSearchBinMEUUp/syst_MESUncert_nSearchBinnSearchBinMEUUpnSearchBinMEUUpZ#rightarrow#nu#nu MEC Upsingle")
    hMECDn = f4.Get("nSearchBinMEUDn/syst_MESUncert_nSearchBinnSearchBinMEUDnnSearchBinMEUDnZ#rightarrow#nu#nu MEC Downsingle")

    hJECUp_ratio = hJECUp.Clone("jeu_up")
    hJECUp_ratio.Add(hJECNom, -1.0)
    hJECUp_ratio.Divide(hJECNom)

    hJECDn_ratio = hJECDn.Clone("jeu_down")
    hJECDn_ratio.Add(hJECNom, -1.0)
    hJECDn_ratio.Divide(hJECNom)

    hMECUp_ratio = hMECUp.Clone("meu_up")
    hMECUp_ratio.Add(hMECNom, -1.0)
    hMECUp_ratio.Divide(hMECNom)

    hMECDn_ratio = hMECDn.Clone("meu_down")
    hMECDn_ratio.Add(hMECNom, -1.0)
    hMECDn_ratio.Divide(hMECNom)

    hJECUp_ratio.SetBinContent(NSB, hJECUp_ratio.GetBinContent(NSB-1))
    hJECDn_ratio.SetBinContent(NSB, hJECDn_ratio.GetBinContent(NSB-1))
    hMECUp_ratio.SetBinContent(NSB, hMECUp_ratio.GetBinContent(NSB-1))
    hMECDn_ratio.SetBinContent(NSB, hMECDn_ratio.GetBinContent(NSB-1))

    fout.cd()
    hJECUp_ratio.Write()
    hJECDn_ratio.Write()
    hMECUp_ratio.Write()
    hMECDn_ratio.Write()

    hTrigNom = f4.Get("nSearchBin/TriggerWgt_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu Trigger weight Centralsingle")
    hTrigUp =  f4.Get("nSearchBin/TriggerWgt_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu Trigger weight Upsingle")
    hTrigDn =  f4.Get("nSearchBin/TriggerWgt_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu Trigger weight Downsingle")

    hTrigUp_ratio = hTrigUp.Clone("trig_up")
    hTrigUp_ratio.Add(hTrigNom, -1.0)
    hTrigUp_ratio.Divide(hTrigNom)

    hTrigDn_ratio = hTrigDn.Clone("trig_dn")
    hTrigDn_ratio.Add(hTrigNom, -1.0)
    hTrigDn_ratio.Divide(hTrigNom)

    fout.cd()
    hTrigUp_ratio.Write()
    hTrigDn_ratio.Write()

    # b-tag uncertanties

    hBTagNom = f4.Get("nSearchBin/BTagUncert_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu b tag Centralsingle")
    hBTagUp =  f4.Get("nSearchBin/BTagUncert_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu b tag Upsingle")
    hBTagDn =  f4.Get("nSearchBin/BTagUncert_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu b tag Downsingle")
    
    hBMistagNom = f4.Get("nSearchBin/BMistagUncert_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu b tag Centralsingle")
    hBMistagUp =  f4.Get("nSearchBin/BMistagUncert_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu b mistag Upsingle")
    hBMistagDn =  f4.Get("nSearchBin/BMistagUncert_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu b mistag Downsingle")


    hBTagUp_Ratio = hBTagUp.Clone("btag_up")
    hBTagUp_Ratio.Add(hBTagNom, -1.0)
    hBTagUp_Ratio.Divide(hBTagNom)

    hBTagDn_Ratio = hBTagDn.Clone("btag_dn")
    hBTagDn_Ratio.Add(hBTagNom, -1.0)
    hBTagDn_Ratio.Divide(hBTagNom)

    hBMistagUp_Ratio = hBMistagUp.Clone("bmistag_up")
    hBMistagUp_Ratio.Add(hBMistagNom, -1.0)
    hBMistagUp_Ratio.Divide(hBMistagNom)

    hBMistagDn_Ratio = hBMistagDn.Clone("bmistag_dn")
    hBMistagDn_Ratio.Add(hBMistagNom, -1.0)
    hBMistagDn_Ratio.Divide(hBMistagNom)

    fout.cd()
    hBTagUp_Ratio.Write()
    hBTagDn_Ratio.Write()
    hBMistagUp_Ratio.Write()
    hBMistagDn_Ratio.Write()
    #ISR Uncertainties
    #hISRNom = f4.Get("nSearchBin/ISRUncert_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu isr Centralsingle")
    #hBISRUp =  f4.Get("nSearchBin/ISRUncert_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu isr Upsingle")
    #hISRDn =  f4.Get("nSearchBin/ISRUncert_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu isr Downsingle")

    #hISRUp_Ratio = hISRUp.Clone("isr_up")
    #hISRUp_Ratio.Add(hISRNom, -1.0)
    #hBISRUp_Ratio.Divide(hISRNom)

    #hISRDn_Ratio = hISRDn.Clone("isr_dn")
    #hISRDn_Ratio.Add(hISRNom, -1.0)
    #hISRDn_Ratio.Divide(hBISRNom)

    #fout.cd()
    #hISRUp_Ratio.Write()
    #hISRDn_Ratio.Write()

    # calculate stupid pull histo here
    hZnunu = f4.Get("nSearchBin/nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nusingle")
    hDYll  = f4.Get("nSearchBin/nSearchBinnSearchBinnSearchBinDY#rightarrow#mu#mu no #mu, Z eff+accsingle")

    #hMCPull = TH1D("MCpull", "MCpull", 15, -5, 5)
    #hMCPull.Sumw2()
    #hMCPull2 = hZnunu.Clone("MCPull2")
    #for i in xrange(1, hZnunu.GetNbinsX() + 1):
        #pull = (hZnunu.GetBinContent(i) - hDYll.GetBinContent(i)) / math.sqrt(hZnunu.GetBinError(i)**2 + hDYll.GetBinError(i)**2)
        #hMCPull.Fill(pull)
        #hMCPull2.SetBinContent(i, pull)

    #fout.cd()
    #hMCPull.Write()
    #hMCPull2.Write()

    #Central value
    hPrediction = f4.Get("nSearchBin/TriggerWgt_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu Njet+norm weightsingle").Clone("central_prediction")
    fout.cd()
    hPrediction.Write()

    # make proto data card

    hJEC_ratio_sym = hJECUp_ratio.Clone("hJEC_ratio_sym")
    hMEC_ratio_sym = hMECUp_ratio.Clone("hMEC_ratio_sym")
    hScale_sym     = hScaleUp.Clone("hScale_sym")
    hPDF_sym       = hPDFUp.Clone("hPDF_sym")
    hTrig_sym      = hTrigUp_ratio.Clone("hTrig_sym")
    hBTag_sym      = hBTagUp_Ratio.Clone("hBTag_sym")
    hBMistag_sym   = hBMistagUp_Ratio.Clone("hBMistag_sym")

    for i in xrange(1, hJEC_ratio_sym.GetNbinsX() + 1):
        hJEC_ratio_sym.SetBinContent(i, max(abs(hJECUp_ratio.GetBinContent(i)),  abs(hJECDn_ratio.GetBinContent(i))))
        hMEC_ratio_sym.SetBinContent(i, max(abs(hMECUp_ratio.GetBinContent(i)),  abs(hMECDn_ratio.GetBinContent(i))))
        hScale_sym    .SetBinContent(i, max(abs(hScaleUp.GetBinContent(i)),      abs(hScaleDn.GetBinContent(i))))
        hPDF_sym      .SetBinContent(i, max(abs(hPDFUp.GetBinContent(i)),        abs(hPDFDn.GetBinContent(i))))
        hTrig_sym     .SetBinContent(i, max(abs(hTrigUp_ratio.GetBinContent(i)), abs(hTrigDn_ratio.GetBinContent(i))))

    fout.cd()
    hJEC_ratio_sym.Write()
    hMEC_ratio_sym.Write()
    hScale_sym    .Write()
    hPDF_sym      .Write()
    hTrig_sym     .Write()

    hOther = hJEC_ratio_sym.Clone("hOther")
    for i in xrange(1, hOther.GetNbinsX() + 1):
        hOther.SetBinContent(i, sqrt(hJEC_ratio_sym.GetBinContent(i)**2 + hMEC_ratio_sym.GetBinContent(i)**2 + hScale_sym.GetBinContent(i)**2 + hPDF_sym.GetBinContent(i)**2 + hTrig_sym.GetBinContent(i)**2 + hBTag_sym.GetBinContent(i)**2 + hBMistag_sym.GetBinContent(i)**2))
    hOther.Write()

    hists = [("syst_unc_shape_central_up",   hShape_final),
             ("syst_unc_shape_central_dn",   hShape_final),
             ("syst_unc_shape_stat_up",      hShapeStat),
             ("syst_unc_shape_stat_dn",      hShapeStat),
             #("stat_unc_up",                 hMCstats),
             #("stat_unc_dn",                 hMCstats),
             ("syst_unc_jeu_up",             hJEC_ratio_sym),
             ("syst_unc_jeu_dn",             hJEC_ratio_sym),
             ("syst_unc_meu_up",             hMEC_ratio_sym),
             ("syst_unc_meu_dn",             hMEC_ratio_sym),
             ("syst_unc_scale_up",           hScale_sym),
             ("syst_unc_scale_dn",           hScale_sym),
             ("syst_unc_pdf_up",             hPDF_sym),
             ("syst_unc_pdf_dn",             hPDF_sym),
             ("syst_unc_trig_up",            hTrig_sym),
             ("syst_unc_trig_dn",            hTrig_sym),
             ("syst_unc_btag_up",            hBTagUp_Ratio),
             ("syst_unc_btag_dn",            hBTagDn_Ratio),
             ("syst_unc_bmistag_up",         hBMistagUp_Ratio),
             ("syst_unc_bmistag_dn",         hBMistagDn_Ratio),
             #("syst_unc_isr_up",            hISRUp_Ratio),
             #("syst_unc_isr_dn",            hISRDn_Ratio),
             ]

    #sc = SampleCollection()
    #lumi = sc.sampleCollectionLumiList()["Data_SingleMuon"]

    print "luminosity = %f"%(35861.695285122)#12905.3378)
    print "channels =", NSB
    print "sample = zinv"
    print ""

    print "%-25s = %s"%("channel", ' '.join(["%8s" % ("bin%i" % (i-1)) for i in xrange(1, NSB+1)]))
    print ""

    print "%-25s = %s"%("rate", ' '.join(["%8.5f" % hPrediction.GetBinContent(i) for i in xrange(1, NSB+1)]))
    print ""

    print "%-25s = %s"%("cs_event", ' '.join(["%8.0f" % (1.0 if 0< hNEff.GetBinContent(i) <1 else  math.floor(hNEff.GetBinContent(i))) for i in xrange(1, NSB+1)]))
    #print "%-25s = %s"%("cs_event", ' '.join(["%8.3f" % (hNEff.GetBinContent(i)) for i in xrange(1, NSB+1)]))

    data = []
    for i in xrange(1, hNEff.GetNbinsX() + 1):
        if hNEff.GetBinContent(i) >= 1:
            data.append("%8.5f" % (hPrediction.GetBinContent(i)/math.floor(hNEff.GetBinContent(i))))
        else:
            data.append("%8.5f" % hPrediction.GetBinContent(i))
    print "%-25s = %s"%("avg_weight", ' '.join(data))

    print ""
    #print "stat_unc_up = xxx yy zz"
    #print "stat_unc_dn = xxx yy zz"
    #print ""

    #sf = ScaleFactors()

    #print "%-25s = %s"%("syst_unc_norm_up", ' '.join(NSB*["36f" % 0.0830140 ]))#(sf.getRnormErr()/sf.getRnorm()) ]))
    #print "%-25s = %s"%("syst_unc_norm_dn", ' '.join(NSB*["%36f" % 0.0830140 ]))#(sf.getRnormErr()/sf.getRnorm()) ]))

    print "%-25s = %s"%("syst_unc_norm_up", ' '.join(NSB*["%8.5f" % 0.086 ]))#(sf.getRnormErr()/sf.getRnorm()) ]))
    print "%-25s = %s"%("syst_unc_norm_dn", ' '.join(NSB*["%8.5f" % 0.086 ]))#(sf.getRnormErr()/sf.getRnorm()) ]))

    for (name, h) in hists:
        print "%-25s = %s"%(name, ' '.join(["%8.5f" % (abs(h.GetBinContent(i))) for i in xrange(1, NSB+1)]))

    fout.Close()



def systScalePDF(filename):
    # Open file
    f = TFile.Open(filename)

    # Get scale and PDF histograms
    hScale     = f.Get("nSearchBin/syst_ScaleWgt_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu Njet+norm weightsingle")
    hScaleUp   = f.Get("nSearchBin/syst_ScaleWgt_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu Scale weight Upsingle")
    hScaleDown = f.Get("nSearchBin/syst_ScaleWgt_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu Scale weight Downsingle")
    hPDF     = f.Get("nSearchBin/syst_PDFWgt_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu Njet+norm weightsingle")
    hPDFUp   = f.Get("nSearchBin/syst_PDFWgt_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu PDF weight Upsingle")
    hPDFDown = f.Get("nSearchBin/syst_PDFWgt_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu PDF weight Downsingle")

    # Get normalization histograms
    hnScale     = f.Get("cntNJetsPt30Eta24Zinv/DataMCwwscale_SingleMuon_nj_0b_blnotagcntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvZ#rightarrow#nu#nu Njet+norm weightsingle")
    hnScaleUp   = f.Get("cntNJetsPt30Eta24Zinv/DataMCwwscale_SingleMuon_nj_0b_blnotagcntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvZ#rightarrow#nu#nu Scale weight Upsingle")
    hnScaleDown = f.Get("cntNJetsPt30Eta24Zinv/DataMCwwscale_SingleMuon_nj_0b_blnotagcntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvZ#rightarrow#nu#nu Scale weight Downsingle")
    hnPDF     = f.Get("cntNJetsPt30Eta24Zinv/DataMCwwpdf_SingleMuon_nj_0b_blnotagcntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvZ#rightarrow#nu#nu Njet+norm weightsingle")
    hnPDFUp   = f.Get("cntNJetsPt30Eta24Zinv/DataMCwwpdf_SingleMuon_nj_0b_blnotagcntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvZ#rightarrow#nu#nu PDF weight Upsingle")
    hnPDFDown = f.Get("cntNJetsPt30Eta24Zinv/DataMCwwpdf_SingleMuon_nj_0b_blnotagcntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvZ#rightarrow#nu#nu PDF weight Downsingle")

    # Determine scale factors
    sf_Scale     = hnScale.Integral()
    sf_ScaleUp   = hnScaleUp.Integral()
    sf_ScaleDown = hnScaleDown.Integral()
    sf_PDF     = hnPDF.Integral()
    sf_PDFUp   = hnPDFUp.Integral()
    sf_PDFDown = hnPDFDown.Integral()

    # Normalize histograms
    hScale.Scale(1./sf_Scale)
    hScaleUp.Scale(1./sf_ScaleUp)
    hScaleDown.Scale(1./sf_ScaleDown)
    hPDF.Scale(1./sf_PDF)
    hPDFUp.Scale(1./sf_PDFUp)
    hPDFDown.Scale(1./sf_PDFDown)

    # Get the ratios (Up - nominal)/nominal etc
    hratio_ScaleUp = hScaleUp.Clone("nSearchBin_ratio_scale_up")
    hratio_ScaleUp.Add(hScale,-1.0)
    hratio_ScaleUp.Divide(hScale)
    hratio_ScaleDown = hScale.Clone("nSearchBin_ratio_scale_down")
    hratio_ScaleDown.Add(hScaleDown,-1.0)
    hratio_ScaleDown.Divide(hScale)
    hratio_PDFUp = hPDFUp.Clone("nSearchBin_ratio_pdf_up")
    hratio_PDFUp.Add(hPDF,-1.0)
    hratio_PDFUp.Divide(hPDF)
    hratio_PDFDown = hPDF.Clone("nSearchBin_ratio_pdf_down")
    hratio_PDFDown.Add(hPDFDown,-1.0)
    hratio_PDFDown.Divide(hPDF)

    # Write the histograms
    fout = TFile.Open("syst_scalePDF.root","RECREATE")
    fout.cd()

    hScale.Write("nSearchBin_scale_nominal")
    hScaleUp.Write("nSearchBin_scale_up")
    hScaleDown.Write("nSearchBin_scale_down")
    hratio_ScaleUp.Write()
    hratio_ScaleDown.Write()
    hPDF.Write("nSearchBin_pdf_nominal")
    hPDFUp.Write("nSearchBin_pdf_up")
    hPDFDown.Write("nSearchBin_pdf_down")
    hratio_PDFUp.Write()
    hratio_PDFDown.Write()

    # Print some info
    print "Scale uncertainty: "
    for i in range(hratio_ScaleUp.GetNbinsX()):
        print "Bin %s: %.3f, %.3f" % (i, hratio_ScaleUp.GetBinContent(i+1), hratio_ScaleDown.GetBinContent(i+1))
    print "PDF uncertainty: "
    for i in range(hratio_PDFUp.GetNbinsX()):
        print "Bin %s: %.3f, %.3f" % (i, hratio_PDFUp.GetBinContent(i+1), hratio_PDFDown.GetBinContent(i+1))

    fout.Close()
    f.Close()

def extrapolationSyst(filename):
    f = TFile.Open(filename)
    print "Opened file", filename, f

    dataName = "%(varName)s/SystPlots_DataMCw_SingleMuon_%(varLabel)s_%(cutStr)s%(varName)s%(varName)sDatadata"
    baseName = "%(varName)s/SystPlots_DataMCw_SingleMuon_%(varLabel)s_%(cutStr)s%(varName)s%(varName)s%(sample)s"

    varDict = {"met":("cleanMetPt", [0, 100, 250, 300, 450, 2000]),#[0, 100, 250, 350, 550, 2000]),
               "ht":("HTZinv", [0, 150, 300, 450, 600, 2000]),
               "nt":("nTopCandSortedCntZinv", None),
               "mt2":("best_had_brJet_MT2Zinv", [0, 100, 200, 300, 400, 2000]),
               "nb":("cntCSVSZinv", None),
               "nj":("cntNJetsPt30Eta24Zinv", None),
               #"jpt":"",
               #"mupt":"",
               "nSearchBin":("nSearchBin", None)}

    cutLists = {"nb":[(False,  "muZinv_0b_loose0_mt2_MET"),
                      (True,   "muZinv_1b_loose0_mt2_MET"),
                      (True,   "muZinv_2b_loose0_mt2_MET"),
                      (True,   "muZinv_3b_loose0_mt2_MET")],
                "nt":[#(False,  "muZinv_0t_loose0_mt2_MET"),
                      (True,   "muZinv_1t_loose0_mt2_MET"),
                      (True,   "muZinv_2t_loose0_mt2_MET")],
                "met":[#(False, "muZinv_met_0_100_loose0_mt2_MET"),
                       (False, "muZinv_met_100_200_loose0_mt2_MET"),
                       (True,  "muZinv_met_200_300_loose0_mt2_MET"),
                       (True,  "muZinv_met_300_400_loose0_mt2_MET"),
                       (True,  "muZinv_met_gt400_loose0_mt2_MET")],
                "mt2":[(False, "muZinv_mt2_0_100_loose0_mt2_MET"),
                       (False, "muZinv_mt2_100_200_loose0_mt2_MET"),
                       (True,  "muZinv_mt2_200_300_loose0_mt2_MET"),
                       (True,  "muZinv_mt2_300_400_loose0_mt2_MET"),
                       (True,  "muZinv_mt2_gt400_loose0_mt2_MET")],
                "ht":[#(False,  "muZinv_ht_200_300_loose0_mt2_MET"),
                      (True,  "muZinv_ht_300_400_loose0_mt2_MET"),
                      (True,  "muZinv_ht_400_500_loose0_mt2_MET"),
                      (True,   "muZinv_ht_gt500_loose0_mt2_MET")]
                }

    samples = ["DYstack",
               #"DY HT<100stack",
               "t#bar{t}stack",
               "Single topstack",
               "t#bar{t}Zstack",
               "Dibosonstack",
               "Rarestack"
               ]

    fout = TFile.Open("looseToTight.root", "RECREATE")

    for varLabel, (varName, theBins) in varDict.iteritems():
        print "Processing variable: " , varName
        hDataTotal = {}
        hBGTotal = {}
        hDataTight = {}
        hBGTight = {}
        for cutNames, cutStrs in cutLists.iteritems():
            tmpDataTotal = []
            tmpBGTotal = []
            tmpDataTight = []
            tmpBGTight = []
            print "VarName: ", varName, "  CutName: ", cutNames
            for (isTight, cutStr) in cutStrs:
                hData = f.Get(dataName%locals())
                tmpDataTotal.append(hData)
                if isTight:
                    tmpDataTight.append(hData)
                hBGs = []
                hBGsTight = []
                for sample in samples:
                    hBGs.append(f.Get(baseName%locals()))
                    if isTight:
                        hBGsTight.append(f.Get(baseName%locals()))
                tmpBGTotal.append(add(hBGs))
                if len(hBGsTight):
                    tmpBGTight.append(add(hBGsTight))
            hDataTotal[varLabel+"_cut_"+cutNames] = add(tmpDataTotal)
            hBGTotal[varLabel+"_cut_"+cutNames] = add(tmpBGTotal)
            hDataTight[varLabel+"_cut_"+cutNames] = add(tmpDataTight)
            hBGTight[varLabel+"_cut_"+cutNames] = add(tmpBGTight)

        # The first loop sank into the swamp, so we built it again!
        # but no really my boy, we had to calculate the total histograms from the individual
        # slices first, then make the float ratios here
        # Yes, I know, this is stupidly inefficient
        for cutNames, cutStrs in cutLists.iteritems():
            hRatioTotal = makeRatio(hDataTotal[varLabel+"_cut_"+cutNames], [hBGTotal[varLabel+"_cut_"+cutNames]], bins=theBins, newname="".join(["totalRatio_", varLabel, "_cut_", cutNames]))
            hRatioTotal.Write()
            for (isTight, cutStr) in cutStrs:
                hData = f.Get(dataName%locals())
                hBGs = []
                for sample in samples:
                    hBGs.append(f.Get(baseName%locals()))
                hRatio = makeRatio(hData, hBGs, bins=theBins, newname="_".join(["BoringRatio", varLabel, cutStr]))
                hRatio.Write()
                hDoubleRatio = makeRatio(hRatio, [hRatioTotal], newname="_".join(["DoubleRatio", varLabel, cutStr]))
                hDoubleRatio.Write()
            hTightRatio = makeRatio(hDataTight[varLabel+"_cut_"+cutNames], [hBGTight[varLabel+"_cut_"+cutNames]], bins=theBins, newname="_".join(["tightRatio", varLabel, "cut", cutNames]))
            hTightRatio.Write()
            hDoubleRatioTL = makeRatio(hTightRatio, [hRatioTotal], newname="_".join(["DoubleRatioTight", varLabel, "cut", cutNames]))
            hDoubleRatioTL.Write()
    f.Close()
    fout.Close()


def calcBeffs(filePath):

    fileNames = ["bTagEfficiency_TTbarSingleLepT.root",
                 "bTagEfficiency_TTbarDiLep.root",
                 "bTagEfficiency_TTbarSingleLepTbar.root",
                 "bTagEfficiency_DYJetsToLL_HT_100to200.root",
                 "bTagEfficiency_DYJetsToLL_HT_200to400.root",
                 "bTagEfficiency_DYJetsToLL_HT_400to600.root",
                 "bTagEfficiency_DYJetsToLL_HT_600toInf.root",
                 "bTagEfficiency_ZJetsToNuNu_HT_600toInf.root",
                 "bTagEfficiency_ZJetsToNuNu_HT_400to600.root",
                 "bTagEfficiency_ZJetsToNuNu_HT_200to400.root",
                 "bTagEfficiency_ZJetsToNuNu_HT_100to200.root",
                 "bTagEfficiency_ttHJetTobb.root",
                 "bTagEfficiency_ttHJetToNonbb.root",
                 "bTagEfficiency_tW_top.root",
                 "bTagEfficiency_tW_antitop.root",
                 "bTagEfficiency_ZZ.root",
                 "bTagEfficiency_WZ.root",
                 "bTagEfficiency_WW.root",
                 "bTagEfficiency_WWZ.root",
                 "bTagEfficiency_WZZ.root",
                 "bTagEfficiency_ZZZ.root",
                 "bTagEfficiency_TTZToQQ.root",
                 "bTagEfficiency_TTZToLLNuNu.root",
                 "bTagEfficiency_TTWJetsToQQ.root",
                 "bTagEfficiency_TTWJetsToLNu.root",
                 "bTagEfficiency_TTGJets.root",
                 "bTagEfficiency_DYJetsToLL_Inc.root"]

    fout = TFile.Open("bTagEffHists.root", "RECREATE")

    for filename in fileNames:

        f = TFile.Open(filePath + filename)

        if not f is None:

            n_eff_b = f.Get("h_eff_b")
            n_eff_c = f.Get("h_eff_c")
            n_eff_udsg = f.Get("h_eff_udsg")
            d_eff_b = f.Get("d_eff_b")
            d_eff_c = f.Get("d_eff_c")
            d_eff_udsg = f.Get("d_eff_udsg")

            n_eff_b.Sumw2()
            n_eff_c.Sumw2()
            n_eff_udsg.Sumw2()
            d_eff_b.Sumw2()
            d_eff_c.Sumw2()
            d_eff_udsg.Sumw2()

            dataset = filename.replace(".root", "").replace("bTagEfficiency_", "")
            eff_b = n_eff_b.Clone("eff_b_" + dataset)
            eff_c = n_eff_c.Clone("eff_c_" + dataset)
            eff_udsg = n_eff_udsg.Clone("eff_udsg_" + dataset)

            eff_b.Divide(d_eff_b)
            eff_c.Divide(d_eff_c)
            eff_udsg.Divide(d_eff_udsg)

            fout.cd()
            eff_b.Write()
            eff_c.Write()
            eff_udsg.Write()

        f.Close()

    fout.Close()



if __name__ ==  "__main__":

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename",
                      help="Grab histogram from FILE", metavar="FILE")
    parser.add_option("--norm", dest="normweight", default=False,
                      help="Compute the normalization weight", action='store_true')
    parser.add_option("--njet", dest="njetweight", default=False,
                      help="Compute the njet weights", action='store_true')
    parser.add_option("--shape", dest="shape", default=False,
                      help="Compute the shape systematics", action='store_true')
    parser.add_option("--systHarvest", dest="systHarvest", default=False,
                      help="Harvest systematics", action='store_true')
    parser.add_option("--systScale", dest="systScale", default=False,
                      help="Grab information for scale and PDF systematics", action='store_true')
    parser.add_option("--extrapolationSyst", dest="extrapolationSyst", default=False,
                      help="Grab information for scale and PDF systematics", action='store_true')
    parser.add_option("--corr", dest="corr", default=False,
                      help="Get the ratios for correlation study", action='store_true')
    parser.add_option("--btag", dest="btag", default=False,
                      help="Calculate the b-tag efficiencies.  Treats -f input as a path, not a file.", action='store_true')

    (options, args) = parser.parse_args()

    if options.njetweight:
        njetWeights(options.filename)
    if options.normweight:
        normWeight(options.filename)
    if options.shape:
        shapeSyst(options.filename)
    if options.systHarvest:
        systHarvest(options.filename)
    if options.systScale:
        systScalePDF(options.filename)
    if options.extrapolationSyst:
        extrapolationSyst(options.filename)
    if options.corr:
        corrCheck()
    if options.btag:
        calcBeffs(options.filename)

