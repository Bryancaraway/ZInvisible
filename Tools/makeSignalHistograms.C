#include "NTupleReader.h"
#include "samples.h"
#include "baselineDef.h"
//#include "../../searchBins.h"
#include "derivedTupleVariables.h"

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <getopt.h>

#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TRandom3.h"
#include "TChain.h"

class HistContainer
{
private:
    std::map<std::string, TH1*> hists_;

    int mMass_;
    int dMass_;
    
    void makeHist(const std::string name, int N, float ll, float ul)
        {
            char hname[128];
            if(mMass_ >= 0 && dMass_ >= 0) sprintf(hname, "Sig_%d_%d_%s", mMass_, dMass_, name.c_str());
            else                         sprintf(hname, "%s", name.c_str());
            hists_[name] = new TH1D(hname, hname, N, ll, ul);
        }

    void bookHists()
        {
            makeHist("met",             150, 0, 3000);
            makeHist("mt2",             150, 0, 3000);
            makeHist("nt",               10, 0,   10);
            makeHist("nb",               10, 0,   10);
            makeHist("nj",               20, 0,   20);
            makeHist("loose0_met",      150, 0, 3000);
            makeHist("loose0_mt2",      150, 0, 3000);
            makeHist("loose0_nt",        10, 0,   10);
            makeHist("loose0_nb",        10, 0,   10);
            makeHist("loose0_nj",        20, 0,   20);
            makeHist("loose0Nt_met",    150, 0, 3000);
            makeHist("loose0Nt_mt2",    150, 0, 3000);
            makeHist("loose0Nt_nt",      10, 0,   10);
            makeHist("loose0Nt_nb",      10, 0,   10);
            makeHist("loose0Nt_nj",      20, 0,   20);
            makeHist("baselineNob_met", 150, 0, 3000);
            makeHist("baselineNob_mt2", 150, 0, 3000);
            makeHist("baselineNob_nt",   10, 0,   10);
            makeHist("baselineNob_nb",   10, 0,   10);
            makeHist("baselineNob_nj",   20, 0,   20);
        }
    
public:
    HistContainer(int mMass, int dMass) : mMass_(mMass), dMass_(dMass) 
        {
            bookHists();
        }

    void fill(const NTupleReader& tr, const float weight)
        {
            const float& met                 = tr.getVar<float>("cleanMetPt");
            const float& best_had_brJet_MT2  = tr.getVar<float>("best_had_brJet_MT2Zinv");
            const float& HTZinv              = tr.getVar<float>("HTZinv");
            const int& cntCSVS                = tr.getVar<int>("cntCSVSZinv");
            const int& nTopCandSortedCnt      = tr.getVar<int>("nTopCandSortedCntZinv");
            const int& cntNJetsPt30Eta24Zinv  = tr.getVar<int>("cntNJetsPt30Eta24Zinv");
            
            const bool& passBaselineNoTagZinv = tr.getVar<bool>("passBaselineNoTagZinv");
            const bool& passNoiseEventFilterZinv = tr.getVar<bool>("passNoiseEventFilterZinv");
            const bool& passnJetsZinv = tr.getVar<bool>("passnJetsZinv");
            const bool& passdPhisZinv = tr.getVar<bool>("passdPhisZinv");
            const bool& passMuZinvSel = tr.getVar<bool>("passMuZinvSel");

            bool passLoose0 = passNoiseEventFilterZinv && passMuZinvSel && (HTZinv > 200) && passnJetsZinv && passdPhisZinv;
            bool passLoose0Nt = passLoose0 && (nTopCandSortedCnt >= 1);
            bool passBaselineNob = passBaselineNoTagZinv && passMuZinvSel && (nTopCandSortedCnt >= 1);

            hists_["met"]->Fill(met, weight);
            hists_["mt2"]->Fill(best_had_brJet_MT2, weight);
            hists_["nt"]->Fill(nTopCandSortedCnt, weight);
            hists_["nb"]->Fill(cntCSVS, weight);
            hists_["nj"]->Fill(cntNJetsPt30Eta24Zinv, weight);

            if(passBaselineNob)
            {
                hists_["baselineNob_met"]->Fill(met, weight);
                hists_["baselineNob_mt2"]->Fill(best_had_brJet_MT2, weight);
                hists_["baselineNob_nt"]->Fill(nTopCandSortedCnt, weight);
                hists_["baselineNob_nb"]->Fill(cntCSVS, weight);
                hists_["baselineNob_nj"]->Fill(cntNJetsPt30Eta24Zinv, weight);
            }

            if(passLoose0)
            {
                hists_["loose0_met"]->Fill(met, weight);
                hists_["loose0_mt2"]->Fill(best_had_brJet_MT2, weight);
                hists_["loose0_nt"]->Fill(nTopCandSortedCnt, weight);
                hists_["loose0_nb"]->Fill(cntCSVS, weight);
                hists_["loose0_nj"]->Fill(cntNJetsPt30Eta24Zinv, weight);
            }

            if(passLoose0Nt)
            {
                hists_["loose0Nt_met"]->Fill(met, weight);
                hists_["loose0Nt_mt2"]->Fill(best_had_brJet_MT2, weight);
                hists_["loose0Nt_nt"]->Fill(nTopCandSortedCnt, weight);
                hists_["loose0Nt_nb"]->Fill(cntCSVS, weight);
                hists_["loose0Nt_nj"]->Fill(cntNJetsPt30Eta24Zinv, weight);
            }
        }

    void write()
        {
            for(auto& hist : hists_) hist.second->Write();
        }
};

//void calcSearchBin(NTupleReader& tr)
//{
//    const float& met                = tr.getVar<float>("met");
//    const float& best_had_brJet_MT2 = tr.getVar<float>("best_had_brJet_MT2");
//    const int& cntCSVS               = tr.getVar<int>("cntCSVS");
//    const int& nTopCandSortedCnt     = tr.getVar<int>("nTopCandSortedCnt");
//
//    int nSearchBin = find_Binning_Index(cntCSVS, nTopCandSortedCnt, best_had_brJet_MT2, met);
//
//    tr.registerDerivedVar("nSearchBin", nSearchBin);
//}

int main(int argc, char* argv[])
{
    TH1::AddDirectory(false);

    int opt;
    int option_index = 0;
    static struct option long_options[] = {
        {"condor",           no_argument, 0, 'c'},
        {"dataSets",   required_argument, 0, 'D'},
        {"numFiles",   required_argument, 0, 'N'},
        {"startFile",  required_argument, 0, 'M'},
        {"numEvts",    required_argument, 0, 'E'},
    };

    bool runOnCondor = false;
    int nFiles = -1, startFile = 0, nEvts = -1;
    std::string dataSets = "";

    while((opt = getopt_long(argc, argv, "cD:N:M:E:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
        case 'c':
            runOnCondor = true;
            break;

        case 'D':
            dataSets = optarg;
            break;

        case 'N':
            nFiles = int(atoi(optarg));
            break;

        case 'M':
            startFile = int(atoi(optarg));
            break;

        case 'E':
            nEvts = int(atoi(optarg));
            break;
        }
    }

    std::string filename = "effhists.root";
    std::string sampleloc = AnaSamples::fileDir;

    //if running on condor override all optional settings
    if(runOnCondor)
    {
        char thistFile[128];
        sprintf(thistFile, "effhists_%s_%d.root", dataSets.c_str(), startFile);
        filename = thistFile;
        sampleloc = "condor";
    }

    // the old version
    //AnaSamples::SampleSet        ss(sampleloc, lumi);
    //AnaSamples::SampleCollection sc(ss);
    // the new version
    AnaSamples::SampleSet        ss("sampleSets.txt");
    AnaSamples::SampleCollection sc("sampleCollections.txt", ss);

    //topTagger::type3TopTagger* type3Ptr;

    try
    {
        //AnaFunctions::prepareTopTagger(type3Ptr);

        std::set<std::string> activatedBranch;
        for(auto& branch : AnaConsts::activatedBranchNames_DataOnly) activatedBranch.insert(branch);
        for(auto& branch : AnaConsts::activatedBranchNames) activatedBranch.insert(branch);
        activatedBranch.insert("muonsCharge");
        activatedBranch.insert("elesCharge");
        activatedBranch.insert("SusyMotherMass");
        activatedBranch.insert("SusyLSPMass");

        std::map<std::pair<int, int>, HistContainer> histVec;

        //for(auto& fs : sc["Signal_fastsim_T2tt_scan"])
        auto& fs = ss[dataSets];
        {
            TChain * t = new TChain(fs.treePath.c_str());

            std::cout << "Processing file(s): " << fs.filePath << std::endl;

            fs.addFilesToChain(t, startFile, nFiles);

            std::string fastsim = "";
            if(fs.filePath.find("SMS") != std::string::npos) fastsim = "fastsim";
            BaselineVessel blv(*static_cast<NTupleReader*>(nullptr), "", fastsim);
            BaselineVessel blvZinv(*static_cast<NTupleReader*>(nullptr), "Zinv", fastsim);
            plotterFunctions::LepInfo lepInfo;

            NTupleReader tr(t, activatedBranch);
            tr.setReThrow(false);
            tr.registerFunction(blv);
            tr.registerFunction(lepInfo);
            tr.registerFunction(blvZinv);
            //tr.registerFunction(&calcSearchBin);

            while(tr.getNextEvent())
            {
                if(tr.getEvtNum() % 10000 == 0) std::cout << "Event #: " << tr.getEvtNum() << std::endl;

                const float& SusyMotherMass_ref  = tr.getVar<float>("SusyMotherMass");
                const float& SusyLSPMass_ref     = tr.getVar<float>("SusyLSPMass");

                float SusyMotherMass = -999;
                float SusyLSPMass = -999;

                if(&SusyMotherMass_ref != nullptr && &SusyLSPMass_ref != nullptr)
                {
                    SusyMotherMass = SusyMotherMass_ref;
                    SusyLSPMass = SusyLSPMass_ref;
                }

                std::pair<int, int> iMP((int)SusyMotherMass, (int)SusyLSPMass);

                auto iter = histVec.find(iMP);
                if(iter == histVec.end()) iter = histVec.emplace(iMP, HistContainer(iMP.first, iMP.second)).first;

                iter->second.fill(tr, fs.getWeight());
            }
        }

        TFile *f = TFile::Open(("signalHists_" + dataSets + "_" + std::to_string(startFile) + ".root").c_str(), "RECREATE");
        f->cd();
        for(auto& p : histVec) p.second.write();
        f->Close();
    }
    catch(const std::string e)
    {
        std::cout << e << std::endl;
    }
}
