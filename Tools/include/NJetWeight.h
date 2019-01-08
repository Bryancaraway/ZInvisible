#ifndef NJETWEIGHT
#define NJETWEIGHT

#include "TypeDefinitions.h"
#include "PhotonTools.h"

#include "SusyAnaTools/Tools/NTupleReader.h"
#include "SusyAnaTools/Tools/customize.h"
#include "SusyAnaTools/Tools/searchBins.h"
#include "TopTagger/Tools/cpp/TaggerUtility.h"
#include "TopTagger/Tools/cpp/PlotUtility.h"
#include "ScaleFactors.h"
#include "ScaleFactorsttBar.h"

#include "TopTagger.h"
#include "TTModule.h"
#include "TopTaggerUtilities.h"
#include "TopTaggerResults.h"
#include "TopTagger/Tools/cpp/PlotUtility.h"

#include "TopTagger/TopTagger/interface/TopObject.h"

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TMath.h"
#include "TLorentzVector.h"
#include "Math/VectorUtil.h"
#include "TRandom3.h"
#include "TVector2.h"

#include <vector>
#include <iostream>
#include <string>
#include <set>

namespace plotterFunctions
{
    class NJetWeight
    {
    private:
        TH1* njWTTbar_0b;
        TH1* njWDYZ_0b;
        TH1* njWTTbar_g1b;
        TH1* njWDYZ_g1b;

        TH1* njWGJets;
        TH1* njWGJetsNorm;
        TH1* njWGJets_all;

        TH1* MCfake1b;
        TH1* MCfake2b;
        TH1* MCfake3b;

        void generateWeight(NTupleReader& tr)
        {
            const auto& cntNJetsPt30Eta24Zinv   = tr.getVar<int>("cntNJetsPt30Eta24Zinv");
            //const auto& nJets                 =  tr.getVar<int>("nJets"); // variable not produced in CMSSW8028_2016 ntuples
            const auto& nJets                   =  tr.getVar<int>("cntNJetsPt30Eta24");
            const auto& cntCSVSZinv             = tr.getVar<int>("cntCSVSZinv");

            data_t wTT = 1.0;
            data_t wDY = 1.0;
            data_t wGJets = 1.0;
            data_t wGJetsNorm = 1.0;
            data_t wGJets_all = 1.0;

            if(cntCSVSZinv == 0)
              {
                if(njWTTbar_0b)  wTT = 1.0;//njWTTbar_0b->GetBinContent(njWTTbar_0b->FindBin(cntNJetsPt30Eta24Zinv));
                if(njWDYZ_0b)    wDY = njWDYZ_0b->GetBinContent(njWDYZ_0b->FindBin(cntNJetsPt30Eta24Zinv));
              }
            else
              {
                if(njWTTbar_g1b) wTT = 1.0;//njWTTbar_g1b->GetBinContent(njWTTbar_g1b->FindBin(cntNJetsPt30Eta24Zinv));
                if(njWDYZ_g1b)   wDY = njWDYZ_g1b->GetBinContent(njWDYZ_g1b->FindBin(cntNJetsPt30Eta24Zinv));
              }
            
            if(njWGJets) {
              wGJets = njWGJets->GetBinContent(njWGJets->FindBin(cntNJetsPt30Eta24Zinv));
              wGJetsNorm = njWGJetsNorm->GetBinContent(njWGJetsNorm->FindBin(cntNJetsPt30Eta24Zinv));
            }

            if(njWGJets_all) wGJets_all = njWGJets_all->GetBinContent(njWGJets_all->FindBin(nJets));
            //wGJets = njWGJets->GetBinContent(njWGJets->FindBin(nJets)); 
            //std::cout<<wGJets<<std::endl;
            data_t nJet1bfakeWgt = 1.0;
            data_t nJet2bfakeWgt = 1.0;
            data_t nJet3bfakeWgt = 1.0;

            if(MCfake1b)   nJet1bfakeWgt = MCfake1b->GetBinContent(MCfake1b->FindBin(cntNJetsPt30Eta24Zinv));
            if(MCfake2b)   nJet2bfakeWgt = MCfake2b->GetBinContent(MCfake2b->FindBin(cntNJetsPt30Eta24Zinv));
            if(MCfake3b)   nJet3bfakeWgt = MCfake3b->GetBinContent(MCfake3b->FindBin(cntNJetsPt30Eta24Zinv));

            data_t normWgt0b = ScaleFactors::sf_norm0b();
            data_t normttbar = ScaleFactorsttBar::sf_norm0b(); 


            tr.registerDerivedVar("nJetWgtTTbar", wTT);
            tr.registerDerivedVar("nJetWgtDYZ",   wDY);

            tr.registerDerivedVar("njWGJets",   wGJets);
            tr.registerDerivedVar("njWGJetsNorm",   wGJetsNorm);
            tr.registerDerivedVar("njWGJets_all",   wGJets_all);

            tr.registerDerivedVar("nJet1bfakeWgt", nJet1bfakeWgt);
            tr.registerDerivedVar("nJet2bfakeWgt", nJet2bfakeWgt);
            tr.registerDerivedVar("nJet3bfakeWgt", nJet3bfakeWgt);

            tr.registerDerivedVar("normWgt0b", normWgt0b);
            tr.registerDerivedVar("normttbar", normttbar);
        }

    public:
        NJetWeight()
        {
            TH1::AddDirectory(false);

            njWTTbar_0b  = nullptr;
            njWDYZ_0b    = nullptr;
            njWTTbar_g1b = nullptr;
            njWDYZ_g1b   = nullptr;
            njWGJets = nullptr;
            njWGJets_all = nullptr;
            MCfake1b = nullptr;
            MCfake2b = nullptr;
            MCfake3b = nullptr;

            TFile *f = new TFile("njetWgtHists.root");
            if(f)
            {
                MCfake1b = static_cast<TH1*>(f->Get("h_njRatio_1fake"));
                MCfake2b = static_cast<TH1*>(f->Get("h_njRatio_2fake"));
                MCfake3b = static_cast<TH1*>(f->Get("h_njRatio_3fake"));
                f->Close();
                delete f;
            }
            else
            {
                std::cout << "Failed to open: njetWgtHists.root" << std::endl;
            }

            f = new TFile("dataMCweights.root");
            if(f)
            {
                //njWTTbar_0b  = 1;//static_cast<TH1*>(f->Get("DataMC_nj_elmuZinv_loose0"));
                njWDYZ_0b    = static_cast<TH1*>(f->Get("DataMC_nj_muZinv_0b_loose0_mt2_MET"));//0b_loose0"));
                //njWTTbar_g1b = 1;//static_cast<TH1*>(f->Get("DataMC_nj_elmuZinv_loose0"));
                njWDYZ_g1b   = static_cast<TH1*>(f->Get("DataMC_nj_muZinv_g1b_loose0_mt2_MET"));//g1b_loose0"));
                f->Close();
                delete f;
            }
            else
            {
                std::cout << "Failed to open: dataMCweights.root" << std::endl;
            }
            f = new TFile("dataMCreweight.root");
            if(f)
              {
                njWGJets = static_cast<TH1*>(f->Get("nJetReweight"));
                njWGJetsNorm = static_cast<TH1*>(f->Get("normWeight"));
                f->Close();
                delete f;
              }
            else
              {
                std::cout << "Failed to open: dataMCreweight.root" << std::endl;
              }
            f = new TFile("dataMCreweight_allJets.root");
            if(f)
              {
                njWGJets_all = static_cast<TH1*>(f->Get("dataMC_Photon_nj_LooseLepVetonJetsnJetsDatadata"));
                f->Close();
                delete f;
              }
            else
              {
                std::cout << "Failed to open: dataMCreweight_allJets.root" << std::endl;
              }
        }

        ~NJetWeight()
        {
            //if(njWTTbar) delete njWTTbar;
            //if(njWDYZ)   delete njWDYZ;
        }

        void operator()(NTupleReader& tr)
        {
            generateWeight(tr);
        }
    };
}

#endif
