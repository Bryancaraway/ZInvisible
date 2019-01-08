#ifndef GENERATEPHOTONEFFICIENCY_H
#define GENERATEPHOTONEFFICIENCY_H

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
    class GeneratePhotonEfficiency
    {
    private:
        TH1F* hCrossSectionRatio;
        TH1F* hPhotonAccPt_num;
        TH1F* hPhotonAccPt_den;
        TH1F* hPhotonEffPt_num;
        TH1F* hPhotonEffPt_den;

        void generatePhotonEfficiency(NTupleReader& tr)
        {
            const auto& photonLVecPassLooseID = tr.getVec<TLorentzVector>("photonLVecPassLooseID"); // loose photon
            const auto& passPhotonSelection  = tr.getVar<bool>("passPhotonSelection");            // photon selection
            data_t photonCrossSectionRatio = -1.0;
            data_t photonAcceptance = -1.0;
            data_t photonEfficiencyPt = -1.0;
            data_t photonPt = -1.0;
            photonCrossSectionRatio = hCrossSectionRatio->GetBinContent(1);
            if (passPhotonSelection)
            {
                if (photonLVecPassLooseID.size() == 1)
                {
                    photonPt = photonLVecPassLooseID[0].Pt();
                }
                else
                {
                    std::cout << "ERROR: passPhotonSelection = true, but photonLVecPassLooseID.size() = " << photonLVecPassLooseID.size() << " (should be 1)" << std::endl;
                }
            }
            if (photonPt > 0)
            {
                // get photon acceptance
                photonAcceptance = hPhotonAccPt_num->Integral() / hPhotonAccPt_den->Integral();
                // get photon efficiency
                TH1F* hPhotonEffPt_ratio = (TH1F*)hPhotonEffPt_num->Clone();
                hPhotonEffPt_ratio->Divide(hPhotonEffPt_den);
                int bin_n = hPhotonEffPt_ratio->GetXaxis()->FindBin(photonPt);
                photonEfficiencyPt = hPhotonEffPt_ratio->GetBinContent(bin_n); 
            }
            else
            {
                // no photons passing selection
                photonAcceptance = 1.0;
                photonEfficiencyPt = 1.0;
            }
            data_t photonAcceptanceWeight   = 1.0 / photonAcceptance;
            data_t photonEfficiencyPtWeight = 1.0 / photonEfficiencyPt;
            
            // if (photonPt > 0)
            // {
            //     std::cout << "photonAcceptance = "   << photonAcceptance   << " photonAcceptanceWeight = "   << photonAcceptanceWeight << std::endl;
            //     std::cout << "photonEfficiencyPt = " << photonEfficiencyPt << " photonEfficiencyPtWeight = " << photonEfficiencyPtWeight << std::endl;
            // }
            
            tr.registerDerivedVar("photonCrossSectionRatio",  photonCrossSectionRatio);
            tr.registerDerivedVar("photonAcceptance",         photonAcceptance);
            tr.registerDerivedVar("photonAcceptanceWeight",   photonAcceptanceWeight);
            tr.registerDerivedVar("photonEfficiencyPt",       photonEfficiencyPt);
            tr.registerDerivedVar("photonEfficiencyPtWeight", photonEfficiencyPtWeight);
        }
    
    public:
        GeneratePhotonEfficiency()
        {
            hCrossSectionRatio = nullptr;
            hPhotonAccPt_num = nullptr;
            hPhotonAccPt_den = nullptr;
            hPhotonEffPt_num = nullptr;
            hPhotonEffPt_den = nullptr;
            TH1::AddDirectory(false);
            std::string histFile = "effhists_GJets.root";
            TFile *f = new TFile(histFile.c_str());
            if(f)
            {
                hCrossSectionRatio = static_cast<TH1F*>(f->Get("hCrossSectionRatio"));
                hPhotonAccPt_num = static_cast<TH1F*>(f->Get("hPhotonAccPt_num"));
                hPhotonAccPt_den = static_cast<TH1F*>(f->Get("hPhotonAccPt_den"));
                hPhotonEffPt_num = static_cast<TH1F*>(f->Get("hPhotonEffPt_num"));
                hPhotonEffPt_den = static_cast<TH1F*>(f->Get("hPhotonEffPt_den"));
                f->Close();
                delete f;
            }
            else
            {
                std::cout << "Failed to open the file " << histFile << std::endl;
            }
        }
        ~GeneratePhotonEfficiency() {}

        void operator()(NTupleReader& tr)
        {
            generatePhotonEfficiency(tr);
        }
    };
}

#endif
