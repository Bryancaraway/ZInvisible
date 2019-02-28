#ifndef BASICLEPTON_H 
#define BASICLEPTON_H 

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
    class BasicLepton
    {
    private:
        void basicLepton(NTupleReader& tr)
        {
            const auto& muonsLVec                           = tr.getVec<TLorentzVector>("MuonTLV");
            const auto& muonsRelIso                         = tr.getVec<data_t>("Muon_miniPFRelIso_all");
            const auto& muonsMiniIso                        = tr.getVec<char>("Muon_miniIsoId");
            const auto& muonsCharge                         = tr.getVec<int>("Muon_charge");
            const auto& muonsJetIndex                       = tr.getVec<int>("Muon_jetIdx");
            //const auto& muonspfActivity                     = tr.getVec<data_t>("muonspfActivity");
            // Muon_mediumId
            //const auto& muonsFlagIDVec                      = tr.getVec<bool_t>("Muon_Stop0l");
            const auto& muonsFlagIDVec                      = tr.getVec<bool_t>("Muon_mediumId");

            const auto& elesLVec                            = tr.getVec<TLorentzVector>("ElectronTLV");
            const auto& elesMiniIso                         = tr.getVec<data_t>("Electron_miniPFRelIso_all");
            const auto& elesCharge                          = tr.getVec<int>("Electron_charge");
            const auto& elesJetIndex                        = tr.getVec<int>("Electron_jetIdx");
            //const auto& elesisEB                            = tr.getVec<unsigned int>("elesisEB");
            //const auto& elespfActivity                      = tr.getVec<data_t>("elespfActivity");
            //const auto& elesFlagIDVec                       = tr.getVec<bool_t>("Electron_Stop0l");
            const auto& elesFlagIDVec                       = tr.getVec<int>("Electron_cutBasedNoIso");

            //muons
            auto* cutMuVec                  = new std::vector<TLorentzVector>();
            auto* cutMuVecRecoOnly          = new std::vector<TLorentzVector>();
            auto* cutMuCharge               = new std::vector<int>();
            auto* cutMuJetIndex             = new std::vector<int>();
            //auto* cutMuActivity             = new std::vector<data_t>();
            
            //electrons
            auto* cutElecVec                = new std::vector<TLorentzVector>();
            auto* cutElecVecRecoOnly        = new std::vector<TLorentzVector>();
            auto* cutElecCharge             = new std::vector<int>();
            auto* cutElecJetIndex           = new std::vector<int>();
            //auto* cutElecActivity           = new std::vector<data_t>();

            //std::vector<TLorentzVector> cutMuVecRecoOnly;
            //std::vector<TLorentzVector> cutElecVecRecoOnly;
            
            //std::vector<TLorentzVector>* Zrecopt = new std::vector<TLorentzVector>();

            //muon selections
            int cutMuSummedCharge = 0;
            int nTriggerMuons = 0;

            for(int i = 0; i < muonsLVec.size(); ++i)
            {
                //if(muonsFlagIDVec[i])
                if(AnaFunctions::passMuon( muonsLVec[i], 0.0, 0.0, muonsFlagIDVec[i], AnaConsts::muonsMiniIsoArr)) // emulates muons with pt but no iso requirements (should this be 0.0 or -1, compare to electrons).
                {
                    cutMuVecRecoOnly->push_back(muonsLVec[i]);
                }
                //if(muonsFlagIDVec[i])
                if(AnaFunctions::passMuon( muonsLVec[i], muonsMiniIso[i], 0.0, muonsFlagIDVec[i], AnaConsts::muonsMiniIsoArr))
                {
                    if(AnaFunctions::passMuon( muonsLVec[i], muonsRelIso[i], 0.0, muonsFlagIDVec[i], AnaConsts::muonsMiniIsoArr))
                    {
                        if(nTriggerMuons == 0 && muonsLVec[i].Pt() > 17)  nTriggerMuons++;
                        else if(muonsLVec[i].Pt() > 8)  nTriggerMuons++;
                    }
                    cutMuVec->push_back(muonsLVec[i]);
                    cutMuCharge->push_back(muonsCharge[i]);
                    cutMuJetIndex->push_back(muonsJetIndex[i]);
                    //std::cout<<"cutMuVec PT "<<muonsLVec[i].Pt()<<std::endl; 
                    //cutMuActivity->push_back(muonspfActivity[i]);
                    if(muonsCharge[i] > 0) cutMuSummedCharge++;
                    else                   cutMuSummedCharge--;
                }
            }
            //std::cout<<"New Muon Selection "<<(*cutMuVec).size()<<std::endl;
            //electron selection
            int cutElecSummedCharge = 0;
            for(int i = 0; i < elesLVec.size(); ++i)
            {
                // Electron_cutBased    Int_t   cut-based ID Fall17 V2 (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)
                // Electron_cutBasedNoIso: Removed isolation requirement from eGamma ID;  Int_t  (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)
                bool passElectonID = (elesFlagIDVec[i] == 3);
                //if(elesFlagIDVec[i])
                if(AnaFunctions::passElectron(elesLVec[i], 0.0, -1, passElectonID, AnaConsts::elesMiniIsoArr)) // emulates electrons with pt but no iso requirements.
                {
                    cutElecVecRecoOnly->push_back(elesLVec[i]);
                }

                //if(elesFlagIDVec[i])
                if(AnaFunctions::passElectron(elesLVec[i], elesMiniIso[i], -1, passElectonID, AnaConsts::elesMiniIsoArr))
                {
                    cutElecVec->push_back(elesLVec[i]);
                    cutElecCharge->push_back(elesCharge[i]);
                    cutElecJetIndex->push_back(elesJetIndex[i]);
                    //cutElecActivity->push_back(elespfActivity[i]);
                    if(elesCharge[i] > 0) cutElecSummedCharge++;
                    else                  cutElecSummedCharge--;
                }
            }

            //muons
            tr.registerDerivedVec("cutMuVec",             cutMuVec);
            tr.registerDerivedVec("cutMuVecRecoOnly",     cutMuVecRecoOnly);
            //tr.registerDerivedVec("cutMuActivity",        cutMuActivity);
            tr.registerDerivedVec("cutMuCharge",          cutMuCharge);
            tr.registerDerivedVar("cutMuSummedCharge",    cutMuSummedCharge);
            tr.registerDerivedVec("cutMuJetIndex",        cutMuJetIndex);
            tr.registerDerivedVar("nTriggerMuons",        nTriggerMuons);
            
            //electrons
            tr.registerDerivedVec("cutElecVec",           cutElecVec);
            tr.registerDerivedVec("cutElecVecRecoOnly",   cutElecVecRecoOnly);
            //tr.registerDerivedVec("cutElecActivity",      cutElecActivity);
            tr.registerDerivedVec("cutElecCharge",        cutElecCharge);
            tr.registerDerivedVar("cutElecSummedCharge",  cutElecSummedCharge);
            tr.registerDerivedVec("cutElecJetIndex",      cutElecJetIndex);
            

        }

    public:
        BasicLepton()
        {
        }

        void operator()(NTupleReader& tr)
        {
            basicLepton(tr);
        }
    };

}
#endif
