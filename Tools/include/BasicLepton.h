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
            const auto& muonsLVec        = tr.getVec<TLorentzVector>("MuonTLV");
            const auto& muonsMiniIso     = tr.getVec<data_t>("Muon_miniPFRelIso_all");
	    const auto& muonsPFIsoId     = tr.getVec<unsigned char>("Muon_pfIsoId") ;
            const auto& muonsCharge      = tr.getVec<int>("Muon_charge");
            const auto& muonsJetIndex    = tr.getVec<int>("Muon_jetIdx");
            const auto& muonsFlagIDVec   = tr.getVec<bool_t>("Muon_mediumId");
	    const auto& muonsFlagIDVec_tight = tr.getVec<bool_t>("Muon_tightId");
            const auto& elesLVec         = tr.getVec<TLorentzVector>("ElectronTLV");
            const auto& elesMiniIso      = tr.getVec<data_t>("Electron_miniPFRelIso_all");
            const auto& elesCharge       = tr.getVec<int>("Electron_charge");
            const auto& elesJetIndex     = tr.getVec<int>("Electron_jetIdx");
            const auto& elesFlagIDVec    = tr.getVec<int>("Electron_cutBasedNoIso");
	    const auto& elesFlagIDVec_cutbasedid = tr.getVec<int>("Electron_cutBased");
	    const auto& Pass_MuonVeto    = tr.getVar<bool>("Pass_MuonVeto");
	    const auto& Pass_ElecVeto    = tr.getVar<bool>("Pass_ElecVeto");

            //muons
            auto* cutMuVec            = new std::vector<TLorentzVector>();
            auto* cutMuVecRecoOnly    = new std::vector<TLorentzVector>();
            auto* cutMuCharge         = new std::vector<int>();
            auto* cutMuJetIndex       = new std::vector<int>();
	    //
	    auto* muonFlagId          = new std::vector<int>();
	    //electrons
            auto* cutElecVec          = new std::vector<TLorentzVector>();
            auto* cutElecVecRecoOnly  = new std::vector<TLorentzVector>();
            auto* cutElecCharge       = new std::vector<int>();
            auto* cutElecJetIndex     = new std::vector<int>();

            int cutMuSummedCharge = 0;
            int nTriggerMuons = 0;

	    bool passLooseSingleLep = false;
	    bool passBCSingleLep = false;
	    int n_looseMu    = 0;
	    int n_medMu      = 0;
	    int n_looseElec  = 0;
	    int n_tightElec  = 0;
	    
            //muon selections
            for(int i = 0; i < muonsLVec.size(); ++i)
            {
	      if (muonsLVec[i].Pt() >= 30 && abs(muonsLVec[i].Eta()) <= 2.4) n_looseMu += 1;
	      if (muonsLVec[i].Pt() >= 30 && abs(muonsLVec[i].Eta()) <= 2.4 && muonsFlagIDVec[i]) n_medMu += 1;
	      muonFlagId->push_back(static_cast<int>(muonsFlagIDVec[i])+static_cast<int>(muonsFlagIDVec_tight[i]));
	      //if(muonsFlagIDVec[i])
	        //if(AnaFunctions::passMuon( muonsLVec[i], 0.0, 0.0, true, AnaConsts::muonsMiniIsoArr)) // emulates muons with pt but no iso requirements (should this be 0.0 or -1, compare to electrons).
                if(AnaFunctions::passMuon( muonsLVec[i], 0.0, 0.0, muonsFlagIDVec[i], AnaConsts::muonsMiniIsoArr)) // emulates muons with pt but no iso requirements (should this be 0.0 or -1, compare to electrons).
                {
                    cutMuVecRecoOnly->push_back(muonsLVec[i]);
                }
                //if(muonsFlagIDVec[i])
                //if(AnaFunctions::passMuon( muonsLVec[i], muonsMiniIso[i] / muonsLVec[i].Pt(), 0.0, muonsFlagIDVec[i], AnaConsts::muonsMiniIsoArr))

                if(AnaFunctions::passMuon( muonsLVec[i], muonsMiniIso[i], 0.0, muonsFlagIDVec[i], AnaConsts::muonsMiniIsoArr))

		// if(AnaFunctions::passMuon( muonsLVec[i], muonsMiniIso[i], 0.0, true, AnaConsts::muonsBryanArr) && static_cast<int>(muonsPFIsoId[i]) >= 0)
                //if(AnaFunctions::passMuon( muonsLVec[i], muonsMiniIso[i], 0.0, true, AnaConsts::muonsMiniIsoArr)) // true ---> loose ID
                {
                    //if(AnaFunctions::passMuon( muonsLVec[i], muonsRelIso[i], 0.0, muonsFlagIDVec[i], AnaConsts::muonsMiniIsoArr))
                    //{
                    //    if(nTriggerMuons == 0 && muonsLVec[i].Pt() > 17)  nTriggerMuons++;
                    //    else if(muonsLVec[i].Pt() > 8)  nTriggerMuons++;
                    //}
                    
                    if(nTriggerMuons == 0 && muonsLVec[i].Pt() > 17)  nTriggerMuons++;
                    else if(muonsLVec[i].Pt() > 8)  nTriggerMuons++;
                    
                    cutMuVec->push_back(muonsLVec[i]);
                    cutMuCharge->push_back(muonsCharge[i]);
                    cutMuJetIndex->push_back(muonsJetIndex[i]);
                    
                    if(muonsCharge[i] > 0) cutMuSummedCharge++;
                    else                   cutMuSummedCharge--;
                }
            }
            
            //electron selection
            int cutElecSummedCharge = 0;
            for(int i = 0; i < elesLVec.size(); ++i)
            {
	      if (elesFlagIDVec[i] >=2 && elesLVec[i].Pt() >= 30 && abs(elesLVec[i].Eta()) <= 2.5) n_looseElec += 1;
	      if (elesFlagIDVec[i] >=4 && elesLVec[i].Pt() >= 30 && abs(elesLVec[i].Eta()) <= 2.5) n_tightElec += 1;
                // Electron_cutBased    Int_t   cut-based ID Fall17 V2 (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)
                // Electron_cutBasedNoIso: Removed isolation requirement from eGamma ID;  Int_t  (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)
	      bool passElectronID = (elesFlagIDVec[i] >= 4);
	      //bool passElectonID = (elesFlagIDVec[i] >= 1); // veto ID 
                //if(elesFlagIDVec[i])
	      if(AnaFunctions::passElectron(elesLVec[i], 0.0, -1, passElectronID, AnaConsts::elesMiniIsoArr)) // emulates electrons with pt but no iso requirements.
                {
		  cutElecVecRecoOnly->push_back(elesLVec[i]);
                }

	      //if(elesFlagIDVec[i])
	      //if(AnaFunctions::passElectron(elesLVec[i], elesMiniIso[i] / elesLVec[i].Pt(), -1, passElectonID, AnaConsts::elesMiniIsoArr))
	      if(AnaFunctions::passElectron(elesLVec[i], elesMiniIso[i], -1, passElectronID, AnaConsts::elesMiniIsoArr))
	      //passElectronID   = (elesFlagIDVec_cutbasedid[i] == 4);
	      //if(AnaFunctions::passElectron(elesLVec[i], 0.0, -1, passElectronID, AnaConsts::elesBryanArr)) // emulates electrons with pt but no iso requirements.
                {
		  cutElecVec->push_back(elesLVec[i]);
		  cutElecCharge->push_back(elesCharge[i]);
		  cutElecJetIndex->push_back(elesJetIndex[i]);
		  //cutElecActivity->push_back(elespfActivity[i]);
		  if(elesCharge[i] > 0) cutElecSummedCharge++;
		  else                  cutElecSummedCharge--;
                }
            }
	    if (( n_looseMu + n_looseElec ) > 0) passLooseSingleLep = true;
	    if (( n_medMu   + n_tightElec ) > 0) passBCSingleLep = true;
	    // For tt+Z/H to bb with single lepton, pt > 30
	    bool passSingleLepElec = false;
	    int  n_elec_ptg30      = 0;
	    int  elec_ptg30_index  = 0;
	    bool passSingleLepMu   = false;
	    int  n_mu_ptg30        = 0;
	    int  mu_ptg30_index    = 0;
	    float Lep_pt  = 0; 
	    float Lep_phi = 0; 
	    float Lep_eta = 0; 
	    float Lep_E   = 0;
	    for( int i = 0; i < cutElecVec->size(); ++i){
	      if (cutElecVec->at(i).Pt() > 30){
		n_elec_ptg30++;
		elec_ptg30_index = i;
	      }
	    }
	    for( int i = 0; i < cutMuVec->size(); ++i){
	      if (cutMuVec->at(i).Pt() > 30){
		n_mu_ptg30++;
		mu_ptg30_index = i;
	      }
	    }
	    if ((n_elec_ptg30 == 1) && (n_mu_ptg30 == 0)){   
	      passSingleLepElec = true;
	      Lep_pt  = cutElecVec->at(elec_ptg30_index).Pt();
	      Lep_eta = cutElecVec->at(elec_ptg30_index).Eta();
	      Lep_phi = cutElecVec->at(elec_ptg30_index).Phi();
	      Lep_E   = cutElecVec->at(elec_ptg30_index).E();
	    }
	    if ((n_mu_ptg30 == 1) && (n_elec_ptg30 == 0)){   
	      passSingleLepMu = true;
	      Lep_pt  = cutMuVec->at(mu_ptg30_index).Pt();
	      Lep_eta = cutMuVec->at(mu_ptg30_index).Eta();
	      Lep_phi = cutMuVec->at(mu_ptg30_index).Phi();
	      Lep_E   = cutMuVec->at(mu_ptg30_index).E();
	    }
	    bool passSingleLep = passSingleLepElec || passSingleLepMu;
            //muons 
            tr.registerDerivedVec("cutMuVec",             cutMuVec);
            tr.registerDerivedVec("cutMuVecRecoOnly",     cutMuVecRecoOnly);
            tr.registerDerivedVec("cutMuCharge",          cutMuCharge);
            tr.registerDerivedVar("cutMuSummedCharge",    cutMuSummedCharge);
            tr.registerDerivedVec("cutMuJetIndex",        cutMuJetIndex);
            tr.registerDerivedVar("nTriggerMuons",        nTriggerMuons);
	    tr.registerDerivedVar("passSingleLepMu",      passSingleLepMu);
	    //
	    tr.registerDerivedVec("Muon_FlagId", muonFlagId);
	    //
            //electrons
            tr.registerDerivedVec("cutElecVec",           cutElecVec);
            tr.registerDerivedVec("cutElecVecRecoOnly",   cutElecVecRecoOnly);
            tr.registerDerivedVec("cutElecCharge",        cutElecCharge);
            tr.registerDerivedVar("cutElecSummedCharge",  cutElecSummedCharge);
            tr.registerDerivedVec("cutElecJetIndex",      cutElecJetIndex);
	    tr.registerDerivedVar("passSingleLepElec",    passSingleLepElec);
	    //single lep tlv info
	    tr.registerDerivedVar("Lep_pt",  Lep_pt );
	    tr.registerDerivedVar("Lep_eta", Lep_eta);
	    tr.registerDerivedVar("Lep_phi", Lep_phi);
	    tr.registerDerivedVar("Lep_E",   Lep_E  );
	    //
	    tr.registerDerivedVar("passLooseSingleLep", passLooseSingleLep);
	    tr.registerDerivedVar("passBCSingleLep", passBCSingleLep);
	    tr.registerDerivedVar("passSingleLep", passSingleLep);
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
