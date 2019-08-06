#ifndef LEPINFO_H 
#define LEPINFO_H 

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

#include "TFile.h"
#include "TGraphAsymmErrors.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TMath.h"
#include "TLorentzVector.h"
#include "Math/VectorUtil.h"
#include "TRandom3.h"
#include "TVector2.h"

#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <set>

namespace plotterFunctions
{
    class LepInfo
    {
    private:
        // use shared_ptr, which will delete and clear dynamically allocated memory
        std::shared_ptr<TRandom3> tr3;
        std::string year_;
        std::map<std::string, std::string>        trigger_eff_file_map;
        std::map<std::string, TGraphAsymmErrors*> trigger_eff_obj_map;
        TGraphAsymmErrors* Efficiency_Electron_pt;
        TGraphAsymmErrors* Efficiency_Electron_eta;
        TGraphAsymmErrors* Efficiency_Muon_pt;
        TGraphAsymmErrors* Efficiency_Muon_eta;
        bool file_exists    = false;
        bool year_valid     = false;
        bool use_lepton_eff = false;
        
        void lepInfo(NTupleReader& tr)
        {
            std::vector<int> genDecayPdgIdVec;
            std::vector<int> genMotherPdgIdVec;
            std::vector<int> GenPart_statusFlags;
            std::vector<TLorentzVector> genDecayLVec;
            if (tr.checkBranch("GenPartTLV"))
            {
	      genDecayPdgIdVec                    = tr.getVec<int>("GenPart_pdgId");
	      genMotherPdgIdVec                   = tr.getVec<int>("GenPart_genPartIdxMother");
	      GenPart_statusFlags                 = tr.getVec<int>("GenPart_statusFlags");
	      genDecayLVec                        = tr.getVec<TLorentzVector>("GenPartTLV");
            }
            const auto& muonsLVec                           = tr.getVec<TLorentzVector>("MuonTLV");
            const auto& muonsCharge                         = tr.getVec<int>("Muon_charge");
            const auto& jetsLVec                            = tr.getVec<TLorentzVector>("JetTLV");
            const auto& cutMuVec                            = tr.getVec<TLorentzVector>("cutMuVec"); 
            const auto& cutMuVecRecoOnly                    = tr.getVec<TLorentzVector>("cutMuVecRecoOnly"); 
            const auto& cutMuSummedCharge                   = tr.getVar<int>("cutMuSummedCharge"); 
            const auto& nTriggerMuons                       = tr.getVar<int>("nTriggerMuons"); 
            const auto& cutElecVec                          = tr.getVec<TLorentzVector>("cutElecVec"); 
            const auto& cutElecVecRecoOnly                  = tr.getVec<TLorentzVector>("cutElecVecRecoOnly"); 
            const auto& cutElecSummedCharge                 = tr.getVar<int>("cutElecSummedCharge"); 
            const auto& met                                 = tr.getVar<data_t>("MET_pt");
            const auto& metphi                              = tr.getVar<data_t>("MET_phi");

            bool Pass_MuonVeto = false;
            bool Pass_ElecVeto = false;
	    bool isZToLL       = false;
	    bool isTAllHad     = false;
            
            try
	      {
                const auto& Pass_MuonVetoTmp = tr.getVar<bool>("Pass_MuonVeto");
                const auto& Pass_ElecVetoTmp = tr.getVar<bool>("Pass_ElecVeto");
                //const auto& isZToLLTmp = tr.getVar<bool>("isZToLL");
                if(&Pass_MuonVetoTmp != nullptr) Pass_MuonVeto = Pass_MuonVetoTmp;
                if(&Pass_ElecVetoTmp != nullptr) Pass_ElecVeto = Pass_ElecVetoTmp;
		//if(&isZToLLTmp != nullptr) isZToLL = isZToLLTmp;
		
	      }
            catch(const std::string e)
            {
                std::cout << "In LepInfo.h: Caught exception, variable \"" << e << "\" not found" << std::endl;
            }

            auto* genMatchIsoElecInAcc      = new std::vector<TLorentzVector>();
            auto* genMatchElecInAcc         = new std::vector<TLorentzVector>();
            auto* genMatchElecInAccRes      = new std::vector<data_t>();
            auto* genElecInAcc              = new std::vector<TLorentzVector>();
            auto* genElec                   = new std::vector<TLorentzVector>();
            auto* genMatchIsoMuInAcc        = new std::vector<TLorentzVector>();
            auto* genMatchMuInAcc           = new std::vector<const TLorentzVector*>();
            auto* genMatchMuInAccRes        = new std::vector<data_t>();
            auto* genMuInAcc                = new std::vector<TLorentzVector>();
            auto* genMu                     = new std::vector<const TLorentzVector*>();

            //mu45 non-iso trigger emulation
            const double effsnom2012ABC[] = {0.928,0.8302,0.8018};
            const double upedge2012ABC[] = { 0.9, 1.2, 2.1};
            bool muTrigMu45 = false;
            for(const TLorentzVector& mu : cutMuVec)
            {
                if(mu.Pt() > 50)
                {
                    for(int iBin = 0; iBin < sizeof(effsnom2012ABC)/sizeof(double); ++iBin)
                    {
                        if(mu.Eta() < upedge2012ABC[iBin] && tr3 && tr3->Uniform() < effsnom2012ABC[iBin])
                        {
                            muTrigMu45 = true;
                            break;
                        }
                    }
                    if(muTrigMu45) break;
                }
            }

            // muon trigger turn on:     50 GeV
            // electron trigger turn on: 50 GeV
            const double   minMuPt = 20.0,   highMuPt = 50.0;
            const double minElecPt = 20.0, highElecPt = 40.0;

            //Gen info parsing
            const int GENPARTMASK = 0x2100;
            
            if(tr.checkBranch("GenPartTLV") && &genDecayLVec != nullptr)
            {
	      for (int i = 0; i < genDecayPdgIdVec.size(); ++i)
		{
		  ///// added by Bryan to check if Z goes to ll or nunu in ttztollnunu /////
		  /// Check if we can directly identify a leptonic daughter from the Z
		  if ((genDecayPdgIdVec[genMotherPdgIdVec[i]] == 23) && ((abs(genDecayPdgIdVec[i]) == 11) || (abs(genDecayPdgIdVec[i]) == 13) || (abs(genDecayPdgIdVec[i]) == 15))) { 
		    isZToLL = true;
		  }
		  /// If no direct daughter is found, check to see if the lepton in the event is from the main interaction + has a OSSF partner with the same mother
		  else if ((abs(genDecayPdgIdVec[i]) == 11) || (abs(genDecayPdgIdVec[i]) == 13) || (abs(genDecayPdgIdVec[i]) == 15)) {
		    if (((genMotherPdgIdVec[i] == -1) || (genMotherPdgIdVec[i] == 0)) || 
			((abs(genDecayPdgIdVec[genMotherPdgIdVec[i]]) == 1) || (abs(genDecayPdgIdVec[genMotherPdgIdVec[i]]) == 2) || (genDecayPdgIdVec[genMotherPdgIdVec[i]] == 21))) {
		      for (int j = i; j < genDecayPdgIdVec.size(); ++j){
			if ((genMotherPdgIdVec[i] == genMotherPdgIdVec[j]) && (genMotherPdgIdVec[i] == (-1*genMotherPdgIdVec[j]))){
			  isZToLL = true ;
			}
		      }
		    }		    
		  }
		  /// Check if the top decays all hadronically or not
		  // First, find the positive W that is from the top
		  if ((genDecayPdgIdVec[genMotherPdgIdVec[i]] == 6) && (genDecayPdgIdVec[i]) == 24) {
		    // Then Check if it decays hadronically
		    std::vector<int> wDaughters;
		    for (int j = 0; j < genDecayPdgIdVec.size(); ++j){
		      if(genDecayPdgIdVec[genMotherPdgIdVec[j]] == 24) wDaughters.push_back(genDecayPdgIdVec[j]);
		    }
		    int countDaughters = 0;
		    for (int j = 0; j < wDaughters.size(); ++j){
		      if (wDaughters[j] < 10) countDaughters += 1;
		    }
		    if (countDaughters >= 2){
		      // Second, check if negative W decays all hadronically
		      for (int j = 0; j < genDecayPdgIdVec.size(); ++j){ 
			if ((genDecayPdgIdVec[genMotherPdgIdVec[j]] == -6) && (genDecayPdgIdVec[j]) == -24) {
			  // Then Check if it decays hadronically
			  std::vector<int> wDaughters;
			  for (int k = 0; k < genDecayPdgIdVec.size(); ++k){
			    if(genDecayPdgIdVec[genMotherPdgIdVec[j]] == -24) wDaughters.push_back(genDecayPdgIdVec[j]);
			  }
			  int countDaughters = 0;
			  for (int k = 0; k < wDaughters.size(); ++k){
			    if (wDaughters[k] < 10) countDaughters += 1;
			  }
			  if (countDaughters >= 2) isTAllHad = true;
			}
		      }
		    }
		  }
		  //////////////////////////////////////////////////////////////////////////
		  //int i = W_emuVec[index];
		  int maskedStatusFlag = (GenPart_statusFlags[i] & GENPARTMASK);
		  bool isGoodGenPart = (maskedStatusFlag == GENPARTMASK);
		  if (isGoodGenPart)
                    {
		      //muon efficiency and acceptance
		      if(abs(genDecayPdgIdVec[i]) == 13)
                        {
			  genMu->push_back(&genDecayLVec[i]);
			  if(AnaFunctions::passMuonAccOnly(genDecayLVec[i], AnaConsts::muonsMiniIsoArr) && genDecayLVec[i].Pt() > minMuPt)
                            {
			      genMuInAcc->push_back(genDecayLVec[i]);
			      double dRMin = 999.9;
			      double matchPt = -999.9;
			      for(int j = 0; j < cutMuVecRecoOnly.size(); ++j)
                                {
				  // difference in angle between gen and reco muons 
				  double dR = ROOT::Math::VectorUtil::DeltaR(genDecayLVec[i], cutMuVecRecoOnly[j]);
				  if(dR < dRMin)
                                    {
                                        dRMin = dR;
                                        matchPt = cutMuVecRecoOnly[j].Pt();
                                    }
                                }
                                if(dRMin < 0.02)
                                {
                                    genMatchMuInAcc->push_back(&genDecayLVec[i]);
                                    genMatchMuInAccRes->push_back((genDecayLVec[i].Pt() - matchPt)/genDecayLVec[i].Pt());
                                }

                                dRMin = 999.9;
                                for(int j = 0; j < cutMuVec.size(); ++j)
                                {
                                    // difference in angle between gen and cut muons 
                                    double dR = ROOT::Math::VectorUtil::DeltaR(genDecayLVec[i], cutMuVec[j]);
                                    if(dR < dRMin)
                                    {
                                        dRMin = dR;
                                    }
                                }
                                if(dRMin < 0.02)
                                {
                                    genMatchIsoMuInAcc->push_back(genDecayLVec[i]);
                                }
                            }
                        }

                        //electron efficiency and acceptance
                        if(abs(genDecayPdgIdVec[i]) == 11)
                        {
                            genElec->push_back(genDecayLVec[i]);
                            if(AnaFunctions::passElectronAccOnly(genDecayLVec[i], AnaConsts::elesMiniIsoArr) && genDecayLVec[i].Pt() > minElecPt)
                            {
                                genElecInAcc->push_back(genDecayLVec[i]);
                                double dRMin = 999.9;
                                double matchPt = -999.9;
                                for(int j = 0; j < cutElecVecRecoOnly.size(); ++j)
                                {
                                    // difference in angle between gen and reco electrons
                                    double dR = ROOT::Math::VectorUtil::DeltaR(genDecayLVec[i], cutElecVecRecoOnly[j]);
                                    if(dR < dRMin)
                                    {
                                        dRMin = dR;
                                        matchPt = cutElecVecRecoOnly[j].Pt();
                                    }
                                }
                                if(dRMin < 0.02)
                                {
                                    genMatchElecInAcc->push_back(genDecayLVec[i]);
                                    genMatchElecInAccRes->push_back((genDecayLVec[i].Pt() - matchPt)/genDecayLVec[i].Pt());
                                }

                                dRMin = 999.9;
                                for(int j = 0; j < cutElecVec.size(); ++j)
                                {
                                    // difference in angle between gen and cut electrons
                                    double dR = ROOT::Math::VectorUtil::DeltaR(genDecayLVec[i], cutElecVec[j]);
                                    if(dR < dRMin)
                                    {
                                        dRMin = dR;
                                    }
                                }
                                if(dRMin < 0.02)
                                {
                                    genMatchIsoElecInAcc->push_back(genDecayLVec[i]);
                                }
                            }
                        }
                    }
                }
            }

            bool debug = false;

            data_t genZPt = -999.9, genZEta = -999.9, genZmass = -999.9, genZPhi;
            int nZ = 0;
            TLorentzVector genZ;
            int pdgIdZDec = 0;
            if(tr.checkBranch("GenPartTLV") && &genDecayLVec != nullptr)
            {
                for(int j = 0; j <  genDecayPdgIdVec.size(); ++j)
                {
                    //WARNING: make sure to use parentheses; otherwise == happens first and & second!!!
                    int maskedStatusFlag = (GenPart_statusFlags[j] & GENPARTMASK);
                    bool isGoodGenPart = (maskedStatusFlag == GENPARTMASK);
                    if(debug) printf("flag = 0x%x; maskedFlag = 0x%x; mask = 0x%x; pdgid = %d",GenPart_statusFlags[j], maskedStatusFlag, GENPARTMASK, genDecayPdgIdVec[j]);
                    if (isGoodGenPart)
                    {
                        if(abs(genDecayPdgIdVec[j]) == 23)
                        {
                            if (debug) printf(" - behold, a Z boson");
                            nZ++;
                            genZ = genDecayLVec[j];
                            genZPt = genDecayLVec[j].Pt();
                            genZEta = genDecayLVec[j].Eta();
                            genZPhi = genDecayLVec[j].Phi();
                            genZmass = genDecayLVec[j].M();
                        }
                    }
                    if(debug) std::cout << std::endl;
                }
                if (debug)
                {
                    std::cout << "nZ = " << nZ;
                    // be quiet... this prints a lot for ZZTo4Q :-)
                    //if(nZ > 1) std::cout << " - WARNING: MORE THAN 1 Z FOUND!!!";
                    std::cout << std::endl;
                }
                // be quiet... this prints a lot for ZZTo4Q :-)
                //else
                //{
                //    if(nZ > 1) std::cout << " - WARNING: MORE THAN 1 Z FOUND!!! Set debug = true for more info." << std::endl;
                //}

            }

            bool passDiMuTrig  = nTriggerMuons >= 2;

            // the Z mass cut logic assumes that zMassMin < zMassLow
            const double zMassMin  =  50.0;
            const double zMassLow  =  81.0;
            const double zMass     =  91.0;
            const double zMassHigh = 101.0;

            double zMuMassCurrent = 1.0e300, zEff = 1.0e100, zAcc = 1.0e100;
            TLorentzVector bestRecoMuZ;
            TLorentzVector Zrecopt;
            for(int i = 0; i < cutMuVec.size(); ++i)
            {
            Zrecopt =  muonsLVec[0]+muonsLVec[1];//cutMuVec[0] + cutMuVec[1];
                if(cutMuVec[i].Pt() < minMuPt) continue;
                for(int j = 0; j < i && j < cutMuVec.size(); ++j)
                {
                    if(cutMuVec[j].Pt() < minMuPt) continue;
                    double zm = (cutMuVec[i] + cutMuVec[j]).M();
                    // check that zm is non-zero... otherwise 0.0 is chosen over values > 182.0
                    if(zm > 0.0 && fabs(zm - zMass) < fabs(zMuMassCurrent - zMass))
                    {
                        bestRecoMuZ = cutMuVec[i] + cutMuVec[j];
                        zMuMassCurrent = zm;
                    }
                }
            }
            double zElecMassCurrent = 1.0e300;
            TLorentzVector bestRecoElecZ;
            for(int i = 0; i < cutElecVec.size(); ++i)
            {
                if(cutElecVec[i].Pt() < minElecPt) continue;
                for(int j = 0; j < i && j < cutElecVec.size(); ++j)
                {
                    if(cutElecVec[j].Pt() < minElecPt) continue;
                    double zm = (cutElecVec[i] + cutElecVec[j]).M();
                    // check that zm is non-zero... otherwise 0.0 is chosen over values > 182.0
                    if(zm > 0.0 && fabs(zm - zMass) < fabs(zElecMassCurrent - zMass))
                    {
                        bestRecoElecZ = cutElecVec[i] + cutElecVec[j];
                        zElecMassCurrent = zm;
                    }
                }
            }

            double zElMuMassCurrent = 1.0e300;
            TLorentzVector bestRecoElMuZ;
            for(int i = 0; i < cutMuVec.size(); ++i)
            {
                if(cutMuVec[i].Pt() < minMuPt) continue;
                for(int j = 0; j < cutElecVec.size(); ++j)
                {
                    if(cutElecVec[j].Pt() < minMuPt) continue;
                    double zm = (cutMuVec[i] + cutElecVec[j]).M();
                    if(fabs(zm - zMass) < fabs(zElMuMassCurrent - zMass))
                    {
                        bestRecoElMuZ = cutMuVec[i] + cutElecVec[j];
                        zElMuMassCurrent = zm;
                    }
                }
            }

            TLorentzVector metV, metZ;
            metV.SetPtEtaPhiM(met, 0.0, metphi, 0.0);

            // initialized by (0., 0., 0., 0.)
            TLorentzVector bestRecoZ;
            
            // compare masses if both are non-zero
            if (bestRecoElecZ.M() > 0.0 && bestRecoMuZ.M() > 0.0)
            {
                if (fabs(bestRecoElecZ.M() - zMass) < fabs(bestRecoMuZ.M() - zMass))
                {
                    bestRecoZ = bestRecoElecZ;
                }
                else
                {
                    bestRecoZ = bestRecoMuZ;
                }
            }
            // otherwise pick non-zero mass (if any are non-zero)
            else
            {
                if (bestRecoElecZ.M() > 0.0)
                {
                    bestRecoZ = bestRecoElecZ;
                }
                else if (bestRecoMuZ.M() > 0.0)
                {
                    bestRecoZ = bestRecoMuZ;
                }
            }
            
            metZ.SetPtEtaPhiM(bestRecoZ.Pt(), 0.0, bestRecoZ.Phi(), 0.0);
            TLorentzVector cleanMet = metV + metZ;
            
            // di-lepton selections
            bool passMuPt      = (cutMuVec.size() == 2   && cutMuVec[0].Pt() > highMuPt     && cutMuVec[1].Pt() > minMuPt);
            bool passElecPt    = (cutElecVec.size() == 2 && cutElecVec[0].Pt() > highElecPt && cutElecVec[1].Pt() > minElecPt);
            bool passDiMuSel   = (passMuPt   && cutMuSummedCharge == 0);
            bool passDiElecSel = (passElecPt && cutElecSummedCharge == 0);
            bool passElMuSel   = (cutMuVec.size() == 1   && cutElecVec.size() == 1   && cutElecSummedCharge == -cutMuSummedCharge 
                                                         && ( (cutMuVec[0].Pt() > highMuPt && cutElecVec[0].Pt() > minElecPt) || (cutMuVec[0].Pt() > minMuPt && cutElecVec[0].Pt() > highElecPt) ) );
            // Z mass peak selections
            bool passMuZMassMin       = bestRecoMuZ.M()  > zMassMin;
            bool passMuOnZMassPeak    = (bestRecoMuZ.M() > zMassLow)    && (bestRecoMuZ.M() < zMassHigh);
            bool passMuOffZMassPeak   = passMuZMassMin && ! passMuOnZMassPeak;
            bool passElecZMassMin     = bestRecoElecZ.M()  > zMassMin;
            bool passElecOnZMassPeak  = (bestRecoElecZ.M() > zMassLow)  && (bestRecoElecZ.M() < zMassHigh);
            bool passElecOffZMassPeak = passElecZMassMin && ! passElecOnZMassPeak;
            bool passElMuZMassMin     = bestRecoElMuZ.M()  > zMassMin;
            bool passElMuOnZMassPeak  = (bestRecoElMuZ.M() > zMassLow)  && (bestRecoElMuZ.M() < zMassHigh);
            bool passElMuOffZMassPeak = passElMuZMassMin && ! passElMuOnZMassPeak;

            bool passMuZinvSel                = Pass_ElecVeto   && passDiMuSel && passMuZMassMin;
            bool passMuZinvSelOnZMassPeak     = passMuZinvSel   && passMuOnZMassPeak;
            bool passMuZinvSelOffZMassPeak    = passMuZinvSel   && passMuOffZMassPeak;
            bool passElecZinvSel              = Pass_MuonVeto   && passDiElecSel && passElecZMassMin;
            bool passElecZinvSelOnZMassPeak   = passElecZinvSel && passElecOnZMassPeak;
            bool passElecZinvSelOffZMassPeak  = passElecZinvSel && passElecOffZMassPeak;
            bool passElMuZinvSel              = passElMuSel     && passElMuZMassMin;
            bool passElMuZinvSelOnZMassPeak   = passElMuSel     && passElMuOnZMassPeak;
            bool passElMuZinvSelOffZMassPeak  = passElMuSel     && passElMuOffZMassPeak;

            data_t genMuPt     = -999.9;
            data_t genMuEta    = -999.9;
            data_t cutMuPt1    = -999.9;
            data_t cutMuPt2    = -999.9;
            data_t cutMuEta1   = -999.9;
            data_t cutMuEta2   = -999.9;
            data_t cutElecPt1  = -999.9;
            data_t cutElecPt2  = -999.9;
            data_t cutElecEta1 = -999.9;
            data_t cutElecEta2 = -999.9;

            if(genMu->size() >= 1)
            {
                genMuPt  = genMu->at(0)->Pt();
                genMuEta = genMu->at(0)->Eta();
            }
            if(cutMuVec.size() >= 1) 
            {
                cutMuPt1  = cutMuVec.at(0).Pt();
                cutMuEta1 = cutMuVec.at(0).Eta();
            }
            if(cutMuVec.size() >= 2) 
            {
                cutMuPt2  = cutMuVec.at(1).Pt();
                cutMuEta2 = cutMuVec.at(1).Eta();
            }
            if(cutElecVec.size() >= 1) 
            {
                cutElecPt1  = cutElecVec.at(0).Pt();
                cutElecEta1 = cutElecVec.at(0).Eta();
            }
            if(cutElecVec.size() >= 2) 
            {
                cutElecPt2  = cutElecVec.at(1).Pt();
                cutElecEta2 = cutElecVec.at(1).Eta();
            }

            double mindPhiMetJ = 999.9;
            int jc = 0;
            for(const TLorentzVector& jet : jetsLVec)
            {
                if(jc >= 3) break;
                jc++;
                mindPhiMetJ = std::min(mindPhiMetJ, fabs(ROOT::Math::VectorUtil::DeltaPhi(genZ, jet)));
            }

            // Z values
            data_t bestRecoZPt  = bestRecoZ.Pt();
            data_t Zrecoptpt    = Zrecopt.Pt();
            data_t metWithLL    = cleanMet.Pt();
            data_t metphiWithLL = cleanMet.Phi();
            
            // di-lepton efficiencies
            data_t DiMuTriggerEffPt    = 1.0;
            data_t DiMuTriggerEffEta   = 1.0;
            data_t DiElecTriggerEffPt  = 1.0;
            data_t DiElecTriggerEffEta = 1.0;
            if (cutMuVec.size() == 2)
            {
                std::vector<double> leptonPts  = {cutMuPt1, cutMuPt2};
                std::vector<double> leptonEtas = {cutMuEta1, cutMuEta2};
                DiMuTriggerEffPt  = getEfficiency("Muon_pt", leptonPts);
                DiMuTriggerEffEta = getEfficiency("Muon_eta", leptonEtas);
            }
            if (cutElecVec.size() == 2)
            {
                std::vector<double> leptonPts  = {cutElecPt1, cutElecPt2};
                std::vector<double> leptonEtas = {cutElecEta1, cutElecEta2};
                DiElecTriggerEffPt  = getEfficiency("Electron_pt", leptonPts);
                DiElecTriggerEffEta = getEfficiency("Electron_eta", leptonEtas);
            }
            
            // print passMuZinvSelOnZMassPeak conditions
            bool printMuon = false;
            if (printMuon)
            {
                printf("num gen mu: %d num mu: %d num cut mu: %d num cut mu reco only: %d\n", genMu->size(), muonsLVec.size(), cutMuVec.size(), cutMuVecRecoOnly.size());
                if (cutMuVec.size() > 0)
                {
                    printf("Pass_ElecVeto: %d ", Pass_ElecVeto);
                    printf("cutMuVec.size() == 2: %d ", cutMuVec.size() == 2);
                    printf("cutMuSummedCharge == 0: %d ", cutMuSummedCharge == 0);
                    printf("cutMuVec[0].Pt() > highMuPt: %d ", cutMuVec[0].Pt() > highMuPt);
                    printf("cutMuVec[1].Pt() > minMuPt: %d ", cutMuVec[1].Pt() > minMuPt);
                    printf("bestRecoMuZ.M() > zMassLow: %d ", bestRecoMuZ.M() > zMassLow);
                    printf("bestRecoMuZ.M() < zMassHigh: %d ", bestRecoMuZ.M() < zMassHigh);
                    if (passMuZinvSelOnZMassPeak)
                    {
                        printf(" --- passMuZinvSelOnZMassPeak");
                    }
                    printf("\n");
                }
                else
                {
                    printf("WARNING: no cut muons found in event.\n");
                }
            }

            bool printEff = true;
            if (printEff)
            {
                if (cutMuVec.size() == 2)
                {
                    printf("cutMuPt1 = %f ",  cutMuPt1);
                    printf("cutMuPt2 = %f ",  cutMuPt2);
                    printf("cutMuEta1 = %f ", cutMuEta1);
                    printf("cutMuEta2 = %f ", cutMuEta2);
                    printf("DiMuTriggerEffPt = %f ",  DiMuTriggerEffPt);
                    printf("DiMuTriggerEffEta = %f ", DiMuTriggerEffEta);
                    printf("\n");
                }
                if (cutElecVec.size() == 2)
                {
                    printf("cutElecPt1 = %f ",  cutElecPt1);
                    printf("cutElecPt2 = %f ",  cutElecPt2);
                    printf("cutElecEta1 = %f ", cutElecEta1);
                    printf("cutElecEta2 = %f ", cutElecEta2);
                    printf("DiElecTriggerEffPt = %f ",  DiElecTriggerEffPt);
                    printf("DiElecTriggerEffEta = %f ", DiElecTriggerEffEta);
                    printf("\n");
                }
            }
            
            tr.registerDerivedVar("bestRecoZPt", bestRecoZPt);
            tr.registerDerivedVar("bestRecoZM", bestRecoZ.M());
            tr.registerDerivedVar("metWithLL", metWithLL);
            tr.registerDerivedVar("metphiWithLL", metphiWithLL);
            tr.registerDerivedVar("cutMuPt1",    cutMuPt1);
            tr.registerDerivedVar("cutMuPt2",    cutMuPt2);
            tr.registerDerivedVar("cutMuEta1",   cutMuEta1);
            tr.registerDerivedVar("cutMuEta2",   cutMuEta2);
            tr.registerDerivedVar("cutElecPt1",  cutElecPt1);
            tr.registerDerivedVar("cutElecPt2",  cutElecPt2);
            tr.registerDerivedVar("cutElecEta1", cutElecEta1);
            tr.registerDerivedVar("cutElecEta2", cutElecEta2);
            tr.registerDerivedVar("DiMuTriggerEffPt",    DiMuTriggerEffPt);
            tr.registerDerivedVar("DiMuTriggerEffEta",   DiMuTriggerEffEta);
            tr.registerDerivedVar("DiElecTriggerEffPt",  DiElecTriggerEffPt);
            tr.registerDerivedVar("DiElecTriggerEffEta", DiElecTriggerEffEta);
            tr.registerDerivedVar("mindPhiMetJ", mindPhiMetJ);
            tr.registerDerivedVar("ZPtRes", (bestRecoZPt - genZPt)/genZPt);
            tr.registerDerivedVar("ZEtaRes", bestRecoZ.Eta() - genZEta);
            tr.registerDerivedVar("ZPhiRes", bestRecoZ.Phi() - genZPhi);
            tr.registerDerivedVar("ZMRes", (bestRecoZ.M() - genZmass)/genZmass);
            tr.registerDerivedVec("genMu", genMu);
            const auto& genMu_test = tr.getVec<const TLorentzVector*>("genMu");
            tr.registerDerivedVar("ngenMu", static_cast<data_t>(genMu->size()));
            tr.registerDerivedVec("genMuInAcc", genMuInAcc);
            tr.registerDerivedVar("ngenMuInAcc", static_cast<data_t>(genMuInAcc->size()));
            tr.registerDerivedVec("genMatchMuInAcc", genMatchMuInAcc);
            tr.registerDerivedVec("genMatchMuInAccRes", genMatchMuInAccRes);
            tr.registerDerivedVec("genMatchIsoMuInAcc", genMatchIsoMuInAcc);
            tr.registerDerivedVar("ngenMatchMuInAcc", static_cast<data_t>(genMatchMuInAcc->size()));
            tr.registerDerivedVec("genElec", genElec);
            tr.registerDerivedVar("ngenElec", static_cast<data_t>(genElec->size()));
            tr.registerDerivedVec("genElecInAcc", genElecInAcc);
            tr.registerDerivedVar("ngenElecInAcc", static_cast<data_t>(genElecInAcc->size()));
            tr.registerDerivedVec("genMatchElecInAcc", genMatchElecInAcc);
            tr.registerDerivedVec("genMatchElecInAccRes", genMatchElecInAccRes);
            tr.registerDerivedVec("genMatchIsoElecInAcc", genMatchIsoElecInAcc);
            tr.registerDerivedVar("ngenMatchElecInAcc", static_cast<data_t>(genMatchElecInAcc->size()));
            tr.registerDerivedVar("genZPt", genZPt);
            tr.registerDerivedVar("genZEta", genZEta);
            tr.registerDerivedVar("genZPhi", genZPhi);
            tr.registerDerivedVar("genZmass", genZmass);
            tr.registerDerivedVar("pdgIdZDec", pdgIdZDec);
            tr.registerDerivedVar("passDiMuIsoTrig", passDiMuTrig);
            tr.registerDerivedVar("passSingleMu45", muTrigMu45);
            tr.registerDerivedVar("passMuPt",                    passMuPt);
            tr.registerDerivedVar("passElecPt",                  passElecPt);
            tr.registerDerivedVar("passDiMuSel",                 passDiMuSel);
            tr.registerDerivedVar("passDiElecSel",               passDiElecSel);
            tr.registerDerivedVar("passElMuSel",                 passElMuSel);
            tr.registerDerivedVar("passMuZinvSel",               passMuZinvSel);
            tr.registerDerivedVar("passMuZinvSelOnZMassPeak",    passMuZinvSelOnZMassPeak);
            tr.registerDerivedVar("passMuZinvSelOffZMassPeak",   passMuZinvSelOffZMassPeak);
            tr.registerDerivedVar("passElecZinvSel",             passElecZinvSel);
            tr.registerDerivedVar("passElecZinvSelOnZMassPeak",  passElecZinvSelOnZMassPeak);
            tr.registerDerivedVar("passElecZinvSelOffZMassPeak", passElecZinvSelOffZMassPeak);
            tr.registerDerivedVar("passElMuZinvSel",             passElMuZinvSel);
            tr.registerDerivedVar("passElMuZinvSelOnZMassPeak",  passElMuZinvSelOnZMassPeak);
            tr.registerDerivedVar("passElMuZinvSelOffZMassPeak", passElMuZinvSelOffZMassPeak);
            tr.registerDerivedVar("isZToLL", isZToLL);
            tr.registerDerivedVar("isTAllHad", isTAllHad);
            tr.registerDerivedVar("Zrecopt", Zrecoptpt);
        }

        double getEfficiency(std::string kinematic, std::vector<double> values)
        {
            bool verbose = false;
            bool kinematic_valid = trigger_eff_obj_map.find(kinematic) != trigger_eff_obj_map.end();
            if (use_lepton_eff && kinematic_valid)
            {
                TGraphAsymmErrors* eff = trigger_eff_obj_map[kinematic];
                std::vector<float> efficiencies;
                for (const auto& value : values)
                {
                    int x_i = 0;
                    while (x_i < eff->GetN() && eff->GetX()[x_i] - eff->GetErrorXlow(x_i) < value)
                    {
                         ++x_i;
                    }
                    efficiencies.push_back(eff->GetY()[x_i]);
                }
                if (verbose)
                {
                    printf("Trigger Eff %s: ", kinematic.c_str());
                }
                float product = 1.0;
                for (const auto& e : efficiencies)
                {
                    product *= (1.0 - e);
                    if (verbose)
                    {
                        printf("%f ", e);
                    }
                }
                float efficiency = 1.0 - product;
                if (verbose)
                {
                    printf("; eff = %f\n", efficiency);
                }
                return efficiency;
            }
            else
            {
                if (verbose)
                {
                    printf("Setting trigger efficiency to 1.0.\n");
                }
                return 1.0;
            }
        }

    public:
        LepInfo(std::string year = "") : tr3(new TRandom3()), year_(year)
        {
            std::string trigger_eff_file_name = "";
            Efficiency_Electron_pt = nullptr;
            Efficiency_Muon_pt     = nullptr;
            trigger_eff_file_map["2016"] = "2016_trigger_eff.root";
            trigger_eff_file_map["2017"] = "2017_trigger_eff.root";
            trigger_eff_file_map["2018"] = "2018_trigger_eff.root";
            // check that year is valid
            year_valid = trigger_eff_file_map.find(year_) != trigger_eff_file_map.end();
            if (year_valid)
            {
                trigger_eff_file_name = trigger_eff_file_map.at(year_);
                // check if file exists
                struct stat buffer;  
                file_exists = bool(stat(trigger_eff_file_name.c_str(), &buffer) == 0);
                TFile *f = new TFile(trigger_eff_file_name.c_str());
                if(file_exists && f)
                {
                    Efficiency_Electron_pt  = static_cast<TGraphAsymmErrors*>(f->Get("Electron_pt"));
                    Efficiency_Electron_eta = static_cast<TGraphAsymmErrors*>(f->Get("Electron_eta"));
                    Efficiency_Muon_pt      = static_cast<TGraphAsymmErrors*>(f->Get("Muon_pt"));
                    Efficiency_Muon_eta     = static_cast<TGraphAsymmErrors*>(f->Get("Muon_eta"));
                    trigger_eff_obj_map["Electron_pt"]  = Efficiency_Electron_pt;
                    trigger_eff_obj_map["Electron_eta"] = Efficiency_Electron_eta;
                    trigger_eff_obj_map["Muon_pt"]      = Efficiency_Muon_pt;
                    trigger_eff_obj_map["Muon_eta"]     = Efficiency_Muon_eta;
                    f->Close();
                    delete f;
                }
                else
                {
                    std::cout << "Failed to open the file " << trigger_eff_file_name << ". The lepton trigger efficiencies will not be used."<< std::endl;
                }
            }
            else
            {
                std::cout << "The year " << year_ << " is not valid. The lepton trigger efficiencies will not be used."<< std::endl;
            }
            use_lepton_eff = year_valid && file_exists;
        }

        void operator()(NTupleReader& tr)
        {
            lepInfo(tr);
        }
    };

}

#endif
