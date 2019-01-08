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
    class LepInfo
    {
    private:
        TRandom3 *tr3;
        void lepInfo(NTupleReader& tr)
        {
            const auto& genDecayPdgIdVec                    = tr.getVec<int>("genDecayPdgIdVec");
            const auto& genDecayIdxVec                      = tr.getVec<int>("genDecayIdxVec");
            const auto& genDecayMomIdxVec                   = tr.getVec<int>("genDecayMomIdxVec");
            const auto& genDecayLVec                        = tr.getVec<TLorentzVector>("genDecayLVec");
            const auto& muonsLVec                           = tr.getVec<TLorentzVector>("muonsLVec");
            const auto& muonsRelIso                         = tr.getVec<data_t>("muonsRelIso");
            const auto& muonsMiniIso                        = tr.getVec<data_t>("muonsMiniIso");
            const auto& W_emuVec                            = tr.getVec<int>("W_emuVec");
            const auto& muonsCharge                         = tr.getVec<data_t>("muonsCharge");
            const auto& jetsLVec                            = tr.getVec<TLorentzVector>("jetsLVec");
            const auto& recoJetschargedEmEnergyFraction     = tr.getVec<data_t>("recoJetschargedEmEnergyFraction");
            const auto& recoJetschargedHadronEnergyFraction = tr.getVec<data_t>("recoJetschargedHadronEnergyFraction");
            const auto& muonsFlagIDVec                      = tr.getVec<int>("muonsFlagMedium");
            const auto& elesFlagIDVec                       = tr.getVec<int>("elesFlagVeto");

            const auto& cutMuVec                            = tr.getVec<TLorentzVector>("cutMuVec"); 
            const auto& cutMuVecRecoOnly                    = tr.getVec<TLorentzVector>("cutMuVecRecoOnly"); 
            const auto& cutMuCharge                         = tr.getVec<data_t>("cutMuCharge"); 
            const auto& cutMuActivity                       = tr.getVec<data_t>("cutMuActivity"); 
            const auto& cutMuSummedCharge                   = tr.getVar<int>("cutMuSummedCharge"); 
            const auto& nTriggerMuons                       = tr.getVar<int>("nTriggerMuons"); 

            const auto& cutElecVec                          = tr.getVec<TLorentzVector>("cutElecVec"); 
            const auto& cutElecVecRecoOnly                  = tr.getVec<TLorentzVector>("cutElecVecRecoOnly"); 
            const auto& cutElecCharge                       = tr.getVec<data_t>("cutElecCharge"); 
            const auto& cutElecActivity                     = tr.getVec<data_t>("cutElecActivity"); 
            const auto& cutElecSummedCharge                 = tr.getVar<int>("cutElecSummedCharge"); 

            const std::vector<data_t>& muonspfActivity      = tr.getVec<data_t>("muonspfActivity");
            const std::vector<data_t>& elespfActivity       = tr.getVec<data_t>("elespfActivity");
            const std::vector<data_t>& W_emu_pfActivityVec  = tr.getVec<data_t>("W_emu_pfActivityVec");

            const auto& met                                 = tr.getVar<data_t>("met");
            const auto& metphi                              = tr.getVar<data_t>("metphi");

            const std::vector<TLorentzVector, std::allocator<TLorentzVector> > elesLVec = tr.getVec<TLorentzVector>("elesLVec");
            const auto& elesMiniIso    = tr.getVec<data_t>("elesMiniIso");
            const auto& elesCharge     = tr.getVec<data_t>("elesCharge");
            const auto& elesisEB       = tr.getVec<unsigned int>("elesisEB");

            //const int& nTopCandSortedCnt = tr.getVar<int>("nTopCandSortedCntZinv");

            bool passMuonVeto = false;
            bool passEleVeto = false;


            try
            {
                const auto& passMuonVetoTmp  = tr.getVar<bool>("passMuonVeto");
                const auto& passEleVetoTmp   = tr.getVar<bool>("passEleVeto");
                if(&passMuonVetoTmp != nullptr) passMuonVeto = passMuonVetoTmp;
                if(&passEleVetoTmp != nullptr) passEleVeto = passEleVetoTmp;
            }
            catch(const std::string e)
            {
                //std::cout << "void muInfo(NTupleReader& tr): Caught exception, variable \"" << e << "\" not found" << std::endl;
            }

            auto* genMatchIsoElecInAcc      = new std::vector<TLorentzVector>();
            auto* genMatchElecInAcc         = new std::vector<TLorentzVector>();
            auto* genMatchElecInAccRes      = new std::vector<data_t>();
            auto* genElecInAcc              = new std::vector<TLorentzVector>();
            auto* genElec                   = new std::vector<TLorentzVector>();
            auto* genMatchIsoElecInAccAct   = new std::vector<data_t>();
            auto* genMatchElecInAccAct      = new std::vector<data_t>();
            auto* genElecInAccAct           = new std::vector<data_t>();
            auto* genElecAct                = new std::vector<data_t>();

            auto* genMatchIsoMuInAcc        = new std::vector<TLorentzVector>();
            auto* genMatchMuInAcc           = new std::vector<const TLorentzVector*>();
            auto* genMatchMuInAccRes        = new std::vector<data_t>();
            //auto* genMuInAcc                = new std::vector<const TLorentzVector*>();
            auto* genMuInAcc                = new std::vector<TLorentzVector>();
            auto* genMu                     = new std::vector<const TLorentzVector*>();
            auto* genMatchIsoMuInAccAct     = new std::vector<data_t>();
            auto* genMatchMuInAccAct        = new std::vector<data_t>();
            auto* genMuInAccAct             = new std::vector<data_t>();
            auto* genMuAct                  = new std::vector<data_t>();
            
            // Registered in BasicLepton.h 
            //auto* cutMuVec                  = new std::vector<TLorentzVector>();
            //auto* cutMuCharge               = new std::vector<data_t>();
            //auto* cutMuActivity             = new std::vector<data_t>();
            //
            //auto* cutElecVec                = new std::vector<TLorentzVector>();
            //auto* cutElecCharge             = new std::vector<data_t>();
            //auto* cutElecActivity           = new std::vector<data_t>();

            //std::vector<TLorentzVector> cutMuVecRecoOnly;
            //std::vector<TLorentzVector> cutElecVecRecoOnly;
            
            //std::vector<TLorentzVector>* Zrecopt = new std::vector<TLorentzVector>();

            // Registered in BasicLepton.h 
            
            // //muon selections
            // int cutMuSummedCharge = 0;
            // int nTriggerMuons = 0;

            // for(int i = 0; i < muonsLVec.size(); ++i)
            // {
            //     if(AnaFunctions::passMuon( muonsLVec[i], 0.0, 0.0, muonsFlagIDVec[i], AnaConsts::muonsMiniIsoArr)) // emulates muons with pt but no iso requirements (should this be 0.0 or -1, compare to electrons).
            //     {
            //         cutMuVecRecoOnly.push_back(muonsLVec[i]);
            //     }
            //     if(AnaFunctions::passMuon( muonsLVec[i], muonsMiniIso[i], 0.0, muonsFlagIDVec[i], AnaConsts::muonsMiniIsoArr))
            //     {
            //         if(AnaFunctions::passMuon( muonsLVec[i], muonsRelIso[i], 0.0, muonsFlagIDVec[i], AnaConsts::muonsMiniIsoArr))
            //         {
            //             if(nTriggerMuons == 0 && muonsLVec[i].Pt() > 17)  nTriggerMuons++;
            //             else if(muonsLVec[i].Pt() > 8)  nTriggerMuons++;
            //         }
            //         cutMuVec->push_back(muonsLVec[i]);
            //         //std::cout<<"cutMuVec PT "<<muonsLVec[i].Pt()<<std::endl; 
            //         cutMuCharge->push_back(muonsCharge[i]);
            //         cutMuActivity->push_back(muonspfActivity[i]);
            //         if(muonsCharge[i] > 0) cutMuSummedCharge++;
            //         else                   cutMuSummedCharge--;
            //     }
            // }
            // //std::cout<<"New Muon Selection "<<(*cutMuVec).size()<<std::endl;
            // //electron selection
            // int cutElecSummedCharge = 0;
            // for(int i = 0; i < elesLVec.size(); ++i)
            // {
            //     if(AnaFunctions::passElectron(elesLVec[i], 0.0, -1, elesisEB[i], elesFlagIDVec[i], AnaConsts::elesMiniIsoArr)) // emulates electrons with pt but no iso requirements.
            //     {
            //         cutElecVecRecoOnly.push_back(elesLVec[i]);
            //     }

            //     if(AnaFunctions::passElectron(elesLVec[i], elesMiniIso[i], -1, elesisEB[i], elesFlagIDVec[i], AnaConsts::elesMiniIsoArr))
            //     {
            //         cutElecVec->push_back(elesLVec[i]);
            //         cutElecCharge->push_back(elesCharge[i]);
            //         cutElecActivity->push_back(AnaFunctions::getElectronActivity(elesLVec[i], jetsLVec, recoJetschargedHadronEnergyFraction, AnaConsts::elesAct));
            //         if(elesCharge[i] > 0) cutElecSummedCharge++;
            //         else                  cutElecSummedCharge--;
            //     }
            // }


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

            double genHt = 0.0;

            // muon trigger of 50 GeV
            // electron trigger of 33 GeV
            const double   minMuPt = 20.0,   highMuPt = 50.0;
            const double minElecPt = 20.0, highElecPt = 50.0;
            double nuPt1 = -999.9, nuPt2 = -999.9;

            // gen tops
            std::vector<TLorentzVector>* genTops = nullptr;
            //Gen info parsing
            if(tr.checkBranch("genDecayPdgIdVec") && &genDecayLVec != nullptr)
            {
                // gen tops
                genTops = new std::vector<TLorentzVector>(ttUtility::GetHadTopLVec(genDecayLVec, genDecayPdgIdVec, genDecayIdxVec, genDecayMomIdxVec));

                for(int i = 0; i < genDecayPdgIdVec.size() && i < genDecayLVec.size(); ++i)
                {
                    // genHt
                    if((abs(genDecayPdgIdVec[i]) != 0 &&  abs(genDecayPdgIdVec[i]) < 6) || (abs(genDecayPdgIdVec[i]) > 100 && abs(genDecayPdgIdVec[i]) < 10000)) genHt += genDecayLVec[i].Pt();

                    if(genDecayPdgIdVec[i] ==  13) nuPt1 = genDecayLVec[i].Pt(); // mu+
                    if(genDecayPdgIdVec[i] == -13) nuPt2 = genDecayLVec[i].Pt(); // mu-
                }

                for(int index = 0; index < W_emuVec.size(); ++index)
                {
                    int i = W_emuVec[index];

                    //muon efficiency and acceptance
                    if(abs(genDecayPdgIdVec[i]) == 13)
                    {
                        genMu->push_back(&genDecayLVec[i]);
                        genMuAct->push_back(W_emu_pfActivityVec[index]);
                        if(AnaFunctions::passMuonAccOnly(genDecayLVec[i], AnaConsts::muonsMiniIsoArr) && genDecayLVec[i].Pt() > minMuPt)
                        {
                            genMuInAcc->push_back(genDecayLVec[i]);
                            genMuInAccAct->push_back(genMuAct->back());
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
                                genMatchMuInAccAct->push_back(genMuAct->back());
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
                                genMatchIsoMuInAccAct->push_back(genMuAct->back());
                            }
                        }
                    }

                    //Elec efficiency and acceptance
                    if(abs(genDecayPdgIdVec[i]) == 11)
                    {
                        genElec->push_back(genDecayLVec[i]);
                        genElecAct->push_back(W_emu_pfActivityVec[index]);
                        if(AnaFunctions::passElectronAccOnly(genDecayLVec[i], AnaConsts::elesMiniIsoArr) && genDecayLVec[i].Pt() > minElecPt)
                        {
                            genElecInAcc->push_back(genDecayLVec[i]);
                            genElecInAccAct->push_back(genElecAct->back());
                            //printf("genElecInAcc p_t = %f\n", genDecayLVec[i].Pt());
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
                                genMatchElecInAccAct->push_back(genElecAct->back());
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
                                genMatchIsoElecInAccAct->push_back(genElecAct->back());
                            }
                        }
                    }
                }
            }
            // create empty vector if it is nullptr
            if (genTops == nullptr)
            {
                genTops = new std::vector<TLorentzVector>();
            }

            data_t genZPt = -999.9, genZEta = -999.9, genZmass = -999.9, genZPhi;
            int nZ = 0;
            TLorentzVector genZ;
            int pdgIdZDec = 0;
            if(&genDecayPdgIdVec != nullptr)
            {
                for(int j = 0; j <  genDecayPdgIdVec.size(); ++j)
                {
                    if(abs(genDecayPdgIdVec[j]) == 23)
                    {
                        nZ++;
                        genZ = genDecayLVec[j];
                        genZPt = genDecayLVec[j].Pt();
                        genZEta = genDecayLVec[j].Eta();
                        genZPhi = genDecayLVec[j].Phi();
                        genZmass = genDecayLVec[j].M();
                    }
                }
                if(nZ > 1) std::cout << "!!!WARNING MORE THAN 1 Z FOUND!!!" << std::endl;

                if(&W_emuVec != nullptr)
                {
                    if(W_emuVec.size() == 0) pdgIdZDec = 15;
                    else if(W_emuVec.size() == 2)
                    {
                        if(abs(genDecayPdgIdVec[W_emuVec[0]]) == 11) pdgIdZDec = 11;
                        else if(abs(genDecayPdgIdVec[W_emuVec[0]]) == 13) pdgIdZDec = 13;
                    }
                }

            }

            bool passDiMuTrig  = nTriggerMuons >= 2;

            const double zMassMin = 81.0;
            const double zMass    = 91.0;
            const double zMassMax = 101.0;

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
                    //if(zm > zMassMin && zm < zMassMax && fabs(zm - zMass) < fabs(zMassCurrent - zMass))
                    if(fabs(zm - zMass) < fabs(zMuMassCurrent - zMass))
                    {
                        bestRecoMuZ = cutMuVec[i] + cutMuVec[j];
                        zMuMassCurrent = zm;
                    }
                }
            }
            //std::cout<<Zrecopt.Pt()<<" the fourth one"<<std::endl;
            //Zrecopt = muonsLVec[0]+muonsLVec[1];//cutMuVec[0] + cutMuVec[1];
            double zElecMassCurrent = 1.0e300;
            TLorentzVector bestRecoElecZ;
            for(int i = 0; i < cutElecVec.size(); ++i)
            {
                if(cutElecVec[i].Pt() < minElecPt) continue;
                for(int j = 0; j < i && j < cutElecVec.size(); ++j)
                {
                    if(cutElecVec[j].Pt() < minElecPt) continue;
                    double zm = (cutElecVec[i] + cutElecVec[j]).M();
                    //if(zm > zMassMin && zm < zMassMax && fabs(zm - zMass) < fabs(zMassCurrent - zMass))
                    if(fabs(zm - zMass) < fabs(zElecMassCurrent - zMass))
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
                    //if(zm > zMassMin && zm < zMassMax && fabs(zm - zMass) < fabs(zMassCurrent - zMass))
                    if(fabs(zm - zMass) < fabs(zElMuMassCurrent - zMass))
                    {
                        bestRecoElMuZ = cutMuVec[i] + cutElecVec[j];
                        zElMuMassCurrent = zm;
                    }
                }
            }

            TLorentzVector metV, metZ;
            metV.SetPtEtaPhiM(met, 0.0, metphi, 0.0);

            TLorentzVector bestRecoZ = (fabs(bestRecoElecZ.M() - zMass) > fabs(bestRecoMuZ.M() - zMass)) ? (bestRecoMuZ) : (bestRecoElecZ);
            //if(fabs(bestRecoZ.M() - zMass) > fabs(bestRecoElMuZ.M() - zMass)) bestRecoZ = bestRecoElMuZ;

            metZ.SetPtEtaPhiM(bestRecoZ.Pt(), 0.0, bestRecoZ.Phi(), 0.0);
            TLorentzVector cleanMet = metV + metZ;
            //std::cout<<"metZ "<<metZ.Pt()<<std::endl;
            //std::cout<<"metV "<<metV.Pt()<<std::endl;
            bool passDiMuSel   = passEleVeto  && (cutMuVec.size() == 2   && cutMuSummedCharge == 0   && cutMuVec[0].Pt() > highMuPt     && cutMuVec[1].Pt() > minMuPt);
            bool passDiElecSel = passMuonVeto && (cutElecVec.size() == 2 && cutElecSummedCharge == 0 && cutElecVec[0].Pt() > highElecPt && cutElecVec[1].Pt() > minElecPt);
            bool passElMuSel = (cutMuVec.size() == 1 && cutElecVec.size() == 1 && cutElecSummedCharge == -cutMuSummedCharge && cutMuVec[0].Pt() > highMuPt && cutElecVec[0].Pt() > minMuPt);

            bool passMuZinvSel         = passEleVeto && (cutMuVec.size() == 2   && cutMuSummedCharge == 0   && cutMuVec[0].Pt() > highMuPt     && cutMuVec[1].Pt() > minMuPt)     && (bestRecoMuZ.M() > zMassMin)   && (bestRecoMuZ.M() < zMassMax);
            bool passMuZinvSel_lowpt   = passEleVeto && (cutMuVec.size() == 2   && cutMuSummedCharge == 0   && cutMuVec[0].Pt() > minMuPt      && cutMuVec[1].Pt() > minMuPt)     && (bestRecoMuZ.M() > zMassMin)   && (bestRecoMuZ.M() < zMassMax);
            bool passElecZinvSel       = passMuonVeto && (cutElecVec.size() == 2 && cutElecSummedCharge == 0 && cutElecVec[0].Pt() > highElecPt && cutElecVec[1].Pt() > minElecPt) && (bestRecoElecZ.M() > zMassMin) && (bestRecoElecZ.M() < zMassMax);
            bool passElecZinvSel_lowpt = passMuonVeto && (cutElecVec.size() == 2 && cutElecSummedCharge == 0 && cutElecVec[0].Pt() > minElecPt  && cutElecVec[1].Pt() > minElecPt) && (bestRecoElecZ.M() > zMassMin) && (bestRecoElecZ.M() < zMassMax);
            bool passElMuZinvSel       = (cutMuVec.size() == 1 && cutElecVec.size() == 1 && cutElecSummedCharge == -cutMuSummedCharge && cutMuVec[0].Pt() > highMuPt && cutElecVec[0].Pt() > minMuPt) && (bestRecoElMuZ.M() > zMassMin) && (bestRecoElMuZ.M() < zMassMax);

            double genMuPt   = -999.9;
            double genMuEta  = -999.9;
            double cutMuPt1  = -999.9;
            double cutMuPt2  = -999.9;
            double cutMuEta1 = -999.9;
            double cutMuEta2 = -999.9;

            // print number of muons
            //printf("num gen mu: %d num mu: %d num cut mu: %d num cut mu reco only: %d\n", genMu->size(), muonsLVec.size(), cutMuVec.size(), cutMuVecRecoOnly.size());
            
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

            double cutElecPt1 = -999.9;
            double cutElecPt2 = -999.9;
            if(cutElecVec.size() >= 1) cutElecPt1 = cutElecVec.at(0).Pt();
            if(cutElecVec.size() >= 2) cutElecPt2 = cutElecVec.at(1).Pt();

            double mindPhiMetJ = 999.9;
            int jc = 0;
            for(const TLorentzVector& jet : jetsLVec)
            {
                if(jc >= 3) break;
                jc++;
                mindPhiMetJ = std::min(mindPhiMetJ, fabs(ROOT::Math::VectorUtil::DeltaPhi(genZ, jet)));
            }
            //vTopsNCandNewMVA->push_back(top.getNConstituents());
/*
            int indexMuTrigger, indexElecTrigger, indexHTMHTTrigger, indexMuHTTrigger;
            topTagger::type3TopTagger t3tagger;
            TopTagger *tt, *ttMVA, *ttAllComb;
            Mt2::ChengHanBisect_Mt2_332_Calculator mt2Calculator;
            TopCat topMatcher_;

            const TopTaggerResults& ttrMVA = ttMVA->getResults();
            std::vector<TopObject> topMVACands = ttrMVA.getTopCandidates();
            int monoJet;
            int diJet;
            int triJet;
            for(int iTop = 0; iTop < ttrMVA.getTops().size(); ++iTop)
            {
                auto& top = *ttrMVA.getTops()[iTop];
            if top.getNConstituents() =1 monoJet++;
            if top.getNConstituents() =2 diJet++;
            if top.getNConstituents() =3 triJet++;
            }
*/
            //const int& nTopCandSortedCnt = tr.getVar<int>("nTopCandSortedCntZinv");
            /*
            std::shared_ptr<TopTagger> ttPtr;
            int monoJet;
            const TopTaggerResults& ttr = ttPtr->getResults();
            std::vector<TopObject*> Ntop = ttr.getTops();
            for(int i=1; i<nTopCandSortedCnt; i++){
            if(Ntop[i]->getNConstituents() == 1) monoJet++;
            }
            std::cout<<monoJet<<std::endl;
            */
            //tr.getTops();
            //TopTagger *tt, *ttMVA, *ttAllComb;
            //ttMVA = new TopTagger();
            //ttMVA->setCfgFile("TopTagger.cfg");
            //const TopTaggerResults& ttResults = ttMVA->getResults();
            //std::vector<TopObject>& topCandidates = ttResults.getTopCandidates();
            //if(topCandidates.getNConstituents() = 1) monoJet++;
            //TopTagger *tt, *ttMVA, *ttAllComb;
            //ttMVA->setCfgFile("TopTagger.cfg");
            /*
            const TopTaggerResults& ttrMVA = ttMVA->getResults();
            for(int iTop = 0; iTop < ttrMVA.getTops().size(); ++iTop)
            {
                auto& top = *ttrMVA.getTops()[iTop];
             
             if(top.getNConstituents() == 1) monoJet++;
             }
            std::cout<< monoJet << std::endl;
            */
            //if(genZPt > 600) std::cout << "HELLO THERE!!!!" << std::endl;
            //if(genZPt > 600 && mindPhiMetJ < 0.5) std::cout << "BONJOUR!!! \t" << genZPt << "\t" << mindPhiMetJ << "\t" << run << "\t" << lumi << "\t" << event << std::endl;
            //std::cout<<"cleanMetPt "<<cleanMet.Pt()<<std::endl;
            //std::cout<<" "<<std::endl;

            //printf("ngenElec = %d; ngenElecInAcc = %d; ngenMatchElecInAcc = %d\n", genElec->size(), genElecInAcc->size(), genMatchElecInAcc->size());

            data_t bestRecoZPt = bestRecoZ.Pt();
            data_t cleanMetPt  = cleanMet.Pt();
            data_t cleanMetPhi = cleanMet.Phi();
            data_t Zrecoptpt = Zrecopt.Pt();
            //data_t cleanMet2Pt = static_cast<data_t>(cleanMet2.Pt());
            tr.registerDerivedVar("bestRecoZPt", bestRecoZPt);
            tr.registerDerivedVar("bestRecoZM", bestRecoZ.M());
            tr.registerDerivedVar("cleanMetPt", cleanMetPt);
            tr.registerDerivedVar("cleanMetPhi", cleanMetPhi);
            //tr.registerDerivedVar("cleanMet2Pt", cleanMet2Pt);
            tr.registerDerivedVar("genHt", genHt);
            tr.registerDerivedVar("cutMuPt1", cutMuPt1);
            tr.registerDerivedVar("cutMuPt2", cutMuPt2);
            tr.registerDerivedVar("cutMuEta1", cutMuEta1);
            tr.registerDerivedVar("cutMuEta2", cutMuEta2);
            tr.registerDerivedVar("cutElecPt1", cutElecPt1);
            tr.registerDerivedVar("cutElecPt2", cutElecPt2);
            tr.registerDerivedVar("mindPhiMetJ", mindPhiMetJ);

            tr.registerDerivedVar("ZPtRes", (bestRecoZPt - genZPt)/genZPt);
            tr.registerDerivedVar("ZEtaRes", bestRecoZ.Eta() - genZEta);
            tr.registerDerivedVar("ZPhiRes", bestRecoZ.Phi() - genZPhi);
            tr.registerDerivedVar("ZMRes", (bestRecoZ.M() - genZmass)/genZmass);

            // Registered in BasicLepton.h 
            //tr.registerDerivedVec("cutMuVec", cutMuVec);
            //tr.registerDerivedVec("cutElecVec", cutElecVec);
            //tr.registerDerivedVec("cutMuActivity", cutMuActivity);
            //tr.registerDerivedVec("cutElecActivity", cutElecActivity);

            tr.registerDerivedVec("genMu", genMu);
            const auto& genMu_test = tr.getVec<const TLorentzVector*>("genMu");
            tr.registerDerivedVar("ngenMu", static_cast<data_t>(genMu->size()));
            tr.registerDerivedVec("genMuInAcc", genMuInAcc);
            tr.registerDerivedVec("genMuAct", genMuAct);
            tr.registerDerivedVar("ngenMuInAcc", static_cast<data_t>(genMuInAcc->size()));
            tr.registerDerivedVec("genMuInAccAct", genMuInAccAct);
            tr.registerDerivedVec("genMatchMuInAcc", genMatchMuInAcc);
            tr.registerDerivedVec("genMatchMuInAccRes", genMatchMuInAccRes);
            tr.registerDerivedVec("genMatchIsoMuInAcc", genMatchIsoMuInAcc);
            tr.registerDerivedVar("ngenMatchMuInAcc", static_cast<data_t>(genMatchMuInAcc->size()));
            tr.registerDerivedVec("genMatchMuInAccAct", genMatchMuInAccAct);
            tr.registerDerivedVec("genMatchIsoMuInAccAct", genMatchIsoMuInAccAct);

            tr.registerDerivedVec("genElec", genElec);
            tr.registerDerivedVar("ngenElec", static_cast<data_t>(genElec->size()));
            tr.registerDerivedVec("genElecInAcc", genElecInAcc);
            tr.registerDerivedVec("genElecAct", genElecAct);
            tr.registerDerivedVar("ngenElecInAcc", static_cast<data_t>(genElecInAcc->size()));
            tr.registerDerivedVec("genElecInAccAct", genElecInAccAct);
            tr.registerDerivedVec("genMatchElecInAcc", genMatchElecInAcc);
            tr.registerDerivedVec("genMatchElecInAccRes", genMatchElecInAccRes);
            tr.registerDerivedVec("genMatchIsoElecInAcc", genMatchIsoElecInAcc);
            tr.registerDerivedVar("ngenMatchElecInAcc", static_cast<data_t>(genMatchElecInAcc->size()));
            tr.registerDerivedVec("genMatchElecInAccAct", genMatchElecInAccAct);
            tr.registerDerivedVec("genMatchIsoElecInAccAct", genMatchIsoElecInAccAct);

            tr.registerDerivedVar("genZPt", genZPt);
            tr.registerDerivedVar("genZEta", genZEta);
            tr.registerDerivedVar("genZPhi", genZPhi);
            tr.registerDerivedVar("genZmass", genZmass);
            tr.registerDerivedVar("pdgIdZDec", pdgIdZDec);
            tr.registerDerivedVar("passDiMuIsoTrig", passDiMuTrig);
            tr.registerDerivedVar("passSingleMu45", muTrigMu45);

            tr.registerDerivedVar("passDiMuSel", passDiMuSel);
            tr.registerDerivedVar("passDiElecSel", passDiElecSel);
            tr.registerDerivedVar("passElMuSel", passElMuSel);

            tr.registerDerivedVar("passMuZinvSel",         passMuZinvSel);
            tr.registerDerivedVar("passMuZinvSel_lowpt",   passMuZinvSel_lowpt);
            tr.registerDerivedVar("passElecZinvSel",       passElecZinvSel);
            tr.registerDerivedVar("passElecZinvSel_lowpt", passElecZinvSel_lowpt);
            tr.registerDerivedVar("passElMuZinvSel",       passElMuZinvSel);

            tr.registerDerivedVar("Zrecopt", Zrecoptpt);
            // gen tops
            tr.registerDerivedVec("genTops", genTops);

        }

    public:
        LepInfo()
        {
            tr3 = new TRandom3();
        }

        void operator()(NTupleReader& tr)
        {
            lepInfo(tr);
        }
    };

}
#endif
