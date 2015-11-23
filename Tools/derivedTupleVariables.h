#include "SusyAnaTools/Tools/NTupleReader.h"
#include "SusyAnaTools/Tools/customize.h"
#include "SusyAnaTools/Tools/baselineDef.h"
#include "SusyAnaTools/Tools/searchBins.h"

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TMath.h"
#include "TLorentzVector.h"
#include "Math/VectorUtil.h"
#include "TRandom3.h"

#include <vector>
#include <iostream>

namespace plotterFunctions
{
    class GenerateWeight
    {
    private:
        TH1* muEff;
        TH1* muEffReco;
        TH2* muEffIso;
        TH2* elecEffReco;
        TH1* elecEffIso;
        TH2* muAcc;
        TH1* hZEff;
        TH1* hZAcc;
        TH1* hZAccElec;

        void generateWeight(NTupleReader& tr)
        {
            const std::vector<TLorentzVector>& jetsLVec         = tr.getVec<TLorentzVector>("jetsLVecLepCleaned");
            const std::vector<TLorentzVector>& cutMuVec         = tr.getVec<TLorentzVector>("cutMuVec");
            const std::vector<TLorentzVector>& cutElecVec       = tr.getVec<TLorentzVector>("cutElecVec");
            const std::vector<double>& cutMuActivity            = tr.getVec<double>("cutMuActivity");
            const std::vector<double>& cutElecActivity          = tr.getVec<double>("cutElecActivity");
            const std::vector<TLorentzVector*>& genMu           = tr.getVec<TLorentzVector*>("genMu");
            const std::vector<TLorentzVector*>& genMuInAcc      = tr.getVec<TLorentzVector*>("genMuInAcc");
            const std::vector<TLorentzVector*>& genMatchMuInAcc = tr.getVec<TLorentzVector*>("genMatchMuInAcc");

            const int& pdgIdZDec      = tr.getVar<int>("pdgIdZDec");
            const bool& passMuZinvSel = tr.getVar<bool>("passMuZinvSel");
            const bool& passElecZinvSel = tr.getVar<bool>("passElecZinvSel");
            const double& ht          = tr.getVar<double>("ht");
            const double& bestRecoZPt = tr.getVar<double>("bestRecoZPt");
            const double& genZPt      = tr.getVar<double>("genZPt");

            // Calculate PU weight

            // Calculate Z-eff weight

            const double zMassMin = 71.0;
            const double zMass    = 91.0;
            const double zMassMax = 111.0;

            double zMassCurrent = 1.0e300, zEff = 1.0e100, zAcc = 1.0e100;
            for(int i = 0; i < cutMuVec.size(); ++i)
            {
                if(cutMuVec[i].Pt() < 10) continue;
                for(int j = 0; j < i && j < cutMuVec.size(); ++j)
                {
                    if(cutMuVec[j].Pt() < 10) continue;
                    double zm = (cutMuVec[i] + cutMuVec[j]).M();
                    if(fabs(zm - zMass) < fabs(zMassCurrent - zMass))
                    {
                        zMassCurrent = zm;
                        double mu1pt = cutMuVec[i].Pt();
                        double mu2pt = cutMuVec[j].Pt();

                        double mu1Act = cutMuActivity[i];
                        double mu2Act = cutMuActivity[j];

                        //set to not overflow histograms
                        if(mu1pt >= 2000.0) mu1pt = 1999.9;
                        if(mu2pt >= 2000.0) mu2pt = 1999.9;
                        
                        //Get mu efficiencies
                        double muEff1 = 0.0, muEff2 = 0.0;

                        if(muEff && muEffReco && muEffIso)
                        {
                            //Fit to reco eff (eff = p0 + p1*pt + p2*pt^2
                            //PHYS14
                            //const double fitStart = 200.0; // extended to 1400 GeV
                            //const double p0 =     0.955847; // +/- 0.461944    
                            //const double p1 = -2.24431e-05; // +/- 0.00128305  
                            //const double p2 = -5.68907e-08; // +/- 7.85913e-07
                            //Sprint15
                            //const double fitStart = 200.0; // extended to 1400 GeV
                            //const double p0 =  9.83467e-01; // +/- 1.54469e+00
                            //const double p1 = -7.81897e-06; // +/- 4.16487e-03
                            //const double p2 = -1.22092e-08; // +/- 2.18556e-06
                            const double fitStart = 200.0;  // extended to 2000 GeV
                            const double p0 =     0.979238; //   +/-   1.17313     
                            const double p1 = -6.47338e-06; //   +/-   0.00342715  
                            const double p2 = -1.16258e-08; //   +/-   1.87837e-06 

                            double muRecoEff = 0.0;

                            int recoPtBin = muEffReco->GetXaxis()->FindBin(mu1pt);
                            if(mu1pt > fitStart) muRecoEff = p0 + p1*mu1pt + p2*mu1pt*mu1pt;
                            else                 muRecoEff = muEffReco->GetBinContent(recoPtBin);
                            int isoPtBin = muEffIso->GetXaxis()->FindBin(mu1pt);
                            if(isoPtBin >= muEffIso->GetNbinsX()) isoPtBin = muEffIso->GetNbinsX();
                            int isoActBin = muEffIso->GetYaxis()->FindBin(mu1Act);
                            if(isoActBin >= muEffIso->GetNbinsY()) isoActBin = muEffIso->GetNbinsY();
                            muEff1 = muRecoEff * muEffIso->GetBinContent(isoPtBin, isoActBin);

                            muRecoEff = 0.0;
                            recoPtBin = muEffReco->GetXaxis()->FindBin(mu2pt);
                            if(mu2pt > fitStart) muRecoEff = p0 + p1*mu2pt + p2*mu2pt*mu2pt;
                            else                 muRecoEff = muEffReco->GetBinContent(recoPtBin);
                            isoPtBin = muEffIso->GetXaxis()->FindBin(mu2pt);
                            if(isoPtBin >= muEffIso->GetNbinsX()) isoPtBin = muEffIso->GetNbinsX();
                            isoActBin = muEffIso->GetYaxis()->FindBin(mu2Act);
                            if(isoActBin >= muEffIso->GetNbinsY()) isoActBin = muEffIso->GetNbinsY();
                            muEff2 = muRecoEff * muEffIso->GetBinContent(isoPtBin, isoActBin);
                        }

                        
                        if((mu1pt > 20 && muEff1 < 1.0e-5) || (mu2pt > 20 && muEff2 < 1.0e-5)) 
                        {
                            std::cout << "SMALL muEff!!! muEff1: " << muEff1 << "\tmuEff2: " << muEff2 << "\t" << mu1pt << "\t" << mu2pt << std::endl;
                            zEff = 1.0e-10;
                        }
                        else zEff = muEff1 * muEff2;
                    }
                }
            }

            zMassCurrent = 1.0e300;
            double zEffElec = 1.0e100, zAccElec = 1.0e100;
            for(int i = 0; i < cutElecVec.size(); ++i)
            {
                if(cutElecVec[i].Pt() < 10) continue;
                for(int j = 0; j < i && j < cutElecVec.size(); ++j)
                {
                    if(cutElecVec[j].Pt() < 10) continue;
                    double zm = (cutElecVec[i] + cutElecVec[j]).M();
                    if(fabs(zm - zMass) < fabs(zMassCurrent - zMass))
                    {
                        zMassCurrent = zm;
                        double elec1pt = cutElecVec[i].Pt();
                        double elec2pt = cutElecVec[j].Pt();

                        double elec1Act = cutElecActivity[i];
                        double elec2Act = cutElecActivity[j];

                        //set to not overflow histograms
                        if(elec1pt >= 2000.0) elec1pt = 1999.9;
                        if(elec2pt >= 2000.0) elec2pt = 1999.9;

                        //Get elec efficiencies
                        double elecEff1 = 0.0, elecEff2 = 0.0;
                        if(elecEffReco && elecEffIso)
                        {
                            //Fit to iso eff (eff = p0 + p1*pt + p2*pt^2
                            const double fitStart = 200.0;  // extended to 2000 GeV
                            const double p0 =     0.989411; //   +/-   1.18467     
                            const double p1 =  3.66321e-06; //   +/-   0.00346729  
                            const double p2 = -5.68292e-09; //   +/-   1.90334e-06 

                            double elecIsoEff = 0.0;

                            int isoPtBin = elecEffIso->GetXaxis()->FindBin(elec1pt);
                            if(elec1pt > fitStart) elecIsoEff = p0 + p1*elec1pt + p2*elec1pt*elec1pt;
                            else                   elecIsoEff = elecEffIso->GetBinContent(isoPtBin);
                        
                            int recoPtBin = elecEffReco->GetXaxis()->FindBin(elec1pt);
                            if(recoPtBin >= elecEffReco->GetNbinsX()) recoPtBin = elecEffReco->GetNbinsX();
                        
                            int recoActBin = elecEffReco->GetYaxis()->FindBin(elec1Act);
                            if(recoActBin >= elecEffReco->GetNbinsY()) recoActBin = elecEffReco->GetNbinsY();
                        
                            elecEff1 = elecIsoEff * elecEffReco->GetBinContent(recoPtBin, recoActBin);
                            //std::cout << elec1pt << "\t" << recoPtBin << "\t" <<  elec1Act << "\t" << recoActBin << "\t" << elecEff1 << "\t" << elecIsoEff << "\t" << elecEffReco->GetBinContent(recoPtBin, recoActBin) << std::endl;
                        
                            elecIsoEff = 0.0;
                            isoPtBin = elecEffIso->GetXaxis()->FindBin(elec2pt);
                            if(elec2pt > fitStart) elecIsoEff = p0 + p1*elec2pt + p2*elec2pt*elec2pt;
                            else                   elecIsoEff = elecEffIso->GetBinContent(isoPtBin);
                        
                            recoPtBin = elecEffReco->GetXaxis()->FindBin(elec2pt);
                            if(recoPtBin >= elecEffReco->GetNbinsX()) recoPtBin = elecEffReco->GetNbinsX();
                        
                            recoActBin = elecEffReco->GetYaxis()->FindBin(elec2Act);
                            if(recoActBin >= elecEffReco->GetNbinsY()) recoActBin = elecEffReco->GetNbinsY();
                        
                            elecEff2 = elecIsoEff * elecEffReco->GetBinContent(recoPtBin, recoActBin);
                            //std::cout << elec2pt << "\t" << recoPtBin << "\t" << elec2Act << "\t" << recoActBin << "\t" << elecEff2 << "\t" << elecIsoEff << "\t" << elecEffReco->GetBinContent(recoPtBin, recoActBin) << std::endl;
                        }

                        
                        if((elec1pt > 33 && elec2pt) && (elecEff1 < 1.0e-5 || elecEff2 < 1.0e-5)) 
                        {
                            std::cout << "SMALL elecEff!!! elecEff1: " << elecEff1 << "\telecEff2: " << elecEff2 << "\t" << elec1pt << "\t" << elec2pt << std::endl;
                            zEffElec = 1.0e-10;
                        }
                        else zEffElec = elecEff1 * elecEff2;
                    }
                }
            }

            double genCleanHt = ht;
            for(auto& tlvp : genMuInAcc) if(tlvp->Pt() > 50) genCleanHt -= tlvp->Pt();

            const double MHT_jetPtMin = 30.0;
            TLorentzVector MHT;
            double mu1dRMin = 99.9, mu2dRMin = 99.9;
            for(auto& jet : jetsLVec)
            {
                double mu1dR = 999.9, mu2dR = 999.9;
                if(cutMuVec.size() >= 1) mu1dR = ROOT::Math::VectorUtil::DeltaR(jet, cutMuVec[0]);
                if(cutMuVec.size() >= 2) mu2dR = ROOT::Math::VectorUtil::DeltaR(jet, cutMuVec[1]);
                mu1dRMin = std::min(mu1dRMin, mu1dR);
                mu2dRMin = std::min(mu2dRMin, mu2dR);

                if(jet.Pt() > MHT_jetPtMin) MHT += jet;
            }
        
            double mudR = 99.9, genMudR = 99.9, genMudPhi = 99.9, genMudEta = 99.9;

            if(cutMuVec.size() >= 2) mudR    = ROOT::Math::VectorUtil::DeltaR(cutMuVec[0], cutMuVec[1]);
            if(genMu.size() > 2) std::cout << "MORE THAN 2 GEN MUONS: " << genMu.size() << std::endl;
            if(genMu.size() >= 2)    
            {
                genMudR = ROOT::Math::VectorUtil::DeltaR(*(genMu[0]), *(genMu[1]));
                genMudPhi = ROOT::Math::VectorUtil::DeltaPhi(*(genMu[0]), *(genMu[1]));
                genMudEta = genMu[0]->Eta() - genMu[1]->Eta();
            }

            // muon Z pt acceptance corrections
            //functional form [2] - exp([0] + [1]*x)
            //PHYS14
            //double acc_p0 = -2.91374e-01;
            //double acc_p1 = -4.42884e-03;
            //double acc_p2 =  9.51190e-01;
            //Spring15
            double acc_p0 = -2.64921e-01;
            double acc_p1 = -4.65305e-03;
            double acc_p2 =  9.50493e-01;
        
            if(hZAcc) 
            {
                if(bestRecoZPt < 100) zAcc = hZAcc->GetBinContent(hZAcc->GetXaxis()->FindBin(bestRecoZPt));
                else                  zAcc = acc_p2 - exp(acc_p0 + acc_p1 * bestRecoZPt);
            }

            if(passMuZinvSel && zAcc < 0.05) 
            {
                std::cout << "WARNING: Muon Z acceptance < 0.05, forcing weight to zero! Acc_Z: " << zAcc << std::endl;
                zAcc = 1.0e101;
            }

            if(passMuZinvSel && zEff < 0.04) 
            {
                std::cout << "WARNING: Muon Z efficiency < 0.05, forcing weight to zero! Eff_Z: " << zEff << "\tZ(Pt): " << bestRecoZPt <<  std::endl;
                zEff = 1.0e101;
            }

            // electron Z pt acceptance corrections
            //functional form [2] - exp([0] + [1]*x)
            //Spring15
            //double elecAcc_p0 = -3.18955e-02;
            //double elecAcc_p1 = -5.16522e-03;
            //double elecAcc_p2 = 8.92280e-01;
            double elecAcc_p0 = -2.50679e-01;
            double elecAcc_p1 = -5.34976e-03;
            double elecAcc_p2 = 9.33562e-01;
        
            if(hZAccElec)
            {
                if(bestRecoZPt < 100) zAccElec = hZAccElec->GetBinContent(hZAccElec->GetXaxis()->FindBin(bestRecoZPt));
                else                  zAccElec = elecAcc_p2 - exp(elecAcc_p0 + elecAcc_p1 * bestRecoZPt);
            }

            if(passElecZinvSel && zAccElec < 0.05) 
            {
                std::cout << "WARNING: Elec Z acceptance < 0.05, forcing weight to zero! Acc_Z: " << zAccElec << std::endl;
                zAccElec = 1.0e101;
            }

            if(passElecZinvSel && zEffElec < 0.04) 
            {
                std::cout << "WARNING: Elec Z efficiency < 0.05, forcing weight to zero! Eff_Z: " << zEffElec << "\tZ(Pt): " << bestRecoZPt <<  std::endl;
                zEffElec = 1.0e101;
            }

            tr.registerDerivedVar("mu1dRMin", mu1dRMin);
            tr.registerDerivedVar("mu2dRMin", mu2dRMin);
            tr.registerDerivedVar("mudR", mudR);
            tr.registerDerivedVar("genMudR", genMudR);
            tr.registerDerivedVar("genMudPhi", genMudPhi);
            tr.registerDerivedVar("genMudEta", genMudEta);

            tr.registerDerivedVar("zEff", zEff);
            tr.registerDerivedVar("zEffWgt", 1.0/zEff);
            tr.registerDerivedVar("zAcc", zAcc);
            tr.registerDerivedVar("zAccWgt", 1.0/zAcc);

            tr.registerDerivedVar("zEffElec", zEffElec);
            tr.registerDerivedVar("zEffWgtElec", 1.0/zEffElec);
            tr.registerDerivedVar("zAccElec", zAccElec);
            tr.registerDerivedVar("zAccWgtElec", 1.0/zAccElec);

            tr.registerDerivedVar("cleanMHt", MHT.Pt());
            tr.registerDerivedVar("cleanMHtPhi", MHT.Phi());
        }

    public:
        GenerateWeight()
        {
            TH1::AddDirectory(false);
            TFile *f = new TFile("lepEffHists.root");
            if(f)
            {
                muEff        = static_cast<TH1*>(f->Get("hMuEffPt_ratio"));
                muEffReco    = static_cast<TH1*>(f->Get("hMuEffPtReco_ratio"));
                muEffIso     = static_cast<TH2*>(f->Get("hMuEffPtActIso_ratio"));
                muAcc        = static_cast<TH2*>(f->Get("hMuAcc_ratio"));
                hZEff        = static_cast<TH1*>(f->Get("hZEffPt_ratio"));
                hZAcc        = static_cast<TH1*>(f->Get("hZAccPtSmear_ratio"));
                elecEffReco  = static_cast<TH2*>(f->Get("hElecEffPtActReco_ratio"));
                elecEffIso   = static_cast<TH1*>(f->Get("hElecEffPtIso_ratio"));
                hZAccElec    = static_cast<TH1*>(f->Get("hZElecAccPtSmear_ratio"));
                f->Close();
                delete f;
            }
            else
            {
                std::cout << "Failed to open: muEffHists.root" << std::endl;
            }
        }

        void operator()(NTupleReader& tr)
        {
            generateWeight(tr);
        }
    };

    class LepInfo
    {
    private:
        TRandom3 *tr3;

        void lepInfo(NTupleReader& tr)
        {
            const std::vector<int>& genDecayPdgIdVec        = tr.getVec<int>("genDecayPdgIdVec");
            const std::vector<TLorentzVector>& genDecayLVec = tr.getVec<TLorentzVector>("genDecayLVec");
            const std::vector<TLorentzVector>& muonsLVec    = tr.getVec<TLorentzVector>("muonsLVec");
            const std::vector<double>& muonsRelIso          = tr.getVec<double>("muonsRelIso");
            const std::vector<double>& muonsMiniIso         = tr.getVec<double>("muonsMiniIso");
            const std::vector<int>& W_emuVec                = tr.getVec<int>("W_emuVec");
            const std::vector<double>& muonsCharge          = tr.getVec<double>("muonsCharge");
            const std::vector<TLorentzVector>& jetsLVec     = tr.getVec<TLorentzVector>("jetsLVec");
            const std::vector<double>& recoJetschargedEmEnergyFraction     = tr.getVec<double>("recoJetschargedEmEnergyFraction"); 
            const std::vector<double>& recoJetschargedHadronEnergyFraction = tr.getVec<double>("recoJetschargedHadronEnergyFraction");
            const std::vector<int> & muonsFlagIDVec = tr.getVec<int>("muonsFlagMedium");
            const std::vector<int>&  elesFlagIDVec  = tr.getVec<int>("elesFlagVeto");

            const std::vector<double>& muonspfActivity      = tr.getVec<double>("muonspfActivity");
            const std::vector<double>& elespfActivity       = tr.getVec<double>("elespfActivity");
            const std::vector<double>& W_emu_pfActivityVec  = tr.getVec<double>("W_emu_pfActivityVec");

            const double& ht                             = tr.getVar<double>("ht");
            const double& met                            = tr.getVar<double>("met");
            const double& metphi                         = tr.getVar<double>("metphi");

            const std::vector<TLorentzVector, std::allocator<TLorentzVector> > elesLVec = tr.getVec<TLorentzVector>("elesLVec");
            const std::vector<double>& elesMiniIso          = tr.getVec<double>("elesMiniIso");
            const std::vector<double>& elesCharge           = tr.getVec<double>("elesCharge");
            const std::vector<unsigned int>& elesisEB       = tr.getVec<unsigned int>("elesisEB");

            bool passMuonVeto = false;
            bool passEleVeto = false;

            try
            {
                const bool& passMuonVetoTmp  = tr.getVar<bool>("passMuonVeto");
                const bool& passEleVetoTmp   = tr.getVar<bool>("passEleVeto");
                if(&passMuonVetoTmp != nullptr) passMuonVeto = passMuonVetoTmp;
                if(&passEleVetoTmp != nullptr) passEleVeto = passEleVetoTmp;
            }
            catch(const std::string e)
            {
                //std::cout << "void muInfo(NTupleReader& tr): Caught exception, variable \"" << e << "\" not found" << std::endl;
            }
            
            std::vector<const TLorentzVector*>* genMatchIsoElecInAcc = new std::vector<const TLorentzVector*>();
            std::vector<const TLorentzVector*>* genMatchElecInAcc = new std::vector<const TLorentzVector*>();
            std::vector<double>* genMatchElecInAccRes = new std::vector<double>();
            std::vector<const TLorentzVector*>* genElecInAcc = new std::vector<const TLorentzVector*>();
            std::vector<const TLorentzVector*>* genElec = new std::vector<const TLorentzVector*>();
            std::vector<double>* genMatchIsoElecInAccAct = new std::vector<double>();
            std::vector<double>* genMatchElecInAccAct = new std::vector<double>();
            std::vector<double>* genElecInAccAct = new std::vector<double>();
            std::vector<double>* genElecAct = new std::vector<double>();

            std::vector<TLorentzVector>* cutElecVec = new std::vector<TLorentzVector>();
            std::vector<double>* cutElecCharge = new std::vector<double>();
            std::vector<double>* cutElecActivity = new std::vector<double>();

            std::vector<TLorentzVector> cutElecVecRecoOnly;

            std::vector<const TLorentzVector*>* genMatchIsoMuInAcc = new std::vector<const TLorentzVector*>();
            std::vector<const TLorentzVector*>* genMatchMuInAcc = new std::vector<const TLorentzVector*>();
            std::vector<double>* genMatchMuInAccRes = new std::vector<double>();
            std::vector<const TLorentzVector*>* genMuInAcc = new std::vector<const TLorentzVector*>();
            std::vector<const TLorentzVector*>* genMu = new std::vector<const TLorentzVector*>();
            std::vector<double>* genMatchIsoMuInAccAct = new std::vector<double>();
            std::vector<double>* genMatchMuInAccAct = new std::vector<double>();
            std::vector<double>* genMuInAccAct = new std::vector<double>();
            std::vector<double>* genMuAct = new std::vector<double>();
            std::vector<TLorentzVector>* cutMuVec = new std::vector<TLorentzVector>();
            std::vector<double>* cutMuCharge = new std::vector<double>();
            std::vector<double>* cutMuActivity = new std::vector<double>();

            std::vector<TLorentzVector> cutMuVecRecoOnly;

            //muon selections 
            int sumMuCharge = 0;
            int nTriggerMuons = 0;
            for(int i = 0; i < muonsLVec.size(); ++i)
            {
                if(AnaFunctions::passMuon( muonsLVec[i], 0.0, 0.0, muonsFlagIDVec[i], AnaConsts::muonsMiniIsoArr)) // emulates muons with pt but no iso requirements.  
                {
                    cutMuVecRecoOnly.push_back(muonsLVec[i]);
                }
                if(AnaFunctions::passMuon( muonsLVec[i], muonsMiniIso[i], 0.0, muonsFlagIDVec[i], AnaConsts::muonsMiniIsoArr))
                {
                    if(AnaFunctions::passMuon( muonsLVec[i], muonsRelIso[i], 0.0, muonsFlagIDVec[i], AnaConsts::muonsMiniIsoArr)) 
                    {
                        if(nTriggerMuons == 0 && muonsLVec[i].Pt() > 17)  nTriggerMuons++;
                        else if(muonsLVec[i].Pt() > 8)  nTriggerMuons++;
                    }
                    cutMuVec->push_back(muonsLVec[i]);
                    cutMuCharge->push_back(muonsCharge[i]);
                    cutMuActivity->push_back(muonspfActivity[i]);
                    if(muonsCharge[i] > 0) sumMuCharge++;
                    else                   sumMuCharge--;
                }
            }

            //electron selection
            int sumElecCharge = 0;
            for(int i = 0; i < elesLVec.size(); ++i)
            {
                if(AnaFunctions::passElectron(elesLVec[i], 0.0, -1, elesisEB[i], elesFlagIDVec[i], AnaConsts::elesMiniIsoArr)) // emulates muons with pt but no iso requirements.  
                {
                    cutElecVecRecoOnly.push_back(elesLVec[i]);
                }

                if(AnaFunctions::passElectron(elesLVec[i], elesMiniIso[i], -1, elesisEB[i], elesFlagIDVec[i], AnaConsts::elesMiniIsoArr))
                {
                    cutElecVec->push_back(elesLVec[i]);
                    cutElecCharge->push_back(elesCharge[i]);
                    cutElecActivity->push_back(AnaFunctions::getElectronActivity(elesLVec[i], jetsLVec, recoJetschargedHadronEnergyFraction, AnaConsts::elesAct));
                    if(elesCharge[i] > 0) sumElecCharge++;
                    else                  sumElecCharge--;
                }
            }        


            //mu45 non-iso trigger emulation 
            const double effsnom2012ABC[] = {0.928,0.8302,0.8018};
            const double upedge2012ABC[] = { 0.9, 1.2, 2.1};
            bool muTrigMu45 = false;
            for(TLorentzVector& mu : *cutMuVec)
            {
                if(mu.Pt() > 45)
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

            const double   minMuPt = 20.0,   highMuPt = 45.0;
            const double minElecPt = 33.0, highElecPt = 33.0;
            double nuPt1 = -999.9, nuPt2 = -999.9;

            //Gen info parsing 
            if(&genDecayPdgIdVec != nullptr && &genDecayLVec != nullptr)
            {
                for(int i = 0; i < genDecayPdgIdVec.size() && i < genDecayLVec.size(); ++i)
                {
                    if((abs(genDecayPdgIdVec[i]) != 0 &&  abs(genDecayPdgIdVec[i]) < 6) || (abs(genDecayPdgIdVec[i]) > 100 && abs(genDecayPdgIdVec[i]) < 10000)) genHt += genDecayLVec[i].Pt();

                    if(genDecayPdgIdVec[i] ==  13) nuPt1 = genDecayLVec[i].Pt();
                    if(genDecayPdgIdVec[i] == -13) nuPt2 = genDecayLVec[i].Pt();
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
                            genMuInAcc->push_back(&genDecayLVec[i]);
                            genMuInAccAct->push_back(genMuAct->back());
                            double dRMin = 999.9;
                            double matchPt = -999.9;
                            for(int j = 0; j < cutMuVecRecoOnly.size(); ++j)
                            {
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
                            for(int j = 0; j < cutMuVec->size(); ++j)
                            {
                                double dR = ROOT::Math::VectorUtil::DeltaR(genDecayLVec[i], (*cutMuVec)[j]);
                                if(dR < dRMin)
                                {
                                    dRMin = dR;
                                }
                            }
                            if(dRMin < 0.02)
                            {
                                genMatchIsoMuInAcc->push_back(&genDecayLVec[i]);
                                genMatchIsoMuInAccAct->push_back(genMuAct->back());
                            }
                        }
                    }

                    //Elec efficiency and acceptance
                    if(abs(genDecayPdgIdVec[i]) == 11)
                    {
                        genElec->push_back(&genDecayLVec[i]);
                        genElecAct->push_back(W_emu_pfActivityVec[index]);
                        if(AnaFunctions::passElectronAccOnly(genDecayLVec[i], AnaConsts::elesMiniIsoArr) && genDecayLVec[i].Pt() > 20)
                        {
                            genElecInAcc->push_back(&genDecayLVec[i]);
                            genElecInAccAct->push_back(genElecAct->back());
                            double dRMin = 999.9;
                            double matchPt = -999.9;
                            for(int j = 0; j < cutElecVecRecoOnly.size(); ++j)
                            {
                                double dR = ROOT::Math::VectorUtil::DeltaR(genDecayLVec[i], cutElecVecRecoOnly[j]);
                                if(dR < dRMin)
                                {
                                    dRMin = dR;
                                    matchPt = cutElecVecRecoOnly[j].Pt();
                                }
                            }
                            if(dRMin < 0.02)
                            {
                                genMatchElecInAcc->push_back(&genDecayLVec[i]);
                                genMatchElecInAccAct->push_back(genElecAct->back());
                                genMatchElecInAccRes->push_back((genDecayLVec[i].Pt() - matchPt)/genDecayLVec[i].Pt());
                            }
                    
                            dRMin = 999.9;
                            for(int j = 0; j < cutElecVec->size(); ++j)
                            {
                                double dR = ROOT::Math::VectorUtil::DeltaR(genDecayLVec[i], (*cutElecVec)[j]);
                                if(dR < dRMin)
                                {
                                    dRMin = dR;
                                }
                            }
                            if(dRMin < 0.02)
                            {
                                genMatchIsoElecInAcc->push_back(&genDecayLVec[i]);
                                genMatchIsoElecInAccAct->push_back(genElecAct->back());
                            }
                        }
                    }
                }
            }

            double genZPt = -999.9, genZEta = -999.9, genZmass = -999.9, genZPhi;
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
        
            const double zMassMin = 71.0;
            const double zMass    = 91.0;
            const double zMassMax = 111.0;

            double zMuMassCurrent = 1.0e300, zEff = 1.0e100, zAcc = 1.0e100;
            TLorentzVector bestRecoMuZ;
            for(int i = 0; i < cutMuVec->size(); ++i)
            {
                if((*cutMuVec)[i].Pt() < minMuPt) continue;
                for(int j = 0; j < i && j < cutMuVec->size(); ++j)
                {
                    if((*cutMuVec)[j].Pt() < minMuPt) continue;
                    double zm = ((*cutMuVec)[i] + (*cutMuVec)[j]).M();
                    //if(zm > zMassMin && zm < zMassMax && fabs(zm - zMass) < fabs(zMassCurrent - zMass))
                    if(fabs(zm - zMass) < fabs(zMuMassCurrent - zMass))
                    {
                        bestRecoMuZ = (*cutMuVec)[i] + (*cutMuVec)[j];
                        zMuMassCurrent = zm;
                    }
                }
            }

            double zElecMassCurrent = 1.0e300;
            TLorentzVector bestRecoElecZ;
            for(int i = 0; i < cutElecVec->size(); ++i)
            {
                if((*cutElecVec)[i].Pt() < minElecPt) continue;
                for(int j = 0; j < i && j < cutElecVec->size(); ++j)
                {
                    if((*cutElecVec)[j].Pt() < minElecPt) continue;
                    double zm = ((*cutElecVec)[i] + (*cutElecVec)[j]).M();
                    //if(zm > zMassMin && zm < zMassMax && fabs(zm - zMass) < fabs(zMassCurrent - zMass))
                    if(fabs(zm - zMass) < fabs(zElecMassCurrent - zMass))
                    {
                        bestRecoElecZ = (*cutElecVec)[i] + (*cutElecVec)[j];
                        zElecMassCurrent = zm;
                    }
                }
            }

            double zElMuMassCurrent = 1.0e300;
            TLorentzVector bestRecoElMuZ;
            for(int i = 0; i < cutMuVec->size(); ++i)
            {
                if((*cutMuVec)[i].Pt() < minMuPt) continue;
                for(int j = 0; j < cutElecVec->size(); ++j)
                {
                    if((*cutElecVec)[j].Pt() < minMuPt) continue;
                    double zm = ((*cutMuVec)[i] + (*cutElecVec)[j]).M();
                    //if(zm > zMassMin && zm < zMassMax && fabs(zm - zMass) < fabs(zMassCurrent - zMass))
                    if(fabs(zm - zMass) < fabs(zElMuMassCurrent - zMass))
                    {
                        bestRecoElMuZ = (*cutMuVec)[i] + (*cutElecVec)[j];
                        zElMuMassCurrent = zm;
                    }
                }
            }

            TLorentzVector metV, metZ;
            metV.SetPtEtaPhiM(met, 0.0, metphi, 0.0);

            TLorentzVector bestRecoZ = (fabs(bestRecoElecZ.M() - zMass) > fabs(bestRecoMuZ.M() - zMass))?(bestRecoMuZ):(bestRecoElecZ);
            if(fabs(bestRecoZ.M() - zMass) > fabs(bestRecoElMuZ.M() - zMass)) bestRecoZ = bestRecoElMuZ;
        
            metZ.SetPtEtaPhiM(bestRecoZ.Pt(), 0.0, bestRecoZ.Phi(), 0.0);
            TLorentzVector cleanMet = metV + metZ;

            bool passMuZinvSel   =  passEleVeto && (cutMuVec->size() == 2   && sumMuCharge == 0   && (*cutMuVec)[0].Pt() > highMuPt     && (*cutMuVec)[1].Pt() > minMuPt)     && (bestRecoMuZ.M() > zMassMin)   && (bestRecoMuZ.M() < zMassMax);
            bool passElecZinvSel = passMuonVeto && (cutElecVec->size() == 2 && sumElecCharge == 0 && (*cutElecVec)[0].Pt() > highElecPt && (*cutElecVec)[1].Pt() > minElecPt) && (bestRecoElecZ.M() > zMassMin) && (bestRecoElecZ.M() < zMassMax);
            bool passElMuZinvSel = (cutMuVec->size() == 1 && cutElecVec->size() == 1 && sumElecCharge == -sumMuCharge && (*cutMuVec)[0].Pt() > highMuPt && (*cutElecVec)[0].Pt() > minMuPt) && (bestRecoElMuZ.M() > zMassMin) && (bestRecoElMuZ.M() < zMassMax);

            double cutMuPt1 = -999.9;
            double cutMuPt2 = -999.9;
            if(cutMuVec->size() >= 1) cutMuPt1 = cutMuVec->at(0).Pt();
            if(cutMuVec->size() >= 2) cutMuPt2 = cutMuVec->at(1).Pt();

            double mindPhiMetJ = 999.9;
            int jc = 0;
            for(const TLorentzVector& jet : jetsLVec)
            {
                if(jc >= 3) break;
                jc++;
                mindPhiMetJ = std::min(mindPhiMetJ, fabs(ROOT::Math::VectorUtil::DeltaPhi(genZ, jet)));
            }
            //if(genZPt > 600) std::cout << "HELLO THERE!!!!" << std::endl;
            //if(genZPt > 600 && mindPhiMetJ < 0.5) std::cout << "BONJOUR!!! \t" << genZPt << "\t" << mindPhiMetJ << "\t" << run << "\t" << lumi << "\t" << event << std::endl;

            double bestRecoZPt = bestRecoZ.Pt();
            double cleanMetPt = cleanMet.Pt();
            //double cleanMet2Pt = cleanMet2.Pt();
            tr.registerDerivedVar("bestRecoZPt", bestRecoZPt);
            tr.registerDerivedVar("bestRecoZM", bestRecoZ.M());
            tr.registerDerivedVar("cleanMetPt", cleanMetPt);
            tr.registerDerivedVar("cleanMetPhi", cleanMet.Phi());
            //tr.registerDerivedVar("cleanMet2Pt", cleanMet2Pt);
            tr.registerDerivedVar("genHt", genHt);
            tr.registerDerivedVar("cutMuPt1", cutMuPt1);
            tr.registerDerivedVar("cutMuPt2", cutMuPt2);
            tr.registerDerivedVar("mindPhiMetJ", mindPhiMetJ);

            tr.registerDerivedVar("ZPtRes", (bestRecoZPt - genZPt)/genZPt);
            tr.registerDerivedVar("ZEtaRes", bestRecoZ.Eta() - genZEta);
            tr.registerDerivedVar("ZPhiRes", bestRecoZ.Phi() - genZPhi);
            tr.registerDerivedVar("ZMRes", (bestRecoZ.M() - genZmass)/genZmass);

            tr.registerDerivedVec("cutMuVec", cutMuVec);
            tr.registerDerivedVec("cutElecVec", cutElecVec);
            tr.registerDerivedVec("cutMuActivity", cutMuActivity);
            tr.registerDerivedVec("cutElecActivity", cutElecActivity);
            tr.registerDerivedVec("genMu", genMu);
            tr.registerDerivedVar("ngenMu", static_cast<double>(genMu->size()));
            tr.registerDerivedVec("genMuInAcc", genMuInAcc);
            tr.registerDerivedVec("genMuAct", genMuAct);
            tr.registerDerivedVar("ngenMuInAcc", static_cast<double>(genMuInAcc->size()));
            tr.registerDerivedVec("genMuInAccAct", genMuInAccAct);
            tr.registerDerivedVec("genMatchMuInAcc", genMatchMuInAcc);
            tr.registerDerivedVec("genMatchMuInAccRes", genMatchMuInAccRes);
            tr.registerDerivedVec("genMatchIsoMuInAcc", genMatchIsoMuInAcc);
            tr.registerDerivedVar("ngenMatchMuInAcc", static_cast<double>(genMatchMuInAcc->size()));
            tr.registerDerivedVec("genMatchMuInAccAct", genMatchMuInAccAct);
            tr.registerDerivedVec("genMatchIsoMuInAccAct", genMatchIsoMuInAccAct);
        
            tr.registerDerivedVec("genElec", genElec);
            tr.registerDerivedVar("ngenElec", static_cast<double>(genElec->size()));
            tr.registerDerivedVec("genElecInAcc", genElecInAcc);
            tr.registerDerivedVec("genElecAct", genElecAct);
            tr.registerDerivedVar("ngenElecInAcc", static_cast<double>(genElecInAcc->size()));
            tr.registerDerivedVec("genElecInAccAct", genElecInAccAct);
            tr.registerDerivedVec("genMatchElecInAcc", genMatchElecInAcc);
            tr.registerDerivedVec("genMatchElecInAccRes", genMatchElecInAccRes);
            tr.registerDerivedVec("genMatchIsoElecInAcc", genMatchIsoElecInAcc);
            tr.registerDerivedVar("ngenMatchElecInAcc", static_cast<double>(genMatchElecInAcc->size()));
            tr.registerDerivedVec("genMatchElecInAccAct", genMatchElecInAccAct);
            tr.registerDerivedVec("genMatchIsoElecInAccAct", genMatchIsoElecInAccAct);

            tr.registerDerivedVar("genZPt", genZPt);
            tr.registerDerivedVar("genZEta", genZEta);
            tr.registerDerivedVar("genZmass", genZmass);
            tr.registerDerivedVar("pdgIdZDec", pdgIdZDec);
            tr.registerDerivedVar("passDiMuIsoTrig", passDiMuTrig);
            tr.registerDerivedVar("passSingleMu45", muTrigMu45);
        
            tr.registerDerivedVar("passMuZinvSel", passMuZinvSel);
            tr.registerDerivedVar("passElecZinvSel", passElecZinvSel);
            tr.registerDerivedVar("passElMuZinvSel", passElMuZinvSel);
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

    void fakebtagvectors(NTupleReader& tr)
    {
        const std::vector<double>& cleanJetpt30ArrBTag = tr.getVec<double>("recoJetsBtag_forTaggerZinv");

        double maxCSV = 0.0;
        double secCSV = 0.0;
        double tenCSV = 0.0;
        int iMaxCSV = -1;
        int iSecCSV = -1;
        int iTenCSV = -1;

        //find index of 3 highest CSV values
        for(int i = 0; i < cleanJetpt30ArrBTag.size(); ++i)
        {
            if(cleanJetpt30ArrBTag[i] > maxCSV)
            {
                tenCSV = secCSV;
                secCSV = maxCSV;
                maxCSV = cleanJetpt30ArrBTag[i];
                iTenCSV = iSecCSV;
                iSecCSV = iMaxCSV;
                iMaxCSV = i;
            }
            else if(cleanJetpt30ArrBTag[i] > secCSV)
            {
                tenCSV = secCSV;
                secCSV = cleanJetpt30ArrBTag[i];
                iTenCSV = iSecCSV;
                iSecCSV = i;
            }
            else if(cleanJetpt30ArrBTag[i] > tenCSV)
            {
                tenCSV = cleanJetpt30ArrBTag[i];
                iTenCSV = i;
            }
        }

        std::vector<double>* cleanJetpt30ArrBTag1fake = new std::vector<double>(cleanJetpt30ArrBTag);
        std::vector<double>* cleanJetpt30ArrBTag2fake = new std::vector<double>(cleanJetpt30ArrBTag);
        std::vector<double>* cleanJetpt30ArrBTag3fake = new std::vector<double>(cleanJetpt30ArrBTag);
        std::vector<double>* fakedCSVValues = new std::vector<double>();

        if(iMaxCSV >= 0) (*cleanJetpt30ArrBTag1fake)[iMaxCSV] = 0.99;

        if(iMaxCSV >= 0) (*cleanJetpt30ArrBTag2fake)[iMaxCSV] = 0.99;
        if(iSecCSV >= 0) (*cleanJetpt30ArrBTag2fake)[iSecCSV] = 0.99;

        if(iMaxCSV >= 0) (*cleanJetpt30ArrBTag3fake)[iMaxCSV] = 0.99;
        if(iSecCSV >= 0) (*cleanJetpt30ArrBTag3fake)[iSecCSV] = 0.99;
        if(iTenCSV >= 0) (*cleanJetpt30ArrBTag3fake)[iTenCSV] = 0.99;

        if(iMaxCSV >= 0) fakedCSVValues->push_back(maxCSV);
        if(iSecCSV >= 0) fakedCSVValues->push_back(secCSV);
        if(iTenCSV >= 0) fakedCSVValues->push_back(tenCSV);

        //Calculate the combinatoric weights for b-jet faking
        double weight1fakeb = TMath::Binomial(cleanJetpt30ArrBTag.size(), 1);
        double weight2fakeb = TMath::Binomial(cleanJetpt30ArrBTag.size(), 2);
        double weight3fakeb = TMath::Binomial(cleanJetpt30ArrBTag.size(), 3);
        //check for nans
        if(weight1fakeb != weight1fakeb) weight1fakeb = 0.0;
        if(weight2fakeb != weight2fakeb) weight2fakeb = 0.0;
        if(weight3fakeb != weight3fakeb) weight3fakeb = 0.0;
        
        tr.registerDerivedVar("weight1fakeb", weight1fakeb);
        tr.registerDerivedVar("weight2fakeb", weight2fakeb);
        tr.registerDerivedVar("weight3fakeb", weight3fakeb);

        tr.registerDerivedVec("cleanJetpt30ArrBTag1fake", cleanJetpt30ArrBTag1fake);
        tr.registerDerivedVec("cleanJetpt30ArrBTag2fake", cleanJetpt30ArrBTag2fake);
        tr.registerDerivedVec("cleanJetpt30ArrBTag3fake", cleanJetpt30ArrBTag3fake);
        tr.registerDerivedVec("fakedCSVValues", fakedCSVValues);
        tr.registerDerivedVar("maxCSV", maxCSV);
        
    }

    void getSearchBin(NTupleReader& tr)
    {
        const int& cntCSVS = tr.getVar<int>("cntCSVSZinv");
        const int& nTopCandSortedCnt = tr.getVar<int>("nTopCandSortedCntZinv");
        const int& nTopCandSortedCnt1b = tr.getVar<int>("nTopCandSortedCntZinv1b");
        const int& nTopCandSortedCnt2b = tr.getVar<int>("nTopCandSortedCntZinv2b");
        const int& nTopCandSortedCnt3b = tr.getVar<int>("nTopCandSortedCntZinv3b");
        const double& cleanMet = tr.getVar<double>("cleanMetPt");
        const double& cleanMetPhi = tr.getVar<double>("cleanMetPhi");
        const double& MT2 = tr.getVar<double>("best_had_brJet_MT2Zinv");
        const double& MT2_1b = tr.getVar<double>("best_had_brJet_MT2Zinv1b");
        const double& MT2_2b = tr.getVar<double>("best_had_brJet_MT2Zinv2b");
        const double& MT2_3b = tr.getVar<double>("best_had_brJet_MT2Zinv3b");
        const double& weight1fakeb = tr.getVar<double>("weight1fakeb");
        const double& weight2fakeb = tr.getVar<double>("weight2fakeb");
        const double& weight3fakeb = tr.getVar<double>("weight3fakeb");

        int nSearchBin = find_Binning_Index(cntCSVS, nTopCandSortedCnt, MT2, cleanMet);

        std::vector<std::pair<double, double> > * nb0Bins = new std::vector<std::pair<double, double> >();
        std::vector<double> * nb0BinsNW = new std::vector<double>();

        //weights based on total N(b) yields vs. N(b) = 0 control region
        //These weights are derived from the rato of events in the N(t) = 1, 2, 3 bins after all baseline cuts except b tag between the 
        //N(b) = 0 control region and each N(b) signal region using Z->nunu MC.  They account for both the combinatoric reweighting factor
        //as well as the different event yields between the control region and each signal region.  
        const double wnb01 = 3.478840e-02;//6.26687e-2;
        const double wnb02 = 2.586369e-03;//5.78052e-3;
        const double wnb03 = 1.640077e-04;//7.08235e-4;

	// weights to apply when doing b-faking
	const double w1b = wnb01 * weight1fakeb;
	const double w2b = wnb02 * weight2fakeb;
	const double w3b = wnb03 * weight3fakeb;
	
        if(cntCSVS == 0)
        {
            //nb0Bins->push_back(std::make_pair(find_Binning_Index(0, nTopCandSortedCnt, MT2, cleanMet), 1.0));
            nb0Bins->emplace_back(std::pair<double, double>(find_Binning_Index(1, nTopCandSortedCnt1b, MT2_1b, cleanMet), wnb01 * weight1fakeb));
            nb0Bins->emplace_back(std::pair<double, double>(find_Binning_Index(2, nTopCandSortedCnt2b, MT2_2b, cleanMet), wnb02 * weight2fakeb));
            nb0Bins->emplace_back(std::pair<double, double>(find_Binning_Index(3, nTopCandSortedCnt3b, MT2_3b, cleanMet), wnb03 * weight3fakeb));

            nb0BinsNW->emplace_back(find_Binning_Index(1, nTopCandSortedCnt1b, MT2_1b, cleanMet));
            nb0BinsNW->emplace_back(find_Binning_Index(2, nTopCandSortedCnt2b, MT2_2b, cleanMet));
            nb0BinsNW->emplace_back(find_Binning_Index(3, nTopCandSortedCnt3b, MT2_3b, cleanMet));
        }

        tr.registerDerivedVar("nSearchBin", nSearchBin);
        tr.registerDerivedVec("nb0BinsNW", nb0BinsNW);
        tr.registerDerivedVec("nb0Bins", nb0Bins);
	tr.registerDerivedVar("weight1fakebComb", w1b);
	tr.registerDerivedVar("weight2fakebComb", w2b);
	tr.registerDerivedVar("weight3fakebComb", w3b);
    }

    void printInterestingEvents(NTupleReader& tr)
    {
        const unsigned int& run   = tr.getVar<unsigned int>("run");
        const unsigned int& event = tr.getVar<unsigned int>("event");

        const double& met                            = tr.getVar<double>("met");
        const double& metphi                         = tr.getVar<double>("metphi");

        const int& nMuons_CUT        = tr.getVar<int>("nMuons_CUT");
        const int& nElectrons_CUT    = tr.getVar<int>("nElectrons_CUT");
        const int& cntNJetsPt50Eta24 = tr.getVar<int>("cntNJetsPt50Eta24");
        const int& cntNJetsPt30Eta24 = tr.getVar<int>("cntNJetsPt30Eta24");
        const int& cntNJetsPt30      = tr.getVar<int>("cntNJetsPt30");

        const double& mht    = tr.getVar<double>("mht");
        const double& mhtphi = tr.getVar<double>("mhtphi");
        const double& ht     = tr.getVar<double>("ht");

        //if(met > 1000) std::cout << "run: " << run << "\tevent: " << event << "\tmet: " << met << "\tmetphi: " << metphi << "\tnMuons_CUT: " << nMuons_CUT << "\t nElectrons_CUT: " << nElectrons_CUT << "\tcntNJetsPt30: " << cntNJetsPt30 << "\tcntNJetsPt30Eta24: " << cntNJetsPt30Eta24 << "\tcntNJetsPt50Eta24: " << cntNJetsPt50Eta24 << "\tmht: " << mht << "\tmhtphi: " << mhtphi << "\tht: " << ht << std::endl;
    }
    
    class RegisterFunctions
    {
    private:
        BaselineVessel blvZinv;
        BaselineVessel blvZinv1b;
        BaselineVessel blvZinv2b;
        BaselineVessel blvZinv3b;
        GenerateWeight weights;
        LepInfo lepInfo;

    public:
        RegisterFunctions() : blvZinv("Zinv"), blvZinv1b("Zinv1b"), blvZinv2b("Zinv2b"), blvZinv3b("Zinv3b")
        {            
        }
        
        void registerFunctions(NTupleReader& tr)
        {
            //Make some global "constants" here

            //register functions with NTupleReader
            tr.registerFunction(lepInfo);
            tr.registerFunction(weights);
            tr.registerFunction(blvZinv);
            tr.registerFunction(&fakebtagvectors);
            tr.registerFunction(blvZinv1b);
            tr.registerFunction(blvZinv2b);
            tr.registerFunction(blvZinv3b);
            tr.registerFunction(&getSearchBin);
            //tr.registerFunction(&printInterestingEvents);
        }
    };

    void activateBranches(std::set<std::string>& activeBranches)
    {
        for(auto& bn : AnaConsts::activatedBranchNames) activeBranches.insert(bn);
        for(auto& bn : AnaConsts::activatedBranchNames_DataOnly) activeBranches.insert(bn);
        activeBranches.insert("ht");
        activeBranches.insert("run");
        activeBranches.insert("lumi");
        activeBranches.insert("event");
        activeBranches.insert("mht");
        activeBranches.insert("mhtphi");
        activeBranches.insert("genDecayPdgIdVec");
        activeBranches.insert("genDecayLVec");
        activeBranches.insert("muonsLVec");
        activeBranches.insert("muonsRelIso");
        activeBranches.insert("muonsMiniIso");
        activeBranches.insert("W_emuVec");
        activeBranches.insert("muonsCharge");
        activeBranches.insert("muonsMtw");
        activeBranches.insert("met");
        activeBranches.insert("metphi");
        activeBranches.insert("jetsLVec");
        activeBranches.insert("elesLVec");
        activeBranches.insert("elesRelIso");
        activeBranches.insert("recoJetsBtag_0");
        activeBranches.insert("loose_isoTrks_mtw");
        activeBranches.insert("elesMtw");
        activeBranches.insert("loose_isoTrks_iso");
        activeBranches.insert("loose_isoTrksLVec");
        activeBranches.insert("muMatchedJetIdx");
        activeBranches.insert("eleMatchedJetIdx");
        activeBranches.insert("recoJetschargedEmEnergyFraction");
        activeBranches.insert("recoJetsneutralEmEnergyFraction");
        activeBranches.insert("recoJetschargedHadronEnergyFraction");
        activeBranches.insert("prodJetsNoMu_recoJetschargedEmEnergyFraction");
        activeBranches.insert("prodJetsNoMu_recoJetsneutralEmEnergyFraction");
        activeBranches.insert("prodJetsNoMu_recoJetschargedHadronEnergyFraction");
        activeBranches.insert("elesisEB");
        activeBranches.insert("elesMiniIso");
        activeBranches.insert("elesCharge");
    }
}
