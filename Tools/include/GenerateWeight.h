#ifndef GENERATEWEIGHT_H
#define GENERATEWEIGHT_H

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

#include "TopTagger/TopTagger/include/TopObject.h"

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
    class GenerateWeight
    {
    private:
        TH1* muEff;
        TH1* muEffReco;
        TH2* muEffIso;
        TH2* elecEffReco;
        TH2* elecEffIso;
        TH2* muAcc;
        TH1* hZEff;
        TH1* hZAcc;
        TH1* hZAccElec;
        TH1* hZAccLepPt;
        TH1* hZAccLepPtElec;

        void generateWeight(NTupleReader& tr)
        {
            const auto& jetsLVec        = tr.getVec<TLorentzVector>("prodJetsNoLep_jetsLVec");
            const auto& cutMuVec        = tr.getVec<TLorentzVector>("cutMuVec");
            const auto& cutElecVec      = tr.getVec<TLorentzVector>("cutElecVec");
            const auto& cutMuActivity   = tr.getVec<data_t>("cutMuActivity");
            const auto& cutElecActivity = tr.getVec<data_t>("cutElecActivity");
            const auto& genMu           = tr.getVec<const TLorentzVector*>("genMu");
            //const auto& genMuInAcc      = tr.getVec<const TLorentzVector*>("genMuInAcc");
            const auto& genMuInAcc      = tr.getVec<TLorentzVector>("genMuInAcc");
            const auto& genMatchMuInAcc = tr.getVec<const TLorentzVector*>("genMatchMuInAcc");

            const auto& pdgIdZDec       = tr.getVar<int>("pdgIdZDec");
            const auto& passMuZinvSel   = tr.getVar<bool>("passMuZinvSel");
            const auto& passElecZinvSel = tr.getVar<bool>("passElecZinvSel");
            const auto& ht              = tr.getVar<data_t>("genHT");
            const auto& bestRecoZPt     = tr.getVar<data_t>("bestRecoZPt");
            const auto& genZPt          = tr.getVar<data_t>("genZPt");

            //const auto& nJets           =  tr.getVar<int>("nJets"); // variable not produced in CMSSW8028_2016 ntuples
            const auto& nJets           =  tr.getVar<int>("nJets_CUT");
            const auto& stored_weight   = tr.getVar<data_t>("stored_weight");

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
                        data_t mu1Act = cutMuActivity[i];
                        data_t mu2Act = cutMuActivity[j];

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
                            //const douBLE p0 =  9.83467e-01; // +/- 1.54469e+00
                            //const double p1 = -7.81897e-06; // +/- 4.16487e-03
                            //const double p2 = -1.22092e-08; // +/- 2.18556e-06
                            //Spring15 low stats
                            //const double fitStart = 200.0;  // extended to 2000 GeV
                            //const double p0 =     0.979238; //   +/-   1.17313
                            //const double p1 = -6.47338e-06; //   +/-   0.00342715
                            //const double p2 = -1.16258e-08; //   +/-   1.87837e-06
                            //Spring15 extended samples
                            const double fitStart = 200.0;  // extended to 2000 GeV
                            const double p0 =      0.97431; //   +/-   1.17119     
                            const double p1 =  2.87484e-05; //   +/-   0.00341371  
                            const double p2 = -5.37058e-08; //   +/-   1.86309e-06 

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

                        data_t elec1Act = cutElecActivity[i];
                        data_t elec2Act = cutElecActivity[j];

                        //set to not overflow histograms
                        if(elec1pt >= 2000.0) elec1pt = 1999.9;
                        if(elec2pt >= 2000.0) elec2pt = 1999.9;

                        //Get elec efficiencies
                        double elecEff1 = 0.0, elecEff2 = 0.0;
                        if(elecEffReco && elecEffIso)
                        {
                            //Fit to iso eff (eff = p0 + p1*pt + p2*pt^2
                            const double fitStart =200.0;  // extended to 2000 GeV
                            const double p0 =     0.989411; //   +/-   1.18467
                            const double p1 =  3.66321e-06; //   +/-   0.00346729
                            const double p2 = -5.68292e-09; //   +/-   1.90334e-06

                            double elecIsoEff = 0.0;

                            //int isoPtBin = elecEffIso->GetXaxis()->FindBin(elec1pt);
                            //if(elec1pt > fitStart) elecIsoEff = p0 + p1*elec1pt + p2*elec1pt*elec1pt;
                            //else                   elecIsoEff = elecEffIso->GetBinContent(isoPtBin);

                            int recoPtBin = elecEffReco->GetXaxis()->FindBin(elec1pt);
                            if(recoPtBin >= elecEffReco->GetNbinsX()) recoPtBin = elecEffReco->GetNbinsX();

                            int recoActBin = elecEffReco->GetYaxis()->FindBin(elec1Act);
                            if(recoActBin >= elecEffReco->GetNbinsY()) recoActBin = elecEffReco->GetNbinsY();

                            elecIsoEff = elecEffIso->GetBinContent(recoPtBin, recoActBin);
                            elecEff1 = elecIsoEff * elecEffReco->GetBinContent(recoPtBin, recoActBin);
                            //std::cout << elec1pt << "\t" << recoPtBin << "\t" <<  elec1Act << "\t" << recoActBin << "\t" << elecEff1 << "\t" << elecIsoEff << "\t" << elecEffReco->GetBinContent(recoPtBin, recoActBin) << std::endl;

                            elecIsoEff = 0.0;
                            //isoPtBin = elecEffIso->GetXaxis()->FindBin(elec2pt);
                            //if(elec2pt > fitStart) elecIsoEff = p0 + p1*elec2pt + p2*elec2pt*elec2pt;
                            //else                   elecIsoEff = elecEffIso->GetBinContent(isoPtBin);

                            recoPtBin = elecEffReco->GetXaxis()->FindBin(elec2pt);
                            if(recoPtBin >= elecEffReco->GetNbinsX()) recoPtBin = elecEffReco->GetNbinsX();

                            recoActBin = elecEffReco->GetYaxis()->FindBin(elec2Act);
                            if(recoActBin >= elecEffReco->GetNbinsY()) recoActBin = elecEffReco->GetNbinsY();

                            elecIsoEff = elecEffIso->GetBinContent(recoPtBin, recoActBin);
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
            for(auto& tlvp : genMuInAcc) if(tlvp.Pt() > 50) genCleanHt -= tlvp.Pt();

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
                //genMudEta = (*(genMu[0])).Eta() - (*(genMu[1])).Eta();
            }
            //std::cout<<"cutMuVec at size of two "<<cutMuVec.size()<<std::endl;
            // muon Z pt acceptance corrections
            //functional form [2] - exp([0] + [1]*x)
            //PHYS14
            //double acc_p0 = -2.91374e-01;
            //double acc_p1 = -4.42884e-03;
            //double acc_p2 =  9.51190e-01;
            //Spring15 low stats
            //double acc_p0 = -2.64921e-01;
            //double acc_p1 = -4.65305e-03;
            //double acc_p2 =  9.50493e-01;
            //Spring15 extended sample
            const double acc_p0 = -2.60663e-01;
            const double acc_p1 = -4.60623e-03;
            const double acc_p2 = 9.49434e-01;

            if(hZAcc && hZAccLepPt)
            {
                //Eta portion of acceptance
                if(bestRecoZPt < 150) zAcc = hZAcc->GetBinContent(hZAcc->GetXaxis()->FindBin(bestRecoZPt));
                else                  zAcc = acc_p2 - exp(acc_p0 + acc_p1 * bestRecoZPt);

                //pt portion of acceptance
                zAcc *= hZAccLepPt->GetBinContent(hZAccLepPt->GetXaxis()->FindBin(bestRecoZPt));
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

            if(hZAccElec && hZAccLepPtElec)
            {
                //Eta portion of the acceptance
                if(bestRecoZPt < 100) zAccElec = hZAccElec->GetBinContent(hZAccElec->GetXaxis()->FindBin(bestRecoZPt));
                else                  zAccElec = elecAcc_p2 - exp(elecAcc_p0 + elecAcc_p1 * bestRecoZPt);

                //Pt portion of the acceptance
                zAccElec = hZAccLepPtElec->GetBinContent(hZAccLepPtElec->GetXaxis()->FindBin(bestRecoZPt));
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

            // Process the generator weight
            double genWeight = 1.;
            // Never apply this weight for data! In the old ntuple version <=3 this is "-1", in the newer ones it is "0"
            if(stored_weight < 0)
              genWeight = -1.;

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

            tr.registerDerivedVar("genWeight", genWeight);
        }

    public:
        GenerateWeight()
        {
            muEff          = nullptr;
            muEffReco      = nullptr;
            muEffIso       = nullptr;
            muAcc          = nullptr;
            hZEff          = nullptr;
            hZAcc          = nullptr;
            elecEffReco    = nullptr;
            elecEffIso     = nullptr;
            hZAccElec      = nullptr;
            hZAccLepPt     = nullptr;
            hZAccLepPtElec = nullptr;

            TH1::AddDirectory(false);
            TFile *f = new TFile("lepEffHists.root");
            if(f)
            {
                muEff          = static_cast<TH1*>(f->Get("hMuEffPt_ratio"));
                muEffReco      = static_cast<TH1*>(f->Get("hMuEffPtReco_ratio"));
                muEffIso       = static_cast<TH2*>(f->Get("hMuEffPtActIso_ratio"));
                muAcc          = static_cast<TH2*>(f->Get("hMuAcc_ratio"));
                hZEff          = static_cast<TH1*>(f->Get("hZEffPt_ratio"));
                hZAcc          = static_cast<TH1*>(f->Get("hZAccPtSmear_ratio"));
                elecEffReco    = static_cast<TH2*>(f->Get("hElecEffPtActReco_ratio"));
                elecEffIso     = static_cast<TH2*>(f->Get("hElecEffPtActIso_ratio"));
                hZAccElec      = static_cast<TH1*>(f->Get("hZElecAccPtSmear_ratio"));
                hZAccLepPt     = static_cast<TH1*>(f->Get("hZAccPtMuPtSmear_ratio"));
                hZAccLepPtElec = static_cast<TH1*>(f->Get("hZElecAccPtPtSmear_ratio"));
                f->Close();
                delete f;
            }
            else
            {
                std::cout << "Failed to open: lepEffHists.root" << std::endl;
            }
        }

        ~GenerateWeight()
        {
            //if(muEff)          delete muEff;
            //if(muEffReco)      delete muEffReco;
            //if(muEffIso)       delete muEffIso;
            //if(muAcc)          delete muAcc;
            //if(hZEff)          delete hZEff;
            //if(hZAcc)          delete hZAcc;
            //if(elecEffReco)    delete elecEffReco;
            //if(elecEffIso)     delete elecEffIso;
            //if(hZAccElec)      delete hZAccElec;
            //if(hZAccLepPt)     delete hZAccLepPt;
            //if(hZAccLepPtElec) delete hZAccLepPtElec;
        }

        void operator()(NTupleReader& tr)
        {
            generateWeight(tr);
        }
    };
}


#endif
