#ifndef GAMMA_H
#define GAMMA_H

#include "TypeDefinitions.h"
#include "ZInvisible/Tools/PhotonTools.h"
#include "SusyAnaTools/Tools/NTupleReader.h"
#include "SusyAnaTools/Tools/customize.h"
#include "SusyAnaTools/Tools/searchBins.h"
#include "ZInvisible/Tools/ScaleFactors.h"
#include "ZInvisible/Tools/ScaleFactorsttBar.h"
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
    class Gamma {

    private:
        std::string year_;
        bool verbose = false;
        enum ID{Loose, Medium, Tight};

    void generateGamma(NTupleReader& tr) {
        const auto& PhotonTLV              = tr.getVec<TLorentzVector>("PhotonTLV");  // reco photon
        const auto& Photon_jetIdx          = tr.getVec<int>("Photon_jetIdx");
        const auto& Photon_Stop0l          = tr.getVec<unsigned char>("Photon_Stop0l");
        const auto& met                    = tr.getVar<data_t>("MET_pt");
        const auto& metphi                 = tr.getVar<data_t>("MET_phi");
        
        std::vector<TLorentzVector> GenPartTLV;
        std::vector<int> GenPart_genPartIdxMother;
        std::vector<int> GenPart_pdgId;
        std::vector<int> GenPart_status;
        std::vector<int> GenPart_statusFlags;
        // Photon_genPartIdx and Photon_genPartFlav are broken due to GenPart skimming
        //std::vector<int> Photon_genPartIdx;
        //std::vector<unsigned char> Photon_genPartFlav;
        
        // choose ID to use
        enum ID myID = Medium;
        // the scale factors only exist in MC, not in Data
        std::vector<data_t> Photon_SF;
        std::vector<data_t> Photon_SF_Err;
            
        data_t met_jesTotalUp       = 0.0;
        data_t met_jesTotalDown     = 0.0;
        data_t metphi_jesTotalUp    = 0.0;
        data_t metphi_jesTotalDown  = 0.0;
        
        // get gen variables if they exist (MC only)
        bool isData = ! tr.checkBranch("GenPart_pt");
        if (! isData)
        {
            GenPartTLV                  = tr.getVec<TLorentzVector>("GenPartTLV");
            GenPart_genPartIdxMother    = tr.getVec<int>("GenPart_genPartIdxMother");
            GenPart_pdgId               = tr.getVec<int>("GenPart_pdgId");
            GenPart_status              = tr.getVec<int>("GenPart_status");
            GenPart_statusFlags         = tr.getVec<int>("GenPart_statusFlags");
            //Photon_genPartIdx     = tr.getVec<int>("Photon_genPartIdx");
            //Photon_genPartFlav    = tr.getVec<unsigned char>("Photon_genPartFlav");
            met_jesTotalUp              = tr.getVar<data_t>("MET_pt_jesTotalUp");
            met_jesTotalDown            = tr.getVar<data_t>("MET_pt_jesTotalDown");
            metphi_jesTotalUp           = tr.getVar<data_t>("MET_phi_jesTotalUp");
            metphi_jesTotalDown         = tr.getVar<data_t>("MET_phi_jesTotalDown");
            
            // scale factor
            // Loose and Medium SF available; use Medium SF for Medium and Tight ID
            if (myID == Loose)
            {
                Photon_SF       = tr.getVec<data_t>("Photon_LooseSF"); 
                Photon_SF_Err   = tr.getVec<data_t>("Photon_LooseSFErr"); 
            }
            else               
            {
                Photon_SF       = tr.getVec<data_t>("Photon_MediumSF");
                Photon_SF_Err   = tr.getVec<data_t>("Photon_MediumSFErr");
            }
        }

        // --- Photon ID --- //
        // 2016:      Use Photon_cutBased       : Int_t cut-based Spring16-V2p2 ID (0:fail, 1: :loose, 2:medium, 3:tight)
        // 2017,2018: Use Photon_cutBasedBitmap : Int_t cut-based ID bitmap, 2^(0:loose, 1: medium, 2:tight); should be 2017 V2
        std::vector<int>  tempID;
        std::vector<bool> Photon_ID;
        std::vector<bool> Photon_PassLooseID;
        std::vector<bool> Photon_PassMediumID;
        std::vector<bool> Photon_PassTightID;
        // 2016
        if (year_.compare("2016") == 0 || tr.checkBranch("Photon_cutBased"))
        {
            tempID = tr.getVec<int>("Photon_cutBased");
            for (const auto& id : tempID)
            {
                Photon_PassLooseID.push_back(  bool(id > 0) );
                Photon_PassMediumID.push_back( bool(id > 1) );
                Photon_PassTightID.push_back(  bool(id > 2) );
            }
        }
        // 2017, 2018
        else
        {
            tempID = tr.getVec<int>("Photon_cutBasedBitmap");
            for (const auto& id : tempID)
            {
                Photon_PassLooseID.push_back(  bool(id & 1) );
                Photon_PassMediumID.push_back( bool(id & 2) );
                Photon_PassTightID.push_back(  bool(id & 4) );
            }
        }
        if (myID == Loose) 
        {
            Photon_ID = Photon_PassLooseID;
        }
        else if (myID == Medium) 
        {
            Photon_ID = Photon_PassMediumID;
        }
        else
        {
            Photon_ID = Photon_PassTightID;
        }
        // for testing ID selection
        //for (int i = 0; i < tempID.size(); ++i)
        //{
        //    std::cout << "ID = " << tempID[i] << "; (L, M, T) = (" << Photon_PassLooseID[i] << ", " << Photon_PassMediumID[i] << ", " << Photon_PassTightID[i] << "); myID = " << Photon_ID[i] << std::endl;
        //}
        float metWithPhoton                 = -999.9;
        float metWithPhoton_jesTotalUp      = -999.9;
        float metWithPhoton_jesTotalDown    = -999.9;
        float metphiWithPhoton              = -999.9;
        float metphiWithPhoton_jesTotalUp   = -999.9;
        float metphiWithPhoton_jesTotalDown = -999.9;
        float cutPhotonPt                   = -999.9;
        float cutPhotonEta                  = -999.9;
        float photonSF                      = 1.0;
        float photonSF_Up                   = 1.0;
        float photonSF_Down                 = 1.0;
        bool passPhotonSelection            = false;
        bool passPhotonSelectionDirect      = false;
        bool passPhotonSelectionFragmented  = false;
        bool passPhotonSelectionFake        = false;
        bool passQCDSelection               = true;  // default should be true
        
        // if you use new, you need to register it or destroy it yourself to clear memory; otherwise there will be memory leaks
        // use createDerivedVec to avoid this issue 
        auto& GenPartonTLV                  = tr.createDerivedVec<TLorentzVector>("GenPartonTLV"); 
        auto& GenPhotonTLV                  = tr.createDerivedVec<TLorentzVector>("GenPhotonTLV"); 
        auto& GenPhotonTLVEta               = tr.createDerivedVec<TLorentzVector>("GenPhotonTLVEta"); 
        auto& GenPhotonTLVEtaPt             = tr.createDerivedVec<TLorentzVector>("GenPhotonTLVEtaPt"); 
        auto& GenPhotonTLVEtaPtMatched      = tr.createDerivedVec<TLorentzVector>("GenPhotonTLVEtaPtMatched"); 
        auto& RecoPhotonTLV                 = tr.createDerivedVec<TLorentzVector>("RecoPhotonTLV");
        auto& RecoPhotonTLVEta              = tr.createDerivedVec<TLorentzVector>("RecoPhotonTLVEta");
        auto& RecoPhotonTLVEtaPt            = tr.createDerivedVec<TLorentzVector>("RecoPhotonTLVEtaPt");
        auto& RecoPhotonTLVEtaPtMatched     = tr.createDerivedVec<TLorentzVector>("RecoPhotonTLVEtaPtMatched");
        auto& RecoPhotonTLVIso              = tr.createDerivedVec<TLorentzVector>("RecoPhotonTLVIso"); 
        auto& LoosePhotonTLV                = tr.createDerivedVec<TLorentzVector>("LoosePhotonTLV");
        auto& MediumPhotonTLV               = tr.createDerivedVec<TLorentzVector>("MediumPhotonTLV");
        auto& TightPhotonTLV                = tr.createDerivedVec<TLorentzVector>("TightPhotonTLV");
        auto& PromptPhotons                 = tr.createDerivedVec<TLorentzVector>("PromptPhotons");
        auto& DirectPhotons                 = tr.createDerivedVec<TLorentzVector>("DirectPhotons");
        auto& FragmentedPhotons             = tr.createDerivedVec<TLorentzVector>("FragmentedPhotons");
        auto& FakePhotons                   = tr.createDerivedVec<TLorentzVector>("FakePhotons");
        auto& cutPhotonTLV                  = tr.createDerivedVec<TLorentzVector>("cutPhotonTLV");
        auto& cutPhotonJetIndex             = tr.createDerivedVec<int>("cutPhotonJetIndex");
        auto& cutPhotonSF                   = tr.createDerivedVec<float>("cutPhotonSF");
        auto& cutPhotonSF_Up                = tr.createDerivedVec<float>("cutPhotonSF_Up");
        auto& cutPhotonSF_Down              = tr.createDerivedVec<float>("cutPhotonSF_Down");
        auto& dR_GenPhotonGenParton         = tr.createDerivedVec<float>("dR_GenPhotonGenParton");
        auto& dR_RecoPhotonGenParton        = tr.createDerivedVec<float>("dR_RecoPhotonGenParton");
        auto& dR_RecoPhotonGenPhoton        = tr.createDerivedVec<float>("dR_RecoPhotonGenPhoton");
        

        //NanoAOD Gen Particles Ref: https://cms-nanoaod-integration.web.cern.ch/integration/master-102X/mc102X_doc.html#GenPart
        //Particle Status Codes Ref: http://home.thep.lu.se/~torbjorn/pythia81html/ParticleProperties.html
        //Particle ID Numbering Ref: http://pdg.lbl.gov/2018/reviews/rpp2018-rev-monte-carlo-numbering.pdf

        // Determine GenPhotons and GenPartons from GenPart (gen particles) 
        if (! isData)
        {
            for (int i = 0; i < GenPartTLV.size(); ++i)
            {
                int genPartIdxMother    = GenPart_genPartIdxMother[i];
                int pdgId               = GenPart_pdgId[i];
                int status              = GenPart_status[i];
                int statusFlags         = GenPart_statusFlags[i];
                

                // mother particle: default pdgId is 0 which means no mother particle found
                int mother_pdgId        = 0;
                // check that index is in range
                if (genPartIdxMother >= 0 && genPartIdxMother < GenPart_pdgId.size())
                {
                    mother_pdgId        = GenPart_pdgId[genPartIdxMother];
                }
                
                // testing
                //printf("INFO: pdgId = %d, status = %d, statusFlags = %d, genPartIdxMother = %d, mother_pdgId = %d\n", pdgId, status, statusFlags, genPartIdxMother, mother_pdgId);
                
                
                // --- old version for gen partons --- //
                //if ( (abs(pdgId) > 0 && abs(pdgId) < 7) || pdgId == 9 || pdgId == 21 )
                //{
                //    // old version
                //    if (status == 23 && (statusFlags & 0x80 == 0x80) )
                //    {
                //        if(verbose) printf("Found GenParton: pdgId = %d, status = %d, statusFlags = %d, genPartIdxMother = %d, mother_pdgId = %d\n", pdgId, status, statusFlags, genPartIdxMother, mother_pdgId);
                //        GenPartonTLV.push_back(GenPartTLV[i]);
                //    }
                //}
                
                // Particle IDs
                // quarks: +/- (1 to 6)
                // gluons: + (9 and 21)
                // statusFlag: isPrompt; statusFlag & 1 == 1
                // outgoing particles of the hardest subprocess: GenPart_status = 23
                // isHardProcess: GenPart_statusFlags & (0x80) == (0x80)
                
                if ( (abs(pdgId) > 0 && abs(pdgId) < 7) || pdgId == 9 || pdgId == 21 )
                {
                    if (statusFlags & 1 == 1)
                    {
                        if(verbose) printf("Found GenParton: pdgId = %d, status = %d, statusFlags = %d, genPartIdxMother = %d, mother_pdgId = %d\n", pdgId, status, statusFlags, genPartIdxMother, mother_pdgId);
                        GenPartonTLV.push_back(GenPartTLV[i]);
                    }
                }

                // Particle IDs
                // photons: +22
                // stable: status == 1
                // statusFlag: isPrompt; statusFlag & 1 == 1
                
                if (pdgId == 22 && status == 1 && (statusFlags & 1 == 1) )
                {
                    if(verbose) printf("Found GenPhoton: pdgId = %d, status = %d, statusFlags = %d, genPartIdxMother = %d, mother_pdgId = %d\n", pdgId, status, statusFlags, genPartIdxMother, mother_pdgId);
                    GenPhotonTLV.push_back(GenPartTLV[i]);
                }
            }
            //Apply cuts to Gen Photons
            for(int i = 0; i < GenPhotonTLV.size(); ++i)
            {
                // ECAL eta cuts
                if (PhotonFunctions::passPhotonECAL(GenPhotonTLV[i]))
                {
                    GenPhotonTLVEta.push_back(GenPhotonTLV[i]);
                }
                // passing pt and eta cuts
                if (PhotonFunctions::passPhotonEtaPt(GenPhotonTLV[i]))
                {
                    GenPhotonTLVEtaPt.push_back(GenPhotonTLV[i]);
                    // passing ECAL barrel/endcap eta cuts and reco match
                    if (PhotonFunctions::isRecoMatched(GenPhotonTLV[i], PhotonTLV)) 
                    {
                        GenPhotonTLVEtaPtMatched.push_back(GenPhotonTLV[i]);
                    }
                }
                // calculate dR
                for (const auto& genParton : GenPartonTLV)
                {
                    float dR = ROOT::Math::VectorUtil::DeltaR(GenPhotonTLV[i], genParton);
                    dR_GenPhotonGenParton.push_back(dR);
                    if (dR > 0.4)
                    {
                        if (verbose) printf("Rejecting QCD event: DR(gen photon, gen parton) = %f\n", dR);
                        passQCDSelection = false;
                    }
                }
            }
        }
        
        // toggle debugging print statements
        bool debug = false;
        // check photon vector lengths
        bool passTest1 = (PhotonTLV.size() == Photon_Stop0l.size());
        if (debug || !passTest1) // print debugging statements
        {
            printf("PhotonTLV == Photon_Stop0l: %d == %d --- %s\n", int(PhotonTLV.size()), int(Photon_Stop0l.size()),
            passTest1 ? "passTest1" : "failTest1");
        }
        if (!passTest1)
        {
            printf(" - ERROR in include/Gamma.h: vectors of photon variables do not have the same size.\n");
            printf(" - Set debug=true in include/Gamma.h for more information.\n");
            // throw exception
            throw 20;
        }

        //Select reco photons within the ECAL acceptance region and Pt > 200 GeV 
        for(int i = 0; i < PhotonTLV.size(); ++i)
        {
            RecoPhotonTLV.push_back(PhotonTLV[i]);
            if (PhotonFunctions::passPhotonECAL(PhotonTLV[i])) 
            {
                RecoPhotonTLVEta.push_back(PhotonTLV[i]);
                // pt and eta cuts
                if (PhotonFunctions::passPhotonEtaPt(PhotonTLV[i])) 
                {
                    RecoPhotonTLVEtaPt.push_back(PhotonTLV[i]);
                    // get all IDs for testing
                    if (Photon_PassLooseID[i])  LoosePhotonTLV.push_back(PhotonTLV[i]);
                    if (Photon_PassMediumID[i]) MediumPhotonTLV.push_back(PhotonTLV[i]);
                    if (Photon_PassTightID[i])  TightPhotonTLV.push_back(PhotonTLV[i]);
                    if(Photon_ID[i])  
                    {
                        if (verbose) std::cout << "ID = " << Photon_ID[i];
                        if (verbose) printf(" Found CutPhoton; ");
                        RecoPhotonTLVIso.push_back(PhotonTLV[i]);
                        cutPhotonTLV.push_back(PhotonTLV[i]);
                        cutPhotonJetIndex.push_back(Photon_jetIdx[i]);
                        // MC Only
                        if (! isData)
                        {
                            // get scale factor for MC
                            cutPhotonSF.push_back(Photon_SF[i]);
                            cutPhotonSF_Up.push_back(Photon_SF[i] + Photon_SF_Err[i]);
                            cutPhotonSF_Down.push_back(Photon_SF[i] - Photon_SF_Err[i]);
                            // calculate dR 
                            for (const auto& genParton : GenPartonTLV)
                            {
                                float dR = ROOT::Math::VectorUtil::DeltaR(PhotonTLV[i], genParton);
                                dR_RecoPhotonGenParton.push_back(dR);
                            }
                            for (const auto& genPhoton : GenPhotonTLV)
                            {
                                float dR = ROOT::Math::VectorUtil::DeltaR(PhotonTLV[i], genPhoton);
                                dR_RecoPhotonGenPhoton.push_back(dR);
                            }
                            // -- specify different types of photons --- //
                            if (PhotonFunctions::isGenMatched_Method1(PhotonTLV[i], GenPhotonTLV))
                            {
                                if (verbose) printf("Found PromptPhoton; ");
                                RecoPhotonTLVEtaPtMatched.push_back(PhotonTLV[i]);
                                PromptPhotons.push_back(PhotonTLV[i]);
                                if (PhotonFunctions::isFragmentationPhoton(PhotonTLV[i], GenPartonTLV))
                                {
                                    if (verbose) printf("Found FragmentedPhoton\n");
                                    FragmentedPhotons.push_back(PhotonTLV[i]);
                                }
                                // direct photon if not fragmented
                                else
                                {
                                    if (verbose) printf("Found DirectPhoton\n");
                                    DirectPhotons.push_back(PhotonTLV[i]);
                                }
                            }
                            // fake photon if not prompt
                            else
                            {
                                if (verbose) printf("Found FakePhoton\n");
                                FakePhotons.push_back(PhotonTLV[i]);
                            }
                        }
                        else
                        {
                            // set scale factor to 1.0 for data
                            cutPhotonSF.push_back(1.0);
                            cutPhotonSF_Up.push_back(1.0);
                            cutPhotonSF_Down.push_back(1.0);
                        }
                    }
                }
            }
        }

        // calculate min dR
        float min_dR_GenPhotonGenParton     = -999.0;  
        float min_dR_RecoPhotonGenParton    = -999.0; 
        float min_dR_RecoPhotonGenPhoton    = -999.0;  
        // MC Only
        if (! isData)
        {
            if (!dR_GenPhotonGenParton.empty())  min_dR_GenPhotonGenParton  = *std::min_element(dR_GenPhotonGenParton.begin(), dR_GenPhotonGenParton.end());
            if (!dR_RecoPhotonGenParton.empty()) min_dR_RecoPhotonGenParton = *std::min_element(dR_RecoPhotonGenParton.begin(), dR_RecoPhotonGenParton.end());
            if (!dR_RecoPhotonGenPhoton.empty()) min_dR_RecoPhotonGenPhoton = *std::min_element(dR_RecoPhotonGenPhoton.begin(), dR_RecoPhotonGenPhoton.end());
        }

        if (verbose) fflush(stdout);

        // all IDs for testing
        bool passPhotonSelectionLoose   = bool(LoosePhotonTLV.size() == 1);
        bool passPhotonSelectionMedium  = bool(MediumPhotonTLV.size() == 1);
        bool passPhotonSelectionTight   = bool(TightPhotonTLV.size() == 1);
        
        // -------------------- //
        // --- Modified MET --- //
        // -------------------- //
        
        // WARNING: don't use new if it will not be registered or destroyed
        TLorentzVector metWithPhotonLVec;
        TLorentzVector metWithPhotonLVec_jesTotalUp;
        TLorentzVector metWithPhotonLVec_jesTotalDown;

        // set default met LVec using met and metphi
        // Pt, Eta, Phi, M
        metWithPhotonLVec.SetPtEtaPhiM(                 met,                0.0, metphi,                0.0);
        metWithPhotonLVec_jesTotalUp.SetPtEtaPhiM(      met_jesTotalUp,     0.0, metphi_jesTotalUp,     0.0);
        metWithPhotonLVec_jesTotalDown.SetPtEtaPhiM(    met_jesTotalDown,   0.0, metphi_jesTotalDown,   0.0);
        metWithPhoton                   = metWithPhotonLVec.Pt();
        metWithPhoton_jesTotalUp        = metWithPhotonLVec_jesTotalUp.Pt();
        metWithPhoton_jesTotalDown      = metWithPhotonLVec_jesTotalDown.Pt();
        metphiWithPhoton                = metWithPhotonLVec.Phi();
        metphiWithPhoton_jesTotalUp     = metWithPhotonLVec_jesTotalUp.Phi();
        metphiWithPhoton_jesTotalDown   = metWithPhotonLVec_jesTotalDown.Phi();
        // pass photon selection and add to MET
        if (cutPhotonTLV.size() == 1)
        {
            cutPhotonPt  = cutPhotonTLV[0].Pt();
            cutPhotonEta = cutPhotonTLV[0].Eta();
            // Add LVecs of MET and Photon
            metWithPhotonLVec               += cutPhotonTLV[0];
            metWithPhotonLVec_jesTotalUp    += cutPhotonTLV[0];
            metWithPhotonLVec_jesTotalDown  += cutPhotonTLV[0];
            metWithPhoton                   = metWithPhotonLVec.Pt();
            metWithPhoton_jesTotalUp        = metWithPhotonLVec_jesTotalUp.Pt();
            metWithPhoton_jesTotalDown      = metWithPhotonLVec_jesTotalDown.Pt();
            metphiWithPhoton                = metWithPhotonLVec.Phi();
            metphiWithPhoton_jesTotalUp     = metWithPhotonLVec_jesTotalUp.Phi();
            metphiWithPhoton_jesTotalDown   = metWithPhotonLVec_jesTotalDown.Phi();
            photonSF            = cutPhotonSF[0];
            photonSF_Up         = cutPhotonSF_Up[0];
            photonSF_Down       = cutPhotonSF_Down[0];
            passPhotonSelection = true;
            // MC Only
            if (! isData)
            {
                if      (DirectPhotons.size() == 1)     passPhotonSelectionDirect       = true;
                else if (FragmentedPhotons.size() == 1) passPhotonSelectionFragmented   = true;
                else if (FakePhotons.size() == 1)       passPhotonSelectionFake         = true;
            }                                               
        }
        bool printEff = false;
        if (printEff && passPhotonSelection)
        {
            printf("cutPhotonPt = %f ",             cutPhotonPt);
            printf("cutPhotonEta = %f ",            cutPhotonEta);
            printf("photonSF = %f ",                photonSF);
            printf("photonSF_Up = %f ",             photonSF_Up);
            printf("photonSF_Down = %f ",           photonSF_Down);
            printf("passPhotonSelection = %i ",     passPhotonSelection);
            printf("\n");
        }
        
        // Register derived variables
        tr.registerDerivedVar("metWithPhoton",                  metWithPhoton);
        tr.registerDerivedVar("metWithPhoton_jesTotalUp",       metWithPhoton_jesTotalUp);
        tr.registerDerivedVar("metWithPhoton_jesTotalDown",     metWithPhoton_jesTotalDown);
        tr.registerDerivedVar("metphiWithPhoton",               metphiWithPhoton);
        tr.registerDerivedVar("metphiWithPhoton_jesTotalUp",    metphiWithPhoton_jesTotalUp);
        tr.registerDerivedVar("metphiWithPhoton_jesTotalDown",  metphiWithPhoton_jesTotalDown);
        tr.registerDerivedVar("photonSF",                       photonSF);
        tr.registerDerivedVar("photonSF_Up",                    photonSF_Up);
        tr.registerDerivedVar("photonSF_Down",                  photonSF_Down);
        tr.registerDerivedVar("cutPhotonPt",                    cutPhotonPt);
        tr.registerDerivedVar("cutPhotonEta",                   cutPhotonEta);
        tr.registerDerivedVar("passPhotonSelectionLoose",       passPhotonSelectionLoose);
        tr.registerDerivedVar("passPhotonSelectionMedium",      passPhotonSelectionMedium);
        tr.registerDerivedVar("passPhotonSelectionTight",       passPhotonSelectionTight);
        tr.registerDerivedVar("passPhotonSelection",            passPhotonSelection);
        tr.registerDerivedVar("passPhotonSelectionDirect",      passPhotonSelectionDirect);
        tr.registerDerivedVar("passPhotonSelectionFragmented",  passPhotonSelectionFragmented);
        tr.registerDerivedVar("passPhotonSelectionFake",        passPhotonSelectionFake);
        tr.registerDerivedVar("passQCDSelection",               passQCDSelection);
        tr.registerDerivedVar("min_dR_GenPhotonGenParton",      min_dR_GenPhotonGenParton);
        tr.registerDerivedVar("min_dR_RecoPhotonGenParton",     min_dR_RecoPhotonGenParton);
        tr.registerDerivedVar("min_dR_RecoPhotonGenPhoton",     min_dR_RecoPhotonGenPhoton);
    }

    public:

        Gamma(std::string year = "") : year_(year)
        {
        }

        ~Gamma(){}

        void operator()(NTupleReader& tr)
        {
            generateGamma(tr);
        }
    };
}

#endif
