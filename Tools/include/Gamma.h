#ifndef GAMMA_H
#define GAMMA_H

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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                          //
// Photon Objects in CMSSW8028_2016 ntuples                                                                                 //
//                                                                                                                          //
// Parton TLorentzVector:                  genPartonLVec       form genParticles   pt > 10                                  //
// Generated Photon TLorentzVector:        photonLVecGen        from genParticles   pt > 10                                  //
// Accepted Photon Variable (Loose):       loosePhotonID       from photonCands    passAcc                                  // 
// Accepted Photon Variable (Medium):      mediumPhotonID      from photonCands    passAcc                                  // 
// Accepted Photon Variable (Tight):       tightPhotonID       from photonCands    passAcc                                  // 
// Reconstructed Photon TLorentzVector:    photonLVec           from photonCands    no cuts                                  //
// Reconstructed Photon Variable:          photonGenMatched          from photonCands    passAcc                                  // 
// Full ID Isolated Photon Variable:       fullID              from photonCands    passAcc and passID and passIso           //
// Loose ID Isolated Photon Variable:      extraLooseID        from photonCands    passAcc passIDLoose and passIsoLoose     //
//                                                                                                                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace plotterFunctions
{
    class Gamma {

    private:

      void generateGamma(NTupleReader& tr) {
        //std::cout << "Running Gamma.h" << std::endl;

        const auto& photonLVec            = tr.getVec<TLorentzVector>("photonLVec");     // reco photon
        const auto& photonLVecGen         = tr.getVec<TLorentzVector>("photonLVecGen");  // gen photon
        const auto& genPartonLVec        = tr.getVec<TLorentzVector>("genPartonLVec"); // gen parton 
        const auto& loosePhotonID        = tr.getVec<unsigned int>("loosePhotonID");
        const auto& mediumPhotonID       = tr.getVec<unsigned int>("mediumPhotonID");
        const auto& tightPhotonID        = tr.getVec<unsigned int>("tightPhotonID");
        const auto& photonGenMatched           = tr.getVec<data_t>("photonGenMatched");
        const auto& photonSigmaIetaIeta        = tr.getVec<data_t>("photonSigmaIetaIeta");
        const auto& photonPFNeutralIsoRhoCorr  = tr.getVec<data_t>("photonPFNeutralIsoRhoCorr");
        const auto& photonPFGammaIsoRhoCorr    = tr.getVec<data_t>("photonPFGammaIsoRhoCorr");
        const auto& photonPFChargedIsoRhoCorr  = tr.getVec<data_t>("photonPFChargedIsoRhoCorr");
        const auto& photonHadTowOverEM         = tr.getVec<data_t>("photonHadTowOverEM");
        const auto& met                  = tr.getVar<data_t>("met");
        const auto& metphi               = tr.getVar<data_t>("metphi");


        // toggle debugging print statements
        bool debug = false;

        //variables to be used in the analysis code
        //float photonMet = -999.9;
        //float photonPtCut = 200.0;
        float metWithPhoton = -999.9;
        float metphiWithPhoton = -999.9;
        bool passPhotonSelection = false;
        
        auto* photonLVecGenEta           = new std::vector<TLorentzVector>(); 
        auto* photonLVecGenEtaPt         = new std::vector<TLorentzVector>(); 
        auto* photonLVecGenEtaPtMatched  = new std::vector<TLorentzVector>(); 
        auto* photonLVecReco             = new std::vector<TLorentzVector>();
        auto* photonLVecRecoEta          = new std::vector<TLorentzVector>();
        auto* photonLVecRecoEtaPt        = new std::vector<TLorentzVector>();
        auto* photonLVecRecoEtaPtMatched = new std::vector<TLorentzVector>();
        auto* photonLVecRecoIso          = new std::vector<TLorentzVector>(); 
        auto* photonLVecPassLooseID      = new std::vector<TLorentzVector>();
        auto* photonLVecPassMediumID     = new std::vector<TLorentzVector>();
        auto* photonLVecPassTightID      = new std::vector<TLorentzVector>();
        auto* metLVec                   = new TLorentzVector();
        auto* metWithPhotonLVec         = new TLorentzVector();
        
        //auto* promptPhotons             = new std::vector<TLorentzVector>(); 
        //auto* fakePhotons               = new std::vector<TLorentzVector>();
        //auto* fragmentationQCD          = new std::vector<TLorentzVector>();
        //auto* loosePhotons              = new std::vector<TLorentzVector>();
        //auto* mediumPhotons             = new std::vector<TLorentzVector>();
        //auto* tightPhotons              = new std::vector<TLorentzVector>();
        //auto* directPhotons             = new std::vector<TLorentzVector>();
        //auto* totalPhotons              = new std::vector<TLorentzVector>();


        //Pass cuts; use some variables from ntuples
        
        //Select gen photons
        for(int i = 0; i < photonLVecGen.size(); ++i)
        {
          // ECAL eta cuts
          if (PhotonFunctions::passPhotonECAL(photonLVecGen[i]))
          {
            photonLVecGenEta->push_back(photonLVecGen[i]);
          }
          // passing pt and eta cuts
          if (PhotonFunctions::passPhotonEtaPt(photonLVecGen[i]))
          {
            photonLVecGenEtaPt->push_back(photonLVecGen[i]);
            // passing ECAL barrel/endcap eta cuts and reco match
            if (PhotonFunctions::isRecoMatched(photonLVecGen[i], photonLVec)) 
            {
              photonLVecGenEtaPtMatched->push_back(photonLVecGen[i]);
            }
          }
        }

        //Select reco photons; only eta cuts for now
        for(int i = 0; i < photonLVec.size(); ++i)
        {
            photonLVecReco->push_back(photonLVec[i]);
          // passing ECAL barrel/endcap eta cuts
          // this needs to be done prior to any other cuts (pt, gen matched, etc)
          // this cut should match passAcc which is done in StopTupleMaker/SkimsAUX/plugins/PhotonIDisoProducer.cc
          if (PhotonFunctions::passPhotonECAL(photonLVec[i])) 
          {
            photonLVecRecoEta->push_back(photonLVec[i]);
          }
        }
        
        // check vector lengths: photonLVecRecoEta should have the same length as photon ntuple values for which passAcc=true
        bool passTest1 = true;
        bool passTest2 = true;
        if (photonLVecReco->size()    != photonLVecRecoEta->size()) passTest1 = false;
        if (photonLVecRecoEta->size() != photonGenMatched.size())        passTest2 = false;
        if (photonLVecRecoEta->size() != loosePhotonID.size())     passTest2 = false;
        if (photonLVecRecoEta->size() != mediumPhotonID.size())    passTest2 = false;
        if (photonLVecRecoEta->size() != tightPhotonID.size())     passTest2 = false;
        if (debug || !passTest2) // print debugging statements
        {
          printf("photonLVecGen photonLVecReco photonLVecRecoEta photonGenMatched loosePhotonID mediumPhotonID tightPhotonID: %d | %d == %d == %d %d %d %d --- %s, %s\n", \
            int(photonLVecGen.size()), int(photonLVecReco->size()), int(photonLVecRecoEta->size()), int(photonGenMatched.size()), \
            int(loosePhotonID.size()), int(mediumPhotonID.size()), int(tightPhotonID.size()), passTest1 ? "passTest1" : "failTest1", passTest2 ? "passTest2" : "failTest2");
        }
        if (!passTest2)
        {
          // we should probably throw an exception here
          printf(" - ERROR in include/Gamma.h: TLorentzVector photonLVecRecoEta for reco photons does not have the same length as one or more photon ntuple vectors.\n");
          printf(" - Set debug=true in include/Gamma.h for more information.\n");
          // throw exception
          try
          {
            throw 20;
          }
          catch (int e)
          {
            std::cout << "Exception: TLorentzVector photonLVecRecoEta for reco photons does not have the same length as one or more photon ntuple vectors." << std::endl;
          }
        }

        //Select reco photons within the ECAL acceptance region and Pt > 200 GeV 
        for(int i = 0; i < photonLVecRecoEta->size(); ++i)
        {
          // pt and eta cuts
          if (PhotonFunctions::passPhotonEtaPt((*photonLVecRecoEta)[i])) 
          {
            photonLVecRecoEtaPt->push_back((*photonLVecRecoEta)[i]);
            //Select iso photons passing passAcc, passIDLoose and passIsoLoose
            if(bool(loosePhotonID[i]))  photonLVecPassLooseID  -> push_back((*photonLVecRecoEta)[i]);
            if(bool(mediumPhotonID[i])) photonLVecPassMediumID -> push_back((*photonLVecRecoEta)[i]);
            if(bool(tightPhotonID[i]))  photonLVecPassTightID  -> push_back((*photonLVecRecoEta)[i]);
            if(bool(loosePhotonID[i]))
            {
              photonLVecRecoIso->push_back((*photonLVecRecoEta)[i]);
              // gen match
              //if (PhotonFunctions::isGenMatched_Method1((*photonLVecRecoEta)[i], photonLVecGen))
              if (bool(photonGenMatched[i]))
              {
                photonLVecRecoEtaPtMatched->push_back((*photonLVecRecoEta)[i]);
              }
            }
          }
        }

        // set met LVec
        // Pt, Eta, Phi, E
        //metLVec->SetPtEtaPhiE(met, 0.0, metphi, met);
        // Pt, Eta, Phi, M
        metLVec->SetPtEtaPhiM(met, 0.0, metphi, 0.0);
        metWithPhotonLVec = metLVec;
        metWithPhoton     = metLVec->Pt();
        metphiWithPhoton  = metLVec->Phi();
        // pass photon selection and add to MET
        if (photonLVecRecoIso->size() == 1)
        {
            // Add LVecs of MET and Photon
            *metWithPhotonLVec += (*photonLVecRecoIso)[0];
            metWithPhoton       = metWithPhotonLVec->Pt();
            metphiWithPhoton    = metWithPhotonLVec->Phi();
            passPhotonSelection = true;
        }

// --- Beginning of section not used (as of October 19, 2018)        
//
//        //photonMet = met;
//        //Get TLorentz vector for Loose, Medium and Tight ID photon selection
//        for(int i = 0; i < photonLVecRecoEta->size(); i++){
//          if ((*photonLVecRecoEta)[i].Pt() > photonPtCut){
//            totalPhotons->push_back((*photonLVecRecoEta)[i]);
//            
//            if(loosePhotonID[i]) loosePhotons->push_back((*photonLVecRecoEta)[i]);
//            if(mediumPhotonID[i]) mediumPhotons->push_back((*photonLVecRecoEta)[i]);
//            if(tightPhotonID[i]) tightPhotons->push_back((*photonLVecRecoEta)[i]);
//
//            //add loose photon pt to ptmiss
//            //if(loosePhotonID[i]) photonMet += (*photonLVecRecoEta)[i].Pt();
//          } 
//        }
//
//        //Gen-Matching Photons (Pt > photonPtCut in GeV)
//        if(   tr.checkBranch("photonLVecGen")  && &photonLVecGen != nullptr
//           && tr.checkBranch("genPartonLVec") && &genPartonLVec != nullptr)
//        {
//          for(int i = 0; i < photonLVecRecoEta->size(); i++)
//          {
//            if((*photonLVecRecoEta)[i].Pt() > photonPtCut && loosePhotonID[i])
//            {
//              if(PhotonFunctions::isGenMatched_Method2((*photonLVecRecoEta)[i],photonLVecGen))
//              {
//                promptPhotons->push_back((*photonLVecRecoEta)[i]);
//                if(PhotonFunctions::isDirectPhoton((*photonLVecRecoEta)[i],genPartonLVec)) directPhotons->push_back((*photonLVecRecoEta)[i]);
//                if(PhotonFunctions::isFragmentationPhoton((*photonLVecRecoEta)[i],genPartonLVec)) fragmentationQCD->push_back((*photonLVecRecoEta)[i]);
//              }
//              else fakePhotons->push_back((*photonLVecRecoEta)[i]);
//            }
//          }
//        }
//
// --- End of section not used (as of October 19, 2018)        

        // Register derived variables
        tr.registerDerivedVar("metWithPhoton", metWithPhoton);
        tr.registerDerivedVar("metphiWithPhoton", metphiWithPhoton);
        tr.registerDerivedVar("passPhotonSelection", passPhotonSelection);
        
        tr.registerDerivedVec("photonLVecPassLooseID", photonLVecPassLooseID);
        tr.registerDerivedVec("photonLVecPassMediumID", photonLVecPassMediumID);
        tr.registerDerivedVec("photonLVecPassTightID", photonLVecPassTightID);
        tr.registerDerivedVec("photonLVecGenEta", photonLVecGenEta);
        tr.registerDerivedVec("photonLVecGenEtaPt", photonLVecGenEtaPt);
        tr.registerDerivedVec("photonLVecGenEtaPtMatched", photonLVecGenEtaPtMatched);
        tr.registerDerivedVec("photonLVecReco", photonLVecReco);
        tr.registerDerivedVec("photonLVecRecoEta", photonLVecRecoEta);
        tr.registerDerivedVec("photonLVecRecoEtaPt", photonLVecRecoEtaPt);
        tr.registerDerivedVec("photonLVecRecoEtaPtMatched", photonLVecRecoEtaPtMatched);
        tr.registerDerivedVec("photonLVecRecoIso", photonLVecRecoIso);
        
        //tr.registerDerivedVar("photonMet", photonMet);
        
        //tr.registerDerivedVec("cutPhotons", loosePhotons);
        //tr.registerDerivedVec("totalPhotons", totalPhotons);
        //tr.registerDerivedVec("promptPhotons", promptPhotons);
        //tr.registerDerivedVec("fakePhotons", fakePhotons);
        //tr.registerDerivedVec("fragmentationQCD", fragmentationQCD);
        //tr.registerDerivedVec("directPhotons", directPhotons);
        //tr.registerDerivedVar("nPhotonNoID", totalPhotons->size());
        //tr.registerDerivedVar("nPhoton", loosePhotons->size());
        //tr.registerDerivedVar("nFakes", fakePhotons->size());
        //tr.registerDerivedVar("nPrompt", promptPhotons->size());
        
        //tr.registerDerivedVar("passNphoton", totalPhotons->size() >= 1);
        //tr.registerDerivedVar("passNloose", loosePhotons->size() >= 1);
        //tr.registerDerivedVar("passNmedium", mediumPhotons->size() >= 1);
        //tr.registerDerivedVar("passNtight", tightPhotons->size() >= 1);
        //tr.registerDerivedVar("passFakes", fakePhotons->size() >= 1);
        //tr.registerDerivedVar("passPrompt", promptPhotons->size() >= 1);
        //tr.registerDerivedVar("passDirect", directPhotons->size() >= 1);
        //tr.registerDerivedVar("passFragmentation", fragmentationQCD->size() >= 1);
      }

    public:

      Gamma(){}

      ~Gamma(){}

      void operator()(NTupleReader& tr)
      {
        generateGamma(tr);
      }
    };
}

#endif
