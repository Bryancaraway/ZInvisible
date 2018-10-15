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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                          //
// Photon Objects in CMSSW8028_2016 ntuples                                                                                 //
//                                                                                                                          //
// Parton TLorentzVector:                  genPartonLVec       form genParticles   pt > 10                                  //
// Generated Photon TLorentzVector:        photonLVecGen       from genParticles   pt > 10                                  //
// Accepted Photon Variable (Loose):       loosePhotonID       from photonCands    passAcc                                  // 
// Accepted Photon Variable (Medium):      mediumPhotonID      from photonCands    passAcc                                  // 
// Accepted Photon Variable (Tight):       tightPhotonID       from photonCands    passAcc                                  // 
// Reconstructed Photon TLorentzVector:    photonLVec          from photonCands    no cuts                                  //
// Reconstructed Photon Variable:          photongenMatched    from photonCands    passAcc                                  // 
// Full ID Isolated Photon Variable:       fullID              from photonCands    passAcc and passID and passIso           //
// Loose ID Isolated Photon Variable:      extraLooseID        from photonCands    passAcc passIDLoose and passIsoLoose     //
//                                                                                                                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace plotterFunctions
{
    class Gamma {

    private:

      void generateGamma(NTupleReader& tr) {

        const auto& photonLVec                 = tr.getVec<TLorentzVector>("photonLVec");     // reco photon
        const auto& photonLVecGen              = tr.getVec<TLorentzVector>("photonLVecGen");  // gen photon
        const auto& genPartonLVec              = tr.getVec<TLorentzVector>("genPartonLVec"); // gen parton 
        const auto& loosePhotonID              = tr.getVec<unsigned int>("loosePhotonID");
        const auto& mediumPhotonID             = tr.getVec<unsigned int>("mediumPhotonID");
        const auto& tightPhotonID              = tr.getVec<unsigned int>("tightPhotonID");
        const auto& photongenMatched           = tr.getVec<data_t>("photongenMatched");
        const auto& photonsigmaIetaIeta        = tr.getVec<data_t>("photonsigmaIetaIeta");
        const auto& photonpfNeutralIsoRhoCorr  = tr.getVec<data_t>("photonpfNeutralIsoRhoCorr");
        const auto& photonpfGammaIsoRhoCorr    = tr.getVec<data_t>("photonpfGammaIsoRhoCorr");
        const auto& photonpfChargedIsoRhoCorr  = tr.getVec<data_t>("photonpfChargedIsoRhoCorr");
        const auto& photonhadTowOverEM         = tr.getVec<data_t>("photonhadTowOverEM");
        const auto& MT2                        = tr.getVar<data_t>("best_had_brJet_MT2");
        const auto& met                        = tr.getVar<data_t>("met");
        const auto& nJets                      = tr.getVar<int>("cntNJetsPt30Eta24Zinv");
        const auto& ht                         = tr.getVar<data_t>("HT");
        const auto& nbJets                     = tr.getVar<int>("cntCSVS");
        const auto& ntops                      = tr.getVar<int>("nTopCandSortedCnt");

        // toggle debugging print statements
        bool debug = true;

        //variables to be used in the analysis code
        double photonPtCut = 200.0;
        double photonMet = -999.9;
        auto* photonLVecGenEta           = new std::vector<TLorentzVector>(); 
        auto* photonLVecGenEtaPt         = new std::vector<TLorentzVector>(); 
        auto* photonLVecGenEtaPtMatched  = new std::vector<TLorentzVector>(); 
        auto* photonLVecReco             = new std::vector<TLorentzVector>();
        auto* photonLVecRecoEta          = new std::vector<TLorentzVector>();
        auto* photonLVecRecoEtaPt        = new std::vector<TLorentzVector>();
        auto* photonLVecRecoEtaPtMatched = new std::vector<TLorentzVector>();
        auto* photonLVecRecoIso          = new std::vector<TLorentzVector>(); 
        auto* promptPhotons              = new std::vector<TLorentzVector>(); 
        auto* fakePhotons                = new std::vector<TLorentzVector>();
        auto* fragmentationQCD           = new std::vector<TLorentzVector>();
        auto* loosePhotons               = new std::vector<TLorentzVector>();
        auto* mediumPhotons              = new std::vector<TLorentzVector>();
        auto* tightPhotons               = new std::vector<TLorentzVector>();
        auto* directPhotons              = new std::vector<TLorentzVector>();
        auto* totalPhotons               = new std::vector<TLorentzVector>();

        //Pass cuts; use some variables from ntuples
        
        //Select gen photons
        for(int i = 0; i < photonLVecGen.size(); ++i) {
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
        for(int i = 0; i < photonLVec.size(); ++i) {
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
        bool passed = true;
        if (photonLVecRecoEta->size() != photonLVecReco->size())  passed = false;
        if (photonLVecReco->size()    != photongenMatched.size()) passed = false;
        if (photonLVecReco->size()    != loosePhotonID.size())    passed = false;
        if (photonLVecReco->size()    != mediumPhotonID.size())   passed = false;
        if (photonLVecReco->size()    != tightPhotonID.size())    passed = false;
        if (debug) // print debugging statements
        {
          printf("photonLVecGen | photonLVecRecoEta == photonLVecReco == photongenMatched loosePhotonID mediumPhotonID tightPhotonID: %d | %d == %d == %d %d %d %d --- %s\n", \
            int(photonLVecGen.size()), int(photonLVecRecoEta->size()), int(photonLVecReco->size()), int(photongenMatched.size()), \
            int(loosePhotonID.size()), int(mediumPhotonID.size()), int(tightPhotonID.size()), passed ? "pass" : "fail");
        }
        if (!passed)
        {
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
            if(bool(loosePhotonID[i]))
            {
              photonLVecRecoIso->push_back((*photonLVecRecoEta)[i]);
              // gen match
              //if (PhotonFunctions::isGenMatched_Method1((*photonLVecRecoEta)[i], photonLVecGen))
              if (bool(photongenMatched[i]))
              {
                photonLVecRecoEtaPtMatched->push_back((*photonLVecRecoEta)[i]);
              }
            }
          }
        }
        

        photonMet = met;
        //Get TLorentz vector for Loose, Medium and Tight ID photon selection
        for(int i = 0; i < photonLVecRecoEta->size(); i++){
          if ((*photonLVecRecoEta)[i].Pt() > photonPtCut){
            totalPhotons->push_back((*photonLVecRecoEta)[i]);
            
            if(loosePhotonID[i]) loosePhotons->push_back((*photonLVecRecoEta)[i]);
            if(mediumPhotonID[i]) mediumPhotons->push_back((*photonLVecRecoEta)[i]);
            if(tightPhotonID[i]) tightPhotons->push_back((*photonLVecRecoEta)[i]);

            //add loose photon pt to ptmiss
            if(loosePhotonID[i]) photonMet += (*photonLVecRecoEta)[i].Pt();
          } 
        }

        //Gen-Matching Photons (Pt > photonPtCut in GeV)
        if(   tr.checkBranch("photonLVecGen")  && &photonLVecGen != nullptr
           && tr.checkBranch("genPartonLVec") && &genPartonLVec != nullptr)
        {
          for(int i = 0; i < photonLVecRecoEta->size(); i++)
          {
            if((*photonLVecRecoEta)[i].Pt() > photonPtCut && loosePhotonID[i])
            {
              if(PhotonFunctions::isGenMatched_Method2((*photonLVecRecoEta)[i],photonLVecGen))
              {
                promptPhotons->push_back((*photonLVecRecoEta)[i]);
                if(PhotonFunctions::isDirectPhoton((*photonLVecRecoEta)[i],genPartonLVec)) directPhotons->push_back((*photonLVecRecoEta)[i]);
                if(PhotonFunctions::isFragmentationPhoton((*photonLVecRecoEta)[i],genPartonLVec)) fragmentationQCD->push_back((*photonLVecRecoEta)[i]);
              }
              else fakePhotons->push_back((*photonLVecRecoEta)[i]);
            }
          }
        }

        // register derived variables
        tr.registerDerivedVar("photonMet", photonMet);
        tr.registerDerivedVar("passNphoton", totalPhotons->size() >= 1);
        tr.registerDerivedVar("passNloose", loosePhotons->size() >= 1);
        tr.registerDerivedVar("passNmedium", mediumPhotons->size() >= 1);
        tr.registerDerivedVar("passNtight", tightPhotons->size() >= 1);
        tr.registerDerivedVar("passFakes", fakePhotons->size() >= 1);
        tr.registerDerivedVar("passPrompt", promptPhotons->size() >= 1);
        tr.registerDerivedVar("passDirect", directPhotons->size() >= 1);
        tr.registerDerivedVar("passFragmentation", fragmentationQCD->size() >= 1);
        tr.registerDerivedVec("photonLVecGenEta", photonLVecGenEta);
        tr.registerDerivedVec("photonLVecGenEtaPt", photonLVecGenEtaPt);
        tr.registerDerivedVec("photonLVecGenEtaPtMatched", photonLVecGenEtaPtMatched);
        tr.registerDerivedVec("photonLVecReco", photonLVecReco);
        tr.registerDerivedVec("photonLVecRecoEta", photonLVecRecoEta);
        tr.registerDerivedVec("photonLVecRecoEtaPt", photonLVecRecoEtaPt);
        tr.registerDerivedVec("photonLVecRecoEtaPtMatched", photonLVecRecoEtaPtMatched);
        tr.registerDerivedVec("photonLVecRecoIso", photonLVecRecoIso);
        tr.registerDerivedVec("cutPhotons", loosePhotons);
        tr.registerDerivedVec("totalPhotons", totalPhotons);
        tr.registerDerivedVec("promptPhotons", promptPhotons);
        tr.registerDerivedVec("fakePhotons", fakePhotons);
        tr.registerDerivedVec("fragmentationQCD", fragmentationQCD);
        tr.registerDerivedVec("directPhotons", directPhotons);
        tr.registerDerivedVar("nPhotonNoID", totalPhotons->size());
        tr.registerDerivedVar("nPhoton", loosePhotons->size());
        tr.registerDerivedVar("nFakes", fakePhotons->size());
        tr.registerDerivedVar("nPrompt", promptPhotons->size());
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
