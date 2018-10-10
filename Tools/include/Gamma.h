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
// Generated Photon TLorentzVector:        gammaLVecGen        from genParticles   pt > 10                                  //
// Accepted Photon Variable (Loose):       loosePhotonID       from photonCands    passAcc                                  // 
// Accepted Photon Variable (Medium):      mediumPhotonID      from photonCands    passAcc                                  // 
// Accepted Photon Variable (Tight):       tightPhotonID       from photonCands    passAcc                                  // 
// Reconstructed Photon TLorentzVector:    gammaLVec           from photonCands    no cuts                                  //
// Reconstructed Photon Variable:          genMatched          from photonCands    passAcc                                  // 
// Full ID Isolated Photon Variable:       fullID              from photonCands    passAcc and passID and passIso           //
// Loose ID Isolated Photon Variable:      extraLooseID        from photonCands    passAcc passIDLoose and passIsoLoose     //
//                                                                                                                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace plotterFunctions
{
    class Gamma {

    private:

      void generateGamma(NTupleReader& tr) {

        const auto& gammaLVec            = tr.getVec<TLorentzVector>("gammaLVec");     // reco photon
        const auto& gammaLVecGen         = tr.getVec<TLorentzVector>("gammaLVecGen");  // gen photon
        const auto& genPartonLVec        = tr.getVec<TLorentzVector>("genPartonLVec"); // gen parton 
        const auto& loosePhotonID        = tr.getVec<unsigned int>("loosePhotonID");
        const auto& mediumPhotonID       = tr.getVec<unsigned int>("mediumPhotonID");
        const auto& tightPhotonID        = tr.getVec<unsigned int>("tightPhotonID");
        const auto& genMatched           = tr.getVec<data_t>("genMatched");
        const auto& sigmaIetaIeta        = tr.getVec<data_t>("sigmaIetaIeta");
        const auto& pfNeutralIsoRhoCorr  = tr.getVec<data_t>("pfNeutralIsoRhoCorr");
        const auto& pfGammaIsoRhoCorr    = tr.getVec<data_t>("pfGammaIsoRhoCorr");
        const auto& pfChargedIsoRhoCorr  = tr.getVec<data_t>("pfChargedIsoRhoCorr");
        const auto& hadTowOverEM         = tr.getVec<data_t>("hadTowOverEM");
        const auto& MT2                  = tr.getVar<data_t>("best_had_brJet_MT2");
        const auto& met                  = tr.getVar<data_t>("met");
        const auto& nJets                = tr.getVar<int>("cntNJetsPt30Eta24Zinv");
        const auto& ht                   = tr.getVar<data_t>("HT");
        const auto& nbJets               = tr.getVar<int>("cntCSVS");
        const auto& ntops                = tr.getVar<int>("nTopCandSortedCnt");

        // toggle debugging print statements
        bool debug = false;

        //variables to be used in the analysis code
        double photonPtCut = 200.0;
        double photonMet = -999.9;
        auto* gammaLVecGenEta           = new std::vector<TLorentzVector>(); 
        auto* gammaLVecGenEtaPt         = new std::vector<TLorentzVector>(); 
        auto* gammaLVecGenEtaPtMatched  = new std::vector<TLorentzVector>(); 
        auto* gammaLVecReco             = new std::vector<TLorentzVector>();
        auto* gammaLVecRecoEta          = new std::vector<TLorentzVector>();
        auto* gammaLVecRecoEtaPt        = new std::vector<TLorentzVector>();
        auto* gammaLVecRecoEtaPtMatched = new std::vector<TLorentzVector>();
        auto* gammaLVecRecoIso          = new std::vector<TLorentzVector>(); 
        auto* promptPhotons             = new std::vector<TLorentzVector>(); 
        auto* fakePhotons               = new std::vector<TLorentzVector>();
        auto* fragmentationQCD          = new std::vector<TLorentzVector>();
        auto* loosePhotons              = new std::vector<TLorentzVector>();
        auto* mediumPhotons             = new std::vector<TLorentzVector>();
        auto* tightPhotons              = new std::vector<TLorentzVector>();
        auto* directPhotons             = new std::vector<TLorentzVector>();
        auto* totalPhotons              = new std::vector<TLorentzVector>();

        //Pass cuts; use some variables from ntuples
        
        //Select gen photons
        for(int i = 0; i < gammaLVecGen.size(); ++i) {
          // ECAL eta cuts
          if (PhotonFunctions::passPhotonECAL(gammaLVecGen[i]))
          {
            gammaLVecGenEta->push_back(gammaLVecGen[i]);
          }
          // passing pt and eta cuts
          if (PhotonFunctions::passPhotonEtaPt(gammaLVecGen[i]))
          {
            gammaLVecGenEtaPt->push_back(gammaLVecGen[i]);
            // passing ECAL barrel/endcap eta cuts and reco match
            if (PhotonFunctions::isRecoMatched(gammaLVecGen[i], gammaLVec)) 
            {
              gammaLVecGenEtaPtMatched->push_back(gammaLVecGen[i]);
            }
          }
        }

        //Select reco photons; only eta cuts for now
        for(int i = 0; i < gammaLVec.size(); ++i) {
          gammaLVecReco->push_back(gammaLVec[i]);
          // passing ECAL barrel/endcap eta cuts
          // this needs to be done prior to any other cuts (pt, gen matched, etc)
          // this cut should match passAcc which is done in StopTupleMaker/SkimsAUX/plugins/PhotonIDisoProducer.cc
          if (PhotonFunctions::passPhotonECAL(gammaLVec[i])) 
          {
            gammaLVecRecoEta->push_back(gammaLVec[i]);
          }
        }
        
        // check vector lengths: gammaLVecRecoEta should have the same length as photon ntuple values for which passAcc=true
        bool passed = true;
        if (gammaLVecRecoEta->size() != genMatched.size())      passed = false;
        if (gammaLVecRecoEta->size() != loosePhotonID.size())   passed = false;
        if (gammaLVecRecoEta->size() != mediumPhotonID.size())  passed = false;
        if (gammaLVecRecoEta->size() != tightPhotonID.size())   passed = false;
        if (debug) // print debugging statements
        {
          printf("gen reco genMatched loosePhotonID mediumPhotonID tightPhotonID: %d %d --- %d %d %d %d --- %s\n", \
            int(gammaLVecGen.size()), int(gammaLVecRecoEta->size()), int(genMatched.size()), \
            int(loosePhotonID.size()), int(mediumPhotonID.size()), int(tightPhotonID.size()), passed ? "pass" : "fail");
        }
        if (!passed)
        {
          printf(" - ERROR in include/Gamma.h: TLorentzVector gammaLVecRecoEta for reco photons does not have the same length as one or more photon ntuple vectors.\n");
          printf(" - Set debug=true in include/Gamma.h for more information.\n");
          // throw exception
          try
          {
            throw 20;
          }
          catch (int e)
          {
            std::cout << "Exception: TLorentzVector gammaLVecRecoEta for reco photons does not have the same length as one or more photon ntuple vectors." << std::endl;
          }
        }

        //Select reco photons within the ECAL acceptance region and Pt > 200 GeV 
        for(int i = 0; i < gammaLVecRecoEta->size(); ++i)
        {
          // pt and eta cuts
          if (PhotonFunctions::passPhotonEtaPt((*gammaLVecRecoEta)[i])) 
          {
            gammaLVecRecoEtaPt->push_back((*gammaLVecRecoEta)[i]);
            //Select iso photons passing passAcc, passIDLoose and passIsoLoose
            if(bool(loosePhotonID[i]))
            {
              gammaLVecRecoIso->push_back((*gammaLVecRecoEta)[i]);
              // gen match
              //if (PhotonFunctions::isGenMatched_Method1((*gammaLVecRecoEta)[i], gammaLVecGen))
              if (bool(genMatched[i]))
              {
                gammaLVecRecoEtaPtMatched->push_back((*gammaLVecRecoEta)[i]);
              }
            }
          }
        }
        

        photonMet = met;
        //Get TLorentz vector for Loose, Medium and Tight ID photon selection
        for(int i = 0; i < gammaLVecRecoEta->size(); i++){
          if ((*gammaLVecRecoEta)[i].Pt() > photonPtCut){
            totalPhotons->push_back((*gammaLVecRecoEta)[i]);
            
            if(loosePhotonID[i]) loosePhotons->push_back((*gammaLVecRecoEta)[i]);
            if(mediumPhotonID[i]) mediumPhotons->push_back((*gammaLVecRecoEta)[i]);
            if(tightPhotonID[i]) tightPhotons->push_back((*gammaLVecRecoEta)[i]);

            //add loose photon pt to ptmiss
            if(loosePhotonID[i]) photonMet += (*gammaLVecRecoEta)[i].Pt();
          } 
        }

        //Gen-Matching Photons (Pt > photonPtCut in GeV)
        if(   tr.checkBranch("gammaLVecGen")  && &gammaLVecGen != nullptr
           && tr.checkBranch("genPartonLVec") && &genPartonLVec != nullptr)
        {
          for(int i = 0; i < gammaLVecRecoEta->size(); i++)
          {
            if((*gammaLVecRecoEta)[i].Pt() > photonPtCut && loosePhotonID[i])
            {
              if(PhotonFunctions::isGenMatched_Method2((*gammaLVecRecoEta)[i],gammaLVecGen))
              {
                promptPhotons->push_back((*gammaLVecRecoEta)[i]);
                if(PhotonFunctions::isDirectPhoton((*gammaLVecRecoEta)[i],genPartonLVec)) directPhotons->push_back((*gammaLVecRecoEta)[i]);
                if(PhotonFunctions::isFragmentationPhoton((*gammaLVecRecoEta)[i],genPartonLVec)) fragmentationQCD->push_back((*gammaLVecRecoEta)[i]);
              }
              else fakePhotons->push_back((*gammaLVecRecoEta)[i]);
            }
          }
        }

        tr.registerDerivedVar("photonMet", photonMet);
        tr.registerDerivedVar("passNphoton", totalPhotons->size() >= 1);
        tr.registerDerivedVar("passNloose", loosePhotons->size() >= 1);
        tr.registerDerivedVar("passNmedium", mediumPhotons->size() >= 1);
        tr.registerDerivedVar("passNtight", tightPhotons->size() >= 1);
        tr.registerDerivedVar("passFakes", fakePhotons->size() >= 1);
        tr.registerDerivedVar("passPrompt", promptPhotons->size() >= 1);
        tr.registerDerivedVar("passDirect", directPhotons->size() >= 1);
        tr.registerDerivedVar("passFragmentation", fragmentationQCD->size() >= 1);
        tr.registerDerivedVec("gammaLVecGenEta", gammaLVecGenEta);
        tr.registerDerivedVec("gammaLVecGenEtaPt", gammaLVecGenEtaPt);
        tr.registerDerivedVec("gammaLVecGenEtaPtMatched", gammaLVecGenEtaPtMatched);
        tr.registerDerivedVec("gammaLVecReco", gammaLVecReco);
        tr.registerDerivedVec("gammaLVecRecoEta", gammaLVecRecoEta);
        tr.registerDerivedVec("gammaLVecRecoEtaPt", gammaLVecRecoEtaPt);
        tr.registerDerivedVec("gammaLVecRecoEtaPtMatched", gammaLVecRecoEtaPtMatched);
        tr.registerDerivedVec("gammaLVecRecoIso", gammaLVecRecoIso);
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
