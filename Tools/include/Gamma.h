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
        //const auto& fullID              = tr.getVec<unsigned int>("fullID"); // not in CMSSW8028_2016 right now
        const auto& extraLooseID         = tr.getVec<unsigned int>("extraLooseID");
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
        auto* gammaLVecGenPtCut         = new std::vector<TLorentzVector>(); 
        auto* gammaLVecGenAcc           = new std::vector<TLorentzVector>(); 
        auto* gammaLVecGenRecoMatched   = new std::vector<TLorentzVector>(); 
        auto* gammaLVecGenIso           = new std::vector<TLorentzVector>(); 
        auto* gammaLVecRecoAcc          = new std::vector<TLorentzVector>();
        auto* gammaLVecRecoIso          = new std::vector<TLorentzVector>(); 
        auto* promptPhotons             = new std::vector<TLorentzVector>(); 
        auto* fakePhotons               = new std::vector<TLorentzVector>();
        auto* fragmentationQCD          = new std::vector<TLorentzVector>();
        auto* loosePhotons              = new std::vector<TLorentzVector>();
        auto* mediumPhotons             = new std::vector<TLorentzVector>();
        auto* tightPhotons              = new std::vector<TLorentzVector>();
        auto* directPhotons             = new std::vector<TLorentzVector>();
        auto* totalPhotons              = new std::vector<TLorentzVector>();

        // // check vector lengths
        // bool passed = true;
        // if (gammaLVec.size() != genMatched.size())      passed = false;
        // if (gammaLVec.size() != extraLooseID.size())    passed = false;
        // if (gammaLVec.size() != loosePhotonID.size())   passed = false;
        // if (gammaLVec.size() != mediumPhotonID.size())  passed = false;
        // if (gammaLVec.size() != tightPhotonID.size())   passed = false;
        // printf("gen reco genMatched extraLooseID loosePhotonID mediumPhotonID tightPhotonID: %d %d --- %d %d %d %d %d --- %s\n", \
        //   int(gammaLVecGen.size()), int(gammaLVec.size()), int(genMatched.size()), int(extraLooseID.size()),        \
        //   int(loosePhotonID.size()), int(mediumPhotonID.size()), int(tightPhotonID.size()), passed ? "pass" : "fail");

        //Pass cuts; use some variables from ntuples
        
        //Select gen photons
        for(int i = 0; i < gammaLVecGen.size(); ++i) {
          // passing pt cut
          if (PhotonFunctions::passPhotonPt(gammaLVecGen[i]))
          {
            gammaLVecGenPtCut->push_back(gammaLVecGen[i]);
          }
          // passing pt and eta cuts
          if (PhotonFunctions::passPhotonPtEta(gammaLVecGen[i]))
          {
            gammaLVecGenAcc->push_back(gammaLVecGen[i]);
            // passing ECAL barrel/endcap eta cuts and reco match
            if (  PhotonFunctions::passPhotonECAL(gammaLVecGen[i])
               && bool(PhotonFunctions::isRecoMatched(gammaLVecGen[i], gammaLVec))
               ) 
            {
              gammaLVecGenRecoMatched->push_back(gammaLVec[i]);
              //Select iso photons passing passAcc, passIDLoose and passIsoLoose
              if(bool(extraLooseID[i]))
              {
                gammaLVecGenIso->push_back(gammaLVec[i]);
              }
            }
          }
        }

        //Select reco photons; no pt cuts at the moment
        for(int i = 0; i < gammaLVec.size(); ++i) {
          // passing pt and eta cuts
          if (PhotonFunctions::passPhotonEta(gammaLVec[i]))
          {
            // passing ECAL barrel/endcap eta cuts and reco match
            // this needs to be done prior to any other cuts (pt, gen matched, etc)
            // this cut should match passAcc which is done in StopTupleMaker/SkimsAUX/plugins/PhotonIDisoProducer.cc
            if (PhotonFunctions::passPhotonECAL(gammaLVec[i])) 
            {
              gammaLVecRecoAcc->push_back(gammaLVec[i]);
              //Select iso photons passing passAcc, passIDLoose and passIsoLoose
              if(bool(extraLooseID[i]))
              {
                gammaLVecRecoIso->push_back(gammaLVec[i]);
              }
            }
          }
        }
        
        // check vector lengths
        bool passed = true;
        if (gammaLVecRecoAcc->size() != genMatched.size())      passed = false;
        if (gammaLVecRecoAcc->size() != extraLooseID.size())    passed = false;
        if (gammaLVecRecoAcc->size() != loosePhotonID.size())   passed = false;
        if (gammaLVecRecoAcc->size() != mediumPhotonID.size())  passed = false;
        if (gammaLVecRecoAcc->size() != tightPhotonID.size())   passed = false;
        if (debug) // print debugging statements
          printf("gen reco genMatched extraLooseID loosePhotonID mediumPhotonID tightPhotonID: %d %d --- %d %d %d %d %d --- %s\n", \
            int(gammaLVecGen.size()), int(gammaLVecRecoAcc->size()), int(genMatched.size()), int(extraLooseID.size()),      \
            int(loosePhotonID.size()), int(mediumPhotonID.size()), int(tightPhotonID.size()), passed ? "pass" : "fail");

        //Select reco photons within the ECAL acceptance region and Pt > 200 GeV 
        for(int i = 0; i < gammaLVec.size(); ++i) {
          if (   
                PhotonFunctions::passPhotonPtEta(gammaLVec[i])
             && PhotonFunctions::passPhotonECAL(gammaLVec[i])
             && bool(genMatched[i])
             ) 
          {
            gammaLVecRecoAcc->push_back(gammaLVec[i]);
            //Select iso photons passing passAcc, passIDLoose and passIsoLoose
            if(bool(extraLooseID[i]))
            {
              gammaLVecRecoIso->push_back(gammaLVec[i]);
            }
          }
        }
        

        photonMet = met;
        //Get TLorentz vector for Loose, Medium and Tight ID photon selection
        for(int i = 0; i < gammaLVecRecoAcc->size(); i++){
          if ((*gammaLVecRecoAcc)[i].Pt() > photonPtCut){
            totalPhotons->push_back((*gammaLVecRecoAcc)[i]);
            
            if(loosePhotonID[i]) loosePhotons->push_back((*gammaLVecRecoAcc)[i]);
            if(mediumPhotonID[i]) mediumPhotons->push_back((*gammaLVecRecoAcc)[i]);
            if(tightPhotonID[i]) tightPhotons->push_back((*gammaLVecRecoAcc)[i]);

            //add loose photon pt to ptmiss
            if(loosePhotonID[i]) photonMet += (*gammaLVecRecoAcc)[i].Pt();
          } 
        }

        //Gen-Matching Photons (Pt > photonPtCut in GeV)
        if(   tr.checkBranch("gammaLVecGen")  && &gammaLVecGen != nullptr
           && tr.checkBranch("genPartonLVec") && &genPartonLVec != nullptr)
        {
          for(int i = 0; i < gammaLVecRecoAcc->size(); i++)
          {
            if((*gammaLVecRecoAcc)[i].Pt() > photonPtCut && loosePhotonID[i])
            {
              if(PhotonFunctions::isGenMatched_Method2((*gammaLVecRecoAcc)[i],gammaLVecGen))
              {
                promptPhotons->push_back((*gammaLVecRecoAcc)[i]);
                if(PhotonFunctions::isDirectPhoton((*gammaLVecRecoAcc)[i],genPartonLVec)) directPhotons->push_back((*gammaLVecRecoAcc)[i]);
                if(PhotonFunctions::isFragmentationPhoton((*gammaLVecRecoAcc)[i],genPartonLVec)) fragmentationQCD->push_back((*gammaLVecRecoAcc)[i]);
              }
              else fakePhotons->push_back((*gammaLVecRecoAcc)[i]);
            }
          }
        }

        tr.registerDerivedVar("photonMet", photonMet);
        tr.registerDerivedVar("passNphoton",totalPhotons->size() >= 1);
        tr.registerDerivedVar("passNloose",loosePhotons->size() >= 1);
        tr.registerDerivedVar("passNmedium",mediumPhotons->size() >= 1);
        tr.registerDerivedVar("passNtight",tightPhotons->size() >= 1);
        tr.registerDerivedVar("passFakes", fakePhotons->size() >= 1);
        tr.registerDerivedVar("passPrompt", promptPhotons->size() >= 1);
        tr.registerDerivedVar("passDirect", directPhotons->size() >= 1);
        tr.registerDerivedVar("passFragmentation", fragmentationQCD->size() >= 1);
        tr.registerDerivedVec("gammaLVecGenAcc",gammaLVecGenAcc);
        tr.registerDerivedVec("gammaLVecGenRecoMatched",gammaLVecGenRecoMatched);
        tr.registerDerivedVec("gammaLVecGenIso",gammaLVecGenIso);
        tr.registerDerivedVec("gammaLVecRecoAcc",gammaLVecRecoAcc);
        tr.registerDerivedVec("gammaLVecRecoIso",gammaLVecRecoIso);
        tr.registerDerivedVec("cutPhotons",loosePhotons);
        tr.registerDerivedVec("totalPhotons",totalPhotons);
        tr.registerDerivedVec("promptPhotons",promptPhotons);
        tr.registerDerivedVec("fakePhotons",fakePhotons);
        tr.registerDerivedVec("fragmentationQCD",fragmentationQCD);
        tr.registerDerivedVec("directPhotons",directPhotons);
        tr.registerDerivedVar("nPhotonNoID",totalPhotons->size());
        tr.registerDerivedVar("nPhoton",loosePhotons->size());
        tr.registerDerivedVar("nFakes",fakePhotons->size());
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
