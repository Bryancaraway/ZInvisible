#include "RegisterFunctions.h"
#include "NTupleReader.h"
#include "derivedTupleVariables.h"
#include "baselineDef.h"
#include "Systematic.h"
#include "PDFUncertainty.h"
#include "BTagCorrector.h"
#include "ISRCorrector.h"
#include "PileupWeights.h"

const std::set<std::string> RegisterFunctions::getMiniTupleSet()
{
    //if you want to not fill the minituple return std::set<std::string>({});
  return std::set<std::string>({"HTZinv","cleanMetPt","cleanMetPhi","best_had_brJet_MT2Zinv","cntCSVSZinv","nTopCandSortedCntZinv","cntNJetsPt30Eta24Zinv","nSearchBin","cutMuVec","cutElecVec","jetsLVec_forTaggerZinv", "recoJetsBtag_forTaggerZinv","zEffWgt","zAccWgt","cuts","passMuTrigger","genHT","genWeight","bTagSF_EventWeightSimple_Central","isr_Unc_Cent","_PUweightFactor"});
    //return std::set<std::string>({"HTZinv","cleanMetPt","cleanMetPhi","best_had_brJet_MT2Zinv","cntCSVSZinv","nTopCandSortedCntZinv","cntNJetsPt30Eta24Zinv","nSearchBin","cutMuVec","cutElecVec","jetsLVec_forTaggerZinv", "recoJetsBtag_forTaggerZinv","zEffWgt","zAccWgt","cuts","passMuTrigger","genHT","genWeight"});
}

const std::set<std::string> RegisterFunctions::getMiniTupleSetData()
{
  return std::set<std::string>({"HTZinv","cleanMetPt","cleanMetPhi","best_had_brJet_MT2Zinv","cntCSVSZinv","nTopCandSortedCntZinv","cntNJetsPt30Eta24Zinv","nSearchBin","cutMuVec","cutElecVec","jetsLVec_forTaggerZinv", "recoJetsBtag_forTaggerZinv","zEffWgt","zAccWgt","cuts","passMuTrigger"});
}

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
    activeBranches.insert("recoJetsCSVv2");
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
    activeBranches.insert("tau1");
    activeBranches.insert("tau2");
    activeBranches.insert("tau3");
    activeBranches.insert("subjetBdisc");
    activeBranches.insert("puppiJetsLVec");
    activeBranches.insert("ak8mass");
    activeBranches.insert("ak8rapidity");
    activeBranches.insert("softDropMass");
    activeBranches.insert("prunedMass");
    activeBranches.insert("slimmedJetsAK8");
    activeBranches.insert("ak8pt");
    activeBranches.insert("nJetsAk8");
    activeBranches.insert("ak82dRMin");
    activeBranches.insert("ak81dRMin");
    activeBranches.insert("tru_npv");
    activeBranches.insert("vtxSize");
    //Photon branches Andres                                                                                                              
    activeBranches.insert("isEB");
    activeBranches.insert("genMatched");
    activeBranches.insert("hadTowOverEM");
    activeBranches.insert("sigmaIetaIeta");
    activeBranches.insert("pfChargedIso");
    activeBranches.insert("pfNeutralIso");
    activeBranches.insert("pfGammaIso");
    activeBranches.insert("pfChargedIsoRhoCorr");
    activeBranches.insert("pfNeutralIsoRhoCorr");
    activeBranches.insert("pfGammaIsoRhoCorr");
    activeBranches.insert("hasPixelSeed");
    activeBranches.insert("passElectronVeto");
    activeBranches.insert("nonPrompt");
    activeBranches.insert("fullID");
    activeBranches.insert("photonPt");
    activeBranches.insert("photonEta");
    activeBranches.insert("photonPhi");
    activeBranches.insert("totalPhotons");
    activeBranches.insert("gammaLVec");
    activeBranches.insert("isZToLL");
    
}

////////////////////////////////

RegisterFunctionsNTuple::RegisterFunctionsNTuple(bool isCondor, std::string sbEra, std::string year) : RegisterFunctions()
{            
    // Important: create objects!!
    
    //AnaFunctions::prepareTopTagger();
    myBLV               = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "");
    blvZinv             = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "Zinv");
    blvNoVeto           = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "NoVeto");
    blvPFLeptonCleaned  = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "PFLeptonCleaned");
    blv_drLeptonCleaned = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "_drLeptonCleaned");
    blv_drPhotonCleaned = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "_drPhotonCleaned");
    //blvZinvJEUUp = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "ZinvJEUUp");
    //blvZinvJEUDn = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "ZinvJEUDn");
    //blvZinvMEUUp = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "ZinvMEUUp");
    //blvZinvMEUDn = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "ZinvMEUDn");


    getVectors                   = new GetVectors;
    cleanedJets                  = new CleanedJets;
    // RunTopTagger(std::string taggerCfg = "TopTagger.cfg", std::string suffix = "", bool doLeptonCleaning = false, bool doPhotonCleaning = false)
    runTopTagger                 = new RunTopTagger;
    runTopTagger_drLeptonCleaned = new RunTopTagger("TopTagger.cfg","_drLeptonCleaned",true,false);
    runTopTagger_drPhotonCleaned = new RunTopTagger("TopTagger.cfg","_drPhotonCleaned",false,true);
    gamma                        = new plotterFunctions::Gamma;
    weights                      = new plotterFunctions::GenerateWeight;
    generatePhotonEfficiency     = new plotterFunctions::GeneratePhotonEfficiency;
    njWeight                     = new plotterFunctions::NJetWeight;
    lepInfo                      = new plotterFunctions::LepInfo(year);
    basicLepton                  = new plotterFunctions::BasicLepton;
    fakebtagvectors              = new plotterFunctions::Fakebtagvectors;
    getSearchBin                 = new plotterFunctions::GetSearchBin;
    triggerInfo                  = new plotterFunctions::TriggerInfo;
    prepareMiniTupleVars         = new plotterFunctions::PrepareMiniTupleVars(true);
    systematicPrep               = new plotterFunctions::SystematicPrep;
    systematicCalc               = new plotterFunctions::SystematicCalc(sbEra);

    //taudiv                    = new plotterFunctions::Taudiv;
    taudiv                       = new plotterFunctions::Taudiv(blvZinv->GetTopTaggerPtr());
    nJetAk8                      = new plotterFunctions::NJetAk8;
    ak8DrMatch                   = new plotterFunctions::Ak8DrMatch;

    myPDFUnc = new PDFUncertainty();
    bTagCorrector = nullptr;
    ISRcorrector = nullptr;
    pileup = nullptr;

    // bTagCorrector not used anymore
    //if(isCondor)
    //{
    //    bTagCorrector = new BTagCorrector("allINone_bTagEff.root", "", false);
    //}
    //else
    //{
    //    bTagCorrector = new BTagCorrector("allINone_bTagEff.root", "/uscms/home/caleb/nobackup/SusyAnalysis/CMSSW_9_4_4/src/ZInvisible/Tools", false);
    //}

    //if(isCondor)
    //{
    //   ISRcorrector = new ISRCorrector("allINone_ISRJets.root","","");   
    //}
    //else
    //{  
    //    ISRcorrector = new ISRCorrector("allINone_ISRJets.root","/uscms/home/caleb/nobackup/SusyAnalysis/CMSSW_9_4_4/src/ZInvisible/Tools","");
    //}
    //
    //if(isCondor)
    //{
    //    pileup = new Pileup_Sys("PileupHistograms_0121_69p2mb_pm4p6.root");
    //}
    //else
    //{
    //    pileup = new Pileup_Sys("PileupHistograms_0121_69p2mb_pm4p6.root");
    //} 
    
}
RegisterFunctionsNTuple::~RegisterFunctionsNTuple()
{
    if(getVectors)                   delete getVectors;
    if(cleanedJets)                  delete cleanedJets;
    if(runTopTagger)                 delete runTopTagger;
    if(runTopTagger_drLeptonCleaned) delete runTopTagger_drLeptonCleaned;
    if(runTopTagger_drPhotonCleaned) delete runTopTagger_drPhotonCleaned;
    if(myBLV)                        delete myBLV;
    if(blvZinv)                      delete blvZinv;
    if(blvNoVeto)                    delete blvNoVeto;
    if(blvPFLeptonCleaned)           delete blvPFLeptonCleaned;
    if(blv_drLeptonCleaned)          delete blv_drLeptonCleaned;
    if(blv_drPhotonCleaned)          delete blv_drPhotonCleaned;
    //if(blvZinv1b)                  delete blvZinv1b;
    //if(blvZinv2b)                  delete blvZinv2b;
    //if(blvZinv3b)                  delete blvZinv3b;
    //if(blvZinvJEUUp)               delete blvZinvJEUUp;
    //if(blvZinvJEUDn)               delete blvZinvJEUDn;
    //if(blvZinvMEUUp)               delete blvZinvMEUUp;
    //if(blvZinvMEUDn)               delete blvZinvMEUDn;
    if(weights)                      delete weights;
    if(generatePhotonEfficiency)     delete generatePhotonEfficiency;
    if(njWeight)                     delete njWeight;
    if(lepInfo)                      delete lepInfo;
    if(basicLepton)                  delete basicLepton;
    if(fakebtagvectors)              delete fakebtagvectors;
    if(getSearchBin)                 delete getSearchBin;
    if(triggerInfo)                  delete triggerInfo;
    if(prepareMiniTupleVars)         delete prepareMiniTupleVars;
    if(myPDFUnc)                     delete myPDFUnc;
    if(systematicPrep)               delete systematicPrep;
    if(systematicCalc)               delete systematicCalc;
    if(bTagCorrector)                delete bTagCorrector;
    if(ISRcorrector)                 delete ISRcorrector;
    if(pileup)                       delete pileup;
    if(gamma)                        delete gamma;
}
        
void RegisterFunctionsNTuple::registerFunctions(NTupleReader& tr)
{
    //register functions with NTupleReader
    
    // order matters
    // get photons and leptons
    // use photons and leptons to clean jets
    tr.registerFunction(*getVectors);
    tr.registerFunction(*gamma);
    tr.registerFunction(*basicLepton);
    //tr.registerFunction(*generatePhotonEfficiency);
    tr.registerFunction(*cleanedJets);
    tr.registerFunction(*runTopTagger);
    tr.registerFunction(*runTopTagger_drLeptonCleaned);
    tr.registerFunction(*runTopTagger_drPhotonCleaned);
    tr.registerFunction(*myBLV);
    tr.registerFunction(*lepInfo);
    tr.registerFunction(*blv_drLeptonCleaned);
    tr.registerFunction(*blv_drPhotonCleaned);
    tr.registerFunction(*getSearchBin);
    
    // old version including JEC and systematics
    // here for reference only
    //tr.registerFunction(*weights);
    //tr.registerFunction(*blvZinv);
    //tr.registerFunction(*blvNoVeto);
    //tr.registerFunction(*blvPFLeptonCleaned);
    //tr.registerFunction(*blv_drLeptonCleaned);
    //tr.registerFunction(*blv_drPhotonCleaned);
    //tr.registerFunction(*njWeight);
    //tr.registerFunction(*fakebtagvectors);
    //tr.registerFunction(*blvZinv1b);
    //tr.registerFunction(*blvZinv2b);
    //tr.registerFunction(*blvZinv3b);
    //tr.registerFunction(*systematicPrep);
    //tr.registerFunction(*blvZinvJEUUp);
    //tr.registerFunction(*blvZinvJEUDn);
    //tr.registerFunction(*blvZinvMEUUp);
    //tr.registerFunction(*blvZinvMEUDn);
    //tr.registerFunction(*systematicCalc);
    //tr.registerFunction(*triggerInfo);
    //tr.registerFunction(*prepareMiniTupleVars);
    //tr.registerFunction(&printInterestingEvents);
    //tr.registerFunction(*myPDFUnc);
    //tr.registerFunction(*bTagCorrector);
    //tr.registerFunction(*nJetAk8);
    //tr.registerFunction(*taudiv);
    //tr.registerFunction(*ak8DrMatch);
    //tr.registerFunction(*ISRcorrector);
    //tr.registerFunction(*pileup);
}

void RegisterFunctionsNTuple::activateBranches(std::set<std::string>& activeBranches)
{
    ::activateBranches(activeBranches);
}

void RegisterFunctionsNTuple::remakeBTagCorrector(std::string sampleName)
{
    if(sampleName.find("Data") == std::string::npos){
        if(bTagCorrector) bTagCorrector->resetEffs(sampleName);
    }
}


void RegisterFunctionsNTuple::remakeISRreweight(std::string sampleName)
{
    if(sampleName.find("Data") == std::string::npos){
        if(ISRcorrector) ISRcorrector->resetSample(sampleName);
    }
}

////////////////////////////////

RegisterFunctionsMiniTuple::RegisterFunctionsMiniTuple() : RegisterFunctions()
{            
    njWeight             = new plotterFunctions::NJetWeight;
    triggerInfo          = new plotterFunctions::TriggerInfo(true);
    prepareMiniTupleVars = new plotterFunctions::PrepareMiniTupleVars(false);
    metSmear             = new plotterFunctions::MetSmear;
}

RegisterFunctionsMiniTuple::~RegisterFunctionsMiniTuple()
{
    if(njWeight) delete njWeight;
    if(triggerInfo) delete triggerInfo;
    if(prepareMiniTupleVars) delete prepareMiniTupleVars;
    if(metSmear) delete metSmear;
}
        
void RegisterFunctionsMiniTuple::registerFunctions(NTupleReader& tr)
{
    //register functions with NTupleReader
    tr.registerFunction(*njWeight);
    tr.registerFunction(*triggerInfo);
    tr.registerFunction(*prepareMiniTupleVars);
    tr.registerFunction(*metSmear);
}

void RegisterFunctionsMiniTuple::activateBranches(std::set<std::string>& activeBranches)
{
}

////////////////////////////////

RegisterFunctionsCalcEff::RegisterFunctionsCalcEff() : RegisterFunctions()
{
    //AnaFunctions::prepareTopTagger();

    myBLV   = new BaselineVessel(*static_cast<NTupleReader*>(nullptr));
    blvZinv = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "Zinv");
    getVectors                = new GetVectors;
    cleanedJets               = new CleanedJets;
    runTopTagger              = new RunTopTagger;
    gamma                     = new plotterFunctions::Gamma;
    basicLepton               = new plotterFunctions::BasicLepton;
    generatePhotonEfficiency  = new plotterFunctions::GeneratePhotonEfficiency;
}

RegisterFunctionsCalcEff::~RegisterFunctionsCalcEff()
{
    if(myBLV) delete myBLV;
    if(blvZinv) delete blvZinv;
    if(getVectors) delete getVectors;
    if(cleanedJets) delete cleanedJets;
    if(runTopTagger) delete runTopTagger;
    if(gamma) delete gamma;
    if(basicLepton) delete basicLepton;
    if(generatePhotonEfficiency) delete generatePhotonEfficiency;
}

void RegisterFunctionsCalcEff::registerFunctions(NTupleReader& tr)
{
    //register functions with NTupleReader
    tr.registerFunction(*getVectors);
    tr.registerFunction(*gamma);
    tr.registerFunction(*basicLepton);
    //tr.registerFunction(*generatePhotonEfficiency);
    tr.registerFunction(*cleanedJets);
    tr.registerFunction(*runTopTagger);
    tr.registerFunction(*myBLV);
    tr.registerFunction(*blvZinv);

}

void RegisterFunctionsCalcEff::activateBranches(std::set<std::string>& activeBranches)
{
    ::activateBranches(activeBranches);
}

////////////////////////////////

void RegisterFunctionsSyst::addFunction(std::function<void(NTupleReader&)> func)
{
    funcs_.emplace_back(func);
}

RegisterFunctionsSyst::RegisterFunctionsSyst() : RegisterFunctions()
{
    njWeight             = new plotterFunctions::NJetWeight;
    prepareMiniTupleVars = new plotterFunctions::PrepareMiniTupleVars(false);
    triggerInfo          = new plotterFunctions::TriggerInfo(true);
    systWeights          = new SystWeights;
}

RegisterFunctionsSyst::~RegisterFunctionsSyst()
{
    if(njWeight) delete njWeight;
    if(prepareMiniTupleVars) delete prepareMiniTupleVars;
    if(triggerInfo) delete triggerInfo;
    if(systWeights) delete systWeights;
}

void RegisterFunctionsSyst::registerFunctions(NTupleReader& tr)
{
    for(auto& f : funcs_) tr.registerFunction(f);

    tr.registerFunction(*njWeight);
    tr.registerFunction(*prepareMiniTupleVars);
    tr.registerFunction(*triggerInfo);
    //tr.registerFunction(*systWeights);
}

////////////////////////////////

RegisterFunctions2Dplot::RegisterFunctions2Dplot()
{
    prepareMiniTupleVars = new plotterFunctions::PrepareMiniTupleVars(false);
}

RegisterFunctions2Dplot::~RegisterFunctions2Dplot()
{
    if(prepareMiniTupleVars) delete prepareMiniTupleVars;
}

void RegisterFunctions2Dplot::registerFunctions(NTupleReader& tr)
{
    tr.registerFunction(*prepareMiniTupleVars);
}

////////////////////////////////

RegisterFunctionsTopStudy::RegisterFunctionsTopStudy()
{
    myBLV = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "TopTag", "");
    triggerInfo = new plotterFunctions::TriggerInfo(false, true);
}

RegisterFunctionsTopStudy::~RegisterFunctionsTopStudy()
{
    if(myBLV) delete myBLV;
}

void RegisterFunctionsTopStudy::registerFunctions(NTupleReader& tr)
{
    tr.registerFunction(*myBLV);
    tr.registerFunction(*triggerInfo);
}

/////////////////////////////////

void drawSBregionDefCopy(const double ymin_Yields, const double ymax_Yields, const bool logscale)
{
    SearchBins::drawSBregionDef(ymin_Yields, ymax_Yields, logscale);
}
