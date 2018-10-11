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
    activeBranches.insert("photonLVec");
}

////////////////////////////////

RegisterFunctionsNTuple::RegisterFunctionsNTuple(bool isCondor, std::string sbEra) : RegisterFunctions()
{            
    //AnaFunctions::prepareTopTagger();
    myBLV     = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "", "");
    blvZinv   = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "Zinv");
    //blvZinv1b = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "Zinv1b");
    //blvZinv2b = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "Zinv2b");
    //blvZinv3b = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "Zinv3b");
    blvZinvJEUUp = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "ZinvJEUUp");
    blvZinvJEUDn = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "ZinvJEUDn");
    blvZinvMEUUp = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "ZinvMEUUp");
    blvZinvMEUDn = new BaselineVessel(*static_cast<NTupleReader*>(nullptr), "ZinvMEUDn");

    weights              = new plotterFunctions::GenerateWeight;
    njWeight             = new plotterFunctions::NJetWeight;
    lepInfo              = new plotterFunctions::LepInfo;
    fakebtagvectors      = new plotterFunctions::Fakebtagvectors;
    getSearchBin         = new plotterFunctions::GetSearchBin(sbEra);
    triggerInfo          = new plotterFunctions::TriggerInfo;
    prepareMiniTupleVars = new plotterFunctions::PrepareMiniTupleVars(true);
    systematicPrep       = new plotterFunctions::SystematicPrep;
    systematicCalc       = new plotterFunctions::SystematicCalc(sbEra);

    //taudiv               = new plotterFunctions::Taudiv;
    taudiv               = new plotterFunctions::Taudiv(blvZinv->GetTopTaggerPtr());
    nJetAk8              = new plotterFunctions::NJetAk8;
    ak8DrMatch           = new plotterFunctions::Ak8DrMatch;

    myPDFUnc = new PDFUncertainty();
    bTagCorrector = nullptr;

    if(isCondor)
    {
        bTagCorrector = new BTagCorrector("allINone_bTagEff.root", "", false);
    }
    else
    {
        //bTagCorrector = new BTagCorrector("bTagEffHists.root", "/uscms_data/d3/snorberg/CMSSW_8_0_23_patch1/src/ZInvisible/Tools", false);
        bTagCorrector = new BTagCorrector("allINone_bTagEff.root", "/uscms_data/d3/nstrobbe/HadronicStop/CMSSW_8_0_25/src/ZInvisible/Tools/", false);
    }

    ISRcorrector = nullptr;
    if(isCondor)
    {
       ISRcorrector = new ISRCorrector("allINone_ISRJets.root","","");   
    }
    else
    {  
        ISRcorrector = new ISRCorrector("allINone_ISRJets.root","/uscms_data/d3/nstrobbe/HadronicStop/CMSSW_8_0_25/src/SusyAnaTools/Tools/ISR_Root_Files/","");
    }
    
    if(isCondor)
    {
        pileup = new Pileup_Sys("PileupHistograms_0121_69p2mb_pm4p6.root");
    }
    else
    {
        pileup = new Pileup_Sys("PileupHistograms_0121_69p2mb_pm4p6.root");
    } 
    
}
RegisterFunctionsNTuple::~RegisterFunctionsNTuple()
{
    if(myBLV) delete myBLV;
    if(blvZinv) delete blvZinv;
    //if(blvZinv1b) delete blvZinv1b;
    //if(blvZinv2b) delete blvZinv2b;
    //if(blvZinv3b) delete blvZinv3b;
    if(blvZinvJEUUp) delete blvZinvJEUUp;
    if(blvZinvJEUDn) delete blvZinvJEUDn;
    if(blvZinvMEUUp) delete blvZinvMEUUp;
    if(blvZinvMEUDn) delete blvZinvMEUDn;
    if(weights) delete weights;
    if(njWeight) delete njWeight;
    if(lepInfo) delete lepInfo;
    if(fakebtagvectors) delete fakebtagvectors;
    if(getSearchBin) delete getSearchBin;
    if(triggerInfo) delete triggerInfo;
    if(prepareMiniTupleVars) delete prepareMiniTupleVars;
    if(myPDFUnc) delete myPDFUnc;
    if(systematicPrep) delete systematicPrep;
    if(systematicCalc) delete systematicCalc;
    if(bTagCorrector) delete bTagCorrector;
    if(ISRcorrector) delete ISRcorrector;
    if(pileup)       delete pileup;
    if(gamma) delete gamma;
}
        
void RegisterFunctionsNTuple::registerFunctions(NTupleReader& tr)
{
    //Make some global "constants" here

    //register functions with NTupleReader
    tr.registerFunction(*myBLV);
    tr.registerFunction(*lepInfo);
    tr.registerFunction(*weights);
    tr.registerFunction(*blvZinv);
    tr.registerFunction(*njWeight);
    tr.registerFunction(*fakebtagvectors);
    //tr.registerFunction(*blvZinv1b);
   // tr.registerFunction(*blvZinv2b);
    //tr.registerFunction(*blvZinv3b);
    tr.registerFunction(*systematicPrep);
    tr.registerFunction(*blvZinvJEUUp);
    tr.registerFunction(*blvZinvJEUDn);
    tr.registerFunction(*blvZinvMEUUp);
    tr.registerFunction(*blvZinvMEUDn);
    tr.registerFunction(*getSearchBin);
    tr.registerFunction(*systematicCalc);
    tr.registerFunction(*triggerInfo);
    tr.registerFunction(*prepareMiniTupleVars);
    //tr.registerFunction(&printInterestingEvents);
    tr.registerFunction(*myPDFUnc);
    tr.registerFunction(*bTagCorrector);
    tr.registerFunction(*nJetAk8);
    tr.registerFunction(*taudiv);
    tr.registerFunction(*ak8DrMatch);
    tr.registerFunction(*ISRcorrector);
    tr.registerFunction(*pileup);
    tr.registerFunction(*gamma);
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
    //AnaFunctions::prepareTopTagger();

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
    //Make some global "constants" here

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

    myBLV     = new BaselineVessel(*static_cast<NTupleReader*>(nullptr));
    lepInfo   = new plotterFunctions::LepInfo;
}

RegisterFunctionsCalcEff::~RegisterFunctionsCalcEff()
{
    if(myBLV) delete myBLV;
    if(lepInfo) delete lepInfo;
}

void RegisterFunctionsCalcEff::registerFunctions(NTupleReader& tr)
{
    //Make some global "constants" here

    //register functions with NTupleReader
    tr.registerFunction(*myBLV);
    tr.registerFunction(*lepInfo);
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
    //prepareTopVars = new plotterFunctions::PrepareTopVars();
    triggerInfo = new plotterFunctions::TriggerInfo(false, true);
}

RegisterFunctionsTopStudy::~RegisterFunctionsTopStudy()
{
    if(myBLV) delete myBLV;
    //if(prepareTopVars) delete prepareTopVars;
}

void RegisterFunctionsTopStudy::registerFunctions(NTupleReader& tr)
{
    tr.registerFunction(*myBLV);
    //tr.registerFunction(*prepareTopVars);
    tr.registerFunction(*triggerInfo);
//    tr.registerFunction(*taudiv);
//    tr.registerFunction(*ak8DrMatch);
}

/////////////////////////////////

void drawSBregionDefCopy(const double ymin_Yields, const double ymax_Yields, const bool logscale)
{
    SearchBins::drawSBregionDef(ymin_Yields, ymax_Yields, logscale);
}
