#ifndef REGISTERFUNCTIONS_H
#define REGISTERFUNCTIONS_H

#include <set>
#include <string>
#include <vector>
#include <functional>
#include "SusyAnaTools/Tools/CleanedJets.h"

class NTupleReader;
class BaselineVessel;
class PDFUncertainty;
class BTagCorrector;
class ISRCorrector;
class Pileup_Sys; 

namespace plotterFunctions
{
    class GenerateWeight;
    class GeneratePhotonEfficiency;
    class LepInfo;
    class BasicLepton;
    class Fakebtagvectors;
    class GetSearchBin;
    class TriggerInfo;
    class PrepareMiniTupleVars;
    class NJetWeight;
    class SystematicPrep;
    class SystematicCalc;
    class MetSmear;
    class PrepareTopVars;
    class Taudiv;
    class NJetAk8;
    class Ak8DrMatch; 
    class Gamma;
}

class SystWeights;

class RegisterFunctions
{
private:

public:
    RegisterFunctions() {}
        
    virtual void registerFunctions(NTupleReader& tr) {};
    virtual void activateBranches(std::set<std::string>& activeBranches) {};
    virtual const std::set<std::string> getMiniTupleSet();
    virtual const std::set<std::string> getMiniTupleSetData();
    virtual void remakeBTagCorrector(std::string sampleName) {};
    virtual void remakeISRreweight(std::string sampleName) {};
    
};

class RegisterFunctionsNTuple : public RegisterFunctions
{
private:
    BaselineVessel *myBLV;
    BaselineVessel *blvZinv;
    BaselineVessel *blvNoVeto;
    BaselineVessel *blvPFLeptonCleaned;
    BaselineVessel *blvDRLeptonCleaned;
    BaselineVessel *blvDRPhotonCleaned;
    //BaselineVessel *blvZinv1b;
    //BaselineVessel *blvZinv2b;
    //BaselineVessel *blvZinv3b;
    BaselineVessel *blvZinvJEUUp;
    BaselineVessel *blvZinvJEUDn;
    BaselineVessel *blvZinvMEUUp;
    BaselineVessel *blvZinvMEUDn;
    plotterFunctions::GenerateWeight *weights;
    plotterFunctions::GeneratePhotonEfficiency *generatePhotonEfficiency;
    plotterFunctions::NJetWeight *njWeight;
    plotterFunctions::BasicLepton *basicLepton;
    plotterFunctions::LepInfo *lepInfo;
    plotterFunctions::Fakebtagvectors *fakebtagvectors;
    plotterFunctions::GetSearchBin *getSearchBin;
    plotterFunctions::TriggerInfo *triggerInfo;
    plotterFunctions::PrepareMiniTupleVars *prepareMiniTupleVars;
    plotterFunctions::SystematicPrep *systematicPrep;
    plotterFunctions::SystematicCalc *systematicCalc;
    CleanedJets *cleanedJets;
    plotterFunctions::Gamma *gamma; //Andres
    PDFUncertainty *myPDFUnc;
    BTagCorrector *bTagCorrector;
    ISRCorrector *ISRcorrector;
    Pileup_Sys *pileup;
    plotterFunctions::Taudiv *taudiv;
    plotterFunctions::NJetAk8 *nJetAk8;
    plotterFunctions::Ak8DrMatch *ak8DrMatch;

public:
    RegisterFunctionsNTuple(bool isCondor = false, std::string sbEra = "SB_v1_2017");
    ~RegisterFunctionsNTuple();
    void registerFunctions(NTupleReader& tr);
    void activateBranches(std::set<std::string>& activeBranches);
    void remakeBTagCorrector(std::string sampleName);
    void remakeISRreweight(std::string sampleName);
};

class RegisterFunctionsMiniTuple : public RegisterFunctions
{
private:
    plotterFunctions::NJetWeight *njWeight;
    plotterFunctions::TriggerInfo *triggerInfo;
    plotterFunctions::PrepareMiniTupleVars *prepareMiniTupleVars;
    plotterFunctions::MetSmear *metSmear;

public:
    RegisterFunctionsMiniTuple();
    ~RegisterFunctionsMiniTuple();
    void registerFunctions(NTupleReader& tr);
    void activateBranches(std::set<std::string>& activeBranches);
};

class RegisterFunctionsCalcEff : public RegisterFunctions
{
private:
    BaselineVessel *myBLV;
    plotterFunctions::Gamma *gamma; //Caleb :-)
    plotterFunctions::LepInfo *lepInfo;
    plotterFunctions::BasicLepton *basicLepton;

public:
    RegisterFunctionsCalcEff();
    ~RegisterFunctionsCalcEff();
    void registerFunctions(NTupleReader& tr);
    void activateBranches(std::set<std::string>& activeBranches);
};

class RegisterFunctionsSyst : public RegisterFunctions
{
private:
    std::vector<std::function<void(NTupleReader&)> > funcs_;
    plotterFunctions::PrepareMiniTupleVars *prepareMiniTupleVars;
    plotterFunctions::NJetWeight *njWeight;
    plotterFunctions::TriggerInfo *triggerInfo;
    SystWeights *systWeights;
public:
    RegisterFunctionsSyst();
    ~RegisterFunctionsSyst();
    void addFunction(std::function<void(NTupleReader&)> func);
    void registerFunctions(NTupleReader& tr);
};

class RegisterFunctions2Dplot : public RegisterFunctions
{
private:
    plotterFunctions::PrepareMiniTupleVars *prepareMiniTupleVars;
public:
    RegisterFunctions2Dplot();
    ~RegisterFunctions2Dplot();
    void registerFunctions(NTupleReader& tr);
};

class RegisterFunctionsTopStudy : public RegisterFunctions
{
private:
    BaselineVessel *myBLV;
    plotterFunctions::PrepareTopVars *prepareTopVars;
    plotterFunctions::Taudiv *taudiv;
    plotterFunctions::Ak8DrMatch *ak8DrMatch;
    plotterFunctions::TriggerInfo *triggerInfo;

public:
    RegisterFunctionsTopStudy();
    ~RegisterFunctionsTopStudy();
    void registerFunctions(NTupleReader& tr);
};

void drawSBregionDefCopy(const double ymin_Yields = 0.05, const double ymax_Yields = 500., const bool logscale = true);

#endif
