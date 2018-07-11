#include "Plotter.h"
#include "RegisterFunctions.h"
#include "SusyAnaTools/Tools/searchBins.h"
#include "SusyAnaTools/Tools/samples.h"

#include <getopt.h>
#include <iostream>

int main(int argc, char* argv[])
{
    using namespace std;

    int opt;
    int option_index = 0;
    static struct option long_options[] = {
        {"plot",             no_argument, 0, 'p'},
        {"savehist",         no_argument, 0, 's'},
        {"savetuple",        no_argument, 0, 't'},
        {"fromFile",         no_argument, 0, 'f'},
        {"condor",           no_argument, 0, 'c'},
        {"histFile",   required_argument, 0, 'H'},
        {"dataSets",   required_argument, 0, 'D'},
        {"numFiles",   required_argument, 0, 'N'},
        {"startFile",  required_argument, 0, 'M'},
        {"numEvts",    required_argument, 0, 'E'},
        {"plotDir",    required_argument, 0, 'P'},
        {"luminosity", required_argument, 0, 'L'},
        {"sbEra",      required_argument, 0, 'S'}
    };

    bool doPlots = true, doSave = true, doTuple = true, fromTuple = true, runOnCondor = false;
    string histFile = "", dataSets = "", sampleloc = AnaSamples::fileDir, plotDir = "plots";
    int nFiles = -1, startFile = 0, nEvts = -1;
    float lumi = AnaSamples::luminosity;
    std::string sbEra = "SB_v1_2017";//"SB_v1_2017";

    while((opt = getopt_long(argc, argv, "pstfcH:D:N:M:E:P:L:S:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
        case 'p':
            if(doPlots) doSave  = doTuple = false;
            else        doPlots = true;
            break;

        case 's':
            if(doSave) doPlots = doTuple = false;
            else       doSave  = true;
            break;

        case 't':
            if(doTuple) doPlots = doSave = false;
            else        doTuple  = true;
            break;

        case 'f':
            fromTuple = false;
            break;

        case 'c':
            runOnCondor = true;
            break;

        case 'H':
            histFile = optarg;
            break;

        case 'D':
            dataSets = optarg;
            break;

        case 'N':
            nFiles = int(atoi(optarg));
            break;

        case 'M':
            startFile = int(atoi(optarg));
            break;

        case 'E':
            nEvts = int(atoi(optarg));
            break;

        case 'P':
            plotDir = optarg;
            break;

        case 'L':
            lumi = atof(optarg);
            break;

        case 'S':
            sbEra = optarg;
            break;
        }
    }

    //if running on condor override all optional settings
    if(runOnCondor)
    {
        char thistFile[128];
        sprintf(thistFile, "histoutput_%s_%d.root", dataSets.c_str(), startFile);
        histFile = thistFile;
        //doSave = true;
        //doPlots = false;
        fromTuple = true;
        sampleloc = "condor";
    }

    // the old version
    //AnaSamples::SampleSet        ss(sampleloc, lumi);
    //AnaSamples::SampleCollection sc(ss);
    // the new version
    AnaSamples::SampleSet        ss("sampleSets.txt");
    AnaSamples::SampleCollection sc("sampleCollections.txt", ss);

    const float zAcc = 1.0;
    //    const float zAcc = 0.5954;
    //    const float zAcc = 0.855;
    const float znunu_mumu_ratio = 5.942;
    const float znunu_ee_ratio   = 5.942;

    map<string, vector<AnaSamples::FileSummary>> fileMap;

    //Select approperiate datasets here
    if(dataSets.compare("TEST") == 0)
    {
        fileMap["DYJetsToLL"]  = {ss["DYJetsToLL_HT_1200to2500"]};
        fileMap["ZJetsToNuNu"] = {ss["ZJetsToNuNu_HT_2500toInf"]};
        fileMap["DYJetsToLL_HT_600to800"] = {ss["DYJetsToLL_HT_600to800"]};
        fileMap["ZJetsToNuNu_HT_2500toInf"] = {ss["ZJetsToNuNu_HT_2500toInf"]};
        fileMap["TTbarDiLep"] = {ss["TTbarDiLep"]};
        fileMap["TTbarNoHad"] = {ss["TTbarDiLep"]};
        fileMap["Data_SingleMuon"] = {ss["Data_SingleMuon"]};
    }
    else if(dataSets.compare("TEST2") == 0)
    {
        fileMap["DYJetsToLL"]  = {ss["DYJetsToLL_HT_600to800"]};
        fileMap["DYJetsToLL_HT_600to800"] = {ss["DYJetsToLL_HT_600to800"]};
        fileMap["IncDY"] = {ss["DYJetsToLL_Inc"]}; 
        fileMap["TTbarDiLep"] = {ss["TTbarDiLep"]};
        fileMap["TTbarNoHad"] = {ss["TTbarDiLep"]};
        fileMap["Data_SingleMuon"] = {ss["Data_SingleMuon"]};
    }
    else
    {
        if(ss[dataSets] != ss.null())
        {
            fileMap[dataSets] = {ss[dataSets]};
            for(const auto& colls : ss[dataSets].getCollections())
            {
                fileMap[colls] = {ss[dataSets]};
            }
        }
        else if(sc[dataSets] != sc.null())
        {
            fileMap[dataSets] = {sc[dataSets]};
            int i = 0;
            for(const auto& fs : sc[dataSets])
            {
                fileMap[sc.getSampleLabels(dataSets)[i++]].push_back(fs);
            }
        }
    }

    // Number of searchbins
    SearchBins sb(sbEra);
    int NSB = sb.nSearchBins();//37; // 45

    // Shortcuts for axis labels
    std::string label_met = "p_{T}^{miss} [GeV]";
    //std::string label_met = "#slash{E}_{T} [GeV]";
    std::string label_ht  = "H_{T} [GeV]";
    std::string label_mht = "MH_{T} [GeV]";
    std::string label_nj  = "N_{jets}";
    std::string label_nb  = "N_{b}";
    std::string label_nt  = "N_{t}";
    std::string label_mt2 = "M_{T2} [GeV]";
    std::string label_mupt  = "#mu p_{T} [GeV]";
    std::string label_mu1pt = "#mu_{1} p_{T} [GeV]";
    std::string label_mu2pt = "#mu_{2} p_{T} [GeV]";
    std::string label_elpt  = "e p_{T} [GeV]";
    std::string label_el1pt = "e_{1} p_{T} [GeV]";
    std::string label_el2pt = "e_{2} p_{T} [GeV]";
    std::string label_jpt  = "j p_{T} [GeV]";
    std::string label_j1pt = "j_{1} p_{T} [GeV]";
    std::string label_j2pt = "j_{2} p_{T} [GeV]";
    std::string label_j3pt = "j_{3} p_{T} [GeV]";
    std::string label_mll  = "m_{ll} [GeV]";
    std::string label_toppt = "top p_{T} [GeV]";
    std::string label_phopt = "p_{T}^{#gamma} [GeV]";
    std::string label_metg = "p_{T}^{#gamma (miss)} [GeV]";

    vector<Plotter::HistSummary> vh;

    // Datasetsummaries we are using                                                
    // no weight (genWeight deals with negative weights); also add btag weights here                                                    
    Plotter::DatasetSummary dsData_SinglePhoton("Data", fileMap["Data_SinglePhoton"], "passPhotonTrigger", "");
    Plotter::DatasetSummary dsTTG("t#bar{t}#gamma", fileMap["TTGJets"], "", "bTagSF_EventWeightSimple_Central;_PUweightFactor;genWeight");
    Plotter::DatasetSummary dsPhoton("#gamma+ jets", fileMap["GJets"], "","bTagSF_EventWeightSimple_Central;_PUweightFactor");
    Plotter::DatasetSummary dsQCD("QCD", fileMap["QCD"], "passQCDHighMETFilter", "bTagSF_EventWeightSimple_Central;_PUweightFactor");
    Plotter::DatasetSummary dsWJets("W(l#nu)+jets", fileMap["WJetsToLNu"], "", "bTagSF_EventWeightSimple_Central;_PUweightFactor");
    Plotter::DatasetSummary dstt2l("t#bar{t}", fileMap["TTbarAll"], "", "bTagSF_EventWeightSimple_Central;isr_Unc_Cent;_PUweightFactor");
    Plotter::DatasetSummary dsttZ("t#bar{t}Z",fileMap["TTZ"],"","bTagSF_EventWeightSimple_Central;genWeight;_PUweightFactor");
    Plotter::DatasetSummary dsVV("Diboson",fileMap["Diboson"],"","bTagSF_EventWeightSimple_Central;_PUweightFactor;genWeight");
    Plotter::DatasetSummary dsRare("Rare",fileMap["Rare"],"","bTagSF_EventWeightSimple_Central;genWeight;_PUweightFactor");
    Plotter::DatasetSummary dstW("tW", fileMap["tW"], "", "bTagSF_EventWeightSimple_Central;_PUweightFactor;genWeight");

    // Shape Scale Factor applied (genWeight deals with negative weights); also add btag weights here                                    
    Plotter::DatasetSummary dsPhotonW("#gamma+ jets", fileMap["GJets"], "","bTagSF_EventWeightSimple_Central;_PUweightFactor;njWGJets");
    // Shape Scale Factor and normalization from tight mumu CR applied
    Plotter::DatasetSummary dsPhotonWW("#gamma+ jets", fileMap["GJets"], "","bTagSF_EventWeightSimple_Central;_PUweightFactor;njWGJets;normWgt0b");
    // Shape Scale Factor applied (normalized with gjets nJets); also add btag weights here                                        
    Plotter::DatasetSummary dsPhotonWN("#gamma+ jets", fileMap["GJets"], "","bTagSF_EventWeightSimple_Central;_PUweightFactor;njWGJetsNorm");

    std::vector<std::vector<Plotter::DatasetSummary>> stack_gammaMC = {{dsPhoton},{dsQCD},{dsWJets},{dsTTG},{dstt2l},{dstW},{dsVV},{dsRare,dsttZ}};
    std::vector<std::vector<Plotter::DatasetSummary>> stackw_gammaMC = {{dsPhotonW},{dsQCD},{dsWJets},{dsTTG},{dstt2l},{dstW},{dsVV},{dsRare,dsttZ}};
    std::vector<std::vector<Plotter::DatasetSummary>> stackww_gammaMC = {{dsPhotonWW},{dsQCD},{dsWJets},{dsTTG},{dstt2l},{dstW},{dsVV},{dsRare,dsttZ}};
    std::vector<std::vector<Plotter::DatasetSummary>> stackwn_gammaMC = {{dsPhotonWN},{dsQCD},{dsWJets},{dsTTG},{dstt2l},{dstW},{dsVV},{dsRare,dsttZ}};

    //std::vector<std::vector<Plotter::DatasetSummary>> stack_gammaMC = {{dsttZ},{dsRare},{dsVV},{dstW},{dstt2l},{dsWJets},{dsTTG},{dsQCD},{dsPhoton}};

    //Fake Rate Studies
    Plotter::DataCollection dcFakes("single", "nFakes", {dsPhoton});
    Plotter::DataCollection dcPhotonPt("single", "cutPhotons(pt)", {dsPhoton});
    Plotter::DataCollection dcTotalPhotonPt("single", "totalPhotons(pt)", {dsPhoton});
    Plotter::DataCollection dcFakePhotonPt("single", "fakePhotons(pt)", {dsPhoton});
    Plotter::DataCollection dcFakeQCDPt("single", "fakePhotons(pt)", {dsQCD});
    Plotter::DataCollection dcFragPhotonPt("single", "fragmentationQCD(pt)", {dsQCD});
    Plotter::DataCollection dcDirectPhotonPt("single", "directPhotons(pt)", {dsPhoton});
    Plotter::DataCollection dcDirectQCDPt("single", "directPhotons(pt)", {dsQCD});
    Plotter::DataCollection dcPromptPhotonPt("single", "promptPhotons(pt)", {dsPhoton});
    Plotter::DataCollection dcQCDPromptPt("single", "promptPhotons(pt)", {dsQCD});

    //Single Photon data for analysis variables                                                                                            
    //Photon count                                                                                                                           
    Plotter::DataCollection dcPhotonCntData("data", "nPhotonNoID", {dsData_SinglePhoton});
    Plotter::DataCollection dcCutPhotonCntData("data", "nPhoton", {dsData_SinglePhoton});
    //Pt
    Plotter::DataCollection dcPhotonPtData("data", "totalPhotons(pt)", {dsData_SinglePhoton});
    Plotter::DataCollection dcCutPhotonPtData("data", "cutPhotons(pt)", {dsData_SinglePhoton});
    //Photon Eta
    Plotter::DataCollection dcPhotonEtaData("data", "totalPhotons(eta)", {dsData_SinglePhoton});
    Plotter::DataCollection dcCutPhotonEtaData("data", "cutPhotons(eta)", {dsData_SinglePhoton});
    //Met Data                                                                                                                           
    Plotter::DataCollection dcPhotonMetData("data", "met", {dsData_SinglePhoton});
    //Photon Met Data                                                                                                                        
    Plotter::DataCollection dcMetGammaData("data", "photonMet", {dsData_SinglePhoton});
    //photon HT                                                                                                                             
    Plotter::DataCollection dcPhotonHtData("data", "HT", {dsData_SinglePhoton});
    //photon MT2                                                                                                                            
    Plotter::DataCollection dcPhotonMt2Data("data", "best_had_brJet_MT2", {dsData_SinglePhoton});
    //photon Njets                                                                                                                          
    Plotter::DataCollection dcPhotonNjData("data", "cntNJetsPt30Eta24Zinv", {dsData_SinglePhoton});
    //photon Ntops                                                                                                                          
    Plotter::DataCollection dcPhotonNtData("data", "nTopCandSortedCnt", {dsData_SinglePhoton});
    //photon NbJets                                                                                                                         
    Plotter::DataCollection dcPhotonNbData("data", "cntCSVS", {dsData_SinglePhoton});

    //Photon stacked MC for analysis variables
    //photon Pt
    Plotter::DataCollection dcPhotonPtMC("stack", "totalPhotons(pt)", {stack_gammaMC});
    Plotter::DataCollection dcCutPhotonPtMC("stack", "cutPhotons(pt)", {stack_gammaMC});
    //Photon Eta                                                                                                                             
    Plotter::DataCollection dcPhotonEtaMC("stack", "totalPhotons(eta)", {stack_gammaMC});
    Plotter::DataCollection dcCutPhotonEtaMC("stack", "cutPhotons(eta)", {stack_gammaMC});
    //Met MC
    Plotter::DataCollection dcPhotonMetMC("stack", "met", {stack_gammaMC});
    //Photon MET MC
    Plotter::DataCollection dcMetGammaMC("stack", "photonMet", {stack_gammaMC});
    //HT MC
    Plotter::DataCollection dcPhotonHtMC("stack", "HT", {stack_gammaMC});
    //MT2 MC
    Plotter::DataCollection dcPhotonMt2MC("stack", "best_had_brJet_MT2", {stack_gammaMC});
    //photon Njets MC
    Plotter::DataCollection dcPhotonNjMC("stack", "cntNJetsPt30Eta24Zinv", {stack_gammaMC});
    //photon Ntops MC 
    Plotter::DataCollection dcPhotonNtMC("stack", "nTopCandSortedCnt", {stack_gammaMC});
    //photon NbJets
    Plotter::DataCollection dcPhotonNbMC("stack", "cntCSVS", {stack_gammaMC});
    //Photon count
    Plotter::DataCollection dcPhotonCntMC("stack", "nPhotonNoID", {stack_gammaMC});
    Plotter::DataCollection dcCutPhotonCntMC("stack", "nPhoton", {stack_gammaMC});

    //Photon stacked MC for analysis variables with the Njets scale factor applied 
    //photon Pt                                                                                                                              
    Plotter::DataCollection dcWPhotonPtMC("stack", "totalPhotons(pt)", {stackw_gammaMC});
    Plotter::DataCollection dcWCutPhotonPtMC("stack", "cutPhotons(pt)", {stackw_gammaMC});
    //Photon Eta                                                                                                                             
    Plotter::DataCollection dcWPhotonEtaMC("stack", "totalPhotons(eta)", {stackw_gammaMC});
    Plotter::DataCollection dcWCutPhotonEtaMC("stack", "cutPhotons(eta)", {stackw_gammaMC});
    //Met MC                                                                                                                                 
    Plotter::DataCollection dcWPhotonMetMC("stack", "met", {stackw_gammaMC});
    //Photon MET MC                                                                                                                          
    Plotter::DataCollection dcWMetGammaMC("stack", "photonMet", {stackw_gammaMC});
    //HT MC                                                                                                                                  
    Plotter::DataCollection dcWPhotonHtMC("stack", "HT", {stackw_gammaMC});
    //MT2 MC                                                                                                                                 
    Plotter::DataCollection dcWPhotonMt2MC("stack", "best_had_brJet_MT2", {stackw_gammaMC});
    //photon Njets MC                                                                                                                        
    Plotter::DataCollection dcWPhotonNjMC("stack", "cntNJetsPt30Eta24Zinv", {stackw_gammaMC});
    //photon Ntops MC                                                                                                                        
    Plotter::DataCollection dcWPhotonNtMC("stack", "nTopCandSortedCnt", {stackw_gammaMC});
    //photon NbJets                                                                                                                          
    Plotter::DataCollection dcWPhotonNbMC("stack", "cntCSVS", {stackw_gammaMC});
    //Photon count                                                                                                                           
    Plotter::DataCollection dcWPhotonCntMC("stack", "nPhotonNoID", {stackw_gammaMC});
    Plotter::DataCollection dcWCutPhotonCntMC("stack", "nPhoton", {stackw_gammaMC});

    //Photon stacked MC for analysis variables with the Njets normalized scale factor applied                                                
    //photon Pt                                                                                                                              
    Plotter::DataCollection dcWWPhotonPtMC("stack", "totalPhotons(pt)", {stackww_gammaMC});
    Plotter::DataCollection dcWWCutPhotonPtMC("stack", "cutPhotons(pt)", {stackww_gammaMC});
    //Photon Eta                                                                                                                             
    Plotter::DataCollection dcWWPhotonEtaMC("stack", "totalPhotons(eta)", {stackww_gammaMC});
    Plotter::DataCollection dcWWCutPhotonEtaMC("stack", "cutPhotons(eta)", {stackww_gammaMC});
    //Met MC                                                                                                                                 
    Plotter::DataCollection dcWWPhotonMetMC("stack", "met", {stackww_gammaMC});
    //Photon MET MC                                                                                                                          
    Plotter::DataCollection dcWWMetGammaMC("stack", "photonMet", {stackww_gammaMC});
    //HT MC                                                                                                                                  
    Plotter::DataCollection dcWWPhotonHtMC("stack", "HT", {stackww_gammaMC});
    //MT2 MC                                                                                                                                 
    Plotter::DataCollection dcWWPhotonMt2MC("stack", "best_had_brJet_MT2", {stackww_gammaMC});
    //photon Njets MC                                                                                                                        
    Plotter::DataCollection dcWWPhotonNjMC("stack", "cntNJetsPt30Eta24Zinv", {stackww_gammaMC});
    //photon Ntops MC                                                                                                                        
    Plotter::DataCollection dcWWPhotonNtMC("stack", "nTopCandSortedCnt", {stackww_gammaMC});
    //photon NbJets                                                                                                                          
    Plotter::DataCollection dcWWPhotonNbMC("stack", "cntCSVS", {stackww_gammaMC});
    //Photon count                                                                                                                           
    Plotter::DataCollection dcWWCutPhotonCntMC("stack", "nPhoton", {stackww_gammaMC});

    //Photon stacked MC for analysis variables with the Njets normalized scale factor applied
    //photon Pt                                                                                                                              
    Plotter::DataCollection dcWNPhotonPtMC("stack", "totalPhotons(pt)", {stackwn_gammaMC});
    Plotter::DataCollection dcWNCutPhotonPtMC("stack", "cutPhotons(pt)", {stackwn_gammaMC});
    //Photon Eta                                                                                                                             
    Plotter::DataCollection dcWNPhotonEtaMC("stack", "totalPhotons(eta)", {stackwn_gammaMC});
    Plotter::DataCollection dcWNCutPhotonEtaMC("stack", "cutPhotons(eta)", {stackwn_gammaMC});
    //Met MC                                                                                                                                 
    Plotter::DataCollection dcWNPhotonMetMC("stack", "met", {stackwn_gammaMC});
    //Photon MET MC                                                                                                                          
    Plotter::DataCollection dcWNMetGammaMC("stack", "photonMet", {stackwn_gammaMC});
    //HT MC                                                                                                                                  
    Plotter::DataCollection dcWNPhotonHtMC("stack", "HT", {stackwn_gammaMC});
    //MT2 MC                                                                                                                                 
    Plotter::DataCollection dcWNPhotonMt2MC("stack", "best_had_brJet_MT2", {stackwn_gammaMC});
    //photon Njets MC                                                                                                                        
    Plotter::DataCollection dcWNPhotonNjMC("stack", "cntNJetsPt30Eta24Zinv", {stackwn_gammaMC});
    //photon Ntops MC                                                                                                                        
    Plotter::DataCollection dcWNPhotonNtMC("stack", "nTopCandSortedCnt", {stackwn_gammaMC});
    //photon NbJets                                                                                                                          
    Plotter::DataCollection dcWNPhotonNbMC("stack", "cntCSVS", {stackwn_gammaMC});
    //Photon count                                                                                                                           
    Plotter::DataCollection dcWNCutPhotonCntMC("stack", "nPhoton", {stackwn_gammaMC});
                                               
    //Photon extra variables                   
    //Photon count
    Plotter::DataCollection dcPhotonCnt("single", "nPhotonNoID", {dsPhoton});
    Plotter::DataCollection dcCutPhotonCnt("single", "nPhoton", {dsPhoton});
    //Photon Pt
    Plotter::DataCollection dcPhoton_pt("single", "totalPhotons(pt)", {dsPhoton});
    Plotter::DataCollection dcCutPhoton_pt("single", "cutPhotons(pt)", {dsPhoton});
    //Photon Eta
    Plotter::DataCollection dcPhoton_eta("single", "totalPhotons(eta)", {dsPhoton});
    Plotter::DataCollection dcCutPhoton_eta("single", "cutPhotons(eta)", {dsPhoton});
    //Met
    Plotter::DataCollection dcPhoton_met("single", "met", {dsPhoton});
    //Photon MET
    Plotter::DataCollection dcMetGamma("single", "photonMet", {dsPhoton});
    //Photon Ntops
    Plotter::DataCollection dcPhoton_nt("single", "nTopCandSortedCnt", {dsPhoton});
    //Photon MT2
    Plotter::DataCollection dcPhoton_mt2("single", "best_had_brJet_MT2", {dsPhoton});
    //NbJets photon
    Plotter::DataCollection dcPhoton_nb("single", "cntCSVS", {dsPhoton});
    //NJets photon
    Plotter::DataCollection dcPhoton_nj("single", "cntNJetsPt30Eta24Zinv", {dsPhoton});
    //Photon HT
    Plotter::DataCollection dcPhoton_ht("single", "HT", {dsPhoton});

    //other
    Plotter::DataCollection dcPhoton_jpt("single", "jetsLVecLepCleaned(pt)", {dsPhoton});
    Plotter::DataCollection dcPhoton_j1pt("single", "jetsLVecLepCleaned[0](pt)", {dsPhoton});
    Plotter::DataCollection dcPhoton_j2pt("single", "jetsLVecLepCleaned[1](pt)", {dsPhoton});
    Plotter::DataCollection dcPhoton_j3pt("single", "jetsLVecLepCleaned[2](pt)", {dsPhoton});

    // pair of cutlevels                                                                                                                   
    std::vector<std::pair<std::string,std::string>> cutlevels_photon = {
        {"empty",                     ""},
        //{"passLoose",                 "passNloose"},
        //{"passMedium",                "passNmedium"},
        //{"passTight",                 "passNtight"},
        {"nosel",                     "passNoiseEventFilter"},
        {"passt",                     "passNoiseEventFilter;passnJetsZinv;passdPhis"},
        {"pass1",                     "passNoiseEventFilter;passnJetsZinv;passdPhisZinv"},
        {"pass2",                     "passNoiseEventFilter;passnJetsZinv;passdPhis"},
        {"pass3",                     "passNoiseEventFilter;passnJetsZinv;passdPhis;HT>300"},
        {"allCutsNoID",               "passNoiseEventFilter;passnJetsZinv;passdPhis;HT>300;passNphoton"},
        {"allCutsLoose",              "passNoiseEventFilter;passnJetsZinv;passdPhis;HT>300;passNloose"},
        //{"allCutsMedium",             "passNoiseEventFilter;passnJets;passdPhis;HT>300;passNmedium"},
        //{"allCutsTight",              "passNoiseEventFilter;passnJets;passdPhis;HT>300;passNtight"},
        //{"NoIDlepVeto",               "passNoiseEventFilter;passnJetsZinv;passdPhis;HT>300;passNphoton;passLeptVeto"},
        {"LooseLepVeto",              "passNoiseEventFilter;passnJetsZinv;passdPhis;HT>300;passNloose;passLeptVeto"},
        {"LooseLepVetoZinv",          "passNoiseEventFilterZinv;passnJetsZinv;passdPhisZinv;HT>300;passNloose;passLeptVeto"},
        //{"MediumLepVeto",             "passNoiseEventFilter;passnJets;passdPhis;HT>300;passNmedium;passLeptVeto"},
        //{"TightLepVeto",              "passNoiseEventFilter;passnJets;passdPhis;HT>300;passNtight;passLeptVeto"},
        //{"CRelecVeto",                "passNoiseEventFilter;passnJets;passdPhis;HT>300;passNloose;passMuonVeto"},
        //{"CRmuonVeto",                "passNoiseEventFilter;passnJets;passdPhis;HT>300;passNloose;passEleVeto"},
        //{"CRisoTrack",                "passNoiseEventFilter;passnJets;passdPhis;HT>300;passNloose;passIsoTrkVeto"},
        //{"CRLepVeto",                 "passNoiseEventFilter;passnJets;passdPhis;HT>300;passNloose;passMuonVeto;passEleVeto"},
        //FakeRate Studies
        //{"FakesCR",                   "passNoiseEventFilter;passnJets;passdPhis;HT>300;passFakes;passLeptVeto"},
    };

    //push the histograms in a loop, save some copy-paste time                                                                           
    for(std::pair<std::string,std::string>& cut : cutlevels_photon)
    {
        //FakeRate 
        vh.push_back(PHS("Photon_Pt_"+cut.first,{dcPhotonPt},{1, 1},cut.second, 60, 0, 1500,true, false,label_phopt, "Events"));
        vh.push_back(PHS("QCDPrompt_Pt_"+cut.first,{dcQCDPromptPt},{1, 1},cut.second, 60, 0, 1500,true, false,label_phopt, "Events"));
        vh.push_back(PHS("Fake_Photon_Pt_"+cut.first,{dcFakePhotonPt},{1, 1},cut.second, 60, 0, 1500,true, false,label_phopt, "Events"));
        vh.push_back(PHS("Fake_QCDPhoton_Pt_"+cut.first,{dcFakeQCDPt},{1, 1},cut.second, 60, 0, 1500,true, false,label_phopt, "Events"));
        vh.push_back(PHS("Prompt_Photon_Pt_"+cut.first,{dcPromptPhotonPt},{1, 1},cut.second, 60, 0, 1500,true, false,label_phopt,"Events"));
        vh.push_back(PHS("Direct_Photon_Pt_"+cut.first,{dcDirectPhotonPt},{1, 1},cut.second, 60, 0, 1500,true, false,label_phopt,"Events"));
        vh.push_back(PHS("Direct_QCDPhoton_Pt_"+cut.first,{dcDirectQCDPt},{1, 1},cut.second, 60, 0, 1500,true, false,label_phopt,"Events"));
        vh.push_back(PHS("Fragmentation_Photon_Pt_"+cut.first,{dcFragPhotonPt},{1, 1},cut.second, 60, 0, 1500,true, false,label_phopt,"Events"));
        //nPhoton
        vh.push_back(PHS("nPhotonNoID_"+cut.first,{dcPhotonCnt},{1, 1},cut.second, 5, 0, 5,false, false,"Photons", "Events"));
        vh.push_back(PHS("nPhoton_"+cut.first,{dcCutPhotonCnt},{1, 1},cut.second, 5, 0, 5,false, false,"Photons", "Events"));
        //Photon Pt
        vh.push_back(PHS("Total_Photon_Pt_"+cut.first,{dcPhoton_pt},{1, 1},cut.second, 60, 0, 1500,true, false,label_phopt, "Events"));
        vh.push_back(PHS("Photon_Pt_"+cut.first,{dcCutPhoton_pt},{1, 1},cut.second, 60, 0, 1500,true, false,label_phopt, "Events"));
        //Photon Eta
        vh.push_back(PHS("Total_Photon_eta_"+cut.first,{dcPhoton_eta},{1, 1},cut.second, 32, -4, 4,false, false, "#eta" , "Events"));
        vh.push_back(PHS("Photon_eta_"+cut.first,{dcCutPhoton_eta},{1, 1},cut.second, 32, -4, 4,false, false, "#eta" , "Events"));
        //Photon MET and MET Gamma
        vh.push_back(PHS("Photon_met_"+cut.first,{dcPhoton_met},{1, 2},cut.second, 60, 0, 1500,true, false,label_met , "Events"));
        vh.push_back(PHS("Photon_metGamma_"+cut.first,{dcMetGamma},{1, 2},cut.second, 60, 0, 1500,true, false,label_metg , "Events"));
        //Ntops photon
        vh.push_back(PHS("Photon_nt_"+cut.first,{dcPhoton_nt},{1, 1},cut.second, 5, 0, 5,false, false,label_nt, "Events"));
        //Photon MT2
        vh.push_back(PHS("Photon_mt2_"+cut.first,{dcPhoton_mt2},{1, 1},cut.second, 60, 0, 1500,true, false,label_mt2, "Events"));
        //NbJets photon
        vh.push_back(PHS("Photon_nb_"+cut.first,{dcPhoton_nb},{1, 1},cut.second, 10, 0, 10,false, false,label_nb, "Events"));
        //NJets
        vh.push_back(PHS("Photon_nj_"+cut.first,{dcPhoton_nj},{1, 1},cut.second, 30, 0, 30,false, false,label_nj, "Events"));
        //Photon HT 
        vh.push_back(PHS("Photon_ht_"+cut.first,{dcPhoton_ht},{1, 1},cut.second, 60, 0, 1500,true, false,label_ht, "Events"));

        //Make Data/MC plots
        //nPhoton
        vh.push_back(PHS("DataMC_SinglePhoton_np_"+cut.first,{dcCutPhotonCntData,dcCutPhotonCntMC},{1, 2},cut.second, 5, 0, 5,false, false,"Photons","Events"));
        //Photon Pt
        vh.push_back(PHS("DataMC_SinglePhoton_pt_"+cut.first,{dcCutPhotonPtData,dcCutPhotonPtMC},{1, 2},cut.second, 60, 0, 1500,true,false,label_phopt,"Events"));
        //Photon Eta
        vh.push_back(PHS("DataMC_SinglePhoton_eta_"+cut.first,{dcCutPhotonEtaData,dcCutPhotonEtaMC},{1, 2},cut.second, 32, -4, 4,false, false,"#eta","Events"));
        //Photon MET
        vh.push_back(PHS("DataMC_SinglePhoton_met_"+cut.first,{dcMetGammaData,dcMetGammaMC},{1, 2},cut.second, 60, 0, 1500,true, false,label_metg, "Events"));
        //Photon MT2                                                                                                                          
        vh.push_back(PHS("DataMC_SinglePhoton_mt2_"+cut.first,{dcPhotonMt2Data,dcPhotonMt2MC},{1, 2},cut.second, 60, 0, 1500,true, false,label_mt2, "Events"));
        //NbJets photon
        vh.push_back(PHS("DataMC_SinglePhoton_nb_"+cut.first,{dcPhotonNbData,dcPhotonNbMC},{1, 2},cut.second, 10, 0, 10,false, false,label_nb, "Events"));
        vh.push_back(PHS("DataMC_SinglePhoton_nb_Log_"+cut.first,{dcPhotonNbData,dcPhotonNbMC},{1, 2},cut.second, 10, 0, 10,true, false,label_nb, "Events"));
        //NJets
        vh.push_back(PHS("DataMC_SinglePhoton_nj_"+cut.first,{dcPhotonNjData,dcPhotonNjMC},{1, 2},cut.second, 30, 0, 30,false, false,label_nj, "Events"));
        vh.push_back(PHS("DataMC_SinglePhoton_nj_Log_"+cut.first,{dcPhotonNjData,dcPhotonNjMC},{1, 2},cut.second, 30, 0, 30,true, false,label_nj, "Events"));
        //NTops
        vh.push_back(PHS("DataMC_SinglePhoton_nt_"+cut.first,{dcPhotonNtData,dcPhotonNtMC},{1, 2},cut.second, 5, 0, 5,false, false,label_nt, "Events"));
        vh.push_back(PHS("DataMC_SinglePhoton_nt_Log_"+cut.first,{dcPhotonNtData,dcPhotonNtMC},{1, 2},cut.second, 5, 0, 5,true, false,label_nt, "Events"));
        //Photon HT
        vh.push_back(PHS("DataMC_SinglePhoton_ht_"+cut.first,{dcPhotonHtData,dcPhotonHtMC},{1, 2},cut.second, 60, 0, 1500,true, false,label_ht, "Events"));

        //data/MC comparison with shape corrections applied
        //nPhoton 
        vh.push_back(PHS("DataMCw_SinglePhoton_np_"+cut.first,{dcCutPhotonCntData,dcWCutPhotonCntMC},{1, 2},cut.second, 5, 0, 5,false, false,"Photons","Events"));
        //Photon Pt                                                                                                                                                                   
        vh.push_back(PHS("DataMCw_SinglePhoton_pt_"+cut.first,{dcCutPhotonPtData,dcWCutPhotonPtMC},{1, 2},cut.second, 60, 0, 1500,true,false,label_phopt,"Events"));
        //Photon Eta                                                                                                                                                                  
        vh.push_back(PHS("DataMCw_SinglePhoton_eta_"+cut.first,{dcCutPhotonEtaData,dcWCutPhotonEtaMC},{1, 2},cut.second, 32, -4, 4,false, false,"#eta","Events"));
        //Photon Met                                                                                                                                                    
        vh.push_back(PHS("DataMCw_SinglePhoton_met_"+cut.first,{dcMetGammaData,dcWMetGammaMC},{1, 2},cut.second, 60, 0, 1500,true, false,label_metg, "Events"));
        //Photon MT2                                                                                                                                                                  
        vh.push_back(PHS("DataMCw_SinglePhoton_mt2_"+cut.first,{dcPhotonMt2Data,dcWPhotonMt2MC},{1, 2},cut.second, 60, 0, 1500,true, false,label_mt2, "Events"));
        //NbJets photon                                                                                                                                                               
        vh.push_back(PHS("DataMCw_SinglePhoton_nb_"+cut.first,{dcPhotonNbData,dcWPhotonNbMC},{1, 2},cut.second, 10, 0, 10,false, false,label_nb, "Events"));
        vh.push_back(PHS("DataMCw_SinglePhoton_nb_Log_"+cut.first,{dcPhotonNbData,dcWPhotonNbMC},{1, 2},cut.second, 10, 0, 10,true, false,label_nb, "Events"));
        //NJets                                                                                                                                                                       
        vh.push_back(PHS("DataMCw_SinglePhoton_nj_"+cut.first,{dcPhotonNjData,dcWPhotonNjMC},{1, 2},cut.second, 30, 0, 30,false, false,label_nj, "Events"));
        vh.push_back(PHS("DataMCw_SinglePhoton_nj_Log_"+cut.first,{dcPhotonNjData,dcWPhotonNjMC},{1, 2},cut.second, 30, 0, 30,true, false,label_nj, "Events"));
        //NTops                                                                                                                                                                       
        vh.push_back(PHS("DataMCw_SinglePhoton_nt_"+cut.first,{dcPhotonNtData,dcWPhotonNtMC},{1, 2},cut.second, 5, 0, 5,false, false,label_nt, "Events"));
        vh.push_back(PHS("DataMCw_SinglePhoton_nt_Log_"+cut.first,{dcPhotonNtData,dcWPhotonNtMC},{1, 2},cut.second, 5, 0, 5,true, false,label_nt, "Events"));
        //Photon HT                                                                                                                                                                   
        vh.push_back(PHS("DataMCw_SinglePhoton_ht_"+cut.first,{dcPhotonHtData,dcWPhotonHtMC},{1, 2},cut.second, 60, 0, 1500,true, false,label_ht, "Events"));


        //data/MC comparison with shape corrections and normalization from tight mumu CR applied                                                                                     
        //nPhoton                                                                                                                                                                     
        vh.push_back(PHS("DataMCww_SinglePhoton_np_"+cut.first,{dcCutPhotonCntData,dcWWCutPhotonCntMC},{1, 2},cut.second, 5, 0, 5,false, false,"Photons","Events"));
        //Photon Pt                                                                                                                                                                   
        vh.push_back(PHS("DataMCww_SinglePhoton_pt_"+cut.first,{dcCutPhotonPtData,dcWWCutPhotonPtMC},{1, 2},cut.second, 60, 0, 1500,true,false,label_phopt,"Events"));
        //Photon Eta                                                                                                                                                                  
        vh.push_back(PHS("DataMCww_SinglePhoton_eta_"+cut.first,{dcCutPhotonEtaData,dcWWCutPhotonEtaMC},{1, 2},cut.second, 32, -4, 4,false, false,"#eta","Events"));
        //Photon MET and Met Gamma                                                                                                                                                    
        vh.push_back(PHS("DataMCww_SinglePhoton_met_"+cut.first,{dcMetGammaData,dcWWMetGammaMC},{1, 2},cut.second, 60, 0, 1500,true, false,label_metg, "Events"));
        //Photon MT2                                                                                                                                                                  
        vh.push_back(PHS("DataMCww_SinglePhoton_mt2_"+cut.first,{dcPhotonMt2Data,dcWWPhotonMt2MC},{1, 2},cut.second, 60, 0, 1500,true, false,label_mt2, "Events"));
        //NbJets photon                                                                                                                                                               
        vh.push_back(PHS("DataMCww_SinglePhoton_nb_"+cut.first,{dcPhotonNbData,dcWWPhotonNbMC},{1, 2},cut.second, 10, 0, 10,false, false,label_nb, "Events"));
        vh.push_back(PHS("DataMCww_SinglePhoton_nb_Log_"+cut.first,{dcPhotonNbData,dcWWPhotonNbMC},{1, 2},cut.second, 10, 0, 10,true, false,label_nb, "Events"));
        //NJets                                                                                                                                                                       
        vh.push_back(PHS("DataMCww_SinglePhoton_nj_"+cut.first,{dcPhotonNjData,dcWWPhotonNjMC},{1, 2},cut.second, 30, 0, 30,false, false,label_nj, "Events"));
        vh.push_back(PHS("DataMCww_SinglePhoton_nj_Log_"+cut.first,{dcPhotonNjData,dcWWPhotonNjMC},{1, 2},cut.second, 30, 0, 30,true, false,label_nj, "Events"));
        //NTops                                                                                                                                                                       
        vh.push_back(PHS("DataMCww_SinglePhoton_nt_"+cut.first,{dcPhotonNtData,dcWWPhotonNtMC},{1, 2},cut.second, 5, 0, 5,false, false,label_nt, "Events"));
        vh.push_back(PHS("DataMCww_SinglePhoton_nt_Log_"+cut.first,{dcPhotonNtData,dcWWPhotonNtMC},{1, 2},cut.second, 5, 0, 5,true, false,label_nt, "Events"));
        //Photon HT                                                                                                                                                                   
        vh.push_back(PHS("DataMCww_SinglePhoton_ht_"+cut.first,{dcPhotonHtData,dcWWPhotonHtMC},{1, 2},cut.second, 60, 0, 1500,true, false,label_ht, "Events"));

        //data/MC comparison with shape corrections and normalization from njets applied                                                                                              
        vh.push_back(PHS("DataMCwn_SinglePhoton_np_"+cut.first,{dcCutPhotonCntData,dcWNCutPhotonCntMC},{1, 2},cut.second, 5, 0, 5,false, false,"Photons","Events"));
        //Photon Pt                                                                                                                                                                   
        vh.push_back(PHS("DataMCwn_SinglePhoton_pt_"+cut.first,{dcCutPhotonPtData,dcWNCutPhotonPtMC},{1, 2},cut.second, 60, 0, 1500,true,false,label_phopt,"Events"));
        //Photon Eta                                                                                                                                                                  
        vh.push_back(PHS("DataMCwn_SinglePhoton_eta_"+cut.first,{dcCutPhotonEtaData,dcWNCutPhotonEtaMC},{1, 2},cut.second, 32, -4, 4,false, false,"#eta","Events"));
        //Photon MET and Met Gamma                                                                                                                                                    
        vh.push_back(PHS("DataMCwn_SinglePhoton_met_"+cut.first,{dcMetGammaData,dcWNMetGammaMC},{1, 2},cut.second, 60, 0, 1500,true, false,label_metg, "Events"));
        //Photon MT2                                                                                                                                                                  
        vh.push_back(PHS("DataMCwn_SinglePhoton_mt2_"+cut.first,{dcPhotonMt2Data,dcWNPhotonMt2MC},{1, 2},cut.second, 60, 0, 1500,true, false,label_mt2, "Events"));
        //NbJets photon                                                                                                                                                               
        vh.push_back(PHS("DataMCwn_SinglePhoton_nb_"+cut.first,{dcPhotonNbData,dcWNPhotonNbMC},{1, 2},cut.second, 10, 0, 10,false, false,label_nb, "Events"));
        vh.push_back(PHS("DataMCwn_SinglePhoton_nb_Log_"+cut.first,{dcPhotonNbData,dcWNPhotonNbMC},{1, 2},cut.second, 10, 0, 10,true, false,label_nb, "Events"));
        //NJets                                                                                                                                                                       
        vh.push_back(PHS("DataMCwn_SinglePhoton_nj_"+cut.first,{dcPhotonNjData,dcWNPhotonNjMC},{1, 2},cut.second, 30, 0, 30,false, false,label_nj, "Events"));
        vh.push_back(PHS("DataMCwn_SinglePhoton_nj_Log_"+cut.first,{dcPhotonNjData,dcWNPhotonNjMC},{1, 2},cut.second, 30, 0, 30,true, false,label_nj, "Events"));
        //NTops                                                                                                                                                                       
        vh.push_back(PHS("DataMCwn_SinglePhoton_nt_"+cut.first,{dcPhotonNtData,dcWNPhotonNtMC},{1, 2},cut.second, 5, 0, 5,false, false,label_nt, "Events"));
        vh.push_back(PHS("DataMCwn_SinglePhoton_nt_Log_"+cut.first,{dcPhotonNtData,dcWNPhotonNtMC},{1, 2},cut.second, 5, 0, 5,true, false,label_nt, "Events"));
        //Photon HT                                                                                                                                                                   
        vh.push_back(PHS("DataMCwn_SinglePhoton_ht_"+cut.first,{dcPhotonHtData,dcWNPhotonHtMC},{1, 2},cut.second, 60, 0, 1500,true, false,label_ht, "Events"));
    }

    //Generate cutflows 
    vector<string> cfsGamma = {"",
                               "passNoiseEventFilter",
                               "passNoiseEventFilter;passnJets",
                               "passNoiseEventFilter;passnJets;passdPhis",
                               "passNoiseEventFilter;passnJets;passdPhis",
                               "passNoiseEventFilter;passnJets;passdPhis;HT>300",
    };  
    vector<Plotter::CutFlowSummary> cutFlowSummaries;

    cutFlowSummaries.emplace_back(Plotter::CutFlowSummary("GJets",             PDC("", "", {dsPhoton}),            cfsGamma));

    set<AnaSamples::FileSummary> vvf;
    for(auto& fsVec : fileMap) for(auto& fs : fsVec.second) vvf.insert(fs);

    RegisterFunctions* rf = new RegisterFunctionsNTuple(runOnCondor, sbEra);

    Plotter plotter(vh, vvf, fromTuple, histFile, nFiles, startFile, nEvts);
    plotter.setCutFlows(cutFlowSummaries);
    plotter.setLumi(lumi);
    plotter.setPlotDir(plotDir);
    plotter.setDoHists(doSave || doPlots);
    plotter.setDoTuple(doTuple);
    plotter.setRegisterFunction(rf);
    plotter.read();
    if(doSave && fromTuple)  plotter.saveHists();
    if(doPlots)              plotter.plot();
}
