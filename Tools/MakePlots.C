#include "Plotter.h"
#include "samples.h"
#include "RegisterFunctions.h"
#include "SusyAnaTools/Tools/searchBins.h"
#include "SusyAnaTools/Tools/samples.h"
#include "SusyAnaTools/Tools/SusyUtility.h"
#include "TMath.h"

#include <getopt.h>
#include <iostream>

void stripRoot(std::string &path)
{
    int dot = path.rfind(".root");
    if (dot != std::string::npos)
    {
        path.resize(dot);
    }
}

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
        {"dophotons",        no_argument, 0, 'g'},
        {"doleptons",        no_argument, 0, 'l'},
        {"filename",   required_argument, 0, 'I'},
        {"dataSets",   required_argument, 0, 'D'},
        {"numFiles",   required_argument, 0, 'N'},
        {"startFile",  required_argument, 0, 'M'},
        {"numEvts",    required_argument, 0, 'E'},
        {"plotDir",    required_argument, 0, 'P'},
        {"luminosity", required_argument, 0, 'L'},
        {"sbEra",      required_argument, 0, 'S'}
    };

    bool runOnCondor = false;
    bool doWeights = false;
    bool doLeptons = false;
    bool doPhotons = false;
    bool doGJetsAndZnunu = false;
    bool doDYAndZnunu = true;
    bool doSearchBins = false;
    bool doPlots = true;
    bool doSave = true;
    bool doTuple = true;
    bool fromTuple = true;
    string filename = "histoutput.root", dataSets = "", sampleloc = AnaSamples::fileDir, plotDir = "plots";
    int nFiles = -1, startFile = 0, nEvts = -1;
    double lumi = AnaSamples::luminosity;
    std::string sbEra = "SB_v1_2017";//"SB_v1_2017";

    while((opt = getopt_long(argc, argv, "pstfcglI:D:N:M:E:P:L:S:", long_options, &option_index)) != -1)
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
        
        case 'g':
            doPhotons = true;
            break;

        case 'l':
            doLeptons = true;
            break;

        case 'I':
            filename = optarg;
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
        stripRoot(filename);
        sprintf(thistFile, "%s_%s_%d.root", filename.c_str(), dataSets.c_str(), startFile);
        filename = thistFile;
        std::cout << "Filename modified for use with condor: " << filename << std::endl;
        //doSave = true;
        //doPlots = false;
        fromTuple = true;
        doPhotons = true;
        doLeptons = false;
        sampleloc = "condor";
    }

    std::cout << "input filename: " << filename << std::endl;
    std::cout << "Sample location: " << sampleloc << std::endl;

    // follow this syntax; order matters for your arguments
    //SampleSet::SampleSet(std::string file, bool isCondor, double lumi)
    AnaSamples::SampleSet        ss("sampleSets.cfg", runOnCondor, AnaSamples::luminosity);
    //SampleCollection::SampleCollection(const std::string& file, SampleSet& samples) : ss_(samples)
    AnaSamples::SampleCollection sc("sampleCollections.cfg", ss);
    
    // // --- modify weights to compare GJets to ZJetsToNuNu
    // // --- do this before creating fileMap

    // //AnaSamples::FileSummary fsg = ss["GJets_HT-200To400"];
    // //printf("%s: xsec = %f kfactor = %f weight = %f\n", fsg.tag.c_str(), fsg.xsec, fsg.kfactor, fsg.getWeight());
    // 
    // std::vector<std::string> sampleTags1 = {"GJets_HT-200To400", "GJets_HT-400To600", "GJets_HT-600ToInf"};
    // std::vector<std::string> sampleTags2 = {"ZJetsToNuNu_HT_200to400", "ZJetsToNuNu_HT_400to600", "ZJetsToNuNu_HT_600to800", "ZJetsToNuNu_HT_800to1200", "ZJetsToNuNu_HT_1200to2500", "ZJetsToNuNu_HT_2500toInf"};
    // std::string sampleTag1 = "GJets";
    // std::string sampleTag2 = "ZJetsToNuNu";

    // if (ss[dataSets] != ss.null())
    // {
    //     printf("--- Using ss -----------\n");
    //     printf("--- Original Weights ---\n");
    //     for (const auto& tag : sampleTags1) printf("%s: weight = %f\n", tag.c_str(), ss[tag].getWeight());
    //     for (const auto& tag : sampleTags2) printf("%s: weight = %f\n", tag.c_str(), ss[tag].getWeight());

    //     // Definition in SusyAnaTools/Tools/samples.cc
    //     ss.modifyWeights(sampleTags1, sampleTags2);

    //     printf("--- Modified Weights ---\n");
    //     for (const auto& tag : sampleTags1) printf("%s: weight = %f\n", tag.c_str(), ss[tag].getWeight());
    //     for (const auto& tag : sampleTags2) printf("%s: weight = %f\n", tag.c_str(), ss[tag].getWeight());
    // }
    // else if (sc[dataSets] != sc.null())
    // {
    //     printf("--- Using sc -----------\n");
    //     printf("--- Original Weights ---\n");
    //     for (const auto& fs : sc[sampleTag1]) printf("%s: weight = %f\n", fs.tag.c_str(), fs.getWeight());
    //     for (const auto& fs : sc[sampleTag2]) printf("%s: weight = %f\n", fs.tag.c_str(), fs.getWeight());

    //     // Definition in SusyAnaTools/Tools/samples.cc
    //     sc.modifyWeights(sampleTag1, sampleTag2);
    //     
    //     printf("--- Modified Weights ---\n");
    //     for (const auto& fs : sc[sampleTag1]) printf("%s: weight = %f\n", fs.tag.c_str(), fs.getWeight());
    //     for (const auto& fs : sc[sampleTag2]) printf("%s: weight = %f\n", fs.tag.c_str(), fs.getWeight());
    // }


    const double zAcc = 1.0;
    // const double zAcc = 0.5954;
    // const double zAcc = 0.855;
    const double znunu_mumu_ratio = 5.942;
    const double znunu_ee_ratio   = 5.942;

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

    // min and max values for histos
    int nBins = 40;
    // p_t in GeV
    double minPt = 0.0;
    double maxPt = 2000.0;
    // Energy in GeV
    double minEnergy = 0.0;
    double maxEnergy = 2000.0;
    // mass in GeV
    // mass of electron: 0.511 MeV = 5.11 * 10^-4 GeV
    // mass of muon: 106 MeV = 0.106 GeV
    // mass of photon: 0.00 eV
    double minMassElec = 0.0;
    double maxMassElec = TMath::Power(10.0, -3);
    double minMassMu = 0.0;
    double maxMassMu = 0.2;
    double minMassPhoton = -2.0 * TMath::Power(10.0, -4);
    double maxMassPhoton =  2.0 * TMath::Power(10.0, -4);
    double minEta = -5.0;
    double maxEta = 5.0;
    double minPhi = -1.0 * TMath::Pi();
    double maxPhi = TMath::Pi();

    // Shortcuts for axis labels
    std::string label_Events = "Events";
    std::string label_met = "p_{T}^{miss} [GeV]";
    std::string label_metphi = "#phi_{MET}";
    std::string label_ht  = "H_{T} [GeV]";
    std::string label_mht = "MH_{T} [GeV]";
    std::string label_nj  = "N_{jets}";
    std::string label_nb  = "N_{bottoms}";
    std::string label_nt  = "N_{tops}";
    std::string label_dr  = "#DeltaR";
    std::string label_dphi0  = "#Delta#phi_{0}";
    std::string label_dphi1  = "#Delta#phi_{1}";
    std::string label_dphi2  = "#Delta#phi_{2}";
    std::string label_mt2 = "M_{T2} [GeV]";
    std::string label_eta = "#eta";
    std::string label_MuPt = "p_{T}^{#mu} [GeV]";
    std::string label_MuEnergy = "E^{#mu} [GeV]";
    std::string label_MuMass = "m^{#mu} [GeV]";
    std::string label_MuEta = "#eta^{#mu}";
    std::string label_MuPhi = "#phi^{#mu}";
    std::string label_genmupt  = "gen #mu p_{T} [GeV]";
    std::string label_genmueta = "gen #mu #eta";
    std::string label_Mu1pt = "#mu_{1} p_{T} [GeV]";
    std::string label_Mu2pt = "#mu_{2} p_{T} [GeV]";
    std::string label_Mu1eta = "#mu_{1} #eta";
    std::string label_Mu2eta = "#mu_{2} #eta";

    std::string label_ElecPt = "p_{T}^{e} [GeV]";
    std::string label_ElecEnergy = "E^{e} [GeV]";
    std::string label_ElecMass = "m^{e} [GeV]";
    std::string label_ElecEta = "#eta^{e}";
    std::string label_ElecPhi = "#phi^{e}";
    std::string label_Elec1pt = "e_{1} p_{T} [GeV]";
    std::string label_Elec2pt = "e_{2} p_{T} [GeV]";
    std::string label_PhotonPt = "p_{T}^{#gamma} [GeV]";
    std::string label_PhotonEnergy = "E^{#gamma} [GeV]";
    std::string label_PhotonMass = "m^{#gamma} [GeV]";
    std::string label_PhotonEta = "#eta^{#gamma}";
    std::string label_PhotonPhi = "#phi^{#gamma}";
    std::string label_jetpt  = "jet p_{T} [GeV]";
    std::string label_jeteta = "jet #eta [GeV]";
    std::string label_jetphi = "jet #phi [GeV]";
    std::string label_jetE   = "jet E [GeV]";
    std::string label_j1pt = "j_{1} p_{T} [GeV]";
    std::string label_j2pt = "j_{2} p_{T} [GeV]";
    std::string label_j3pt = "j_{3} p_{T} [GeV]";
    std::string label_mll  = "m_{ll} [GeV]";
    std::string label_topPt = "top p_{T} [GeV]";
    std::string label_genTopPt = "gen top p_{T} [GeV]";
    std::string label_phopt = "p_{T}^{#gamma} [GeV]";
    std::string label_metg = "p_{T}^{#gamma (miss)} [GeV]";
    std::string label_ptcut_single = "GenEtaPt & GenPt";
    std::string label_ptcut_ratio  = "GenEtaPt / GenPt";
    std::string label_acc_single = "RecoEta & Reco";
    std::string label_acc_ratio  = "RecoEta / Reco";
    std::string label_matched_single = "RecoEtaPtMatched & RecoEtaPt";
    std::string label_matched_ratio  = "RecoEtaPtMatched / RecoEtaPt";
    std::string label_iso_single = "RecoIso & RecoEtaPtMatched";
    std::string label_iso_ratio  = "RecoIso / RecoEtaPtMatched";
    // make a map of labels
    // there is no Gen Iso
    //std::vector<std::pair<std::string,std::string>> cutlevels_electrons
    std::map<std::string, std::string> label_map = {
        {"GenAcc_single",     "GenEta & Gen"},
        {"GenAcc_ratio",      "GenEta / Gen"},
        {"GenMatch_single",   "GenEtaPtMatched & GenEtaPt"},
        {"GenMatch_ratio",    "GenEtaPtMatched / GenEtaPt"},
        {"RecoAcc_single",    "RecoEta & Reco"},
        {"RecoAcc_ratio",     "RecoEta / Reco"},
        {"RecoIso_single",    "RecoIso & RecoEtaPt"},
        {"RecoIso_ratio",     "RecoIso / RecoEtaPt"},
        {"RecoMatch_single",  "RecoEtaPtMatched & RecoIso"},
        {"RecoMatch_ratio",   "RecoEtaPtMatched / RecoIso"},
    };

    vector<Plotter::HistSummary> vh;

    // Datasetsummaries we are using                                                                                                        
    // no weight (genWeight deals with negative weights); also add btag weights here                                                        
    Plotter::DatasetSummary dsData_SingleMuon("Data",         fileMap["Data_SingleMuon"], "passMuTrigger",   "");
    Plotter::DatasetSummary dsDY_mu(          "DY #mu",       fileMap["DYJetsToLL"],      "",        "muTrigWgt;bTagSF_EventWeightSimple_Central;_PUweightFactor");
    Plotter::DatasetSummary dsDYInc_mu(       "DY HT<100",    fileMap["IncDY"],           "genHT<100",   "muTrigWgt;bTagSF_EventWeightSimple_Central;_PUweightFactor");
    Plotter::DatasetSummary dsDY_elec(        "DY e",         fileMap["DYJetsToLL"],      "",          "bTagSF_EventWeightSimple_Central;_PUweightFactor"); // do not use muTrigWgt for electrons (it is 0.0)
    Plotter::DatasetSummary dsDYInc_elec(     "DY HT<100",    fileMap["IncDY"],           "genHT<100",   "bTagSF_EventWeightSimple_Central;_PUweightFactor"); // do not use muTrigWgt for electrons (it is 0.0)
    Plotter::DatasetSummary dsPhoton(         "#gamma+ jets", fileMap["GJets"],         "",            "bTagSF_EventWeightSimple_Central;_PUweightFactor");
    Plotter::DatasetSummary dstt2l(           "t#bar{t}",     fileMap["TTbarNoHad"],      "",            "muTrigWgt;bTagSF_EventWeightSimple_Central;isr_Unc_Cent;_PUweightFactor");
    Plotter::DatasetSummary dstW(             "Single t",     fileMap["SingleTopZinv"],   "",            "muTrigWgt;bTagSF_EventWeightSimple_Central;_PUweightFactor;genWeight");
    Plotter::DatasetSummary dsttZ(            "t#bar{t}Z",    fileMap["TTZ"],             "",            "muTrigWgt;bTagSF_EventWeightSimple_Central;genWeight;_PUweightFactor");
    Plotter::DatasetSummary dsT1tttt_gluino1200_lsp800("T1tttt_gluino1200_lsp800",     fileMap["Signal_T1tttt_mGluino1200_mLSP800"], "",  "");
    Plotter::DatasetSummary dsT1tttt_gluino1500_lsp100("T1tttt_gluino1500_lsp100",     fileMap["Signal_T1tttt_mGluino1500_mLSP100"], "",  "");
    Plotter::DatasetSummary dsT1tttt_gluino2000_lsp100("T1tttt_gluino2000_lsp100",     fileMap["Signal_T1tttt_mGluino2000_mLSP100"], "",  "");
    Plotter::DatasetSummary dsVV(             "Diboson",      fileMap["Diboson"],        "",            "muTrigWgt;bTagSF_EventWeightSimple_Central;_PUweightFactor;genWeight");
    Plotter::DatasetSummary dsRare(           "Rare ",        fileMap["Rare"],           "",            "muTrigWgt;bTagSF_EventWeightSimple_Central;genWeight;_PUweightFactor");
    std::vector<std::vector<Plotter::DatasetSummary>> stack_MC = {{dsDY_mu, dsDYInc_mu}, {dstt2l}, {dstW}, {dsRare, dsVV, dsttZ}};

    // Apply data/mc njet weight for DY and ttbar                                                                                                                                    
    Plotter::DatasetSummary dswDY(             "DY",         fileMap["DYJetsToLL"],      "",            "muTrigWgt;bTagSF_EventWeightSimple_Central;njWGJets;_PUweightFactor");
    Plotter::DatasetSummary dswDYInc(          "DY HT<100",  fileMap["IncDY"],           "genHT<100",   "muTrigWgt;bTagSF_EventWeightSimple_Central;njWGJets;_PUweightFactor");
    Plotter::DatasetSummary dswtt2l(           "t#bar{t}",   fileMap["TTbarNoHad"],      "",            "muTrigWgt;bTagSF_EventWeightSimple_Central;nJetWgtTTbar;isr_Unc_Cent;_PUweightFactor");
    Plotter::DatasetSummary dswtW(             "Single t",   fileMap["SingleTopZinv"],   "",            "muTrigWgt;bTagSF_EventWeightSimple_Central;_PUweightFactor;genWeight");
    Plotter::DatasetSummary dswttZ(            "t#bar{t}Z",  fileMap["TTZ"],             "",            "muTrigWgt;bTagSF_EventWeightSimple_Central;genWeight;_PUweightFactor");
    Plotter::DatasetSummary dswVV(             "Diboson",    fileMap["Diboson"],         "",            "muTrigWgt;bTagSF_EventWeightSimple_Central;_PUweightFactor;genWeight");
    Plotter::DatasetSummary dswRare(           "Rare ",      fileMap["Rare"],            "",            "muTrigWgt;bTagSF_EventWeightSimple_Central;genWeight;_PUweightFactor");
    std::vector<std::vector<Plotter::DatasetSummary>> stackw_MC = {{dswDY, dswDYInc}, {dswtt2l}, {dswtW}, {dswRare, dswVV, dswttZ}};

    Plotter::DatasetSummary dswwDY(             "DY",         fileMap["DYJetsToLL"],      "",            "bTagSF_EventWeightSimple_Central;njWGJets;normWgt0b");
    Plotter::DatasetSummary dswwDYInc(          "DY HT<100",  fileMap["IncDY"],           "genHT<100",   "bTagSF_EventWeightSimple_Central;njWGJets;normWgt0b");
    std::vector<std::vector<Plotter::DatasetSummary>> stackww_MC = {{dswwDY, dswwDYInc}, {dswtt2l}, {dswtW}, {dswttZ}, {dswVV}, {dswRare, dswVV, dswttZ}};
    
 
    //lambda is your friend
    //for electrons do not use muTrigWgt (it is 0.0 for electrons)
    auto makePDSMu     = [&](const std::string& label) {return Plotter::DatasetSummary("DYJetsToLL "+label, fileMap["DYJetsToLL"], "", "muTrigWgt;bTagSF_EventWeightSimple_Central;_PUweightFactor"); };
    auto makePDSElec   = [&](const std::string& label) {return Plotter::DatasetSummary("DYJetsToLL "+label, fileMap["DYJetsToLL"], "", "bTagSF_EventWeightSimple_Central;_PUweightFactor"); };
    auto makePDSPhoton = [&](const std::string& label, const std::string& sample="GJets", const std::string& cuts="passPhotonSelection;HTZinv>200") {return Plotter::DatasetSummary("GJets "+label, fileMap[sample], cuts, "photonAcceptanceWeight;photonEfficiencyPtWeight;photonCrossSectionRatio"); };
    
    // acceptance
    // muons
    Plotter::DataCollection dcMC_ngenMu(            "single", "ngenMu",                     {dsDY_mu});
    Plotter::DataCollection dcMC_ngenMatchMu(       "single", "ngenMatchMu",                {dsDY_mu});
    Plotter::DataCollection dcMC_ngenMuInAcc(       "single", "ngenMuInAcc",                {dsDY_mu});
    Plotter::DataCollection dcMC_ngenMatchMuInAcc(  "single", "ngenMatchMuInAcc",           {dsDY_mu});
    Plotter::DataCollection dcMC_genMuPt(           "single", "genMu(pt)",                  {dsDY_mu});
    Plotter::DataCollection dcMC_genMuInAccPt(      "single", "genMuInAcc(pt)",             {dsDY_mu});
    Plotter::DataCollection dcMC_genMatchMuInAccPt( "single", "genMatchMuInAcc(pt)",        {dsDY_mu});
    Plotter::DataCollection dcMC_genMuEta(          "single", "genMu(eta)",                 {dsDY_mu});
    Plotter::DataCollection dcMC_genMuInAccEta(     "single", "genMuInAcc(eta)",            {dsDY_mu});
    Plotter::DataCollection dcMC_genMatchMuInAccEta("single", "genMatchMuInAcc(eta)",       {dsDY_mu});
    // electrons
    Plotter::DataCollection dcMC_ngenElec(            "single", "ngenElec",                 {dsDY_elec});
    Plotter::DataCollection dcMC_ngenMatchElec(       "single", "ngenMatchElec",            {dsDY_elec});
    Plotter::DataCollection dcMC_ngenElecInAcc(       "single", "ngenElecInAcc",            {dsDY_elec});
    Plotter::DataCollection dcMC_ngenMatchElecInAcc(  "single", "ngenMatchElecInAcc",       {dsDY_elec});
    Plotter::DataCollection dcMC_genElecPt(           "single", "genElec(pt)",              {dsDY_elec});
    Plotter::DataCollection dcMC_genElecInAccPt(      "single", "genElecInAcc(pt)",         {dsDY_elec});
    Plotter::DataCollection dcMC_genMatchElecInAccPt( "single", "genMatchElecInAcc(pt)",    {dsDY_elec});
    Plotter::DataCollection dcMC_genElecEta(          "single", "genElec(eta)",             {dsDY_elec});
    Plotter::DataCollection dcMC_genElecInAccEta(     "single", "genElecInAcc(eta)",        {dsDY_elec});
    Plotter::DataCollection dcMC_genMatchElecInAccEta("single", "genMatchElecInAcc(eta)",   {dsDY_elec});
    // magic lambda functions... give it pt, eta, etc
    // leptons
    // acceptance = genInAcc / gen (acceptance / MC)
    auto makePDCElecAcc_single = [&](const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"genElecInAcc("+var+")", makePDSElec("e acc")}, {"genElec("+var+")", makePDSElec("e gen")}}); };
    auto makePDCMuAcc_single   = [&](const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"genMuInAcc("+var+")",   makePDSMu("#mu acc")}, {"genMu("+var+")",   makePDSMu("#mu gen")}}); };
    auto makePDCElecAcc_ratio = [&](const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"genElecInAcc("+var+")", makePDSElec("e acc over gen")}, {"genElec("+var+")", makePDSElec("e acc over gen")}}); };
    auto makePDCMuAcc_ratio   = [&](const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"genMuInAcc("+var+")",   makePDSMu("#mu acc over gen")}, {"genMu("+var+")",   makePDSMu("#mu acc over gen")}}); };
    // reco efficiency = genMatchInAcc / genInAcc (reco / acceptance)
    auto makePDCElecReco_single = [&](const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"genMatchElecInAcc("+var+")", makePDSElec("e reco")}, {"genElecInAcc("+var+")", makePDSElec("e acc")}}); };
    auto makePDCMuReco_single   = [&](const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"genMatchMuInAcc("+var+")",   makePDSMu("#mu reco")}, {"genMuInAcc("+var+")",   makePDSMu("#mu acc")}}); };
    auto makePDCElecReco_ratio = [&](const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"genMatchElecInAcc("+var+")", makePDSElec("e reco over acc")}, {"genElecInAcc("+var+")", makePDSElec("e reco over acc")}}); };
    auto makePDCMuReco_ratio   = [&](const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"genMatchMuInAcc("+var+")",   makePDSMu("#mu reco over acc")}, {"genMuInAcc("+var+")",   makePDSMu("#mu reco over acc")}}); };
    // iso efficiency = genMatchIsoInAcc / genMatchInAcc (iso / reco)
    auto makePDCElecIso_single  = [&](const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"genMatchIsoElecInAcc("+var+")", makePDSElec("e iso")}, {"genMatchElecInAcc("+var+")", makePDSElec("e reco")}}); };
    auto makePDCMuIso_single    = [&](const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"genMatchIsoMuInAcc("+var+")",   makePDSMu("#mu iso")}, {"genMatchMuInAcc("+var+")",   makePDSMu("#mu reco")}}); };
    auto makePDCElecIso_ratio  = [&](const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"genMatchIsoElecInAcc("+var+")", makePDSElec("e iso over reco")}, {"genMatchElecInAcc("+var+")", makePDSElec("e iso over reco")}}); };
    auto makePDCMuIso_ratio    = [&](const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"genMatchIsoMuInAcc("+var+")",   makePDSMu("#mu iso over reco")}, {"genMatchMuInAcc("+var+")",   makePDSMu("#mu iso over reco")}}); };
    // photons
    // Tag must be Gen or Reco
    // acceptance = photonLVecTagEtaPt / photonLVecTag (acc / tag)
    auto makePDCPhotonAcc_single   = [&](const std::string& tag, const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"photonLVec"+tag+"Eta("+var+")", makePDSPhoton(tag+"Eta")},           {"photonLVec"+tag+"("+var+")", makePDSPhoton(tag)}}); };
    auto makePDCPhotonAcc_ratio    = [&](const std::string& tag, const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"photonLVec"+tag+"Eta("+var+")", makePDSPhoton(tag+"Eta over "+tag)}, {"photonLVec"+tag+"("+var+")", makePDSPhoton(tag+"Eta over "+tag)}}); };
    // iso efficiency = photonLVecTagIso / photonLVecTagEtaPt (iso / acc)
    // iso is only for reco!
    auto makePDCPhotonIso_single   = [&](const std::string& tag, const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"photonLVec"+tag+"Iso("+var+")", makePDSPhoton(tag+"Iso")},                   {"photonLVec"+tag+"EtaPt("+var+")", makePDSPhoton(tag+"EtaPt")}}); };
    auto makePDCPhotonIso_ratio    = [&](const std::string& tag, const std::string& var, const std::string& style) {return Plotter::DataCollection(style, {{"photonLVec"+tag+"Iso("+var+")", makePDSPhoton(tag+"Iso over "+tag+"EtaPt")}, {"photonLVec"+tag+"EtaPt("+var+")", makePDSPhoton(tag+"Iso over "+tag+"EtaPt")}}); };
    // Gen matched to Reco efficiency = photonLVecGenMatched / photonLVecGenEtaPt  (match / acc)
    // Reco matched to Gen efficiency = photonLVecRecoMatched / photonLVecRecoIso  (match / iso)
    auto makePDCPhotonMatch_single = [&](const std::string& tag, const std::string& var, const std::string& style) 
    {
        if (tag.compare("Gen") == 0)    return Plotter::DataCollection(style, {{"photonLVec"+tag+"EtaPtMatched("+var+")", makePDSPhoton(tag+"EtaPtMatched")}, {"photonLVec"+tag+"EtaPt("+var+")", makePDSPhoton(tag+"EtaPt")}}); 
        if (tag.compare("Reco") == 0)   return Plotter::DataCollection(style, {{"photonLVec"+tag+"EtaPtMatched("+var+")", makePDSPhoton(tag+"EtaPtMatched")}, {"photonLVec"+tag+"Iso("+var+")", makePDSPhoton(tag+"Iso")}});
        printf("ERROR: Tag must be Gen or Reco for Photon matching.\n");
    };
    auto makePDCPhotonMatch_ratio  = [&](const std::string& tag, const std::string& var, const std::string& style)
    {
        if (tag.compare("Gen") == 0)    return Plotter::DataCollection(style, {{"photonLVec"+tag+"EtaPtMatched("+var+")", makePDSPhoton(tag+"EtaPtMatched over "+tag+"EtaPt")}, {"photonLVec"+tag+"EtaPt("+var+")", makePDSPhoton(tag+"EtaPtMatched over "+tag+"EtaPt")}}); 
        if (tag.compare("Reco") == 0)   return Plotter::DataCollection(style, {{"photonLVec"+tag+"EtaPtMatched("+var+")", makePDSPhoton(tag+"EtaPtMatched over "+tag+"Iso")},   {"photonLVec"+tag+"Iso("+var+")", makePDSPhoton(tag+"EtaPtMatched over "+tag+"Iso")}}); 
        printf("ERROR: Tag must be Gen or Reco for Photon matching.\n");
    };
    

    // photons gen: photonLVecGen
    // photons acc: photonLVecGenEtaPt
    // photons reco: photonLVecGenRecoMatched
    // photons iso: photonLVecGenIso
    Plotter::DataCollection dcMC_genPhotonPt(     "single", "photonLVecGenEtaPt(pt)",      {dsPhoton});
    Plotter::DataCollection dcMC_genPhotonEta(    "single", "photonLVecGenEtaPt(eta)",     {dsPhoton});
    Plotter::DataCollection dcMC_matchedPhotonPt( "single", "promptPhotons(pt)",        {dsPhoton});
    Plotter::DataCollection dcMC_matchedPhotonEta("single", "promptPhotons(eta)",       {dsPhoton});

    // tops
    Plotter::DataCollection dcMC_T1tttt("single",  "genTops(pt)", {dsT1tttt_gluino1200_lsp800, dsT1tttt_gluino1500_lsp100, dsT1tttt_gluino2000_lsp100});

    // nj                                                                                                                                                                            
    Plotter::DataCollection dcData_SingleMuon_nj("data",   "cntNJetsPt30Eta24Zinv", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_nj(             "stack",  "cntNJetsPt30Eta24Zinv", stack_MC);
    Plotter::DataCollection dcwMC_nj(            "stack",  "cntNJetsPt30Eta24Zinv", stackw_MC);
    Plotter::DataCollection dcwwMC_nj(           "stack",  "cntNJetsPt30Eta24Zinv", stackww_MC);
 
    // gen Z pt: genZPt
    Plotter::DataCollection dcData_DY_gen_pt("data",  "genZPt", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_DY_gen_pt("stack",   "genZPt", stack_MC);
  
    // reco Z pt: bestRecoZPt
    Plotter::DataCollection dcData_DY_reco_pt("data",  "bestRecoZPt", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_DY_reco_pt("stack",   "bestRecoZPt", stack_MC);

    // gen Z eta
    Plotter::DataCollection dcData_DY_eta("data",  "genZEta", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_DY_eta("stack",   "genZEta", stack_MC);

    // gen Z phi
    Plotter::DataCollection dcData_DY_phi("data",  "genZPhi", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_DY_phi("stack",   "genZPhi", stack_MC);

    // invariant mass
    Plotter::DataCollection dcData_DY_mass("data",  "genZmass", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_DY_mass("stack",   "genZmass", stack_MC);

    // met                                                                                                                                                                           
    Plotter::DataCollection dcData_SingleMuon_met("data",   "cleanMetPt", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_met(             "stack",  "cleanMetPt", stack_MC);
    Plotter::DataCollection dcwMC_met(            "stack",  "cleanMetPt", stackw_MC);
    Plotter::DataCollection dcwwMC_met(           "stack",  "cleanMetPt", stackww_MC);
    // ntops                                                                                                                                                                         
    Plotter::DataCollection dcData_SingleMuon_nt("data",   "nTopCandSortedCntZinv", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_nt(             "stack",  "nTopCandSortedCntZinv", stack_MC);
    Plotter::DataCollection dcwMC_nt(            "stack",  "nTopCandSortedCntZinv", stackw_MC);
    Plotter::DataCollection dcwwMC_nt(           "stack",  "nTopCandSortedCntZinv", stackww_MC);
    // MT2                                                                                                                                                                           
    Plotter::DataCollection dcData_SingleMuon_mt2("data",   "best_had_brJet_MT2Zinv", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_mt2(             "stack",  "best_had_brJet_MT2Zinv", stack_MC);
    Plotter::DataCollection dcwMC_mt2(            "stack",  "best_had_brJet_MT2Zinv", stackw_MC);
    Plotter::DataCollection dcwwMC_mt2(           "stack",  "best_had_brJet_MT2Zinv", stackww_MC);
    // nb                                                                                                                                  
    Plotter::DataCollection dcData_SingleMuon_nb("data",   "cntCSVSZinv", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_nb(             "stack",  "cntCSVSZinv", stack_MC);
    Plotter::DataCollection dcwMC_nb(            "stack",  "cntCSVSZinv", stackw_MC);
    Plotter::DataCollection dcwwMC_nb(           "stack",  "cntCSVSZinv", stackww_MC);
    // ht                                                                                                                                                                            
    Plotter::DataCollection dcData_SingleMuon_ht("data",   "HTZinv", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_ht(             "stack",  "HTZinv", stack_MC);
    Plotter::DataCollection dcwMC_ht(            "stack",  "HTZinv", stackw_MC);
    Plotter::DataCollection dcwwMC_ht(           "stack",  "HTZinv", stackww_MC);
    // gen mu pt
    // gen mu eta
    // mu1pt                                                                                                                                                                         
    Plotter::DataCollection dcData_SingleMuon_mu1pt("data",   "cutMuPt1", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_mu1pt(             "stack",  "cutMuPt1", stack_MC);
    Plotter::DataCollection dcwMC_mu1pt(            "stack",  "cutMuPt1", stackw_MC);
    // mu1eta                                                                                                                                                                         
    Plotter::DataCollection dcData_SingleMuon_mu1eta("data",   "cutMuEta1", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_mu1eta(             "stack",  "cutMuEta1", stack_MC);
    Plotter::DataCollection dcwMC_mu1eta(            "stack",  "cutMuEta1", stackw_MC);
    // mu2pt                                                                                                                                                                         
    Plotter::DataCollection dcData_SingleMuon_mu2pt("data",   "cutMuPt2", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_mu2pt(             "stack",  "cutMuPt2", stack_MC);
    Plotter::DataCollection dcwMC_mu2pt(            "stack",  "cutMuPt2", stackw_MC);
    // mu2eta                                                                                                                                                                         
    Plotter::DataCollection dcData_SingleMuon_mu2eta("data",   "cutMuEta2", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_mu2eta(             "stack",  "cutMuEta2", stack_MC);
    Plotter::DataCollection dcwMC_mu2eta(            "stack",  "cutMuEta2", stackw_MC);
    // mll                                                                                                                                                                           
    Plotter::DataCollection dcData_SingleMuon_mll("data",   "bestRecoZM", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_mll(             "stack",  "bestRecoZM", stack_MC);
    Plotter::DataCollection dcwMC_mll(            "stack",  "bestRecoZM", stackw_MC);
    // nsearchbins                                                                                                                                                                   
    Plotter::DataCollection dcData_SingleMuon_nSearchBin("data",   "nSearchBin", {dsData_SingleMuon});
    Plotter::DataCollection dcMC_nSearchBin(             "stack",  "nSearchBin", stack_MC);
    Plotter::DataCollection dcwMC_nSearchBin(            "stack",  "nSearchBin", stackw_MC);
    Plotter::DataCollection dcwwMC_nSearchBin(           "stack",  "nSearchBin", stackww_MC);

    Plotter::DataCollection dcData_SingleMuon_mht("data",   "cleanMHt", {dsData_SingleMuon});
    Plotter::DataCollection dcwwMC_mht(           "stack",  "cleanMHt", stackww_MC);
    Plotter::DataCollection dcwMC_mht(            "stack",  "cleanMHt", stackw_MC);


    // electron cut levels
    std::vector<std::pair<std::string,std::string>> cutlevels_electrons = {
        //{"nothing",                   ""},
        {"nosel",                     "passNoiseEventFilterZinv"},
        //{"elecZinv",                  "passNoiseEventFilterZinv;passElecZinvSel"},
    };

    // muon cut levels; commented some cut leves to make less plots
    std::vector<std::pair<std::string,std::string>> cutlevels_muon = {
        //{"nothing",                   ""},
        {"nosel",                     "passNoiseEventFilterZinv"},
        //{"muZinv",                    "passNoiseEventFilterZinv;passMuZinvSel"},
        //{"muZinv_blnotag",            "passMuZinvSel;passBaselineNoTagZinv"},
        //{"muZinv_bl",                 "passMuZinvSel;passBaselineZinv"},
        //{"muZinv_0b_blnotag",         "passMuZinvSel;cntCSVSZinv=0;passBaselineNoTagZinv"},
        //{"muZinv_g1b_blnotag",        "passMuZinvSel;passBJetsZinv;passBaselineNoTagZinv"},
        //{"muZinv_g1b",                "passNoiseEventFilterZinv;passMuZinvSel;passBJetsZinv"},
        //{"muZinv_0b_loose0",          "passNoiseEventFilterZinv;passMuZinvSel;cntCSVSZinv=0;HTZinv>300;passnJetsZinv;passdPhisZinv"},
        //{"muZinv_g1b_loose0",         "passNoiseEventFilterZinv;passMuZinvSel;passBJetsZinv;HTZinv>200;passnJetsZinv;passdPhisZinv"},
        //{"muZinv_loose0",             "passNoiseEventFilterZinv;passMuZinvSel;HTZinv>300;passnJetsZinv;passdPhisZinv"},
    };

    // for reference from Plotter.h
    // HistSummary(std::string l, std::vector<Plotter::DataCollection> ns, std::pair<int, int> ratio, std::string cuts, int nb, double ll, double ul, bool log, bool norm, std::string xal, std::string yal, bool isRatio = true);
    // HistSummary(std::string l, std::vector<Plotter::DataCollection> ns, std::pair<int, int> ratio, std::string cuts, std::vector<double> be, bool log, bool norm, std::string xal, std::string yal, bool isRatio = true);

    // structs: ordering is the same as the HistSummary constructor
    struct plotStruct
    {
        std::string particle;                   // Elec, Mu, etc.
        std::string measurement;                // Acc, RecoEff, IsoEff, etc.
        std::string variable;                   // Pt, Eta, Phi, Energy, Mass, etc.
        //std::string shortVar;                 // pt, eta, phi, E, M, etc.
        Plotter::DataCollection dataCollection; // Data Collection returned by makePDCElecAcc, makePDCElecReco, etc.
        std::string style;                      // ratio, single, etc.
        int nbins;                              // number of bins
        double xMin;                            // x min for histo
        double xMax;                            // x max for histo
        bool logBool;                           // bool for log scale (true for pt)
        bool normBool;                          // bool for norm (false)
        std::string xLabel;                     // x axis label
        std::string yLabel;                     // y axis label
        bool ratioBool;                         // bool for ratio (true)
    };

    // vector of structs

    // electrons
    // measurements: Acc, Reco, Iso
    // variables: Pt, Energy, Mass Eta, Phi
    
    std::vector<plotStruct> plotParamsElec;
    
    // Acc
    plotParamsElec.push_back({"Elec", "Acc", "Pt",     makePDCElecAcc_single("pt","single"),  "single", nBins, minPt, maxPt, true, false, label_ElecPt, label_acc_single, true});
    plotParamsElec.push_back({"Elec", "Acc", "Energy", makePDCElecAcc_single("E","single"),   "single", nBins, minEnergy, maxEnergy, true, false, label_ElecEnergy, label_acc_single, true});
    plotParamsElec.push_back({"Elec", "Acc", "Mass",   makePDCElecAcc_single("M","single"),   "single", nBins, minMassElec, maxMassElec, false, false, label_ElecMass, label_acc_single, true});
    plotParamsElec.push_back({"Elec", "Acc", "Eta",    makePDCElecAcc_single("eta","single"), "single", nBins, minEta, maxEta, false, false, label_ElecEta, label_acc_single, true});
    plotParamsElec.push_back({"Elec", "Acc", "Phi",    makePDCElecAcc_single("phi","single"), "single", nBins, minPhi, maxPhi, false, false, label_ElecPhi, label_acc_single, true});
    plotParamsElec.push_back({"Elec", "Acc", "Pt",     makePDCElecAcc_ratio("pt","ratio"),    "ratio",  nBins, minPt, maxPt, false, false, label_ElecPt, label_acc_ratio, true});
    plotParamsElec.push_back({"Elec", "Acc", "Energy", makePDCElecAcc_ratio("E","ratio"),     "ratio",  nBins, minEnergy, maxEnergy, false, false, label_ElecEnergy, label_acc_ratio, true});
    plotParamsElec.push_back({"Elec", "Acc", "Mass",   makePDCElecAcc_ratio("M","ratio"),     "ratio",  nBins, minMassElec, maxMassElec, false, false, label_ElecMass, label_acc_ratio, true});
    plotParamsElec.push_back({"Elec", "Acc", "Eta",    makePDCElecAcc_ratio("eta","ratio"),   "ratio",  nBins, minEta, maxEta, false, false, label_ElecEta, label_acc_ratio, true});
    plotParamsElec.push_back({"Elec", "Acc", "Phi",    makePDCElecAcc_ratio("phi","ratio"),   "ratio",  nBins, minPhi, maxPhi, false, false, label_ElecPhi, label_acc_ratio, true});
    // Reco
    plotParamsElec.push_back({"Elec", "Reco", "Pt",     makePDCElecReco_single("pt","single"),  "single", nBins, minPt, maxPt, true, false, label_ElecPt, label_matched_single, true});
    plotParamsElec.push_back({"Elec", "Reco", "Energy", makePDCElecReco_single("E","single"),   "single", nBins, minEnergy, maxEnergy, true, false, label_ElecEnergy, label_matched_single, true});
    plotParamsElec.push_back({"Elec", "Reco", "Mass",   makePDCElecReco_single("M","single"),   "single", nBins, minMassElec, maxMassElec, false, false, label_ElecMass, label_matched_single, true});
    plotParamsElec.push_back({"Elec", "Reco", "Eta",    makePDCElecReco_single("eta","single"), "single", nBins, minEta, maxEta, false, false, label_ElecEta, label_matched_single, true});
    plotParamsElec.push_back({"Elec", "Reco", "Phi",    makePDCElecReco_single("phi","single"), "single", nBins, minPhi, maxPhi, false, false, label_ElecPhi, label_matched_single, true});
    plotParamsElec.push_back({"Elec", "Reco", "Pt",     makePDCElecReco_ratio("pt","ratio"),    "ratio",  nBins, minPt, maxPt, false, false, label_ElecPt, label_matched_ratio, true});
    plotParamsElec.push_back({"Elec", "Reco", "Energy", makePDCElecReco_ratio("E","ratio"),     "ratio",  nBins, minEnergy, maxEnergy, false, false, label_ElecEnergy, label_matched_ratio, true});
    plotParamsElec.push_back({"Elec", "Reco", "Mass",   makePDCElecReco_ratio("M","ratio"),     "ratio",  nBins, minMassElec, maxMassElec, false, false, label_ElecMass, label_matched_ratio, true});
    plotParamsElec.push_back({"Elec", "Reco", "Eta",    makePDCElecReco_ratio("eta","ratio"),   "ratio",  nBins, minEta, maxEta, false, false, label_ElecEta, label_matched_ratio, true});
    plotParamsElec.push_back({"Elec", "Reco", "Phi",    makePDCElecReco_ratio("phi","ratio"),   "ratio",  nBins, minPhi, maxPhi, false, false, label_ElecPhi, label_matched_ratio, true});
    // Iso
    plotParamsElec.push_back({"Elec", "Iso", "Pt",     makePDCElecIso_single("pt","single"),  "single", nBins, minPt, maxPt, true, false, label_ElecPt, label_iso_single, true});
    plotParamsElec.push_back({"Elec", "Iso", "Energy", makePDCElecIso_single("E","single"),   "single", nBins, minEnergy, maxEnergy, true, false, label_ElecEnergy, label_iso_single, true});
    plotParamsElec.push_back({"Elec", "Iso", "Mass",   makePDCElecIso_single("M","single"),   "single", nBins, minMassElec, maxMassElec, false, false, label_ElecMass, label_iso_single, true});
    plotParamsElec.push_back({"Elec", "Iso", "Eta",    makePDCElecIso_single("eta","single"), "single", nBins, minEta, maxEta, false, false, label_ElecEta, label_iso_single, true});
    plotParamsElec.push_back({"Elec", "Iso", "Phi",    makePDCElecIso_single("phi","single"), "single", nBins, minPhi, maxPhi, false, false, label_ElecPhi, label_iso_single, true});
    plotParamsElec.push_back({"Elec", "Iso", "Pt",     makePDCElecIso_ratio("pt","ratio"),    "ratio",  nBins, minPt, maxPt, false, false, label_ElecPt, label_iso_ratio, true});
    plotParamsElec.push_back({"Elec", "Iso", "Energy", makePDCElecIso_ratio("E","ratio"),     "ratio",  nBins, minEnergy, maxEnergy, false, false, label_ElecEnergy, label_iso_ratio, true});
    plotParamsElec.push_back({"Elec", "Iso", "Mass",   makePDCElecIso_ratio("M","ratio"),     "ratio",  nBins, minMassElec, maxMassElec, false, false, label_ElecMass, label_iso_ratio, true});
    plotParamsElec.push_back({"Elec", "Iso", "Eta",    makePDCElecIso_ratio("eta","ratio"),   "ratio",  nBins, minEta, maxEta, false, false, label_ElecEta, label_iso_ratio, true});
    plotParamsElec.push_back({"Elec", "Iso", "Phi",    makePDCElecIso_ratio("phi","ratio"),   "ratio",  nBins, minPhi, maxPhi, false, false, label_ElecPhi, label_iso_ratio, true});
    
    // muons
    // measurements: Acc, Reco, Iso
    // variables: Pt, Energy, Mass Eta, Phi
    
    std::vector<plotStruct> plotParamsMu;
    
    // Acc
    plotParamsMu.push_back({"Mu", "Acc", "Pt",     makePDCMuAcc_single("pt","single"),  "single", nBins, minPt, maxPt, true, false, label_MuPt, label_acc_single, true});
    plotParamsMu.push_back({"Mu", "Acc", "Energy", makePDCMuAcc_single("E","single"),   "single", nBins, minEnergy, maxEnergy, true, false, label_MuEnergy, label_acc_single, true});
    plotParamsMu.push_back({"Mu", "Acc", "Mass",   makePDCMuAcc_single("M","single"),   "single", nBins, minMassMu, maxMassMu, false, false, label_MuMass, label_acc_single, true});
    plotParamsMu.push_back({"Mu", "Acc", "Eta",    makePDCMuAcc_single("eta","single"), "single", nBins, minEta, maxEta, false, false, label_MuEta, label_acc_single, true});
    plotParamsMu.push_back({"Mu", "Acc", "Phi",    makePDCMuAcc_single("phi","single"), "single", nBins, minPhi, maxPhi, false, false, label_MuPhi, label_acc_single, true});
    plotParamsMu.push_back({"Mu", "Acc", "Pt",     makePDCMuAcc_ratio("pt","ratio"),   "ratio",  nBins, minPt, maxPt, false, false, label_MuPt, label_acc_ratio, true});
    plotParamsMu.push_back({"Mu", "Acc", "Energy", makePDCMuAcc_ratio("E","ratio"),    "ratio",  nBins, minEnergy, maxEnergy, false, false, label_MuEnergy, label_acc_ratio, true});
    plotParamsMu.push_back({"Mu", "Acc", "Mass",   makePDCMuAcc_ratio("M","ratio"),    "ratio",  nBins, minMassMu, maxMassMu, false, false, label_MuMass, label_acc_ratio, true});
    plotParamsMu.push_back({"Mu", "Acc", "Eta",    makePDCMuAcc_ratio("eta","ratio"),  "ratio",  nBins, minEta, maxEta, false, false, label_MuEta, label_acc_ratio, true});
    plotParamsMu.push_back({"Mu", "Acc", "Phi",    makePDCMuAcc_ratio("phi","ratio"),  "ratio",  nBins, minPhi, maxPhi, false, false, label_MuPhi, label_acc_ratio, true});
    // Reco
    plotParamsMu.push_back({"Mu", "Reco", "Pt",     makePDCMuReco_single("pt","single"),  "single", nBins, minPt, maxPt, true, false, label_MuPt, label_matched_single, true});
    plotParamsMu.push_back({"Mu", "Reco", "Energy", makePDCMuReco_single("E","single"),   "single", nBins, minEnergy, maxEnergy, true, false, label_MuEnergy, label_matched_single, true});
    plotParamsMu.push_back({"Mu", "Reco", "Mass",   makePDCMuReco_single("M","single"),   "single", nBins, minMassMu, maxMassMu, false, false, label_MuMass, label_matched_single, true});
    plotParamsMu.push_back({"Mu", "Reco", "Eta",    makePDCMuReco_single("eta","single"), "single", nBins, minEta, maxEta, false, false, label_MuEta, label_matched_single, true});
    plotParamsMu.push_back({"Mu", "Reco", "Phi",    makePDCMuReco_single("phi","single"), "single", nBins, minPhi, maxPhi, false, false, label_MuPhi, label_matched_single, true});
    plotParamsMu.push_back({"Mu", "Reco", "Pt",     makePDCMuReco_ratio("pt","ratio"),   "ratio",  nBins, minPt, maxPt, false, false, label_MuPt, label_matched_ratio, true});
    plotParamsMu.push_back({"Mu", "Reco", "Energy", makePDCMuReco_ratio("E","ratio"),    "ratio",  nBins, minEnergy, maxEnergy, false, false, label_MuEnergy, label_matched_ratio, true});
    plotParamsMu.push_back({"Mu", "Reco", "Mass",   makePDCMuReco_ratio("M","ratio"),    "ratio",  nBins, minMassMu, maxMassMu, false, false, label_MuMass, label_matched_ratio, true});
    plotParamsMu.push_back({"Mu", "Reco", "Eta",    makePDCMuReco_ratio("eta","ratio"),  "ratio",  nBins, minEta, maxEta, false, false, label_MuEta, label_matched_ratio, true});
    plotParamsMu.push_back({"Mu", "Reco", "Phi",    makePDCMuReco_ratio("phi","ratio"),  "ratio",  nBins, minPhi, maxPhi, false, false, label_MuPhi, label_matched_ratio, true});
    // Iso
    plotParamsMu.push_back({"Mu", "Iso", "Pt",     makePDCMuIso_single("pt","single"),  "single", nBins, minPt, maxPt, true, false, label_MuPt, label_iso_single, true});
    plotParamsMu.push_back({"Mu", "Iso", "Energy", makePDCMuIso_single("E","single"),   "single", nBins, minEnergy, maxEnergy, true, false, label_MuEnergy, label_iso_single, true});
    plotParamsMu.push_back({"Mu", "Iso", "Mass",   makePDCMuIso_single("M","single"),   "single", nBins, minMassMu, maxMassMu, false, false, label_MuMass, label_iso_single, true});
    plotParamsMu.push_back({"Mu", "Iso", "Eta",    makePDCMuIso_single("eta","single"), "single", nBins, minEta, maxEta, false, false, label_MuEta, label_iso_single, true});
    plotParamsMu.push_back({"Mu", "Iso", "Phi",    makePDCMuIso_single("phi","single"), "single", nBins, minPhi, maxPhi, false, false, label_MuPhi, label_iso_single, true});
    plotParamsMu.push_back({"Mu", "Iso", "Pt",     makePDCMuIso_ratio("pt","ratio"),   "ratio",  nBins, minPt, maxPt, false, false, label_MuPt, label_iso_ratio, true});
    plotParamsMu.push_back({"Mu", "Iso", "Energy", makePDCMuIso_ratio("E","ratio"),    "ratio",  nBins, minEnergy, maxEnergy, false, false, label_MuEnergy, label_iso_ratio, true});
    plotParamsMu.push_back({"Mu", "Iso", "Mass",   makePDCMuIso_ratio("M","ratio"),    "ratio",  nBins, minMassMu, maxMassMu, false, false, label_MuMass, label_iso_ratio, true});
    plotParamsMu.push_back({"Mu", "Iso", "Eta",    makePDCMuIso_ratio("eta","ratio"),  "ratio",  nBins, minEta, maxEta, false, false, label_MuEta, label_iso_ratio, true});
    plotParamsMu.push_back({"Mu", "Iso", "Phi",    makePDCMuIso_ratio("phi","ratio"),  "ratio",  nBins, minPhi, maxPhi, false, false, label_MuPhi, label_iso_ratio, true});

    // photons 
    // measurements: Acc, Reco, Iso
    // variables: Pt, Energy, Mass Eta, Phi
    
    std::vector<plotStruct> plotParamsPhoton;
    
    // Tag must be Gen or Reco
    std::vector<std::string> tags = {"Gen", "Reco"};
    for (std::string tag : tags)
    {
      // Acc
      plotParamsPhoton.push_back({"Photon", tag+"Acc", "Pt",     makePDCPhotonAcc_single(tag, "pt","single"),  "single", nBins, minPt, maxPt,                 true,  false, label_PhotonPt,     label_map[tag+"Acc_single"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Acc", "Energy", makePDCPhotonAcc_single(tag, "E","single"),   "single", nBins, minEnergy, maxEnergy,         true,  false, label_PhotonEnergy, label_map[tag+"Acc_single"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Acc", "Mass",   makePDCPhotonAcc_single(tag, "M","single"),   "single", nBins, minMassPhoton, maxMassPhoton, false, false, label_PhotonMass,   label_map[tag+"Acc_single"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Acc", "Eta",    makePDCPhotonAcc_single(tag, "eta","single"), "single", nBins, minEta, maxEta,               false, false, label_PhotonEta,    label_map[tag+"Acc_single"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Acc", "Phi",    makePDCPhotonAcc_single(tag, "phi","single"), "single", nBins, minPhi, maxPhi,               false, false, label_PhotonPhi,    label_map[tag+"Acc_single"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Acc", "Pt",     makePDCPhotonAcc_ratio(tag, "pt","ratio"),    "ratio",  nBins, minPt, maxPt,                 false, false, label_PhotonPt,     label_map[tag+"Acc_ratio"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Acc", "Energy", makePDCPhotonAcc_ratio(tag, "E","ratio"),     "ratio",  nBins, minEnergy, maxEnergy,         false, false, label_PhotonEnergy, label_map[tag+"Acc_ratio"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Acc", "Mass",   makePDCPhotonAcc_ratio(tag, "M","ratio"),     "ratio",  nBins, minMassPhoton, maxMassPhoton, false, false, label_PhotonMass,   label_map[tag+"Acc_ratio"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Acc", "Eta",    makePDCPhotonAcc_ratio(tag, "eta","ratio"),   "ratio",  nBins, minEta, maxEta,               false, false, label_PhotonEta,    label_map[tag+"Acc_ratio"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Acc", "Phi",    makePDCPhotonAcc_ratio(tag, "phi","ratio"),   "ratio",  nBins, minPhi, maxPhi,               false, false, label_PhotonPhi,    label_map[tag+"Acc_ratio"], true});
      // Iso
      // only iso for reco
      if (tag.compare("Reco") == 0)
      {
        plotParamsPhoton.push_back({"Photon", tag+"Iso", "Pt",     makePDCPhotonIso_single(tag, "pt","single"),  "single", nBins, minPt, maxPt,                     true, false,  label_PhotonPt,     label_map[tag+"Iso_single"], true});
        plotParamsPhoton.push_back({"Photon", tag+"Iso", "Energy", makePDCPhotonIso_single(tag, "E","single"),   "single", nBins, minEnergy, maxEnergy,             true, false,  label_PhotonEnergy, label_map[tag+"Iso_single"], true});
        plotParamsPhoton.push_back({"Photon", tag+"Iso", "Mass",   makePDCPhotonIso_single(tag, "M","single"),   "single", nBins, minMassPhoton, maxMassPhoton,     false, false, label_PhotonMass,   label_map[tag+"Iso_single"], true});
        plotParamsPhoton.push_back({"Photon", tag+"Iso", "Eta",    makePDCPhotonIso_single(tag, "eta","single"), "single", nBins, minEta, maxEta,                   false, false, label_PhotonEta,    label_map[tag+"Iso_single"], true});
        plotParamsPhoton.push_back({"Photon", tag+"Iso", "Phi",    makePDCPhotonIso_single(tag, "phi","single"), "single", nBins, minPhi, maxPhi,                   false, false, label_PhotonPhi,    label_map[tag+"Iso_single"], true});
        plotParamsPhoton.push_back({"Photon", tag+"Iso", "Pt",     makePDCPhotonIso_ratio(tag, "pt","ratio"),    "ratio",  nBins, minPt, maxPt,                     false, false, label_PhotonPt,     label_map[tag+"Iso_ratio"], true});
        plotParamsPhoton.push_back({"Photon", tag+"Iso", "Energy", makePDCPhotonIso_ratio(tag, "E","ratio"),     "ratio",  nBins, minEnergy, maxEnergy,             false, false, label_PhotonEnergy, label_map[tag+"Iso_ratio"], true});
        plotParamsPhoton.push_back({"Photon", tag+"Iso", "Mass",   makePDCPhotonIso_ratio(tag, "M","ratio"),     "ratio",  nBins, minMassPhoton, maxMassPhoton,     false, false, label_PhotonMass,   label_map[tag+"Iso_ratio"], true});
        plotParamsPhoton.push_back({"Photon", tag+"Iso", "Eta",    makePDCPhotonIso_ratio(tag, "eta","ratio"),   "ratio",  nBins, minEta, maxEta,                   false, false, label_PhotonEta,    label_map[tag+"Iso_ratio"], true});
        plotParamsPhoton.push_back({"Photon", tag+"Iso", "Phi",    makePDCPhotonIso_ratio(tag, "phi","ratio"),   "ratio",  nBins, minPhi, maxPhi,                   false, false, label_PhotonPhi,    label_map[tag+"Iso_ratio"], true});
      }
      // Match
      plotParamsPhoton.push_back({"Photon", tag+"Match", "Pt",     makePDCPhotonMatch_single(tag, "pt","single"),  "single", nBins, minPt, maxPt,                 true, false,  label_PhotonPt,     label_map[tag+"Match_single"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Match", "Energy", makePDCPhotonMatch_single(tag, "E","single"),   "single", nBins, minEnergy, maxEnergy,         true, false,  label_PhotonEnergy, label_map[tag+"Match_single"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Match", "Mass",   makePDCPhotonMatch_single(tag, "M","single"),   "single", nBins, minMassPhoton, maxMassPhoton, false, false, label_PhotonMass,   label_map[tag+"Match_single"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Match", "Eta",    makePDCPhotonMatch_single(tag, "eta","single"), "single", nBins, minEta, maxEta,               false, false, label_PhotonEta,    label_map[tag+"Match_single"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Match", "Phi",    makePDCPhotonMatch_single(tag, "phi","single"), "single", nBins, minPhi, maxPhi,               false, false, label_PhotonPhi,    label_map[tag+"Match_single"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Match", "Pt",     makePDCPhotonMatch_ratio(tag, "pt","ratio"),    "ratio",  nBins, minPt, maxPt,                 false, false, label_PhotonPt,     label_map[tag+"Match_ratio"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Match", "Energy", makePDCPhotonMatch_ratio(tag, "E","ratio"),     "ratio",  nBins, minEnergy, maxEnergy,         false, false, label_PhotonEnergy, label_map[tag+"Match_ratio"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Match", "Mass",   makePDCPhotonMatch_ratio(tag, "M","ratio"),     "ratio",  nBins, minMassPhoton, maxMassPhoton, false, false, label_PhotonMass,   label_map[tag+"Match_ratio"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Match", "Eta",    makePDCPhotonMatch_ratio(tag, "eta","ratio"),   "ratio",  nBins, minEta, maxEta,               false, false, label_PhotonEta,    label_map[tag+"Match_ratio"], true});
      plotParamsPhoton.push_back({"Photon", tag+"Match", "Phi",    makePDCPhotonMatch_ratio(tag, "phi","ratio"),   "ratio",  nBins, minPhi, maxPhi,               false, false, label_PhotonPhi,    label_map[tag+"Match_ratio"], true});
    }
    
    if (doLeptons)
    {
        // loop over electron cut levels
        for(std::pair<std::string,std::string>& cut : cutlevels_electrons)
        {
            //ratios 
            for (plotStruct& p : plotParamsElec)
            {
                if (p.style.compare("single") == 0)
                    vh.push_back(PHS("MC_"+p.particle+p.measurement+p.variable+"_"+p.style+"_"+cut.first, {p.dataCollection}, {1, 2}, cut.second, p.nbins, p.xMin, p.xMax, p.logBool, p.normBool, p.xLabel, p.yLabel, p.ratioBool));
                else if (p.style.compare("ratio") == 0)
                    vh.push_back(PHS("MC_"+p.particle+p.measurement+p.variable+"_"+p.style+"_"+cut.first, {p.dataCollection}, {1, 1}, cut.second, p.nbins, p.xMin, p.xMax, p.logBool, p.normBool, p.xLabel, p.yLabel, p.ratioBool));
            }
            
        }

        // loop over muon cut levels
        for(std::pair<std::string,std::string>& cut : cutlevels_muon)
        {
            
            //ratios 
            for (plotStruct& p : plotParamsMu)
            {
                if (p.style.compare("single") == 0)
                    vh.push_back(PHS("MC_"+p.particle+p.measurement+p.variable+"_"+p.style+"_"+cut.first, {p.dataCollection}, {1, 2}, cut.second, p.nbins, p.xMin, p.xMax, p.logBool, p.normBool, p.xLabel, p.yLabel, p.ratioBool));
                else if (p.style.compare("ratio") == 0)
                    vh.push_back(PHS("MC_"+p.particle+p.measurement+p.variable+"_"+p.style+"_"+cut.first, {p.dataCollection}, {1, 1}, cut.second, p.nbins, p.xMin, p.xMax, p.logBool, p.normBool, p.xLabel, p.yLabel, p.ratioBool));
            }
            
            //Z things (DY)
            // vh.push_back(PHS("DataMC_DY_gen_pt_"                 +cut.first,  {dcData_DY_gen_pt, dcMC_DY_gen_pt},              {1, 2}, cut.second, 60, minPt, maxPt, true, false,   "gen Z Pt",   "Events"));
            // vh.push_back(PHS("DataMC_DY_reco_pt_"                +cut.first,  {dcData_DY_reco_pt, dcMC_DY_reco_pt},            {1, 2}, cut.second, 60, minPt, maxPt, true, false,   "reco Z Pt",  "Events"));
            // vh.push_back(PHS("DataMC_DY_eta_"                    +cut.first,  {dcData_DY_eta, dcMC_DY_eta},                    {1, 2}, cut.second, 60, -10, 10, true, false,   "gen Z Eta",  "Events"));
            // vh.push_back(PHS("DataMC_DY_phi_"                    +cut.first,  {dcData_DY_phi, dcMC_DY_phi},                    {1, 2}, cut.second, 60, -10, 10, true, false,   "gen Z Phi",  "Events"));
            // vh.push_back(PHS("DataMC_DY_mass_"                   +cut.first,  {dcData_DY_mass, dcMC_DY_mass},                  {1, 2}, cut.second, 60, 0, 200, true, false,    "gen Z Mass", "Events"));
            
            if (doWeights)
            {
                //no weights
                vh.push_back(PHS("DataMC_SingleMuon_met_"            +cut.first,  {dcData_SingleMuon_met,   dcMC_met},             {1, 2}, cut.second, 60, 0, 1500, true, false,  label_met,      "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_ht_"             +cut.first,  {dcData_SingleMuon_ht,    dcMC_ht},              {1, 2}, cut.second, 60, 0, 1500, true, false,  label_ht,       "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_nt_"             +cut.first,  {dcData_SingleMuon_nt,    dcMC_nt},              {1, 2}, cut.second, 5,  0, 5,    false, false,  label_nt,      "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_mt2_"            +cut.first,  {dcData_SingleMuon_mt2,   dcMC_mt2},             {1, 2}, cut.second, 60, 0, 1500, true, false,  label_mt2,      "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_nb_"             +cut.first,  {dcData_SingleMuon_nb,    dcMC_nb},              {1, 2}, cut.second, 10, 0, 10,   true, false,  label_nb,       "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_nj_"             +cut.first,  {dcData_SingleMuon_nj,    dcMC_nj},              {1, 2}, cut.second, 30, 0, 30,   true, false,  label_nj,       "Events"));
                //vh.push_back(PHS("DataMC_SingleMuon_genMuPt_"        +cut.first,  {dcData_SingleMuon_genMuPt, dcMC_genMuPt},       {1, 2}, cut.second, 50, 0, 1000, true, false,  label_genmupt,  "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_mu1pt_"          +cut.first,  {dcData_SingleMuon_mu1pt, dcMC_mu1pt},           {1, 2}, cut.second, 50, 0, 1000, true, false,  label_Mu1pt,    "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_mu2pt_"          +cut.first,  {dcData_SingleMuon_mu2pt, dcMC_mu2pt},           {1, 2}, cut.second, 50, 0, 1000, true, false,  label_Mu2pt,    "Events"));
                ///vh.push_back(PHS("DataMC_SingleMuon_genMuEta_"       +cut.first,  {dcData_SingleMuon_genMuEta, dcMC_genMuEta},     {1, 2}, cut.second, 40, -10, 10, true, false,  label_genmueta, "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_mu1eta_"         +cut.first,  {dcData_SingleMuon_mu1eta, dcMC_mu1eta},         {1, 2}, cut.second, 40, -10, 10, true, false,  label_Mu1eta,   "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_mu2eta_"         +cut.first,  {dcData_SingleMuon_mu2eta, dcMC_mu2eta},         {1, 2}, cut.second, 40, -10, 10, true, false,  label_Mu2eta,   "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_mll_"            +cut.first,  {dcData_SingleMuon_mll,   dcMC_mll},             {1, 2}, cut.second, 40, 0, 200,  true, false,  label_mll,      "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_nSearchBin_"     +cut.first,  {dcData_SingleMuon_nSearchBin, dcMC_nSearchBin}, {1, 2}, cut.second, NSB, 0, NSB, true, false,  "Search Bin",   "Events"));
                //Shape correction weights
                vh.push_back(PHS("DataMC_SingleMuon_met_Wgt_"        +cut.first,  {dcData_SingleMuon_met,   dcwMC_met},             {1, 2}, cut.second, 60, 0, 1500, true, false,  label_met,  "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_ht_Wgt_"         +cut.first,  {dcData_SingleMuon_ht,    dcwMC_ht},              {1, 2}, cut.second, 60, 0, 1500, true, false,  label_ht,   "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_nt_Wgt_"         +cut.first,  {dcData_SingleMuon_nt,    dcwMC_nt},              {1, 2}, cut.second, 5,  0, 5,    false, false,  label_nt,  "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_mt2_Wgt_"        +cut.first,  {dcData_SingleMuon_mt2,   dcwMC_mt2},             {1, 2}, cut.second, 60, 0, 1500, true, false,  label_mt2,  "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_nb_Wgt_"         +cut.first,  {dcData_SingleMuon_nb,    dcwMC_nb},              {1, 2}, cut.second, 10, 0, 10,   true, false,  label_nb,   "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_nj_Wgt_"         +cut.first,  {dcData_SingleMuon_nj,    dcwMC_nj},              {1, 2}, cut.second, 30, 0, 30,   true, false,  label_nj,   "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_mu1pt_Wgt_"      +cut.first,  {dcData_SingleMuon_mu1pt, dcwMC_mu1pt},           {1, 2}, cut.second, 50, 0, 1000, true, false,  label_Mu1pt, "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_mu2pt_Wgt_"      +cut.first,  {dcData_SingleMuon_mu2pt, dcwMC_mu2pt},           {1, 2}, cut.second, 50, 0, 1000, true, false,  label_Mu2pt, "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_mu1eta_Wgt_"     +cut.first,  {dcData_SingleMuon_mu1eta, dcwMC_mu1eta},         {1, 2}, cut.second, 40, -10, 10, true, false,  label_Mu1eta, "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_mu2eta_Wgt_"     +cut.first,  {dcData_SingleMuon_mu2eta, dcwMC_mu2eta},         {1, 2}, cut.second, 40, -10, 10, true, false,  label_Mu2eta, "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_mll_Wgt_"        +cut.first,  {dcData_SingleMuon_mll,   dcwMC_mll},             {1, 2}, cut.second, 40, 0, 200,  true, false,  label_mll,  "Events"));
                vh.push_back(PHS("DataMC_SingleMuon_nSearchBin_Wgt_" +cut.first,  {dcData_SingleMuon_nSearchBin, dcwMC_nSearchBin}, {1, 2}, cut.second, NSB, 0, NSB, true, false,  "Search Bin", "Events"));
            }
        }
    }

    if (doWeights)
    {
        for(std::pair<std::string,std::string>& cut : cutlevels_muon)
        {
            vector<double> metBins = {0, 50, 100, 150, 200, 275, 300, 350, 400, 450, 2000};
            vector<double> mt2Bins = {0, 50, 100, 150, 200, 250, 300, 350, 400, 2000};
            vh.push_back(PHS("DataMCw_SingleMuon_met_"        +cut.first,  {dcData_SingleMuon_met,        dcwMC_met},        {1, 2}, cut.second, 25, 0, 1500, true, false,  label_met,     "Events / 60 GeV"));
            vh.push_back(PHS("DataMCw_SingleMuon_rebin_met_"  +cut.first,  {dcData_SingleMuon_met,        dcwMC_met},        {1, 2}, cut.second, metBins,     true, false,  label_met,     "Events"));
            vh.push_back(PHS("DataMCw_SingleMuon_ht_"         +cut.first,  {dcData_SingleMuon_ht,         dcwMC_ht},         {1, 2}, cut.second, 60, 0, 1500, true, false,  label_ht,      "Events"));
            vh.push_back(PHS("DataMCw_SingleMuon_mht_"        +cut.first,  {dcData_SingleMuon_mht,        dcwMC_mht},        {1, 2}, cut.second, 60, 0, 1500, true, false,  label_mht,     "Events"));
            vh.push_back(PHS("DataMCw_SingleMuon_nt_"         +cut.first,  {dcData_SingleMuon_nt,         dcwMC_nt},         {1, 2}, cut.second, 5,  0, 5,    true, false,  label_nt,      "Events"));
            vh.push_back(PHS("DataMCw_SingleMuon_mt2_"        +cut.first,  {dcData_SingleMuon_mt2,        dcwMC_mt2},        {1, 2}, cut.second, 60, 0, 1500, true, false,  label_mt2,     "Events"));
            vh.push_back(PHS("DataMCw_SingleMuon_rebin_mt2_"  +cut.first,  {dcData_SingleMuon_mt2,        dcwMC_mt2},        {1, 2}, cut.second, mt2Bins,     true, false,  label_mt2,     "Events"));
            vh.push_back(PHS("DataMCw_SingleMuon_nb_"         +cut.first,  {dcData_SingleMuon_nb,         dcwMC_nb},         {1, 2}, cut.second, 7, 0, 7,   true, false,  label_nb,        "Events / bin"));
            vh.push_back(PHS("DataMCw_SingleMuon_nj_"         +cut.first,  {dcData_SingleMuon_nj,         dcwMC_nj},         {1, 2}, cut.second, 30, 0, 30,   true, false,  label_nj,      "Events"));
            vh.push_back(PHS("DataMCw_SingleMuon_nSearchBin_" +cut.first,  {dcData_SingleMuon_nSearchBin, dcwMC_nSearchBin}, {1, 2}, cut.second, NSB, 0, NSB, true, false,  "Search Bin",  "Events"));

            //// Normalization weight applied, only for blnotag selections
            if(cut.first.rfind("blnotag") == (cut.first.size()-7))
            {
                // DataMC weights applied
                vh.push_back(PHS("DataMCww_SingleMuon_met_"   +cut.first,  {dcData_SingleMuon_met,   dcwwMC_met},   {1, 2}, cut.second, 50, 0, 1500, true, false,  label_met,                    "Events"));
                vh.push_back(PHS("DataMCww_SingleMuon_ht_"    +cut.first,  {dcData_SingleMuon_ht,    dcwwMC_ht},    {1, 2}, cut.second, 50, 0, 1500, true, false,  label_ht,                     "Events"));
                vh.push_back(PHS("DataMCww_SingleMuon_mht_"   +cut.first,  {dcData_SingleMuon_mht,   dcwwMC_mht},   {1, 2}, cut.second, 50, 0, 1500, true, false,  label_mht,                    "Events"));
                vh.push_back(PHS("DataMCww_SingleMuon_nt_"    +cut.first,  {dcData_SingleMuon_nt,    dcwwMC_nt},    {1, 2}, cut.second, 5,  0, 5,    true, false,  label_nt,                     "Events"));
                vh.push_back(PHS("DataMCww_SingleMuon_mt2_"   +cut.first,  {dcData_SingleMuon_mt2,   dcwwMC_mt2},   {1, 2}, cut.second, 50, 0, 1500, true, false,  label_mt2,                    "Events"));
                vh.push_back(PHS("DataMCww_SingleMuon_nb_"    +cut.first,  {dcData_SingleMuon_nb,    dcwwMC_nb},    {1, 2}, cut.second, 10, 0, 10,   true, false,  label_nb,                     "Events"));
                vh.push_back(PHS("DataMCww_SingleMuon_nj_"    +cut.first,  {dcData_SingleMuon_nj,    dcwwMC_nj},    {1, 2}, cut.second, 30, 0, 30,   true, false,  label_nj,                     "Events"));
                vh.push_back(PHS("DataMCww_SingleMuon_nSearchBin_" +cut.first,  {dcData_SingleMuon_nSearchBin, dcwwMC_nSearchBin}, {1, 2}, cut.second, NSB, 0, NSB,  true, false,  "Search Bin", "Events"));
            }
        }
    }

    // photons
    if (doPhotons)
    {
        for (plotStruct& p : plotParamsPhoton)
        {
            if (p.style.compare("single") == 0)
                vh.push_back(PHS("MC_"+p.particle+p.measurement+p.variable+"_"+p.style, {p.dataCollection}, {1, 2}, "", p.nbins, p.xMin, p.xMax, p.logBool, p.normBool, p.xLabel, p.yLabel, p.ratioBool));
            else if (p.style.compare("ratio") == 0)
                vh.push_back(PHS("MC_"+p.particle+p.measurement+p.variable+"_"+p.style, {p.dataCollection}, {1, 1}, "", p.nbins, p.xMin, p.xMax, p.logBool, p.normBool, p.xLabel, p.yLabel, p.ratioBool));
        }
        
            
    }

    //tops
    //vh.push_back(PHS("DataMC_T1tttt", {dcMC_T1tttt}, {1, 1}, "passNoiseEventFilterZinv", 60, minPt, maxPt, true, false, label_genTopPt, "Events"));

    // Znunu
    Plotter::DatasetSummary dsDY_nunu("Z#rightarrow#nu#nu", fileMap["ZJetsToNuNu"], "passLeptVeto", "");
    // nothing
    Plotter::DatasetSummary dsDY_nunu_njetnorm_TriggerCentral(          "Z#rightarrow#nu#nu Trigger Central ",        fileMap["ZJetsToNuNu"], "passLeptVeto",    "");
    Plotter::DatasetSummary dsDY_nunu_njetnorm_TriggerUp(               "Z#rightarrow#nu#nu Trigger Up ",             fileMap["ZJetsToNuNu"], "passLeptVeto",    "");
    Plotter::DatasetSummary dsDY_nunu_njetnorm_TriggerDown(             "Z#rightarrow#nu#nu Trigger Down ",           fileMap["ZJetsToNuNu"], "passLeptVeto",    "");
    Plotter::DatasetSummary dsDY_nunu_njetnorm(                         "Z#rightarrow#nu#nu Njet+norm ",              fileMap["ZJetsToNuNu"], "passLeptVeto",    "");
    // only b jet scale factor
    Plotter::DatasetSummary dsDY_nunu_njetnorm_TriggerCentral_scaled(   "Z#rightarrow#nu#nu Trigger scale Central ", fileMap["ZJetsToNuNu"], "passLeptVeto",    "bTagSF_EventWeightSimple_Central");
    Plotter::DatasetSummary dsDY_nunu_njetnorm_TriggerUp_scaled(        "Z#rightarrow#nu#nu Trigger scale Up ",      fileMap["ZJetsToNuNu"], "passLeptVeto",    "bTagSF_EventWeightSimple_Central");
    Plotter::DatasetSummary dsDY_nunu_njetnorm_TriggerDown_scaled(      "Z#rightarrow#nu#nu Trigger scale Down ",    fileMap["ZJetsToNuNu"], "passLeptVeto",    "bTagSF_EventWeightSimple_Central");
    Plotter::DatasetSummary dsDY_nunu_njetnorm_scaled(                  "Z#rightarrow#nu#nu Njet+norm scale ",       fileMap["ZJetsToNuNu"], "passLeptVeto",    "bTagSF_EventWeightSimple_Central");
    // bjet scale factor, shape factor and normalization
    Plotter::DatasetSummary dsDY_nunu_njetnorm_TriggerCentral_weighted( "Z#rightarrow#nu#nu Trigger weight Central ", fileMap["ZJetsToNuNu"], "passLeptVeto",    "bTagSF_EventWeightSimple_Central;njWGJets;normWgt0b;TriggerEffMC");
    Plotter::DatasetSummary dsDY_nunu_njetnorm_TriggerUp_weighted(      "Z#rightarrow#nu#nu Trigger weight Up ",      fileMap["ZJetsToNuNu"], "passLeptVeto",    "bTagSF_EventWeightSimple_Central;njWGJets;normWgt0b;TriggerEffUpMC");
    Plotter::DatasetSummary dsDY_nunu_njetnorm_TriggerDown_weighted(    "Z#rightarrow#nu#nu Trigger weight Down ",    fileMap["ZJetsToNuNu"], "passLeptVeto",    "bTagSF_EventWeightSimple_Central;njWGJets;normWgt0b;TriggerEffDownMC");
    Plotter::DatasetSummary dsDY_nunu_njetnorm_weighted(                "Z#rightarrow#nu#nu Njet+norm weight ",       fileMap["ZJetsToNuNu"], "passLeptVeto",    "bTagSF_EventWeightSimple_Central;njWGJets;normWgt0b");
    Plotter::DataCollection trigger_nSearchBin( "single", {{"nSearchBin",    dsDY_nunu_njetnorm_TriggerCentral}, {"nSearchBin",    dsDY_nunu_njetnorm_TriggerUp}, {"nSearchBin",    dsDY_nunu_njetnorm_TriggerDown}, {"nSearchBin",    dsDY_nunu_njetnorm}  });
    Plotter::DataCollection trigger_nSearchBin_scaled( "single", {{"nSearchBin",    dsDY_nunu_njetnorm_TriggerCentral_scaled}, {"nSearchBin",    dsDY_nunu_njetnorm_TriggerUp_scaled}, {"nSearchBin",    dsDY_nunu_njetnorm_TriggerDown_scaled}, {"nSearchBin",    dsDY_nunu_njetnorm_scaled}  });
    Plotter::DataCollection trigger_nSearchBin_weighted( "single", {{"nSearchBin",    dsDY_nunu_njetnorm_TriggerCentral_weighted}, {"nSearchBin",    dsDY_nunu_njetnorm_TriggerUp_weighted}, {"nSearchBin",    dsDY_nunu_njetnorm_TriggerDown_weighted}, {"nSearchBin",    dsDY_nunu_njetnorm_weighted}  });
    
    // Znunu
    auto makePDSZnunu       = [&](const std::string& label, const std::string& cuts="HTZinv>200") {return Plotter::DatasetSummary("ZJetsToNuNu "+label, fileMap["ZJetsToNuNu"], cuts, ""); };
    auto makePDCGJetsZnunu  = [&](const std::string& var, const std::string& style, const std::string& label, const std::string& cuts) {return Plotter::DataCollection(style, {{var, makePDSPhoton(label, "GJets", "passPhotonSelection;" + cuts)}, {var, makePDSZnunu(label, cuts)}}); };
    
    // study jet collections and jet cleaning
    // photon jet cleaning
    std::string met_cut = "";
    //std::string met_cut = "met>250";
    std::string photon_cut = "passPhotonSelection"; 
    std::string lepton_cut = "passLeptVeto"; 
    // use met cut if it is not an empty string
    if (! met_cut.empty() )
    {
      photon_cut += ";" + met_cut;
      lepton_cut += ";" + met_cut;
    }
    auto makePDSGJets = [&](const std::string& label, const std::string& cuts) {return Plotter::DatasetSummary("GJets "+label, fileMap["GJets"], cuts, ""); };
    //Plotter::DataCollection dc_GJets_nj_all                ("single", "cntNJetsPt20Eta24NoVeto",   {makePDSGJets("all jets", met_cut)}                     );
    //Plotter::DataCollection dc_GJets_nj_all_onephoton      ("single", "cntNJetsPt20Eta24NoVeto",   {makePDSGJets("all jets one photon", photon_cut)}       );
    //Plotter::DataCollection dc_GJets_nj_phoclean           ("single", "cntNJetsPt20Eta24DRPhotonCleaned", {makePDSGJets("photon cleaned", met_cut)}               );
    //Plotter::DataCollection dc_GJets_nj_phoclean_onephoton ("single", "cntNJetsPt20Eta24DRPhotonCleaned", {makePDSGJets("photon cleaned one photon", photon_cut)} );
    
    //vh.push_back(PHS("MC_GJets_nj", {dc_GJets_nj_all, dc_GJets_nj_all_onephoton, dc_GJets_nj_phoclean, dc_GJets_nj_phoclean_onephoton}, {1, 1}, "", 10, 0, 10, true, false, label_nj, label_Events));
    
    // lepton jet cleaning
    // variables: HT, MET, METPHI, dPhi, n_j, n_t, n_b 
    auto makePDSDY = [&](const std::string& label, const std::string& cuts) {return Plotter::DatasetSummary("DYJetsToLL "+label, fileMap["DYJetsToLL"], cuts, ""); };
    //Plotter::DataCollection dc_DY_nj_all              ("single", "cntNJetsPt20Eta24NoVeto",   {makePDSDY("all jets", met_cut)}                      );
    //Plotter::DataCollection dc_DY_nj_all_lepveto      ("single", "cntNJetsPt20Eta24NoVeto",   {makePDSDY("all jets lepton veto", lepton_cut)}       );
    //Plotter::DataCollection dc_DY_nj_lepclean         ("single", "cntNJetsPt20Eta24NoLepton", {makePDSDY("lepton cleaned", met_cut)}                );
    //Plotter::DataCollection dc_DY_nj_lepclean_lepveto ("single", "cntNJetsPt20Eta24NoLepton", {makePDSDY("lepton cleaned lepton veto", lepton_cut)} );
    //Plotter::DataCollection dc_DY_nt_all              ("single", "nTopCandSortedCntNoVeto",   {makePDSDY("all jets", met_cut)}                      );
    //Plotter::DataCollection dc_DY_nt_lepclean         ("single", "nTopCandSortedCntNoLepton", {makePDSDY("lepton cleaned", met_cut)}                );
    
    //vh.push_back(PHS("MC_DY_nj", {dc_DY_nj_all, dc_DY_nj_all_lepveto, dc_DY_nj_lepclean, dc_DY_nj_lepclean_lepveto}, {1, 1}, "", 10, 0, 10, true, false, label_nj, label_Events));
    
    if (doDYAndZnunu)
    {
        struct simplePlotStruct
        {
            std::string variable;                                       // met, HT, etc.
            std::vector<Plotter::DataCollection> dataCollectionVector;  // vector of Data Collections
            int nBins;                                                  // number of bins
            double xMin;                                                // x min for histo
            double xMax;                                                // x max for histo
            bool logBool;                                               // bool for log scale (true for pt)
            bool normBool;                                              // bool for norm (false)
            std::string xLabel;                                         // x axis label
            std::string yLabel;                                         // y axis label
        };
        
        // map of variable names to vector of data collections
        std::map<std::string, std::vector<Plotter::DataCollection> > dataCollectionMap;
        // vetor of variable names
        std::vector<std::string> variables = {"cntNJetsPt20Eta24", "nTopCandSortedCnt", "cntCSVS", "HT"};
        // vector of pais with tags and labels
        std::vector< std::pair<std::string, std::string> > tagVector;
        //tagVector.emplace_back("NoVeto",          "all jets");
        tagVector.emplace_back("PFLeptonCleaned", "PF lepton cleaned jets");
        tagVector.emplace_back("DRLeptonCleaned", "DR lepton cleaned jets");
        // vector of DY selections
        std::vector<std::string> selectionVec = {"Elec", "Mu"};
        // vector of styles
        std::vector<string> styleVec = {"", "_ratio"};
        // map of jets 
        std::map<std::string, std::string> jetMap;
        
        // no pt or eta cuts
        //jetMap["NoVeto"]          = "jetsLVec";
        //jetMap["PFLeptonCleaned"] = "prodJetsNoLep_jetsLVec";
        //jetMap["DRLeptonCleaned"] = "jetsLVec_drLeptonCleaned";
        // with pt and eta cuts
        jetMap["NoVeto"]          = "jetsLVec_pt20eta24";
        jetMap["PFLeptonCleaned"] = "prodJetsNoLep_jetsLVec_pt20eta24";
        jetMap["DRLeptonCleaned"] = "jetsLVec_drLeptonCleaned_pt20eta24";

        // map of y axis limits
        std::map< std::string, std::vector<float> > YAxisLimits;
        YAxisLimits["jetphi"] = {    pow(10.0, 2),      pow(10.0, 5)};
        YAxisLimits["jeteta"] = {    pow(10.0, 2),      pow(10.0, 6)};
        YAxisLimits["metphi"] = {    pow(10.0, 2),      pow(10.0, 4)};
        YAxisLimits["dphi0"]  = {5 * pow(10.0, -1), 5 * pow(10.0, 5)};
        YAxisLimits["dphi1"]  = {5 * pow(10.0, 1),      pow(10.0, 5)};
        YAxisLimits["dphi2"]  = {5 * pow(10.0, 1),      pow(10.0, 5)};

        // plot parameters
        std::vector<simplePlotStruct> plotParamsDY;

        // try combining the following sections into one loop
        // selection, then variable, then tag (standard pattern)
        // selection, then tag, then one line per variable (unique pattern)
        
        std::string selectionLL = "";
        std::string selectionNuNu = "passBaselineNoVeto";
        // fill map
        for (const auto& s : selectionVec) 
        {
            for (const auto& variable : variables)
            {
                // z nunu
                dataCollectionMap[variable + "_" + s].emplace_back( Plotter::DataCollection("single", variable + "NoVeto", {makePDSZnunu("all jets", selectionNuNu)} ) );
                for (const auto& tag : tagVector)
                {
                    // note that DY to LL and Z to NuNu have different selections
                    // DY to LL has dilepton selection, while Z to NuNu does not
                    selectionLL = "passBaseline" + tag.first + ";pass" + s + "ZinvSel_lowpt";
                    // DY
                    dataCollectionMap[variable + "_" + s].emplace_back(           Plotter::DataCollection("single", variable + tag.first, {makePDSDY(tag.second, selectionLL)} ) );
                    dataCollectionMap[variable + "_" + s+ "_ratio"].emplace_back( Plotter::DataCollection("ratio",  variable + tag.first, {makePDSDY(tag.second, selectionLL), makePDSZnunu(tag.second, selectionNuNu)} ) );
                }
            }
            
            // z nunu
            dataCollectionMap["jetpt_" + s].emplace_back(  Plotter::DataCollection("single", jetMap["NoVeto"] + "(pt)",    {makePDSZnunu("all jets", selectionNuNu)} ) );
            dataCollectionMap["jeteta_" + s].emplace_back( Plotter::DataCollection("single", jetMap["NoVeto"] + "(eta)",   {makePDSZnunu("all jets", selectionNuNu)} ) );
            dataCollectionMap["jetphi_" + s].emplace_back( Plotter::DataCollection("single", jetMap["NoVeto"] + "(phi)",   {makePDSZnunu("all jets", selectionNuNu)} ) );
            dataCollectionMap["jetE_" + s].emplace_back(   Plotter::DataCollection("single", jetMap["NoVeto"] + "(E)",     {makePDSZnunu("all jets", selectionNuNu)} ) );
            dataCollectionMap["met_" + s].emplace_back(    Plotter::DataCollection("single", "cleanMetPt",                 {makePDSZnunu("all jets", selectionNuNu)} ) );
            dataCollectionMap["metphi_" + s].emplace_back( Plotter::DataCollection("single", "cleanMetPhi",                {makePDSZnunu("all jets", selectionNuNu)} ) );
            dataCollectionMap["dphi0_" + s].emplace_back(  Plotter::DataCollection("single", "dPhiVecNoVeto[0]",           {makePDSZnunu("all jets", selectionNuNu)} ) );
            dataCollectionMap["dphi1_" + s].emplace_back(  Plotter::DataCollection("single", "dPhiVecNoVeto[1]",           {makePDSZnunu("all jets", selectionNuNu)} ) );
            dataCollectionMap["dphi2_" + s].emplace_back(  Plotter::DataCollection("single", "dPhiVecNoVeto[2]",           {makePDSZnunu("all jets", selectionNuNu)} ) );
            for (const auto& tag : tagVector)
            {
                selectionLL = "passBaseline" + tag.first + ";pass" + s + "ZinvSel_lowpt";
                // DY
                dataCollectionMap["jetpt_" + s].emplace_back(  Plotter::DataCollection("single", jetMap[tag.first] + "(pt)",    {makePDSDY(tag.second, selectionLL)} ) );
                dataCollectionMap["jeteta_" + s].emplace_back( Plotter::DataCollection("single", jetMap[tag.first] + "(eta)",   {makePDSDY(tag.second, selectionLL)} ) );
                dataCollectionMap["jetphi_" + s].emplace_back( Plotter::DataCollection("single", jetMap[tag.first] + "(phi)",   {makePDSDY(tag.second, selectionLL)} ) );
                dataCollectionMap["jetE_" + s].emplace_back(   Plotter::DataCollection("single", jetMap[tag.first] + "(E)",     {makePDSDY(tag.second, selectionLL)} ) );
                dataCollectionMap["met_" + s].emplace_back(    Plotter::DataCollection("single", "cleanMetPt",                  {makePDSDY(tag.second, selectionLL)} ) );
                dataCollectionMap["metphi_" + s].emplace_back( Plotter::DataCollection("single", "cleanMetPhi",                 {makePDSDY(tag.second, selectionLL)} ) );
                dataCollectionMap["dphi0_" + s].emplace_back(  Plotter::DataCollection("single", "dPhiVec" + tag.first + "[0]", {makePDSDY(tag.second, selectionLL)} ) );
                dataCollectionMap["dphi1_" + s].emplace_back(  Plotter::DataCollection("single", "dPhiVec" + tag.first + "[1]", {makePDSDY(tag.second, selectionLL)} ) );
                dataCollectionMap["dphi2_" + s].emplace_back(  Plotter::DataCollection("single", "dPhiVec" + tag.first + "[2]", {makePDSDY(tag.second, selectionLL)} ) );
                dataCollectionMap["jetpt_" + s + "_ratio"].emplace_back(  Plotter::DataCollection("ratio", jetMap[tag.first] + "(pt)",    {makePDSDY(tag.second, selectionLL), makePDSZnunu(tag.second, selectionNuNu)} ) );
                dataCollectionMap["jeteta_" + s + "_ratio"].emplace_back( Plotter::DataCollection("ratio", jetMap[tag.first] + "(eta)",   {makePDSDY(tag.second, selectionLL), makePDSZnunu(tag.second, selectionNuNu)} ) );
                dataCollectionMap["jetphi_" + s + "_ratio"].emplace_back( Plotter::DataCollection("ratio", jetMap[tag.first] + "(phi)",   {makePDSDY(tag.second, selectionLL), makePDSZnunu(tag.second, selectionNuNu)} ) );
                dataCollectionMap["jetE_" + s + "_ratio"].emplace_back(   Plotter::DataCollection("ratio", jetMap[tag.first] + "(E)",     {makePDSDY(tag.second, selectionLL), makePDSZnunu(tag.second, selectionNuNu)} ) );
                dataCollectionMap["met_" + s + "_ratio"].emplace_back(    Plotter::DataCollection("ratio", "cleanMetPt",                  {makePDSDY(tag.second, selectionLL), makePDSZnunu(tag.second, selectionNuNu)} ) );
                dataCollectionMap["metphi_" + s + "_ratio"].emplace_back( Plotter::DataCollection("ratio", "cleanMetPhi",                 {makePDSDY(tag.second, selectionLL), makePDSZnunu(tag.second, selectionNuNu)} ) );
                dataCollectionMap["dphi0_" + s + "_ratio"].emplace_back(  Plotter::DataCollection("ratio", "dPhiVec" + tag.first + "[0]", {makePDSDY(tag.second, selectionLL), makePDSZnunu(tag.second, selectionNuNu)} ) );
                dataCollectionMap["dphi1_" + s + "_ratio"].emplace_back(  Plotter::DataCollection("ratio", "dPhiVec" + tag.first + "[1]", {makePDSDY(tag.second, selectionLL), makePDSZnunu(tag.second, selectionNuNu)} ) );
                dataCollectionMap["dphi2_" + s + "_ratio"].emplace_back(  Plotter::DataCollection("ratio", "dPhiVec" + tag.first + "[2]", {makePDSDY(tag.second, selectionLL), makePDSZnunu(tag.second, selectionNuNu)} ) );
            }

            selectionLL = "passBaselineDRLeptonCleaned;pass" + s + "ZinvSel_lowpt";
            dataCollectionMap["dr_" + s].emplace_back( Plotter::DataCollection("single", "dR_jetsLVec_drLeptonCleaned", {makePDSDY("all jets", selectionLL)} ) );
           
            
            // fill plot parameters
            for (const auto& style : styleVec)
            {
                std::string y_axis_label = label_Events;
                if (style.compare("_ratio") == 0)
                {
                    y_axis_label = "DYJetsToLL / ZJetsToNuNu";
                }
             
                plotParamsDY.push_back({"jetpt_" + s + style,   dataCollectionMap["jetpt_" + s + style],              nBins, 0.0, 200.0, true, false, label_jetpt, y_axis_label});
                plotParamsDY.push_back({"jeteta_" + s + style,  dataCollectionMap["jeteta_" + s + style],             nBins, minEta, maxEta, true, false, label_jeteta, y_axis_label});
                plotParamsDY.push_back({"jetphi_" + s + style,  dataCollectionMap["jetphi_" + s + style],             nBins, minPhi, maxPhi, true, false, label_jetphi, y_axis_label});
                plotParamsDY.push_back({"jetE_" + s + style,    dataCollectionMap["jetE_" + s + style],               nBins, 0.0, 2000.0, true, false, label_jetE, y_axis_label});
                plotParamsDY.push_back({"nj_" + s + style,      dataCollectionMap["cntNJetsPt20Eta24_" + s + style],  20, 0, 20, true, false, label_nj, y_axis_label});
                plotParamsDY.push_back({"nt_" + s + style,      dataCollectionMap["nTopCandSortedCnt_" + s + style],  20, 0, 20, true, false, label_nt, y_axis_label});
                plotParamsDY.push_back({"nb_" + s + style,      dataCollectionMap["cntCSVS_" + s + style],            20, 0, 20, true, false, label_nb, y_axis_label});
                plotParamsDY.push_back({"ht_" + s + style,      dataCollectionMap["HT_" + s + style],                 nBins, 0.0, 2000.0, true, false, label_ht, y_axis_label});
                plotParamsDY.push_back({"met_" + s + style,     dataCollectionMap["met_" + s + style],                nBins, 0.0, 2000.0, true, false, label_met, y_axis_label});
                plotParamsDY.push_back({"metphi_" + s + style,  dataCollectionMap["metphi_" + s + style],             nBins, minPhi, maxPhi, true, false, label_metphi, y_axis_label});
                plotParamsDY.push_back({"dphi0_" + s + style,   dataCollectionMap["dphi0_" + s + style],              nBins, 0.0, maxPhi, true, false, label_dphi0, y_axis_label});
                plotParamsDY.push_back({"dphi1_" + s + style,   dataCollectionMap["dphi1_" + s + style],              nBins, 0.0, maxPhi, true, false, label_dphi1, y_axis_label});
                plotParamsDY.push_back({"dphi2_" + s + style,   dataCollectionMap["dphi2_" + s + style],              nBins, 0.0, maxPhi, true, false, label_dphi2, y_axis_label});
            }
                plotParamsDY.push_back({"dr_" + s,      dataCollectionMap["dr_" + s],                 nBins, 0.0, 1.0, true, false, label_dr, label_Events});
        }
        
        for (const auto& p : plotParamsDY)
        {
            // check for ratio before checking for y-axis limits
            // dr and ratio options
            std::vector<std::string> splitVar = SusyUtility::getVecFromString(p.variable, '_');
            std::string var = splitVar[0];
            if (p.variable.find("dr") != std::string::npos || p.variable.find("ratio") != std::string::npos)
            {
                vh.push_back(PHS("MC_DY_" + p.variable, {p.dataCollectionVector}, {1, 1}, "", p.nBins, p.xMin, p.xMax, p.logBool, p.normBool, p.xLabel, p.yLabel));
            }
            // user defined y-axis limits
            else if (YAxisLimits.find(var) != YAxisLimits.end())
            {
                vh.push_back(PHS("MC_DY_" + p.variable, {p.dataCollectionVector}, {2, 1}, "", p.nBins, p.xMin, p.xMax, YAxisLimits[var][0], YAxisLimits[var][1], p.logBool, p.normBool, p.xLabel, p.yLabel));
            }
            // default case
            else
            {
                vh.push_back(PHS("MC_DY_" + p.variable, {p.dataCollectionVector}, {2, 1}, "", p.nBins, p.xMin, p.xMax, p.logBool, p.normBool, p.xLabel, p.yLabel));
            }
        }
    }
    
    
    // photon cuts: passPhotonSelection;passLeptVeto;HTZinv>200

    if (doGJetsAndZnunu)
    {
        // Stack HT Plot
        // GJets_HT-200To400
        // GJets_HT-400To600
        // GJets_HT-600ToInf
        
        std::string style_stack = "stack";
        std::vector<std::vector<Plotter::DatasetSummary>> Photon_HT_stack_cuts = {
                                                                                   {makePDSPhoton("200 < HT < 400", "GJets_HT-200To400", "HTZinv>200;metWithPhoton>250")}, 
                                                                                   {makePDSPhoton("400 < HT < 600", "GJets_HT-400To600", "HTZinv>200;metWithPhoton>250")},
                                                                                   {makePDSPhoton("600 > HT",       "GJets_HT-600ToInf", "HTZinv>200;metWithPhoton>250")}
                                                                                 };
        std::vector<std::vector<Plotter::DatasetSummary>> Photon_HT_stack_baseline = {
                                                                                       {makePDSPhoton("200 < HT < 400", "GJets_HT-200To400", "passBaselineZinv")}, 
                                                                                       {makePDSPhoton("400 < HT < 600", "GJets_HT-400To600", "passBaselineZinv")},
                                                                                       {makePDSPhoton("600 > HT",       "GJets_HT-600ToInf", "passBaselineZinv")}
                                                                                     };
  
        Plotter::DataCollection dc_GJets_ht_cuts(     "stack", "HTZinv",  Photon_HT_stack_cuts);
        Plotter::DataCollection dc_GJets_ht_baseline( "stack", "HTZinv",  Photon_HT_stack_baseline);
        Plotter::DataCollection dc_Znunu_ht_cuts(     "data",  "HTZinv",  {makePDSZnunu("HT > 200", "HTZinv>200;metWithPhoton>250")});
        Plotter::DataCollection dc_Znunu_ht_baseline( "data",  "HTZinv",  {makePDSZnunu("HT > 200", "passBaselineZinv")});
        vh.push_back(PHS("MC_GJets_ZJetsToNuNu_ht200_met250_ht_" + style_stack, {dc_GJets_ht_cuts, dc_Znunu_ht_cuts},         {1, 2}, "", 100, 0.0, 2000.0, true, false, label_ht, label_Events));
        vh.push_back(PHS("MC_GJets_ZJetsToNuNu_baseline_ht_"     + style_stack, {dc_GJets_ht_baseline, dc_Znunu_ht_baseline}, {1, 2}, "", 100, 0.0, 2000.0, true, false, label_ht, label_Events));
        
        std::vector<std::string> styles = {"single", "ratio"};
        // Z#rightarrow#nu#nu
        // ratio = GJets / ZJetsToNuNu
        for (const auto & style : styles)
        {
            // denominator index (2 for single, 1 for ratio)
            int d = 0;
            bool log_scale = false;
            std::string y_axis_label;
            std::string legend_label;
            if (style.compare("single") == 0)
            {
                d = 2;
                log_scale = true;
                y_axis_label = label_Events;
                legend_label = "MC";
            }
            else if (style.compare("ratio") == 0)
            {
                d = 1;
                log_scale = false;
                y_axis_label = "GJets / ZJetsToNuNu";
                legend_label = "over ZJetsToNuNu";
            }
            // use metWithPhoton instead of met (it had the photon pt added to it)
            vh.push_back(PHS("MC_GJets_ZJetsToNuNu_ht200_met250_met_"    + style, {makePDCGJetsZnunu("metWithPhoton",         style, legend_label, "HTZinv>200;metWithPhoton>250")}, {1, d}, "", 80, minPt,  maxPt,  log_scale, false, label_met, y_axis_label));
            vh.push_back(PHS("MC_GJets_ZJetsToNuNu_ht200_met250_metphi_" + style, {makePDCGJetsZnunu("metphiWithPhoton",      style, legend_label, "HTZinv>200;metWithPhoton>250")}, {1, d}, "", 80, minPhi, maxPhi, log_scale, false, label_metphi, y_axis_label));
            vh.push_back(PHS("MC_GJets_ZJetsToNuNu_ht200_met250_ht_"     + style, {makePDCGJetsZnunu("HTZinv",                style, legend_label, "HTZinv>200;metWithPhoton>250")}, {1, d}, "", 80, 0.0, 2000.0,    log_scale, false, label_ht,  y_axis_label));
            vh.push_back(PHS("MC_GJets_ZJetsToNuNu_ht200_met250_nj_"     + style, {makePDCGJetsZnunu("cntNJetsPt20Eta24Zinv", style, legend_label, "HTZinv>200;metWithPhoton>250")}, {1, d}, "", 10, 0, 10,           log_scale, false, label_nj,  y_axis_label));
            vh.push_back(PHS("MC_GJets_ZJetsToNuNu_ht200_met250_nb_"     + style, {makePDCGJetsZnunu("cntCSVSZinv",           style, legend_label, "HTZinv>200;metWithPhoton>250")}, {1, d}, "", 10, 0, 10,           log_scale, false, label_nb,  y_axis_label));
            vh.push_back(PHS("MC_GJets_ZJetsToNuNu_ht200_met250_nt_"     + style, {makePDCGJetsZnunu("nTopCandSortedCntZinv", style, legend_label, "HTZinv>200;metWithPhoton>250")}, {1, d}, "", 10, 0, 10,           log_scale, false, label_nt,  y_axis_label));
            vh.push_back(PHS("MC_GJets_ZJetsToNuNu_ht200_met250_dr_"     + style, {makePDCGJetsZnunu("dR_jetsLVec_drPhotonCleaned",  style, legend_label, "HTZinv>200;metWithPhoton>250")}, {1, d}, "", 80, 0, 10.0,        log_scale, false, label_dr,  y_axis_label));
            vh.push_back(PHS("MC_GJets_ZJetsToNuNu_baseline_met_"        + style, {makePDCGJetsZnunu("metWithPhoton",         style, legend_label, "passBaselineZinv")},             {1, d}, "", 80, minPt,  maxPt,  log_scale, false, label_met, y_axis_label));
            vh.push_back(PHS("MC_GJets_ZJetsToNuNu_baseline_metphi_"     + style, {makePDCGJetsZnunu("metphiWithPhoton",      style, legend_label, "passBaselineZinv")},             {1, d}, "", 80, minPhi, maxPhi, log_scale, false, label_metphi, y_axis_label));
            vh.push_back(PHS("MC_GJets_ZJetsToNuNu_baseline_ht_"         + style, {makePDCGJetsZnunu("HTZinv",                style, legend_label, "passBaselineZinv")},             {1, d}, "", 80, 0.0, 2000.0,    log_scale, false, label_ht,  y_axis_label));
            vh.push_back(PHS("MC_GJets_ZJetsToNuNu_baseline_nj_"         + style, {makePDCGJetsZnunu("cntNJetsPt20Eta24Zinv", style, legend_label, "passBaselineZinv")},             {1, d}, "", 10, 0, 10,           log_scale, false, label_nj,  y_axis_label));
            vh.push_back(PHS("MC_GJets_ZJetsToNuNu_baseline_nb_"         + style, {makePDCGJetsZnunu("cntCSVSZinv",           style, legend_label, "passBaselineZinv")},             {1, d}, "", 10, 0, 10,           log_scale, false, label_nb,  y_axis_label));
            vh.push_back(PHS("MC_GJets_ZJetsToNuNu_baseline_nt_"         + style, {makePDCGJetsZnunu("nTopCandSortedCntZinv", style, legend_label, "passBaselineZinv")},             {1, d}, "", 10, 0, 10,           log_scale, false, label_nt,  y_axis_label));
            vh.push_back(PHS("MC_GJets_ZJetsToNuNu_baseline_dr_"         + style, {makePDCGJetsZnunu("dR_jetsLVec_drPhotonCleaned",  style, legend_label, "passBaselineZinv")},             {1, d}, "", 80, 0, 10.0,        log_scale, false, label_dr,  y_axis_label));
        }
    }

    if (doSearchBins)
    {
        vh.push_back(PHS("Trigger_",         {trigger_nSearchBin},           {2, 1}, "passBaseline",     NSB,  0, NSB, false, false,  "Search Bin", "Events", true));
        vh.push_back(PHS("TriggerScl_",      {trigger_nSearchBin_scaled},    {2, 1}, "passBaseline",     NSB,  0, NSB, false, false,  "Search Bin", "Events", true));
        vh.push_back(PHS("TriggerWgt_",      {trigger_nSearchBin_weighted},  {2, 1}, "passBaseline",     NSB,  0, NSB, false, false,  "Search Bin", "Events", true));
        vh.push_back(PHS("Trigger_Zinv_",    {trigger_nSearchBin},           {2, 1}, "passBaselineZinv", NSB,  0, NSB, false, false,  "Search Bin", "Events", true));
        vh.push_back(PHS("TriggerScl_Zinv_", {trigger_nSearchBin_scaled},    {2, 1}, "passBaselineZinv", NSB,  0, NSB, false, false,  "Search Bin", "Events", true));
        vh.push_back(PHS("TriggerWgt_Zinv_", {trigger_nSearchBin_weighted},  {2, 1}, "passBaselineZinv", NSB,  0, NSB, false, false,  "Search Bin", "Events", true));
    }
    //Generate cutflows 
    vector<string> cfsZ = {"",
                           "passNoiseEventFilterZinv",
                           "passNoiseEventFilterZinv;passLeptVeto",
                           "passNoiseEventFilterZinv;passLeptVeto",
                           "passNoiseEventFilterZinv;passLeptVeto;passnJetsZinv",
                           "passNoiseEventFilterZinv;passLeptVeto;passnJetsZinv;passBJetsZinv",
                           "passNoiseEventFilterZinv;passLeptVeto;passnJetsZinv;passBJetsZinv;passTaggerZinv",
                           "passNoiseEventFilterZinv;passLeptVeto;passnJetsZinv;passBJetsZinv;passTaggerZinv;passMETZinv",
                           "passNoiseEventFilterZinv;passLeptVeto;passnJetsZinv;passdPhisZinv;passTaggerZinv;passMETZinv;passBJetsZinv",
                           "passNoiseEventFilterZinv;passLeptVeto;passnJetsZinv;passdPhisZinv;passHTZinv;passMETZinv;passBJetsZinv;passTaggerZinv",
                           "passNoiseEventFilterZinv;passLeptVeto;passnJetsZinv;passdPhisZinv;passHTZinv;passMETZinv;passBJetsZinv;passTaggerZinv;passMT2Zinv",
                           "passLeptVeto;passBaselineZinv"};
    vector<Plotter::CutFlowSummary> cutFlowSummaries;

    cutFlowSummaries.emplace_back(Plotter::CutFlowSummary("ZtoNuNu",           PDC("", "", {dsDY_nunu}),           cfsZ));

    set<AnaSamples::FileSummary> vvf;
    for(auto& fsVec : fileMap) for(auto& fs : fsVec.second) vvf.insert(fs);

    RegisterFunctions* rf = new RegisterFunctionsNTuple(runOnCondor, sbEra);

    //std::cout << "Creating Plotter: Plotter plotter(vh, vvf, fromTuple, filename, nFiles, startFile, nEvts);" << std::endl;
    //printf("    fromTuple: %s\n", fromTuple ? "true" : "false"); fflush(stdout);
    //printf("    filename: %s\n", filename.c_str());              fflush(stdout);
    //printf("    nFiles: %d\n", nFiles);                          fflush(stdout);
    //printf("    startFile: %d\n", startFile);                    fflush(stdout);
    //printf("    nEvts: %d\n", nEvts);                            fflush(stdout);
  
    Plotter plotter(vh, vvf, fromTuple, filename, nFiles, startFile, nEvts);
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
