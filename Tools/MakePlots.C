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
        {"verbose",          no_argument, 0, 'v'},
        {"filename",   required_argument, 0, 'I'},
        {"dataSets",   required_argument, 0, 'D'},
        {"numFiles",   required_argument, 0, 'N'},
        {"startFile",  required_argument, 0, 'M'},
        {"numEvts",    required_argument, 0, 'E'},
        {"plotDir",    required_argument, 0, 'P'},
        {"luminosity", required_argument, 0, 'L'},
        {"sbEra",      required_argument, 0, 'S'},
        {"era",        required_argument, 0, 'Y'}
    };

    bool runOnCondor        = false;
    bool doDataMCElectron   = true;
    bool doDataMCMuon       = true;
    bool doDataMCPhoton     = true;
    bool doWeights = false;
    bool doLeptons = false;
    bool doPhotons = false;
    bool doGJetsAndZnunu = false;
    bool doDYAndZnunu = false;
    bool doSearchBins = true;
    bool doPlots = true;
    bool doSave = true;
    bool doTuple = true;
    bool fromTuple = true;
    bool verbose = false;
    string filename = "histoutput.root", dataSets = "", sampleloc = AnaSamples::fileDir, plotDir = "plots";
    int nFiles = -1, startFile = 0, nEvts = -1;
    double lumi      = -1.0;
    double lumi_2016 = AnaSamples::luminosity_2016;
    double lumi_2017 = AnaSamples::luminosity_2017;
    double lumi_2018 = AnaSamples::luminosity_2018;
    std::string sbEra = "SB_v1_2017";
    std::string era  = "";
    std::string year = "";
    while((opt = getopt_long(argc, argv, "pstfcglvI:D:N:M:E:P:L:S:Y:", long_options, &option_index)) != -1)
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

        case 'v':
            verbose = true;
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
        
        case 'Y':
            era = optarg;
            break;
        }
    }

    // get year from era
    // if era is 2018_AB, year is 2018
    if (era.length() >= 4)
    {
        year = era.substr(0, 4);
    }
    
    // datasets
    std::string ElectronDataset = "Data_SingleElectron";
    std::string MuonDataset     = "Data_SingleMuon";
    std::string PhotonDataset   = "Data_SinglePhoton";
    // year and periods
    std::string eraTag    = "_" + era; 
    std::string yearTag   = "_" + year; 
    std::string periodTag = ""; 
    // HEM veto for 2018 periods C and C
    std::string HEMVeto                 = "";
    std::string HEMVeto_drLeptonCleaned = "";
    std::string HEMVeto_drPhotonCleaned = "";
    std::string semicolon_HEMVeto                 = "";
    std::string semicolon_HEMVeto_drLeptonCleaned = "";
    std::string semicolon_HEMVeto_drPhotonCleaned = "";
    // PrefireWeight
    std::string PrefireWeight = "";

    // lumi for Plotter
    if (era.compare("2016") == 0)
    {
        lumi            = lumi_2016;
        PrefireWeight   = ";PrefireWeight";
    }
    else if (era.compare("2017") == 0)
    {
        lumi            = lumi_2017;
        PrefireWeight   = ";PrefireWeight";
    }
    else if (era.compare("2018") == 0)
    {
        // use lumi for periods A + B + C + D
        lumi            = lumi_2018;
        ElectronDataset = "Data_EGamma";
        PhotonDataset   = "Data_EGamma";
    }
    else if (era.compare("2018_AB") == 0)
    {
        // use lumi for periods A + B
        lumi_2018       = AnaSamples::luminosity_2018_AB;
        lumi            = lumi_2018;
        periodTag       = "_PeriodsAB";
        ElectronDataset = "Data_EGamma";
        PhotonDataset   = "Data_EGamma";
    }
    else if (era.compare("2018_CD") == 0)
    {
        // use lumi for periods C + D
        lumi_2018               = AnaSamples::luminosity_2018_CD;
        lumi                    = lumi_2018;
        // HEM vetos: use ";veto_name" so that it can be appended to cuts
        HEMVeto                 = "SAT_Pass_HEMVeto30";
        HEMVeto_drLeptonCleaned = "SAT_Pass_HEMVeto30_drLeptonCleaned";
        HEMVeto_drPhotonCleaned = "SAT_Pass_HEMVeto30_drPhotonCleaned";
        semicolon_HEMVeto                 = ";" + HEMVeto;
        semicolon_HEMVeto_drLeptonCleaned = ";" + HEMVeto_drLeptonCleaned;
        semicolon_HEMVeto_drPhotonCleaned = ";" + HEMVeto_drPhotonCleaned;
        periodTag               = "_PeriodsCD";
        ElectronDataset         = "Data_EGamma";
        PhotonDataset           = "Data_EGamma";
    }
    else
    {
        std::cout << "Please enter 2016, 2017, 2018, 2018_AB or 2018_CD for the era using the -Y option." << std::endl;
        exit(1);
    }
    
    // add year and period tags
    ElectronDataset = ElectronDataset + yearTag + periodTag;
    MuonDataset     = MuonDataset     + yearTag + periodTag;
    PhotonDataset   = PhotonDataset   + yearTag + periodTag;
    // testing
    //printf("ElectronDataset: %s\n", ElectronDataset.c_str());
    //printf("MuonDataset: %s\n",     MuonDataset.c_str());
    //printf("PhotonDataset: %s\n",   PhotonDataset.c_str());

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


    struct sampleStruct
    {
        AnaSamples::SampleSet           sample_set;
        AnaSamples::SampleCollection    sample_collection;
        std::string                     sample_year;
    };

    // --- follow the syntax; order matters for your arguments --- //
    
    //SampleSet::SampleSet(std::string file, bool isCondor, double lumi)
    AnaSamples::SampleSet        SS_2016("sampleSets_PostProcessed_2016.cfg", runOnCondor, lumi_2016);
    AnaSamples::SampleSet        SS_2017("sampleSets_PostProcessed_2017.cfg", runOnCondor, lumi_2017);
    AnaSamples::SampleSet        SS_2018("sampleSets_PostProcessed_2018.cfg", runOnCondor, lumi_2018);
    
    //SampleCollection::SampleCollection(const std::string& file, SampleSet& samples)
    AnaSamples::SampleCollection SC_2016("sampleCollections_2016.cfg", SS_2016);
    AnaSamples::SampleCollection SC_2017("sampleCollections_2017.cfg", SS_2017);
    AnaSamples::SampleCollection SC_2018("sampleCollections_2018.cfg", SS_2018);

    // Warning: keep years together when you add them to sampleList:
    std::vector<sampleStruct> sampleList;
    sampleList.push_back({SS_2016, SC_2016, "2016"});
    sampleList.push_back({SS_2017, SC_2017, "2017"});
    sampleList.push_back({SS_2018, SC_2018, "2018"});
    
    const double zAcc = 1.0;
    // const double zAcc = 0.5954;
    // const double zAcc = 0.855;
    const double znunu_mumu_ratio = 5.942;
    const double znunu_ee_ratio   = 5.942;

    map<string, vector<AFS>> fileMap;

    //Select approperiate datasets here
    if(dataSets.compare("TEST") == 0)
    {
        fileMap["DYJetsToLL" + yearTag]                = {SS_2016["DYJetsToLL_HT_1200to2500" + yearTag]};
        fileMap["ZJetsToNuNu" + yearTag]               = {SS_2016["ZJetsToNuNu_HT_2500toInf" + yearTag]};
        fileMap["DYJetsToLL_HT_600to800" + yearTag]    = {SS_2016["DYJetsToLL_HT_600to800" + yearTag]};
        fileMap["ZJetsToNuNu_HT_2500toInf" + yearTag]  = {SS_2016["ZJetsToNuNu_HT_2500toInf" + yearTag]};
        fileMap["TTbarDiLep" + yearTag]                = {SS_2016["TTbarDiLep" + yearTag]};
        fileMap["TTbarNoHad" + yearTag]                = {SS_2016["TTbarDiLep" + yearTag]};
        fileMap[MuonDataset]                           = {SS_2016[MuonDataset]};
    }
    else if(dataSets.compare("TEST2") == 0)
    {
        fileMap["DYJetsToLL" + yearTag]              = {SS_2016["DYJetsToLL_HT_600to800" + yearTag]};
        fileMap["DYJetsToLL_HT_600to800" + yearTag]  = {SS_2016["DYJetsToLL_HT_600to800" + yearTag]};
        fileMap["IncDY" + yearTag]                   = {SS_2016["DYJetsToLL_Inc" + yearTag]}; 
        fileMap["TTbarDiLep" + yearTag]              = {SS_2016["TTbarDiLep" + yearTag]};
        fileMap["TTbarNoHad" + yearTag]              = {SS_2016["TTbarDiLep" + yearTag]};
        fileMap[MuonDataset]                         = {SS_2016[MuonDataset]};
    }
    else
    {
        for (const auto& sample : sampleList)
        {
            AnaSamples::SampleSet           ss = sample.sample_set;
            AnaSamples::SampleCollection    sc = sample.sample_collection;
            std::string                     sy = sample.sample_year; 
            // --- calculate total luminosity for data --- //
            //printf("year: %s\n", sy.c_str());
            //printf("%s: lumi = %f\n", (ElectronDataset).c_str(),    sc.getSampleLumi(ElectronDataset));
            //printf("%s: lumi = %f\n", (MuonDataset).c_str(),        sc.getSampleLumi(MuonDataset));
            //printf("%s: lumi = %f\n", (PhotonDataset).c_str(),      sc.getSampleLumi(PhotonDataset));
            // ------------------------------------------- // 
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
    }


    //SearchBins sb(sbEra);
    // Number of searchbins
    int NSB = 204;
    // search bins for low and high dm
    // Low DM, 53 bins: 0 - 52
    // High DM, 151 bins: 53 - 203
    // Total 204 bins: 0 - 203
    int min_sb_low_dm = 0;
    int max_sb_low_dm = 53;
    int min_sb_high_dm = 53;
    int max_sb_high_dm = 204;
    //Validation Bins
    // Low DM, 15 bins: 0 - 14
    // Low DM High MET, 4 bins: 15 - 18
    // High DM, 24 bins: 22 - 45
    // Total 43 bins: 0 - 18 and 22 - 45
    int min_vb_low_dm = 0;
    int max_vb_low_dm = 15;
    int min_vb_low_dm_high_met = 15;
    int max_vb_low_dm_high_met = 19;
    int min_vb_high_dm = 22;
    int max_vb_high_dm = 46;

    // min and max values for histos
    int nBins = 40;
    // p_t in GeV
    double minPt = 0.0;
    double maxPt = 1000.0;
    // Energy in GeV
    double minEnergy = 0.0;
    double maxEnergy = 2000.0;
    int minJets = 0;
    int maxJets = 20;
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

    // met bin edges
    std::vector<double> metBinEdges      = {250.0, 300.0, 400.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0};
    std::vector<double> photonPtBinEdges = {220.0, 300.0, 400.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0};

    // Shortcuts for axis labels
    std::string label_Events = "Events";
    //std::string label_met = "p_{T}^{miss} [GeV]";
    std::string label_met = "#slash{E}_{T} [GeV]";
    std::string label_metWithLL = "#slash{E}_{T}^{LL} [GeV]";
    std::string label_metWithPhoton = "#slash{E}_{T}^{#gamma} [GeV]";
    std::string label_metphi = "#phi_{MET}";
    std::string label_metphiWithLL = "#phi_{MET}^{LL}";
    std::string label_metphiWithPhoton = "#phi_{MET}^{#gamma}";
    std::string label_bestRecoZM  = "m_{LL} [GeV]";
    std::string label_bestRecoZPt = "p_{T}(LL) [GeV]";
    std::string label_ht  = "H_{T} [GeV]";
    std::string label_mtb = "M_{T}(b_{1,2}, #slash{E}_{T}) [GeV]";
    std::string label_ptb = "p_{T}(b) [GeV]";
    std::string label_ISRJetPt = "ISR Jet p_{T} [GeV]"; 
    std::string label_mht = "MH_{T} [GeV]";
    std::string label_nj  = "N_{jets}";
    std::string label_nb  = "N_{bottoms}";
    std::string label_nt  = "N_{tops}";
    std::string label_dr  = "#DeltaR";
    // start with dphi1 for leading jet 1
    std::string label_dphi1  = "#Delta#phi_{1}";
    std::string label_dphi2  = "#Delta#phi_{2}";
    std::string label_dphi3  = "#Delta#phi_{3}";
    std::string label_dphi4  = "#Delta#phi_{4}";
    std::string label_dphi5  = "#Delta#phi_{5}";
    std::vector<std::string> vec_label_dphi = {label_dphi1, label_dphi2, label_dphi3, label_dphi4, label_dphi5}; 
    std::string label_mt2 = "M_{T2} [GeV]";
    std::string label_eta = "#eta";
    std::string label_MuPt = "p_{T}^{#mu} [GeV]";
    std::string label_MuEnergy = "E^{#mu} [GeV]";
    std::string label_MuMass = "m^{#mu} [GeV]";
    std::string label_MuEta = "#eta^{#mu}";
    std::string label_MuPhi = "#phi^{#mu}";
    std::string label_genmupt  = "gen #mu p_{T} [GeV]";
    std::string label_genmueta = "gen #mu #eta";
    std::string label_MuPt1 = "#mu_{1} p_{T} [GeV]";
    std::string label_MuPt2 = "#mu_{2} p_{T} [GeV]";
    std::string label_MuEta1 = "#mu_{1} #eta";
    std::string label_MuEta2 = "#mu_{2} #eta";
    std::string label_ElecPt = "p_{T}^{e} [GeV]";
    std::string label_ElecEnergy = "E^{e} [GeV]";
    std::string label_ElecMass = "m^{e} [GeV]";
    std::string label_ElecEta = "#eta^{e}";
    std::string label_ElecPhi = "#phi^{e}";
    std::string label_ElecPt1 = "e_{1} p_{T} [GeV]";
    std::string label_ElecPt2 = "e_{2} p_{T} [GeV]";
    std::string label_ElecEta1 = "e_{1} #eta";
    std::string label_ElecEta2 = "e_{2} #eta";
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

    //vector<Plotter::HistSummary> vh;
    vector<PHS> vh;
    

    //////// TESTING SOMETHING ADDED BY BRYAN //////////////
    //////// CUTS :: DiLepton Cut ("passDiElecSel","passDiMuSel") , Tagged Top ("nMergedTops_drLeptonCleaned>0")
    //////// Kenimatics: diLeption Mass spectrum (invariant mass). Z(ll) pt spectrum of the two leptons, Resolved top pt, up to 1TeV, n_Tops
    //////// Samples : TTbarSingleLep, TTbarDiLep, WZ, ZZ, ttW, SingleT, ttH, tZq, VVV, DYJetsToLL, ttZ
    typedef std::tuple<std::string,std::string, float, float, bool> Variable;
    typedef std::pair<std::string,std::string>                      StrPair;
    /////// Validation of Selection Cuts ///////
    bool testSel  = false;
    bool doZptCut = true; 
    ///////

    std::vector<StrPair>  mc_samples = { 
      {"DY",               "DYJetsToLL"},
      {"t#bar{t}Z_ll",     "TTZToLLNuNu"},
      {"t#bar{t}Z_nunu",   "TTZToLLNuNu"},
      {"t#bar{t}Z_qq",     "TTZToQQ"},
      {"t#bar{t}W",        "TTW"},
      {"tZq",              "ST_tZq_ll"},
      {"VVV",              "Triboson"},
      {"t#bar{t}H",        "ttH"},
      {"t#bar{t}t#bar{t}", "TTTT"},
      {"WZ",               "WZ_amcatnlo"},
      {"Diboson",          "Diboson"},
      {"t",                "SingleTop"},
      {"t#bar{t}_2l",      "TTbarDiLep"},
      {"t#bar{t}_1l",      "TTbarSingleLep"}};
      
    std::vector<Variable> kenimatic_vars = { // Usage: label, variable, range, logScale? 
      {"Z_ll_mass", "bestRecoZM",                                    0.0, 200.0,  false},
      {"Z_ll_pt",   "bestRecoZPt",                   doZptCut? 300 : 0.0, 1000.0, !doZptCut},
      {"TopPt",     "JetTLV_drLeptonCleaned[0](pt)",                 0.0, 1000.0, !doZptCut}};

    if (testSel) {                                                                 // validate selections
      kenimatic_vars.push_back({"nRTops",    "nResolvedTops_drLeptonCleaned", 0.0, 10.0,   true});
      kenimatic_vars.push_back({"nMTops",    "nMergedTops_drLeptonCleaned",   0.0, 10.0,   true});
      kenimatic_vars.push_back({"nBot",      "nBottoms_drLeptonCleaned",      0.0, 10.0,   true});
    }
    std::string          Zpt_selection  = "bestRecoZPt>300";
    std::vector<StrPair> bot_selections = {
      {"nb0",  "nBottoms_drLeptonCleaned=0"},//{"nb1","nb>=1"},
      {"nbg0", "nBottoms_drLeptonCleaned>0"},
      {"nbg1", "nBottoms_drLeptonCleaned>1"}};
    //{"","NONE"}};{"nb2","nb>=2"}};
    if (testSel){  // validate selections
      bot_selections.push_back({"nbge2","nBottoms_drLeptonCleaned>=2"}); // really >= 1
    }
    std::vector<StrPair> topR_selections = {
      {"nRt0",  "nResolvedTops_drLeptonCleaned=0"},
      {"nRtg0",  "nResolvedTops_drLeptonCleaned>0"},
      {"nRtg1", "nResolvedTops_drLeptonCleaned>1"},
      {"","NONE"}};
    std::vector<StrPair> topM_selections = {
      {"nMt0",  "nMergedTops_drLeptonCleaned=0"},
      {"nMtg0",  "nMergedTops_drLeptonCleaned>0"},
      {"nMtg1", "nMergedTops_drLeptonCleaned>1"},
      {"","NONE"}};
    std::vector<StrPair> topRM_selections = {
      {"nRMtl3","(nResolvedTops_drLeptonCleaned+nMergedTops_drLeptonCleaned)<3"},
      {"","NONE"}};
    std::vector<StrPair> top_selections;
    //
    for (StrPair& r : topR_selections){
      for (StrPair& m : topM_selections){
	if      (r.second == "NONE" && m.second == "NONE") top_selections.push_back({"","NONE"});
	else if (r.second != "NONE" && m.second == "NONE") top_selections.push_back({r.first,r.second});
	else if (r.second == "NONE" && m.second != "NONE") top_selections.push_back({m.first,m.second});
	else                                               top_selections.push_back({r.first+"_"+m.first, r.second+";"+m.second});
      }
    }
    for (StrPair& t : top_selections){
      for (StrPair& rm : topRM_selections){
	if (t.second  == "NONE") continue;
	else if (rm.second == "NONE") continue;
	else top_selections.push_back({t.first+"_"+rm.first, t.second+";"+rm.second});
      }
    }
    if (testSel) {                                                               // validate selections
      top_selections.push_back({"nRtge2","nResolvedTops_drLeptonCleaned>=2"}); // really >= 1
      top_selections.push_back({"nRteg2","nResolvedTops_drLeptonCleaned=>2"}); // doesnt work
      top_selections.push_back({"nRt2","nResolvedTops_drLeptonCleaned=2"});    // works
      top_selections.push_back({"nMtge2","nMergedTops_drLeptonCleaned>=2"});   // really >= 1
      top_selections.push_back({"nMteg2","nMergedTops_drLeptonCleaned=>2"});   // doesnt work
      top_selections.push_back({"nMt2","nMergedTops_drLeptonCleaned=2"});      // works
      }
    // === PDS === // 
    std::string ttz_elec_cuts      = "passElecZinvSelOnZMassPeak";//"passDiElecSel";
    std::string ttz_mu_cuts        = "passMuZinvSelOnZMassPeak";//"passDiMuSel";
    std::string ttz_elec_diLepCuts = "passDiElecSel";
    std::string ttz_mu_diLepCuts   = "passDiMuSel";
    //
    std::vector<std::vector<PDS>> dsTTZ_elec_stack ;
    std::vector<std::vector<PDS>> dsTTZ_mu_stack ;
    std::vector<std::vector<PDS>> dsTTZ_elec_loose_stack ;
    std::vector<std::vector<PDS>> dsTTZ_mu_loose_stack ;
    //
    for ( StrPair& sample : mc_samples){
      if (sample.first == "t#bar{t}Z_ll"){
	dsTTZ_elec_stack.push_back({PDS( sample.first, fileMap[sample.second + yearTag], ttz_elec_cuts+";isZToLL", "")});
	dsTTZ_mu_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_mu_cuts+";isZToLL",   "")});
      }
      else if (sample.first == "t#bar{t}Z_nunu"){
	dsTTZ_elec_stack.push_back({PDS( sample.first, fileMap[sample.second + yearTag], ttz_elec_cuts+";!isZToLL", "")});
	dsTTZ_mu_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_mu_cuts+";!isZToLL",   "")});
      }
      else{
	dsTTZ_elec_stack.push_back({PDS( sample.first, fileMap[sample.second + yearTag], ttz_elec_cuts, "")});
	dsTTZ_mu_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_mu_cuts, "")});
      }
      //
      dsTTZ_elec_loose_stack.push_back({PDS( sample.first, fileMap[sample.second + yearTag], ttz_elec_diLepCuts, "")});
      dsTTZ_mu_loose_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_mu_diLepCuts, "")});
    }
    PDS dsElectronData("Data", fileMap[ElectronDataset],  ttz_elec_cuts, "");
    PDS dsMuonData(    "Data", fileMap[MuonDataset],      ttz_mu_cuts, "");
    // === PDC === //
    std::vector<PDC> dsMC_DiLep_elec;
    std::vector<PDC> dsMC_DiLep_mu;
    std::vector<PDC> dsMC_DiLepLoose_elec;
    std::vector<PDC> dsMC_DiLepLoose_mu;
    std::vector<PDC> dsData_Elec;
    std::vector<PDC> dsData_Mu;
    //
    for ( Variable& var : kenimatic_vars ){
      dsMC_DiLep_elec.push_back(PDC(     "stack", std::get<1>(var), dsTTZ_elec_stack));
      dsMC_DiLep_mu.push_back(PDC(       "stack", std::get<1>(var), dsTTZ_mu_stack));
      //
      //dsMC_DiLepLoose_elec.push_back(PDC("stack", std::get<1>(var), dsTTZ_elec_loose_stack));
      //dsMC_DiLepLoose_mu.push_back(PDC(  "stack", std::get<1>(var), dsTTZ_mu_loose_stack));
      //
      dsData_Elec.push_back(PDC(         "data",  std::get<1>(var), {dsElectronData}));
      dsData_Mu.push_back(PDC(           "data",  std::get<1>(var), {dsMuonData}));
    }
    dsMC_DiLepLoose_elec.push_back(PDC("stack", "bestRecoZM", dsTTZ_elec_loose_stack));
    dsMC_DiLepLoose_mu.push_back(PDC(  "stack", "bestRecoZM", dsTTZ_mu_loose_stack));

    // === PHS === //
    for(int i = 0 ; i < dsMC_DiLep_elec.size() ; i++){
      for(StrPair& b_sel : bot_selections){
	for(StrPair& top_sel : top_selections){
	  std::string sel_label, sel_cut;
	  bool addData = false; // for viability tests, only add data for very loose selection requirements
	  if (b_sel.second == "NONE" && top_sel.second == "NONE"){
	    sel_label = "";
	    sel_cut   = "";
	    addData = true;
	  }
	  else if (b_sel.second == "NONE" && top_sel.second != "NONE"){ 
	    sel_label = "_"+top_sel.first; 
	    sel_cut   = top_sel.second; 
	  }
	  else if (b_sel.second != "NONE" && top_sel.second == "NONE"){
	    sel_label = "_"+b_sel.first; 
	    sel_cut   = b_sel.second; 
	  }
	  else{
	    sel_label = "_"+b_sel.first+"_"+top_sel.first;
	    sel_cut   = b_sel.second+";"+top_sel.second;
	    if (doZptCut) sel_cut = sel_cut + ";"+Zpt_selection;
	  }
	  ////
	  if (addData){
	    vh.push_back(PHS("MCData_DiLep_elec_"+std::get<0>(kenimatic_vars[i])+sel_label+eraTag,            {dsData_Elec[i], dsMC_DiLep_elec[i]},      {1,1}, sel_cut, 
			     60, std::get<2>(kenimatic_vars[i]), std::get<3>(kenimatic_vars[i]), std::get<4>(kenimatic_vars[i]), false, std::get<0>(kenimatic_vars[i]), "Events"));
	    vh.push_back(PHS("MCData_DiLep_mu_"+std::get<0>(kenimatic_vars[i])+sel_label+eraTag,              {dsData_Mu[i], dsMC_DiLep_mu[i]},          {1,1}, sel_cut, 
			     60, std::get<2>(kenimatic_vars[i]), std::get<3>(kenimatic_vars[i]), std::get<4>(kenimatic_vars[i]), false, std::get<0>(kenimatic_vars[i]), "Events"));
	    vh.push_back(PHS("MCData_DiLep_NoZMassCut_elec_"+std::get<0>(kenimatic_vars[0])+sel_label+eraTag, {dsData_Elec[i], dsMC_DiLepLoose_elec[0]}, {1,1}, sel_cut, 
			     60, std::get<2>(kenimatic_vars[0]), std::get<3>(kenimatic_vars[0]), std::get<4>(kenimatic_vars[0]), false, std::get<0>(kenimatic_vars[0]), "Events"));
	    vh.push_back(PHS("MCData_DiLep_NoZMassCut_mu_"+std::get<0>(kenimatic_vars[0])+sel_label+eraTag,   {dsData_Mu[i],   dsMC_DiLepLoose_mu[0]},   {1,1}, sel_cut, 
			     60, std::get<2>(kenimatic_vars[0]), std::get<3>(kenimatic_vars[0]), std::get<4>(kenimatic_vars[0]), false, std::get<0>(kenimatic_vars[0]), "Events"));
	  }
	  else{
	    vh.push_back(PHS("MC_DiLep_elec_"+std::get<0>(kenimatic_vars[i])+sel_label+eraTag,            {dsMC_DiLep_elec[i]},      {1,1}, sel_cut, 
			     60, std::get<2>(kenimatic_vars[i]), std::get<3>(kenimatic_vars[i]), std::get<4>(kenimatic_vars[i]), false, std::get<0>(kenimatic_vars[i]), "Events"));
	    vh.push_back(PHS("MC_DiLep_mu_"+std::get<0>(kenimatic_vars[i])+sel_label+eraTag,              {dsMC_DiLep_mu[i]},        {1,1}, sel_cut, 
			     60, std::get<2>(kenimatic_vars[i]), std::get<3>(kenimatic_vars[i]), std::get<4>(kenimatic_vars[i]), false, std::get<0>(kenimatic_vars[i]), "Events"));
	    vh.push_back(PHS("MC_DiLep_NoZMassCut_elec_"+std::get<0>(kenimatic_vars[0])+sel_label+eraTag, {dsMC_DiLepLoose_elec[0]}, {1,1}, sel_cut, 
			     60, std::get<2>(kenimatic_vars[0]), std::get<3>(kenimatic_vars[0]), std::get<4>(kenimatic_vars[0]), false, std::get<0>(kenimatic_vars[0]), "Events"));
	    vh.push_back(PHS("MC_DiLep_NoZMassCut_mu_"+std::get<0>(kenimatic_vars[0])+sel_label+eraTag,   {dsMC_DiLepLoose_mu[0]},   {1,1}, sel_cut, 
			     60, std::get<2>(kenimatic_vars[0]), std::get<3>(kenimatic_vars[0]), std::get<4>(kenimatic_vars[0]), false, std::get<0>(kenimatic_vars[0]), "Events"));
	  }
	}
      }
    }
    //vh.push_back(PHS("MC_WZ", {{PDC("stack", "bestRecoZM", {PDS("WZ", fileMap["WZ_amcatnlo" + yearTag], "", "")})}}, {1,1}, "", 60, 0, 500, true, false, "Z_ll_mass", "Events"));

    //////// END OF  THE  TEST ADDED BY BRYAN //////////////
    

    set<AFS> vvf;
    for(auto& fsVec : fileMap) for(auto& fs : fsVec.second) vvf.insert(fs);

    RegisterFunctions* rf = new RegisterFunctionsNTuple(runOnCondor, sbEra, year);

    if (verbose)
    {
        std::cout << "Creating Plotter: Plotter plotter(vh, vvf, fromTuple, filename, nFiles, startFile, nEvts);" << std::endl;
        printf("    fromTuple: %s\n", fromTuple ? "true" : "false"); fflush(stdout);
        printf("    filename: %s\n", filename.c_str());              fflush(stdout);
        printf("    nFiles: %d\n", nFiles);                          fflush(stdout);
        printf("    startFile: %d\n", startFile);                    fflush(stdout);
        printf("    nEvts: %d\n", nEvts);                            fflush(stdout);
    }
  
    Plotter plotter(vh, vvf, fromTuple, filename, nFiles, startFile, nEvts);
    plotter.setLumi(lumi);
    plotter.setPlotDir(plotDir);
    plotter.setDoHists(doSave || doPlots);
    plotter.setDoTuple(doTuple);
    plotter.setRegisterFunction(rf);
    plotter.read();
    if(doSave && fromTuple)  plotter.saveHists();
    if(doPlots)              plotter.plot();
}
