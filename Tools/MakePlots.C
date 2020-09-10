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
      {"doPre",            no_argument, 0, 'b'},
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
    bool doPre   = false;
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
    while((opt = getopt_long(argc, argv, "bpstfcglvI:D:N:M:E:P:L:S:Y:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
	case 'b':
	  doPre = true;
	  break;
	  
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
    std::string sample_step = "Post";
    if (doPre)
      {
	sample_step = "Pre";
      }
     
    AnaSamples::SampleSet        SS_2016("sampleSets_"+sample_step+"Processed_2016.cfg", runOnCondor, lumi_2016);
    AnaSamples::SampleSet        SS_2017("sampleSets_"+sample_step+"Processed_2017.cfg", runOnCondor, lumi_2017);
    AnaSamples::SampleSet        SS_2018("sampleSets_"+sample_step+"Processed_2018.cfg", runOnCondor, lumi_2018);
    
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
    //////////////===================================================================================================/////////////////////////
    std::vector<Plotter::Scanner> scanners;
    std::string tag = "Training";
    std::string lClean = "_drLeptonCleaned";
    std::set<std::string> vars = {
      
      "nResolvedTops"+lClean, "nMergedTops"+lClean, "nBottoms"+lClean, "nSoftBottoms"+lClean, "nJets30"+lClean,              // validation
      "MET_pt", "MET_phi", "Pass_IsoTrkVeto", "Pass_TauVeto", "Pass_ElecVeto", "Pass_MuonVeto",                              // validation
      "passSingleLepElec", "passSingleLepMu", "Lep_pt", "Lep_eta" ,"Lep_phi", "Lep_E",                                       // validation

      "Muon_pt","Muon_eta","Muon_phi","Muon_mass",
      "Muon_miniPFRelIso_all",
      "Muon_FlagId", "Muon_pfRelIso04_all",
      "Electron_pt","Electron_eta","Electron_phi","Electron_mass",
      "Electron_miniPFRelIso_all", "Electron_cutBasedNoIso", "Electron_cutBased", 

      "ResolvedTopCandidate_discriminator",                                                                                  // validation
      "ResolvedTopCandidate_j1Idx", "ResolvedTopCandidate_j2Idx", "ResolvedTopCandidate_j3Idx",                              // validation
      //"ResolvedTopCandidate_pt", "ResolvedTopCandidate_eta", "ResolvedTopCandidate_phi", "ResolvedTopCandidate_mass",        // validation

      "Jet_btagDeepB"+lClean, "Jet_deepFlavourb"+lClean, "Jet_deepFlavourbb"+lClean,                                            // training 1 (AK4)
      "Jet_deepFlavourlepb"+lClean, "Jet_deepFlavouruds"+lClean,                                                                // training 1 (AK4)
      "Jet_pt"+lClean, "Jet_eta"+lClean, "Jet_phi"+lClean, "Jet_mass"+lClean,                                                   // training 1 (AK4)
      //"Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass",                                                                               // training 1 (AK4)
      "Jet_lepcleaned_idx", // test

      "FatJet_pt"+lClean, "FatJet_eta"+lClean, "FatJet_phi"+lClean, "FatJet_mass"+lClean,                                       // training 2 (AK8)
      "FatJet_msoftdrop"+lClean, 
      "FatJet_rawFactor"+lClean,"FatJet_area"+lClean,
      "SubJet_rawFactor",

      //"FatJet_pt", "FatJet_eta", "FatJet_phi", "FatJet_E",                                                                   // training 2 (AK8)
      //"FatJet_tau1"+lClean, "FatJet_tau2"+lClean, "FatJet_tau3"+lClean, "FatJet_tau4"+lClean,                                // training 2 (AK8) 
      "FatJet_deepTag_WvsQCD"+lClean, "FatJet_deepTag_TvsQCD"+lClean, "FatJet_deepTag_ZvsQCD"+lClean,                        // training 2 (AK8)
      "FatJet_deepTagMD_H4qvsQCD"+lClean, "FatJet_deepTagMD_HbbvsQCD"+lClean, "FatJet_deepTagMD_TvsQCD"+lClean,              // training 2 (AK8)
      "FatJet_deepTagMD_WvsQCD"+lClean, "FatJet_deepTagMD_ZHbbvsQCD"+lClean, "FatJet_deepTagMD_ZHccvsQCD"+lClean,            // training 2 (AK8)
      "FatJet_deepTagMD_ZbbvsQCD"+lClean, "FatJet_deepTagMD_ZvsQCD"+lClean, "FatJet_deepTagMD_bbvsLight"+lClean,             // training 2 (AK8)
      "FatJet_deepTagMD_ccvsLight"+lClean,                                                                                   // training 2 (AK8)
      "FatJet_btagDeepB"+lClean, "FatJet_btagHbb"+lClean,                                                                    // training 2 (AK8)
      "FatJet_subJetIdx1"+lClean, "FatJet_subJetIdx2"+lClean,                                                                // training 2 (ak8)
      "SubJet_pt", "SubJet_eta", "SubJet_phi", "SubJet_mass", "SubJet_btagDeepB",                                            // training 2 (ak8) 
      
      "SAT_Pass_HEMVeto_DataOnly"+lClean, "SAT_Pass_HEMVeto_DataAndMC"+lClean, "SAT_HEMVetoWeight"+lClean, // 2018 only
      "SAT_Pass_HEMVeto_DataOnly", "SAT_Pass_HEMVeto_DataAndMC", "SAT_HEMVetoWeight", // 2018 only
      "run",
      "Pass_trigger_muon", "Pass_trigger_electron",           // Trigger
      "HLT_IsoMu24" , "HLT_IsoMu27", "HLT_Mu50", 
      "HLT_Ele27_WPTight_Gsf", "HLT_Photon175",
      "HLT_Ele115_CaloIdVT_GsfTrkIdT", "HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165"
    };

    std::set<std::string> mc_vars = {
      "GenPart_pdgId", "GenPart_genPartIdxMother", "GenPart_status" , "genTtbarId",                                                     // gen validation
      "GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass","genWeight",

      "GenJetAK8_pt", "GenJetAK8_eta", "GenJetAK8_phi", "GenJetAK8_mass",
      //"SubGenJetAK8_pt", "SubGenJetAK8_eta", "SubGenJetAK8_phi", "SubGenJetAK8_mass",
      

      "LHEScaleWeight","PSWeight", // systematic

      "BTagWeight","puWeight","ISRWeight",
      "BTagWeightHeavy","BTagWeightLight",
      "Stop0l_topptWeight","Stop0l_topMGPowWeight", //"Stop0l_topptOnly"    // Weights // for 2018
      // JES JER related variables
      "Jet_lepcleaned_idx_jesTotalUp","Jet_lepcleaned_idx_jesTotalDown","Jet_lepcleaned_idx_jerUp","Jet_lepcleaned_idx_jerDown", // test

      "Jet_pt_jesTotalUp"+lClean, "Jet_pt_jesTotalDown"+lClean, "Jet_mass_jesTotalUp"+lClean, "Jet_mass_jesTotalDown"+lClean,// Systematics
      "Jet_pt_jerUp"+lClean, "Jet_pt_jerDown"+lClean, "Jet_mass_jerUp"+lClean, "Jet_mass_jerDown"+lClean,                    // Systematics
      //"Jet_pt_jesTotalUp", "Jet_pt_jesTotalDown", "Jet_mass_jesTotalUp", "Jet_mass_jesTotalDown",// Systematics
      //"Jet_pt_jerUp", "Jet_pt_jerDown", "Jet_mass_jerUp", "Jet_mass_jerDown",                    // Systematics
      "MET_pt_jesTotalUp", "MET_pt_jesTotalDown", "MET_pt_jerUp", "MET_pt_jerDown",

      // HEM veto weight, njets , nbottoms
      "nJets30_JESUp"+lClean,"nJets30_JESDown"+lClean,"nJets30_JERUp"+lClean,"nJets30_JERDown"+lClean,
      "nBottoms_JESUp"+lClean,"nBottoms_JESDown"+lClean,"nBottoms_JERUp"+lClean,"nBottoms_JERDown"+lClean,

      "ResolvedTopCandidate_JESUp_discriminator",                                                                                  // validation
      "ResolvedTopCandidate_JESUp_j1Idx", "ResolvedTopCandidate_JESUp_j2Idx", "ResolvedTopCandidate_JESUp_j3Idx",                              // validation
      //"ResolvedTopCandidate_JESUp_pt", "ResolvedTopCandidate_JESUp_eta", "ResolvedTopCandidate_JESUp_phi", "ResolvedTopCandidate_JESUp_mass",        // validation
      "ResolvedTopCandidate_JESDown_discriminator",                                                                                  // validation
      "ResolvedTopCandidate_JESDown_j1Idx", "ResolvedTopCandidate_JESDown_j2Idx", "ResolvedTopCandidate_JESDown_j3Idx",                              // validation
      //"ResolvedTopCandidate_JESDown_pt", "ResolvedTopCandidate_JESDown_eta", "ResolvedTopCandidate_JESDown_phi", "ResolvedTopCandidate_JESDown_mass",        // validation
      // end JES, JER variations
      "BTagWeight_Up", "BTagWeight_Down",                                                                                    // Systematics
      "BTagWeightLight_Up", "BTagWeightLight_Down",                                                                                    // Systematics
      "BTagWeightHeavy_Up", "BTagWeightHeavy_Down",                                                                                    // Systematics
      "puWeight_Up","puWeight_Down", "pdfWeight_Up","pdfWeight_Down",
      //"Stop0l_topptOnly_Up","Stop0l_topptOnly_Down",   // dont work atm       // Systematics
      "ISRWeight_Up","ISRWeight_Down",
      //
      "SAT_HEMVetoWeight_JESUp"+lClean, "SAT_HEMVetoWeight_JESDown"+lClean,
      "SAT_HEMVetoWeight_JERUp"+lClean, "SAT_HEMVetoWeight_JERDown"+lClean,
      "SAT_HEMVetoWeight_JESUp", "SAT_HEMVetoWeight_JESDown",
      "SAT_HEMVetoWeight_JERUp", "SAT_HEMVetoWeight_JERDown"
      
      //"Stop0l_trigger_eff_Electron_pt", "Stop0l_trigger_eff_Muon_pt", 
      //"Stop0l_trigger_eff_Electron_eta", "Stop0l_trigger_eff_Muon_eta", 
      //"Stop0l_trigger_eff_Electron_pt_up", "Stop0l_trigger_eff_Muon_pt_up", 
      //"Stop0l_trigger_eff_Electron_eta_up", "Stop0l_trigger_eff_Muon_eta_up", 
      //"Stop0l_trigger_eff_Electron_pt_down", "Stop0l_trigger_eff_Muon_pt_down", 
      //"Stop0l_trigger_eff_Electron_eta_down", "Stop0l_trigger_eff_Muon_eta_down", 
      //
      //"Stop0l_trigger_eff_Photon_pt", "Stop0l_trigger_eff_Photon_eta_up",
      //"Stop0l_trigger_eff_Photon_pt_up",  "Stop0l_trigger_eff_Photon_pt_down", 
      //"Stop0l_trigger_eff_Photon_eta_up", "Stop0l_trigger_eff_Photon_eta_down", 
    };
    std::set<std::string> hlt_vars_non2016 = {
      "HLT_Ele35_WPTight_Gsf", "HLT_Ele32_WPTight_Gsf_L1DoubleEG",
      "HLT_Photon200", "HLT_Ele28_eta2p1_WPTight_Gsf_HT150", "HLT_Ele32_WPTight_Gsf",
      "HLT_OldMu100", "HLT_TkMu100"

    };
    //std::set<std::string> hlt_vars_non2017 = {
    //  "HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
    //  "HLT_Ele115_CaloIdVT_GsfTrkIdT"
    //};
    std::set<std::string> mc_vars_non2018 = {
      "PrefireWeight",
      "PrefireWeight_Up","PrefireWeight_Down"
    };
    std::cout<<"\n"<<dataSets<<"\t"<<MuonDataset<<"\t"<<dataSets.find(MuonDataset)<<"\t"<<era<<"\n";
    if (era.find("2016")!=std::string::npos){
      vars.insert("HLT_IsoTkMu24");
      vars.insert("HLT_TkMu50");
      vars.insert("HLT_Ele45_CaloIdVT_GsfTrkIdT_PFJet200_PFJet50");
      
    }
    //if (era.find("2018")!=std::string::npos){
    //  //vars.insert("HLT_Ele32_WPTight_Gsf");
    //  vars.insert("HLT_OldMu100");
    //  vars.insert("HLT_TkMu100");
    //}
    if (era.find("2016")==std::string::npos){ 
      vars.insert(hlt_vars_non2016.begin(),hlt_vars_non2016.end());
    }
    //if (era.find("2017")==std::string::npos){ 
    //  vars.insert(hlt_vars_non2017.begin(),hlt_vars_non2017.end());
    //}
    if (era.find("2018")==std::string::npos){
      mc_vars.insert(mc_vars_non2018.begin(), mc_vars_non2018.end());
    }
    // check if not data
    if (dataSets.find(ElectronDataset)==std::string::npos && dataSets.find(MuonDataset)==std::string::npos){
      vars.insert(mc_vars.begin(), mc_vars.end());
    }

    //// fill in PDS here (DataSets to train on) 
    std::vector< std::pair<std::string,std::string> > lepTags = {
      //{tag,        "passElecZinvSelOnZMassPeak"},
      //{tag,        "passMuZinvSelOnZMassPeak"},
      //{tag+"_Inv", "Pass_LeptonVeto;MET_pt>200"},
      //{tag+"_bb",  "passSingleLepMu;passFatJetPt;nBottoms_drLeptonCleaned>=2;nJets30_drLeptonCleaned>=4;MET_pt>=20"},
      //{tag+"_bb",  "passSingleLepElec;passFatJetPt;nBottoms_drLeptonCleaned>=2;nJets30_drLeptonCleaned>=4;MET_pt>=20"}};
      //{tag+"_bb",  "passLooseSingleLep;passFatJetPt;nBottoms_drLeptonCleaned>=2;nJets30_drLeptonCleaned>=4;MET_pt>=20"}};
      //{tag+"_bb",  "passBCSingleLep;passFatJetPt;pass_BC_nbs_drLeptonCleaned;pass_BC_njet_drLeptonCleaned"}};
      {tag+"_bb",  "passSingleLep;passBCFatJetPt_drLeptonCleaned;pass_BC_nbs_drLeptonCleaned;pass_BC_njet_drLeptonCleaned"}};
    for ( std::pair<std::string,std::string> lepType : lepTags) {
	PDS dsDY      = PDS("DY",        fileMap["DYJetsToLL"+eraTag],  lepType.second,              "");
	PDS dsZJ      = PDS("ZJets",     fileMap["ZJetsToNuNu"+eraTag], lepType.second,              "");
	PDS dsWJ      = PDS("WJets",     fileMap["WJetsToLNu"+eraTag],  lepType.second,              "");
	PDS dsTTZ     = PDS("TTZ",       fileMap["TTZToLLNuNu"+eraTag], lepType.second,              "");
	//PDS dsTTHad   = PDS("TTBarHad",  fileMap["TTbarHad"+eraTag],    lepType.second+";isTAllHad",  ""); // for non powheg
	//PDS dsTTLep   = PDS("TTBarLep",  fileMap["TTbarNoHad"+eraTag],  lepType.second,              "");
	PDS dsTTHad_pow   = PDS("TTBarHad_pow",  fileMap["TTbarHad_pow"+eraTag],    lepType.second,             "");
	PDS dsTTSemi_pow   = PDS("TTBarSemi_pow",  fileMap["TTbarSemi_pow"+eraTag],  lepType.second,              "");
	PDS dsTTDi_pow   = PDS("TTBarDi_pow",  fileMap["TTbarDi_pow"+eraTag],  lepType.second,              "");
	PDS dsTTbbHad_pow   = PDS("TTbbHad_pow",  fileMap["TTbbHad_pow"+eraTag],  lepType.second,              "");
	PDS dsTTbbSemi_pow   = PDS("TTbbSemi_pow",  fileMap["TTbbSemi_pow"+eraTag],  lepType.second,              "");
	PDS dsTTbbDi_pow   = PDS("TTbbDi_pow",  fileMap["TTbbDi_pow"+eraTag],  lepType.second,              "");
	// ttbar dedicated systematcis 
	PDS dsTTHad_pow_erdOn   = PDS("TTBarHad_pow_erdOn",      fileMap["TTbarHad_pow_erdOn"+eraTag],    lepType.second,             "");
	PDS dsTTHad_pow_UEDown   = PDS("TTBarHad_pow_UEDown",     fileMap["TTbarHad_pow_UEDown"+eraTag],    lepType.second,             "");
	PDS dsTTHad_pow_UEUp   = PDS("TTBarHad_pow_UEDUp",      fileMap["TTbarHad_pow_UEUp"+eraTag],    lepType.second,             "");
	PDS dsTTHad_pow_hdampDown   = PDS("TTBarHad_pow_hdampDown",  fileMap["TTbarHad_pow_hdampDown"+eraTag],    lepType.second,             "");
	PDS dsTTHad_pow_hdampUp   = PDS("TTBarHad_pow_hdampUp",    fileMap["TTbarHad_pow_hdampUp"+eraTag],    lepType.second,             "");
	//
	PDS dsTTSemi_pow_erdOn   = PDS("TTBarSemi_pow_erdOn",      fileMap["TTbarSemi_pow_erdOn"+eraTag],    lepType.second,             "");
	PDS dsTTSemi_pow_UEDown   = PDS("TTBarSemi_pow_UEDown",     fileMap["TTbarSemi_pow_UEDown"+eraTag],    lepType.second,             "");
	PDS dsTTSemi_pow_UEUp   = PDS("TTBarSemi_pow_UEDUp",      fileMap["TTbarSemi_pow_UEUp"+eraTag],    lepType.second,             "");
	PDS dsTTSemi_pow_hdampDown   = PDS("TTBarSemi_pow_hdampDown",  fileMap["TTbarSemi_pow_hdampDown"+eraTag],    lepType.second,             "");
	PDS dsTTSemi_pow_hdampUp   = PDS("TTBarSemi_pow_hdampUp",    fileMap["TTbarSemi_pow_hdampUp"+eraTag],    lepType.second,             "");
	//
	PDS dsTTDi_pow_erdOn   = PDS("TTBarDi_pow_erdOn",      fileMap["TTbarDi_pow_erdOn"+eraTag],    lepType.second,             "");
	PDS dsTTDi_pow_UEDown   = PDS("TTBarDi_pow_UEDown",     fileMap["TTbarDi_pow_UEDown"+eraTag],    lepType.second,             "");
	PDS dsTTDi_pow_UEUp   = PDS("TTBarDi_pow_UEDUp",      fileMap["TTbarDi_pow_UEUp"+eraTag],    lepType.second,             "");
	PDS dsTTDi_pow_hdampDown   = PDS("TTBarDi_pow_hdampDown",  fileMap["TTbarDi_pow_hdampDown"+eraTag],    lepType.second,             "");
	PDS dsTTDi_pow_hdampUp   = PDS("TTBarDi_pow_hdampUp",    fileMap["TTbarDi_pow_hdampUp"+eraTag],    lepType.second,             "");
	//
	PDS dsTTbbHad_pow_hdampDown   = PDS("TTbbHad_pow_hdampDown",  fileMap["TTbbHad_pow_hdampDown"+eraTag],    lepType.second,             "");
	PDS dsTTbbHad_pow_hdampUp   = PDS("TTbbHad_pow_hdampUp",    fileMap["TTbbHad_pow_hdampUp"+eraTag],    lepType.second,             "");
	//
	PDS dsTTbbSemi_pow_hdampDown   = PDS("TTbbSemi_pow_hdampDown",  fileMap["TTbbSemi_pow_hdampDown"+eraTag],    lepType.second,             "");
	PDS dsTTbbSemi_pow_hdampUp   = PDS("TTbbSemi_pow_hdampUp",    fileMap["TTbbSemi_pow_hdampUp"+eraTag],    lepType.second,             "");
	//
	PDS dsTTbbDi_pow_hdampDown   = PDS("TTbbDi_pow_hdampDown",  fileMap["TTbbDi_pow_hdampDown"+eraTag],    lepType.second,             "");
	PDS dsTTbbDi_pow_hdampUp   = PDS("TTbbDi_pow_hdampUp",    fileMap["TTbbDi_pow_hdampUp"+eraTag],    lepType.second,             "");
	//
	PDS dsVV      = PDS("DiBoson",   fileMap["Diboson"+eraTag],     lepType.second,              "");
	PDS dsVVV     = PDS("TriBoson",  fileMap["Triboson"+eraTag],    lepType.second,              "");
	PDS dsTTX     = PDS("TTX",       fileMap["TTX"+eraTag],         lepType.second,              "");
	PDS dsTTXqq   = PDS("TTX",       fileMap["TTXqq"+eraTag],       lepType.second,              "");
	PDS dsQCD     = PDS("QCD",       fileMap["QCD"+eraTag],         lepType.second,              "");	
	PDS dsTTZH    = PDS("TTZH",      fileMap["TTZH"+eraTag],        lepType.second,              "");
	PDS dsttZ_bb  = PDS("TTZ_bb",    fileMap["TTZtoBB"+eraTag],     lepType.second,              "");
	PDS dsMuData  = PDS("MuData",    fileMap["Data_SingleMuon"+eraTag],              lepType.second,              "");
	PDS dsEleData = PDS("EleData",   fileMap["Data_SingleElectron"+eraTag],          lepType.second,              "");
	PDS dsEGamma  = PDS("EleData",   fileMap["Data_EGamma"+eraTag],                  lepType.second,              "");
	//
	//if (lepType.first == "Training"){
	//  scanners.push_back(Plotter::Scanner(lepType.first, vars, {dsDY, dsWJ, dsTTZ, dsTTHad, dsTTLep, dsVV, dsVVV, dsTTX}));
	//}
	//else if (lepType.first == "Training_Inv"){
	//  scanners.push_back(Plotter::Scanner(lepType.first, vars, {dsZJ, dsWJ, dsTTZ, dsTTHad, dsTTLep, dsVV, dsVVV, dsTTX, dsQCD}));
	//}
	if (lepType.first == "Training_bb"){
	  scanners.push_back(Plotter::Scanner(lepType.first, vars, {dsTTZH, dsttZ_bb, dsDY, dsZJ, dsWJ, 
		  dsTTHad_pow, dsTTHad_pow_erdOn, dsTTHad_pow_UEDown, dsTTHad_pow_UEUp, dsTTHad_pow_hdampDown,  dsTTHad_pow_hdampUp,  
		  dsTTSemi_pow, dsTTSemi_pow_erdOn, dsTTSemi_pow_UEDown, dsTTSemi_pow_UEUp, dsTTSemi_pow_hdampDown,  dsTTSemi_pow_hdampUp,  
		  dsTTDi_pow, dsTTDi_pow_erdOn, dsTTDi_pow_UEDown, dsTTDi_pow_UEUp, dsTTDi_pow_hdampDown,  dsTTDi_pow_hdampUp,  
		  dsTTbbSemi_pow, dsTTbbSemi_pow_hdampDown, dsTTbbSemi_pow_hdampUp,
		  dsTTbbHad_pow, dsTTbbHad_pow_hdampDown, dsTTbbHad_pow_hdampUp,
		  dsTTbbDi_pow, dsTTbbDi_pow_hdampDown, dsTTbbDi_pow_hdampUp,
		  dsVV, dsVVV, dsTTXqq, dsQCD, dsMuData, dsEleData, dsEGamma}));
	  //scanners.push_back(Plotter::Scanner(lepType.first, vars, {dsTTZH, dsDY, dsZJ, dsWJ, dsTTHad, dsTTLep, dsVV, dsVVV, dsTTXqq, dsQCD, dsMuData, dsEleData}));
	}
    }
    //////// TESTING SOMETHING ADDED BY BRYAN //////////////
    //////// CUTS :: DiLepton Cut ("passDiElecSel","passDiMuSel") , Tagged Top ("nMergedTops_drLeptonCleaned>0")
    //////// Kenimatics: diLeption Mass spectrum (invariant mass). Z(ll) pt spectrum of the two leptons, Resolved top pt, up to 1TeV, n_Tops
    //////// Samples : TTbarSingleLep, TTbarDiLep, WZ, ZZ, ttW, SingleT, ttH, tZq, VVV, DYJetsToLL, ttZ
    typedef std::tuple<std::string,std::string, float, float, bool> Variable;
    typedef std::pair<std::string,std::string>                      StrPair;
    /////// Validation of Selection Cuts ///////
    // Config for plots
    bool testSel    = false;

    bool doZptCut   = true;
 
    bool doJetSel   = false; 
    bool dolooseZ   = false;
    bool doData     = false;
    bool doFakeData = true;
    bool doZqq      = false;

    //
    //bool analyzeGen = false; // if true, dont use Z selection cuts
    ///////

    std::vector<StrPair>  mc_samples = { 
      {"DY",                   "DYJetsToLL"},
      //{"t#bar{t}Had_Z_ll",     "TTZToLLNuNu"},
      //{"t#bar{t}Lep_Z_ll",     "TTZToLLNuNu"},
      //{"t#bar{t}_Z_nunu",      "TTZToLLNuNu"},
      //{"t#bar{t}Z_qq",         "TTZToQQ"},
      {"t#bar{t}Z_pass",         "TTZ"},
      {"t#bar{t}Z_fail",         "TTZ"},
      //{"t#bar{t}W",            "TTW"},
      //{"tZq",                  "ST_tZq_ll"},
      {"VVV",                  "Triboson"},
      //{"t#bar{t}H",            "ttH"},
      //{"t#bar{t}t#bar{t}",     "TTTT"},
      //{"WZ",                   "WZ_amcatnlo"},
      {"Diboson",              "Diboson"}, //    bck
      {"t(#bar{t})X",          "TTX"},     //    bck
      //{"t",                    "SingleTop"},
      //{"t#bar{t}_2l",          "TTbarDiLep"},
      //{"t#bar{t}_1l",          "TTbarSingleLep"}
      {"t#bar{t}",             "TTbar"}};
      
    std::vector<Variable> kenimatic_vars = { // Usage: label, variable, range, logScale? 
      {"Z_ll_mass", "bestRecoZM",                                    60.0,  120.0,  false},
      {"Z_ll_pt",   "bestRecoZPt",                   doZptCut? 300.0 : 0.0,   1000.0, false},
      //{"Z_qq_mass", "bestRecoZqqM",                                    60.0,  120.0,  false},
      //{"Z_qq_pt",   "bestRecoZqqPt",                                  200.0,   1000.0, true},
      //{"first_GenTopPt",     "genTops(pt)[0]",                           0.0, 1000.0,  true},
      //{"second_GenTopPt",    "genTops(pt)[1]",                           0.0, 1000.0,  true},
      //{"first_GenWPt",       "genWs(pt)[0]",                             0.0, 1000.0,  true},
      //{"second_GenWPt",       "genWs(pt)[1]",                            0.0, 1000.0,  true}};
      //{"NWs",       "nWs_drLeptonCleaned",                           0.0,   5.0   , true}};
      //{"NWs_qcd",       "nWs_qcd",                           0.0,   5.0   , true},
      //{"NWs_tau",       "nWs_tau",                           0.0,   5.0   , true},
      //{"NZs_qcd",       "nZs_qcd",                           0.0,   5.0   , true},
      //{"NZs_tau",       "nZs_tau",                           0.0,   5.0   , true},
      {"TopDisc",     "ResolvedTopsDisc",                       0.7,   1.0,   false},
      {"nRt_ttz",       "nRt_ttz",                              0.0,   7.0,   true}};
      //{"TopPt",     "ResolvedTopCandidate_pt[0]",                       0.0,   1000.0, !doZptCut},
      //{"TopM",      "ResolvedTopCandidate_mass[0]",                     123.0, 223.0,  !doZptCut},
      //{"GenW",      "genWeight",                                      -1.5,   1.5,    true},
      //{"TopM_in",   "ResolvedTopCandidate_mass[0]{ResolvedTopCandidate_mass[0]>160;ResolvedTopCandidate_mass[0]<190}",
      // 155.0,   195.0,  !doZptCut},
      //{"TopM_out",  "ResolvedTopCandidate_mass[0]{ResolvedTopCandidate_mass[0]<160;ResolvedTopCandidate_mass[0]>190}",
      //0.0, 400.0,  !doZptCut},
      //{"nRT",       "nResolvedTops_drLeptonCleaned",                    0.0,   7.0,    !doZptCut},
      //{"NJets",                  "nJets",                               0.0,   10.0,    false},
      //{"NJets_LepCleaned",       "nJets_drLeptonCleaned",               0.0,   10.0,    false},
      //{"NJets30",                "nJets30",                             0.0,   10.0,    false},
      //{"NJets30_LepCleaned",     "nJets30_drLeptonCleaned",             0.0,   10.0,    false},
      //{"Bottom_dR", "b_dR_drLeptonCleaned",                             0.0,   5.0,    !doZptCut}};
    
    if (testSel) {                                                                 // validate selections
      kenimatic_vars.push_back({"nRTops",    "nResolvedTops_drLeptonCleaned", 0.0, 10.0,   true});
      kenimatic_vars.push_back({"nMTops",    "nMergedTops_drLeptonCleaned",   0.0, 10.0,   true});
      kenimatic_vars.push_back({"nBot",      "nBottoms_drLeptonCleaned",      0.0, 10.0,   true});
    }
    std::string          Zpt_selection  = "bestRecoZPt>300";
    std::vector<StrPair> bot_selections = {
      //{"nb0",  "nBottoms_drLeptonCleaned=0"},//{"nb1","nb>=1"},
      //{"nb1", "nBottoms_drLeptonCleaned=1"},
      {"nb2", "nBottoms_drLeptonCleaned=2"},
      {"nbg1", "nBottoms_drLeptonCleaned>1"},
      {"","NONE"}};//{"nb2","nb>=2"}};
    if (testSel){  // validate selections
      bot_selections.push_back({"nbge2","nBottoms_drLeptonCleaned>=2"}); // really >= 1
    }
    std::vector<StrPair> topR_selections = {
      //{"nRt0",  "nResolvedTops_drLeptonCleaned=0"},
      //{"nRt1", "nResolvedTops_drLeptonCleaned=1"},
      {"nRtg0", "nResolvedTops_drLeptonCleaned>=1"},
      //{"nRt2", "nResolvedTops_drLeptonCleaned=2"},
      //{"nRtg2", "nResolvedTops_drLeptonCleaned>2"},
      //{"nRt0",  "nRt_ttz=0"},
      //{"nRt1", "nRt_ttz=1"},
      //{"nRt2", "nRt_ttz=2"},
      //{"nRtg2", "nRt_ttz>2"},
      {"",      "NONE"}};
    std::vector<StrPair> topM_selections = {
      //{"nMt0",  "nMergedTops_drLeptonCleaned=0"},
      //{"nMtg0", "nMergedTops_drLeptonCleaned>0"},
      //{"nMtg1", "nMergedTops_drLeptonCleaned>1"},
      //{"nZq0",     "nZs_qcd=0"},
      //{"nZq1",     "nZs_qcd=1"},
      //{"nZqg1",    "nZs_qcd>1"},
      //{"nZt0",     "nZs_tau=0"},
      //{"nZt1",     "nZs_tau=1"},
      //{"nZtg1",    "nZs_tau>1"},
      {""     , "NONE"}};
    std::vector<StrPair> topRM_selections = {
      //{"nRMtl3","(nResolvedTops_drLeptonCleaned+nMergedTops_drLeptonCleaned)<3"},
      {"","NONE"}};
    std::vector<StrPair> nJet_selections = {
      {"nJ23", "nJets30_drLeptonCleaned>=2;nJets30_drLeptonCleaned<=3"},
      {"nJ45", "nJets30_drLeptonCleaned>=4;nJets30_drLeptonCleaned<=5"},
      {"nJ6" , "nJets30_drLeptonCleaned>=6"},
      {""    , "NONE"}};
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
    // Baseline Cuts
    std::string ttz_elec_cuts      = "passElecZinvSelOnZMassPeak";//"passDiElecSel";
    std::string ttz_mu_cuts        = "passMuZinvSelOnZMassPeak";//"passDiMuSel";
    if (doZqq){
      ttz_elec_cuts      = "passElecZinvSelOffZMassPeak";
      ttz_mu_cuts        = "passMuZinvSelOffZMassPeak";
    }
    std::string ttz_elmu_cuts      = "passElMuZinvSelOffZMassPeak";
    std::string ttz_elec_diLepCuts = "passDiElecSel";
    std::string ttz_mu_diLepCuts   = "passDiMuSel";
    //if (analyzeGen){
    //  ttz_elec_cuts      = "";
    //  ttz_mu_cuts        = "";
    //  ttz_elec_diLepCuts = "";
    //  ttz_mu_diLepCuts   = "";
    //}
    // Baseline Weights
    std::string weight             = "genWeight"; 
    //
    std::vector<std::vector<PDS>> dsTTZ_elec_stack ;
    std::vector<std::vector<PDS>> dsTTZ_mu_stack ;
    std::vector<std::vector<PDS>> dsTTZ_elmu_stack ;
    std::vector<std::vector<PDS>> dsTTZ_elec_loose_stack ;
    std::vector<std::vector<PDS>> dsTTZ_mu_loose_stack ;
    std::vector<std::vector<PDS>> TTZ_stack;
    std::vector<std::vector<PDS>> TTZ_noGenW_stack;
    //
    for ( StrPair& sample : mc_samples){
      if      (sample.first == "t#bar{t}Had_Z_ll"){
	dsTTZ_elec_stack.push_back({PDS( sample.first, fileMap[sample.second + yearTag], ttz_elec_cuts+";isZToLL;isTAllHad", weight)});
	dsTTZ_mu_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_mu_cuts+";isZToLL;isTAllHad",   weight)});
	if (doZqq) dsTTZ_elmu_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_elmu_cuts+";isZToLL;isTAllHad",   weight)});
	if (dolooseZ){
	  dsTTZ_elec_loose_stack.push_back({PDS( sample.first, fileMap[sample.second + yearTag], ttz_elec_diLepCuts+";isZToLL;isTAllHad", weight)});
	  dsTTZ_mu_loose_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_mu_diLepCuts+";isZToLL;isTAllHad", weight)}); 
	}
	//
	TTZ_stack.push_back({PDS(        sample.first, fileMap[sample.second + yearTag], "isZToLL;isTAllHad",                weight)});
	TTZ_noGenW_stack.push_back({PDS(        sample.first, fileMap[sample.second + yearTag], "isZToLL;isTAllHad",                "")});
      }
      else if (sample.first == "t#bar{t}Lep_Z_ll"){
	dsTTZ_elec_stack.push_back({PDS( sample.first, fileMap[sample.second + yearTag], ttz_elec_cuts+";isZToLL;!isTAllHad", weight)});
	dsTTZ_mu_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_mu_cuts+";isZToLL;!isTAllHad",   weight)});
	if (doZqq) dsTTZ_elmu_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_elmu_cuts+";isZToLL;!isTAllHad",   weight)});
	if (dolooseZ){
	  dsTTZ_elec_loose_stack.push_back({PDS( sample.first, fileMap[sample.second + yearTag], ttz_elec_diLepCuts+";isZToLL;!isTAllHad", weight)});
	  dsTTZ_mu_loose_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_mu_diLepCuts+";isZToLL;!isTAllHad", weight)}); 
	}
	//
	TTZ_stack.push_back({PDS(        sample.first, fileMap[sample.second + yearTag], "isZToLL;!isTAllHad",                weight)});
	TTZ_noGenW_stack.push_back({PDS(        sample.first, fileMap[sample.second + yearTag], "isZToLL;!isTAllHad",                "")});
      }
      else if (sample.first == "t#bar{t}_Z_nunu"){
	dsTTZ_elec_stack.push_back({PDS( sample.first, fileMap[sample.second + yearTag], ttz_elec_cuts+";!isZToLL", weight)});
	dsTTZ_mu_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_mu_cuts+";!isZToLL",   weight)});
	if (doZqq) dsTTZ_elmu_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_elmu_cuts+";!isZToLL",   weight)});
	if (dolooseZ){
	  dsTTZ_elec_loose_stack.push_back({PDS( sample.first, fileMap[sample.second + yearTag], ttz_elec_diLepCuts+";!isZToLL", weight)});
	  dsTTZ_mu_loose_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_mu_diLepCuts+";!isZToLL", weight)});
	}
	//
	TTZ_stack.push_back({PDS(        sample.first, fileMap[sample.second + yearTag], "!isZToLL",                weight)});
	TTZ_noGenW_stack.push_back({PDS(        sample.first, fileMap[sample.second + yearTag], "!isZToLL",                "")});
      }
      //else if (sample.first == "t#bar{t}Lep_Z_nunu"){
      //	dsTTZ_elec_stack.push_back({PDS( sample.first, fileMap[sample.second + yearTag], ttz_elec_cuts+";!isZToLL;!isTAllHad", weight)});
      //	dsTTZ_mu_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_mu_cuts+";!isZToLL;!isTAllHad",   weight)});
      //	TTZ_stack.push_back({PDS(        sample.first, fileMap[sample.second + yearTag], "!isZToLL;!isTAllHad",                weight)});
      //}
      else if (sample.first == "t#bar{t}Z_pass"){
 	dsTTZ_elec_stack.push_back({PDS( sample.first, fileMap[sample.second + yearTag], ttz_elec_cuts+";passGenCutsEE", weight)});
	dsTTZ_mu_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_mu_cuts+";passGenCutsMuMu", weight)});
	//
	TTZ_stack.push_back({PDS(        sample.first, fileMap[sample.second + yearTag], "passGenCuts",                weight)});
	TTZ_noGenW_stack.push_back({PDS(        sample.first, fileMap[sample.second + yearTag], "passGenCuts",             "")});
      }
      else if (sample.first == "t#bar{t}Z_fail"){
	dsTTZ_elec_stack.push_back({PDS( sample.first, fileMap[sample.second + yearTag], ttz_elec_cuts+";!passGenCutsEE", weight)});
	dsTTZ_mu_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_mu_cuts+";!passGenCutsMuMu", weight)});
	//
	TTZ_stack.push_back({PDS(        sample.first, fileMap[sample.second + yearTag], "!passGenCuts",                weight)});
	TTZ_noGenW_stack.push_back({PDS(        sample.first, fileMap[sample.second + yearTag], "!passGenCuts",             "")});
      }
      else{
	dsTTZ_elec_stack.push_back({PDS( sample.first, fileMap[sample.second + yearTag], ttz_elec_cuts, weight)});
	dsTTZ_mu_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_mu_cuts, weight)});
	if (doZqq) dsTTZ_elmu_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_elmu_cuts, weight)});
	if (dolooseZ){
	  dsTTZ_elec_loose_stack.push_back({PDS( sample.first, fileMap[sample.second + yearTag], ttz_elec_diLepCuts, weight)});
	  dsTTZ_mu_loose_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], ttz_mu_diLepCuts, weight)}); 
	}
	//
	TTZ_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], "", weight)});
	TTZ_noGenW_stack.push_back({PDS(   sample.first, fileMap[sample.second + yearTag], "", "")});
      }
    }
    PDS dsElectronData("Data", fileMap[ElectronDataset],  ttz_elec_cuts, "");
    PDS dsMuonData(    "Data", fileMap[MuonDataset],      ttz_mu_cuts, "");
    if (doFakeData){
      PDS dsElectronData("Data", fileMap["TTZ"+yearTag],    ttz_elec_cuts+";passGenCuts", weight);
      PDS dsMuonData(    "Data", fileMap["TTZ"+yearTag],      ttz_mu_cuts+";passGenCuts", weight);
    }
    // === PDC === //
    std::vector<PDC> dsMC_DiLep_elec;
    std::vector<PDC> dsMC_DiLep_mu;
    std::vector<PDC> dsMC_DiLep_elmu;
    std::vector<PDC> dsMC_DiLepLoose_elec;
    std::vector<PDC> dsMC_DiLepLoose_mu;
    std::vector<PDC> dsData_Elec;
    std::vector<PDC> dsData_Mu;
    std::vector<PDC> dsMC_TTZ;
    std::vector<PDC> dsMC_TTZ_noGenW;
    //
    for ( Variable& var : kenimatic_vars ){
      dsMC_DiLep_elec.push_back(PDC(     "stack", std::get<1>(var), dsTTZ_elec_stack));
      dsMC_DiLep_mu.push_back(PDC(       "stack", std::get<1>(var), dsTTZ_mu_stack));
      if (doZqq) dsMC_DiLep_elmu.push_back(PDC(       "stack", std::get<1>(var), dsTTZ_elmu_stack));
      //
      //dsMC_DiLepLoose_elec.push_back(PDC("stack", std::get<1>(var), dsTTZ_elec_loose_stack));
      //dsMC_DiLepLoose_mu.push_back(PDC(  "stack", std::get<1>(var), dsTTZ_mu_loose_stack));
      //
      dsData_Elec.push_back(PDC(         "data",  std::get<1>(var), {dsElectronData}));
      dsData_Mu.push_back(PDC(           "data",  std::get<1>(var), {dsMuonData}));
    }
    if (dolooseZ){
      dsMC_DiLepLoose_elec.push_back(PDC("stack", "bestRecoZM", dsTTZ_elec_loose_stack));
      dsMC_DiLepLoose_mu.push_back(PDC(  "stack", "bestRecoZM", dsTTZ_mu_loose_stack));
    }
    // test
    dsMC_TTZ.push_back(PDC(           "stack", "genWeight", TTZ_stack));
    vh.push_back(PHS("MC_TTZ_GenW"+eraTag,        {dsMC_TTZ[0]},        {1,1}, "", 
		     60, -3.0, 3.0, true, false, "GenW", "Events"));
    dsMC_TTZ_noGenW.push_back(PDC(           "stack", "genWeight", TTZ_noGenW_stack));
    vh.push_back(PHS("MC_TTZ_noGenW_GenW"+eraTag, {dsMC_TTZ_noGenW[0]}, {1,1}, "", 
		     60, -3.0, 3.0, true, false, "GenW", "Events"));
    // end test
    if (doJetSel) top_selections = nJet_selections; // super duper hacky (rewrite this Bryan!!!)
    // === PHS === //
    for(int i = 0 ; i < dsMC_DiLep_elec.size() ; i++){
      for(StrPair& b_sel : bot_selections){
	for(StrPair& top_sel : top_selections){
	  std::string sel_label, sel_cut;
	  bool addData = false; // for viability tests, only add data for very loose selection requirements
	  if (doFakeData) addData = true;
	  if (b_sel.second == "NONE" && top_sel.second == "NONE"){
	    sel_label = "";
	    sel_cut   = "";
	    if (doData) addData = true;
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
	  }
	  if (doZptCut){
	    if (sel_cut == "" || sel_cut == ";") sel_cut = Zpt_selection;
	    else sel_cut = sel_cut+";"+Zpt_selection;
	  }
	  //printf("SELCUT:\t%s\n",sel_cut.c_str());
	  ////
	  if (addData){
	    vh.push_back(PHS("MCData_DiLep_elec_"+std::get<0>(kenimatic_vars[i])+sel_label+eraTag,            {dsData_Elec[i], dsMC_DiLep_elec[i]},      {1,1}, sel_cut, 
			     30, std::get<2>(kenimatic_vars[i]), std::get<3>(kenimatic_vars[i]), std::get<4>(kenimatic_vars[i]), false, std::get<0>(kenimatic_vars[i]), "Events"));
	    vh.push_back(PHS("MCData_DiLep_mu_"+std::get<0>(kenimatic_vars[i])+sel_label+eraTag,              {dsData_Mu[i], dsMC_DiLep_mu[i]},          {1,1}, sel_cut, 
			     30, std::get<2>(kenimatic_vars[i]), std::get<3>(kenimatic_vars[i]), std::get<4>(kenimatic_vars[i]), false, std::get<0>(kenimatic_vars[i]), "Events"));
	    if (dolooseZ){
	      vh.push_back(PHS("MCData_DiLep_NoZMassCut_elec_"+std::get<0>(kenimatic_vars[0])+sel_label+eraTag, {dsData_Elec[i], dsMC_DiLepLoose_elec[0]}, {1,1}, sel_cut, 
			       30, std::get<2>(kenimatic_vars[0]), std::get<3>(kenimatic_vars[0]), std::get<4>(kenimatic_vars[0]), false, std::get<0>(kenimatic_vars[0]), "Events"));
	      vh.push_back(PHS("MCData_DiLep_NoZMassCut_mu_"+std::get<0>(kenimatic_vars[0])+sel_label+eraTag,   {dsData_Mu[i],   dsMC_DiLepLoose_mu[0]},   {1,1}, sel_cut, 
			       30, std::get<2>(kenimatic_vars[0]), std::get<3>(kenimatic_vars[0]), std::get<4>(kenimatic_vars[0]), false, std::get<0>(kenimatic_vars[0]), "Events"));
	    }
	  }
	  else{
	    vh.push_back(PHS("MC_DiLep_elec_"+std::get<0>(kenimatic_vars[i])+sel_label+eraTag,            {dsMC_DiLep_elec[i]},      {1,1}, sel_cut, 
			     30, std::get<2>(kenimatic_vars[i]), std::get<3>(kenimatic_vars[i]), std::get<4>(kenimatic_vars[i]), false, std::get<0>(kenimatic_vars[i]), "Events"));
	    vh.push_back(PHS("MC_DiLep_mu_"+std::get<0>(kenimatic_vars[i])+sel_label+eraTag,              {dsMC_DiLep_mu[i]},        {1,1}, sel_cut, 
			     30, std::get<2>(kenimatic_vars[i]), std::get<3>(kenimatic_vars[i]), std::get<4>(kenimatic_vars[i]), false, std::get<0>(kenimatic_vars[i]), "Events"));
	    if (doZqq) vh.push_back(PHS("MC_DiLep_elmu_"+std::get<0>(kenimatic_vars[i])+sel_label+eraTag,              {dsMC_DiLep_elmu[i]},        {1,1}, sel_cut, 
					30, std::get<2>(kenimatic_vars[i]), std::get<3>(kenimatic_vars[i]), std::get<4>(kenimatic_vars[i]), false, std::get<0>(kenimatic_vars[i]), "Events"));
	    if (dolooseZ){
	      vh.push_back(PHS("MC_DiLep_NoZMassCut_elec_"+std::get<0>(kenimatic_vars[0])+sel_label+eraTag, {dsMC_DiLepLoose_elec[0]}, {1,1}, sel_cut, 
			       30, std::get<2>(kenimatic_vars[0]), std::get<3>(kenimatic_vars[0]), std::get<4>(kenimatic_vars[0]), false, std::get<0>(kenimatic_vars[0]), "Events"));
	      vh.push_back(PHS("MC_DiLep_NoZMassCut_mu_"+std::get<0>(kenimatic_vars[0])+sel_label+eraTag,   {dsMC_DiLepLoose_mu[0]},   {1,1}, sel_cut, 
			       30, std::get<2>(kenimatic_vars[0]), std::get<3>(kenimatic_vars[0]), std::get<4>(kenimatic_vars[0]), false, std::get<0>(kenimatic_vars[0]), "Events"));
	    }
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
    plotter.setScanners(scanners);
    plotter.setLumi(lumi);
    plotter.setPlotDir(plotDir);
        //plotter.setDoHists(doSave || doPlots);
    plotter.setDoTuple(doTuple);
    plotter.setRegisterFunction(rf);
    plotter.read();
        //if(doSave && fromTuple)  plotter.saveHists();
        //if(doPlots)              plotter.plot();
}
