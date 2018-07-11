#include "Plotter.h"
#include "RegisterFunctions.h"
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
	{"luminosity", required_argument, 0, 'L'}
    };

    bool doPlots = true, doSave = true, doTuple = true, fromTuple = true, runOnCondor = false;
    string histFile = "", dataSets = "", sampleloc = "/uscms/home/pastika/nobackup/zinv/dev/CMSSW_7_4_8/src/ZInvisible/Tools/condor/", plotDir = "Mini_plots";
    int nFiles = -1, startFile = 0, nEvts = -1;
    float lumi = AnaSamples::luminosity;

    while((opt = getopt_long(argc, argv, "pstfcH:D:N:M:E:P:L:", long_options, &option_index)) != -1)
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

    AnaSamples::SampleSet        ss(sampleloc, lumi);
    AnaSamples::SampleCollection sc(ss);

    const float zAcc = 1.0;
//    const float zAcc = 0.5954;
//    const float zAcc = 0.855;
    const float znunu_mumu_ratio = 5.942;
    const float znunu_ee_ratio   = 5.942;

    map<string, vector<AnaSamples::FileSummary>> fileMap;

    //Select approperiate datasets here
    if(dataSets.compare("TEST") == 0)
    {
        fileMap["DYJetsToLL"]  = {ss["DYJetsToLL_HT_600toInf"]};
        //fileMap["ZJetsToNuNu"] = {ss["ZJetsToNuNu_HT_600toInf"]};
        fileMap["DYJetsToLL_HT_600toInf"] = {ss["DYJetsToLL_HT_600toInf"]};
        //fileMap["ZJetsToNuNu_HT_600toInf"] = {ss["ZJetsToNuNu_HT_600toInf"]};
        fileMap["TTbarDiLep"] = {ss["TTbarDiLep"]};
    }
    if(dataSets.compare("ALL") == 0)
    {
        fileMap["Data_SingleMuon"] = sc["Data_SingleMuon"];
        fileMap["Data_DoubleEG"]   = sc["Data_DoubleEG"];
        fileMap["DYJetsToLL"]      = sc["DYJetsToLL"];
        fileMap["IncDY"]           = sc["IncDY"];
        fileMap["TTbarNoHad"]      = sc["TTbarNoHad"];
        fileMap["tW"]              = sc["tW"];
        fileMap["TTZ"]             = sc["TTZ"];
        fileMap["Diboson"]         = sc["Diboson"];
        fileMap["Rare"]            = sc["Rare"];
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


    // Shortcuts for axis labels
    std::string label_met = "p_{T}^{miss} [GeV]";
    std::string label_ht  = "H_{T} [GeV]";
    std::string label_mht = "MH_{T} [GeV]";
    std::string label_nj  = "N_{jets}";
    std::string label_nb  = "N_{b-jets}";
    std::string label_nt  = "N_{tops}";
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

    vector<Plotter::HistSummary> vh;

    // -----------------
    // - Data/MC plots -
    // -----------------

    // Datasetsummaries we are using
    // no weight (genWeight deals with negative weights)
    Plotter::DatasetSummary dsData_SingleMuon("Data",       fileMap["Data_SingleMuon"], "passMuTrigger",   "");
    Plotter::DatasetSummary dsData_DoubleEG(  "Data",       fileMap["Data_DoubleEG"],   "passElecTrigger", "");
    Plotter::DatasetSummary dsDY(             "DY",         fileMap["DYJetsToLL"],      "",                "");
    Plotter::DatasetSummary dsDYInc(          "DY HT<100",  fileMap["IncDY"],           "genHT<100",       "");
    Plotter::DatasetSummary dstt2l(           "t#bar{t}",   fileMap["TTbarNoHad"],      "",                "");
    Plotter::DatasetSummary dstW(             "single top", fileMap["tW"],              "",                "");
    //Plotter::DatasetSummary dsttZ(            "t#bar{t}Z",  fileMap["TTZ"],             "",                "genWeight");
    Plotter::DatasetSummary dsttZ(            "t#bar{t}Z",  fileMap["TTZ"],             "",                "");
    Plotter::DatasetSummary dsVV(             "Diboson",    fileMap["Diboson"],         "",                "");
    //Plotter::DatasetSummary dsRare(           "Rare",       fileMap["Rare"],            "",                "genWeight");
    Plotter::DatasetSummary dsRare(           "Rare",       fileMap["Rare"],            "",                "");
    std::vector<Plotter::DatasetSummary> stack_MC = {dsDY, dsDYInc, dstt2l, dstW, dsttZ, dsVV, dsRare};
    // eff*Acc*B(Z)/B(DY)
    Plotter::DatasetSummary dsData_SingleMuon_effAcc("Data",       fileMap["Data_SingleMuon"], "passMuTrigger",   "zEffWgt;zAccWgt",           znunu_mumu_ratio);
    Plotter::DatasetSummary dsData_DoubleEG_effAcc(  "Data",       fileMap["Data_DoubleEG"],   "passElecTrigger", "zEffWgt;zAccWgt",           znunu_mumu_ratio);
    Plotter::DatasetSummary dsDY_effAcc(             "DY",         fileMap["DYJetsToLL"],      "",                "zEffWgt;zAccWgt",           znunu_mumu_ratio);
    Plotter::DatasetSummary dsDYInc_effAcc(          "DY HT<100",  fileMap["IncDY"],           "genHT<100",       "zEffWgt;zAccWgt",           znunu_mumu_ratio);
    Plotter::DatasetSummary dstt2l_effAcc(           "t#bar{t}",   fileMap["TTbarNoHad"],      "",                "zEffWgt;zAccWgt",           znunu_mumu_ratio);
    Plotter::DatasetSummary dstW_effAcc(             "single top", fileMap["tW"],              "",                "zEffWgt;zAccWgt",           znunu_mumu_ratio);
    //Plotter::DatasetSummary dsttZ_effAcc(            "t#bar{t}Z",  fileMap["TTZ"],             "",                "zEffWgt;zAccWgt;genWeight", znunu_mumu_ratio);
    Plotter::DatasetSummary dsttZ_effAcc(            "t#bar{t}Z",  fileMap["TTZ"],             "",                "zEffWgt;zAccWgt;", znunu_mumu_ratio);
    Plotter::DatasetSummary dsVV_effAcc(             "Diboson",    fileMap["Diboson"],         "",                "zEffWgt;zAccWgt",           znunu_mumu_ratio);
    //Plotter::DatasetSummary dsRare_effAcc(           "Rare",       fileMap["Rare"],            "",                "zEffWgt;zAccWgt;genWeight", znunu_mumu_ratio);
    Plotter::DatasetSummary dsRare_effAcc(           "Rare",       fileMap["Rare"],            "",                "zEffWgt;zAccWgt", znunu_mumu_ratio);
    std::vector<Plotter::DatasetSummary> stack_MC_effAcc = {dsDY_effAcc, dsDYInc_effAcc, dstt2l_effAcc, dstW_effAcc, dsttZ_effAcc, dsVV_effAcc, dsRare_effAcc};

    // Apply data/mc njet weight for DY and ttbar
    Plotter::DatasetSummary dswDY(             "DY",         fileMap["DYJetsToLL"],      "",            "nJetWgtDYZ");
    Plotter::DatasetSummary dswDYInc(          "DY HT<100",  fileMap["IncDY"],           "genHT<100",   "nJetWgtDYZ");
    Plotter::DatasetSummary dswtt2l(           "t#bar{t}",   fileMap["TTbarNoHad"],      "",            "nJetWgtTTbar");
    Plotter::DatasetSummary dswtW(             "single top", fileMap["tW"],              "",            "");
    //Plotter::DatasetSummary dswttZ(            "t#bar{t}Z",  fileMap["TTZ"],             "",            "genWeight");
    Plotter::DatasetSummary dswttZ(            "t#bar{t}Z",  fileMap["TTZ"],             "",            "");
    Plotter::DatasetSummary dswVV(             "Diboson",    fileMap["Diboson"],         "",            "");
    //Plotter::DatasetSummary dswRare(           "Rare",       fileMap["Rare"],            "",            "genWeight");
    Plotter::DatasetSummary dswRare(           "Rare",       fileMap["Rare"],            "",            "");
    std::vector<Plotter::DatasetSummary> stackw_MC = {dswDY, dswDYInc, dswtt2l, dswtW, dswttZ, dswVV, dswRare};

    // Extra MC stack that have njet weight applied for ttbar but not for DY (to make plots for AN)
    std::vector<Plotter::DatasetSummary> stackwtt_MC = {dsDY, dsDYInc, dswtt2l, dstW, dsttZ, dsVV, dsRare};

    // Apply data/mc njet weight for DY and ttbar & apply the normalization weight
    Plotter::DatasetSummary dswwDY(             "DY",         fileMap["DYJetsToLL"],      "",            "nJetWgtDYZ;normWgt0b");
    Plotter::DatasetSummary dswwDYInc(          "DY HT<100",  fileMap["IncDY"],           "genHT<100",   "nJetWgtDYZ;normWgt0b");
    std::vector<Plotter::DatasetSummary> stackww_MC = {dswwDY, dswwDYInc, dswtt2l, dswtW, dswttZ, dswVV, dswRare};


    // Collections for all variables, no cuts applied yet
    // met
    Plotter::DataCollection dcData_SingleMuon_met("data",   "cleanMetPt", {dsData_SingleMuon});
    Plotter::DataCollection dcData_DoubleEG_met(  "data",   "cleanMetPt", {dsData_DoubleEG});
    Plotter::DataCollection dcMC_met(             "stack",  "cleanMetPt", stack_MC);
    Plotter::DataCollection dcwMC_met(            "stack",  "cleanMetPt", stackw_MC);
    Plotter::DataCollection dcwttMC_met(          "stack",  "cleanMetPt", stackwtt_MC);
    Plotter::DataCollection dcwwMC_met(           "stack",  "cleanMetPt", stackww_MC);
    // ntops
    Plotter::DataCollection dcData_SingleMuon_nt("data",   "nTopCandSortedCntZinv", {dsData_SingleMuon});
    Plotter::DataCollection dcData_DoubleEG_nt(  "data",   "nTopCandSortedCntZinv", {dsData_DoubleEG});
    Plotter::DataCollection dcMC_nt(             "stack",  "nTopCandSortedCntZinv", stack_MC);
    Plotter::DataCollection dcwMC_nt(            "stack",  "nTopCandSortedCntZinv", stackw_MC);
    Plotter::DataCollection dcwttMC_nt(          "stack",  "nTopCandSortedCntZinv", stackwtt_MC);
    Plotter::DataCollection dcwwMC_nt(           "stack",  "nTopCandSortedCntZinv", stackww_MC);
    // MT2
    Plotter::DataCollection dcData_SingleMuon_mt2("data",   "best_had_brJet_MT2Zinv", {dsData_SingleMuon});
    Plotter::DataCollection dcData_DoubleEG_mt2(  "data",   "best_had_brJet_MT2Zinv", {dsData_DoubleEG});
    Plotter::DataCollection dcMC_mt2(             "stack",  "best_had_brJet_MT2Zinv", stack_MC);
    Plotter::DataCollection dcwMC_mt2(            "stack",  "best_had_brJet_MT2Zinv", stackw_MC);
    Plotter::DataCollection dcwttMC_mt2(          "stack",  "best_had_brJet_MT2Zinv", stackwtt_MC);
    Plotter::DataCollection dcwwMC_mt2(           "stack",  "best_had_brJet_MT2Zinv", stackww_MC);
    // nb
    Plotter::DataCollection dcData_SingleMuon_nb("data",   "cntCSVSZinv", {dsData_SingleMuon});
    Plotter::DataCollection dcData_DoubleEG_nb(  "data",   "cntCSVSZinv", {dsData_DoubleEG});
    Plotter::DataCollection dcMC_nb(             "stack",  "cntCSVSZinv", stack_MC);
    Plotter::DataCollection dcwMC_nb(            "stack",  "cntCSVSZinv", stackw_MC);
    Plotter::DataCollection dcwttMC_nb(          "stack",  "cntCSVSZinv", stackwtt_MC);
    Plotter::DataCollection dcwwMC_nb(           "stack",  "cntCSVSZinv", stackww_MC);
    // nj
    Plotter::DataCollection dcData_SingleMuon_nj("data",   "cntNJetsPt30Eta24Zinv", {dsData_SingleMuon});
    Plotter::DataCollection dcData_DoubleEG_nj(  "data",   "cntNJetsPt30Eta24Zinv", {dsData_DoubleEG});
    Plotter::DataCollection dcMC_nj(             "stack",  "cntNJetsPt30Eta24Zinv", stack_MC);
    Plotter::DataCollection dcwMC_nj(            "stack",  "cntNJetsPt30Eta24Zinv", stackw_MC);
    Plotter::DataCollection dcwttMC_nj(          "stack",  "cntNJetsPt30Eta24Zinv", stackwtt_MC);
    Plotter::DataCollection dcwwMC_nj(           "stack",  "cntNJetsPt30Eta24Zinv", stackww_MC);
    // ht
    Plotter::DataCollection dcData_SingleMuon_ht("data",   "HTZinv", {dsData_SingleMuon});
    Plotter::DataCollection dcData_DoubleEG_ht(  "data",   "HTZinv", {dsData_DoubleEG});
    Plotter::DataCollection dcMC_ht(             "stack",  "HTZinv", stack_MC);
    Plotter::DataCollection dcwMC_ht(            "stack",  "HTZinv", stackw_MC);
    Plotter::DataCollection dcwttMC_ht(          "stack",  "HTZinv", stackwtt_MC);
    Plotter::DataCollection dcwwMC_ht(           "stack",  "HTZinv", stackww_MC);
    // mht
    Plotter::DataCollection dcData_SingleMuon_mht("data",   "cleanMHt", {dsData_SingleMuon});
    Plotter::DataCollection dcData_DoubleEG_mht(  "data",   "cleanMHt", {dsData_DoubleEG});
    Plotter::DataCollection dcMC_mht(             "stack",  "cleanMHt", stack_MC);
    Plotter::DataCollection dcwMC_mht(            "stack",  "cleanMHt", stackw_MC);
    Plotter::DataCollection dcwttMC_mht(          "stack",  "cleanMHt", stackwtt_MC);
    Plotter::DataCollection dcwwMC_mht(           "stack",  "cleanMHt", stackww_MC);
    // nsearchbins
    Plotter::DataCollection dcData_SingleMuon_nSearchBin("data",   "nSearchBin", {dsData_SingleMuon});
    Plotter::DataCollection dcData_DoubleEG_nSearchBin(  "data",   "nSearchBin", {dsData_DoubleEG});
    Plotter::DataCollection dcMC_nSearchBin(             "stack",  "nSearchBin", stack_MC);
    Plotter::DataCollection dcwMC_nSearchBin(            "stack",  "nSearchBin", stackw_MC);
    Plotter::DataCollection dcwttMC_nSearchBin(          "stack",  "nSearchBin", stackwtt_MC);
    Plotter::DataCollection dcwwMC_nSearchBin(           "stack",  "nSearchBin", stackww_MC);

    // pair of cutlevels
    std::vector<std::pair<std::string,std::string>> cutlevels_muon = {
	{"muZinv_loose0",             "passNoiseEventFilterZinv;passMuZinvSel;HTZinv>200;passnJetsZinv;passdPhisZinv"},
	{"muZinv_loose0_mt2",         "passNoiseEventFilterZinv;passMuZinvSel;HTZinv>200;passnJetsZinv;passdPhisZinv;best_had_brJet_MT2Zinv>0"},
	//{"muZinv_loose0_ntop",        "passNoiseEventFilterZinv;passMuZinvSel;HTZinv>200;passnJetsZinv;passdPhisZinv;nTopCandSortedCntZinv>0"},
	//{"muZinv_loose0_ht500",       "passNoiseEventFilterZinv;passMuZinvSel;HTZinv>500;passnJetsZinv;passdPhisZinv"},
	{"muZinv_blnotagmt2",         "passMuZinvSel;passBaselineNoTagMT2Zinv"},
	//{"muZinv_bl",                 "passMuZinvSel;passBaselineZinv"},
	//{"muZinv_0b_loose0",          "passNoiseEventFilterZinv;passMuZinvSel;cntCSVSZinv=0;HTZinv>200;passnJetsZinv;passdPhisZinv"},
	//{"muZinv_0b_loose0_ht500",    "passNoiseEventFilterZinv;passMuZinvSel;cntCSVSZinv=0;HTZinv>500;passnJetsZinv;passdPhisZinv"},
	//{"muZinv_0b_blnotagmt2",      "passMuZinvSel;cntCSVSZinv=0;passBaselineNoTagMT2Zinv"},
	//{"muZinv_0b_blnotag",         "passMuZinvSel;cntCSVSZinv=0;passBaselineNoTagZinv"},
	//{"muZinv_g1b_loose0",         "passNoiseEventFilterZinv;passMuZinvSel;passBJetsZinv;HTZinv>200;passnJetsZinv;passdPhisZinv"},
	//{"muZinv_g1b_loose0_ht500",   "passNoiseEventFilterZinv;passMuZinvSel;passBJetsZinv;HTZinv>500;passnJetsZinv;passdPhisZinv"},
	//{"muZinv_g1b_blnotagmt2",     "passMuZinvSel;passBJetsZinv;passBaselineNoTagMT2Zinv"},
	//{"muZinv_g1b_blnotag",        "passMuZinvSel;passBJetsZinv;passBaselineNoTagZinv"},
    };

    // perturbed met variables
    vector<string> metNames = {//"met_sub_10",
                               //"met_sub_20",
                               //"met_sub_30",
                               //"met_sub_40",
                               //"met_sub_50",
                               //"met_sub_60",
                               //
                               //"met_prop_1",
                               //"met_prop_2",
                               //"met_prop_3",
                               //"met_prop_4",
                               //"met_prop_5",
                               //"met_prop_7",
                               //"met_prop_10",
                               //"met_prop_15",
                               //"met_prop_20",
                               //
                               //"met_quad_1_5",
                               //"met_quad_1_10",
                               //"met_quad_1_15",
                               //"met_quad_1_18",
                               //"met_quad_1_20",
                               //"met_quad_1_22",
                               //"met_quad_1_24",
                               //"met_quad_2_5",
                               //"met_quad_2_10",
                               //"met_quad_2_15",
                               //"met_quad_2_18",
                               //"met_quad_2_20",
                               //"met_quad_2_22",
                               //"met_quad_2_24",
                               //"met_quad_3_5",
                               //"met_quad_3_10",
                               //"met_quad_3_15",
                               //"met_quad_3_20",
                               //"met_quad_2_20_ip_300",
                               //"met_quad_2_20_ip_350",
                               //"met_quad_2_20_ip_500",
                               //"met_quad_2_20_ip_450",
                               //
                               //"met_gaus_5",
                               //"met_gaus_10",
                               //"met_gaus_15",
                               //"met_gaus_20",
                               //"met_gaus_25",
                               //"met_gaus_30",
                               //"met_gaus_40",
                               //"met_gaus_50"

                               "met_logi_1",
                               "met_logi_2",
                               "met_logi_3",
                               "met_logi_4",
                               "met_logi_5",
                               "met_logi_6",
                               "met_logi_7",
                               "met_logi_8",
                               "met_logi_9"};

    for(auto& var : metNames)
    {
        Plotter::DatasetSummary dswDY_pmet(      "DY",         fileMap["DYJetsToLL"],      "",            "nJetWgtDYZ");
        Plotter::DatasetSummary dswDYInc_pmet(   "DY HT<100",  fileMap["IncDY"],           "genHT<100",   "nJetWgtDYZ");
        Plotter::DatasetSummary dswtt2l_pmet(    "t#bar{t}",   fileMap["TTbarNoHad"],      "",            "nJetWgtTTbar");
        Plotter::DatasetSummary dswtW_pmet(      "single top", fileMap["tW"],              "",            "");
        //Plotter::DatasetSummary dswttZ_pmet(   "t#bar{t}Z",  fileMap["TTZ"],             "",            "genWeight");
        Plotter::DatasetSummary dswttZ_pmet(     "t#bar{t}Z",  fileMap["TTZ"],             "",            "");
        Plotter::DatasetSummary dswVV_pmet(      "Diboson",    fileMap["Diboson"],         "",            "");
        //Plotter::DatasetSummary dswRare_pmet(  "Rare",       fileMap["Rare"],            "",            "genWeight");
        Plotter::DatasetSummary dswRare_pmet(    "Rare",       fileMap["Rare"],            "",            "");
        std::vector<Plotter::DatasetSummary> stackw_MC_pmet = {dswDY_pmet, dswDYInc_pmet, dswtt2l_pmet, dswtW_pmet, dswttZ_pmet, dswVV_pmet, dswRare_pmet};

        Plotter::DataCollection dcwMC_pmet(            "stack", var, stackw_MC_pmet);

        for(std::pair<std::string,std::string>& cut : cutlevels_muon)
        {
            vh.push_back(PHS("Corrolation_MET-MT2_" + var + "_" + cut.first, {dcData_SingleMuon_met,   dcwMC_pmet}, {1, 2}, cut.second, 50, 0, 1500, true, false,  label_met, ""));
        }
    }

    //push the histograms in a loop, save some copy-paste time
    //for(std::pair<std::string,std::string>& cut : cutlevels_muon)
    //{
    //    // unweighted
    //    vh.push_back(PHS("DataMC_SingleMuon_met_"        +cut.first,  {dcData_SingleMuon_met,   dcMC_met},             {1, 2}, cut.second, 50, 0, 1500, true, false,  label_met,            ""));
    //    vh.push_back(PHS("DataMC_SingleMuon_ht_"         +cut.first,  {dcData_SingleMuon_ht,    dcMC_ht},              {1, 2}, cut.second, 50, 0, 1500, true, false,  label_ht,             ""));
    //    vh.push_back(PHS("DataMC_SingleMuon_mht_"        +cut.first,  {dcData_SingleMuon_mht,   dcMC_mht},             {1, 2}, cut.second, 50, 0, 1500, true, false,  label_mht,            ""));
    //    vh.push_back(PHS("DataMC_SingleMuon_nt_"         +cut.first,  {dcData_SingleMuon_nt,    dcMC_nt},              {1, 2}, cut.second, 5,  0, 5,    true, false,  label_nt,           ""));
    //    vh.push_back(PHS("DataMC_SingleMuon_mt2_"        +cut.first,  {dcData_SingleMuon_mt2,   dcMC_mt2},             {1, 2}, cut.second, 50, 0, 1500, true, false,  label_mt2,            ""));
    //    vh.push_back(PHS("DataMC_SingleMuon_nb_"         +cut.first,  {dcData_SingleMuon_nb,    dcMC_nb},              {1, 2}, cut.second, 10, 0, 10,   true, false,  label_nb,             ""));
    //    vh.push_back(PHS("DataMC_SingleMuon_nj_"         +cut.first,  {dcData_SingleMuon_nj,    dcMC_nj},              {1, 2}, cut.second, 20, 0, 20,   true, false,  label_nj,             ""));
    //    vh.push_back(PHS("DataMC_SingleMuon_nSearchBin_" +cut.first,  {dcData_SingleMuon_nSearchBin, dcMC_nSearchBin}, {1, 2}, cut.second, 45, 0, 45,   true, false,  "Search Bin",     ""));
    //
    //    // DataMC weights applied
    //    vh.push_back(PHS("DataMCw_SingleMuon_met_"   +cut.first,  {dcData_SingleMuon_met,   dcwMC_met},   {1, 2}, cut.second, 50, 0, 1500, true, false,  label_met,            ""));
    //    vh.push_back(PHS("DataMCw_SingleMuon_ht_"    +cut.first,  {dcData_SingleMuon_ht,    dcwMC_ht},    {1, 2}, cut.second, 50, 0, 1500, true, false,  label_ht,             ""));
    //    vh.push_back(PHS("DataMCw_SingleMuon_mht_"   +cut.first,  {dcData_SingleMuon_mht,   dcwMC_mht},   {1, 2}, cut.second, 50, 0, 1500, true, false,  label_mht,            ""));
    //    vh.push_back(PHS("DataMCw_SingleMuon_nt_"    +cut.first,  {dcData_SingleMuon_nt,    dcwMC_nt},    {1, 2}, cut.second, 5,  0, 5,    true, false,  label_nt,           ""));
    //    vh.push_back(PHS("DataMCw_SingleMuon_mt2_"   +cut.first,  {dcData_SingleMuon_mt2,   dcwMC_mt2},   {1, 2}, cut.second, 50, 0, 1500, true, false,  label_mt2,            ""));
    //    vh.push_back(PHS("DataMCw_SingleMuon_nb_"    +cut.first,  {dcData_SingleMuon_nb,    dcwMC_nb},    {1, 2}, cut.second, 10, 0, 10,   true, false,  label_nb,             ""));
    //    vh.push_back(PHS("DataMCw_SingleMuon_nj_"    +cut.first,  {dcData_SingleMuon_nj,    dcwMC_nj},    {1, 2}, cut.second, 20, 0, 20,   true, false,  label_nj,             ""));
    //    vh.push_back(PHS("DataMCw_SingleMuon_nSearchBin_" +cut.first,  {dcData_SingleMuon_nSearchBin, dcwMC_nSearchBin}, {1, 2}, cut.second, 45, 0, 45,  true, false,  "Search Bin",            ""));
    //
    //    // Normalization weight applied, only for blnotag selections
    //    if(cut.first.rfind("blnotag") == (cut.first.size()-7))
    //    {
    //        // DataMC weights applied
    //        vh.push_back(PHS("DataMCww_SingleMuon_met_"   +cut.first,  {dcData_SingleMuon_met,   dcwwMC_met},   {1, 2}, cut.second, 50, 0, 1500, true, false,  label_met,            ""));
    //        vh.push_back(PHS("DataMCww_SingleMuon_ht_"    +cut.first,  {dcData_SingleMuon_ht,    dcwwMC_ht},    {1, 2}, cut.second, 50, 0, 1500, true, false,  label_ht,             ""));
    //        vh.push_back(PHS("DataMCww_SingleMuon_mht_"   +cut.first,  {dcData_SingleMuon_mht,   dcwwMC_mht},   {1, 2}, cut.second, 50, 0, 1500, true, false,  label_mht,            ""));
    //        vh.push_back(PHS("DataMCww_SingleMuon_nt_"    +cut.first,  {dcData_SingleMuon_nt,    dcwwMC_nt},    {1, 2}, cut.second, 5,  0, 5,    true, false,  label_nt,           ""));
    //        vh.push_back(PHS("DataMCww_SingleMuon_mt2_"   +cut.first,  {dcData_SingleMuon_mt2,   dcwwMC_mt2},   {1, 2}, cut.second, 50, 0, 1500, true, false,  label_mt2,            ""));
    //        vh.push_back(PHS("DataMCww_SingleMuon_nb_"    +cut.first,  {dcData_SingleMuon_nb,    dcwwMC_nb},    {1, 2}, cut.second, 10, 0, 10,   true, false,  label_nb,             ""));
    //        vh.push_back(PHS("DataMCww_SingleMuon_nj_"    +cut.first,  {dcData_SingleMuon_nj,    dcwwMC_nj},    {1, 2}, cut.second, 20, 0, 20,   true, false,  label_nj,             ""));
    //        vh.push_back(PHS("DataMCww_SingleMuon_nSearchBin_" +cut.first,  {dcData_SingleMuon_nSearchBin, dcwwMC_nSearchBin}, {1, 2}, cut.second, 45, 0, 45,  true, false,  "Search Bin",            ""));
    //    }
    //}

    set<AnaSamples::FileSummary> vvf;
    for(auto& fsVec : fileMap) for(auto& fs : fsVec.second) vvf.insert(fs);

    RegisterFunctions* rf = new RegisterFunctionsMiniTuple();

    Plotter plotter(vh, vvf, fromTuple, histFile, nFiles, startFile, nEvts);
    plotter.setLumi(lumi);
    plotter.setPlotDir(plotDir);
    plotter.setDoHists(doSave || doPlots);
    plotter.setDoTuple(false);
    plotter.setRegisterFunction(rf);
    plotter.setPrintInterval(10000);
    plotter.read();
    if(doSave && fromTuple)  plotter.saveHists();
    if(doPlots)              plotter.plot();
}
