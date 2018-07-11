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
        //{"savetuple",        no_argument, 0, 't'},
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
    string histFile = "topStudyOutput.root", dataSets = "", sampleloc = AnaSamples::fileDir, plotDir = "top_plots";
    int nFiles = -1, startFile = 0, nEvts = -1;
    float lumi = AnaSamples::luminosity;
    std::string sbEra = "SB_37_2015";

    while((opt = getopt_long(argc, argv, "psfcH:D:N:M:E:P:L:S:", long_options, &option_index)) != -1)
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

//        case 't':
//            if(doTuple) doPlots = doSave = false;
//            else        doTuple  = true;
//            break;

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
        sprintf(thistFile, "topStudyOutput_%s_%d.root", dataSets.c_str(), startFile);
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

    map<string, vector<AnaSamples::FileSummary>> fileMap;

    //Select approperiate datasets here
    if(dataSets.compare("TEST") == 0)
    {
        fileMap["ZJetsToNuNu"] = {ss["ZJetsToNuNu_HT_200to400"]};//{ss["ZJetsToNuNu_HT_600toInf"]};
        fileMap["ZJetsToNuNu_HT_600toInf"] = {ss["ZJetsToNuNu_HT_600toInf"]};
        fileMap["TTbarSingleLep"] = {ss["TTbarSingleLepT"]};
        fileMap["TTbarNoHad"] = {ss["TTbarSingleLepT"]};
        //fileMap["TTbarDiLep"] = {ss["TTbarDiLep"]};
        //fileMap["TTbarInc"] = {ss["TTbarInc"]};
        fileMap["TTbar"] = sc["TTbar"];
        fileMap["Signal_T2tt_mStop500_mLSP325"] = sc["Signal_T2tt_mStop500_mLSP325"];
        fileMap["Signal_T2tt_mStop850_mLSP100"] = sc["Signal_T2tt_mStop850_mLSP100"];
        fileMap["Signal_T1tttt_mGluino1200_mLSP800"] = sc["Signal_T1tttt_mGluino1200_mLSP800"];
        fileMap["Signal_T1tttt_mGluino1500_mLSP100"] = sc["Signal_T1tttt_mGluino1500_mLSP100"];
        fileMap["Data_SingleMuon_Run2016G"] = {ss["Data_SingleMuon_Run2016G"]};
        fileMap["Data_MET_Run2016G"] = {ss["Data_MET_Run2016G"]};
        fileMap["WJetsToLNu"] = {ss["WJetsToLNu_HT_800to1200"]};
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

    vector<Plotter::HistSummary> vh;

    Plotter::DatasetSummary ds_Znunu(   "Z#rightarrow#nu#nu", fileMap["ZJetsToNuNu"],                       "", "");
    Plotter::DatasetSummary ds_ttbar(   "t#bar{t} inc",       fileMap["TTbar"],                             "", "");
    Plotter::DatasetSummary ds_ttbar1l( "t#bar{t} l1",        fileMap["TTbarSingleLep"],                    "", "");
    Plotter::DatasetSummary ds_T2tt(    "T2tt (500, 325)",    fileMap["Signal_T2tt_mStop500_mLSP325"],      "", "");
    Plotter::DatasetSummary ds_T2tt_2(  "T2tt (850, 100)",    fileMap["Signal_T2tt_mStop850_mLSP100"],      "", "");
    Plotter::DatasetSummary ds_T1tttt(  "T1tttt (1200, 800)", fileMap["Signal_T1tttt_mGluino1200_mLSP800"], "", "");
    Plotter::DatasetSummary ds_T1tttt_2("T1tttt (1500, 100)", fileMap["Signal_T1tttt_mGluino1500_mLSP100"], "", "");

    Plotter::DatasetSummary dsData_SingleMuon("Data",       fileMap["Data_SingleMuon_Run2016G"], "passMuTrigger",          "");
    Plotter::DatasetSummary dsData_DoubleEG(  "Data",       fileMap["Data_DoubleEG"],   "passElecTrigger",        "");
    Plotter::DatasetSummary dsData_HTMHT(     "Data",       fileMap["Data_MET_Run2016G"],      "passMuHTorHTMHTTrigger", "");
    Plotter::DatasetSummary dsDY(             "DY",         fileMap["DYJetsToLL"],      "",                "");
    Plotter::DatasetSummary dsWj(             "W+Jets",     fileMap["WJetsToLNu"],      "",                "");
    Plotter::DatasetSummary dsDYInc(          "DY HT<100",  fileMap["IncDY"],           "genHT<100",       "");
    Plotter::DatasetSummary dsWjInc(          "Wj HT<100",  fileMap["WJetsToLNuInc"],   "genHT<100",       "");
    Plotter::DatasetSummary dstt2l(           "t#bar{t}",   fileMap["TTbarNoHad"],      "",                "");
    Plotter::DatasetSummary dstW(             "Single top", fileMap["tW"],              "",                "");
    Plotter::DatasetSummary dsttZ(            "t#bar{t}Z",  fileMap["TTZ"],             "",                "genWeight");
    Plotter::DatasetSummary dsVV(             "Diboson",    fileMap["Diboson"],         "",                "");
    Plotter::DatasetSummary dsRare(           "Rare",       fileMap["Rare"],            "",                "genWeight");
    Plotter::DatasetSummary dstthad(          "t#bar{t}",   fileMap["TTbarAll"],        "",                "");
    Plotter::DatasetSummary dsQCD(            "QCD",        fileMap["QCD"],             "",                "");
    Plotter::DatasetSummary dstthad_dummy(    "AllMC",      fileMap["TTbarAll"],        "",                "");
    std::vector<std::vector<Plotter::DatasetSummary>> stack_MC_2l = {{dsDY, dsDYInc}, {dstt2l}, {dstW}, {dsttZ}, {dsVV}, {dsRare}};
    std::vector<std::vector<Plotter::DatasetSummary>> stack_MC_1l = {{dstt2l}, {dsWj, dsWjInc}, {dstW}, {dsttZ}, {dsVV}, {dsRare}};
    std::vector<std::vector<Plotter::DatasetSummary>> stack_MC_0l = {{dstthad}, {dsWj, dsWjInc}, {ds_Znunu}, {dsttZ}, {dsQCD}, {dstW}, {dsRare, dstW, dsVV} };
    std::vector<std::vector<Plotter::DatasetSummary>> All_MC_0l   = {{dstthad_dummy, dsWj, dsWjInc, ds_Znunu, dsttZ, dsQCD, dstW, dsVV, dsRare, dstW}};

    auto PDSCutBind = [](Plotter::DatasetSummary pds, std::string cuts) 
    {
        pds.setCuts(cuts); 
        pds.label = pds.label + " " + cuts;
        for(size_t pos = pds.label.find(';', 0); pos != string::npos; pos = pds.label.find(';', pos)) pds.label[pos] = '_';
        return pds; 
    };
    auto PDSCutBindVec = [](std::vector<std::vector<Plotter::DatasetSummary>> vvpds, std::string cuts) 
    {
        for(auto& vpds : vvpds)
        {
            for(auto& pds : vpds)
            {
                std::string orig_cuts = pds.getCuts();
                if(orig_cuts.size())
                {
                    pds.setCuts(orig_cuts + ";" + cuts);
                }
                else
                {
                    pds.setCuts(cuts);
                }
                pds.label = pds.label + " " + cuts;
                for(size_t pos = pds.label.find(';', 0); pos != string::npos; pos = pds.label.find(';', pos)) pds.label[pos] = '_';
            }
        }
        return vvpds; 
    };
    auto PDSLabelBind = [](Plotter::DatasetSummary pds, std::string label) { pds.label = pds.label + " " + label; return pds; };

    auto PDCMaker = [&](std::string var) { return Plotter::DataCollection( "single", var, {ds_T1tttt, ds_T1tttt_2, ds_T2tt, ds_T2tt_2, ds_Znunu, ds_ttbar, ds_ttbar1l}); };

    auto PDCMaker_nTops   = [&](Plotter::DatasetSummary& pds) { return Plotter::DataCollection( "single", {{"nTops", pds}, {"nTopsNew", pds}, {"nTopsNewMVA", pds}, {"vTopsMatchNewMVA(size)", pds}}); };

    auto PDCMaker_distributions   = [&](Plotter::DatasetSummary& pds, std::string  var) 
    { 
        return Plotter::DataCollection( "single", {
                {"MVAvar_finalTop_"    + var,  PDSLabelBind(pds, "final top") }, 
                {"MVAvar_notFinalTop_" + var,  PDSLabelBind(pds, "background top") },
                {"MVAvar_genMatch_"    + var,  PDSLabelBind(pds, "gen match") }, 
                {"MVAvar_notGenMatch_" + var,  PDSLabelBind(pds, "not gen match") }, 
            }); 
    };
 
    auto vPDCMaker_disc    = [&](Plotter::DatasetSummary& pds)
    {
        return std::vector<Plotter::DataCollection>( { Plotter::DataCollection( "single", {{"discriminators", pds}}),
                    Plotter::DataCollection( "fill",  {{"discriminatorsParMatch", PDSLabelBind(pds, "2 + 3 jet match")}, {"discriminatorsMatch", PDSLabelBind(pds, "3 jet match")} }),
                    } );
    };
    auto vPDCMaker_disc2    = [&](Plotter::DatasetSummary& pds)
    {
        return std::vector<Plotter::DataCollection>( { Plotter::DataCollection( "single", {{"discriminators", pds}}),
                                                       Plotter::DataCollection( "fill",  {{"discriminatorsNoMatch", PDSLabelBind(pds, "Not matched")}, {"discriminatorsMatch", PDSLabelBind(pds, "Matched")} }),
                    } );
    };
    auto vPDCMaker_disc3    = [&](Plotter::DatasetSummary& pds)
    {
        return std::vector<Plotter::DataCollection>( { Plotter::DataCollection( "single", {{"discriminators", pds}}),
                                                        Plotter::DataCollection( "fill",  {{"discriminatorsParNoMatch", PDSLabelBind(pds, "Not matched")}, {"discriminatorsParMatch", PDSLabelBind(pds, "Matched")} }),
                    } );
    };
    auto vPDCMaker_disc4    = [&](Plotter::DatasetSummary& pds)
    {
        return std::vector<Plotter::DataCollection>( { Plotter::DataCollection( "single", {{"discriminatorsMatch", PDSLabelBind(pds, "3 match")}, {"discriminatorsMatch2", PDSLabelBind(pds, "2 match")}, {"discriminatorsMatch1", PDSLabelBind(pds, "1 match")}, {"discriminatorsMatch0", PDSLabelBind(pds, "0 match")}}),
                    } );
    };
    auto PDCMaker_topProp = [&](Plotter::DatasetSummary& pds, int iTop = 0, std::string var = "pt")
    {
        std::string afterVar = ((iTop<0)?(""):("[" + std::to_string(iTop) + "]")) + "(" + var + ")";
        return Plotter::DataCollection( "single", { {"vTops"       + afterVar, PDSLabelBind(pds, "ICHEP 2016")},
                                                    {"vTopsNew"    + afterVar, PDSLabelBind(pds, "ICHEP 2016 New Code")},
                                                    {"vTopsNewMVA" + afterVar, PDSLabelBind(pds, "MVA + AK8")} });
    };
    auto vPDCMaker_topProp = [&](Plotter::DatasetSummary& pds, int iTop = 0, std::string var = "pt")
    {
        std::string afterVar = ((iTop<0)?(""):("[" + std::to_string(iTop) + "]")) + "(" + var + ")";
        return std::vector<Plotter::DataCollection>({
                //Plotter::DataCollection( "fill",   { {"vTopsNew"    + afterVar, PDSLabelBind(pds, "New trijet only")}, {"vTopsMatchNew"    + afterVar, PDSLabelBind(pds, "New trijet Match")}, {"vTopsParMatchNew"    + afterVar, PDSLabelBind(pds, "New trijet Match")} }),
                Plotter::DataCollection( "single", { {"vTopsNewMVA" + afterVar, PDSLabelBind(pds, "MVA")}, {"vTopsMatchNewMVA" + afterVar, PDSLabelBind(pds, "MVA 3 Match")}, {"vTopsParMatchNewMVA" + afterVar, PDSLabelBind(pds, "MVA 2+3 Match")} }),
                    });
    };
    auto PDCMaker_topMultComp = [&](Plotter::DatasetSummary& pds, std::string var = "pt")
    {
        std::string varStr = "vTopsNewMVA(" + var + ")";
        return Plotter::DataCollection( "single", {{varStr, PDSCutBind(pds, "nTopsNewMVA=0")}, {varStr, PDSCutBind(pds, "nTopsNewMVA=1")}, {varStr, PDSCutBind(pds, "nTopsNewMVA=2")}, {varStr, PDSCutBind(pds, "nTopsNewMVA>2")} });
    };
    auto vPDCMaker_Eff = [&](Plotter::DatasetSummary & pds, std::string var = "pt")
    {
        std::string varStr = "(" + var + ")";
        return std::vector<Plotter::DataCollection>({//Plotter::DataCollection( "ratio", {{"vTopsGenMatchAllComb" + varStr, PDSLabelBind(pds, "Max eff")}, {"genTops" + varStr, PDSLabelBind(pds, "Max eff")} }),
                                                     Plotter::DataCollection( "ratio", {{"vTopsGenMatchNew"     + varStr, PDSLabelBind(pds, "ICHEP 2016 New Code")},  {"genTops" + varStr, PDSLabelBind(pds, "ICHEP 2016 New Code")}  }),
                                                     Plotter::DataCollection( "ratio", {{"vTopsGenMatchNewMVA"  + varStr, PDSLabelBind(pds, "MVA + AK8")},            {"genTops" + varStr, PDSLabelBind(pds, "MVA + AK8")}            }),
                    //                                 Plotter::DataCollection( "ratio", {{"vTopsGenMatchTriNewMVA"  + varStr, PDSLabelBind(pds, "MVA + AK8 trijet")},  {"genTops" + varStr, PDSLabelBind(pds, "MVA + AK8")}            }),
                    //                                 Plotter::DataCollection( "ratio", {{"vTopsGenMatchDiNewMVA"   + varStr, PDSLabelBind(pds, "MVA + AK8 dijet")},   {"genTops" + varStr, PDSLabelBind(pds, "MVA + AK8")}            }),
                    //                                 Plotter::DataCollection( "ratio", {{"vTopsGenMatchMonoNewMVA" + varStr, PDSLabelBind(pds, "MVA + AK8 monojet")}, {"genTops" + varStr, PDSLabelBind(pds, "MVA + AK8")}            })
                    });
    };
    auto vPDCMaker_Purity = [&](Plotter::DatasetSummary & pds, std::string var = "pt")
    {
        std::string varStr = "(" + var + ")";
        return std::vector<Plotter::DataCollection>({Plotter::DataCollection( "ratio", {{"vTopsMatchNew"    + varStr, PDSLabelBind(pds, "ICHEP 2016 New Code")}, {"vTopsNew"    + varStr, PDSLabelBind(pds, "ICHEP 2016 New Code")}}),
                                                     Plotter::DataCollection( "ratio", {{"vTopsMatchNewMVA" + varStr, PDSLabelBind(pds, "MVA + AK8")},           {"vTopsNewMVA" + varStr, PDSLabelBind(pds, "MVA + AK8")}})
                    });
    };
    auto vPDCMaker_fake = [&](std::string var = "met")
    {
        return std::vector<Plotter::DataCollection>({Plotter::DataCollection( "ratio", {{var, PDSLabelBind(PDSCutBind(ds_Znunu, "nTopsNew>0.5"),    "ICHEP 2016 New Code")}, {var, PDSLabelBind(ds_Znunu, "ICHEP 2016 New Code")}}),
                    Plotter::DataCollection( "ratio", {{var, PDSLabelBind(PDSCutBind(ds_Znunu, "nTopsNewMVA>0.5"), "MVA + AK8")},                                 {var, PDSLabelBind(ds_Znunu, "MVA + AK8")}}),
                    //Plotter::DataCollection( "ratio", {{var, PDSLabelBind(PDSCutBind(ds_Znunu, "nTopsNewMVA>0.5;vTopsNCandNewMVA[0]=3"), "MVA + AK8 trijet")},    {var, PDSLabelBind(ds_Znunu, "MVA + AK8 trijet")}}),
                    //Plotter::DataCollection( "ratio", {{var, PDSLabelBind(PDSCutBind(ds_Znunu, "nTopsNewMVA>0.5;vTopsNCandNewMVA[0]=2"), "MVA + AK8 dijet")},     {var, PDSLabelBind(ds_Znunu, "MVA + AK8 dijet")}}),
                    //Plotter::DataCollection( "ratio", {{var, PDSLabelBind(PDSCutBind(ds_Znunu, "nTopsNewMVA>0.5;vTopsNCandNewMVA[0]=1"), "MVA + AK8 monojet")},   {var, PDSLabelBind(ds_Znunu, "MVA + AK8 monojet")}})
                    });
    };
    auto vPDCMaker_DataMC_1l = [&](std::string var)
    {
        return std::vector<Plotter::DataCollection>({
                Plotter::DataCollection( "data", var, {dsData_HTMHT}),
                Plotter::DataCollection( "stack", var, stack_MC_1l),
            });
    };
    auto vPDCMaker_DataMC_2l = [&](std::string var)
    {
        return std::vector<Plotter::DataCollection>({
                Plotter::DataCollection( "data", var, {dsData_SingleMuon}),
                Plotter::DataCollection( "stack", var, stack_MC_2l),
            });
    };

    auto vPDCMaker_All_0l = [&](std::string var1, std::string var2, int ntops = -1)
    {
        if(ntops < 0)
        {
            return std::vector<Plotter::DataCollection>({
                    Plotter::DataCollection( "stack",  var1, stack_MC_0l),
                    Plotter::DataCollection( "single", var2, All_MC_0l),
                });
        }
        else
        {
            return std::vector<Plotter::DataCollection>({
                    Plotter::DataCollection( "stack",  var1, PDSCutBindVec(stack_MC_0l, "nTopsNewMVA="+std::to_string(ntops) ) ),
                    Plotter::DataCollection( "single", var2, PDSCutBindVec(All_MC_0l,   "nTopsNew="+std::to_string(ntops)    ) )
                });
        }
    };

    
    //std::vector<std::pair<std::string, std::string>> cutslistData = {{"SingleMu_baseline", "passSingleLep;passLeptVetoNoMu;passBJets;passnJets;passMET;HT>300"}, {"SingleMu_lowMet", "passSingleLep;passLeptVetoNoMu;passBJets;passnJets;!passMET;met>150;HT>300"}};
    std::vector<std::pair<std::string, std::string>> cutslistData_1l = {
        {"SingleMu_baseline", "passSingleLep;passLeptVetoNoMu;passBJetsTopTag;passnJetsTopTag;passMETTopTag;HTTopTag>300"}, 
        {"SingleMu_lowMet",   "passSingleLep;passLeptVetoNoMu;passBJetsTopTag;passnJetsTopTag;!passMETTopTag;met>150;HTTopTag>300"},
        {"SingleMu_nJetMet",  "passSingleLep;passLeptVetoNoMu;passnJetsTopTag;met>150"},
    };

    //for(auto& cuts : cutslistData_1l)
    //{
    //    for(std::string catagory : {"MVAvartop_", "MVAvarcand_"})
    //    {
    //        //"cand_m", "j12_m", "j13_m", "j23_m", "j1_p", "j2_p", "j3_p", "dTheta12", "dTheta23", "dTheta13", "j1_CSV", "j2_CSV", "j3_CSV", "j1_QGL", "j2_QGL", "j3_QGL"
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "mTop",      vPDCMaker_DataMC_1l(catagory + "cand_m"),    {1, 2}, cuts.second + "",  50,   0, 300,    false,  false,  "M_{Top}",         "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "mJ12",      vPDCMaker_DataMC_1l(catagory + "j12_m"),     {1, 2}, cuts.second + "",  50,   0, 300,    false,  false,  "M(j1, j2)",       "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "mJ23",      vPDCMaker_DataMC_1l(catagory + "j23_m"),     {1, 2}, cuts.second + "",  50,   0, 300,    false,  false,  "M(j2, j3)",       "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "mJ13",      vPDCMaker_DataMC_1l(catagory + "j13_m"),     {1, 2}, cuts.second + "",  50,   0, 300,    false,  false,  "M(j1, j3)",       "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "pJ1",       vPDCMaker_DataMC_1l(catagory + "j1_p"),      {1, 2}, cuts.second + "",  50,   0, 250,    false,  false,  "p(j1)",           "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "pJ2",       vPDCMaker_DataMC_1l(catagory + "j2_p"),      {1, 2}, cuts.second + "",  50,   0, 250,    false,  false,  "p(j2)",           "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "pJ3",       vPDCMaker_DataMC_1l(catagory + "j3_p"),      {1, 2}, cuts.second + "",  50,   0, 250,    false,  false,  "p(j3)",           "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "dTheta12",  vPDCMaker_DataMC_1l(catagory + "dTheta12"),  {1, 2}, cuts.second + "",  25,   0, 3.14,   false,  false,  "dTheta(J1, J2)",  "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "dTheta23",  vPDCMaker_DataMC_1l(catagory + "dTheta23"),  {1, 2}, cuts.second + "",  25,   0, 3.14,   false,  false,  "dTheta(J2, J3)",  "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "dTheta13",  vPDCMaker_DataMC_1l(catagory + "dTheta13"),  {1, 2}, cuts.second + "",  25,   0, 3.14,   false,  false,  "dTheta(J1, J3)",  "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "csv1",      vPDCMaker_DataMC_1l(catagory + "j1_CSV"),    {1, 2}, cuts.second + "",  50,   0,   1,    false,  false,  "CSV",             "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "csv2",      vPDCMaker_DataMC_1l(catagory + "j2_CSV"),    {1, 2}, cuts.second + "",  50,   0,   1,    false,  false,  "CSV",             "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "csv3",      vPDCMaker_DataMC_1l(catagory + "j3_CSV"),    {1, 2}, cuts.second + "",  50,   0,   1,    false,  false,  "CSV",             "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "qgl1",      vPDCMaker_DataMC_1l(catagory + "j1_QGL"),    {1, 2}, cuts.second + "",  50,   0,   1,    false,  false,  "QGL",             "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "qgl2",      vPDCMaker_DataMC_1l(catagory + "j2_QGL"),    {1, 2}, cuts.second + "",  50,   0,   1,    false,  false,  "QGL",             "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "qgl3",      vPDCMaker_DataMC_1l(catagory + "j3_QGL"),    {1, 2}, cuts.second + "",  50,   0,   1,    false,  false,  "QGL",             "Events"));
    //    }
    //
    //    vh.push_back(PHS("DataMC_TopProp_" + cuts.first + "_" + "nTops",         vPDCMaker_DataMC_1l("vTopsNewMVA(size)"),  {1, 2}, cuts.second + "",   8,   0,    8,    false,  false,  "nTops",         "Events"));
    //    vh.push_back(PHS("DataMC_TopProp_" + cuts.first + "_" + "pt",            vPDCMaker_DataMC_1l("vTopsNewMVA(pt)"),    {1, 2}, cuts.second + "",  50,   0, 1000,     true,  false,  "pt [GeV]",      "Events"));
    //    vh.push_back(PHS("DataMC_TopProp_" + cuts.first + "_" + "eta",           vPDCMaker_DataMC_1l("vTopsNewMVA(eta)"),   {1, 2}, cuts.second + "",  50,  -3,    3,    false,  false,  "#eta",          "Events"));
    //    vh.push_back(PHS("DataMC_TopProp_" + cuts.first + "_" + "discriminator", vPDCMaker_DataMC_1l("discriminators"),     {1, 2}, cuts.second + "",  50,   0,    1,    false,  false,  "discriminator", "Events"));
    //
    //    vh.push_back(PHS("puppi_Top_T_PT"  + cuts.first + "_" + "pt",    vPDCMaker_DataMC_1l("puppiLVectight_top(pt)"),   {1, 2}, cuts.second + "", 20, 0, 1200, true, false,  "top p_{T} [GeV]",   "Events"));
    //    vh.push_back(PHS("DataMC_TopProp_" + cuts.first + "_" + "nTops",         vPDCMaker_DataMC_1l("vTopsNewMVA(size)"),  {1, 2}, cuts.second + "",   8,   0,    8,    false,  false,  "nTops",         "Events"));
    //    vh.push_back(PHS("DataMC_TopProp_" + cuts.first + "_" + "pt",            vPDCMaker_DataMC_1l("vTopsNewMVA(pt)"),    {1, 2}, cuts.second + "",  50,   0, 1000,     true,  false,  "pt [GeV]",      "Events"));
    //    vh.push_back(PHS("DataMC_TopProp_" + cuts.first + "_" + "eta",           vPDCMaker_DataMC_1l("vTopsNewMVA(eta)"),   {1, 2}, cuts.second + "",  50,  -3,    3,    false,  false,  "#eta",          "Events"));
    //    vh.push_back(PHS("DataMC_TopProp_" + cuts.first + "_" + "discriminator", vPDCMaker_DataMC_1l("discriminators"),     {1, 2}, cuts.second + "",  50,   0,    1,    false,  false,  "discriminator", "Events"));
    //    vh.push_back(PHS("puppi_Top_T_PT"+ cuts.first + "_" + "pt",   vPDCMaker_DataMC_1l("puppiLVectight_top(pt)"),  {1, 2}, cuts.second + "", 20, 0, 1200, true, false,  "top p_{T} [GeV]",   "Events"));
    //    vh.push_back(PHS("puppi_Top_T_Num"+ cuts.first + "_" + "nTops",  vPDCMaker_DataMC_1l("puppiLVectight_top(size)"),  {1, 2}, cuts.second + "", 5, 0, 5, false, false,  "N_{t}",   "Events"));
    //    vh.push_back(PHS("puppi_Top_L_PT"+ cuts.first + "_" + "pt", vPDCMaker_DataMC_1l("puppiLVecLoose_top(pt)"),  {1, 2}, cuts.second + "", 20, 0, 1200, true, false,  "top p_{T} [GeV]",   "Events"));
    //    vh.push_back(PHS("puppi_Top_L_Num"+ cuts.first + "_" + "nTops",  vPDCMaker_DataMC_1l("puppiLVecLoose_top(size)"),  {1, 2}, cuts.second + "", 5, 0, 5, false, false,  "N_{t}",   "Events"));
    //    vh.push_back(PHS("puppi_w_T_PT"+ cuts.first + "_" + "PT",  vPDCMaker_DataMC_1l("puppiLVectight_w(pt)"),  {1, 2}, cuts.second + "", 50, 0, 1200, true, false,  "W p_{T} [GeV]",   "Events"));
    //    vh.push_back(PHS("puppi_w_T_Num"+ cuts.first + "_" + "nW", vPDCMaker_DataMC_1l("puppiLVectight_w(size)"),  {1, 2}, cuts.second + "", 5, 0, 5, false, false,  "N_{W}",   "Events"));
    //    vh.push_back(PHS("puppi_w_L_Num"+ cuts.first + "_" + "nW", vPDCMaker_DataMC_1l("puppiLVecLoose_w(size)"),  {1, 2}, cuts.second + "", 5, 0, 5, false, false,  "N_{W}",   "Events"));
    //    vh.push_back(PHS("puppi_w_L_PT"+ cuts.first + "_" + "PT",  vPDCMaker_DataMC_1l("puppiLVecLoose_w(pt)"),  {1, 2}, cuts.second + "", 50, 0, 1200, true, false,  "W p_{T} [GeV]",   "Events"));
    //    vh.push_back(PHS("puppiTau21"  + cuts.first + "_"+ "tau21_W",  vPDCMaker_DataMC_1l("puppitau2Dtau1"),  {1, 2}, cuts.second, 100, 0, 1, false, false,  "N-subjettines \tau_{21}",   "Events"));
    //    vh.push_back(PHS("puppiTauD32"  + cuts.first + "_"+ "tau32_Top",  vPDCMaker_DataMC_1l("puppitau3Dtau2"),  {1, 2}, cuts.second, 100, 0, 1, false, false,  "N-subjettines \tau_{32}",   "Events")); 
    //    vh.push_back(PHS("puppitau2Dtau1_SDM"  + cuts.first + "_"+ "SoftDropMass_W",  vPDCMaker_DataMC_1l("puppitau2Dtau1_SDM"),  {1, 2}, cuts.second, 10, 0, 200, false, false,  "m [GeV]",   "Events"));
    //    vh.push_back(PHS("puppitau3Dtau2_SDM"  + cuts.first + "_"+ "SoftDropMass_Top",  vPDCMaker_DataMC_1l("puppitau3Dtau2_SDM"),  {1, 2}, cuts.second, 10, 0, 200, false, false,  "m [GeV]",   "Events"));
    //}
    //
    //
    //std::vector<std::pair<std::string, std::string>> cutslistData_2l = {
    //    {"DoubleMu_baseline", "passDoubleLep;passLeptVetoNoMu;passBJetsTopTag;passnJetsTopTag;passMETTopTag"}, 
    //    {"DoubleMu_lowMet",   "passDoubleLep;passLeptVetoNoMu;passBJetsTopTag;passnJetsTopTag;!passMETTopTag"},
    //    {"DoubleMu_nJet",     "passDoubleLep;passLeptVetoNoMu;passnJetsTopTag"},
    //};
    //
    //for(auto& cuts : cutslistData_2l)
    //{
    //    for(std::string catagory : {"MVAvartop_", "MVAvarcand_"})
    //    {
    //        //"cand_m", "j12_m", "j13_m", "j23_m", "j1_p", "j2_p", "j3_p", "dTheta12", "dTheta23", "dTheta13", "j1_CSV", "j2_CSV", "j3_CSV", "j1_QGL", "j2_QGL", "j3_QGL"
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "mTop",      vPDCMaker_DataMC_2l(catagory + "cand_m"),    {1, 2}, cuts.second + "",  50,   0, 300,    false,  false,  "M_{Top}",         "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "mJ12",      vPDCMaker_DataMC_2l(catagory + "j12_m"),     {1, 2}, cuts.second + "",  50,   0, 300,    false,  false,  "M(j1, j2)",       "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "mJ23",      vPDCMaker_DataMC_2l(catagory + "j23_m"),     {1, 2}, cuts.second + "",  50,   0, 300,    false,  false,  "M(j2, j3)",       "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "mJ13",      vPDCMaker_DataMC_2l(catagory + "j13_m"),     {1, 2}, cuts.second + "",  50,   0, 300,    false,  false,  "M(j1, j3)",       "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "pJ1",       vPDCMaker_DataMC_2l(catagory + "j1_p"),      {1, 2}, cuts.second + "",  50,   0, 250,    false,  false,  "p(j1)",           "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "pJ2",       vPDCMaker_DataMC_2l(catagory + "j2_p"),      {1, 2}, cuts.second + "",  50,   0, 250,    false,  false,  "p(j2)",           "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "pJ3",       vPDCMaker_DataMC_2l(catagory + "j3_p"),      {1, 2}, cuts.second + "",  50,   0, 250,    false,  false,  "p(j3)",           "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "dTheta12",  vPDCMaker_DataMC_2l(catagory + "dTheta12"),  {1, 2}, cuts.second + "",  25,   0, 3.14,   false,  false,  "dTheta(J1, J2)",  "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "dTheta23",  vPDCMaker_DataMC_2l(catagory + "dTheta23"),  {1, 2}, cuts.second + "",  25,   0, 3.14,   false,  false,  "dTheta(J2, J3)",  "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "dTheta13",  vPDCMaker_DataMC_2l(catagory + "dTheta13"),  {1, 2}, cuts.second + "",  25,   0, 3.14,   false,  false,  "dTheta(J1, J3)",  "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "csv1",      vPDCMaker_DataMC_2l(catagory + "j1_CSV"),    {1, 2}, cuts.second + "",  50,   0,   1,    false,  false,  "CSV",             "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "csv2",      vPDCMaker_DataMC_2l(catagory + "j2_CSV"),    {1, 2}, cuts.second + "",  50,   0,   1,    false,  false,  "CSV",             "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "csv3",      vPDCMaker_DataMC_2l(catagory + "j3_CSV"),    {1, 2}, cuts.second + "",  50,   0,   1,    false,  false,  "CSV",             "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "qgl1",      vPDCMaker_DataMC_2l(catagory + "j1_QGL"),    {1, 2}, cuts.second + "",  50,   0,   1,    false,  false,  "QGL",             "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "qgl2",      vPDCMaker_DataMC_2l(catagory + "j2_QGL"),    {1, 2}, cuts.second + "",  50,   0,   1,    false,  false,  "QGL",             "Events"));
    //        vh.push_back(PHS("DataMC_" + catagory + cuts.first + "_" + "qgl3",      vPDCMaker_DataMC_2l(catagory + "j3_QGL"),    {1, 2}, cuts.second + "",  50,   0,   1,    false,  false,  "QGL",             "Events"));
    //    }
    //
    //    vh.push_back(PHS("DataMC_TopProp_" + cuts.first + "_" + "nTops",         vPDCMaker_DataMC_2l("vTopsNewMVA(size)"),  {1, 2}, cuts.second + "",   8,   0,    8,    false,  false,  "nTops",         "Events"));
    //    vh.push_back(PHS("DataMC_TopProp_" + cuts.first + "_" + "pt",            vPDCMaker_DataMC_2l("vTopsNewMVA(pt)"),    {1, 2}, cuts.second + "",  50,   0, 1000,     true,  false,  "pt [GeV]",      "Events"));
    //    vh.push_back(PHS("DataMC_TopProp_" + cuts.first + "_" + "eta",           vPDCMaker_DataMC_2l("vTopsNewMVA(eta)"),   {1, 2}, cuts.second + "",  50,  -3,    3,    false,  false,  "#eta",          "Events"));
    //    vh.push_back(PHS("DataMC_TopProp_" + cuts.first + "_" + "discriminator", vPDCMaker_DataMC_2l("discriminators"),     {1, 2}, cuts.second + "",  50,   0,    1,    false,  false,  "discriminator", "Events"));
    //
    //    vh.push_back(PHS("puppi_Top_T_PT_"     + cuts.first + "_" + "pt",               vPDCMaker_DataMC_2l("puppiLVectight_top(pt)"),   {1, 2}, cuts.second, 20, 0, 1200,  true, false,  "top p_{T} [GeV]",         "Events"));
    //    vh.push_back(PHS("puppi_Top_T_Num_"    + cuts.first + "_" + "nTops",            vPDCMaker_DataMC_2l("puppiLVectight_top(size)"), {1, 2}, cuts.second, 5, 0, 5,     false, false,  "N_{t}",                   "Events")); 
    //    vh.push_back(PHS("puppi_Top_L_PT_"     + cuts.first + "_" + "pt",               vPDCMaker_DataMC_2l("puppiLVecLoose_top(pt)"),   {1, 2}, cuts.second, 20, 0, 1200,  true, false,  "top p_{T} [GeV]",         "Events"));
    //    vh.push_back(PHS("puppi_Top_L_Num_"    + cuts.first + "_" + "nTops",            vPDCMaker_DataMC_2l("puppiLVecLoose_top(size)"), {1, 2}, cuts.second, 5, 0, 5,     false, false,  "N_{t}",                   "Events"));
    //    vh.push_back(PHS("puppi_w_T_PT_"       + cuts.first + "_" + "pt",               vPDCMaker_DataMC_2l("puppiLVectight_w(pt)"),     {1, 2}, cuts.second, 50, 0, 1200,  true, false,  "W p_{T} [GeV]",           "Events"));
    //    vh.push_back(PHS("puppi_w_T_Num_"      + cuts.first + "_" + "nW",               vPDCMaker_DataMC_2l("puppiLVectight_w(size)"),   {1, 2}, cuts.second, 5, 0, 5,     false, false,  "N_{W}",                   "Events"));
    //    vh.push_back(PHS("puppi_w_L_Num_"      + cuts.first + "_" + "nW",               vPDCMaker_DataMC_2l("puppiLVecLoose_w(size)"),   {1, 2}, cuts.second, 5, 0, 5,     false, false,  "N_{W}",                   "Events"));
    //    vh.push_back(PHS("puppi_w_L_PT_"       + cuts.first + "_" + "pt",               vPDCMaker_DataMC_2l("puppiLVecLoose_w(pt)"),     {1, 2}, cuts.second, 50, 0, 1200,  true, false,  "W p_{T} [GeV]",           "Events")); 
    //    vh.push_back(PHS("puppiTau21_"         + cuts.first + "_" + "tau21_W",          vPDCMaker_DataMC_2l("puppitau2Dtau1"),           {1, 2}, cuts.second, 100, 0, 1,   false, false,  "N-subjettines \tau_{21}", "Events"));
    //    vh.push_back(PHS("puppiTau32_"         + cuts.first + "_" + "tau32_Top",        vPDCMaker_DataMC_2l("puppitau3Dtau2"),           {1, 2}, cuts.second, 100, 0, 1,   false, false,  "N-subjettines \tau_{32}", "Events"));
    //    vh.push_back(PHS("puppi_SDM_"          + cuts.first + "_" + "SoftDropMass_W",   vPDCMaker_DataMC_2l("puppitau2Dtau1_SDM"),       {1, 2}, cuts.second, 10, 0, 200,  false, false,  "m [GeV]",                 "Events"));
    //    vh.push_back(PHS("puppi_SDM_"          + cuts.first + "_" + "SoftDropMass_Top", vPDCMaker_DataMC_2l("puppitau3Dtau2_SDM"),       {1, 2}, cuts.second, 10, 0, 200,  false, false,  "m [GeV]",                 "Events"));
    //}


    std::vector<std::pair<std::string, std::string>> cutslist = {
        {"baseline", "passLeptVetoTopTag;passBJetsTopTag;passnJetsTopTag;passMETTopTag"}, 
        {"lowMet",   "passLeptVetoTopTag;passBJetsTopTag;passnJetsTopTag;!passMETTopTag"}
    };
    std::vector<std::pair<PDS, std::string>> pdsVec = {{ds_T1tttt, "T1tttt_1"}, {ds_T1tttt_2, "T1tttt_2"}, {ds_T2tt, "T2tt"}, {ds_T2tt_2, "T2tt_2"}, {ds_Znunu, "Znunu"}, {ds_ttbar, "ttbar"}, {ds_ttbar1l, "ttbar1l"}};

    for(auto& cuts : cutslist)
    {
        vh.push_back(PHS(cuts.first + "_nTops",    {PDCMaker("nTops")},       {1, 1}, cuts.second,  10, 0, 10,  true,  true,  "N_{T}", "Norm Events"));
        vh.push_back(PHS(cuts.first + "_nTopsNew", {PDCMaker("nTopsNewMVA")}, {1, 1}, cuts.second,  10, 0, 10,  true,  true,  "N_{T}", "Norm Events"));

        for(auto& pds : pdsVec)
        {
            vh.push_back(PHS(cuts.first + "_" + "nTops_"          + pds.second,  {PDCMaker_topProp(pds.first, -1, "size")},  {1, 1}, cuts.second + "",  10,   0, 10,    false,  false,  "N_{T}",             "Events"));
            //vh.push_back(PHS(cuts.first + "_" + "disc_"           + pds.second,  vPDCMaker_disc(pds.first),                  {1, 1}, cuts.second + "",   50,  0, 1,     false,  false,  "disc",              "Events"));
            vh.push_back(PHS(cuts.first + "_" + "disc_3match_"    + pds.second,  vPDCMaker_disc2(pds.first),                 {1, 1}, cuts.second + "",   50,  0, 1,     false,  false,  "disc",              "Events"));
            vh.push_back(PHS(cuts.first + "_" + "disc_2_3match_"  + pds.second,  vPDCMaker_disc3(pds.first),                 {1, 1}, cuts.second + "",   50,  0, 1,     false,  false,  "disc",              "Events"));
            //vh.push_back(PHS(cuts.first + "_" + "disc_byCatagory" + pds.second,  vPDCMaker_disc4(pds.first),                 {1, 1}, cuts.second + "",   50,  0, 1,     false,  true,   "disc",              "Norm Events"));
            vh.push_back(PHS(cuts.first + "_" + "tAll_pt_"        + pds.second,  {PDCMaker_topProp(pds.first, -1, "pt")},    {1, 1}, cuts.second + "",   50,  0, 2000,  false,  false,  "p_{T} top [GeV]",   "Events"));
            vh.push_back(PHS(cuts.first + "_" + "t1_pt_"          + pds.second,  {PDCMaker_topProp(pds.first,  0, "pt")},    {1, 1}, cuts.second + "",   50,  0, 2000,  false,  false,  "p_{T} top 1 [GeV]", "Events"));
            vh.push_back(PHS(cuts.first + "_" + "t2_pt_"          + pds.second,  {PDCMaker_topProp(pds.first,  1, "pt")},    {1, 1}, cuts.second + "",   50,  0, 2000,  false,  false,  "p_{T} top 2 [GeV]", "Events"));
            vh.push_back(PHS(cuts.first + "_" + "t3_pt_"          + pds.second,  {PDCMaker_topProp(pds.first,  2, "pt")},    {1, 1}, cuts.second + "",   50,  0, 2000,  false,  false,  "p_{T} top 3 [GeV]", "Events"));
            vh.push_back(PHS(cuts.first + "_" + "t4_pt_"          + pds.second,  {PDCMaker_topProp(pds.first,  3, "pt")},    {1, 1}, cuts.second + "",   50,  0, 2000,  false,  false,  "p_{T} top 4 [GeV]", "Events"));
            //vh.push_back(PHS(cuts.first + "_" + "tAll_eta_"       + pds.second,  {PDCMaker_topProp(pds.first, -1, "eta")},   {1, 1}, cuts.second + "",  100, -5, 5,     false,  false,  "eta top",           "Events"));
            //vh.push_back(PHS(cuts.first + "_" + "t1_eta_"         + pds.second,  {PDCMaker_topProp(pds.first,  0, "eta")},   {1, 1}, cuts.second + "",  100, -5, 5,     false,  false,  "eta top 1",         "Events"));
            //vh.push_back(PHS(cuts.first + "_" + "t2_eta_"         + pds.second,  {PDCMaker_topProp(pds.first,  1, "eta")},   {1, 1}, cuts.second + "",  100, -5, 5,     false,  false,  "eta top 2",         "Events"));
            //vh.push_back(PHS(cuts.first + "_" + "t3_eta_"         + pds.second,  {PDCMaker_topProp(pds.first,  2, "eta")},   {1, 1}, cuts.second + "",  100, -5, 5,     false,  false,  "eta top 3",         "Events"));
            //vh.push_back(PHS(cuts.first + "_" + "t4_eta_"         + pds.second,  {PDCMaker_topProp(pds.first,  3, "eta")},   {1, 1}, cuts.second + "",  100, -5, 5,     false,  false,  "eta top 4",         "Events"));
            vh.push_back(PHS(cuts.first + "_" + "match_nTops_"    + pds.second,  vPDCMaker_topProp(pds.first, -1, "size"),   {1, 1}, cuts.second + "",  10,   0, 10,    false,  false,  "N_{T}",             "Events"));
            vh.push_back(PHS(cuts.first + "_" + "match_tAll_pt_"  + pds.second,  vPDCMaker_topProp(pds.first, -1, "pt"),     {1, 1}, cuts.second + "",  100,  0, 2000,  false,  false,  "p_{T} top [GeV]",   "Events"));
            //vh.push_back(PHS(cuts.first + "_" + "match_tAll_eta_" + pds.second,  vPDCMaker_topProp(pds.first, -1, "eta"),    {1, 1}, cuts.second + "",  100, -5, 5,     false,  false,  "eta top",           "Events"));
            vh.push_back(PHS(cuts.first + "_" + "tops_pt_"        + pds.second,  {PDCMaker_topMultComp(pds.first,  "pt")},   {1, 1}, cuts.second + "",  100, -5, 2000,  false,  false,  "p_{T} top [GeV]",   "Events"));
            vh.push_back(PHS(cuts.first + "_" + "tops_m_"         + pds.second,  {PDCMaker_topMultComp(pds.first,  "M")},    {1, 1}, cuts.second + "",  100, -5, 500,   false,  false,  "mass top [GeV]",    "Events"));
            vh.push_back(PHS(cuts.first + "_" + "tops_eta_"       + pds.second,  {PDCMaker_topMultComp(pds.first,  "eta")},  {1, 1}, cuts.second + "",  100, -5, 5,     false,  false,  "eta top",           "Events"));

            //vh.push_back(PHS(cuts.first + "_" + "eff_eta_" + pds.second,  vPDCMaker_Eff(pds.first, "eta"),  {1, 1}, cuts.second + "",   20, -5,    5,     false,  false,  "top eta",          "Efficiency"));
            vh.push_back(PHS(cuts.first + "_" + "eff_pt_"  + pds.second,  vPDCMaker_Eff(pds.first, "pt"),   {1, 1}, cuts.second + "",   20,  0, 1000,     false,  false,  "top pt [GeV]",     "Efficiency"));
            //vh.push_back(PHS(cuts.first + "_" + "eff_m_"   + pds.second,  vPDCMaker_Eff(pds.first, "M"),    {1, 1}, cuts.second + "",   20,  0,  500,     false,  false,  "top mass [GeV]",   "Efficiency"));

            //vh.push_back(PHS(cuts.first + "_" + "purity_eta_" + pds.second,  vPDCMaker_Purity(pds.first, "eta"),  {1, 1}, cuts.second + "",  100, -5,    5,     false,  false,  "top eta",          "Purity"));
            vh.push_back(PHS(cuts.first + "_" + "purity_pt_"  + pds.second,  vPDCMaker_Purity(pds.first, "pt"),   {1, 1}, cuts.second + "",   20,  0, 1000,     false,  false,  "top pt [GeV]",     "Purity"));
            //vh.push_back(PHS(cuts.first + "_" + "purity_m_"   + pds.second,  vPDCMaker_Purity(pds.first, "M"),    {1, 1}, cuts.second + "",  100,  0,  500,     false,  false,  "top mass [GeV]",   "Purity"));

            vh.push_back(PHS("MVAVar_" + cuts.first + "_cand_m" + "_"    + pds.second,  {PDCMaker_distributions(pds.first,  "cand_m")},    {1, 1}, cuts.second + "",  100, 0, 300,   false,  true,  "top mass [GeV]",   "Events"));
            vh.push_back(PHS("MVAVar_" + cuts.first + "_j12_m" + "_"     + pds.second,  {PDCMaker_distributions(pds.first,  "j12_m")},     {1, 1}, cuts.second + "",  100, 0, 200,   false,  true,  "dijet (12) mass [GeV]",   "Events"));
            vh.push_back(PHS("MVAVar_" + cuts.first + "_j23_m" + "_"     + pds.second,  {PDCMaker_distributions(pds.first,  "j23_m")},     {1, 1}, cuts.second + "",  100, 0, 200,   false,  true,  "dijet (23) mass [GeV]",   "Events"));
            vh.push_back(PHS("MVAVar_" + cuts.first + "_j13_m" + "_"     + pds.second,  {PDCMaker_distributions(pds.first,  "j13_m")},     {1, 1}, cuts.second + "",  100, 0, 200,   false,  true,  "dijet (13) mass [GeV]",   "Events"));
            vh.push_back(PHS("MVAVar_" + cuts.first + "_j1_p" + "_"      + pds.second,  {PDCMaker_distributions(pds.first,  "j1_p")},      {1, 1}, cuts.second + "",  100, 0, 400,   false,  true,  "jet 1 momentum [GeV]",   "Events"));
            vh.push_back(PHS("MVAVar_" + cuts.first + "_j2_p" + "_"      + pds.second,  {PDCMaker_distributions(pds.first,  "j2_p")},      {1, 1}, cuts.second + "",  100, 0, 400,   false,  true,  "jet 2 momentum [GeV]",   "Events"));
            vh.push_back(PHS("MVAVar_" + cuts.first + "_j3_p" + "_"      + pds.second,  {PDCMaker_distributions(pds.first,  "j3_p")},      {1, 1}, cuts.second + "",  100, 0, 400,   false,  true,  "jet 3 momentum [GeV]",   "Events"));
            vh.push_back(PHS("MVAVar_" + cuts.first + "_dTheta12" + "_"  + pds.second,  {PDCMaker_distributions(pds.first,  "dTheta12")},  {1, 1}, cuts.second + "",  100, 0, 3.14,  false,  true,  "dTheta12",   "Events"));
            vh.push_back(PHS("MVAVar_" + cuts.first + "_dTheta23" + "_"  + pds.second,  {PDCMaker_distributions(pds.first,  "dTheta23")},  {1, 1}, cuts.second + "",  100, 0, 3.14,  false,  true,  "dTheta23",   "Events"));
            vh.push_back(PHS("MVAVar_" + cuts.first + "_dTheta13" + "_"  + pds.second,  {PDCMaker_distributions(pds.first,  "dTheta13")},  {1, 1}, cuts.second + "",  100, 0, 3.14,  false,  true,  "dTheta13",   "Events"));
            vh.push_back(PHS("MVAVar_" + cuts.first + "_j1_CSV" + "_"    + pds.second,  {PDCMaker_distributions(pds.first,  "j1_CSV")},    {1, 1}, cuts.second + "",  100, 0, 1.0,   false,  true,  "j1 CSV",   "Events"));
            vh.push_back(PHS("MVAVar_" + cuts.first + "_j2_CSV" + "_"    + pds.second,  {PDCMaker_distributions(pds.first,  "j2_CSV")},    {1, 1}, cuts.second + "",  100, 0, 1.0,   false,  true,  "j2 CSV",   "Events"));
            vh.push_back(PHS("MVAVar_" + cuts.first + "_j3_CSV" + "_"    + pds.second,  {PDCMaker_distributions(pds.first,  "j3_CSV")},    {1, 1}, cuts.second + "",  100, 0, 1.0,   false,  true,  "j3 CSV",   "Events"));
            vh.push_back(PHS("MVAVar_" + cuts.first + "_j1_QGL" + "_"    + pds.second,  {PDCMaker_distributions(pds.first,  "j1_QGL")},    {1, 1}, cuts.second + "",  100, 0, 1.0,   false,  true,  "j1 QGL",   "Events"));
            vh.push_back(PHS("MVAVar_" + cuts.first + "_j2_QGL" + "_"    + pds.second,  {PDCMaker_distributions(pds.first,  "j2_QGL")},    {1, 1}, cuts.second + "",  100, 0, 1.0,   false,  true,  "j2 QGL",   "Events"));
            vh.push_back(PHS("MVAVar_" + cuts.first + "_j3_QGL" + "_"    + pds.second,  {PDCMaker_distributions(pds.first,  "j3_QGL")},    {1, 1}, cuts.second + "",  100, 0, 1.0,   false,  true,  "j3 QGL",   "Events"));
        }

        //vh.push_back(PHS(cuts.first + "_MCComp_nTops",    vPDCMaker_All_0l("vTopsNewMVA(size)", "vTopsNew(size)"),    {2, 1}, /*cuts.second + */"",  10,  0,   10,  true,  false,  "N_{T}",     "Events"));
        //vh.push_back(PHS(cuts.first + "_MCComp_nbNIT",    vPDCMaker_All_0l("nBNotInTop", "nBNotInTop"),               {2, 1}, /*cuts.second + */"",  10,  0,   10,  true,  false,  "N_{T}",     "Events"));
        //vh.push_back(PHS(cuts.first + "_MCComp_topPt_0t", vPDCMaker_All_0l("vTopsNewMVA(pt)", "vTopsNew(pt)", 0),     {2, 1}, /*cuts.second + */"",  50,  0, 1000,  true,  false,  "Top p_{T} [GeV]", "Events"));
        //vh.push_back(PHS(cuts.first + "_MCComp_topPt_1t", vPDCMaker_All_0l("vTopsNewMVA(pt)", "vTopsNew(pt)", 1),     {2, 1}, /*cuts.second + */"",  50,  0, 1000,  true,  false,  "Top p_{T} [GeV]", "Events"));
        //vh.push_back(PHS(cuts.first + "_MCComp_topPt_2t", vPDCMaker_All_0l("vTopsNewMVA(pt)", "vTopsNew(pt)", 2),     {2, 1}, /*cuts.second + */"",  50,  0, 1000,  true,  false,  "Top p_{T} [GeV]", "Events"));
        //vh.push_back(PHS(cuts.first + "_MCComp_topPt_3t", vPDCMaker_All_0l("vTopsNewMVA(pt)", "vTopsNew(pt)", 3),     {2, 1}, /*cuts.second + */"",  50,  0, 1000,  true,  false,  "Top p_{T} [GeV]", "Events"));
        //vh.push_back(PHS(cuts.first + "_MCComp_met_0t",   vPDCMaker_All_0l("met", "met", 0),                          {2, 1}, /*cuts.second + */"",  50,  0, 1000,  true,  false,  "MET [GeV]", "Events"));
        //vh.push_back(PHS(cuts.first + "_MCComp_met_1t",   vPDCMaker_All_0l("met", "met", 1),                          {2, 1}, /*cuts.second + */"",  50,  0, 1000,  true,  false,  "MET [GeV]", "Events"));
        //vh.push_back(PHS(cuts.first + "_MCComp_met_2t",   vPDCMaker_All_0l("met", "met", 2),                          {2, 1}, /*cuts.second + */"",  50,  0, 1000,  true,  false,  "MET [GeV]", "Events"));
        //vh.push_back(PHS(cuts.first + "_MCComp_met_3t",   vPDCMaker_All_0l("met", "met", 3),                          {2, 1}, /*cuts.second + */"",  50,  0, 1000,  true,  false,  "MET [GeV]", "Events"));
        ////vh.push_back(PHS(cuts.first + "_MCComp_ht_0t",    vPDCMaker_All_0l("HT", "HT", 0),                            {2, 1}, /*cuts.second + */"",  50,  0, 2000,  true,  false,  "HT [GeV]",  "Events"));
        ////vh.push_back(PHS(cuts.first + "_MCComp_ht_1t",    vPDCMaker_All_0l("HT", "HT", 1),                            {2, 1}, /*cuts.second + */"",  50,  0, 2000,  true,  false,  "HT [GeV]",  "Events"));
        ////vh.push_back(PHS(cuts.first + "_MCComp_ht_2t",    vPDCMaker_All_0l("HT", "HT", 2),                            {2, 1}, /*cuts.second + */"",  50,  0, 2000,  true,  false,  "HT [GeV]",  "Events"));
        ////vh.push_back(PHS(cuts.first + "_MCComp_ht_3t",    vPDCMaker_All_0l("HT", "HT", 3),                            {2, 1}, /*cuts.second + */"",  50,  0, 2000,  true,  false,  "HT [GeV]",  "Events"));
        //
        //vh.push_back(PHS(cuts.first + "_" + "fakerate_met",   vPDCMaker_fake("met"),                      {1, 1}, cuts.second + "",   30,  0,  1200,     false,  false,  "met [GeV]",   "Fakerate"));
        //vh.push_back(PHS(cuts.first + "_" + "fakerate_njet",  vPDCMaker_fake("cntNJetsPt50Eta24TopTag"),  {1, 1}, cuts.second + "",   20,  0,    20,     false,  false,  "N Jets",      "Fakerate"));
        //vh.push_back(PHS(cuts.first + "_" + "fakerate_nbjet", vPDCMaker_fake("cntCSVSTopTag"),                   {1, 1}, cuts.second + "",   10,  0,    10,     false,  false,  "N b-jets",    "Fakerate"));
        //
        ////vh.push_back(PHS(cuts.first + "_" + "nbNotInTop", {PDCMaker("nBNotInTop")}, {1, 1}, cuts.second,  10, 0, 10,  true,  false,  "N_{b}", "Events"));
        //vh.push_back(PHS(cuts.first + "_" + "nb",         {PDCMaker("cntCSVSTopTag")},     {1, 1}, cuts.second,  10, 0, 10,  true,  false,  "N_{b}", "Events"));
        //
        //vh.push_back(PHS(cuts.first + "_" + "topMass1", {PDCMaker("vTops[0](M)")}, {1, 1}, cuts.second,  100, 0, 500,  false,  true,  "Top Mass [GeV]", "Norm Events"));
        //vh.push_back(PHS(cuts.first + "_" + "topMass2", {PDCMaker("vTops[1](M)")}, {1, 1}, cuts.second,  100, 0, 500,  false,  true,  "Top Mass [GeV]", "Norm Events"));
        //vh.push_back(PHS(cuts.first + "_" + "topMass3", {PDCMaker("vTops[2](M)")}, {1, 1}, cuts.second,  100, 0, 500,  false,  true,  "Top Mass [GeV]", "Norm Events"));
        //vh.push_back(PHS(cuts.first + "_" + "topMass4", {PDCMaker("vTops[3](M)")}, {1, 1}, cuts.second,  100, 0, 500,  false,  true,  "Top Mass [GeV]", "Norm Events"));
        //
        //vh.push_back(PHS(cuts.first + "_" + "topMassAllNew", {PDCMaker("vTopsNewMVA(M)")},    {1, 1}, cuts.second,  100, 0, 500,  false,  true,  "Top Mass [GeV]", "Norm Events"));
        //vh.push_back(PHS(cuts.first + "_" + "topMass1New",   {PDCMaker("vTopsNewMVA[0](M)")}, {1, 1}, cuts.second,  100, 0, 500,  false,  true,  "Top Mass [GeV]", "Norm Events"));
        //vh.push_back(PHS(cuts.first + "_" + "topMass2New",   {PDCMaker("vTopsNewMVA[1](M)")}, {1, 1}, cuts.second,  100, 0, 500,  false,  true,  "Top Mass [GeV]", "Norm Events"));
        //vh.push_back(PHS(cuts.first + "_" + "topMass3New",   {PDCMaker("vTopsNewMVA[2](M)")}, {1, 1}, cuts.second,  100, 0, 500,  false,  true,  "Top Mass [GeV]", "Norm Events"));
        //vh.push_back(PHS(cuts.first + "_" + "topMass4New",   {PDCMaker("vTopsNewMVA[3](M)")}, {1, 1}, cuts.second,  100, 0, 500,  false,  true,  "Top Mass [GeV]", "Norm Events"));
        //
        //vh.push_back(PHS(cuts.first + "_" + "topPtAllNew", {PDCMaker("vTopsNewMVA(pt)")},    {1, 1}, cuts.second,  100, 0, 2000,  true,  true,  "Top p_{T} [GeV]", "Norm Events"));
        //vh.push_back(PHS(cuts.first + "_" + "topPt1New",   {PDCMaker("vTopsNewMVA[0](pt)")}, {1, 1}, cuts.second,  100, 0, 2000,  true,  true,  "Top p_{T} [GeV]", "Norm Events"));
        //vh.push_back(PHS(cuts.first + "_" + "topPt2New",   {PDCMaker("vTopsNewMVA[1](pt)")}, {1, 1}, cuts.second,  100, 0, 2000,  true,  true,  "Top p_{T} [GeV]", "Norm Events"));
        //vh.push_back(PHS(cuts.first + "_" + "topPt3New",   {PDCMaker("vTopsNewMVA[2](pt)")}, {1, 1}, cuts.second,  100, 0, 2000,  true,  true,  "Top p_{T} [GeV]", "Norm Events"));
        //vh.push_back(PHS(cuts.first + "_" + "topPt4New",   {PDCMaker("vTopsNewMVA[3](pt)")}, {1, 1}, cuts.second,  100, 0, 2000,  true,  true,  "Top p_{T} [GeV]", "Norm Events"));
    }

    vh.push_back(PHS("met", {PDCMaker("met")}, {1, 2}, "",  100, 0, 1000,  true,  true,  "MET [GeV]", "Norm Events"));

    set<AnaSamples::FileSummary> vvf;
    for(auto& fsVec : fileMap) for(auto& fs : fsVec.second) vvf.insert(fs);

    RegisterFunctions* rf = new RegisterFunctionsTopStudy();

    Plotter plotter(vh, vvf, fromTuple, histFile, nFiles, startFile, nEvts);
    plotter.setLumi(lumi);
    plotter.setPlotDir(plotDir);
    plotter.setDoHists(doSave || doPlots);
    plotter.setDoTuple(false);
    plotter.setRegisterFunction(rf);
    plotter.read();
    if(doSave && fromTuple)  plotter.saveHists();
    if(doPlots)              plotter.plot();

//    if(!runOnCondor) throw "NOT ACTAULLY AN ERROR";
}
