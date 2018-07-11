#include "Plotter.h"
#include "RegisterFunctions.h"
#include "SusyAnaTools/Tools/samples.h"
#include "Systematic.h"

#include <getopt.h>
#include <iostream>

int main(int argc, char* argv[])
{
    using namespace std;

    TH1::AddDirectory(false);

    int opt;
    int option_index = 0;
    static struct option long_options[] = {
        {"plot",             no_argument, 0, 'p'},
        {"dataSets",   required_argument, 0, 'D'},
	{"luminosity", required_argument, 0, 'L'}
    };

    string dataSets = "ZJetsToNuNu";
    float lumi = AnaSamples::luminosity;

    while((opt = getopt_long(argc, argv, "pD:L:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
        case 'p':
            break;

        case 'D':
            dataSets = optarg;
            break;

	case 'L':
            lumi = atof(optarg);
            break;
        }
    }

    AnaSamples::SampleSet        ss("/uscms_data/d3/snorberg/CMSSW_8_0_23_patch1/src/ZInvisible/Tools/", lumi);
    AnaSamples::SampleCollection sc(ss);

    map<string, vector<AnaSamples::FileSummary>> fileMap;

    //Select approperiate datasets here
    if(dataSets.compare("TEST") == 0)
    {
        fileMap["DYJetsToLL"]  = {sc["DYJetsToLL"]};
        fileMap["ZJetsToNuNu"] = {sc["ZJetsToNuNu"]};
        fileMap["TTbarDiLep"] = {ss["TTbarDiLep"]};
    }
    else if(dataSets.compare("TEST2") == 0)
    {
        fileMap["DYJetsToLL"]  = {ss["DYJetsToLL_HT_400to600"]};
        fileMap["ZJetsToNuNu"] = {ss["ZJetsToNuNu_HT_400to600"]};
        fileMap["DYJetsToLL_HT_400to600"] = {ss["DYJetsToLL_HT_400to600"]};
        fileMap["ZJetsToNuNu_HT_400to600"] = {ss["ZJetsToNuNu_HT_400to600"]};
    }
    else
    {
        if(sc[dataSets] != sc.null())
        {
            fileMap[dataSets] = {sc[dataSets]};
            int i = 0;
            for(const auto& fs : sc[dataSets])
            {
                fileMap[sc.getSampleLabels(dataSets)[i++]].push_back(fs);
            }
        }
        else if(ss[dataSets] != ss.null())
        {
            fileMap[dataSets] = {ss[dataSets]};
            for(const auto& colls : ss[dataSets].getCollections())
            {
                fileMap[colls] = {ss[dataSets]};
            }
        }
        
    }

    std::cout << dataSets << "\t" << fileMap.size() <<std::endl;

    // for(auto& vfs : fileMap)
    // {
    //     for(auto& fs : vfs.second)
    //     {
    //         size_t start = fs.filePath.rfind('/');
    //         size_t stop  = fs.filePath.rfind('.');
    //         if(int(stop) - (int(start) + 1) > 0)
    //         {
    //             fs.treePath = fs.filePath.substr(start + 1, stop - start - 1);
    //             fs.filePath = "stupidFileList.txt";
    //             fs.readFileList();
    //         }
    //     }
    // }

    TH1 *shapeMET, *shapeMT2, *shapeNT, *shapeNB;
    
    TFile *f = new TFile("syst_shape.root");
    if(f)
    {
        shapeMET = static_cast<TH1*>(f->Get("ShapeRatio_met")->Clone());
        shapeMT2 = static_cast<TH1*>(f->Get("ShapeRatio_mt2")->Clone());
        shapeNT  = static_cast<TH1*>(f->Get("ShapeRatio_nt")->Clone());
        shapeNB  = static_cast<TH1*>(f->Get("ShapeRatio_nb")->Clone());
        f->Close();
        delete f;
    }
    else
    {
        std::cout << "Failed to open: syst_shape.root" << std::endl;
    }

    TH1 *shapeMETGaus, *shapeMT2Gaus, *shapeMETLogi, *shapeMT2Logi;
    TH2 *shapeMT2vMETGaus, *shapeMT2vMETLogi;
   //Used for some correlation 
    TFile *f2 = new TFile("correlations_ratio.root");
    if(f2)
    {
        shapeMETGaus = static_cast<TH1*>(f2->Get("Ratio_MET_Gaus")->Clone());
        shapeMT2Gaus = static_cast<TH1*>(f2->Get("Ratio_MT2_Gaus")->Clone());
        shapeMT2vMETGaus = static_cast<TH2*>(f2->Get("Ratio_MT2vMET_Gaus")->Clone());
        shapeMETLogi  = static_cast<TH1*>(f2->Get("Ratio_MET_Logi")->Clone());
        shapeMT2Logi  = static_cast<TH1*>(f2->Get("Ratio_MT2_Logi")->Clone());
        shapeMT2vMETLogi  = static_cast<TH2*>(f2->Get("Ratio_MT2vMET_Logi")->Clone());
        f2->Close();
        delete f2;
    }
    else
    {
        std::cout << "Failed to open: correlations_ratio.root" << std::endl;
    }
    

    RegisterFunctions *rf = new RegisterFunctionsSyst;

    vector<Plotter::HistSummary> vh;

    Plotter::DatasetSummary dsDY_mm( "DY#rightarrow#mu#mu", fileMap["DYJetsToLL"],  "", "");
    Plotter::DatasetSummary dsDY_ee( "DY#rightarrowee",     fileMap["DYJetsToLL"],  "", "");
    Plotter::DatasetSummary dsZ_nunu("Z#rightarrow#nu#nu",  fileMap["ZJetsToNuNu"], "", "");

    Plotter::DataCollection dcDY_mm_cleamMET( "single", "cleanMetPt", {dsDY_mm});
    Plotter::DataCollection dcDY_ee_cleamMET( "single", "cleanMetPt", {dsDY_ee});
    Plotter::DataCollection dcZ_nunu_cleamMET("single", "cleanMetPt", {dsZ_nunu});

    vh.push_back(PHS("cleanMet",     {dcDY_mm_cleamMET, dcDY_ee_cleamMET, dcZ_nunu_cleamMET}, {1, 2}, "",            100, 0, 1500,  true,  true,  "MET [GeV]",  "Norm Events"));
    vh.push_back(PHS("cleanMet_cut", {dcDY_mm_cleamMET, dcDY_ee_cleamMET, dcZ_nunu_cleamMET}, {1, 2}, "passMETZinv", 100, 0, 1500,  true,  true,  "MET [GeV]",  "Norm Events"));

    TF1 *fit = new TF1("fit","pol1");
    fit->SetParameters(0.98, 0.001);
    Systematic test("systWgtTest", "cleanMetPt", fit);
    test.bookHist(vh, fileMap["ZJetsToNuNu"]);
    static_cast<RegisterFunctionsSyst*>(rf)->addFunction(std::bind(test, std::placeholders::_1));

    //Met shape syst 
    Systematic METSyst("systWgtMET", "cleanMetPt", shapeMET);
    METSyst.bookHist(vh, fileMap["ZJetsToNuNu"]);
    static_cast<RegisterFunctionsSyst*>(rf)->addFunction(std::bind(METSyst, std::placeholders::_1));

    //MT2 shape syst 
    Systematic MT2Syst("systWgtMT2", "best_had_brJet_MT2Zinv", shapeMT2);
    MT2Syst.bookHist(vh, fileMap["ZJetsToNuNu"]);
    static_cast<RegisterFunctionsSyst*>(rf)->addFunction(std::bind(MT2Syst, std::placeholders::_1));

    //NT shape uncertainty
    Systematic NTSyst("systWgtNT", "nTopCandSortedCntZinv", shapeNT);
    NTSyst.bookHist(vh, fileMap["ZJetsToNuNu"]);
    static_cast<RegisterFunctionsSyst*>(rf)->addFunction(std::bind(NTSyst, std::placeholders::_1));

    //NB shape uncertainty
    Systematic NBSyst("systWgtNB", "cntCSVSZinv", shapeNB);
    NBSyst.bookHist(vh, fileMap["ZJetsToNuNu"]);
    static_cast<RegisterFunctionsSyst*>(rf)->addFunction(std::bind(NBSyst, std::placeholders::_1));

    //MET Gaus correlation study uncertainty
    Systematic CorrMETGaus("CorrMETGaus", "cleanMetPt", shapeMETGaus);
    CorrMETGaus.bookHist(vh, fileMap["ZJetsToNuNu"]);
    static_cast<RegisterFunctionsSyst*>(rf)->addFunction(std::bind(CorrMETGaus, std::placeholders::_1));

    //MT2 Gaus correlation study uncertainty
    Systematic CorrMT2Gaus("CorrMT2Gaus", "best_had_brJet_MT2Zinv", shapeMT2Gaus);
    CorrMT2Gaus.bookHist(vh, fileMap["ZJetsToNuNu"]);
    static_cast<RegisterFunctionsSyst*>(rf)->addFunction(std::bind(CorrMT2Gaus, std::placeholders::_1));

    //MT2vMET Gaus correlation study uncertainty
    Systematic CorrMT2vMETGaus("CorrMT2vMETGaus", "cleanMetPt", "best_had_brJet_MT2Zinv", shapeMT2vMETGaus);
    CorrMT2vMETGaus.bookHist(vh, fileMap["ZJetsToNuNu"]);
    static_cast<RegisterFunctionsSyst*>(rf)->addFunction(std::bind(CorrMT2vMETGaus, std::placeholders::_1));

    //MET Logi correlation study uncertainty
    Systematic CorrMETLogi("CorrMETLogi", "cleanMetPt", shapeMETLogi);
    CorrMETLogi.bookHist(vh, fileMap["ZJetsToNuNu"]);
    static_cast<RegisterFunctionsSyst*>(rf)->addFunction(std::bind(CorrMETLogi, std::placeholders::_1));

    //MT2 Logi correlation study uncertainty
    Systematic CorrMT2Logi("CorrMT2Logi", "best_had_brJet_MT2Zinv", shapeMT2Logi);
    CorrMT2Logi.bookHist(vh, fileMap["ZJetsToNuNu"]);
    static_cast<RegisterFunctionsSyst*>(rf)->addFunction(std::bind(CorrMT2Logi, std::placeholders::_1));

    //MT2vMET Logi correlation study uncertainty
    Systematic CorrMT2vMETLogi("CorrMT2vMETLogi", "cleanMetPt", "best_had_brJet_MT2Zinv", shapeMT2vMETLogi);
    CorrMT2vMETLogi.bookHist(vh, fileMap["ZJetsToNuNu"]);
    static_cast<RegisterFunctionsSyst*>(rf)->addFunction(std::bind(CorrMT2vMETLogi, std::placeholders::_1));

    Plotter::DataCollection dcDY_nunu_nJetWgtSyst( "single", {{"njSystWeightedSB", dsZ_nunu}, {"njSystUnweightedSB", dsZ_nunu}});
    vh.emplace_back(PHS("systNJetWgtStat", {dcDY_nunu_nJetWgtSyst}, {2, 1}, "", 45, 0, 45, true, true, "Search Bin", "Events"));

    set<AnaSamples::FileSummary> vvf;
    for(auto& fsVec : fileMap) for(auto& fs : fsVec.second) vvf.insert(fs);

    Plotter plotter(vh, vvf, true, "systematics.root");
    plotter.setLumi(lumi);
    plotter.setPlotDir("Syst_plots");
    plotter.setDoHists(true);
    plotter.setDoTuple(false);
    plotter.setPrintInterval(10000);
    plotter.setRegisterFunction(rf);
    plotter.read();
    plotter.saveHists();
    plotter.plot();
}
