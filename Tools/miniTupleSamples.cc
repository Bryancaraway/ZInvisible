#include "samples.h"

#include <iostream>
#include <cstdio>
#include <cstring>

namespace AnaSamples
{
    void FileSummary::readFileList() const
    {
        if(filelist_.size()) filelist_.clear();
        
        FILE *f = fopen(filePath.c_str(), "r");
        char buff[512];
        if(f)
        {
            while(!feof(f) && fgets(buff, 512, f))
            {
                for(char* k = strchr(buff, '\n'); k != 0; k = strchr(buff, '\n')) *k = '\0';
                filelist_.push_back(buff);
            }
            fclose(f);
        }
        else std::cout << "Filelist file \"" << filePath << "\" not found!!!!!!!" << std::endl;
    }

    void FileSummary::addCollection(std::string colName)
    {
        collections_.insert(colName);
    }

    std::map<std::string, FileSummary>& SampleSet::getMap()
    {
        return sampleSet_;
    }
    
    SampleSet::SampleSet(std::string fDir, float lumi) : fDir_(fDir), lumi_(lumi)
    {
        // ---------------
        // - backgrounds -
        // ---------------

        // branching ratio info from PDG
        float W_Lept_BR = 0.1086*3;
        float TTbar_SingleLept_BR = 0.43930872; // 2*W_Lept_BR*(1-W_Lept_BR)
        float TTbar_DiLept_BR = 0.10614564; // W_Lept_BR^2

//        std::string MCloc = "Spring15_74X_Oct_2015_Ntp_v2X/";
        std::string MCloc = "";
        std::string DATAloc = "";

        if(fDir.compare("condor") == 0)
        {
            fDir_ = "";
            MCloc = "";
            DATAloc = "";
        }


        // Calculated from PDG BRs'. Not from the kt * xSec in McM.
        addSample("TTbarDiLep", fDir_ + MCloc + "minituple_TTbarDiLep.txt",         "TTJets_DiLept_TuneCUETP8M1_13TeV-madgraphMLM-pythia8",             831.76*TTbar_DiLept_BR,         lumi, 30498962, 1.0, kGreen);
        addSample("TTbarSingleLepT", fDir_ + MCloc + "minituple_TTbarSingleLepT.txt",    "TTJets_SingleLeptFromT_TuneCUETP8M1_13TeV-madgraphMLM-pythia8",    831.76*0.5*TTbar_SingleLept_BR, lumi, 60144642, 1.0, kGreen);
        addSample("TTbarSingleLepTbar", fDir_ + MCloc + "minituple_TTbarSingleLepTbar.txt", "TTJets_SingleLeptFromTbar_TuneCUETP8M1_13TeV-madgraphMLM-pythia8", 831.76*0.5*TTbar_SingleLept_BR, lumi, 59816364, 1.0, kGreen);

        //Z -> nunu
        // From https://twiki.cern.ch/twiki/bin/viewauth/CMS/SummaryTable1G25ns#DY_Z, kz = 1.23
        //addSample("ZJetsToNuNu_HT_100to200", fDir_ + MCloc + "minituple_ZJetsToNuNu_HT.txt", "ZJetsToNuNu_HT-100To200_13TeV-madgraph", 280.35, lumi, 5154824,  1.23,  kTeal+4);
        //addSample("ZJetsToNuNu_HT_200to400", fDir_ + MCloc + "minituple_ZJetsToNuNu_HT.txt", "ZJetsToNuNu_HT-200To400_13TeV-madgraph", 77.67,  lumi, 24863552, 1.23,  kTeal+4);
        //addSample("ZJetsToNuNu_HT_400to600", fDir_ + MCloc + "minituple_ZJetsToNuNu_HT.txt", "ZJetsToNuNu_HT-400To600_13TeV-madgraph", 10.73,  lumi, 9591908,  1.23,  kTeal+4);
        //addSample("ZJetsToNuNu_HT_600toInf", fDir_ + MCloc + "minituple_ZJetsToNuNu_HT.txt", "ZJetsToNuNu_HT-600ToInf_13TeV-madgraph", 4.116,  lumi, 10202299, 1.23,  kTeal+4);
        addSample("ZJetsToNuNu_HT_100to200",   fDir_ + MCloc + "minituple-ZJetsToNuNu.txt", "ZJetsToNuNu_HT-100To200_13TeV-madgraph",   280.35,     lumi, 18665303, 1.23,  kTeal+4);
        addSample("ZJetsToNuNu_HT_200to400",   fDir_ + MCloc + "minituple-ZJetsToNuNu.txt", "ZJetsToNuNu_HT-200To400_13TeV-madgraph",   77.67,      lumi, 19914665, 1.23,  kTeal+4);
        addSample("ZJetsToNuNu_HT_400to600",   fDir_ + MCloc + "minituple-ZJetsToNuNu.txt", "ZJetsToNuNu_HT-400To600_13TeV-madgraph",   10.73,      lumi, 9591908,  1.23,  kTeal+4);
        addSample("ZJetsToNuNu_HT_600to800",   fDir_ + MCloc + "minituple-ZJetsToNuNu.txt", "ZJetsToNuNu_HT-600To800_13TeV-madgraph",   0.853*3,    lumi, 5650700,  1.23,  kTeal+4);
        addSample("ZJetsToNuNu_HT_800to1200",  fDir_ + MCloc + "minituple-ZJetsToNuNu.txt", "ZJetsToNuNu_HT-800To1200_13TeV-madgraph",  0.394*3,    lumi, 2156175,  1.23,  kTeal+4);
        addSample("ZJetsToNuNu_HT_1200to2500", fDir_ + MCloc + "minituple-ZJetsToNuNu.txt", "ZJetsToNuNu_HT-1200To2500_13TeV-madgraph", 0.0974*3,   lumi, 365216,   1.23,  kTeal+4);
        addSample("ZJetsToNuNu_HT_2500toInf",  fDir_ + MCloc + "minituple-ZJetsToNuNu.txt", "ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph",  0.00230*3,  lumi, 405752,   1.23,  kTeal+4);

        //DY->ll
        // kz = 1.23
        addSample("DYJetsToLL_HT_100to200", fDir_ + MCloc + "minituple_DYJetsToLL_HT.txt", "DYJetsToLL_M-50_HT-100to200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8", 147.4, lumi, 11079482, 1.23,  kYellow-7);
        addSample("DYJetsToLL_HT_200to400", fDir_ + MCloc + "minituple_DYJetsToLL_HT.txt", "DYJetsToLL_M-50_HT-200to400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8", 40.99, lumi, 10548654, 1.23,  kYellow-7);
        addSample("DYJetsToLL_HT_400to600", fDir_ + MCloc + "minituple_DYJetsToLL_HT.txt", "DYJetsToLL_M-50_HT-400to600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8", 5.678, lumi, 10420186, 1.23,  kYellow-7);
        addSample("DYJetsToLL_HT_600toInf", fDir_ + MCloc + "minituple_DYJetsToLL_HT.txt", "DYJetsToLL_M-50_HT-600toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8", 2.198, lumi, 4631933,  1.23,  kYellow-7);
        // NNLO
        addSample("DYJetsToLL_Inc", fDir_ + MCloc + "minituple_DYJetsToLL_Inc.txt", "DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8", 6025.2, lumi, 9042031, 1.0,  kYellow-7);

        //Other Samples
        // Aprox. NNLO
        addSample("tW_top", fDir_ + MCloc + "minituple_tW.txt", "ST_tW_top_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1", 35.6, lumi,  995600,  1.0,  kYellow);
        // Aprox. NNLO
        addSample("tW_antitop", fDir_ + MCloc + "minituple_tW.txt", "ST_tW_antitop_5f_inclusiveDecays_13TeV-powheg-pythia8_TuneCUETP8M1", 35.6, lumi, 988500,  1.0,  kYellow);

        // NLO --> negative weights!
        // (sign of gen weight) * (lumi*xsec)/(effective number of events): effective number of events = N(evt) with positive weight - N(evt) with negative weight
        addSample("TTZToLLNuNu", fDir_ + MCloc + "minituple_TTZ.txt", "TTZToLLNuNu_M-10_TuneCUETP8M1_13TeV-amcatnlo-pythia8", 0.2529, lumi, 291495 - 106505,  1.0,  kOrange+2);
        addSample("TTZToQQ", fDir_ + MCloc + "minituple_TTZ.txt", "TTZToQQ_TuneCUETP8M1_13TeV-amcatnlo-pythia8",          0.5297, lumi, 550599 - 199201,  1.0,  kOrange+2);

        // NLO --> negative weights!
        addSample("TTWJetsToLNu", fDir_ + MCloc + "minituple_TTWJets.txt", "TTWJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8", 0.2043, lumi, 191379 - 61529,   1.0,  kSpring+2);
        addSample("TTWJetsToQQ", fDir_ + MCloc + "minituple_TTWJets.txt", "TTWJetsToQQ_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8",  0.4062, lumi, 632147 - 201817,  1.0,  kSpring+2);

        // NLO --> negative weights! 
        addSample("TTGJets", fDir_ + MCloc + "minituple_TTGJets.txt", "TTGJets_TuneCUETP8M1_13TeV-amcatnloFXFX-madspin-pythia8", 3.697, lumi,  3199309 - 1632921,  1.0,  kOrange+2);

        // ttH --> negative weights!
	addSample("ttHJetTobb", fDir_ + MCloc + "minituple_ttHJetTobb.txt",    "ttHJetTobb_M125_13TeV_amcatnloFXFX_madspin_pythia8",             0.2934,  lumi, 1047463 - 565141,   1.0,  kOrange+2);
	addSample("ttHJetToNonbb", fDir_ + MCloc + "minituple_ttHJetToNonbb.txt", "ttHJetToNonbb_M125_13TeV_amcatnloFXFX_madspin_pythia8_mWCutfix", 0.2151,  lumi, 5206759 - 2818464,  1.0,  kOrange+2);
        
        // Di-boson
	// Ref. https://indico.cern.ch/event/439995/session/0/contribution/6/attachments/1143460/1638648/diboson_final.pdf (NNLO is given)
        addSample("WW", fDir_ + MCloc + "minituple_WW.txt", "WW_TuneCUETP8M1_13TeV-pythia8", 115.0,  lumi, 993640,  1.0,  kViolet+4); 
	// Ref. https://twiki.cern.ch/twiki/bin/viewauth/CMS/SummaryTable1G25ns (NLO from MCFM)
        addSample("WZ", fDir_ + MCloc + "minituple_WZ.txt", "WZ_TuneCUETP8M1_13TeV-pythia8", 47.13,  lumi, 978512,  1.0,  kViolet+4);
        addSample("ZZ", fDir_ + MCloc + "minituple_ZZ.txt", "ZZ_TuneCUETP8M1_13TeV-pythia8", 16.523, lumi, 996944,  1.0,  kViolet+4);

        // Tri-boson: negative weights!
        addSample("WWZ", fDir_ + MCloc + "minituple_WWZ.txt", "WWZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8", 0.1651,  lumi, 235734 - 14266,  1.0,  kViolet+2);
        addSample("WZZ", fDir_ + MCloc + "minituple_WZZ.txt", "WZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8", 0.05565, lumi, 234584 - 15416,  1.0,  kViolet+2);
        addSample("ZZZ", fDir_ + MCloc + "minituple_ZZZ.txt", "ZZZ_TuneCUETP8M1_13TeV-amcatnlo-pythia8", 0.01398, lumi, 231925 - 18075,  1.0,  kViolet+2);

        // --------
        // - data -
        // --------
	
	addSample("Data_SingleMuon_2015C", fDir_ + DATAloc + "minituple_Data_SingleMuon_2015C.txt",            "Spring15_74X_Jan_2016_Ntp_v5p0_SingleMuon-Run2015C-25ns-05Oct2015", 17.226, 1.0,  kBlack);
	addSample("Data_SingleMuon_2015D_05Oct2015", fDir_ + DATAloc + "minituple_Data_SingleMuon_2015D_05Oct2015.txt",  "Spring15_74X_Jan_2016_Ntp_v5p0_SingleMuon-Run2015D-05Oct2015", 575.34, 1.0,  kBlack);
	addSample("Data_SingleMuon_2015D_PromptReco", fDir_ + DATAloc + "minituple_Data_SingleMuon_2015D_PromptReco.txt", "Spring15_74X_Jan_2016_Ntp_v5p0_SingleMuon-Run2015D-PromptReco", 1670.38, 1.0,  kBlack);
    }

    SampleCollection::SampleCollection(SampleSet& samples)
    {
        //Define sets of samples for later use
        addSampleSet(samples, "TTbarSingleLep", {"TTbarSingleLepT", "TTbarSingleLepTbar"});
        addSampleSet(samples, "TTbarDiLep", {"TTbarDiLep"});
        addSampleSet(samples, "TTbarNoHad", {"TTbarSingleLepT", "TTbarSingleLepTbar", "TTbarDiLep"});

        //addSampleSet(samples, "ZJetsToNuNu", {"ZJetsToNuNu_HT_600toInf", "ZJetsToNuNu_HT_400to600", "ZJetsToNuNu_HT_200to400", "ZJetsToNuNu_HT_100to200"});
        addSampleSet(samples, "ZJetsToNuNu", {"ZJetsToNuNu_HT_100to200", "ZJetsToNuNu_HT_200to400", "ZJetsToNuNu_HT_400to600", "ZJetsToNuNu_HT_600to800", "ZJetsToNuNu_HT_800to1200", "ZJetsToNuNu_HT_1200to2500", "ZJetsToNuNu_HT_2500toInf"});

        addSampleSet(samples, "DYJetsToLL", {"DYJetsToLL_HT_600toInf", "DYJetsToLL_HT_400to600", "DYJetsToLL_HT_200to400", "DYJetsToLL_HT_100to200"});
        addSampleSet(samples, "IncDY", {"DYJetsToLL_Inc"});

        addSampleSet(samples, "tW", {"tW_top", "tW_antitop"});

        addSampleSet(samples, "TTZ", {"TTZToLLNuNu", "TTZToQQ"});
        addSampleSet(samples, "TTW", {"TTWJetsToLNu", "TTWJetsToQQ"});

        addSampleSet(samples, "TTG", {"TTGJets"});

	addSampleSet(samples, "ttH", {"ttHJetTobb", "ttHJetToNonbb"});

        addSampleSet(samples, "WWZ", {"WWZ"});
        addSampleSet(samples, "WZZ", {"WZZ"});
        addSampleSet(samples, "ZZZ", {"ZZZ"});

        addSampleSet(samples, "Diboson", {"WW", "WZ", "ZZ"});

        addSampleSet(samples, "Triboson", {"WWZ", "WZZ", "ZZZ"});

	addSampleSet(samples, "Rare", {"TTWJetsToLNu", "TTWJetsToQQ", "TTGJets", "WWZ", "WZZ", "ZZZ", "ttHJetTobb", "ttHJetToNonbb"});

        addSampleSet(samples, "Data_SingleMuon", {"Data_SingleMuon_2015C", "Data_SingleMuon_2015D_05Oct2015", "Data_SingleMuon_2015D_PromptReco"});

        addSampleSet(samples, "ZinvllAll", {"Data_SingleMuon_2015C", "Data_SingleMuon_2015D_05Oct2015", "Data_SingleMuon_2015D_PromptReco", "TTbarSingleLepT", "TTbarSingleLepTbar", "TTbarDiLep", "DYJetsToLL_HT_600toInf", "DYJetsToLL_HT_400to600", "DYJetsToLL_HT_200to400", "DYJetsToLL_HT_100to200", "DYJetsToLL_Inc", "tW_top", "tW_antitop", "WW", "WZ", "ZZ", "TTWJetsToLNu", "TTWJetsToQQ", "TTGJets", "WWZ", "WZZ", "ZZZ", "ttHJetTobb", "ttHJetToNonbb", "TTZToLLNuNu", "TTZToQQ"});

    }

    void SampleCollection::addSampleSet(SampleSet& samples, std::string name, std::vector<std::string> vss)
    {
        if(vss.size() > 1)
        {
            for(std::string& sn : vss)
            {
                if(sn.compare(name) == 0)
                {
                    std::cout << "You have named a sampleCollection the same as one of its member sampleSets, but it has more than one sampleSet!!!! This is bad!!!  Stop!!! Stop now!!!  This collection will be skipped until it is properly named." << std::endl;
                    return;
                }
            }
        }

        auto& map = samples.getMap();

        for(std::string& sn : vss)
        {
            map[sn].addCollection(name);
            sampleSet_[name].push_back(samples[sn]);
            nameVec_[name].push_back(sn);
            totalLumiMap_[name] += samples[sn].lumi;
        }
    }

    std::vector<std::string>& SampleCollection::getSampleLabels(std::string name)
    {
        return nameVec_[name];
    }

    bool operator< (const FileSummary& lhs, const FileSummary& rhs)
    {
        return lhs.filePath < rhs.filePath || lhs.treePath < rhs.treePath;
    }

    bool operator== (const FileSummary& lhs, const FileSummary& rhs)
    {
        return lhs.filePath == rhs.filePath && lhs.treePath == rhs.treePath && lhs.xsec == rhs.xsec && lhs.lumi == rhs.lumi && lhs.kfactor == rhs.kfactor && lhs.nEvts == rhs.nEvts;
    }

    bool operator!= (const FileSummary& lhs, const FileSummary& rhs)
    {
        return !(lhs == rhs);
    }
}
