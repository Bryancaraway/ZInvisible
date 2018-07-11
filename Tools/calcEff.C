#include "SusyAnaTools/Tools/samples.h"
#include "RegisterFunctions.h"
#include "NTupleReader.h"

#include <iostream>
#include <getopt.h>

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TRandom3.h"
#include "TChain.h"
#include "Math/VectorUtil.h"

int main(int argc, char* argv[])
{
    using namespace std;

    int opt;
    int option_index = 0;
    static struct option long_options[] = {
        {"condor",           no_argument, 0, 'c'},
        {"dataSets",   required_argument, 0, 'D'},
        {"numFiles",   required_argument, 0, 'N'},
        {"startFile",  required_argument, 0, 'M'},
        {"numEvts",    required_argument, 0, 'E'},
    };

    bool runOnCondor = false;
    int nFiles = -1, startFile = 0, nEvts = -1;
    string dataSets = "";

    while((opt = getopt_long(argc, argv, "cD:N:M:E:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
        case 'c':
            runOnCondor = true;
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
        }
    }

    string filename = "effhists.root";
    string sampleloc = AnaSamples::fileDir;

    //if running on condor override all optional settings
    if(runOnCondor)
    {
        char thistFile[128];
        sprintf(thistFile, "effhists_%s_%d.root", dataSets.c_str(), startFile);
        filename = thistFile;
        sampleloc = "condor";
    }

  
    // the old version
    //AnaSamples::SampleSet        ss(sampleloc, lumi);
    //AnaSamples::SampleCollection sc(ss);
    // the new version
    AnaSamples::SampleSet        ss("sampleSets.txt");
    AnaSamples::SampleCollection sc("sampleCollections.txt", ss);


    TFile *f = new TFile(filename.c_str(),"RECREATE");
    f->cd();

    TH1 *hMuEffPt_num = new TH1D("hMuEffPt_num", "hMuEffPt_num", 200, 0, 2000);
    TH1 *hMuEffPt_den = new TH1D("hMuEffPt_den", "hMuEffPt_den", 200, 0, 2000);
    TH1 *hMuAccPt_num = new TH1D("hMuAccPt_num", "hMuAccPt_num", 200, 0, 2000);
    TH1 *hMuAccPt_den = new TH1D("hMuAccPt_den", "hMuAccPt_den", 200, 0, 2000);

    TH1 *hMuEffHt_num = new TH1D("hMuEffHt_num", "hMuEffHt_num", 300, 0, 3000);
    TH1 *hMuEffHt_den = new TH1D("hMuEffHt_den", "hMuEffHt_den", 300, 0, 3000);
    TH1 *hMuAccHt_num = new TH1D("hMuAccHt_num", "hMuAccHt_num", 300, 0, 3000);
    TH1 *hMuAccHt_den = new TH1D("hMuAccHt_den", "hMuAccHt_den", 300, 0, 3000);

    TH2 *hMuEff_num = new TH2D("hMuEff_num", "hMuEff_num", 200, 0, 2000, 300, 0, 3000);
    TH2 *hMuEff_num_pp = new TH2D("hMuEff_num_pp", "hMuEff_num_pp", 200, 0, 2000, 300, 0, 3000);
    TH2 *hMuEff_num_pf = new TH2D("hMuEff_num_pf", "hMuEff_num_pf", 200, 0, 2000, 300, 0, 3000);
    TH2 *hMuEff_num_fp = new TH2D("hMuEff_num_fp", "hMuEff_num_fp", 200, 0, 2000, 300, 0, 3000);

    TH2 *hMuEffPtActReco_num = new TH2D("hMuEffPtActReco_num", "hMuEffPtActReco_num", 200, 0, 2000, 1000, 0, 10);
    TH2 *hMuEffPtActReco_den = new TH2D("hMuEffPtActReco_den", "hMuEffPtActReco_den", 200, 0, 2000, 1000, 0, 10);
    TH2 *hMuEffPtActIso_num = new TH2D("hMuEffPtActIso_num", "hMuEffPtActIso_num",    200, 0, 2000, 1000, 0, 10);
    TH2 *hMuEffPtActIso_den = new TH2D("hMuEffPtActIso_den", "hMuEffPtActIso_den",    200, 0, 2000, 1000, 0, 10);

    TH2 *hElecEffPtActReco_num = new TH2D("hElecEffPtActReco_num", "hElecEffPtActReco_num", 200, 0, 2000, 1000, 0, 10);
    TH2 *hElecEffPtActReco_den = new TH2D("hElecEffPtActReco_den", "hElecEffPtActReco_den", 200, 0, 2000, 1000, 0, 10);
    TH2 *hElecEffPtActIso_num =  new TH2D("hElecEffPtActIso_num",  "hElecEffPtActIso_num",  200, 0, 2000, 1000, 0, 10);
    TH2 *hElecEffPtActIso_den =  new TH2D("hElecEffPtActIso_den",  "hElecEffPtActIso_den",  200, 0, 2000, 1000, 0, 10);

    TH1 *hMuEffPtReco_num = new TH1D("hMuEffPtReco_num", "hMuEffPtActReco_num", 200, 0, 2000);
    TH1 *hMuEffPtReco_den = new TH1D("hMuEffPtReco_den", "hMuEffPtActReco_den", 200, 0, 2000);
    TH1 *hMuEffPtIso_num = new TH1D("hMuEffPtIso_num", "hMuEffPtActIso_num", 200, 0, 2000);
    TH1 *hMuEffPtIso_den = new TH1D("hMuEffPtIso_den", "hMuEffPtActIso_den", 200, 0, 2000);

    TH1 *hElecEffPtReco_num = new TH1D("hElecEffPtReco_num", "hElecEffPtActReco_num", 200, 0, 2000);
    TH1 *hElecEffPtReco_den = new TH1D("hElecEffPtReco_den", "hElecEffPtActReco_den", 200, 0, 2000);
    TH1 *hElecEffPtIso_num =  new TH1D("hElecEffPtIso_num",  "hElecEffPtActIso_num", 200, 0, 2000);
    TH1 *hElecEffPtIso_den =  new TH1D("hElecEffPtIso_den",  "hElecEffPtActIso_den", 200, 0, 2000);

    TH2 *hMuEff_num_rand = new TH2D("hMuEff_num_rand", "hMuEff_num_rand", 200, 0, 2000, 300, 0, 3000);

    TH2 *hMuEff_den = new TH2D("hMuEff_den", "hMuEff_den", 200, 0, 2000, 300, 0, 3000);
    TH2 *hMuAcc_num = new TH2D("hMuAcc_num", "hMuAcc_num", 200, 0, 2000, 300, 0, 3000);
    TH2 *hMuAcc_den = new TH2D("hMuAcc_den", "hMuAcc_den", 200, 0, 2000, 300, 0, 3000);

    TH1 *hZEffPt_num = new TH1D("hZEffPt_num", "hZEffPt_num", 200, 0, 2000);
    TH1 *hZEffPt_den = new TH1D("hZEffPt_den", "hZEffPt_den", 200, 0, 2000);
    TH1 *hZAccPt_num = new TH1D("hZAccPt_num", "hZAccPt_num", 200, 0, 2000);
    TH1 *hZAccPt_den = new TH1D("hZAccPt_den", "hZAccPt_den", 200, 0, 2000);
    TH1 *hZAccPtSmear_num = new TH1D("hZAccPtSmear_num", "hZAccPtSmear_num", 200, 0, 2000);
    TH1 *hZAccPtSmear_den = new TH1D("hZAccPtSmear_den", "hZAccPtSmear_den", 200, 0, 2000);
    TH1 *hZAccPtPtSmear_num = new TH1D("hZAccPtMuPtSmear_num", "hZAccPtMuPtSmear_num", 200, 0, 2000);
    TH1 *hZAccPtPtSmear_den = new TH1D("hZAccPtMuPtSmear_den", "hZAccPtMuPtSmear_den", 200, 0, 2000);

    TH1 *hZElecAccPt_num = new TH1D("hZElecAccPt_num", "hZElecAccPt_num", 200, 0, 2000);
    TH1 *hZElecAccPt_den = new TH1D("hZElecAccPt_den", "hZElecAccPt_den", 200, 0, 2000);
    TH1 *hZElecAccPtSmear_num = new TH1D("hZElecAccPtSmear_num", "hZElecAccPtSmear_num", 200, 0, 2000);
    TH1 *hZElecAccPtSmear_den = new TH1D("hZElecAccPtSmear_den", "hZElecAccPtSmear_den", 200, 0, 2000);
    TH1 *hZElecAccPtPtSmear_num = new TH1D("hZElecAccPtPtSmear_num", "hZElecAccPtPtSmear_num", 200, 0, 2000);
    TH1 *hZElecAccPtPtSmear_den = new TH1D("hZElecAccPtPtSmear_den", "hZElecAccPtPtSmear_den", 200, 0, 2000);

    TH2 *hZEff_num = new TH2D("hZEff_num", "hZEff_num", 200, 0, 2000, 300, 0, 3000);
    TH2 *hZEff_den = new TH2D("hZEff_den", "hZEff_den", 200, 0, 2000, 300, 0, 3000);
    TH2 *hZAcc_num = new TH2D("hZAcc_num", "hZAcc_num", 200, 0, 2000, 300, 0, 3000);
    TH2 *hZAcc_den = new TH2D("hZAcc_den", "hZAcc_den", 200, 0, 2000, 300, 0, 3000);

    //TH2 *hZEff_jActR1_num = new TH2D("hZEff_jActR1_num", "hZEff_jActR1_num", 200, 0, 2000, 500, 0, 5000);
    //TH2 *hZEff_jActR1_den = new TH2D("hZEff_jActR1_den", "hZEff_jActR1_den", 200, 0, 2000, 500, 0, 5000);
    //TH2 *hZEff_jActR2_num = new TH2D("hZEff_jActR2_num", "hZEff_jActR2_num", 200, 0, 2000, 500, 0, 5000);
    //TH2 *hZEff_jActR2_den = new TH2D("hZEff_jActR2_den", "hZEff_jActR2_den", 200, 0, 2000, 500, 0, 5000);

    TH2 *hdPhi1 = new TH2D("hdPhi1", "hdPhi1", 200, 0, 2000, 500, 0, 5000);
    TH2 *hdPhi2 = new TH2D("hdPhi2", "hdPhi2", 200, 0, 2000, 500, 0, 5000);
    TH2 *hdPhi3 = new TH2D("hdPhi3", "hdPhi3", 200, 0, 2000, 500, 0, 5000);

    RegisterFunctionsCalcEff rt;

    std::set<std::string> activeBranches;
    rt.activateBranches(activeBranches);

    TRandom3 *trg = new TRandom3(12321);
    TFile * fZRes = new TFile("zRes.root");
    TH1* hZRes = (TH1*)fZRes->Get("zRes");
    float hZRes_int = hZRes->Integral(hZRes->FindBin(-0.3), hZRes->FindBin(0.3));

    //for(auto& file : sc["DYJetsToLL"]) 
    auto& file = ss[dataSets];
    {
        TChain *t = new TChain(file.treePath.c_str());

        //for(const auto& fn : file.getFilelist()) t->Add(fn.c_str());
        file.addFilesToChain(t, startFile, nFiles);
        //t->Add("root://cmsxrootd-site.fnal.gov//store/user/lpcsusyhad/PHYS14_720_Mar14_2014_v2/pastika/DYJetsToLL_M-50_HT-600toInf_Tune4C_13TeV-madgraph-tauola/PHYS14_PU20bx25_PHYS14_25_V1-FLAT/150328_003328/0000/stopFlatNtuples_15.root");

        std::cout << "Processing file(s): " << file.filePath << std::endl;

        NTupleReader tr(t, activeBranches);

        rt.registerFunctions(tr);

        while(tr.getNextEvent())
        {
            const std::vector<const TLorentzVector*>& genMatchIsoMuInAcc = tr.getVec<const TLorentzVector*>("genMatchIsoMuInAcc");
            const std::vector<const TLorentzVector*>& genMatchMuInAcc    = tr.getVec<const TLorentzVector*>("genMatchMuInAcc");
            const std::vector<const TLorentzVector*>& genMuInAcc         = tr.getVec<const TLorentzVector*>("genMuInAcc");
            const std::vector<const TLorentzVector*>& genMu              = tr.getVec<const TLorentzVector*>("genMu");
            const std::vector<float>& genMuAct                = tr.getVec<float>("genMuAct");
            const std::vector<float>& genMuInAccAct           = tr.getVec<float>("genMuInAccAct");
            const std::vector<float>& genMatchMuInAccAct      = tr.getVec<float>("genMatchMuInAccAct");
            const std::vector<float>& genMatchIsoMuInAccAct   = tr.getVec<float>("genMatchIsoMuInAccAct");

            const std::vector<const TLorentzVector*>& genMatchIsoElecInAcc = tr.getVec<const TLorentzVector*>("genMatchIsoElecInAcc");
            const std::vector<const TLorentzVector*>& genMatchElecInAcc    = tr.getVec<const TLorentzVector*>("genMatchElecInAcc");
            const std::vector<const TLorentzVector*>& genElecInAcc         = tr.getVec<const TLorentzVector*>("genElecInAcc");
            const std::vector<const TLorentzVector*>& genElec              = tr.getVec<const TLorentzVector*>("genElec");
            const std::vector<float>& genElecAct                = tr.getVec<float>("genElecAct");
            const std::vector<float>& genElecInAccAct           = tr.getVec<float>("genElecInAccAct");
            const std::vector<float>& genMatchElecInAccAct      = tr.getVec<float>("genMatchElecInAccAct");
            const std::vector<float>& genMatchIsoElecInAccAct   = tr.getVec<float>("genMatchIsoElecInAccAct");

            //const bool& passZinvBaselineNoTag = tr.getVar<bool>("passZinvBaselineNoTag");
            //const bool& passMuZinvSel = tr.getVar<bool>("passMuZinvSel");

            const float& recoZPt    = tr.getVar<float>("bestRecoZPt");
            const float& genZPt     = tr.getVar<float>("genZPt");
            const float& genZM      = tr.getVar<float>("genZmass");
            const float& cleanHt    = tr.getVar<float>("ht");
            const float& cleanMetPt = tr.getVar<float>("cleanMetPt");
            const int&    pdgIdZDec  = tr.getVar<int>("pdgIdZDec");

            if(pdgIdZDec == 13)
            {

                for(auto& tlv : genMu)
                {
                    hMuAccPt_den->Fill(tlv->Pt(), file.getWeight());
                    hMuAccHt_den->Fill(cleanHt,   file.getWeight());

                    hMuAcc_den->Fill(tlv->Pt(), cleanHt, file.getWeight());
                }

                int count = 0, random = 0;//trg->Integer(400000000) & 1;
                bool oneMatch = false, twoMatch = false;

                float modHt = cleanHt;

                for(auto& tlv : genMuInAcc)
                {
                    if(tlv->Pt() > 50) modHt -= tlv->Pt();
                }

                //const std::vector<float>& jActR1 = tr.getVec<float>("jActR1");
                //const std::vector<float>& jActR2 = tr.getVec<float>("jActR2");

                for(auto& tlv : genMuInAcc)
                {
                    hMuAccPt_num->Fill(tlv->Pt(), file.getWeight());
                    hMuAccHt_num->Fill(cleanHt,   file.getWeight());

                    hMuAcc_num->Fill(tlv->Pt(), cleanHt, file.getWeight());

                    hMuEffPt_den->Fill(tlv->Pt(), file.getWeight());
                    hMuEffHt_den->Fill(cleanHt,   file.getWeight());

                    hMuEff_den->Fill(tlv->Pt(), cleanHt, file.getWeight());

                    //hZEff_jActR1_den->Fill(tlv->Pt(), jActR1[count], file.getWeight());
                    //hZEff_jActR2_den->Fill(tlv->Pt(), jActR2[count], file.getWeight());

                    for(auto& tlv2 : genMatchMuInAcc)
                    {
                        if(     count == 0 && tlv == tlv2) oneMatch = true;
                        else if(count == 1 && tlv == tlv2) twoMatch = true;

                        if(count == random && tlv == tlv2) hMuEff_num_rand->Fill(tlv->Pt(), cleanHt, file.getWeight());

                        if(tlv == tlv2)
                        {
                            //hZEff_jActR1_num->Fill(tlv->Pt(), jActR1[count], file.getWeight());
                            //hZEff_jActR2_num->Fill(tlv->Pt(), jActR2[count], file.getWeight());
                        }
                    }
                    count++;
                }

                for(auto& tlv : genMatchMuInAcc)
                {
                    hMuEffPt_num->Fill(tlv->Pt(), file.getWeight());
                    hMuEffHt_num->Fill(cleanHt,   file.getWeight());

                    hMuEff_num->Fill(tlv->Pt(), cleanHt, file.getWeight());

                    if(      oneMatch && twoMatch) hMuEff_num_pp->Fill(tlv->Pt(), cleanHt, file.getWeight());
                    else if(!oneMatch && twoMatch) hMuEff_num_fp->Fill(tlv->Pt(), cleanHt, file.getWeight());
                    else if(oneMatch && !twoMatch) hMuEff_num_pf->Fill(tlv->Pt(), cleanHt, file.getWeight());
                }

                if(true)//passMuZinvSel)
                {
                    for(int i = 0; i < genMuInAcc.size(); ++i)
                    {
                        hMuEffPtActReco_den->Fill(genMuInAcc[i]->Pt(), genMuInAccAct[i]);

                        hMuEffPtReco_den->Fill(genMuInAcc[i]->Pt());
                    }
                    for(int i = 0; i < genMatchMuInAcc.size(); ++i)
                    {
                        hMuEffPtActReco_num->Fill(genMatchMuInAcc[i]->Pt(), genMatchMuInAccAct[i]);

                        hMuEffPtActIso_den->Fill(genMatchMuInAcc[i]->Pt(), genMatchMuInAccAct[i]);


                        hMuEffPtReco_num->Fill(genMatchMuInAcc[i]->Pt());
                    
                        hMuEffPtIso_den->Fill(genMatchMuInAcc[i]->Pt());
                    }
                    for(int i = 0; i < genMatchIsoMuInAcc.size(); ++i)
                    {
                        hMuEffPtActIso_num->Fill(genMatchIsoMuInAcc[i]->Pt(), genMatchIsoMuInAccAct[i]);

                        hMuEffPtIso_num->Fill(genMatchIsoMuInAcc[i]->Pt());
                    }
                }


                if(true)//passMuZinvSel && passZinvBaselineNoTag)
                {
                    for(int i = hZRes->FindBin(-0.3); i <= hZRes->FindBin(0.3); ++i)
                    {
                        hZAccPtSmear_den->Fill(genZPt*(1+hZRes->GetBinCenter(i)), hZRes->GetBinContent(i)/hZRes_int);
                    }
                    hZAccPt_den->Fill(genZPt, file.getWeight());
                    hZAcc_den->Fill(genZPt, modHt, file.getWeight());
                    if(genMuInAcc.size() >= 2)// && genMuInAcc[0]->Pt() > 45 && genMuInAcc[1]->Pt() > 20 && genZM > 71 && genZM < 111)
                    {
                        // muon iso cut 
                        float muDeltaR = ROOT::Math::VectorUtil::DeltaR(*genMuInAcc[0], *genMuInAcc[1]);
                        float minMuPt = std::min(genMuInAcc[0]->Pt(), genMuInAcc[1]->Pt());
                        float mudRMin = 10.0/minMuPt;
                        if(minMuPt < 50)       mudRMin = 0.2;
                        else if(minMuPt > 200) mudRMin = 0.05;
                        //if(cleanMetPt > 1000) std::cout << muDeltaR << " > " << mudRMin << "\t" << genMuInAcc.size() << std::endl;
                        if(true)//muDeltaR > mudRMin)
                        {
                            for(int i = hZRes->FindBin(-0.3); i <= hZRes->FindBin(0.3); ++i)
                            {
                                hZAccPtSmear_num->Fill(genZPt*(1+hZRes->GetBinCenter(i)), hZRes->GetBinContent(i)/hZRes_int);

                                hZAccPtPtSmear_den->Fill(genZPt*(1+hZRes->GetBinCenter(i)), hZRes->GetBinContent(i)/hZRes_int);
                                if((genMuInAcc[0]->Pt() > 45 && genMuInAcc[1]->Pt() > 20) || (genMuInAcc[1]->Pt() > 45 && genMuInAcc[0]->Pt() > 20)) 
                                {
                                    hZAccPtPtSmear_num->Fill(genZPt*(1+hZRes->GetBinCenter(i)), hZRes->GetBinContent(i)/hZRes_int);
                                }
                            }
                            hZAccPt_num->Fill(genZPt, file.getWeight());
                            hZAcc_num->Fill(genZPt, modHt, file.getWeight());
                
                            hZEffPt_den->Fill(genZPt, file.getWeight());
                            hZEff_den->Fill(genZPt, modHt, file.getWeight());
                            if(genMuInAcc[0]->Pt() > 45 && genMuInAcc[1]->Pt() > 20 && genZM > 71 && genZM < 111)//passMuZinvSel && passZinvBaselineNoTag)//genMatchMuInAcc.size() >= 2)
                            {
                                hZEffPt_num->Fill(genZPt, file.getWeight());
                                hZEff_num->Fill(genZPt, modHt, file.getWeight());
                            }
                        }
                    }
                }
            }

            if(pdgIdZDec == 11)
            {
                //Electrons 

                float modHt = cleanHt;

                for(auto& tlv : genElecInAcc)
                {
                    if(tlv->Pt() > 50) modHt -= tlv->Pt();
                }

                for(int i = 0; i < genElecInAcc.size(); ++i)
                {
                    hElecEffPtActReco_den->Fill(genElecInAcc[i]->Pt(), genElecInAccAct[i]);

                    hElecEffPtReco_den->Fill(genElecInAcc[i]->Pt());
                }
                for(int i = 0; i < genMatchElecInAcc.size(); ++i)
                {
                    hElecEffPtActReco_num->Fill(genMatchElecInAcc[i]->Pt(), genMatchElecInAccAct[i]);

                    hElecEffPtActIso_den->Fill(genMatchElecInAcc[i]->Pt(), genMatchElecInAccAct[i]);


                    hElecEffPtReco_num->Fill(genMatchElecInAcc[i]->Pt());
                    
                    hElecEffPtIso_den->Fill(genMatchElecInAcc[i]->Pt());
                }
                for(int i = 0; i < genMatchIsoElecInAcc.size(); ++i)
                {
                    hElecEffPtActIso_num->Fill(genMatchIsoElecInAcc[i]->Pt(), genMatchIsoElecInAccAct[i]);

                    hElecEffPtIso_num->Fill(genMatchIsoElecInAcc[i]->Pt());
                }


                for(int i = hZRes->FindBin(-0.3); i <= hZRes->FindBin(0.3); ++i)
                {
                    hZElecAccPtSmear_den->Fill(genZPt*(1+hZRes->GetBinCenter(i)), hZRes->GetBinContent(i)/hZRes_int);
                }
                hZElecAccPt_den->Fill(genZPt, file.getWeight());
                if(genElecInAcc.size() >= 2)
                {
                    for(int i = hZRes->FindBin(-0.3); i <= hZRes->FindBin(0.3); ++i)
                    {
                        hZElecAccPtSmear_num->Fill(genZPt*(1+hZRes->GetBinCenter(i)), hZRes->GetBinContent(i)/hZRes_int);

                        hZElecAccPtPtSmear_den->Fill(genZPt*(1+hZRes->GetBinCenter(i)), hZRes->GetBinContent(i)/hZRes_int);
                        if(genElecInAcc[0]->Pt() > 33 && genElecInAcc[1]->Pt()) hZElecAccPtPtSmear_num->Fill(genZPt*(1+hZRes->GetBinCenter(i)), hZRes->GetBinContent(i)/hZRes_int);
                    }
                    hZElecAccPt_num->Fill(genZPt, file.getWeight());
                
                    hZEffPt_den->Fill(genZPt, file.getWeight());
                    hZEff_den->Fill(genZPt, modHt, file.getWeight());
                    if(genElecInAcc[0]->Pt() > 45 && genElecInAcc[1]->Pt() > 20 && genZM > 71 && genZM < 111)//passElecZinvSel && passZinvBaselineNoTag)//genMatchElecInAcc.size() >= 2)
                    {
                        hZEffPt_num->Fill(genZPt, file.getWeight());
                        hZEff_num->Fill(genZPt, modHt, file.getWeight());
                    }
                }
            }

        }
    }

    f->cd();
    hMuEffPt_num->Write();
    hMuEffPt_den->Write();
    hMuAccPt_num->Write();
    hMuAccPt_den->Write();
    hMuEffHt_num->Write();
    hMuEffHt_den->Write();
    hMuAccHt_num->Write();
    hMuAccHt_den->Write();
    hMuEff_num->Write();
    hMuEff_den->Write();
    hMuAcc_num->Write();
    hMuAcc_den->Write();

    hMuEff_num_pp->Write();
    hMuEff_num_pf->Write();
    hMuEff_num_fp->Write();

    hMuEffPtActReco_num->Write();
    hMuEffPtActReco_den->Write();
    hMuEffPtActIso_num->Write();
    hMuEffPtActIso_den->Write();
    hMuEffPtReco_num->Write();
    hMuEffPtReco_den->Write();
    hMuEffPtIso_num->Write();
    hMuEffPtIso_den->Write();

    hElecEffPtActReco_num->Write();
    hElecEffPtActReco_den->Write();
    hElecEffPtActIso_num->Write();
    hElecEffPtActIso_den->Write();
    hElecEffPtReco_num->Write();
    hElecEffPtReco_den->Write();
    hElecEffPtIso_num->Write();
    hElecEffPtIso_den->Write();

    hMuEff_num_rand->Write();

    hZEffPt_num->Write();
    hZEffPt_den->Write();
    hZAccPt_num->Write();
    hZAccPt_den->Write();
    hZAccPtSmear_den->Write();
    hZAccPtSmear_num->Write();
    hZAccPtPtSmear_den->Write();
    hZAccPtPtSmear_num->Write();

    hZElecAccPt_num->Write();
    hZElecAccPt_den->Write();
    hZElecAccPtSmear_den->Write();
    hZElecAccPtSmear_num->Write();
    hZElecAccPtPtSmear_den->Write();
    hZElecAccPtPtSmear_num->Write();

    hZEff_num->Write();
    hZEff_den->Write();
    hZAcc_num->Write();
    hZAcc_den->Write();

    //hZEff_jActR1_num->Write();
    //hZEff_jActR1_den->Write();
    //hZEff_jActR2_num->Write();
    //hZEff_jActR2_den->Write();

    f->Close();
}
