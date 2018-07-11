#include "../../SusyAnaTools/Tools/NTupleReader.h"
#include "../../SusyAnaTools/Tools/samples.h"
#include "derivedTupleVariables.h"

#include <iostream>
#include <string>
#include <vector>

#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TRandom3.h"

void GetnTops(NTupleReader& tr)
{
    try
    {
        int nTops = tr.getVar<int>("nTopCandSortedCntZinv");
        std::vector<TLorentzVector> *vTops = new std::vector<TLorentzVector>();

        throw "FIX ME WITH NEW TAGGER";

        //for(int it=0; it<nTops; it++)
        //{
        //    TLorentzVector topLVec = type3Ptr->buildLVec(tr.getVec<TLorentzVector>("jetsLVec_forTaggerZinv"),
        //                                                 type3Ptr->finalCombfatJets[type3Ptr->ori_pickedTopCandSortedVec[it]]);
        //    vTops->push_back(topLVec);
        //}
        //
        //auto& mt2inputs = type3Ptr->had_brJetLVecMap;
        //std::vector<TLorentzVector> *vMT2Inputs = new std::vector<TLorentzVector>();
        //if(vTops->size())
        //{
        //    vMT2Inputs->push_back(vTops->at(0));
        //}
        //if(!mt2inputs.empty())
        //{
        //    vMT2Inputs->push_back(mt2inputs.begin()->second);
        //}
        //
        //tr.registerDerivedVec("vTops", vTops);
        //tr.registerDerivedVec("vMT2Inputs", vMT2Inputs);
    }
    catch(const std::string e)
    {
        std::cout << e << std::endl;
    }

    return;
}

int main()
{
    TH1::AddDirectory(false);

    TH1* njWTTbar_0b;
    TH1* njWDYZ_0b;
    TH1* njWTTbar_g1b;
    TH1* njWDYZ_g1b;

    TH1* shapeMET;
    TH1* shapeMT2;
    TH1* shapeNT;
    TH1* shapeNB;

    TRandom3 tr3(153474);

    TFile *f = new TFile("dataMCweights.root");
    if(f)
    {
        //njWTTbar_0b  = static_cast<TH1*>(f->Get("DataMC_nj_elmuZinv_loose0")->Clone());//ht200_dphi")->Clone());
        njWDYZ_0b    = static_cast<TH1*>(f->Get("DataMC_nj_muZinv_0b_loose0_mt2_MET")->Clone());//ht200_dphi")->Clone());
       // njWTTbar_g1b = static_cast<TH1*>(f->Get("DataMC_nj_elmuZinv_loose0")->Clone());//ht200_dphi")->Clone());
        njWDYZ_g1b   = static_cast<TH1*>(f->Get("DataMC_nj_muZinv_g1b_loose0_mt2_MET")->Clone());//ht200_dphi")->Clone());
        f->Close();
        delete f;
    }
    else
    {
        std::cout << "Failed to open: dataMCweights.root" << std::endl;
    }

    f = new TFile("syst_shape.root");
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

    AnaSamples::SampleSet        ss("/uscms_data/d3/snorberg/CMSSW_8_0_23_patch1/src/ZInvisible/Tools/", AnaSamples::luminosity);
    AnaSamples::SampleCollection sc(ss);

    //f = TFile::Open("condor/minituple-Feb5.root");

    //Prep variables for nJet and shape systematic weights
    const int NTRIALS = 1000;
    const int NSEARCHBINS = 84;

    TH1 *h[5][NSEARCHBINS];
    std::vector<std::string> hnames = {"njet", "met", "mt2", "nt", "nb"};

    float N0[NSEARCHBINS];
    float N0square[NSEARCHBINS];
    float N[5][NSEARCHBINS][NTRIALS];
    float N0_83=0.0;
    float N0square_83=0.0;
    float N0_divide_83=0.0; 
    float N0_83_top=0.0;
    float N0square_83_top=0.0;
    float N0_divide_83_top=0.0;
    float N0_84=0.0;
    float N0square_84=0.0;
    float N0_divide_84=0.0;
    float N0_84_top=0.0;
    float N0square_84_top=0.0;
    float N0_divide_84_top=0.0;
    int n83_num =0;
    int n83t_num =0;
    int n84_num= 0;
    int n84t_num=0;
    int n80_num=0;

    float variations[5][20][NTRIALS];
    for(int iT = 0; iT < NTRIALS; ++iT)
    {
        for(int i = 0; i < 5; ++i) for(int j = 0; j < 20; ++j) variations[i][j][iT] = 1.0;
        for(int i = 0; i <= njWDYZ_g1b->GetNbinsX() + 1; ++i) if(njWDYZ_g1b->GetBinContent(i) > 1e-10) variations[0][i][iT] = (float)tr3.Gaus(1.0, njWDYZ_g1b->GetBinError(i)/njWDYZ_g1b->GetBinContent(i));
        for(int i = 0; i <= shapeMET  ->GetNbinsX() + 1; ++i) if(shapeMET->GetBinContent(i) > 1e-10)   variations[1][i][iT] = (float)tr3.Gaus(1.0, shapeMET->GetBinError(i)/shapeMET->GetBinContent(i));
        for(int i = 0; i <= shapeMT2  ->GetNbinsX() + 1; ++i) if(shapeMT2->GetBinContent(i) > 1e-10)   variations[2][i][iT] = (float)tr3.Gaus(1.0, shapeMT2->GetBinError(i)/shapeMT2->GetBinContent(i));
        for(int i = 0; i <= shapeNT   ->GetNbinsX() + 1; ++i) if(shapeNT->GetBinContent(i) > 1e-10)    variations[3][i][iT] = (float)tr3.Gaus(1.0, shapeNT->GetBinError(i)/shapeNT->GetBinContent(i));
        for(int i = 0; i <= shapeNB   ->GetNbinsX() + 1; ++i) if(shapeNB->GetBinContent(i) > 1e-10)    variations[4][i][iT] = (float)tr3.Gaus(1.0, shapeNB->GetBinError(i)/shapeNB->GetBinContent(i));
    }

    for(int ih = 0; ih < 5; ++ih)
    {
        for(int i = 0; i < NSEARCHBINS; ++i)
        {
            char name[128];
            sprintf(name, "hSB_%s_%d", hnames[ih].c_str(), i);
            h[ih][i] = new TH1D(name, name, 2000, -1, 3);

            for(int j = 0; j < NTRIALS; ++j)
            {
                N[ih][i][j] = 0.0;
            }
        }
    }

    for(int i = 0; i < NSEARCHBINS; ++i)
    {
        N0square[i] = 0.0;
        N0[i] = 0.0;
    }

    //prep histograms for MET-MT2 corrolation study
    TH1* h_ratio_MET_nom  = new TH1D("h_ratio_MET_nom",      "hMET_nom",  400, 0, 2);
    TH1* h_ratio_MT2_nom  = new TH1D("h_ratio_MT2_nom",      "hMET_nom",  400, 0, 2);
    TH2* h_ratio_2D_nom   = new TH2D("h_ratio_MT2vMET_nom",  "h2D_nom ;MET;MT2", 400, 0, 2, 400, 0, 2);
    TH2* h_ratio_2Dvmet_nom   = new TH2D("h_ratiovmet_MT2vMET_nom",  "h2D_nom ;MET;MT2", 200, 0, 2000, 400, 0, 2);
    TH2* h_ratio_2Dvmt2_nom   = new TH2D("h_ratiovmt2_MT2vMET_nom",  "h2D_nom ;MET;MT2", 200, 0, 2000, 400, 0, 2);

    //more debug histograms
    // angle betwee MET and tops
    TH2* h_met_angle_t1_met = new TH2D("h_met_angle_t1_met", "h_met_angle_t1_met", 160, -3.2, 3.2, 400, 0, 2);
    TH2* h_met_angle_t2_met = new TH2D("h_met_angle_t2_met", "h_met_angle_t2_met", 160, -3.2, 3.2, 400, 0, 2);
    TH2* h_met_angle_t1_t2  = new TH2D("h_met_angle_t1_t2",  "h_met_angle_t1_t2",  160, -3.2, 3.2, 400, 0, 2);
    TH2* h_met_deta_t1_t2   = new TH2D("h_met_deta_t1_t2",   "h_met_deta_t1_t2",   100, -2.5, 2.5, 400, 0, 2);
    TH2* h_met_pt_t1        = new TH2D("h_met_pt_t1",        "h_met_pt_t1",        150,   0, 1500, 400, 0, 2);
    TH2* h_met_pt_t2        = new TH2D("h_met_pt_t2",        "h_met_pt_t2",        150,   0, 1500, 400, 0, 2);
    TH2* h_met_eta_t1       = new TH2D("h_met_eta_t1",       "h_met_eta_t1",       100, -2.5, 2.5, 400, 0, 2);
    TH2* h_met_eta_t2       = new TH2D("h_met_eta_t2",       "h_met_eta_t2",       100, -2.5, 2.5, 400, 0, 2);
    TH2* h_met_metPhi       = new TH2D("h_met_metPhi",       "h_met_metPhi",       160, -3.2, 3.2, 400, 0, 2);
    TH2* h_met_mass_t1      = new TH2D("h_met_mass_t1",      "h_met_mass_t1",      200,   0, 1000, 400, 0, 2);
    TH2* h_met_mass_t2      = new TH2D("h_met_mass_t2",      "h_met_mass_t2",      200,   0, 1000, 400, 0, 2);
    TH2* h_met_tmt_mt2      = new TH2D("h_met_tmt_mt2",      "h_met_tmt_mt2",      400,   0,    2, 400, 0, 2);

    TH2* h_mt2_angle_t1_met = new TH2D("h_mt2_angle_t1_met", "h_mt2_angle_t1_met", 160, -3.2, 3.2, 400, 0, 2);
    TH2* h_mt2_angle_t2_met = new TH2D("h_mt2_angle_t2_met", "h_mt2_angle_t2_met", 160, -3.2, 3.2, 400, 0, 2);
    TH2* h_mt2_angle_t1_t2  = new TH2D("h_mt2_angle_t1_t2",  "h_mt2_angle_t1_t2",  160, -3.2, 3.2, 400, 0, 2);
    TH2* h_mt2_deta_t1_t2   = new TH2D("h_mt2_deta_t1_t2",   "h_mt2_deta_t1_t2",   100, -2.5, 2.5, 400, 0, 2);
    TH2* h_mt2_pt_t1        = new TH2D("h_mt2_pt_t1",        "h_mt2_pt_t1",        150,   0, 1500, 400, 0, 2);
    TH2* h_mt2_pt_t2        = new TH2D("h_mt2_pt_t2",        "h_mt2_pt_t2",        150,   0, 1500, 400, 0, 2);
    TH2* h_mt2_eta_t1       = new TH2D("h_mt2_eta_t1",       "h_mt2_eta_t1",       100, -2.5, 2.5, 400, 0, 2);
    TH2* h_mt2_eta_t2       = new TH2D("h_mt2_eta_t2",       "h_mt2_eta_t2",       100, -2.5, 2.5, 400, 0, 2);
    TH2* h_mt2_metPhi       = new TH2D("h_mt2_metPhi",       "h_mt2_metPhi",       160, -3.2, 3.2, 400, 0, 2);
    TH2* h_mt2_mass_t1      = new TH2D("h_mt2_mass_t1",      "h_mt2_mass_t1",      200,   0, 1000, 400, 0, 2);
    TH2* h_mt2_mass_t2      = new TH2D("h_mt2_mass_t2",      "h_mt2_mass_t2",      200,   0, 1000, 400, 0, 2);
    TH2* h_mt2_tmt_mt2      = new TH2D("h_mt2_tmt_mt2",      "h_mt2_tmt_mt2",      400,   0,    2, 400, 0, 2);


    TH1* hMET_nom  = new TH1D("hMET_nom",  "hMET_nom",  200, 0, 2000);
    TH1* hMET_Gaus = new TH1D("hMET_Gaus", "hMET_Gaus", 200, 0, 2000);
    TH1* hMET_Logi = new TH1D("hMET_Logi", "hMET_Logi", 200, 0, 2000);

    TH1* hMT2_nom  = new TH1D("hMT2_nom",  "hMT2_nom",  200, 0, 2000);
    TH1* hMT2_Gaus = new TH1D("hMT2_Gaus", "hMT2_Gaus", 200, 0, 2000);
    TH1* hMT2_Logi = new TH1D("hMT2_Logi", "hMT2_Logi", 200, 0, 2000);

    TH2* h2D_nom  = new TH2D("hMT2vMET_nom",  "h2D_nom ;MET;MT2", 200, 0, 2000, 200, 0, 2000);
    TH2* h2D_Gaus = new TH2D("hMT2vMET_Gaus", "h2D_Gaus;MET;MT2", 200, 0, 2000, 200, 0, 2000);
    TH2* h2D_Logi = new TH2D("hMT2vMET_Logi", "h2D_Logi;MET;MT2", 200, 0, 2000, 200, 0, 2000);

    //AnaFunctions::prepareTopTagger();

    try
    {

        //for(auto& fs : sc["DYJetsToLL"])
        for(auto& fs : sc["ZJetsToNuNu"])
        {
            //size_t start = fs.filePath.rfind('/');
            //size_t stop  = fs.filePath.rfind('.');
            //std::string treeName = fs.filePath.substr(start + 1, stop - start - 1);

            TChain *t = new TChain(fs.treePath.c_str());
            fs.addFilesToChain(t);

            std::cout << "Tree: " << fs.treePath << std::endl;
            //std::cout << "sigma*lumi: " << fs.getWeight() << std::endl;

            plotterFunctions::PrepareMiniTupleVars pmt(false);
            plotterFunctions::NJetWeight njWeight;
            plotterFunctions::TriggerInfo triggerInfo(true);
            plotterFunctions::MetSmear metSmear;
	    

            NTupleReader tr(t);
            tr.registerFunction(pmt);
            tr.registerFunction(njWeight);
            tr.registerFunction(triggerInfo);
            //tr.registerFunction(metSmear);
            //tr.registerFunction(GetnTops);

            while(tr.getNextEvent())
            {
                if(tr.getEvtNum() % 10000 == 0) std::cout << "Event #: " << tr.getEvtNum() << std::endl;

                const int& nSearchBin = tr.getVar<int>("nSearchBin");
                const int& cntNJetsPt30Eta24Zinv = tr.getVar<int>("cntNJetsPt30Eta24Zinv");
                const float& cleanMetPt = tr.getVar<float>("cleanMetPt");
                const float& best_had_brJet_MT2Zinv = tr.getVar<float>("best_had_brJet_MT2Zinv");
                const int& cntCSVSZinv            = tr.getVar<int>("cntCSVSZinv");
                const int& nTopCandSortedCntZinv  = tr.getVar<int>("nTopCandSortedCntZinv");
                const bool& passBaseline = tr.getVar<bool>("passBaseline");
                const bool& passBaselineZinv = tr.getVar<bool>("passBaselineZinv");
                const bool& passLeptVeto = tr.getVar<bool>("passLeptVeto");
                const float& bTagSF_EventWeightSimple_Central = tr.getVar<float>("bTagSF_EventWeightSimple_Central");

                const float& triggerEffMC = tr.getVar<float>("TriggerEffMC");
                const float& nJetWgtDYZ   = tr.getVar<float>("nJetWgtDYZ");
                const float& normWgt0b    = tr.getVar<float>("normWgt0b");

                if(passBaselineZinv && passLeptVeto && cntCSVSZinv>=1 && nTopCandSortedCntZinv >= 3 && cleanMetPt < 350)
                {
                    float weight_1 = triggerEffMC * nJetWgtDYZ * normWgt0b * bTagSF_EventWeightSimple_Central * fs.getWeight();
                        N0_83 += weight_1;
                        N0square_83 += weight_1*weight_1;
                        //N0_divide_83 = N0_83/N0square_83;
                        //std::cout<<"events bin 83 "<<tr.getEvtNum() <<std::endl;
                        n83_num ++;
                }
                if(passBaselineZinv && passLeptVeto && cntCSVSZinv>=3 && nTopCandSortedCntZinv >= 2 && cleanMetPt < 350)
                {
                    float weight_2 = triggerEffMC * nJetWgtDYZ * normWgt0b * bTagSF_EventWeightSimple_Central * fs.getWeight();
                        N0_83_top += weight_2;
                        N0square_83_top += weight_2*weight_2;
                        //std::cout<<"events bin 83 top "<<tr.getEvtNum() <<std::endl;
                        n83t_num ++;
                }

               if(passBaselineZinv && passLeptVeto && cntCSVSZinv>=2 && nTopCandSortedCntZinv >= 3 && cleanMetPt > 350)
                {
                    float weight_3 = triggerEffMC * nJetWgtDYZ * normWgt0b * bTagSF_EventWeightSimple_Central * fs.getWeight();
                        N0_84 += weight_3;
                        N0square_84 += weight_3*weight_3;
                        n84_num ++;
                }
                if(passBaselineZinv && passLeptVeto && cntCSVSZinv>=3 && nTopCandSortedCntZinv >= 3 && cleanMetPt > 350)
                {
                    float weight_4 = triggerEffMC * nJetWgtDYZ * normWgt0b * bTagSF_EventWeightSimple_Central * fs.getWeight();
                        N0_84_top += weight_4;
                        N0square_84_top += weight_4*weight_4;
                        n84t_num ++;
                }
                if(passBaselineZinv && passLeptVeto && cntCSVSZinv==2 && nTopCandSortedCntZinv >= 3 && cleanMetPt < 400 )
                {
                n80_num++;
                }
                N0_divide_83 = N0square_83/N0_83;
                N0_divide_83_top = N0square_83_top/N0_83_top;
                N0_divide_84 = N0square_84/N0_84;
                N0_divide_84_top = N0square_84_top/N0_84_top;
                //std::cout<<"N0_divide_83 "<<N0_divide_83<<std::endl;
                //fill stat uncertainty histograms here
                if(passBaselineZinv && passLeptVeto)
                {
                    float weight = triggerEffMC * nJetWgtDYZ * normWgt0b * bTagSF_EventWeightSimple_Central * fs.getWeight();

                    if(nSearchBin >= 0 && nSearchBin < NSEARCHBINS)
                    {
                        N0[nSearchBin] += weight;
                        N0square[nSearchBin] += weight*weight;
                        for(int iTrial = 0; iTrial < NTRIALS; ++iTrial)
                        {
                            if(iTrial < NTRIALS)
                            {
                                N[0][nSearchBin][iTrial] += variations[0][njWDYZ_g1b->FindBin(cntNJetsPt30Eta24Zinv)][iTrial] * weight;
                                N[1][nSearchBin][iTrial] += variations[1][shapeMET->FindBin(cleanMetPt)][iTrial]              * weight;
                                N[2][nSearchBin][iTrial] += variations[2][shapeMT2->FindBin(best_had_brJet_MT2Zinv)][iTrial]  * weight;
                                N[3][nSearchBin][iTrial] += variations[3][shapeNT->FindBin(nTopCandSortedCntZinv)][iTrial]    * weight;
                                N[4][nSearchBin][iTrial] += variations[4][shapeNB->FindBin(cntCSVSZinv)][iTrial]              * weight;
                            }
                        }
                    }
                }
            }
        }
    std::cout<<"N0_divide_83 "<<N0_divide_83 <<" number "<< n83_num <<std::endl;
    std::cout<<"N0_divide_83_top "<<N0_divide_83_top <<" number "<< n83t_num <<std::endl;
    std::cout<<"N0_divide_84 "<<N0_divide_84 <<" number "<< n84_num <<std::endl;
    std::cout<<"N0_divide_84_top "<<N0_divide_84_top <<" number "<< n84t_num <<std::endl;
    std::cout<<"Number in bin 80 " << n80_num <<std::endl;
    }
    catch(const std::string e)
    {
        std::cout << e << std::endl;
        return 0;
    }

    for(int ih = 0; ih < 5; ++ih)
    {
        for(int i = 0; i < NSEARCHBINS; ++i)
        {
            for(int iTrial = 0; iTrial < NTRIALS; ++iTrial)
            {
                h[ih][i]->Fill(N[ih][i][iTrial]/N0[i]);
            }
        }
    }

    TFile fout("syst_nJetWgt.root", "RECREATE");
    TH1 *syst68Max = new TH1D("syst68Max", "syst68Max", NSEARCHBINS, 0, NSEARCHBINS);
    for(int ih = 0; ih < 5; ++ih)
    {
        char name[128];
        sprintf(name, "syst68_%s", hnames[ih].c_str());
        TH1 *syst68 = new TH1D(name, name, NSEARCHBINS, 0, NSEARCHBINS);
        sprintf(name, "systRMS_%s", hnames[ih].c_str());
        TH1 *systRMS = new TH1D(name, name, NSEARCHBINS, 0, NSEARCHBINS);

        for(int i = 0; i < NSEARCHBINS; ++i)
        {
            h[ih][i]->Write();
            h[ih][i]->Scale(1/h[ih][i]->Integral(0, h[ih][i]->GetNbinsX() + 1));
            float ll = -999.9, ul = -999.9;
            TH1* hint = (TH1*)h[ih][i]->Clone((std::string(h[ih][i]->GetName())+"_int").c_str());
            for(int iBin = 1; iBin <= h[ih][i]->GetNbinsX(); ++iBin)
            {
                hint->SetBinContent(iBin, h[ih][i]->Integral(0, iBin));
                if     (ll < 0 && h[ih][i]->Integral(0, iBin) > 0.16) ll = h[ih][i]->GetBinCenter(iBin);
                else if(ul < 0 && h[ih][i]->Integral(0, iBin) > 0.84) ul = h[ih][i]->GetBinCenter(iBin);
            }
            if(ll < 0) ll = 0;
            if(ul < 0) ul = 2.0;
            syst68->SetBinContent(i + 1, (ul - ll) / 2.0);
            syst68Max->SetBinContent(i + 1, std::max(syst68Max->GetBinContent(i + 1), (ul - ll) / 2.0));
            systRMS->SetBinContent(i + 1, h[ih][i]->GetRMS());
            //std::cout << "bin: " << i << "\tll: " << ll << "\tul: " << ul << "\tsym err: " << (ul - ll) / 2.0 << "\t:RMS: " << h[ih][i]->GetRMS() << std::endl;
            hint->Write();
        }
        syst68->Write();
        systRMS->Write();
    }
    syst68Max->Write();

    TH1 *centralvalue = new TH1D("centralvalue", "centralvalue", NSEARCHBINS, 0, NSEARCHBINS);
    TH1 *avgWgt = new TH1D("avgWgt", "avgWgt", NSEARCHBINS, 0, NSEARCHBINS);
    TH1 *neff = new TH1D("neff", "neff", NSEARCHBINS, 0, NSEARCHBINS);

    for(int i = 0; i < NSEARCHBINS; ++i)
    {
        centralvalue->SetBinContent(i + 1, N0[i]);
        if(N0[i] > 1e-10) avgWgt->SetBinContent(i + 1, N0square[i]/N0[i]);
        if(N0square[i] > 1e-10) neff->SetBinContent(i + 1, N0[i]*N0[i]/N0square[i]);
    }

    centralvalue->Write();
    avgWgt->Write();
    neff->Write();
    fout.Close();

    //Write out MET-MT2 histograms
    TFile fout2("Corrolation_MET-MT2.root", "RECREATE");

    h_ratio_MET_nom->Write();
    h_ratio_MT2_nom->Write();
    h_ratio_2D_nom ->Write();
    h_ratio_2Dvmet_nom->Write();
    h_ratio_2Dvmt2_nom->Write();

    hMET_nom ->Write();
    hMET_Gaus->Write();
    hMET_Logi->Write();

    hMT2_nom ->Write();
    hMT2_Gaus->Write();
    hMT2_Logi->Write();

    h2D_nom ->Write();
    h2D_Gaus->Write();
    h2D_Logi->Write();

    h_met_angle_t1_met->Write();
    h_met_angle_t2_met->Write();
    h_met_angle_t1_t2 ->Write();
    h_met_deta_t1_t2  ->Write();
    h_met_pt_t1       ->Write();
    h_met_pt_t2       ->Write();
    h_met_eta_t1      ->Write();
    h_met_eta_t2      ->Write();
    h_met_metPhi      ->Write();
    h_met_mass_t1     ->Write();
    h_met_mass_t2     ->Write();
    h_met_tmt_mt2     ->Write();

    h_mt2_angle_t1_met->Write();
    h_mt2_angle_t2_met->Write();
    h_mt2_angle_t1_t2 ->Write();
    h_mt2_deta_t1_t2  ->Write();
    h_mt2_pt_t1       ->Write();
    h_mt2_pt_t2       ->Write();
    h_mt2_eta_t1      ->Write();
    h_mt2_eta_t2      ->Write();
    h_mt2_metPhi      ->Write();
    h_mt2_mass_t1     ->Write();
    h_mt2_mass_t2     ->Write();
    h_mt2_tmt_mt2     ->Write();

    fout2.Close();
}
