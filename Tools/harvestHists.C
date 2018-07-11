#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

#include <iostream>
#include <cstdio>
#include <string>

TH2* rebin2d(TH2* old, float xbins[], int nx, float ybins[], int ny)
{
    //create a new TH2 with your bin arrays spec
    //int nx = sizeof(xbins)/sizeof(float) - 1;

    //int ny = sizeof(ybins)/sizeof(float) - 1;

    char hname[128];
    sprintf(hname, "%s_rebin", old->GetName());
    TH2 *h = new TH2F(hname, old->GetTitle(), nx, xbins, ny, ybins);
    TAxis *xaxis = old->GetXaxis();
    TAxis *yaxis = old->GetYaxis();
    for (int i=0; i<=xaxis->GetNbins() + 1; i++) 
    {
        for(int j=0; j<=yaxis->GetNbins() + 1; j++) 
        {
            h->Fill(xaxis->GetBinCenter(i), yaxis->GetBinCenter(j), old->GetBinContent(i, j));
        }
    }

    return h;
}

void combineActBins(TH2* hnum, TH2* hden, const int N = 5)
{
    TAxis *xaxis = hnum->GetXaxis();
    TAxis *yaxis = hnum->GetYaxis();

    for (int j=1; j<=yaxis->GetNbins(); j++)
    {
        int i;
        float contentNum = 0.0, contentDen = 0.0;
        for(i = xaxis->GetNbins() + 1; i >= 0; i--) 
        {
            contentNum += hnum->GetBinContent(i, j);
            contentDen += hden->GetBinContent(i, j);
            if(contentNum > N) break;
        }
        if(i <= 1) i = 2;
        for(; i <= xaxis->GetNbins() + 1; i++)
        {
            hnum->SetBinContent(i, j, contentNum);
            hnum->SetBinError(i, j, sqrt(contentNum));
            hden->SetBinContent(i, j, contentDen);
            hden->SetBinError(i, j, sqrt(contentDen));
        }
    }
}

static float dzbt[] = {0.0};

void makeRatio1D(std::string label, TFile *fin, TFile *fout, float bins[] = dzbt, int nbins = 0)
{
    fin->cd();
    TH1 *num = (TH1*)fin->Get((label+"_num").c_str());
    TH1 *den = (TH1*)fin->Get((label+"_den").c_str());

    if(nbins > 1)
    {
        num = num->Rebin(nbins - 1, "", bins);
        den = den->Rebin(nbins - 1, "", bins);
    }

    TH1 *ratio = (TH1*)num->Clone((label + "_ratio").c_str());
    ratio->Divide(den);

    fout->cd();
    ratio->Write();
}

void makeRatio2D(std::string label, TFile *fin, TFile *fout, float binsX[] = dzbt, int nx = 0, float binsY[] = dzbt, int ny = 0)
{
    fin->cd();
    TH2 *num = (TH2*)fin->Get((label+"_num").c_str());
    TH2 *den = (TH2*)fin->Get((label+"_den").c_str());

    if(nx > 1 && ny > 1)
    {
        num = rebin2d(num, binsX, nx - 1, binsY, ny - 1);
        den = rebin2d(den, binsX, nx - 1, binsY, ny - 1);
    }

    combineActBins(num, den, 10);

    fout->cd();
    num->Write();
    den->Write();

    TH2 *ratio = (TH2*)num->Clone((label + "_ratio").c_str());
    ratio->Divide(den);

    ratio->Write();
}

int main ()
{
    TH1::AddDirectory(false);

    float xbins[] = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 220.0, 240.0, 260.0, 280.0, 300.0, 340.0, 380.0, 420.0, 460.0, 500.0, 560.0, 2000.0};
    int nxbins = sizeof(xbins)/sizeof(float);

    float ybins[] = {0.0, 10.0, 40.0, 120.0, 200.0, 300.0, 400.0, 650.0, 800.0, 1000.0, 5000.0};
    int nybins = sizeof(ybins)/sizeof(float);

    float zptbins[] = {0.0, 100.0, 140.0, 180.0, 220.0, 300.0, 400.0, 500.0, 600.0, 800.0, 2000.0};
    int nzptbins = sizeof(zptbins)/sizeof(float);

    float zptbins2[] = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0, 380.0, 390.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0, 520.0, 540.0, 560.0, 580.0, 600.0, 640.0, 680.0, 720.0, 760.0, 800.0, 840.0, 920.0, 1000.0, 1400.0, 2000.0};
    int nzptbins2 = sizeof(zptbins2)/sizeof(float);

    float muptbins[] = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 140.0, 180.0, 200.0, 300.0, 400.0, 500.0, 600.0, 800.0, 2000.0};
    int nmuptbins = sizeof(muptbins)/sizeof(float);

    float muptbins2[] = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0, 150.0, 300.0, 500.0, 800.0, 1400.0, 2000.0};
    int nmuptbins2 = sizeof(muptbins2)/sizeof(float);

    float muptbins3[] = {0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 140.0, 180.0, 200.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1200.0, 2000.0};
    int nmuptbins3 = sizeof(muptbins3)/sizeof(float);

    //float actbins[] = {0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 200.0};//, 500.0, 3000.0};
    float actbins[] = {0.0, 0.02, 0.05, 0.10, 0.15, 0.40, 0.60, 1.0, 20.0};
    int nactbins = sizeof(actbins)/sizeof(float);

    TFile  *fin = new TFile("effhists.root");
    TFile *fout = new TFile("lepEffHists2.root", "RECREATE");

    makeRatio1D("hMuEffPt", fin, fout, muptbins, nmuptbins);
    makeRatio1D("hMuAccPt", fin, fout);
    
    makeRatio1D("hMuEffHt", fin, fout);
    makeRatio1D("hMuAccHt", fin, fout);
    
    makeRatio2D("hMuEff", fin, fout);
    makeRatio2D("hMuAcc", fin, fout);
    
    makeRatio1D("hZEffPt", fin, fout, zptbins, nzptbins);
    makeRatio1D("hZAccPt", fin, fout, zptbins, nzptbins);

    makeRatio1D("hZElecAccPt", fin, fout, zptbins2, nzptbins2);
    
    makeRatio1D("hZAccPtSmear", fin, fout, zptbins2, nzptbins2);
    makeRatio1D("hZAccPtMuPtSmear", fin, fout, zptbins2, nzptbins2);

    makeRatio1D("hZElecAccPtSmear", fin, fout, zptbins2, nzptbins2);
    makeRatio1D("hZElecAccPtPtSmear", fin, fout, zptbins2, nzptbins2);
    
    makeRatio2D("hZEff", fin, fout);
    makeRatio2D("hZAcc", fin, fout);

    //makeRatio2D("hZEff_jActR1", fin, fout, zptbins, nzptbins, ybins, nybins);
    //makeRatio2D("hZEff_jActR2", fin, fout);
    
    makeRatio2D("hMuEffPtActReco", fin, fout, muptbins2, nmuptbins2, actbins, nactbins);
    makeRatio2D("hMuEffPtActIso", fin, fout, muptbins2, nmuptbins2, actbins, nactbins);

    makeRatio2D("hElecEffPtActReco", fin, fout, muptbins2, nmuptbins2, actbins, nactbins);
    makeRatio2D("hElecEffPtActIso", fin, fout, muptbins2, nmuptbins2, actbins, nactbins);
    
    makeRatio1D("hMuEffPtReco", fin, fout, muptbins3, nmuptbins3);
    makeRatio1D("hMuEffPtIso",  fin, fout, muptbins3, nmuptbins3);

    makeRatio1D("hElecEffPtReco", fin, fout, muptbins3, nmuptbins3);
    makeRatio1D("hElecEffPtIso",  fin, fout, muptbins3, nmuptbins3);

    fout->Close();

    //Derive N(b) scale factors 
    TFile *fin2 = new TFile("condor/histoutput-Feb5_2016.root");
    TH1 *h_1b =      (TH1*)fin2->Get("nTopCandSortedCntZinv/fake1b_baselineNoTag_nTopnTopCandSortedCntZinvnTopCandSortedCntZinvZ#rightarrow#nu#nu N(b) = 1single");
    TH1 *h_1b_fake = (TH1*)fin2->Get("nTopCandSortedCntZinv/fake1b_baselineNoTag_nTopnTopCandSortedCntZinv1bnTopCandSortedCntZinv1bZ#rightarrow#nu#nu N(b) = 0, 1 fake bsingle");
    TH1 *h_2b =      (TH1*)fin2->Get("nTopCandSortedCntZinv/fake2b_baselineNoTag_nTopnTopCandSortedCntZinvnTopCandSortedCntZinvZ#rightarrow#nu#nu N(b) = 2single");
    TH1 *h_2b_fake = (TH1*)fin2->Get("nTopCandSortedCntZinv/fake2b_baselineNoTag_nTopnTopCandSortedCntZinv2bnTopCandSortedCntZinv2bZ#rightarrow#nu#nu N(b) = 0, 2 fake bsingle");
    TH1 *h_3b =      (TH1*)fin2->Get("nTopCandSortedCntZinv/fake3b_baselineNoTag_nTopnTopCandSortedCntZinvnTopCandSortedCntZinvZ#rightarrow#nu#nu N(b) > 2single");
    TH1 *h_3b_fake = (TH1*)fin2->Get("nTopCandSortedCntZinv/fake3b_baselineNoTag_nTopnTopCandSortedCntZinv3bnTopCandSortedCntZinv3bZ#rightarrow#nu#nu N(b) = 0, 3 fake bsingle");
    
    printf("N(b) extrapolation scale factors\n");
    if(h_1b && h_1b_fake) printf("N(b) = 0 -> 1: %e\n", h_1b->Integral()/h_1b_fake->Integral());
    if(h_2b && h_2b_fake) printf("N(b) = 0 -> 2: %e\n", h_2b->Integral()/h_2b_fake->Integral());
    if(h_3b && h_3b_fake) printf("N(b) = 0 -> 3: %e\n", h_3b->Integral()/h_3b_fake->Integral());

    fout = new TFile("njetWgtHists.root", "RECREATE");
    TH1 *h_nj_1b =      (TH1*)fin2->Get("cntNJetsPt30Eta24Zinv/ClosureNb_nj_nw_1fakeb_loose0cntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvZ#rightarrow#nu#nu, N(b) = 0single");
    TH1 *h_nj_1b_fake = (TH1*)fin2->Get("cntNJetsPt30Eta24Zinv/ClosureNb_nj_nw_1fakeb_loose0cntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvZ#rightarrow#nu#nu, Direct MC, N(b) = 1single");
    TH1 *h_nj_2b =      (TH1*)fin2->Get("cntNJetsPt30Eta24Zinv/ClosureNb_nj_nw_2fakeb_loose0cntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvZ#rightarrow#nu#nu, N(b) = 0single");
    TH1 *h_nj_2b_fake = (TH1*)fin2->Get("cntNJetsPt30Eta24Zinv/ClosureNb_nj_nw_2fakeb_loose0cntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvZ#rightarrow#nu#nu, Direct MC, N(b) = 2single");
    TH1 *h_nj_3b =      (TH1*)fin2->Get("cntNJetsPt30Eta24Zinv/ClosureNb_nj_nw_3fakeb_loose0cntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvZ#rightarrow#nu#nu, N(b) = 0single");
    TH1 *h_nj_3b_fake = (TH1*)fin2->Get("cntNJetsPt30Eta24Zinv/ClosureNb_nj_nw_3fakeb_loose0cntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvZ#rightarrow#nu#nu, Direct MC, N(b) >= 3single");
    
    TH1 *h_njRatio_1fake = (TH1*)h_nj_1b_fake->Clone("h_njRatio_1fake");
    TH1 *h_njRatio_2fake = (TH1*)h_nj_2b_fake->Clone("h_njRatio_2fake");
    TH1 *h_njRatio_3fake = (TH1*)h_nj_3b_fake->Clone("h_njRatio_3fake");

    h_njRatio_1fake->Divide(h_nj_1b);
    h_njRatio_2fake->Divide(h_nj_2b);
    h_njRatio_3fake->Divide(h_nj_3b);

    h_njRatio_1fake->Write();
    h_njRatio_2fake->Write();
    h_njRatio_3fake->Write();

    fout->Close();

    fout = new TFile("syst_searchBinStats.root", "RECREATE");
    TH1 *h_nSB_NW = (TH1*)fin2->Get("nb0BinsNW/DataMCww_SingleMuon_nb0BinsNW_muZinv_0b_blnotagnb0BinsNWnb0BinsNWDYstack")->Clone("h_nSB_MC_NW");
    TH1 *h_nSB =    (TH1*)fin2->Get("nSearchBin/NJetWgt_nSearchBinnSearchBinnSearchBinZ#rightarrow#nu#nu DataMC weightsingle")->Clone("h_nSB_MC_pred");

    TH1 *h_nSB_ZZ_NW = (TH1*)fin2->Get("nb0BinsNW/ClosureNb_nSearchBin_njW_baselinenb0BinsNWnb0BinsNWZ#rightarrow#nu#nu DataMC weightsingle")->Clone("h_nSB_ZZ_MC_NW");
    TH1 *h_nSB_ZZ =    (TH1*)fin2->Get("nb0Bins/ClosureNb_nSearchBin_baselinenb0Binsnb0BinsZ#rightarrow#nu#nusingle")->Clone("h_nSB_ZZ_MC_pred");

    TH1 *h_nSB_DY_NW = (TH1*)fin2->Get("nb0Bins/nSearchBin_stat_lognb0Binsnb0BinsDY#rightarrow#mu#musingle")->Clone("h_nSB_DY_MC_NW");
    TH1 *h_nSB_DY =    (TH1*)fin2->Get("nb0Bins/nSearchBinnb0_lognb0Binsnb0BinsDY#rightarrow#mu#mu no #mu, Z eff+accsingle")->Clone("h_nSB_DY_MC_pred");
    
    TFile *fin3 = new TFile("condor/histoutput-Feb5_2016.root");
    //TFile *fin3 = new TFile("dataplots_muon_Dec15.root");
    
    TH1 *h_nSB_Data_0b = (TH1*)fin3->Get("nb0BinsNW/DataMCww_SingleMuon_nb0BinsNW_muZinv_0b_blnotagnb0BinsNWnb0BinsNWDatadata")->Clone("h_nSB_Data_0b");
    h_nSB_Data_0b->Sumw2(false);
    h_nSB_Data_0b->SetBinErrorOption(TH1::kPoisson);

    TH1 *h_nSB_ratio = (TH1*)h_nSB->Clone("h_nSB_ratio");
    h_nSB_ratio->Divide(h_nSB_NW);

    TH1 *h_nSB_ZZ_ratio = (TH1*)h_nSB->Clone("h_nSB_ZZ_ratio");
    h_nSB_ZZ_ratio->Divide(h_nSB_ZZ_NW);

    TH1 *h_nSB_DY_ratio = (TH1*)h_nSB->Clone("h_nSB_DY_ratio");
    h_nSB_DY_ratio->Divide(h_nSB_DY_NW);

    TH1 *h_nSB_ratio2 = (TH1*)h_nSB_ZZ_ratio->Clone("h_nSB_DY_ratio2");
    h_nSB_ratio2->Multiply(h_nSB_DY_ratio);

    TH1 *h_nSB_DDest = (TH1*)h_nSB_Data_0b->Clone("h_nSB_DDest");
    h_nSB_DDest->Multiply(h_nSB_ratio);

    TH1 *h_nSB_uncertUp = (TH1*)h_nSB_Data_0b->Clone("h_nSB_uncertUp");
    TH1 *h_nSB_uncertDn = (TH1*)h_nSB_Data_0b->Clone("h_nSB_uncertDn");

    printf("%8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s\n", "bin", "data", "data up", "data dn", "ratio", "MC noWgt", "MC pred", "pred up", "pred dn", "% uncert");
    for(int i = 1; i <= h_nSB_ratio2->GetNbinsX(); ++i)
    {
        h_nSB_uncertUp->SetBinContent(i, h_nSB_Data_0b->GetBinErrorUp(i)  * h_nSB_ratio2->GetBinContent(i));
        h_nSB_uncertDn->SetBinContent(i, h_nSB_Data_0b->GetBinErrorLow(i) * h_nSB_ratio2->GetBinContent(i));
        printf("%8i  %8.0lf  +%7.4lf  -%7.4lf  %8.4lf  %8.4lf  %8.4lf  +%7.4lf  -%7.4lf  %8.1lf\n", i - 1, h_nSB_Data_0b->GetBinContent(i), h_nSB_Data_0b->GetBinErrorUp(i), h_nSB_Data_0b->GetBinErrorLow(i), h_nSB_ratio2->GetBinContent(i), h_nSB_NW->GetBinContent(i), h_nSB->GetBinContent(i), (h_nSB_Data_0b->GetBinErrorUp(i)  * h_nSB_ratio2->GetBinContent(i)), (h_nSB_Data_0b->GetBinErrorLow(i)  * h_nSB_ratio2->GetBinContent(i)), 100.0 * (h_nSB_Data_0b->GetBinErrorUp(i)  * h_nSB_ratio2->GetBinContent(i))/h_nSB->GetBinContent(i));
    }

    TH1 *h_nSB_uncertUp_ratio = (TH1*)h_nSB_uncertUp->Clone("h_nSB_uncertUp_ratio");
    TH1 *h_nSB_uncertDn_ratio = (TH1*)h_nSB_uncertDn->Clone("h_nSB_uncertDn_ratio");

    h_nSB_uncertUp_ratio->Divide(h_nSB);
    h_nSB_uncertDn_ratio->Divide(h_nSB);

    fout->cd();
    h_nSB_Data_0b->Write();
    h_nSB_NW->Write();
    h_nSB->Write();
    h_nSB_ratio->Write();
    h_nSB_DDest->Write();
    h_nSB_uncertUp->Write();
    h_nSB_uncertDn->Write();
    h_nSB_uncertUp_ratio->Write();
    h_nSB_uncertDn_ratio->Write();
    h_nSB_ZZ_ratio->Write();
    h_nSB_DY_ratio->Write();
    h_nSB_ratio2->Write();
}
