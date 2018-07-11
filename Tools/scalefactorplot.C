#include "tdrstyle.h"
#include "ScaleFactors.h"

#include "TROOT.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TTree.h"
#include "TH1D.h"
#include "TF1.h"
#include "THStack.h"
#include "TLatex.h"
#include "TPad.h"
#include "TLegend.h"
#include "TGraphAsymmErrors.h"

void smartMax(const TH1* const h, const TLegend* const l, const TPad* const p, float& gmin, float& gmax, float& gpThreshMax)
{
    const bool isLog = p->GetLogy();
    float min = 9e99;
    float max = -9e99;
    float pThreshMax = -9e99;
    int threshold = static_cast<int>(h->GetNbinsX()*(l->GetX1() - p->GetLeftMargin())/((1 - p->GetRightMargin()) - p->GetLeftMargin()));

    for(int i = 1; i <= h->GetNbinsX(); ++i)
    {
        float bin = h->GetBinContent(i);
        if(bin > max) max = bin;
        else if(bin > 1e-10 && bin < min) min = bin;
        if(i >= threshold && bin > pThreshMax) pThreshMax = bin;
    }

    gpThreshMax = std::max(gpThreshMax, pThreshMax);
    gmax = std::max(gmax, max);
    gmin = std::min(gmin, min);
}

void makeplot(TFile* f, std::string hname)
{

    TH1D* h = (TH1D*)f->Get(hname.c_str());

    // Set the style
    setTDRStyle();
    gStyle->SetErrorX(0.5);
    // Prepare canvas
    TCanvas *c;
    float fontScale;
    c = new TCanvas("c1", "c1", 1200, 800);
    c->Divide(1, 1);
    c->cd(1);
    gPad->SetPad("p1", "p1", 0, 0, 1, 1, kWhite, 0, 0);
    gPad->SetBottomMargin(0.12);
    fontScale = 6.5 / 8;
    
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.06);
    gPad->SetTopMargin(0.07 * (8.0 / 6.5) * fontScale);
        
    // Use dummy histogram to set the axes and stuff 
    TH1 *dummy = new TH1F("dummy", "dummy", 1000, h->GetBinLowEdge(1), h->GetBinLowEdge(h->GetNbinsX()) + h->GetBinWidth(h->GetNbinsX()));
    dummy->SetStats(0);
    dummy->SetTitle(0);
    dummy->GetYaxis()->SetTitle("S_{DY} scale factor");
    dummy->GetYaxis()->SetTitleOffset(.9*1.05 / (fontScale));
    dummy->GetXaxis()->SetTitleOffset(1.05);
    dummy->GetXaxis()->SetTitle("N_{jets}");
    dummy->GetXaxis()->SetTitleSize(0.20 * 2 / 6.5 * fontScale);
    dummy->GetXaxis()->SetLabelSize(0.20 * 2 / 6.5 * fontScale);
    dummy->GetYaxis()->SetTitleSize(0.20 * 2 / 6.5 * fontScale);
    dummy->GetYaxis()->SetLabelSize(0.20 * 2 / 6.5 * fontScale);
    if(dummy->GetNdivisions() % 100 > 5) dummy->GetXaxis()->SetNdivisions(6, 5, 0);

    float max = 0.0, lmax = 0.0, min = 1.0e300, minAvgWgt = 1.0e300;
    int iSingle = 0, iRatio = 0;
    char legEntry[128];

    // Format the histogram
    h->SetLineColor(kBlue+2);
    h->SetLineWidth(3);
    iSingle++;
    
    dummy->GetXaxis()->SetRangeUser(4, 20);
    dummy->GetYaxis()->SetRangeUser(0.0, 2.0);
    dummy->Draw();
    
    h->Draw("Esame");

    fixOverlay();

    c->cd(1);
    //char lumistamp[128];
    //sprintf(lumistamp, "%.1f fb^{-1} at 13 TeV", 2156. / 1000.0);
    //TLatex mark;
    //mark.SetTextSize(0.042 * fontScale);
    //mark.SetTextFont(42);
    //mark.SetNDC(true);
    //mark.DrawLatex(gPad->GetLeftMargin(), 0.95, "CMS Preliminary");
    //mark.SetTextAlign(31);
    //mark.DrawLatex(1 - gPad->GetRightMargin(), 0.95, lumistamp);
    
    char lumistamp[128];
    sprintf(lumistamp, "%.1f fb^{-1} (13 TeV)",  35867 / 1000.0);

    TLatex mark;
    mark.SetNDC(true);

    //Draw CMS mark
    mark.SetTextAlign(11);
    mark.SetTextSize(0.042 * fontScale * 1.25);
    //mark.SetTextSize(0.04 * 1.1 * 8 / 6.5 * 1.25 * fontScale);
    mark.SetTextFont(61);
    mark.DrawLatex(gPad->GetLeftMargin(), 1 - (gPad->GetTopMargin() - 0.017), "CMS"); // #scale[0.8]{#it{Preliminary}}");
    mark.SetTextSize(0.042 * fontScale);
    //mark.SetTextSize(0.04 * 1.1 * 8 / 6.5 * fontScale);
    mark.SetTextFont(52);
    //mark.DrawLatex(gPad->GetLeftMargin() + 0.065, 1 - (gPad->GetTopMargin() - 0.017), "Supplementary");
    mark.DrawLatex(gPad->GetLeftMargin() + 0.065, 1 - (gPad->GetTopMargin() - 0.017), "Preliminary");

    //Draw lumistamp
    mark.SetTextFont(42);
    mark.SetTextAlign(31);
    mark.DrawLatex(1 - gPad->GetRightMargin(), 1 - (gPad->GetTopMargin() - 0.017), lumistamp);

    fixOverlay();
    char outname[128];
    sprintf(outname, "SF_njet_%s.pdf", hname.c_str());
    c->Print(outname);
    sprintf(outname, "SF_njet_%s.png", hname.c_str());
    c->Print(outname);

    c->Close();
}

int main(int argc, char* argv[])
{
    // Get the relevant information
    TFile* f1 = TFile::Open("/uscms_data/d3/snorberg/CMSSW_8_0_23_patch1/src/ZInvisible/Tools/dataMCweights.root");
    std::vector<std::string> hnames = {//"DataMC_nj_elmuZinv_",
				       //"DataMC_nj_elmuZinv_0b_ht200_dphi",
				       //"DataMC_nj_elmuZinv_g1b_ht200_dphi",
				       "DataMC_nj_muZinv_g1b_loose0_mt2_MET",
				       "DataMC_nj_muZinv_0b_loose0_mt2_MET",
				       "DataMC_nj_muZinv_loose0_mt2_MET"};
    for(std::string const& hname : hnames)
    {
	makeplot(f1, hname);
    }

    f1->Close();

    return 0;

}
