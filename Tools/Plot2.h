#ifndef PLOT2_h
#define PLOT2_h

//C++ libraries                                                                                                                                      
#include<iostream>
#include<sstream>
#include<vector>
#include<stdlib.h>
#include<iomanip>

//root libraries                                                                                                                                    
#include "TH1D.h"
#include "TFile.h"
#include "TString.h"
#include "THStack.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TPaveText.h"
#include "TLatex.h"
#include "TPad.h"
#include "TStyle.h"
#include "TLine.h"

class Leaf {
 public:
  TString name, xlabel, ylabel, plotTitle;
  int bins, rebin;
  float xmin,xmax;

  void SetValues(TString n, TString x, TString y, TString p, int b, float xm, float xp){
    name = n;
    xlabel = x;
    ylabel = y;
    bins = b;
    xmin = xm;
    xmax = xp;
    plotTitle = p;
  };

  void SetValues(TString n, TString x, TString y, TString p){
    name = n;
    xlabel = x;
    ylabel = y;
    plotTitle = p;
  }

  TString GetName() {return name;}
  TString GetPlotTitle() {return plotTitle;}
  TString GetXlabel() {return xlabel;}
  TString GetYlabel() {return ylabel;}
  int GetNbins() {return bins;}
  int GetRebin() {return rebin;}
  float GetMin() {return xmin;}
  float GetMax() {return xmax;}
};

TTree *GetTree(TString fileName, TString dir) {
  TFile *f = TFile::Open(fileName);
  TTree *tree = (TTree*)f->Get(dir);
  return tree;
}

void SetTextAndTitle(TString title, bool isDivision, bool fixMark) {
  float fontScale;

  if (isDivision && fixMark) fontScale = 1.3;

  else if (isDivision && !fixMark) fontScale = 1.2;

  if (!isDivision) fontScale = 6.5/8; 
  /*
  TPaveLabel *title = new TPaveLabel(0.4,0.97,0.6,0.91,Title,"nbNDC");//(0.4,0.97,0.6,0.9)                                                              
  title->SetFillColor(0);
  title->Draw("same");
  */
  TLatex mark;
  mark.SetNDC(true);

  //Draw CMS mark
  mark.SetTextAlign(11);
  mark.SetTextSize(0.042 * fontScale * 1.25);
  //mark.SetTextSize(0.04 * 1.1 * 8 / 6.5 * 1.25 * fontScale);
  mark.SetTextFont(61);

  if (fixMark == true){ 
    mark.DrawLatex(gPad->GetLeftMargin()+0.05, 1 - (gPad->GetTopMargin() - 0.017), "CMS");
    mark.SetTextSize(0.042 * fontScale);
    mark.SetTextFont(52);
    mark.DrawLatex(gPad->GetLeftMargin() + 0.14, 1 - (gPad->GetTopMargin() - 0.017), "Simulation");
  }
  else {
    mark.DrawLatex(gPad->GetLeftMargin(), 1 - (gPad->GetTopMargin() - 0.017), "CMS");
    mark.SetTextSize(0.042 * fontScale);
    mark.SetTextFont(52);
    mark.DrawLatex(gPad->GetLeftMargin() + 0.095, 1 - (gPad->GetTopMargin() - 0.017), "Simulation");
  }
  //mark.DrawLatex(gPad->GetLeftMargin() + 0.095, 1 - (gPad->GetTopMargin() - 0.017), "Preliminary");
  //mark.DrawLatex(gPad->GetLeftMargin() + 0.095, 1 - (gPad->GetTopMargin() - 0.017), "Supplementary");

  mark.SetTextFont(42);
  mark.SetTextAlign(31);
  mark.DrawLatex(1 - gPad->GetRightMargin(), 1 - (gPad->GetTopMargin() - 0.017), "t#bar{t}, 14 TeV (PU = 200)");
}

void SinglePlot(TH1D *hist, TString name, Leaf leaf, bool scale, bool log){

  TString xlabel = leaf.GetXlabel();
  TString ylabel = leaf.GetYlabel();
  bool fixMark = false;

  TCanvas *cv;

  if (scale == false && log == false) {
    cv = new TCanvas(leaf.GetName()+"_"+name,leaf.GetName()+"_"+name,1000,800);
    fixMark = true;
  }
  else cv = new TCanvas(leaf.GetName()+"_"+name,leaf.GetName()+"_"+name,900,800);

  gStyle->SetOptStat(0);
  hist->Draw("H");

  if (scale == true) hist->Scale(1/hist->Integral());

  if (log == true) cv->SetLogy();

  hist->GetXaxis()->SetTitle(xlabel);
  hist->GetYaxis()->SetTitle(ylabel);

  hist->SetLineWidth(2);
  gStyle->SetOptTitle(0);

  SetTextAndTitle(leaf.GetPlotTitle(),false,fixMark);
  cv->SaveAs("plots/"+leaf.GetName()+"_"+name+"_Single.pdf");
  cv->Close();
}

TH1D *CreateHistogram1(TTree *tree, TH1D *hist, TString fileName, TString dir, TString ID, Leaf leaf){
  //TFile *f = TFile::Open(fileName);
  tree = GetTree(fileName,dir);//(TTree*)f->Get(dir);

  tree->SetBranchStatus("*",0);
  tree->SetBranchStatus(leaf.GetName(),1);
  hist = new TH1D("h","h",leaf.GetNbins(),leaf.GetMin(),leaf.GetMax());

  vector<int>* var = new vector<int>;
  tree->SetBranchAddress(leaf.GetName(), &var);
  int nentries = tree->GetEntries();

  for(int ientry = 0; ientry<nentries; ientry++){
    tree->GetEntry(ientry);
    hist->Fill(var->at(0));
  }
  return hist;
}

void ComparisonPlot(TH1D *hist1, TH1D *hist2, Leaf leaf, TString name1, TString name2, bool scale, bool log){

  TString leafName = leaf.GetName();

  TString xlabel = leaf.GetXlabel();
  TString ylabel = leaf.GetYlabel();

  bool fixMark = false;

  TCanvas *cv;

  if (scale == false && log == false) {
    cv = new TCanvas(leaf.GetName()+"_"+name1+"_vs_"+name2,leaf.GetName()+"_"+name1+"_vs_"+name2,1000,800);
    fixMark = true;
  }
  else cv = new TCanvas(leaf.GetName()+"_"+name1+"_vs_"+name2,leaf.GetName()+"_"+name1+"_vs_"+name2,800,800);
  
  gStyle->SetOptStat(0);

  hist1->Draw("H");
  hist2->Draw("PHsame");

  if (scale == true){
    hist1->Scale(1/hist1->Integral());
    hist2->Scale(1/hist2->Integral());
  }

  if (log == true) cv->SetLogy();

  hist1->SetLineColor(4);
  hist2->SetLineColor(2);

  hist1->SetLineWidth(2);
  hist2->SetLineWidth(2);

  hist1->GetXaxis()->SetTitle(xlabel);
  hist1->GetYaxis()->SetTitle(ylabel);

  gStyle->SetOptTitle(0);

  hist2->SetMarkerStyle(20);
  hist2->SetMarkerSize(0.7);

  TLegend *legend = new TLegend(0.79,0.75,0.89,0.85);
  legend->AddEntry(hist1,name1,"L");
  legend->AddEntry(hist2,name2,"PL");
  legend->SetBorderSize(0);
  legend->Draw("same");

  SetTextAndTitle(leaf.GetPlotTitle(),false,fixMark);

  cv->SaveAs("plots/"+leafName+"_"+name1+"_vs_"+name2+"_Comparison.pdf");
  cv->Close();
}

/*
void CompareAll(TH1D *hist){

}*/

void RatioPlot(TH1D *hist1, TH1D *hist2, Leaf leaf,TString name1, TString name2, bool scale, bool log) {

  TString xlabel = leaf.GetXlabel();
  TString ylabel = leaf.GetYlabel();

  TCanvas *cv;

  bool fixMark = false;

  if (scale == false && log == false) {
    cv = new TCanvas(leaf.GetName()+"_"+name1+"_vs_"+name2+"ratio",leaf.GetName()+"_"+name1+"_vs_"+name2+"ratio",950,800);
    fixMark = true;
  }
  else cv = new TCanvas(leaf.GetName()+"_"+name1+"_vs_"+name2+"ratio",leaf.GetName()+"_"+name1+"_vs_"+name2+"ratio",800,800);

  gStyle->SetOptStat(0);

  //Create first TPad object                                                                                                                               
  TPad *pad1 = new TPad("pad1", "The pad 80% of the height",0.0,0.3,1.0,1.0);
  TPad *pad2 = new TPad("pad2", "The pad 20% of the height",0.0,0.05,1.0,0.3);

  //joins pad1's lower margin to pad2's top margin together                                                                                                
  pad1->SetBottomMargin(0.018);//0.018                                                                                                                     
  pad2->SetTopMargin(0);
  pad2->SetBottomMargin(0.2);//0.2                                                                                                                         

  //draw the 2 pads on cv canvas                                                                                                                           
  pad1->Draw();
  pad2->Draw();

  pad1->cd();

  hist1->Draw("H");
  hist2->Draw("PHsame");

  if (log == true) gPad->SetLogy();

  if (scale == true){
    hist1->Scale(1/hist1->Integral());
    hist2->Scale(1/hist2->Integral());
  }

  hist1->SetLineColor(4);
  hist2->SetLineColor(2);
  hist1->SetLineWidth(2);
  hist2->SetLineWidth(2);
  hist1->GetXaxis()->SetTitle(xlabel);
  hist1->GetYaxis()->SetTitle(ylabel);
  gStyle->SetOptTitle(0);
  hist2->SetMarkerStyle(20);
  hist2->SetMarkerSize(0.7);
  SetTextAndTitle(leaf.GetPlotTitle(),true,fixMark);

  //Add Legend                                                                                                                                             
  TLegend *legend = new TLegend(0.79,0.75,0.89,0.85);
  legend->AddEntry(hist1,name1,"L");
  legend->AddEntry(hist2,name2,"PL");
  legend->SetBorderSize(0);
  legend->Draw("same");

  hist1->GetYaxis()->SetTitle(ylabel);
  hist1->GetXaxis()->SetLabelOffset(999);

  gPad->Modified();
  pad2->cd();
  float xmax = hist1->GetXaxis()->GetXmax();
  float xmin = hist1->GetXaxis()->GetXmin();

  TLine *line = new TLine(xmin,1,xmax,1);
  line->SetLineStyle(6);
  line->SetLineColor(1);

  TH1D *ratio = new TH1D(*hist1);
  ratio->Divide(hist2);

  ratio->GetXaxis()->SetTitle(xlabel);
  ratio->GetYaxis()->SetTitle(name1+"/"+name2);

  if (leaf.GetName() == "nstub") ratio->SetAxisRange(0,2.1,"Y");
  else ratio->SetAxisRange(0.85,1.12,"Y"); 
  ratio->SetMarkerStyle(20);
  ratio->SetMarkerSize(1.2);
  ratio->Draw("E1X0");

  ratio->GetYaxis()->SetLabelSize(0.09);
  ratio->GetXaxis()->SetLabelSize(0.1);
  ratio->GetYaxis()->SetTitleSize(0.1);
  ratio->GetXaxis()->SetTitleSize(0.1);
  ratio->GetYaxis()->SetTitleOffset(0.4);
  ratio->GetXaxis()->SetLabelOffset(0.02);

  gStyle->SetOptTitle(0);

  line->Draw("same");

  gPad->Modified();

  cv->SaveAs("plots/"+leaf.GetName()+"_"+name1+"_vs_"+name2+"_ratio.pdf");
  cv->Close();
}

#endif
