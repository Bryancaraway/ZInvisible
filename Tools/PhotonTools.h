#ifndef PHOTONTOOLS_H
#define PHOTONTOOLS_H

#include <iostream>
#include<sstream>
#include<iomanip>
#include <string>
#include <set>
#include <vector>

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
#include "TMath.h"
#include "TROOT.h"
#include "Math/VectorUtil.h"

namespace PhotonConsts
{
  struct IDselection {
    float HoE, sieie, pfCHadIso,pfNHadIso, factor1, factor2, pfGIso, factor3;
  };

  //barrel
  const IDselection looseIDbarrel{0.105, 0.0103, 2.839, 9.188, 0.0126, 0.000026, 2.956, 0.0035};
  const IDselection mediumIDbarrel{0.035, 0.0103, 1.416, 2.491, 0.0126, 0.000026, 2.952, 0.0040};
  const IDselection tightIDbarrel{0.020, 0.0103, 1.158, 1.267, 0.0126, 0.000026, 2.065, 0.0035};
  //endcap
  const IDselection looseIDendcap{0.029, 0.0276, 2.150, 10.471, 0.0119, 0.000025, 4.895, 0.0040};
  const IDselection mediumIDendcap{0.027, 0.0271, 1.012, 9.131, 0.0119, 0.000025, 4.095, 0.0040};
  const IDselection tightIDendcap{0.025, 0.0271, 0.575, 8.916, 0.0119, 0.000025, 3.272, 0.0040};

  /*
  //Photon Variable Map
  std::map<TString,TString> varNameMap;
  //DY control region
  varNameMap["cleanMetPt"] = "met";
  varNameMap["HTZinv"] = "ht";
  varNameMap["nTopCandSortedCntZinv"] = "nt";
  varNameMap["best_had_brJet_MT2Zinv"] = "mt2";
  varNameMap["cntCSVSZinv"] = "nb";
  varNameMap["cntNJetsPt30Eta24Zinv"] = "nj";
  varNameMap["cutMuPt1"] = "mu1pt";
  varNameMap["nSearchBin"] = "nSearchBin";
  //GJets control region
  varNameMap["photonMet"] = "MetGamma";
  varNameMap["nJets"] = "Photon_nj";
  */
}

namespace PhotonFunctions
{
  
  bool passPhoton(const TLorentzVector& photon){
    const float minPt = 100, barrelMax = 1.4442, endcapMin = 1.566, endcapMax = 2.5;
    float perPhotonPt = photon.Pt(), perPhotonEta = photon.Eta();
    return (minPt == -1 || perPhotonPt > minPt)
      && ((barrelMax == -1 || fabs(perPhotonEta) < barrelMax)
	  || ((endcapMin == -1 || fabs(perPhotonEta) > endcapMin)
	      && (endcapMax == -1 || fabs(perPhotonEta) < endcapMax)));
  }

  bool isBarrelECAL(const TLorentzVector& photon){
    const float barrelMax = 1.4442;
    float perPhotonEta = photon.Eta();
    return (barrelMax == -1 || fabs(perPhotonEta) <= barrelMax);
  }
  
  bool isEndcapECAL(const TLorentzVector& photon){
    const float endcapMin = 1.566, endcapMax = 2.5;
    float perPhotonEta = photon.Eta();
    return (endcapMin == -1 || fabs(perPhotonEta) >= endcapMin)
      || (endcapMax == -1 || fabs(perPhotonEta) <= endcapMax);
  }

  bool isGenMatched(const TLorentzVector& photon, std::vector<TLorentzVector> genPhoton){
    float RecoPt = photon.Pt();
    bool match = false;
    
    //std::cout << "genPhotons: " << genPhoton.size() << std::endl;
    for(int i = 0; i < genPhoton.size(); i++){
      float deltaR = ROOT::Math::VectorUtil::DeltaR(genPhoton[i],photon);
      float GenPt = genPhoton[i].Pt();
      float temp_ratio = GenPt/RecoPt;
      /*
      std::cout << "igenPhoton: " << i+1 << std::endl;
      std::cout << "Reco Photon Pt: " << RecoPt << std::endl;
      std::cout<< "Gen Photon Pt: " << GenPt<< std::endl;
      std::cout<< "GenPt/RecoPt: " << temp_ratio << std::endl;
      std::cout << "deltaR: " << deltaR << std::endl;
      */
      if (temp_ratio > 0.5 && temp_ratio < 2.0 && deltaR < 0.1){
	match = true;
	break;
      }
    }
    //if (match) std::cout << "pass"<< std::endl << std::endl;
    //else std::cout << "fake"<< std::endl << std::endl;
      return match;
    
  }

  bool isDirectPhoton(const TLorentzVector& photon, std::vector<TLorentzVector> genParton){

    bool isDirect = false;

    for(int i = 0; i < genParton.size(); i++){
      float deltaR = ROOT::Math::VectorUtil::DeltaR(photon,genParton[i]);
      //std::cout << "deltaR: " << deltaR << std::endl;
      if (deltaR > 0.4){
        isDirect = true;
        break;
      }
    }
    //if (isDirect) std::cout << "passDirect" << std::endl << std::endl;
    //else std::cout << "fragmentation" << std::endl << std::endl;
    return (isDirect);
  }

  bool isFragmentationPhoton(const TLorentzVector& photon, std::vector<TLorentzVector> genParton){
    bool isFrag = false;

    for(int i = 0; i < genParton.size(); i++){
      float deltaR = ROOT::Math::VectorUtil::DeltaR(photon,genParton[i]);
      //std::cout << "deltaR: " << deltaR << std::endl;
      if (deltaR < 0.4){
        isFrag = true;
        break;
      }
    }
    //if (isFrag) std::cout << "passFragmentation" << std::endl << std::endl;
    return (isFrag);
  }

  void prepareHist(TH1D* hist){

    hist->Sumw2();
    hist->Integral();
  }

  TString GetHistName(TString varName, TString cut, TString tag, TString type){

    TString genericHist, histName;

    if(type == "stack" || type == "data") genericHist = "%s/dataMC_Photon_%s_%s%s%s%s%s";
    
    if(type == "single") genericHist = "%s/%sPhoton_%s_%s%s%s%s%s";

    if(varName == "nJets") {
      histName.Form(genericHist, varName.Data(), "nj", cut.Data(), varName.Data(), varName.Data(), tag.Data(), type.Data()); 
    }

    TString var1,var2;
    var1 = varName+"pt";
    var2 = varName+"(pt)";

    if(varName == "cutPhotons") {
      histName.Form(genericHist, varName.Data(), "", "Pt", cut.Data(), var1.Data(), var2.Data(), tag.Data(), type.Data());
    }

    if(varName == "directPhotons") {
      histName.Form(genericHist, varName.Data(), "Direct_", "Pt", cut.Data(), var1.Data(), var2.Data(), tag.Data(), type.Data());
     }

    if(varName == "fragmentationQCD") {
      histName.Form(genericHist, varName.Data(), "Fragmentation_", "Pt", cut.Data(), var1.Data(), var2.Data(), tag.Data(), type.Data());
     }

    if(varName == "fakePhotons") {
      histName.Form(genericHist, varName.Data(), "Fake_QCD", "Pt", cut.Data(), var1.Data(), var2.Data(), tag.Data(), type.Data());
     }

    return histName;
    
  }

  TString GetName(TString genstring, TString name, TString var){

    std::map<TString,TString> varNameMap;
    //DY control region                                                                                                                
    varNameMap["cleanMetPt"] = "met";
    varNameMap["HTZinv"] = "ht";
    varNameMap["nTopCandSortedCntZinv"] = "nt";
    varNameMap["best_had_brJet_MT2Zinv"] = "mt2";
    varNameMap["cntCSVSZinv"] = "nb";
    varNameMap["cntNJetsPt30Eta24Zinv"] = "nj";
    varNameMap["cutMuPt1"] = "mu1pt";
    varNameMap["nSearchBin"] = "nSearchBin";
    //GJets control region                                                                                                            
    //varNameMap["cntNJetsPt30Eta24Zinv"] = "Photon_nj";
    varNameMap["photonMet"] = "met";
    //varNameMap["nJets"] = "Photon_nj";

    TString tempName, tempMap;

    tempMap = varNameMap[name];

    tempName.Form(genstring,name.Data(),tempMap.Data(),name.Data(),name.Data(),var.Data());

    return tempName;

  }

  std::vector<TH1D*> GetHistVec(std::vector<TString> histogramNames, TString fileName) {

    TFile *file = TFile::Open(fileName);

    std::vector<TH1D*> histograms;

    for(int i = 0; i < histogramNames.size(); i++){ 

      histograms.push_back((TH1D*)file->Get(histogramNames[i]));

    }

    return histograms;

  }

  void Subtract(TH1D* data, std::vector<TH1D*> backgrounds){

    for(int i = 1; i < backgrounds.size(); i++){
      data->Add(backgrounds[i],-1.0);
    } 
    data->Integral();
  }

  void Aesthetics(TH1D* hist, TString name){

    //TStyle* style = new TStyle("style","Style Object");

    gStyle->SetPadTickX(1);
    gStyle->SetPadTickY(1);

    hist->SetLineWidth(2);
    hist->SetLineColor(1);

    hist->SetMarkerStyle(20);
    hist->SetMarkerSize(0.8);
    hist->SetMarkerColor(2);
    if (name == "Purity_") hist->GetYaxis()->SetTitle("Purity");
    if (name == "FakeRate_"){
      hist->GetYaxis()->SetTitle("Fake Rate");
      hist->GetYaxis()->SetTitleOffset(1.5);
    }
    if (name == "ShapeCorr_") hist->GetYaxis()->SetTitle("S_{#gamma} scale factor");
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);

  }

  void CMSmark(){
    TLatex mark;
    mark.SetNDC(true);

    float fontScale = 6.5/8;

    mark.SetTextAlign(11);
    mark.SetTextSize(0.042 * fontScale * 1.25);
    mark.SetTextFont(61);
    mark.DrawLatex(gPad->GetLeftMargin()/*+0.05*/, 1 - (gPad->GetTopMargin() - 0.017), "CMS");
    mark.SetTextSize(0.042 * fontScale);
    mark.SetTextFont(52);
    mark.DrawLatex(gPad->GetLeftMargin()+0.095/*+0.14*/, 1 - (gPad->GetTopMargin() - 0.017), "Preliminary");
    mark.SetTextFont(42);
    mark.SetTextAlign(31);
    mark.DrawLatex(1 - gPad->GetRightMargin(), 1 - (gPad->GetTopMargin() - 0.017), "35.9 fb^{-1} (13 TeV)");
  }

  void makeRatio(TH1D *hist1, TH1D *hist2, TString var){

    //hist1->Scale(1/hist1->Integral());
    //hist2->Scale(1/hist2->Integral());

    hist1->SetLineColor(1);
    hist2->SetLineColor(1);

    hist1->SetMarkerColor(2);
    hist2->SetMarkerColor(3);

    hist1->Rebin(2);
    hist2->Rebin(2);

    hist1->SetLineWidth(2);
    hist2->SetLineWidth(2);

    hist1->SetMarkerStyle(20);
    hist1->SetMarkerSize(0.8);

    hist2->SetMarkerStyle(20);
    hist2->SetMarkerSize(0.8);

    TCanvas *c1 = new TCanvas("c1","c1",900,1000);

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

    hist2->Draw("P");
    hist1->Draw("Psame");

    //hist1->SetAxisRange(200,1500,"X");
    hist2->SetAxisRange(0,3,"Y");

    CMSmark();

    //gPad->SetLogy();

    //Add Legend                                                                                                                                                                    
    TLegend *legend = new TLegend(0.79,0.75,0.89,0.85);
    legend->AddEntry(hist1,"Z#rightarrow#mu#mu","PL");
    legend->AddEntry(hist2,"#gamma+ jets","PL");
    legend->SetBorderSize(0);
    legend->Draw("same");

    hist2->GetYaxis()->SetTitleSize(0.05);
    hist2->GetYaxis()->SetTitle("data/simulation");
    //hist2->GetYaxis()->SetTitleSize(10);
    hist2->GetXaxis()->SetLabelOffset(999);

    //hist2->SetAxisRange(3,14,"X");

    gPad->Modified();
    pad2->cd();
    float xmax = hist2->GetXaxis()->GetXmax();
    float xmin = hist2->GetXaxis()->GetXmin();

    TLine *line = new TLine(2,1,16,1);
    line->SetLineStyle(1);
    line->SetLineColor(1);

    TH1D *ratio = (TH1D*)hist1->Clone();
    //ratio->Print("all"); 
    ratio->Divide(ratio,hist2,1.0,1.0,"B");
    //ratio->Print("all"); 

    ratio->SetAxisRange(-2,2,"Y");

    ratio->SetMarkerStyle(20);
    ratio->SetMarkerSize(1.2);
    ratio->SetMarkerColor(1);
    ratio->Draw("E1X0");

    ratio->GetYaxis()->SetLabelSize(0.09);
    ratio->GetXaxis()->SetLabelSize(0.1);
    ratio->GetYaxis()->SetTitleSize(0.15);
    ratio->GetXaxis()->SetTitleSize(0.15);
    ratio->GetYaxis()->SetTitleOffset(0.27);
    ratio->GetXaxis()->SetLabelOffset(0.02);
    ratio->GetXaxis()->SetTitleOffset(0.5);

    //hist1->Print("all");
    //hist2->Print("all");
    //ratio->Print("all");

    //ratio->SetMinimum(-1.8);
    //ratio->SetMaximum(1.8);

    //ratio->SetAxisRange(200,1500,"X");

    ratio->GetYaxis()->SetTitle("Z/#gamma");
    ratio->GetXaxis()->SetTitle(var);

    gStyle->SetOptTitle(0);

    line->Draw("same");
    //CMSmark();

    gPad->Modified();
    c1->SaveAs("photonPlots/ZtoGamma_ratio_"+var+".pdf");

    delete c1;
  }

  void SavePlot(TH1D* hist, TString name, TString cut){

    TCanvas *c1 = new TCanvas("c1"+name+cut,"c1"+name+cut,800,800);

    Aesthetics(hist,name);

    hist->SetAxisRange(-1,1.5,"Y");
    hist->SetAxisRange(0,15,"X");

    hist->Draw();
    CMSmark();

    c1->SetGridx();
    c1->SetGridy();
    
    c1->SaveAs("photonPlots/"+name+cut+".pdf");

    delete c1;
  }

  void makeRootFile(std::vector<TH1D*> hist){

    gROOT->Reset();

    TFile *MyFile = new TFile("dataMCreweight.root","RECREATE");

    for(int i = 0; i < hist.size(); i++){
      hist[i]->Write();
    }

    MyFile->Close();

  }

  void reweight(TH1D* hist, TH1D* weights){

    for(int i = 0; i < hist->GetNbinsX(); i++){

      int sfbin = weights->FindBin(hist->GetBinCenter(i+1));
      float sf = weights->GetBinContent(sfbin);
      float sf_e = weights->GetBinError(sfbin);

      if (sfbin == 0) continue;

      float bc_old = hist->GetBinContent(i+1);
      float be_old = hist->GetBinError(i+1);

      if (bc_old == 0) continue;

      float bc = bc_old*sf;
      float be = bc*TMath::Sqrt( (be_old/bc_old)*(be_old/bc_old) + (sf_e/sf)*(sf_e/sf) );

      hist->SetBinContent(i+1,bc);
      hist->SetBinError(i+1,be);
    }

  }

  class SelectID
  {
  public:
    
    float HoverE, sietaieta, pfChargedHadronIso, pfNeutralHadronIso, pfGammaIso;
    
    void getIDvar(float HoE, float sieie, float pfCHadIso, float pfNhadIso, float pfPhoIso) {
      HoverE = HoE;
      sietaieta = sieie;
      pfChargedHadronIso = pfCHadIso;
      pfNeutralHadronIso = pfNhadIso;
      pfGammaIso = pfPhoIso;
    }
    
    float GetHoverE() {return HoverE;}
    float GetSieie() {return sietaieta;}
    float GetChargedHadIso() {return pfChargedHadronIso;}
    float GetNeutralHadIso() {return pfNeutralHadronIso;}
    float GetGammaIso() {return pfGammaIso;}
  };

  float pfNhadronIsoCalc(const TLorentzVector& photon, const PhotonConsts::IDselection& paramArr){
    float photonPt = photon.Pt();
    float product = paramArr.pfNHadIso + paramArr.factor1*photonPt+paramArr.factor2*photonPt*photonPt;
    return product;
  }
  
  float pfGammaIsoCalc(const TLorentzVector& photon, const PhotonConsts::IDselection& paramArr){
    float photonPt = photon.Pt();
    float product = paramArr.pfGIso + paramArr.factor3*photonPt;
    return product;
  }
  
  bool GetPhotonID(const TLorentzVector& photon, SelectID& IDobj, TString decision){
    const float HoE = IDobj.GetHoverE();
    const float sieie = IDobj.GetSieie();
    const float pfCHadIso = IDobj.GetChargedHadIso();
    const float pfNhadIso = IDobj.GetNeutralHadIso(); 
    const float pfPhoIso = IDobj.GetGammaIso();
    
    bool passIDbarrel = false;
    bool passIDendcap = false;
    bool passIsoBarrel = false;
    bool passIsoEndcap = false;
    
    PhotonConsts::IDselection barrelParams;
    PhotonConsts::IDselection endcapParams;
    
    if (decision == "Loose"){
      barrelParams = PhotonConsts::looseIDbarrel;
      endcapParams = PhotonConsts::looseIDendcap;
    }
    
    else if (decision == "Medium"){
      barrelParams = PhotonConsts::mediumIDbarrel;
      endcapParams = PhotonConsts::mediumIDendcap;
    }
    
    else if (decision == "Tight"){
      barrelParams = PhotonConsts::tightIDbarrel;
      endcapParams = PhotonConsts::tightIDendcap;
    }

    if(isBarrelECAL(photon)){
      passIDbarrel = (HoE < barrelParams.HoE && sieie < barrelParams.sieie);
      passIsoBarrel = (pfCHadIso < barrelParams.pfCHadIso && pfNhadIso < pfNhadronIsoCalc(photon, barrelParams) 
		       && pfPhoIso < pfGammaIsoCalc(photon, barrelParams));
    }

      else if(isEndcapECAL(photon)){
	passIDendcap = (HoE < endcapParams.HoE && sieie < endcapParams.sieie);
	passIsoBarrel = (pfCHadIso < endcapParams.pfCHadIso && pfNhadIso < pfNhadronIsoCalc(photon, endcapParams)
			 && pfPhoIso < pfGammaIsoCalc(photon, endcapParams));
      }

    return ((passIDbarrel && passIsoBarrel) || (passIDendcap && passIsoEndcap));
  }

}
    
#endif
