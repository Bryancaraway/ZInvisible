#include "PhotonTools.h"

using namespace PhotonFunctions;

void PhotonMethod() {

  TString file1 = "condor/photonPlots9.root";
  TString file2 = "condor/DYCR6.root";
  //TString file2 = "MC_Data_normalized_Shape.root";

  TFile *f1 = TFile::Open(file1);
  TFile *f2 = TFile::Open(file2);

  TString genNameDY ="%s/DataMCw_SingleMuon_%s_muZinv_0b_blnotag%s%s%sstack";                                                               
  TString genDataDY ="%s/DataMCw_SingleMuon_%s_muZinv_0b_blnotag%s%s%sdata";
  TString genNameDY_inc ="%s/DataMC_SingleMuon_%s_muZinv_blnotag%s%s%sstack";
  TString genDataDY_inc ="%s/DataMC_SingleMuon_%s_muZinv_blnotag%s%s%sdata";
  TString genNameGJets = "%s/DataMC_SinglePhoton_%s_LooseLepVeto%s%s%sstack";                                                                    
  TString genDataGJets = "%s/DataMC_SinglePhoton_%s_LooseLepVeto%s%s%sdata"; 

  //Don't display canvas when drawing                                                                                    
  gROOT->SetBatch(kTRUE);

  /*-------------------------------*/
  /*-- ZtoGamma comparison study --*/
  /*-------------------------------*/
  
  TString tempName, test1, test2;

  std::map<TString,TString> varNameMap;
  //DY control region                                                                                                                
  varNameMap["cleanMetPt"] = "p_{T}^{miss}";
  varNameMap["cntNJetsPt30Eta24Zinv"] = "N_{jets}";
  //GJets control region                                                                                                              
  varNameMap["photonMet"] = "p_{T}^{miss}";
  varNameMap["nJets"] = "N_{jets}";

  std::vector<TString> sampleDY;
  sampleDY.push_back("DY");
  sampleDY.push_back("t#bar{t}");
  sampleDY.push_back("Single t");
  sampleDY.push_back("Rare ");

  std::vector<TString> sampleGJets;
  sampleGJets.push_back("#gamma+ jets");
  sampleGJets.push_back("QCD");
  sampleGJets.push_back("t#bar{t}#gamma");
  sampleGJets.push_back("W(l#nu)+jets");
  sampleGJets.push_back("t#bar{t}");
  sampleGJets.push_back("Diboson");
  sampleGJets.push_back("tW");
  sampleGJets.push_back("Rare");
  //sampleGJets.push_back("t#bar{t}Z");

  std::cout << GetName(genDataGJets,"photonMet","Data") << std::endl;
  for(int k = 0; k < sampleGJets.size(); k++){
    std::cout << GetName(genNameGJets,"photonMet",sampleGJets[k]) << std::endl;
  }

  std::vector<TString> varNameDY;
  varNameDY.push_back("cleanMetPt");
  varNameDY.push_back("cntNJetsPt30Eta24Zinv");

  std::vector<TString> varNameGJets;
  varNameGJets.push_back("photonMet");
  varNameGJets.push_back("cntNJetsPt30Eta24Zinv");

  TH1D* dataDY[varNameDY.size()];
  TH1D* dataGJets[varNameGJets.size()];

  std::vector<TString> backgroundsZmumu;
  std::vector<TString> backgroundsGJets;

  for(int i = 0; i < varNameDY.size(); i++){
    dataDY[i] = (TH1D*)f2->Get(GetName(genDataDY_inc,varNameDY[i],"Data"));
    dataGJets[i] = (TH1D*)f1->Get(GetName(genDataGJets,varNameGJets[i],"Data"));

    for(int j = 0; j < sampleDY.size(); j++){
      backgroundsZmumu.push_back(GetName(genNameDY_inc,varNameDY[i],sampleDY[j]));
    }
    std::vector<TH1D*> histsDY = GetHistVec(backgroundsZmumu, file2);

    for(int k = 0; k < sampleGJets.size(); k++){
      backgroundsGJets.push_back(GetName(genNameGJets,varNameGJets[i],sampleGJets[k]));
    }
    std::vector<TH1D*> histsGJets = GetHistVec(backgroundsGJets, file1);

    Subtract(dataDY[i],histsDY);
    Subtract(dataGJets[i],histsGJets);
    //Subtract(dataDY[i],histsDY);

    histsDY[0]->Integral();
    histsGJets[0]->Integral();
    dataDY[i]->Divide(dataDY[i],histsDY[0],1.0,1.0,"B");
    dataGJets[i]->Divide(dataGJets[i],histsGJets[0],1.0,1.0,"B");

    makeRatio(dataDY[i],dataGJets[i],varNameMap[varNameDY[i]]);

    backgroundsZmumu.clear();
    backgroundsGJets.clear();
    histsDY.clear();
    histsGJets.clear();

    delete dataDY[i];
    delete dataGJets[i];
  }
  
  /*----------------------------------*/
  /*-- Fake Rate and Purity Studies --*/
  /*----------------------------------*/
  /*
  std::vector<TString> varName;
  //varName.push_back("cutPhotons");
  varName.push_back("directPhotons");
  varName.push_back("fakePhotons");
  varName.push_back("fragmentationQCD");

  std::vector<TString> cutName;
  cutName.push_back("empty");
  cutName.push_back("LooseLepVeto");
  cutName.push_back("MediumLepVeto");
  cutName.push_back("TightLepVeto");

  for(int i = 0; i < cutName.size(); i++){

    TH1D *direct = (TH1D*)f1->Get(GetHistName("directPhotons",cutName[i],"GJets","single"));
    TH1D *fragmentation = (TH1D*)f1->Get(GetHistName("fragmentationQCD",cutName[i],"QCD","single"));
    TH1D *fakes = (TH1D*)f1->Get(GetHistName("fakePhotons",cutName[i],"QCD","single"));

    prepareHist(direct);
    prepareHist(fragmentation);
    prepareHist(fakes);
  
    TH1D* allPhotons = (TH1D*)direct->Clone();
    TH1D* purity = (TH1D*)direct->Clone();
    TH1D* fakeRate = (TH1D*)fakes->Clone();
    
    allPhotons->Add(fragmentation);
    allPhotons->Add(fakes);
    purity->Add(fragmentation);
    
    fakeRate->Divide(fakeRate,allPhotons,1.0,1.0,"B");
    purity->Divide(purity,allPhotons,1.0,1.0,"B");

    //SavePlot(purity,"Purity_",cutName[i]);
    //SavePlot(fakeRate,"FakeRate_",cutName[i]);
  } //End of fake rate and purity studies
  */
  /*-----------------------*/
  /*-- RNorm Calculation --*/
  /*-----------------------*/
  /*
  //TFile* f3 = TFile::Open("MC_Data_normalized_Shape.root");

  TString folder = "cntCSVSZinv/";

  TH1D *dataHistDY = (TH1D*)f2->Get(folder+"DataMC_SingleMuon_nb_Wgt_muZinv_blnotagcntCSVSZinvcntCSVSZinvDatadata");
  TH1D *dYHist     = (TH1D*)f2->Get(folder+"DataMC_SingleMuon_nb_Wgt_muZinv_blnotagcntCSVSZinvcntCSVSZinvDYstack");

  //std::cout << (TH1D*)f2->Get(folder+"DataMC_SingleMuon_nj_muZinv_0b_blnotagcntNJetsPt30Eta24ZinvcntNJetsPt30Eta24ZinvDatadata") << std::endl;

  std::vector<TString> histNames;
  histNames.push_back(folder+"DataMC_SingleMuon_nb_Wgt_muZinv_blnotagcntCSVSZinvcntCSVSZinvt#bar{t}stack");
  histNames.push_back(folder+"DataMC_SingleMuon_nb_Wgt_muZinv_blnotagcntCSVSZinvcntCSVSZinvSingle tstack");
  histNames.push_back(folder+"DataMC_SingleMuon_nb_Wgt_muZinv_blnotagcntCSVSZinvcntCSVSZinvRare stack");

  TH1D *backgroundsDY[histNames.size()];

  for(int i = 0; i < histNames.size(); i++){
    backgroundsDY[i] = (TH1D*)f2->Get(histNames[i]);
    //std::cout << backgroundsDY[i]->GetBinContent(1) << std::endl;
    prepareHist(backgroundsDY[i]);
    //std::cout << i << std::endl;
    dataHistDY->Add(backgroundsDY[i],-1.0);
  }

  dataHistDY->Divide(dataHistDY,dYHist,1.0,1.0,"B");
  TH1D* newh = (TH1D*)dataHistDY->Clone();

  //newh->Draw();

  std::cout << "Rnorm = " << newh->GetBinContent(1) << " +- " << newh->GetBinError(1) << std::endl;
  */
  /*------------------*/
  /*-- Njet weights --*/
  /*------------------*/

  //TH1D *dataHist = (TH1D*)f1->Get(GetHistName("cntNJetsPt30Eta24Zinv","LooseLepVeto","Data","data"));
  TH1D *gJetsHist = (TH1D*)f1->Get(GetName(genNameGJets,"cntNJetsPt30Eta24Zinv","#gamma+ jets"));
  TH1D *dataHist = (TH1D*)f1->Get(GetName(genDataGJets,"cntNJetsPt30Eta24Zinv","Data"));

  TH1D* gJets = (TH1D*)gJetsHist->Clone();
  TH1D* data = (TH1D*)dataHist->Clone();
  //TH1D *gJetsHist = (TH1D*)f1->Get(GetHistName("cntNJetsPt30Eta24Zinv","LooseLepVeto","GJets","stack"));
  //std::cout << GetName(genDataGJets,"cntNJetsPt30Eta24Zinv","Data") << std::endl;
  //std::cout << GetName(genNameGJets,"cntNJetsPt30Eta24Zinv","#gamma+ jets") << std::endl;
  //std::cout << GetHistName("cntNJetsPt30Eta24Zinv","LooseLepVeto","GJets","stack") << std::endl;
  //dataHist->Integral();
  gJetsHist->Integral();
  //std::cout << "test" << std::endl;
  //dataHist->Scale(1/dataHist->Integral());
  //gJetsHist->Scale(1/gJetsHist->Integral());
  //prepareHist(dataHist);
  //prepareHist(gJetsHist);

  //Background names                                                                                                                                                               
  std::vector<TString> backgroundTag;
  backgroundTag.push_back("QCD");
  backgroundTag.push_back("t#bar{t}#gamma");
  backgroundTag.push_back("W(l#nu)+jets");
  backgroundTag.push_back("t#bar{t}");
  backgroundTag.push_back("Diboson");
  backgroundTag.push_back("tW");
  backgroundTag.push_back("Rare");
  //backgroundTag.push_back("t#bar{t}Z");

  TH1D *backgrounds[backgroundTag.size()];

  for(int i = 0; i < backgroundTag.size(); i++){
    //std::cout << GetName(genNameGJets,"cntNJetsPt30Eta24Zinv",backgroundTag[i]) << std::endl;
    backgrounds[i] = (TH1D*)f1->Get(GetName(genNameGJets,"cntNJetsPt30Eta24Zinv",backgroundTag[i]));
    dataHist->Add(backgrounds[i],-1.0);
  }

  dataHist->Integral();

  TH1D* normWeight = (TH1D*)dataHist->Clone();
  normWeight->SetName("normWeight");
  normWeight->Divide(normWeight,gJetsHist,1.0,1.0,"B");

  gJetsHist->Scale(1/gJetsHist->Integral());
  dataHist->Scale(1/dataHist->Integral());

  dataHist->Divide(dataHist,gJetsHist,1.0,1.0,"B");
  TH1D* nJetReweight = (TH1D*)dataHist->Clone();
  nJetReweight->SetName("nJetReweight");
  SavePlot(dataHist,"ShapeCorr_","LooseLepVeto");
  //end of shape corrections calculation

  //Shape Scale Factor validation
  PhotonFunctions::reweight(gJets,dataHist);

  for(int i = 0; i < backgroundTag.size(); i++){
    backgrounds[i] = (TH1D*)f1->Get(GetName(genNameGJets,"cntNJetsPt30Eta24Zinv",backgroundTag[i]));
    gJets->Add(backgrounds[i]);
  }

  prepareHist(gJets);
  prepareHist(data);

  data->Divide(data,gJets,1.0,1.0,"B");
  SavePlot(data,"ShapeValidation_","LooseLepVeto");

  std::vector<TH1D*> sfHists;
  sfHists.push_back(nJetReweight);
  sfHists.push_back(normWeight);

  makeRootFile(sfHists);
  //f1->Close();

  /*-----------------------*/
  /*-- RNorm Calculation --*/
  /*-----------------------*/
  std::vector<TString> sampleDYnorm;                                                                                                           
  sampleDY.push_back("t#bar{t}");                                                                                                          
  sampleDY.push_back("Single t");                                                                                                          
  sampleDY.push_back("Rare ");

  int bin = 1;
  //TString varName = "cntNJetsPt30Eta24Zinv";
  TString varName = "cntCSVSZinv";                                                                                                         

  TH1D *dataHistDY = (TH1D*)f2->Get(GetName(genDataDY,varName,"Data"));                       
  TH1D *dYHist     = (TH1D*)f2->Get(GetName(genNameDY,varName,"DY"));

  //PhotonFunctions::reweight(dYHist,nJetReweight);

  TH1D *backgroundsDY[sampleDYnorm.size()];
                                                                                                                                           
  for(int i = 0; i < sampleDYnorm.size(); i++){                                                                                               
    backgroundsDY[i] = (TH1D*)f2->Get(GetName(genNameDY,varName,sampleDYnorm[i]));                                            
    dataHistDY->Add(backgroundsDY[i],-1.0);                                                                                                
  }                                                                                                                                        

  float dataSum = dataHistDY->Integral();
  float gJetsSum = dYHist->Integral();

  std::cout << dataSum/gJetsSum << std::endl;

  prepareHist(dYHist);
  prepareHist(dataHistDY);

  dataHistDY->Divide(dataHistDY,dYHist,1.0,1.0,"B");                                                                                       
  TH1D* newh = (TH1D*)dataHistDY->Clone();                                                                                                 
 
  std::cout << "Rnorm = " << newh->GetBinContent(bin) << " +- " << newh->GetBinError(bin) << std::endl;

  f1->Close();
  f2->Close();
}
