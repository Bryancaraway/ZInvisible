void quick_ttH_macro(){
  TTree *tree = (TTree*) gFile->Get("Events") ;
  TString f_name = gFile->GetName();
  TString tth_f  = f_name(115,15);
  tree->Draw("GenPart_pt>>"+tth_f+"(20,0,1000)","GenPart_pdgId == 25");
  //
  TFile *f = new TFile("h_tth.root","update");
  TGraph *gr_full   = (TGraph*) gROOT->FindObject(tth_f)->Clone();
  gr_full->SetTitle(tth_f);
  //
 
  f->Write();
  f->Close();
}
