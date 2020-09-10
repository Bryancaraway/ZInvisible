void quick_ttZ_macro(){
  TTree *tree = (TTree*) gFile->Get("Events") ;
  TString f_name = gFile->GetName();
  TString ttz_f  = f_name(118,15);
  tree->Draw("GenPart_pt>>"+ttz_f+"(20,0,1000)","GenPart_pdgId == 23");
  //
  TFile *f = new TFile("h_ttz.root","update");
  TGraph *gr_full   = (TGraph*) gROOT->FindObject(ttz_f)->Clone();
  gr_full->SetTitle(ttz_f);
  //
 
  f->Write();
  f->Close();
}
