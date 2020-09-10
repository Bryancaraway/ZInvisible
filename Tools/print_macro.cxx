void print_macro(){
  TTree *tree = (TTree*) gFile->Get("Events") ; 
  tree->Print();

}
