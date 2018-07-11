#include "../../SusyAnaTools/Tools/NTupleReader.h"
#include "RegisterFunctions.h"

#include <iostream>
#include <string>
#include <vector>

#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TLegend.h"

int main()
{
    TFile *f = TFile::Open("condor/minituple-Feb5.root");

    std::vector<std::string> treeNames = /*{"ZJetsToNuNu_HT-100To200_13TeV-madgraph", "ZJetsToNuNu_HT-200To400_13TeV-madgraph", "ZJetsToNuNu_HT-400To600_13TeV-madgraph",*/ {"ZJetsToNuNu_HT-600ToInf_13TeV-madgraph"};

    TCanvas c("c","c",800, 800);
    
    TLegend *leg = new TLegend(0.60, 0.60, 0.90, 0.90);

    TH2 *h = new TH2D("met_vs_ht", "hist", 50, 0, 1500, 50, 0, 1500);
    TH2 *h2 = new TH2D("met_vs_mt2", "hist", 50, 0, 1500, 50, 0, 1500);
    TH2 *h3 = new TH2D("nt_vs_nj", "hist", 10, 0, 10, 20, 0, 20);

    for(auto& treeName : treeNames)
    {
        TTree * t = (TTree*)f->Get(treeName.c_str());

        RegisterFunctions2Dplot pvtm;

        NTupleReader tr(t);
        pvtm.registerFunctions(tr);

        while(tr.getNextEvent())
        {
            if(tr.getEvtNum() % 10000 == 0) std::cout << "Event #: " << tr.getEvtNum() << std::endl;

            const float& HT = tr.getVar<float>("HTZinv");
            const float& cleanMetPt = tr.getVar<float>("cleanMetPt");
            const float& best_had_brJet_MT2Zinv = tr.getVar<float>("best_had_brJet_MT2Zinv");
            const float& cntNJetsPt30Eta24Zinv = tr.getVar<int>("cntNJetsPt30Eta24Zinv");
            const float& nTopCandSortedCntZinv = tr.getVar<int>("nTopCandSortedCntZinv");

            const bool& passNoiseEventFilterZinv = tr.getVar<bool>("passNoiseEventFilterZinv");
            const bool& passMuZinvSel = tr.getVar<bool>("passMuZinvSel");
            const bool& passnJetsZinv = tr.getVar<bool>("passnJetsZinv");
            const bool& passdPhisZinv = tr.getVar<bool>("passdPhisZinv");

            //std::cout << passNoiseEventFilterZinv << "\t" << passMuZinvSel << "\t" << passnJetsZinv << "\t" << passdPhisZinv << "\t" << (HT > 200) << std::endl;

            if(passNoiseEventFilterZinv && passnJetsZinv && passdPhisZinv && HT > 200)
            {
                h->Fill(HT, cleanMetPt);
                h2->Fill(best_had_brJet_MT2Zinv, cleanMetPt);
                h3->Fill(cntNJetsPt30Eta24Zinv, nTopCandSortedCntZinv);
            }
            
        }

   }

    TFile *fout = new TFile("test2dHist.root", "RECREATE");
    h->Write();
    h2->Write();
    h3->Write();
    fout->Close();

}
