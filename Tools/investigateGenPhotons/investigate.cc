// investigate.cc
// investigate gen photons
// Caleb J. Smith
// October 1, 2018

#include <string>
#include <vector>
#include "stdio.h"
#include "TH1.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TCanvas.h"
#include "TTreeReader.h"

// simple class
class DrawOptions
{
    public:
        std::string plotName;
        std::string varexp;
        std::string selection;
        int color;
};

void plot(TTree* tree, DrawOptions p)
{
    std::string plotDir = "plots/";
    TCanvas* c1 = new TCanvas("c","c", 600.0, 600.0);
    tree->Draw(p.varexp.c_str(), p.selection.c_str(), "E");
    c1->SaveAs((plotDir + p.plotName + ".png").c_str());
    c1->SaveAs((plotDir + p.plotName + ".pdf").c_str());
    delete c1;
}

void multiplot(TTree* tree, std::vector<DrawOptions> vp, std::string plotName)
{
    // get htemp
    std::string plotDir = "plots/";
    TCanvas* c1 = new TCanvas("c","c", 600.0, 600.0);
    TH1F* htemp = nullptr;
    htemp = (TH1F*)gPad->GetPrimitive("htemp");
    for (const auto & p : vp)
    {
        // draw
        if (htemp)
        {
            htemp->GetXaxis()->SetRangeUser(-5.0, 5.0);
        }
        else
        {
            printf("ERROR: htemp is nullptr\n");
            exit(1);
        }
        tree->SetLineColor(p.color);
        tree->Draw(p.varexp.c_str(), p.selection.c_str(), "same E");
    }
    c1->SaveAs((plotDir + plotName + ".png").c_str());
    c1->SaveAs((plotDir + plotName + ".pdf").c_str());
    delete c1;
}

void investigate(const char* inputFileName)
{
    // open that file please 
    TFile* file = nullptr;
    file = TFile::Open(inputFileName);
    if (!file)
    {
        printf("ERROR: Did not open file %s\n", inputFileName);
        exit(1);
    }
    // get tree
    std::string treeName = "Events";
    TTree* tree = nullptr;
    tree = (TTree*)file->Get(treeName.c_str());
    if (!tree)
    {
        printf("ERROR: Did not open tree %s\n", treeName.c_str());
        exit(1);
    }
    // make plots
    std::vector<DrawOptions> plotOptions;
    std::string plotName = "";
    std::string varexp = "";
    std::string selection = "";
    
    // from Dr. Ken
    // plotName = "genStatus_pdgId=22_pt>10";
    // varexp = "recoGenParticles_prunedGenParticles__PAT.obj.status()";
    // selection = "recoGenParticles_prunedGenParticles__PAT.obj.pdgId() == 22 && recoGenParticles_prunedGenParticles__PAT.obj.pt() > 10.0";
    // DrawOptions p1 = {plotName, varexp, selection};
    // plotOptions.push_back(p1);
    
    plotName = "genEta_eta<5";
    varexp = "recoGenParticles_prunedGenParticles__PAT.obj.eta()";
    selection = "abs(recoGenParticles_prunedGenParticles__PAT.obj.eta()) < 5.0";
    DrawOptions p1 = {plotName, varexp, selection, kRed};
    plotOptions.push_back(p1);
    
    plotName = "genEta_eta<5_pt>10";
    varexp = "recoGenParticles_prunedGenParticles__PAT.obj.eta()";
    selection = "abs(recoGenParticles_prunedGenParticles__PAT.obj.eta()) < 5.0 && recoGenParticles_prunedGenParticles__PAT.obj.pt() > 10.0";
    DrawOptions p2 = {plotName, varexp, selection, kBlue};
    plotOptions.push_back(p2);
    
    plotName = "genEta_eta<5_pt>10_pdgId=22";
    varexp = "recoGenParticles_prunedGenParticles__PAT.obj.eta()";
    selection = "abs(recoGenParticles_prunedGenParticles__PAT.obj.eta()) < 5.0 && recoGenParticles_prunedGenParticles__PAT.obj.pt() > 10.0 && recoGenParticles_prunedGenParticles__PAT.obj.pdgId() == 22";
    DrawOptions p3 = {plotName, varexp, selection, kGreen-2};
    plotOptions.push_back(p3);
    
    // loop over plots
    for (const auto& p : plotOptions)
    {
        plot(tree, p);
    }
    
    // multiplot
    multiplot(tree, plotOptions, "all_the_gen");

    // close that file please
    file->Close();
    // delete file
    delete file;
}

int main()
{
    const char* inputFileName = "FC894077-2DCA-E611-8008-002590DE6E3C.root"; 
    investigate(inputFileName);
}




