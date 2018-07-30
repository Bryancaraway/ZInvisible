#include "../../SusyAnaTools/Tools/NTupleReader.h"
#include "../../SusyAnaTools/Tools/samples.h"
#include "../../SusyAnaTools/Tools/SATException.h"

#include "../../TopTaggerTools/Tools/include/HistoContainer.h"

#include "derivedTupleVariables.h"
#include "baselineDef.h"
#include "BTagCorrector.h"
#include "TTbarCorrector.h"
#include "ISRCorrector.h"
#include "PileupWeights.h"
#include "customize.h"

#include "TopTaggerResults.h"
#include "Constituent.h"

#include <iostream>
#include <string>
#include <vector>
#include <getopt.h>

#include "math.h"

#include "Math/VectorUtil.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TRandom3.h"
#include "TFile.h"

void stripRoot(std::string &path)
{
    int dot = path.rfind(".root");
    if (dot != std::string::npos)
    {
        path.resize(dot);
    }
}

float SF_13TeV(float top_pt){

    return exp(0.0615-0.0005*top_pt);

}

bool filterEvents(NTupleReader& tr)
{
    const std::vector<TLorentzVector>& jetsLVec = tr.getVec<TLorentzVector>("jetsLVec");
    const float& met = tr.getVar<float>("met");

    return jetsLVec.size() >= 4 && jetsLVec[3].Pt() > 30;// && met > 250;
}

int main(int argc, char* argv[])
{

    std::string jetVecLabel           = "jetsLVec";

    int opt;
    int option_index = 0;
    int bct = 0; //TEST
    static struct option long_options[] = {
        {"condor",              no_argument, 0, 'c'},
        {"TTbar weight",        no_argument, 0, 't'},
        {"no event weighting",  no_argument, 0, 'd'},
        {"run stealth version", no_argument, 0, 's'},
        {"dataSets",      required_argument, 0, 'D'},
        {"numFiles",      required_argument, 0, 'N'},
        {"startFile",     required_argument, 0, 'M'},
        {"numEvts",       required_argument, 0, 'E'},
        {"output",        required_argument, 0, 'O'}
    };

    bool runOnCondor = false, enableTTbar = false, doWgt = true, runStealth = false;
    int nFiles = -1, startFile = 0, nEvts = -1;
    std::string dataSets = "Signal_T2tt_mStop850_mLSP100", filename = "example.root";

    while((opt = getopt_long(argc, argv, "ctdsD:N:M:E:O:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
        case 'c':
            runOnCondor = true;
            std::cout << "Configured for condor compatibility." << std::endl;
            break;

        case 't':
            enableTTbar = true;
            std::cout << "Enabled TTbar event weighting." << std::endl;
            break;

        case 'd':
            doWgt = false;
            std::cout << "No Event weighting." << std::endl;
            break;

        case 's':
            runStealth = true;
            std::cout << "Running stealth verison" << std::endl;
            break;

        case 'D':
            dataSets = optarg;
            std::cout << "Running over the " << dataSets << " data sets." << std::endl;
            break;

        case 'N':
            nFiles = int(atoi(optarg));
            std::cout << "Running over " << nFiles << " files." << std::endl;
            break;

        case 'M':
            startFile = int(atoi(optarg));
            std::cout << "Starting on file #" << startFile << std::endl;
            break;

        case 'E':
            nEvts = int(atoi(optarg));
            std::cout << "Events: " << nEvts << std::endl;
            break;

        case 'O':
            filename = optarg;
            std::cout << "Filename: " << filename << std::endl;
        }
    }

    std::string sampleloc = AnaSamples::fileDir;
    //if running on condor override all optional settings
    if(runOnCondor)
    {
        char thistFile[128];
        stripRoot(filename);
        sprintf(thistFile, "%s_%s_%d.root", filename.c_str(), dataSets.c_str(), startFile);
        filename = thistFile;
        std::cout << "Filename modified for use with condor: " << filename << std::endl;
        sampleloc = "condor";
    }

    TH1::AddDirectory(false);

    bool savefile = true;
    if(filename == "-"){
        savefile = false;
        std::cout << "Histogram file will not be saved." << std::endl;
    }

    std::cout << "Sample location: " << sampleloc << std::endl;

    AnaSamples::SampleSet        ss("sampleSets.txt", runOnCondor, AnaSamples::luminosity);
    AnaSamples::SampleCollection sc("sampleCollections.txt", ss);

    if(dataSets.find("Data") != std::string::npos){
       std::cout << "This looks like a data n-tuple. No weighting will be applied." << std::endl;
       doWgt = false;
    }

    if(dataSets.find("TT") != std::string::npos){
       std::cout << "This looks like a TTbar sample. Applying TTbar weighting" << std::endl;
       enableTTbar = true;
    }

    std::cout << "Dataset: " << dataSets << std::endl;

    int events = 0, pevents = 0;


    TRandom* trand = new TRandom3();
    //Create Histos
    TH1* hMET[2]={new TH1D("met"        , "All Events;  met;     events"        , 50, 0, 750),
                  new TH1D("p_met"      , "Passed Cuts; met;     events"        , 50, 0, 750)};
    TH1* jEta[4]={new TH1D("jeta_l"     , "All Events;  Jet Eta; events"        , 50,-3, 3  ),
                  new TH1D("jeta_h"     , "All Events;  Jet Eta; events"        , 50,-3, 3  ),
                  new TH1D("p_jeta_l"   , "Passed Cuts; Jet Eta; events"        , 50,-3, 3  ),
                  new TH1D("p_jeta_h"   , "Passed Cuts; Jet Eta; events"        , 50,-3, 3  )};
    TH1* jPhi[4]={new TH1D("jphi_l"     , "All Events;  Jet Phi; events"        , 50,-3, 3  ),
                  new TH1D("jphi_h"     , "All Events;  Jet Phi; events"        , 50,-3, 3  ),
                  new TH1D("p_jphi_l"   , "Passed Cuts; Jet Phi; events"        , 50,-3, 3  ),
                  new TH1D("p_jphi_h"   , "Passed Cuts; Jet Phi; events"        , 50,-3, 3  )};
    TH1* jPt[4] ={new TH1D("jpt_l"      , "All Events;  Jet Pt;  events"        , 50, 0, 750),
                  new TH1D("jpt_h"      , "All Events;  Jet Pt;  events"        , 50, 0, 750),
                  new TH1D("p_jpt_l"    , "Passed Cuts; Jet Pt;  events"        , 50, 0, 750),
                  new TH1D("p_jpt_h"    , "Passed Cuts; Jet Pt;  events"        , 50, 0, 750)};
    TH1* mEta[4]={new TH1D("meta_l"     , "All Events;  Muon Eta; events"       , 50,-3, 3  ),
                  new TH1D("meta_h"     , "All Events;  Muon Eta; events"       , 50,-3, 3  ),
                  new TH1D("p_meta_l"   , "Passed Cuts; Muon Eta; events"       , 50,-3, 3  ),
                  new TH1D("p_meta_h"   , "Passed Cuts; Muon Eta; events"       , 50,-3, 3  )};
    TH1* mPhi[4]={new TH1D("mphi_l"     , "All Events;  Muon Phi; events"       , 50,-3, 3  ),
                  new TH1D("mphi_h"     , "All Events;  Muon Phi; events"       , 50,-3, 3  ),
                  new TH1D("p_mphi_l"   , "Passed Cuts; Muon Phi; events"       , 50,-3, 3  ),
                  new TH1D("p_mphi_h"   , "Passed Cuts; Muon Phi; events"       , 50,-3, 3  )};
    TH1* mPt[4] ={new TH1D("mpt_l"      , "All Events;  Muon Pt;  events"       , 50, 0, 750),
                  new TH1D("mpt_h"      , "All Events;  Muon Pt;  events"       , 50, 0, 750),
                  new TH1D("p_mpt_l"    , "Passed Cuts; Muon Pt;  events"       , 50, 0, 750),
                  new TH1D("p_mpt_h"    , "Passed Cuts; Muon Pt;  events"       , 50, 0, 750)};
    TH1* eEta[4]={new TH1D("eeta_l"     , "All Events;  Electron Eta; events"   , 50,-3, 3  ),
                  new TH1D("eeta_h"     , "All Events;  Electron Eta; events"   , 50,-3, 3  ),
                  new TH1D("p_eeta_l"   , "Passed Cuts; Electron Eta; events"   , 50,-3, 3  ),
                  new TH1D("p_eeta_h"   , "Passed Cuts; Electron Eta; events"   , 50,-3, 3  )};
    TH1* ePhi[4]={new TH1D("ephi_l"     , "All Events;  Electron Phi; events"   , 50,-3, 3  ),
                  new TH1D("ephi_h"     , "All Events;  Electron Phi; events"   , 50,-3, 3  ),
                  new TH1D("p_ephi_l"   , "Passed Cuts; Electron Phi; events"   , 50,-3, 3  ),
                  new TH1D("p_ephi_h"   , "Passed Cuts; Electron Phi; events"   , 50,-3, 3  )};
    TH1* ePt[4] ={new TH1D("ept_l"      , "All Events;  Electron Pt;  events"   , 50, 0, 750),
                  new TH1D("ept_h"      , "All Events;  Electron Pt;  events"   , 50, 0, 750),
                  new TH1D("p_ept_l"    , "Passed Cuts; Electron Pt;  events"   , 50, 0, 750),
                  new TH1D("p_ept_h"    , "Passed Cuts; Electron Pt;  events"   , 50, 0, 750)};
    TH1* hHT[2] ={new TH1D("ht"         , "All Events;  HT;       events"       , 50, 0, 750),
                  new TH1D("p_ht"       , "Passed Cuts; HT;       events"       , 50, 0, 750)};
    TH1* vEta[4]={new TH1D("veta_l"     , "All Events;  vTop Eta; events"       , 50,-3, 3  ),
                  new TH1D("veta_h"     , "All Events;  vTop Eta; events"       , 50,-3, 3  ),
                  new TH1D("p_veta_l"   , "Passed Cuts; vTop Eta; events"       , 50,-3, 3  ),
                  new TH1D("p_veta_h"   , "Passed Cuts; vTop Eta; events"       , 50,-3, 3  )};
    TH1* vPhi[4]={new TH1D("vphi_l"     , "All Events;  vTop Phi; events"       , 50,-3, 3  ),
                  new TH1D("vphi_h"     , "All Events;  vTop Phi; events"       , 50,-3, 3  ),
                  new TH1D("p_vphi_l"   , "Passed Cuts; vTop Phi; events"       , 50,-3, 3  ),
                  new TH1D("p_vphi_h"   , "Passed Cuts; vTop Phi; events"       , 50,-3, 3  )};
    TH1* vPt[4] ={new TH1D("vpt_l"      , "All Events;  vTop Pt;  events"       , 50, 0, 750),
                  new TH1D("vpt_h"      , "All Events;  vTop Pt;  events"       , 50, 0, 750),
                  new TH1D("p_vpt_l"    , "Passed Cuts; vTop Pt;  events"       , 50, 0, 750),
                  new TH1D("p_vpt_h"    , "Passed Cuts; vTop Pt;  events"       , 50, 0, 750)};
    TH1* hDP[6] ={new TH1D("dphi_1"     , "All Events;  deltaPhi; events"       , 50, 0, 3.2),
                  new TH1D("dphi_2"     , "All Events;  deltaPhi; events"       , 50, 0, 3.2),
                  new TH1D("dphi_3"     , "All Events;  deltaPhi; events"       , 50, 0, 3.2),
                  new TH1D("p_dphi_1"   , "Passed Cuts; deltaPhi; events"       , 50, 0, 3.2),
                  new TH1D("p_dphi_2"   , "Passed Cuts; deltaPhi; events"       , 50, 0, 3.2),
                  new TH1D("p_dphi_3"   , "Passed Cuts; deltaPhi; events"       , 50, 0, 3.2)};
    TH1* x_hMET = new TH1D("x_met"      , "All Events;  regional met;events"    , 50, 0, 750);
    TH1* x_jPt  = new TH1D("x_jpt_l"    , "All Events;  regional Jet Pt;events" , 50, 0, 750);
    TH1* x_hDP[3]={new TH1D("x_dphi_1"  , "All Events;  regional dPhi;events"   , 50, 0, 3.2),
                  new TH1D("x_dphi_2"   , "All Events;  regional dPhi;events"   , 50, 0, 3.2),
                  new TH1D("x_dphi_3"   , "All Events;  regional dPhi;events"   , 50, 0, 3.2)};
    //TH1* hNE[2] ={new TH1D("nElec"      , "All Events;  N Electrons;events"     , 11, 0, 10),
    //              new TH1D("p_nElec"    , "Passed Cuts; N Electrons;events"     , 11, 0, 10)};
    try
    {
        //for(auto& fs : sc[dataSets])
        auto& fs = ss[dataSets];
        {
            TChain *t = new TChain(fs.treePath.c_str());
            fs.addFilesToChain(t, startFile, nFiles);

            std::cout << "File: " << fs.filePath << "/" << fs.fileName << std::endl;
            std::cout << "Tree: " << fs.treePath << std::endl;
            //std::cout << "sigma*lumi: " << fs.getWeight() << std::endl;

            BaselineVessel myBLV(*static_cast<NTupleReader*>(nullptr), "", "");
            NTupleReader tr(t);            
            tr.registerFunction(myBLV);
            float fileWgt = fs.getWeight();

            const int printInterval = 1000;
            int printNumber = 0;

            while(tr.getNextEvent())
            {
                events++;

//                tr.printTupleMembers();
//                return 0;

                if(nEvts > 0 && tr.getEvtNum() > nEvts) break;
                std::cout << "evt - " << tr.getEvtNum() << "\n" << std::endl;
                if(tr.getEvtNum() > 100) break; //DEBUG
                if(tr.getEvtNum() / printInterval > printNumber)
                {
                    printNumber = tr.getEvtNum() / printInterval;
                    std::cout << "Event #: " << printNumber * printInterval << std::endl;
                }

                const float& met    = tr.getVar<float>("met");
                const float& metphi = tr.getVar<float>("metphi");
                const float& ht     = tr.getVar<float>("HT");
                //const int& nElec    = tr.getVar<int>("nElectrons_CUT");  
                const std::vector<TLorentzVector>& jetsLvec     = tr.getVec<TLorentzVector>("jetsLVec");
                const std::vector<TLorentzVector>& muonsLVec    = tr.getVec<TLorentzVector>("muonsLVec");
                const std::vector<TLorentzVector, std::allocator<TLorentzVector> > elesLVec = tr.getVec<TLorentzVector>("elesLVec");
                const std::vector<TLorentzVector>& vTops        = tr.getVec<TLorentzVector>("vTops");
                const std::vector<float>& muonsMiniIso          = tr.getVec<float>("muonsMiniIso");
                const std::vector<int> & muonsFlagIDVec         = tr.getVec<int>("muonsFlagMedium");
                const std::vector<float>& muonsRelIso           = tr.getVec<float>("muonsRelIso");
                const std::vector<int>& elesFlagIDVec           = tr.getVec<int>("elesFlagVeto");
                const std::vector<float>& elesMiniIso           = tr.getVec<float>("elesMiniIso");
                const std::vector<unsigned int>& elesisEB       = tr.getVec<unsigned int>("elesisEB");
                const std::vector<float>& dPhiVec               = tr.getVec<float>("dPhiVec");

//passLeptVeto passnJets passdPhis passBJets passMET passMT2 passHT passTagger passNoiseEventFilter passQCDHighMETFilter passFastsimEventFilter 
//passMuonVeto passEleVeto passIsoTrkVeto passIsoLepTrkVeto passIsoPionTrkVeto
                int end = 9;
                bool pb = tr.getVar<bool>("passBaseline");
                //bool pb = tr.getVar<bool>("passdPhisZinv");
                //bool pbt = (/*tr.getVar<bool>("passLeptVeto")*/ tr.getVar<bool>("passMuonVeto") /*&& tr.getVar<bool>("passEleVeto")*/ && tr.getVar<bool>("passIsoTrkVeto") && tr.getVar<bool>("passIsoLepTrkVeto") && tr.getVar<bool>("passIsoPionTrkVeto") && tr.getVar<bool>("passnJets") && tr.getVar<bool>("passdPhis") && tr.getVar<bool>("passBJets") && tr.getVar<bool>("passMET") && tr.getVar<bool>("passHT") && tr.getVar<bool>("passNoiseEventFilter") && tr.getVar<bool>("passQCDHighMETFilter"));// && tr.getVar<bool>("passFastsimEventFilter")); 
                //if (pb != pbt) {bct++; std::cout<<"!!!Boolean Mismatch!!! "<<bct<<std::endl;}
                TLorentzVector MET;
                MET.SetPtEtaPhiM(met, 0.0, metphi, 0.0);
                // fill histos
                std::cout << "MET: " << met << std::endl;
                hMET[0] ->Fill(met,fileWgt);
                hHT[0]  ->Fill(ht ,fileWgt);
                //hNE[0]  ->Fill(nElec, fileWgt);
                if (pb) {
                    hMET[1] ->Fill(met,fileWgt);
                    hHT[1]  ->Fill(ht ,fileWgt);
                    //hNE[1]  ->Fill(nElec, fileWgt);
                }
                for (int i = 0; i < 3; i++){
                    if (i < dPhiVec.size()){
                        hDP[i]->Fill(dPhiVec[i],fileWgt);
                        if (pb) hDP[i+3]->Fill(dPhiVec[i],fileWgt);
                    }
                }
                for (TLorentzVector jet:jetsLvec) {
                    if ((jet.Eta() > -2.5) && (jet.Eta() < -1.5) && (jet.Pt() > 40)) end = 0;
                    else if ((jet.Eta() > 1.5) && (jet.Eta() < 2.5) && (jet.Pt() > 40)) end = 1;
                    else continue;
                    jEta[end]   ->Fill(jet.Eta(),   fileWgt);
                    jPhi[end]   ->Fill(jet.Phi(),   fileWgt);
                    jPt [end]   ->Fill(jet.Pt() ,   fileWgt);
                    if (pb){
                        jEta[end+2] ->Fill(jet.Eta(),   fileWgt);
                        jPhi[end+2] ->Fill(jet.Phi(),   fileWgt);
                        jPt [end+2] ->Fill(jet.Pt() ,   fileWgt);
                    }
                    break;
                }
                if (jetsLvec.size() && (jetsLvec[0].Eta() > -2.5) && (jetsLvec[0].Eta() < -1.5) && (jetsLvec[0].Phi() < -1) && (jetsLvec[0].Phi() > -2)){
                    for (int i = 0; i < 3; i++) if (i < dPhiVec.size()) x_hDP[i]->Fill(dPhiVec[i],fileWgt);
                    x_jPt       ->Fill(jetsLvec[0].Pt(),    fileWgt);
                    x_hMET       ->Fill(met,                 fileWgt);
                } 
                for (TLorentzVector vTop:vTops) {
                    if ((vTop.Eta() > -2.5) && (vTop.Eta() < -1.5) && (vTop.Pt() > 40)) end = 0;
                    else if ((vTop.Eta() > 1.5) && (vTop.Eta() < 2.5) && (vTop.Pt() > 40)) end = 1;
                    else continue;
                    vEta[end]   ->Fill(vTop.Eta(),  fileWgt);
                    vPhi[end]   ->Fill(vTop.Phi(),  fileWgt);
                    vPt [end]   ->Fill(vTop.Pt() ,  fileWgt);
                    if (pb){
                        vEta[end+2] ->Fill(vTop.Eta(),  fileWgt);
                        vPhi[end+2] ->Fill(vTop.Phi(),  fileWgt);
                        vPt [end+2] ->Fill(vTop.Pt() ,  fileWgt);
                    }
                }
                // dPhiVec (0, 1, 2nd elements if existant)
                for (int i = 0; i < muonsLVec.size(); ++i) {
                    if (AnaFunctions::passMuon( muonsLVec[i], muonsMiniIso[i], 0.0, muonsFlagIDVec[i], AnaConsts::muonsMiniIsoArr) && AnaFunctions::passMuon( muonsLVec[i], muonsRelIso[i], 0.0, muonsFlagIDVec[i], AnaConsts::muonsMiniIsoArr)) {
                        if ((muonsLVec[i].Eta() > -2.5)&&(muonsLVec[i].Eta() < -1.5)) end = 0;
                        else if ((muonsLVec[i].Eta() < 2.5)&&(muonsLVec[i].Eta() > 1.5)) end = 1;
                        else continue;
                        mEta[end]   ->Fill(muonsLVec[i].Eta(),  fileWgt);
                        mPhi[end]   ->Fill(muonsLVec[i].Phi(),  fileWgt);
                        mPt [end]   ->Fill(muonsLVec[i].Pt() ,  fileWgt);
                        if (pb){
                            mEta[end+2] ->Fill(muonsLVec[i].Eta(),  fileWgt);
                            mPhi[end+2] ->Fill(muonsLVec[i].Phi(),  fileWgt);
                            mPt [end+2] ->Fill(muonsLVec[i].Pt() ,  fileWgt);
                        }
                    }
                }
                for(int i = 0; i < elesLVec.size(); ++i){
                    if(AnaFunctions::passElectron(elesLVec[i], elesMiniIso[i], -1, elesisEB[i], elesFlagIDVec[i], AnaConsts::elesMiniIsoArr)){
                        if ((elesLVec[i].Eta() > -2.5)&&(elesLVec[i].Eta() < -1.5)) end = 0;
                        else if ((elesLVec[i].Eta() < 2.5)&&(elesLVec[i].Eta() > 1.5)) end = 1;
                        else continue;
                        eEta[end]   ->Fill(elesLVec[i].Eta(),   fileWgt);
                        ePhi[end]   ->Fill(elesLVec[i].Phi(),   fileWgt);
                        ePt [end]   ->Fill(elesLVec[i].Pt() ,   fileWgt);
                        if (pb){
                            eEta[end+2] ->Fill(elesLVec[i].Eta(),   fileWgt);
                            ePhi[end+2] ->Fill(elesLVec[i].Phi(),   fileWgt);
                            ePt [end+2] ->Fill(elesLVec[i].Pt() ,   fileWgt);
                        }
                    }
                }
            }
        }
    }
    catch(const std::string e)
    {
        std::cout << e << std::endl;
        return 0;
    }
    catch(const TTException e)
    {
        std::cout << e << std::endl;
        return 0;
    }
    catch(const SATException e)
    {
        std::cout << e << std::endl;
        return 0;
    }

    std::cout << "Processed " << events << " events. " << pevents << " passed selection." << std::endl;

    if(savefile)
    {
        std::cout << "Saving root file..." << std::endl;

        TFile *f = new TFile(filename.c_str(),"RECREATE");
        if(f->IsZombie())
        {
            std::cout << "Cannot create " << filename << std::endl;
            throw "File is zombie";
        }
        f->cd();
        //histo->Write()
        for (int i = 0; i < 2; i++){
            hHT[i]  ->Write();
            hMET[i] ->Write();
            //hNE[i]  ->Write();
        }
        for (int i = 0; i < 4; i++){
            jEta[i] ->Write(); 
            jPhi[i] ->Write();
            jPt[i]  ->Write();
            mEta[i] ->Write();
            mPhi[i] ->Write();
            mPt[i]  ->Write();
            eEta[i] ->Write();
            ePhi[i] ->Write();
            ePt[i]  ->Write();
            vEta[i] ->Write();
            vPhi[i] ->Write();
            vPt[i]  ->Write();
        }
        for (int i = 0; i < 6; i++){
            hDP[i]  ->Write();
        }
        x_hMET  ->Write();
        x_jPt   ->Write();
        for (int i = 0; i < 3; i++){
            x_hDP[i]->Write();
        }
        f->Write();
        f->Close();
    }
}
