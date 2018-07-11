#include "Systematic.h"

static const int NSEARCHBINS = 84;//59;//84

void Systematic::bookHist(std::vector<Plotter::HistSummary>& vh, std::vector<AnaSamples::FileSummary>& vfs)
{
    Plotter::DatasetSummary dsDY_nunu_syst(  "varied", vfs, "", name_);
    Plotter::DatasetSummary dsDY_nunu_nosyst("Nominal", vfs, "", "");
    Plotter::DataCollection dcDY_nunu( "single", "nSearchBin", {dsDY_nunu_nosyst, dsDY_nunu_syst});
    if(!var2_.size())
    {
	vh.emplace_back(PHS(name_+"_"+var_, {dcDY_nunu}, {2, 1}, "", NSEARCHBINS, 0, NSEARCHBINS, true, true, "Search Bin", "Events"));
    } else
    {
	vh.emplace_back(PHS(name_+"_"+var_+"_"+var2_, {dcDY_nunu}, {2, 1}, "", NSEARCHBINS, 0, NSEARCHBINS, true, true, "Search Bin", "Events"));
    }
}

void Systematic::modifyParameters(NTupleReader& tr)
{
    std::string type;
    std::string type2;
    tr.getType(var_, type);
    std::vector<std::pair<std::string, std::pair<std::string, float>>> typevarvals = {{type, {var_, -999.}}};

    if(var2_.size())
    {
	tr.getType(var2_,type2);
	typevarvals.emplace_back(std::pair<std::string, std::pair<std::string, float>>({type2, {var2_, -999.}}));
    }

    for(auto& typevarval : typevarvals)
    {
	if(typevarval.first.find("vector") != std::string::npos)
	{
	    throw "Systematic::modifyParameters(...): Variables cannot be vectors. Var: " + typevarval.second.first; 
	}
	else
	{
	    if     (typevarval.first.find("float")       != std::string::npos) typevarval.second.second = tr.getVar<float>(typevarval.second.first);
	    else if(typevarval.first.find("unsigned int") != std::string::npos) typevarval.second.second = static_cast<float>(tr.getVar<unsigned int>(typevarval.second.first));
	    else if(typevarval.first.find("int")          != std::string::npos) typevarval.second.second = static_cast<float>(tr.getVar<int>(typevarval.second.first));
	    else if(typevarval.first.find("float")        != std::string::npos) typevarval.second.second = static_cast<float>(tr.getVar<float>(typevarval.second.first));
	    else if(typevarval.first.find("char")         != std::string::npos) typevarval.second.second = static_cast<float>(tr.getVar<char>(typevarval.second.first));
	    else if(typevarval.first.find("short")        != std::string::npos) typevarval.second.second = static_cast<float>(tr.getVar<short>(typevarval.second.first));
	    else if(typevarval.first.find("long")         != std::string::npos) typevarval.second.second = static_cast<float>(tr.getVar<long>(typevarval.second.first));
	    else
	    {
		throw "Systematic::modifyParameters(...): variable not defined. Var: " + typevarval.second.first;
	    }
	}
    }
    float weight = 1.0;
    if(func_)      weight = func_->Eval(typevarvals[0].second.second);
    else if(hist_) weight = hist_->GetBinContent(hist_->FindBin(typevarvals[0].second.second));
    else if(hist2_) weight = hist2_->GetBinContent(hist2_->GetXaxis()->FindBin(typevarvals[0].second.second),hist2_->GetYaxis()->FindBin(typevarvals[1].second.second));
    tr.registerDerivedVar(name_, weight);
}

SystWeights::SystWeights() : tr3(123384)
{
    TH1::AddDirectory(false);
    
    njWTTbar_0b  = nullptr;
    njWDYZ_0b    = nullptr;
    njWTTbar_g1b = nullptr;
    njWDYZ_g1b   = nullptr;
    
    TFile *f = new TFile("dataMCweights.root");
    if(f)
    {
        //njWTTbar_0b  = static_cast<TH1*>(f->Get("DataMC_nj_elmuZinv_0b_loose0")->Clone());//_ht200_dphi")->Clone());
        njWDYZ_0b    = static_cast<TH1*>(f->Get("DataMC_nj_muZinv_0b_loose0_mt2_MET")->Clone());//ht200_dphi")->Clone());
        //njWTTbar_g1b = static_cast<TH1*>(f->Get("DataMC_nj_elmuZinv_g1b_loose0")->Clone());//ht200_dphi")->Clone());
        njWDYZ_g1b   = static_cast<TH1*>(f->Get("DataMC_nj_muZinv_g1b_loose0_mt2_MET")->Clone());//ht200_dphi")->Clone());
        f->Close();
        delete f;
    }
    else
    {
        std::cout << "Failed to open: dataMCweights.root" << std::endl;
    }
}

SystWeights::~SystWeights()
{
    //if(tr3) delete tr3;
}

void SystWeights::getWeights(NTupleReader& tr)
{
    const int& cntNJetsPt30Eta24Zinv = tr.getVar<int>("cntNJetsPt30Eta24Zinv");
    const int& nSearchBin = tr.getVar<int>("nSearchBin");

    float mean_0b_DY  = njWDYZ_0b   ->GetBinContent(njWDYZ_0b->FindBin(cntNJetsPt30Eta24Zinv));
    float mean_g1b_DY = njWDYZ_g1b  ->GetBinContent(njWDYZ_g1b->FindBin(cntNJetsPt30Eta24Zinv));
    float mean_0b_tt  = njWTTbar_0b ->GetBinContent(njWTTbar_0b->FindBin(cntNJetsPt30Eta24Zinv));
    float mean_g1b_tt = njWTTbar_g1b->GetBinContent(njWTTbar_g1b->FindBin(cntNJetsPt30Eta24Zinv));

    float rms_0b_DY  = njWDYZ_0b   ->GetBinError(njWDYZ_0b->FindBin(cntNJetsPt30Eta24Zinv));
    float rms_g1b_DY = njWDYZ_g1b  ->GetBinError(njWDYZ_g1b->FindBin(cntNJetsPt30Eta24Zinv));
    float rms_0b_tt  = njWTTbar_0b ->GetBinError(njWTTbar_0b->FindBin(cntNJetsPt30Eta24Zinv));
    float rms_g1b_tt = njWTTbar_g1b->GetBinError(njWTTbar_g1b->FindBin(cntNJetsPt30Eta24Zinv));
    
    auto* weightedSB = new std::vector<std::pair<float, float> >();
    
    for(int i = 0; i < 100; ++i)
    {
        float wgt_0b_DY  = tr3.Gaus(1.0, rms_0b_DY/mean_0b_DY);
        float wgt_g1b_DY = tr3.Gaus(1.0, rms_g1b_DY/mean_g1b_DY);
        float wgt_0b_tt  = tr3.Gaus(1.0, rms_0b_tt/mean_0b_tt);
        float wgt_g1b_tt = tr3.Gaus(1.0, rms_g1b_tt/mean_g1b_tt);
        
        weightedSB->emplace_back(std::make_pair(static_cast<float>(nSearchBin), wgt_0b_DY*wgt_g1b_DY*wgt_0b_tt*wgt_g1b_tt));
        //unweightedSB->emplace_back(nSearchBin);
    }

    tr.registerDerivedVec("njSystWeightedSB", weightedSB);
}
