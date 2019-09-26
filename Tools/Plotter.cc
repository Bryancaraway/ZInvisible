#include "Plotter.h"
#include "tdrstyle.h"
#include "MiniTupleMaker.h"
#include "RegisterFunctions.h"
#include "TypeDefinitions.h"
#include "TROOT.h"
#include "TCanvas.h"
#include "TTree.h"
#include "TF1.h"
#include "THStack.h"
#include "TLatex.h"
#include <sstream>

const int colors[] = {
    kRed,
    kBlue,
    kGreen-2,
    kBlack,
    kOrange,
    kCyan,
    kMagenta,
    kYellow,
    kGreen
};
const int NCOLORS = sizeof(colors)/sizeof(int);

const int stackColors[] = {
    kAzure   + 2,
    kRed     - 7,
    kSpring  - 5,
    kOrange  - 3,
    kMagenta - 2,
    kAzure   - 4,
    kRed     - 3,
    kTeal    - 7,
    kAzure   + 0,
    kGreen   + 2,
    kMagenta - 1,
    kYellow  + 4,
    kRed     + 1,
    kAzure   - 4,
    kCyan    + 2,
    kOrange  + 7
};
const int NSTACKCOLORS = sizeof(stackColors) / sizeof(int);

const int hatches[] = {
    1001,
    3345,
    3354,
    3144,
    3001
};
const int NHATCHES = sizeof(hatches) / sizeof(int);

const int lineStyles[] = {
    kSolid,
};
const int LINESTYLES = sizeof(lineStyles) / sizeof(int);

Plotter::Plotter(std::vector<HistSummary>& h, std::set<AnaSamples::FileSummary>& t, const bool readFromTuple, std::string ofname, const int nFile, const int startFile, const int nEvts) : nFile_(nFile), startFile_(startFile), maxEvts_(nEvts)
{
    TH1::AddDirectory(false);

    hists_ = h;
    trees_ = t;
    readFromTuple_ = readFromTuple;
    lumi_ = AnaSamples::luminosity_2016;
    foutTuple_ = nullptr;
    if(ofname.size() == 0) ofname = "histoutput.root";
    if(readFromTuple)
    {
        fout_ = new TFile(ofname.c_str(), "RECREATE");
    }
    else
    {
        fout_ = new TFile(ofname.c_str());
    }
    foutTupleName_ = "minituple_" + ofname;
    doHists_ = doTuple_ = true;
    registerfunc_ = nullptr;
    printInterval_ = 1000;
}

Plotter::~Plotter()
{
    if(fout_)
    {
        fout_->Close();
        delete fout_;
    }
    if(foutTuple_)
    {
        foutTuple_->Close();
        delete foutTuple_;
    }
    if(registerfunc_) delete registerfunc_;
}

void Plotter::read()
{
    if(readFromTuple_)
    {
        createHistsFromTuple();
    }
    else
    {
        createHistsFromFile();
    }
}

Plotter::Cut::Cut(std::string s, char t, bool inv, double v, double v2)
{
    rawName = s;
    type = t;
    inverted = inv;
    val = v;
    val2 = v2;
    parseName();
}

void Plotter::Cut::parseName()
{
    Plotter::parseSingleVar(rawName, name);
}

Plotter::Cuttable::Cuttable(const std::string& c)
{
    setCuts(c);
}

void Plotter::Cuttable::extractCuts(std::set<std::string>& ab) const
{
    for(auto& cut : cutVec_) ab.insert(cut.rawName);
}

Plotter::HistSummary::HistSummary(std::string l, std::vector<Plotter::DataCollection> ns, std::pair<int, int> ratio, std::string cuts, int nb, double ll, double ul, bool log, bool norm, std::string xal, std::string yal, bool isRatio) : Cuttable(cuts), name(l), nBins(nb), low(ll), high(ul), isLog(log), isNorm(norm), xAxisLabel(xal), yAxisLabel(yal), ratio(ratio), isRatio(isRatio), ymin_(-999.9), ymax_(-999.9), setYLimits(false)
{
    parseName(ns);
}

Plotter::HistSummary::HistSummary(std::string l, std::vector<Plotter::DataCollection> ns, std::pair<int, int> ratio, std::string cuts, int nb, double ll, double ul, double ymin, double ymax, bool log, bool norm, std::string xal, std::string yal, bool isRatio) : HistSummary(l, ns, ratio, cuts, nb, ll, ul, log, norm, xal, yal, isRatio)
{
    ymin_ = ymin;
    ymax_ = ymax;
    setYLimits = true;
}

Plotter::HistSummary::HistSummary(std::string l, std::vector<Plotter::DataCollection> ns, std::pair<int, int> ratio, std::string cuts, std::vector<double> be, bool log, bool norm, std::string xal, std::string yal, bool isRatio) : Cuttable(cuts), name(l), nBins(0), low(0.0), high(0.0), binEdges(be), isLog(log), isNorm(norm), xAxisLabel(xal), yAxisLabel(yal), ratio(ratio), isRatio(isRatio), ymin_(-999.9), ymax_(-999.9), setYLimits(false)
{
    parseName(ns);
}

void Plotter::HistSummary::parseName(std::vector<Plotter::DataCollection>& ns)
{
    for(auto& n : ns)
    {
        std::vector<std::shared_ptr<HistCutSummary>> tmphtp;
        for(auto& dataset : n.datasets)
        {
            size_t b1 = 0, b2 = 0;
            VarName var;
            Plotter::parseSingleVar(dataset.first, var);

            // WARNING: semicolons ";" should not be used in histogram names
            // TH1* hist = static_cast<TH1*>(file->Get("h_the;name")); 
            // does not work when semicolons ";" are in the histogram name... in those cases hist will be nullptr.
            // https://stackoverflow.com/questions/2896600/how-to-replace-all-occurrences-of-a-character-in-string
            std::string histname = name +  var.name + ((var.index >= 0)?(std::to_string(var.index)):("")) + var.var + dataset.first + dataset.second.front().label + n.type;
            std::replace(histname.begin(), histname.end(), ';', '_');

            tmphtp.push_back(std::shared_ptr<HistCutSummary>(new HistCutSummary(dataset.second.front().label, histname, var, nullptr, dataset.second)));
        }
        hists.push_back(HistVecAndType(tmphtp, n.type));
    }
}

void Plotter::parseSingleVar(const std::string& name, VarName& var)
{
    size_t a1 = name.find("[");
    size_t a2 = name.find("]");
    size_t b1 = name.find("(");
    size_t b2 = name.find(")");
    size_t c1 = name.find("{");
    size_t c2 = name.find("}");

    if(a1 != std::string::npos && a2 != std::string::npos)
    {
        var.index = static_cast<int>(atoi(name.substr(a1+1, a2 - a1 - 1).c_str()));
    }
    if(b1 != std::string::npos && b2 != std::string::npos)
    {
        var.var = name.substr(b1+1, b2 - b1 - 1);
    }
    if(c1 != std::string::npos && c2 != std::string::npos)
    {
        var.cuts.reset(new Cuttable(name.substr(c1+1, c2 - c1 - 1)));
    }
    if(a1 != std::string::npos || b1 != std::string::npos || c1 != std::string::npos)
    {
        var.name = name.substr(0, std::min(a1, std::min(b1, c1)));
    }
    else
    {
        var.name = name;
    }
}

Plotter::HistSummary::~HistSummary()
{
}

Plotter::HistCutSummary::~HistCutSummary()
{
    if(h) delete h;
}

Plotter::CutFlowSummary::CutFlowSummary(std::string n, Plotter::DataCollection ns, std::vector<std::string> cutLevels) : name(n), dc(ns)
{
    for(auto& str : cutLevels) cuts_.emplace_back(Cuttable(str));

    h = nullptr;
}

void Plotter::CutFlowSummary::generateHist()
{
    if(!h) h = new TH1D(name.c_str(), name.c_str(), cuts_.size(), -0.5, -0.5 + cuts_.size());
}

void Plotter::CutFlowSummary::fillHist(const NTupleReader& tr, const double& weight)
{
    for(int i = 0; i < cuts_.size(); ++i)
    {
        if(cuts_[i].passCuts(tr)) h->Fill(i, weight);
    }
}

Plotter::CutFlowSummary::~CutFlowSummary()
{
    if(h) delete h;
}

Plotter::DatasetSummary::DatasetSummary(std::string lab, std::vector<AnaSamples::FileSummary>& f, std::string cuts, std::string weights, double k) : Cuttable(cuts), label(lab), files(f), kfactor(k), weightStr(weights)
{
    parseWeights();
}

void Plotter::DatasetSummary::parseWeights()
{
    //parse which weights to use here.
    for(size_t pos = 0, npos = 0; npos < weightStr.size(); pos = npos + 1)
    {
        npos = weightStr.find(';', pos + 1);
        if(npos == size_t(-1)) npos = weightStr.size();
        std::string tmp = weightStr.substr(pos, npos - pos);

        weightVec_.push_back(tmp);
    }
}

double Plotter::DatasetSummary::getWeight(const NTupleReader& tr) const
{
    double retval = 1.0;
    for(auto& weightName : weightVec_)
    {
        const double& weight = static_cast<double>(tr.getVar<data_t>(weightName));
        if(weight == weight)
        {
            if(weight < 1e6)
            {
                if(weight > 2700.0) std::cout << weightName << "\t" << weight << std::endl;
                retval *= weight;
            }
        }
        else
        {
            std::cout << weightName << " is NAN!!!" << std::endl;
        }
    }
    return retval;
}

double Plotter::DatasetSummary::extractWeightNames(std::set<std::string>& ab) const
{
    for(auto& w : weightVec_) ab.insert(w);
}

Plotter::DataCollection::DataCollection(std::string type, std::vector<std::pair<std::string, DatasetSummary>> ds) : type(type)
{
    for(auto& dataset : ds)   
    {
        datasets.push_back(std::make_pair(dataset.first, std::vector<DatasetSummary>({dataset.second})));
    }
}

Plotter::DataCollection::DataCollection(std::string type, std::string var, std::vector<DatasetSummary> ds) : type(type)
{
    for(auto& dataset : ds)
    {
        datasets.push_back(std::make_pair(var, std::vector<DatasetSummary>({dataset})));
    }
}

Plotter::DataCollection::DataCollection(std::string type, std::vector<std::pair<std::string, std::vector<DatasetSummary>>> ds) : type(type), datasets(ds)
{
}

Plotter::DataCollection::DataCollection(std::string type, std::string var, std::vector<std::vector<DatasetSummary>> vvds) : type(type)
{
    for(auto& vds : vvds)
    { 
        datasets.push_back(std::make_pair(var, vds));
    }
}

void Plotter::createHistsFromTuple()
{
    for(const AnaSamples::FileSummary& file : trees_)
    {
        std::set<std::string> activeBranches;

        //get file list 
        file.readFileList();

        //make vector of hists to fill
        std::map<std::pair<std::string, const DatasetSummary*>, std::pair<const HistSummary*, std::vector<std::shared_ptr<HistCutSummary>>>> histsToFill;
        for(HistSummary& hs : hists_)
        {
            for(HistVecAndType& histvec : hs.hists)
            {
                for(std::shared_ptr<HistCutSummary>& hist : histvec.hcsVec)
                {
                    //annoying hack to fix vector pointer issue
                    if(hist->hs == nullptr) hist->hs = &hs;

                    // Make histogram if it is blank
                    if(hist->h == nullptr)
                    {
                        if(hs.binEdges.size()) hist->h = new TH1D(hist->name.c_str(), hist->variable.name.c_str(), hs.binEdges.size() - 1, hs.binEdges.data());
                        else                   hist->h = new TH1D(hist->name.c_str(), hist->variable.name.c_str(), hs.nBins, hs.low, hs.high);
                        hist->h->Sumw2();
                        hist->h->GetXaxis()->SetTitle(hs.xAxisLabel.c_str());
                        hist->h->GetYaxis()->SetTitle(hs.yAxisLabel.c_str());
                    }

                    for(const DatasetSummary& ds : hist->dss)
                    {
                        for(const AnaSamples::FileSummary& fileToComp : ds.files)
                        {
                            if(file == fileToComp)
                            {
                                hist->dssp = &ds;
                                auto theCuts = std::make_pair(hs.getCuts(), hist->dssp);
                                auto iter = histsToFill.find(theCuts);
                                if(iter == histsToFill.end())
                                {
                                    histsToFill[std::make_pair(hs.getCuts(), hist->dssp)] = std::make_pair(hist->hs, std::vector<std::shared_ptr<HistCutSummary>>({hist}));
                                }
                                else
                                {
                                    iter->second.second.push_back(hist);
                                }
                            }
                        }
                    }
                }
            }
        }

        //make vector of cutflows to fill
        std::vector<CutFlowSummary*> cutFlowsToFill;
        for(CutFlowSummary& cfs : cutFlows_)
        {
            for(auto& dss : cfs.dc.datasets)
            {
                for(const DatasetSummary& ds : dss.second)
                {
                    for(const AnaSamples::FileSummary& fileToComp : ds.files)
                    {
                        if(file == fileToComp)
                        {
                            cutFlowsToFill.push_back(&cfs);
                            cutFlowsToFill.back()->setDSS(&ds);
                            cutFlowsToFill.back()->generateHist();
                            break;
                        }
                    }
                }
            }
        }

        bool keepGoing = true;
        // Do not process files if there are no histograms asking for it
        if(!keepGoing && !histsToFill.size() && !cutFlowsToFill.size())
        {
            std::cout << "WARNING: skipping file summary " << file.tag << std::endl;
            continue;
        }

        std::cout << "Processing file(s): " << file.filePath << std::endl;

        //Set up mini tuple maker
        TTree *tOut = nullptr;
        MiniTupleMaker* mtm = nullptr;

        if(registerfunc_ == nullptr) registerfunc_ = new RegisterFunctions();
        registerfunc_->remakeBTagCorrector(file.tag);
        registerfunc_->remakeISRreweight(file.tag);

        if(doTuple_)
        {
            size_t start = file.filePath.rfind('/');
            size_t stop  = file.filePath.rfind('.');
            if(int(stop) - (int(start) + 1) > 0)
            {
                std::string treeName = file.filePath.substr(start + 1, stop - start - 1);
                if(!foutTuple_) foutTuple_ = new TFile(foutTupleName_.c_str(), "RECREATE");
                foutTuple_->cd();
                tOut = new TTree(treeName.c_str(), treeName.c_str());
                mtm = new MiniTupleMaker(tOut);
                if(file.isData_) mtm->setTupleVars(registerfunc_->getMiniTupleSetData());
                else             mtm->setTupleVars(registerfunc_->getMiniTupleSet());
                fout_->cd();
            }
        }

        int fileCount = 0, startCount = 0;
        int NEvtsTotal = 0;
        for(const std::string& fname : file.filelist_)
        {
            if(startCount++ < startFile_) continue;
            if(nFile_ > 0 && fileCount++ >= nFile_) break;

            if(nFile_ > 0) NEvtsTotal = 0;
            else if(maxEvts_ >= 0 && NEvtsTotal > maxEvts_) break;

            TFile *f = TFile::Open(fname.c_str());

            if(!f)
            {
                std::cout << "File \"" << fname << "\" not found!!!!!!" << std::endl;
                continue;
            }
            TTree *t = (TTree*)f->Get(file.treePath.c_str());

            if(!t)
            {
                std::cout << "Tree \"" << file.treePath << "\" not found in file \"" << fname << "\"!!!!!!" << std::endl;
                continue;
            }
            std::cout << "\t" << fname << std::endl;

            registerfunc_->activateBranches(activeBranches);

            try
            {
                NTupleReader tr(t, activeBranches);
                tr.setReThrow(false);

                registerfunc_->registerFunctions(tr);

                while(tr.getNextEvent())
                {
                    //Things to run only on first event
                    if(tr.isFirstEvent())
                    {
                        //Initialize the mini tuple branches, needs to be done after first call of tr.getNextEvent()
                        if(tOut && mtm)
                        {
                            try
                            {
                                mtm->initBranches(tr);
                            }
                            catch(const std::string e)
                            {
                                std::cout << "Exception caught in Plotter::createHistsFromTuple(), text follows" << std::endl << e << std::endl;
                            }
                        }
                    }

                    //If maxEvents_ is set, stop after so many events
                    if(maxEvts_ > 0 && NEvtsTotal > maxEvts_) break;
                    if(tr.getEvtNum() % printInterval_ == 0) std::cout << "Event #: " << tr.getEvtNum() << std::endl;

                    //fill histograms
                    if(doHists_)
                    {
                        double fileWgt = file.getWeight();

                        for(auto& histsToFillVec : histsToFill)
                        {
                            // get the dataset summary
                            const DatasetSummary& dss = *histsToFillVec.first.second;
                            
                            // tree level dynamical cuts are applied here
                            if(!dss.passCuts(tr)) continue;

                            const HistSummary& hs = *histsToFillVec.second.first;

                            // parse hist level cuts here
                            if(!hs.passCuts(tr)) continue;

                            // get the weight associated with the dataset
                            double weight = fileWgt * dss.getWeight(tr) * dss.kfactor;

                            for(auto& hist : histsToFillVec.second.second)
                            {
                                //fill histograms here
                                fillHist(hist->h, hist->variable, tr, weight);
                            }
                        }
                    }

                    //fill cut flows
                    for(auto& cutFlow : cutFlowsToFill)
                    {
                        //get event weight here
                        double weight = file.getWeight() * cutFlow->dssp->getWeight(tr) * cutFlow->dssp->kfactor;

                        cutFlow->fillHist(tr, weight);
                    }

                    //fill mini tuple
                    if(tOut && mtm)
                    {
                        if(tr.getVar<bool>("passnJetsZinv"))
                        {
                            if(file.filePath.find("ZJetsToNuNu_") != std::string::npos || tr.getVar<bool>("passMuZinvSel"))
                            {
                                foutTuple_->cd();
                                mtm->fill();
                                fout_->cd();
                            }
                        }
                    }

                    ++NEvtsTotal;
                }
            }
            catch(const std::string e)
            {
                std::cout << e << std::endl;
            }
            t->Reset();
            delete t;
            f->Close();
            delete f;
        }

        if(foutTuple_ && tOut && mtm)
        {
            foutTuple_->cd();
            tOut->Write();
        }
        if(mtm) delete mtm;
    }
}

void Plotter::createHistsFromFile()
{
    for(auto& hs : hists_)
    {
        for(auto& histvec : hs.hists)
        {
            for(auto& hist : histvec.hcsVec)
            {
                std::string& dirname = hist->variable.name;
                std::string histName = hist->name;

                if(fout_) hist->h = static_cast<TH1*>(fout_->Get( (dirname+"/"+histName).c_str() ) );
                else std::cout << "Input file \"" << fout_ << "\" not found!!!!" << std::endl;
                if(!hist->h) std::cout << "Histogram not found: \"" << hist->name << "\"!!!!!! dirname/histname: " << dirname+"/"+hist->name << std::endl;
            }
        }
    }
}

void Plotter::Cuttable::setCuts(const std::string& c)
{
    cuts_ = c;
    cutVec_.clear();
    parseCutString();
}

void Plotter::Cuttable::parseCutString()
{
    for(size_t pos = 0, npos = 0; npos < cuts_.size(); pos = npos + 1)
    {
        npos = cuts_.find(';', pos + 1);
        if(npos == size_t(-1)) npos = cuts_.size();
        std::string tmp = cuts_.substr(pos, npos - pos);
        size_t sepPos = 0, sepLen = 1;
        char cutType = ' ';
        bool inverted = false;
        if((sepPos = tmp.find("!")) != size_t(-1))
        {
            inverted = true;
            tmp = tmp.substr(sepPos + 1, size_t(-1));
        }
        sepPos = 0;
        // important: check for ">=" and "<=" first 
        if((sepPos = tmp.find(">=")) != std::string::npos)
        {
            cutType = 'G';
            sepLen = 2;
        }
        else if((sepPos = tmp.find("<=")) != std::string::npos)
        {
            cutType = 'L';
            sepLen = 2;
        }
        // important: check for ">" and "<" next
        else if((sepPos = tmp.find(">"))  != std::string::npos) cutType = '>';
        else if((sepPos = tmp.find("<"))  != std::string::npos) cutType = '<';
        else if((sepPos = tmp.find("="))  != std::string::npos) cutType = '=';
        else
        {
            cutType = 'B';
            cutVec_.push_back(Cut(tmp, cutType, inverted, 0.0));
            continue;
        }
        std::string t1 = tmp.substr(0, sepPos), t2 = tmp.substr(sepPos + sepLen, std::string::npos);
        t1.erase(remove(t1.begin(),t1.end(),' '),t1.end());
        t2.erase(remove(t2.begin(),t2.end(),' '),t2.end());
        char vname[32];
        sscanf(t1.c_str(), "%s", vname);
        tmp = vname;
        double cutvalue;
        sscanf(t2.c_str(), "%lf", &cutvalue);
        Cut tmpCut(tmp, cutType, inverted, cutvalue);
        cutVec_.push_back(tmpCut);
    }
}

bool Plotter::Cut::passCut(const NTupleReader& tr, const int index) const
{
    switch(type)
    {
    case '<': return translateVar(tr, index) < val;
    case '>': return translateVar(tr, index) > val;
    case '=': return translateVar(tr, index) == val;
    case 'G': return translateVar(tr, index) >= val;
    case 'L': return translateVar(tr, index) <= val;
    case '-': return translateVar(tr, index) > val && translateVar(tr, index) < val2;
    case 'B': return boolReturn(tr);
    default:
        printf("Unrecognized cut type, %c\n", type);
        return false;
    }
}

template<> const double& Plotter::getVarFromVec<TLorentzVector, double>(const VarName& name, const NTupleReader& tr, const int index)
{
    const auto& vec = tr.getVec<TLorentzVector>(name.name);

    if(&vec != nullptr)
    {
        int i;
        if(index >= 0) i = index;
        else           i = name.index;
        if(i < vec.size()) return tlvGetValue(name.var, vec.at(i));
        else return *static_cast<double*>(nullptr);
    }
    return *static_cast<double*>(nullptr);
}

double Plotter::Cut::translateVar(const NTupleReader& tr, const int index) const
{
    std::string type;
    tr.getType(name.name, type);

    if(type.find("vector") != std::string::npos)
    {
        if     (type.find("pair")           != std::string::npos) return Plotter::getVarFromVec<std::pair<double, double>>(name, tr, index).first;
        else if(type.find("double")         != std::string::npos) return static_cast<double>(Plotter::getVarFromVec<double>(name, tr, index));
        else if(type.find("float")          != std::string::npos) return static_cast<float>(Plotter::getVarFromVec<float>(name, tr, index));
        else if(type.find("unsigned int")   != std::string::npos) return static_cast<double>(Plotter::getVarFromVec<unsigned int>(name, tr, index));
        else if(type.find("int")            != std::string::npos) return static_cast<double>(Plotter::getVarFromVec<int>(name, tr, index));
        else if(type.find("TLorentzVector") != std::string::npos) return Plotter::getVarFromVec<TLorentzVector, double>(name, tr, index);
    }
    else
    {
        if     (type.find("double")       != std::string::npos) return tr.getVar<double>(name.name);
        else if(type.find("float")        != std::string::npos) return static_cast<float>(tr.getVar<float>(name.name));
        else if(type.find("unsigned int") != std::string::npos) return static_cast<double>(tr.getVar<unsigned int>(name.name));
        else if(type.find("int")          != std::string::npos) return static_cast<double>(tr.getVar<int>(name.name));
        else if(type.find("char")         != std::string::npos) return static_cast<double>(tr.getVar<char>(name.name));
        else if(type.find("short")        != std::string::npos) return static_cast<double>(tr.getVar<short>(name.name));
        else if(type.find("long")         != std::string::npos) return static_cast<double>(tr.getVar<long>(name.name));
    }
}

bool Plotter::Cut::boolReturn(const NTupleReader& tr) const
{
    return tr.getVar<bool>(name.name);
}

bool Plotter::Cuttable::passCuts(const NTupleReader& tr, const int index) const
{
    bool passCut = true;
    for(const Cut& cut : cutVec_)
    {
        passCut = passCut && ((cut.inverted)?(!cut.passCut(tr, index)):(cut.passCut(tr, index)));
    }
    return passCut;
}

void Plotter::saveHists()
{
    if(!readFromTuple_) return;

    fout_->cd();

    for(HistSummary& hist : hists_)
    {
        for(auto& hvec : hist.hists)
        {
            if(readFromTuple_ && fout_)
            {
                for(auto& h : hvec.hcsVec)
                {
                    std::string dirname = h->variable.name;
                    TDirectory* mydir = fout_->GetDirectory(dirname.c_str());
                    if(mydir == 0)
                    {
                        mydir = fout_->mkdir(dirname.c_str(),dirname.c_str());
                    }
                    mydir->cd();
                    if (h->h)
                    {
                        h->h->Write();
                    }
                    else
                    {
                        printf("Histogram for %s does not exist (nullptr)\n", dirname.c_str());
                    }
                }
            }
        }
    }

    TDirectory* d_cut = fout_->mkdir("CutFlows","CutFlows");
    d_cut->cd();
    for(CutFlowSummary& cutFlow : cutFlows_)
    {
        if(cutFlow.h) cutFlow.h->Write();
    }

    fout_->cd();
}

void Plotter::setPlotDir(const std::string plotDir)
{
    plotDir_ = plotDir + "/";
}

void Plotter::setLumi(const double lumi)
{
    lumi_ = lumi;
}

void Plotter::setDoHists(const bool doHists)
{
    doHists_ = doHists;
}

void Plotter::setDoTuple(const bool doTuple)
{
    doTuple_ = doTuple;
}

void Plotter::setRegisterFunction(RegisterFunctions* rf)
{
    registerfunc_ = rf;
}

void Plotter::setPrintInterval(const int printInterval)
{
    printInterval_ = printInterval;
}

void Plotter::setCutFlows(std::vector<CutFlowSummary> cfs)
{
    cutFlows_ = cfs;
}

double Plotter::getLumi()
{
    return lumi_;
}

void Plotter::plot()
{
    // setTDRStyle() is defined in SusyAnaTools/Tools/tdrstyle.h
    setTDRStyle();
    //gROOT->SetStyle("Plain");
    // make png files
    bool make_png = true;

    for(HistSummary& hist : hists_)
    {
        std::string plotname = plotDir_ + hist.name;
        bool skip = false;
        for(auto& hvec : hist.hists)  for(auto& h : hvec.hcsVec) if(!h->h) skip = true;
        if(skip) continue;
        if (! hist.fhist() )
        {
            // ERROR in red
            std::cout << "\033[1;31m  ERROR for histogram " << hist.name << "; hist.fhist() does not work, skipping plot. Check that histogram is created properly. \033[0m"<< std::endl;
            continue;
        }

        bool showRatio = true;
        if(hist.ratio.first == hist.ratio.second || hist.ratio.first < 1 || hist.ratio.second < 1)
        {
            showRatio = false;
        }

        TCanvas *c;
        double fontScale;
        if(showRatio)
        {
            c = new TCanvas("c1", "c1", 800, 900);
            c->Divide(1, 2);
            c->cd(1);
            gPad->SetPad("p1", "p1", 0, 2.5 / 9.0, 1, 1, kWhite, 0, 0);
            gPad->SetBottomMargin(0.01);
            fontScale = 1.0;
        }
        else
        {
            c = new TCanvas("c1", "c1", 800, 800);
            c->Divide(1, 1);
            c->cd(1);
            gPad->SetPad("p1", "p1", 0, 0, 1, 1, kWhite, 0, 0);
            gPad->SetBottomMargin(0.12);
            fontScale = 6.5 / 8;
        }
        gPad->SetLeftMargin(0.15);
        gPad->SetRightMargin(0.06);
        gPad->SetTopMargin(0.06 * (8.0 / 6.5) * fontScale);

        TH1 *dummy = new TH1F("dummy", "dummy", 1000, hist.fhist()->GetBinLowEdge(1), hist.fhist()->GetBinLowEdge(hist.fhist()->GetNbinsX()) + hist.fhist()->GetBinWidth(hist.fhist()->GetNbinsX()));
        dummy->SetStats(0);
        dummy->SetTitle(0);
        // remove (x 10^N), simply write zeros
        //dummy->GetYaxis()->SetNoExponent(kTRUE);
        dummy->GetYaxis()->SetTitle(hist.yAxisLabel.c_str());
        dummy->GetYaxis()->SetTitleOffset(1.1*1.05 / (fontScale));
        dummy->GetXaxis()->SetTitleOffset(1.05);
        if(showRatio) dummy->GetXaxis()->SetTitle("");
        else          dummy->GetXaxis()->SetTitle(hist.xAxisLabel.c_str());
        dummy->GetXaxis()->SetTitleSize(0.20 * 2 / 6.5 * fontScale);
        dummy->GetXaxis()->SetLabelSize(0.20 * 2 / 6.5 * fontScale);
        dummy->GetYaxis()->SetTitleSize(0.20 * 2 / 6.5 * fontScale);
        dummy->GetYaxis()->SetLabelSize(0.20 * 2 / 6.5 * fontScale);
        if(dummy->GetNdivisions() % 100 > 5) dummy->GetXaxis()->SetNdivisions(6, 5, 0);

        int NlegEntries = 0;
        for(auto& hvec : hist.hists)
        {
            if(hvec.type.compare("data") == 0)
            {
                if(hvec.hcsVec.size()) ++NlegEntries;
            }
            else if(hvec.type.compare("single") == 0)                            NlegEntries += hvec.hcsVec.size();
            else if(hvec.type.compare("ratio") == 0 && hvec.hcsVec.size() >= 2)  ++NlegEntries;
            else if(hvec.type.compare("stack") == 0)                             NlegEntries += hvec.hcsVec.size();
        }

        TLegend *leg = new TLegend(0.50, 0.88 - NlegEntries * 0.045, 0.89, 0.88);
        leg->SetFillStyle(0);
        leg->SetBorderSize(0);
        leg->SetLineWidth(1);
        leg->SetNColumns(1);
        leg->SetTextFont(42);

        double max = 0.0, lmax = 0.0, min = 1.0e300, minAvgWgt = 1.0e300;
        int iSingle = 0, iRatio = 0, iFill = 0;
        char legEntry[128];

        for(auto& hvec : hist.hists)
        {
            if(hvec.type.compare("data") == 0)
            {
                if(hvec.hcsVec.size())
                {
                    hvec.hcsVec.front()->h->SetLineColor(kBlack);
                    hvec.hcsVec.front()->h->SetLineWidth(3);
                    hvec.hcsVec.front()->h->SetMarkerColor(kBlack);
                    hvec.hcsVec.front()->h->SetMarkerStyle(20);
                    double integral = hvec.hcsVec.front()->h->Integral(1, hvec.hcsVec.front()->h->GetNbinsX() + 1);
                    if(     integral < 3.0)   sprintf(legEntry, "%s (%0.2lf)", hvec.hcsVec.front()->label.c_str(), integral);
                    else if(integral < 1.0e5) sprintf(legEntry, "%s (%0.0lf)", hvec.hcsVec.front()->label.c_str(), integral);
                    else                      sprintf(legEntry, "%s (%0.2e)",  hvec.hcsVec.front()->label.c_str(), integral);
                    leg->AddEntry(hvec.hcsVec.front()->h, legEntry, "PE");
                    //if(hist.isNorm) hvec.hcsVec.front()->h->Scale(hist.fhist()->Integral()/hvec.hcsVec.front()->h->Integral());
                    if(hist.isNorm) hvec.hcsVec.front()->h->Scale(1.0/hvec.hcsVec.front()->h->Integral());
                    smartMax(hvec.hcsVec.front()->h, leg, static_cast<TPad*>(gPad), min, max, lmax, true);

                    hvec.h = static_cast<TNamed*>(hvec.hcsVec.front()->h->Clone());
                }
            }
            else if(hvec.type.compare("single") == 0 || hvec.type.compare("fill") == 0)
            {
                int iStyle = 0;
                for(auto& h : hvec.hcsVec)
                {
                    std::string drawOptions = "L";
                    h->h->SetLineColor(colors[iSingle%NCOLORS]);
                    if(hvec.type.compare("fill") == 0)
                    {
                        h->h->SetFillColor(colors[iSingle%NCOLORS]);
                        h->h->SetFillStyle(hatches[iFill%NHATCHES]);
                        drawOptions = "F";
                        iFill++;
                    }
                    h->h->SetLineStyle(lineStyles[iStyle%LINESTYLES]);
                    iStyle++;
                    h->h->SetLineWidth(3);
                    iSingle++;
                    double integral = h->h->Integral(0, h->h->GetNbinsX() + 1);
                    if(     integral < 3.0)   sprintf(legEntry, "%s (%0.2lf)", h->label.c_str(), integral);
                    else if(integral < 1.0e5) sprintf(legEntry, "%s (%0.0lf)", h->label.c_str(), integral);
                    else                      sprintf(legEntry, "%s (%0.2e)",  h->label.c_str(), integral);
                    leg->AddEntry(h->h, legEntry, drawOptions.c_str());
                    //if(hist.isNorm) h->h->Scale(hist.fhist()->Integral()/h->h->Integral());
                    if(hist.isNorm) if(h->h->Integral() > 0.0) h->h->Scale(1.0/h->h->Integral());
                    smartMax(h->h, leg, static_cast<TPad*>(gPad), min, max, lmax);
                    minAvgWgt = std::min(minAvgWgt, h->h->GetSumOfWeights()/h->h->GetEntries());
                }
            }
            else if(hvec.type.compare("ratio") == 0 && hvec.hcsVec.size() >= 2)
            {
                auto hIter = hvec.hcsVec.begin();
                TH1* hratio = static_cast<TH1*>((*hIter)->h->Clone());
                hratio->SetLineColor(colors[iRatio%NCOLORS]);
                hratio->SetLineWidth(3);
                ++iRatio;
                hvec.h = static_cast<TNamed*>(hratio);
                ++hIter;
                //if(hist.isNorm) hratio->Scale((*hIter)->h->Integral()/hratio->Integral());
                hratio->Divide((*hIter)->h);
                leg->AddEntry(hratio, hvec.flabel().c_str());
                smartMax(hratio, leg, static_cast<TPad*>(gPad), min, max, lmax);
                minAvgWgt = std::min(minAvgWgt, 1.0);
            }
            else if(hvec.type.compare("stack") == 0)
            {
                THStack *stack = new THStack((hist.name + hvec.hcsVec.front()->label).c_str(), "stack");
                hvec.h = static_cast<TNamed*>(stack);

                double sow = 0, te = 0;
                TH1* thstacksucks = nullptr;
                int iStack = 0;
                for(auto ih = hvec.hcsVec.begin(); ih != hvec.hcsVec.end(); ++ih)
                {
                    //(*ih)->h->Scale(1.27);
                    (*ih)->h->SetMarkerColor(stackColors[iStack%NSTACKCOLORS]);
                    (*ih)->h->SetLineColor(stackColors[iStack%NSTACKCOLORS]);
                    (*ih)->h->SetFillColor(stackColors[iStack%NSTACKCOLORS]);
                    iStack++;
                    double integral = (*ih)->h->Integral(0, (*ih)->h->GetNbinsX() + 1);
                    if(     integral < 3.0)   sprintf(legEntry, "%s (%0.2lf)", (*ih)->label.c_str(), integral);
                    else if(integral < 1.0e5) sprintf(legEntry, "%s (%0.0lf)", (*ih)->label.c_str(), integral);
                    else                      sprintf(legEntry, "%s (%0.2e)",  (*ih)->label.c_str(), integral);
                    leg->AddEntry((*ih)->h, legEntry, "F");
                    sow += (*ih)->h->GetSumOfWeights();
                    te +=  (*ih)->h->GetEntries();
                    if(thstacksucks == nullptr)
                    {
                        thstacksucks = static_cast<TH1*>((*ih)->h->Clone());
                    }
                    else
                    {
                        thstacksucks->Add((*ih)->h);
                    }
                }
                for(auto ih = hvec.hcsVec.rbegin(); ih != hvec.hcsVec.rend(); ++ih)
                {
                    if(hist.isNorm) (*ih)->h->Scale(1.0/thstacksucks->Integral());
                    stack->Add((*ih)->h);
                }
                if(hist.isNorm) thstacksucks->Scale(1.0/thstacksucks->Integral());
                smartMax(thstacksucks, leg, static_cast<TPad*>(gPad), min, max, lmax);
                minAvgWgt = std::min(minAvgWgt, sow/te);
                if(thstacksucks) delete thstacksucks;
            }
        }

        gPad->SetLogy(hist.isLog);
        if (hist.setYLimits)
        {
            if (hist.ymin_ > 0 && hist.ymax_ > 0)
            {
                dummy->GetYaxis()->SetRangeUser(hist.ymin_, hist.ymax_);
            }
            else
            {
                std::cout << "ERROR for " << hist.name << ": ymin and ymax must be positive" << std::endl;
            }
        }
        else if(hist.isLog)
        {
            double locMin = std::min(0.2*minAvgWgt, std::max(0.00011, 0.05 * min));
            double legSpan = (log10(3*max) - log10(locMin)) * (leg->GetY1() - gPad->GetBottomMargin()) / ((1 - gPad->GetTopMargin()) - gPad->GetBottomMargin());
            double legMin = legSpan + log10(locMin);
            if(log10(lmax) > legMin)
            {
                double scale = (log10(lmax) - log10(locMin)) / (legMin - log10(locMin));
                max = pow(max/locMin, scale)*locMin;
            }
            dummy->GetYaxis()->SetRangeUser(locMin, 3*max);
        }
        else
        {
            double locMin = 0.0;
            double ratioMin = 0.5;
            double legMin = (1.2*max - locMin) * (leg->GetY1() - gPad->GetBottomMargin()) / ((1 - gPad->GetTopMargin()) - gPad->GetBottomMargin());
            if(lmax > legMin)
            {
                max *= (lmax - locMin)/(legMin - locMin);
            }
            dummy->GetYaxis()->SetRangeUser(0.0, max*1.2);
            if(hist.hists.front().type.compare("ratio") == 0) 
            {
                dummy->GetYaxis()->SetRangeUser(ratioMin, 2.0);
            }
        }
        dummy->Draw();

        for(auto& hvec : hist.hists)
        {
            if(!hvec.h)
            {
                for(auto& h : hvec.hcsVec)
                {
                    h->h->Draw("hist same");
                }
            }
            else
            {
                if(     hvec.type.compare("ratio") == 0) hvec.h->Draw("hist same");
                else if(hvec.type.compare("data") != 0)  hvec.h->Draw("hist same");
            }
        }
        // Make sure to always draw data on top
        for(auto& hvec : hist.hists)
        {
            if(hvec.h)
            {
                if(hvec.type.compare("data") == 0)  hvec.h->Draw("same");
            }
        }
        leg->Draw();

        fixOverlay();

        // Add the search bin boundaries for the search bin plots
        // if(hist.name.find("nSearchBin") != std::string::npos)
        // {
        //     drawSBregionDefCopy(dummy->GetMinimum(),dummy->GetMaximum());
        // }

        TH1 *dummy2 = nullptr, *h1 = nullptr, *h2 = nullptr;
        if(showRatio)
        {
            c->cd(2);
            gPad->SetPad("p2", "p2", 0, 0, 1, 2.5 / 9.0, kWhite, 0, 0);
            gPad->SetLeftMargin(0.15);
            gPad->SetRightMargin(0.06);
            gPad->SetTopMargin(0.01);
            gPad->SetBottomMargin(0.37);
            gPad->SetGridy();

            dummy2 = new TH1F("dummy2", "dummy2", 1000, hist.fhist()->GetBinLowEdge(1), hist.fhist()->GetBinLowEdge(hist.fhist()->GetNbinsX()) + hist.fhist()->GetBinWidth(hist.fhist()->GetNbinsX()));
            dummy2->GetXaxis()->SetTitle(hist.xAxisLabel.c_str());
            dummy2->GetXaxis()->SetTitleOffset(0.97);
            //dummy2->GetYaxis()->SetTitle("Data/MC");
            dummy2->GetYaxis()->SetTitle("Ratio");
            dummy2->GetYaxis()->SetTitleOffset(0.42);
            dummy2->GetYaxis()->SetNdivisions(3, 5, 0, true);

            dummy2->GetYaxis()->SetTitleSize(0.16 * 2 / 2.5);
            dummy2->GetYaxis()->SetLabelSize(0.20 * 2 / 2.5);
            dummy2->GetXaxis()->SetTitleSize(0.20 * 2 / 2.5);
            dummy2->GetXaxis()->SetLabelSize(0.20 * 2 / 2.5);

            dummy2->SetStats(0);
            dummy2->SetTitle(0);
            if(dummy2->GetNdivisions() % 100 > 5) dummy2->GetXaxis()->SetNdivisions(6, 5, 0);

            int iHist = 1;
            if(hist.hists.size() == 1 && hist.hists.front().type.compare("single") == 0)
            {
                for(auto& h : hist.hists.front().hcsVec)
                {
                    if(     iHist == hist.ratio.first)  h1 = static_cast<TH1*>(h->h->Clone());
                    else if(iHist == hist.ratio.second) h2 = static_cast<TH1*>(h->h->Clone());
                    ++iHist;
                }
            }
            else
            {
                for(auto& hvec : hist.hists)
                {
                    if(hvec.type.compare("single") == 0 || hvec.type.compare("data") == 0)
                    {
                        if(iHist == hist.ratio.first)  h1 = static_cast<TH1*>(hvec.hcsVec.front()->h->Clone());
                        if(iHist == hist.ratio.second) h2 = static_cast<TH1*>(hvec.hcsVec.front()->h->Clone());
                    }
                    else if(hvec.type.compare("stack") == 0)
                    {
                        bool firstHIS = true;
                        TH1* thstacksucks = 0;
                        if(iHist == hist.ratio.first || iHist == hist.ratio.second)
                        {
                            for(auto& h : hvec.hcsVec)
                            {
                                if(firstHIS)
                                {
                                    firstHIS = false;
                                    thstacksucks = static_cast<TH1*>(h->h->Clone());
                                }
                                else
                                {
                                    thstacksucks->Add(h->h);
                                }
                            }
                            if(iHist == hist.ratio.first)  h1 = static_cast<TH1*>(thstacksucks->Clone());
                            if(iHist == hist.ratio.second) h2 = static_cast<TH1*>(thstacksucks->Clone());
                        }
                        if(thstacksucks) delete thstacksucks;
                    }
                    ++iHist;
                }
            }

            TF1 * fline = new TF1("line", "pol0", hist.fhist()->GetBinLowEdge(1), hist.fhist()->GetBinLowEdge(hist.fhist()->GetNbinsX()) + hist.fhist()->GetBinWidth(hist.fhist()->GetNbinsX()));
            TF1 * fline2 = nullptr;
            TF1 * fline3 = nullptr;
            std::string drawOptions = "";

            if(h1 && h2)
            {
                if(hist.isRatio)
                {
                    fline->SetParameter(0, 1);
                    drawOptions = "same PE1";
                    bool autoScale = false;

                    h1->Divide(h2);
                    h1->SetMarkerStyle(20);
                    h1->SetMarkerColor(h1->GetLineColor());
                    double d2ymin = 0.4;
                    double d2ymax = 1.6;
                    if (autoScale)
                    {
                        d2ymin = -0.1;
                        d2ymax =  1.1;
                        for(int iBin = 1; iBin <= h1->GetNbinsX(); ++iBin)
                        {
                            if(h1->GetBinContent(iBin) < 2.1)
                            {
                                d2ymax = std::max(d2ymax, h1->GetBinContent(iBin));
                            }
                        }
                    }
                    dummy2->GetYaxis()->SetRangeUser(d2ymin, d2ymax);
                }
                else // pull distribution
                {
                    fline->SetParameter(0, 0);
                    drawOptions = "same histP";

                    h1->SetLineStyle(0);
                    h1->SetMarkerStyle(20);
                    h1->SetMarkerColor(h1->GetLineColor());

                    const double absoluteMaxPull = 10.0;
                    double maxPull = 2.0;
                    double sumW2_1 = 0.0, sumW_1 = 0.0;
                    double sumW2_2 = 0.0, sumW_2 = 0.0;
                    for(int iBin = 1; iBin <= h1->GetNbinsX(); ++iBin)
                    {
                        sumW_1 += h1->GetBinContent(iBin);
                        sumW_2 += h2->GetBinContent(iBin);
                        sumW2_1 += pow(h1->GetBinError(iBin), 2);
                        sumW2_2 += pow(h2->GetBinError(iBin), 2);
                    }
                    for(int iBin = 1; iBin <= h1->GetNbinsX(); ++iBin)
                    {
                        double binVal = binVal = h1->GetBinContent(iBin) - h2->GetBinContent(iBin);
                        double binErr = -999.9;
                        if(h1->GetBinContent(iBin) > 1e-10 && h2->GetBinContent(iBin) > 1e-10)
                        {
                            binErr = sqrt(pow(h1->GetBinError(iBin), 2) + pow(h2->GetBinError(iBin), 2));
                        }
                        else if(h1->GetBinContent(iBin) > 1e-10)
                        {
                            binErr = sqrt(pow(h1->GetBinError(iBin), 2) + pow(1.8 * sumW2_1/sumW_1, 2));
                        }
                        else if(h2->GetBinContent(iBin) > 1e-10)
                        {
                            binErr = sqrt(pow(h2->GetBinError(iBin), 2) + pow(1.8 * sumW2_2/sumW_2, 2));
                        }
                        h1->SetBinContent(iBin, binVal/binErr);
                        if(fabs(h1->GetBinContent(iBin)) < absoluteMaxPull) maxPull = std::max(maxPull, fabs(h1->GetBinContent(iBin)));
                    }
                    double d2ymin = -std::min(ceil(maxPull*1.2), absoluteMaxPull);
                    double d2ymax =  std::min(ceil(maxPull*1.2), absoluteMaxPull);
                    dummy2->GetYaxis()->SetRangeUser(d2ymin, d2ymax);
                    dummy2->GetYaxis()->SetTitle("Pull");
                    dummy2->GetYaxis()->SetNdivisions(2, 5, 0);

                    fline2 = new TF1("line", "pol0", hist.fhist()->GetBinLowEdge(1), hist.fhist()->GetBinLowEdge(hist.fhist()->GetNbinsX()) + hist.fhist()->GetBinWidth(hist.fhist()->GetNbinsX()));
                    fline2->SetParameter(0, 1);
                    fline3 = new TF1("line", "pol0", hist.fhist()->GetBinLowEdge(1), hist.fhist()->GetBinLowEdge(hist.fhist()->GetNbinsX()) + hist.fhist()->GetBinWidth(hist.fhist()->GetNbinsX()));
                    fline3->SetParameter(0, -1);
                    fline2->SetLineColor(kBlack);
                    fline2->SetLineStyle(kDashed);
                    fline3->SetLineColor(kBlack);
                    fline3->SetLineStyle(kDashed);
                }

                dummy2->Draw();
                fline->SetLineColor(h2->GetLineColor());
                fline->Draw("same");
                if(fline2) fline2->Draw("same");
                if(fline3) fline3->Draw("same");
                h1->Draw(drawOptions.c_str());

                fixOverlay();
            }
        }

        c->cd(1);
        char lumistamp[128];
        sprintf(lumistamp, "%.1f fb^{-1} (13 TeV)", getLumi() / 1000.0);

        TLatex mark;
        mark.SetNDC(true);

        //Draw CMS mark
        double x_offset = 0.0;
        mark.SetTextAlign(11);
        mark.SetTextSize(0.042 * fontScale * 1.25);
        //mark.SetTextSize(0.04 * 1.1 * 8 / 6.5 * 1.25 * fontScale);
        mark.SetTextFont(61);
        //mark.DrawLatex(gPad->GetLeftMargin(), 1 - (gPad->GetTopMargin() - 0.017), "CMS"); // #scale[0.8]{#it{Preliminary}}");
        mark.DrawLatex(gPad->GetLeftMargin() + x_offset, 1 - (gPad->GetTopMargin() - 0.017), "CMS"); // #scale[0.8]{#it{Preliminary}}");
        mark.SetTextSize(0.042 * fontScale);
        //mark.SetTextSize(0.04 * 1.1 * 8 / 6.5 * fontScale);
        mark.SetTextFont(52);
        //mark.DrawLatex(gPad->GetLeftMargin() + 0.095, 1 - (gPad->GetTopMargin() - 0.017), "Preliminary");
        mark.DrawLatex(gPad->GetLeftMargin() + x_offset + 0.095, 1 - (gPad->GetTopMargin() - 0.017), "Preliminary");
        //mark.DrawLatex(gPad->GetLeftMargin() + 0.095, 1 - (gPad->GetTopMargin() - 0.017), "Supplementary");

        //Draw lumistamp
        mark.SetTextFont(42);
        mark.SetTextAlign(31);
        mark.DrawLatex(1 - gPad->GetRightMargin(), 1 - (gPad->GetTopMargin() - 0.017), lumistamp);

        fixOverlay();
        c->Print((plotDir_ + hist.name+".pdf").c_str());
        if (make_png)
        {
            c->Print((plotDir_ + hist.name+".png").c_str());
        }

        delete leg;
        delete dummy;
        if(dummy2) delete dummy2;
        delete c;
        if(h1) delete h1;
        if(h2) delete h2;
    }
    // Plotter::plot() is complete: green
    std::cout << "\033[1;32m  Plotter::plot() is complete. \033[0m"<< std::endl;
}

void Plotter::fillHist(TH1 * const h, const VarName& name, const NTupleReader& tr, const double weight)
{
    std::string type;
    tr.getType(name.name, type);

    if(type.find("vector") != std::string::npos)
    {

        if(type.find("*") != std::string::npos)
        {
            if(type.find("TLorentzVector") != std::string::npos) 
            {
              if(type.find("const") != std::string::npos) fillHistFromVec<const TLorentzVector*>(h, name, tr, weight);
              else                                        fillHistFromVec<TLorentzVector*>(h, name, tr, weight);
            }
        }
        else
        {
            if     (type.find("pair")           != std::string::npos) fillHistFromVec<std::pair<double, double>>(h, name, tr, weight);
            else if(type.find("double")         != std::string::npos) fillHistFromVec<double>(h, name, tr, weight);
            else if(type.find("float")          != std::string::npos) fillHistFromVec<float>(h, name, tr, weight);
            else if(type.find("unsigned int")   != std::string::npos) fillHistFromVec<unsigned int>(h, name, tr, weight);
            else if(type.find("int")            != std::string::npos) fillHistFromVec<int>(h, name, tr, weight);
            else if(type.find("TLorentzVector") != std::string::npos) fillHistFromVec<TLorentzVector>(h, name, tr, weight);
        }
    }
    else
    {
        if     (type.find("double")         != std::string::npos) h->Fill(tr.getVar<double>(name.name), weight);
        else if(type.find("float")          != std::string::npos) h->Fill(tr.getVar<float>(name.name), weight);
        else if(type.find("unsigned int")   != std::string::npos) h->Fill(tr.getVar<unsigned int>(name.name), weight);
        else if(type.find("int")            != std::string::npos) h->Fill(tr.getVar<int>(name.name), weight);
        else if(type.find("float")          != std::string::npos) h->Fill(tr.getVar<float>(name.name), weight);
        else if(type.find("char")           != std::string::npos) h->Fill(tr.getVar<char>(name.name), weight);
        else if(type.find("short")          != std::string::npos) h->Fill(tr.getVar<short>(name.name), weight);
        else if(type.find("long")           != std::string::npos) h->Fill(tr.getVar<long>(name.name), weight);
    }
}

template<> inline void Plotter::vectorFill(TH1 * const h, const VarName& name, const TLorentzVector& obj, const double weight)
{
    h->Fill(tlvGetValue(name.var, obj), weight);
}

template<> inline void Plotter::vectorFill(TH1 * const h, const VarName& name, const std::pair<double, double>& obj, const double weight)
{
    h->Fill(obj.first, obj.second * weight);
}

void Plotter::smartMax(const TH1* const h, const TLegend* const l, const TPad* const p, double& gmin, double& gmax, double& gpThreshMax, const bool error) const
{
    const bool isLog = p->GetLogy();
    double min = 9e99;
    double max = -9e99;
    double pThreshMax = -9e99;
    int threshold = static_cast<int>(h->GetNbinsX()*(l->GetX1() - p->GetLeftMargin())/((1 - p->GetRightMargin()) - p->GetLeftMargin()));

    for(int i = 1; i <= h->GetNbinsX(); ++i)
    {
        double bin = 0.0;
        if(error) bin = h->GetBinContent(i) + h->GetBinError(i);
        else      bin = h->GetBinContent(i);
        if(bin > max) max = bin;
        else if(bin > 1e-10 && bin < min) min = bin;
        if(i >= threshold && bin > pThreshMax) pThreshMax = bin;
    }

    gpThreshMax = std::max(gpThreshMax, pThreshMax);
    gmax = std::max(gmax, max);
    gmin = std::min(gmin, min);
}
