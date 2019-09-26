#ifndef STOP_PLOTTER_H
#define STOP_PLOTTER_H

#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TGraph.h"
#include "TProfile.h"
#include "TLegend.h"
#include "TPad.h"
#include "TStyle.h"
#include "TPaveText.h"

#include <vector>
#include <utility>
#include <string>
#include <set>
#include <memory>

#include "NTupleReader.h"
#include "samples.h"

class RegisterFunctions;

class Plotter
{
private:
    class HistCutSummary;
    class Cuttable;

    class VarName
    {
    public:
        std::string name;
        std::string var;
        int index;
        std::shared_ptr<Cuttable> cuts;

        VarName() : cuts(nullptr) {index = -1;}
    };

    class Cut
    {
    public:
        std::string rawName;
        VarName name;
        char type;
        double val, val2;
        bool inverted;

        Cut(std::string s, char t, bool inv, double v, double v2 = 0);
        bool passCut(const NTupleReader& tr, const int index = -999) const;
    private:
        void parseName();
        double translateVar(const NTupleReader& tr, const int index = -999) const;
        bool boolReturn(const NTupleReader& tr) const;
    };

    class Cuttable
    {
    public:
        Cuttable() {}
        Cuttable(const std::string& c);
        bool passCuts(const NTupleReader& tr, const int index = -999) const;
        void setCuts(const std::string& c);
        void extractCuts(std::set<std::string>& ab) const;
        const std::string& getCuts() const {return cuts_;}

    private:
        std::string cuts_;
        std::vector<Cut> cutVec_;
        void parseCutString();
    };

    class HistVecAndType
    {
    public:
        TNamed* h;
        std::vector<std::shared_ptr<HistCutSummary>> hcsVec;
        std::string type;

        HistVecAndType(std::vector<std::shared_ptr<HistCutSummary>> hcsVec, std::string type) : hcsVec(hcsVec), type(type) {h = nullptr;}
        ~HistVecAndType() { if(h) delete h;}

        const std::string& flabel() const {return hcsVec.front()->label;}
    };

public:

    class DatasetSummary : public Cuttable
    {
    public:
        std::string label, weightStr;
        std::vector<AnaSamples::FileSummary> files;
        double kfactor;

        DatasetSummary() {}
        DatasetSummary(std::string lab, std::vector<AnaSamples::FileSummary>& f, std::string cuts = "", std::string weights = "", double k = 1.0);

        double getWeight(const NTupleReader& tr) const;
        double extractWeightNames(std::set<std::string>& ab) const;

    private:
        std::vector<std::string> weightVec_;
        void parseWeights();
    };

    class DataCollection
    {
    public:
        std::string type;
        std::vector<std::pair<std::string, std::vector<DatasetSummary>>> datasets;

        DataCollection(std::string type, std::vector<std::pair<std::string, DatasetSummary>> ds);
        DataCollection(std::string type, std::string var, std::vector<DatasetSummary> ds);

        DataCollection(std::string type, std::vector<std::pair<std::string, std::vector<DatasetSummary>>> ds);
        DataCollection(std::string type, std::string var, std::vector<std::vector<DatasetSummary>> vvds);
    };

    class HistSummary : public Cuttable
    {
    public:
        std::vector<HistVecAndType> hists;
        std::string name;
        int nBins;
        double low, high;
        double ymin_, ymax_;
        bool setYLimits;
        bool isLog, isNorm, isRatio;
        std::string xAxisLabel, yAxisLabel;
        std::pair<int, int> ratio;
        std::vector<double> binEdges;

        HistSummary() {}
        HistSummary(std::string l, std::vector<Plotter::DataCollection> ns, std::pair<int, int> ratio, std::string cuts, int nb, double ll, double ul, bool log, bool norm, std::string xal, std::string yal, bool isRatio = true);
        HistSummary(std::string l, std::vector<Plotter::DataCollection> ns, std::pair<int, int> ratio, std::string cuts, int nb, double ll, double ul, double ymin, double max, bool log, bool norm, std::string xal, std::string yal, bool isRatio = true);
        HistSummary(std::string l, std::vector<Plotter::DataCollection> ns, std::pair<int, int> ratio, std::string cuts, std::vector<double> be, bool log, bool norm, std::string xal, std::string yal, bool isRatio = true);
        ~HistSummary();

        TH1* fhist(){if(hists.size()) return hists.front().hcsVec.front()->h;}
    private:
        void parseName(std::vector<Plotter::DataCollection>& ns);
    };

    class CutFlowSummary
    {
    public:
        TH1 *h;
        std::string name;
        Plotter::DataCollection dc;
        const Plotter::DatasetSummary* dssp;

        void setDSS(const Plotter::DatasetSummary* d) { dssp = d; }
        void fillHist(const NTupleReader& tr, const double& weight);
        void generateHist();

        CutFlowSummary(std::string n, Plotter::DataCollection ns, std::vector<std::string> cutLevels);
        ~CutFlowSummary();

    private:
        std::vector<Cuttable> cuts_;
    };

    Plotter(std::vector<HistSummary>& h, std::set<AnaSamples::FileSummary>& t, const bool readFromTuple = true, std::string ofname = "", const int nFile = -1, const int startFile = 0, const int nEvts = -1);
    ~Plotter();

    void read();

    static void parseSingleVar(const std::string& name, VarName& var);

    void setPlotDir(const std::string plotDir);
    void setLumi(const double lumi);
    void setDoHists(const bool doHists);
    void setDoTuple(const bool doTuple);
    void setRegisterFunction(RegisterFunctions* rf);
    void setPrintInterval(const int printInterval);
    void setCutFlows(std::vector<CutFlowSummary> cfs);

    double getLumi();

    void plot();
    void saveHists();

private:
    std::vector<HistSummary> hists_;
    std::vector<CutFlowSummary> cutFlows_;
    std::set<AnaSamples::FileSummary> trees_;
    std::string plotDir_;
    std::string foutTupleName_;
    TFile *fout_, *foutTuple_;
    bool readFromTuple_;
    const int nFile_;
    const int startFile_;
    const int maxEvts_;
    int printInterval_;
    double lumi_;
    bool doHists_, doTuple_;
    RegisterFunctions *registerfunc_;

    class HistCutSummary
    {
    public:
        std::string label, name;
        VarName variable;
        TH1 *h;
        const HistSummary *hs;
        const DatasetSummary* dssp;
        std::vector<DatasetSummary> dss;

    HistCutSummary(const std::string& lab, const std::string& name, const VarName v, const HistSummary* hsum, const std::vector<DatasetSummary>& ds) : label(lab), name(name), h(nullptr), variable(v), hs(hsum), dss(ds) {}
        ~HistCutSummary();
    };

    void createHistsFromTuple();
    void createHistsFromFile();
    void fillHist(TH1 * const h, const VarName& name, const NTupleReader& tr, const double weight);
    void smartMax(const TH1* const h, const TLegend* const l, const TPad* const p, double& gmin, double& gmax, double& gpThreshMax, const bool error = false) const;

    template<typename T> static const double& tlvGetValue(const std::string& name, const T& v)
    {
        if     (name.find("pt")  != std::string::npos)
        {
            const auto& retval = v.Pt();
            return retval;
        }
        else if(name.find("eta") != std::string::npos)
        {
            const auto& retval = v.Eta();
            return retval;
        }
        else if(name.find("phi") != std::string::npos)
        {
            const auto& retval = v.Phi();
            return retval;
        }
        else if(name.find("E")   != std::string::npos)
        {
            const auto& retval = v.E();
            return retval;
        }
        else if(name.find("M")   != std::string::npos)
        {
            const auto& retval = v.M();
            return retval;
        }
        else if(name.find("Mt")  != std::string::npos)
        {
            const auto& retval = v.Mt();
            return retval;
        }
        else
        {
            printf("Invalid lorentz variable: \"%s\", returning nullptr segfault incoming!!!\n", name.c_str());
            fflush(stdin);
            return *static_cast<double*>(nullptr);
        }
    }

    template<typename T> inline const T pointerDeref(T obj) const
    {
        return obj;
    }

    template<typename T> inline const T& pointerDeref(T* const obj) const
    {
        return *obj;
    }

    template<typename T> void fillHistFromVec(TH1* const h, const VarName& name, const NTupleReader& tr, const double weight)
    {
        if(name.var.compare("size") == 0)
        {
            const auto& vec = tr.getVec<T>(name.name);
            if(&vec != nullptr) h->Fill(vec.size(), weight);
        }
        else
        {
            if(name.index >= 0)
            {
                auto& var = getVarFromVec<T>(name, tr);
                if(&var != nullptr) vectorFill(h, name, pointerDeref(var), weight);
            }
            else
            {
                const auto& vec = tr.getVec<T>(name.name);
                if(name.cuts)
                {
                    //per object cuts are defined, so apply these explicitely here (maybe can be made more efficient later)
                    for(int iObj = 0; iObj < vec.size(); ++iObj)
                    {
                        if(name.cuts->passCuts(tr, iObj)) vectorFill(h, name, pointerDeref(vec[iObj]), weight);
                    }
                }
                else
                {
                    //no per object cuts are defined
                    for(auto& var : vec) vectorFill(h, name, pointerDeref(var), weight);
                }
            }
        }
    }

    template<typename T, typename R = T> static const R& getVarFromVec(const VarName& name, const NTupleReader& tr, const int index = -999)
    {
        const auto& vec = tr.getVec<T>(name.name);

        if(&vec != nullptr)
        {
            int i;
            if(index >= 0) i = index;
            else           i = name.index;
            if(i < vec.size()) return vec[i];
            else return *static_cast<R*>(nullptr);
        }
        return *static_cast<R*>(nullptr);
    }

    template<typename T> inline void vectorFill(TH1 * const h, const VarName& name, const T& obj, const double weight)
    {
        h->Fill(obj, weight);
    }
};

typedef Plotter::HistSummary PHS;
typedef AnaSamples::FileSummary AFS;
typedef Plotter::DatasetSummary PDS;
typedef Plotter::DataCollection PDC;

inline bool operator< (const Plotter::DataCollection& lhs, const Plotter::DataCollection& rhs)
{
    return lhs.datasets < rhs.datasets;
}

inline bool operator< (const Plotter::DatasetSummary& lhs, const Plotter::DatasetSummary& rhs)
{
    return lhs.label < rhs.label || lhs.files < rhs.files;
}

#endif
