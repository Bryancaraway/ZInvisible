#ifndef GETSEARCHBIN_H
#define GETSEARCHBIN_H

#include "PhotonTools.h"

#include "SusyAnaTools/Tools/NTupleReader.h"
#include "SusyAnaTools/Tools/customize.h"
#include "SusyAnaTools/Tools/searchBins.h"
#include "TopTagger/Tools/cpp/TaggerUtility.h"
#include "TopTagger/Tools/cpp/PlotUtility.h"
#include "ScaleFactors.h"
#include "ScaleFactorsttBar.h"

#include "TopTagger.h"
#include "TTModule.h"
#include "TopTaggerUtilities.h"
#include "TopTaggerResults.h"
#include "TopTagger/Tools/cpp/PlotUtility.h"

#include "TopTagger/TopTagger/include/TopObject.h"

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TMath.h"
#include "TLorentzVector.h"
#include "Math/VectorUtil.h"
#include "TRandom3.h"
#include "TVector2.h"

#include <vector>
#include <iostream>
#include <string>
#include <set>

namespace plotterFunctions
{
    class GetSearchBin
    {
    private:
        SearchBins sbins;

        void getSearchBin(NTupleReader& tr)
        {
            const int& cntCSVS = tr.getVar<int>("cntCSVSZinv");
            const int& nTopCandSortedCnt = tr.getVar<int>("nTopCandSortedCntZinv");
            //const int& nTopCandSortedCnt1b = tr.getVar<int>("nTopCandSortedCntZinv1b");
            //const int& nTopCandSortedCnt2b = tr.getVar<int>("nTopCandSortedCntZinv2b");
            //const int& nTopCandSortedCnt3b = tr.getVar<int>("nTopCandSortedCntZinv3b");
            const double& cleanMet = tr.getVar<double>("cleanMetPt");
            const double& cleanMetPhi = tr.getVar<double>("cleanMetPhi");
            const double& MT2 = tr.getVar<double>("best_had_brJet_MT2Zinv");
            //const double& MT2_1b = tr.getVar<double>("best_had_brJet_MT2Zinv1b");
            //const double& MT2_2b = tr.getVar<double>("best_had_brJet_MT2Zinv2b");
            //const double& MT2_3b = tr.getVar<double>("best_had_brJet_MT2Zinv3b");
            //const double& weight1fakeb = tr.getVar<double>("weight1fakeb");
            //const double& weight2fakeb = tr.getVar<double>("weight2fakeb");
            //const double& weight3fakeb = tr.getVar<double>("weight3fakeb");
            //
            //const double& nJet1bfakeWgt = tr.getVar<double>("nJet1bfakeWgt");
            //const double& nJet2bfakeWgt = tr.getVar<double>("nJet2bfakeWgt");
            //const double& nJet3bfakeWgt = tr.getVar<double>("nJet3bfakeWgt");
            const double& HT            = tr.getVar<double>("HTZinv");
            //const int& nTopCandSortedCnt = tr.getVar<int>("nTopCandSortedCntZinv");
            //            //top
            /*
            std::shared_ptr<TopTagger> ttPtr;
            int monoJet;
           const TopTaggerResults& ttr = ttPtr->getResults();
               std::vector<TopObject*> Ntop = ttr.getTops();
              for(int i=1; i<nTopCandSortedCnt; i++){
               if(Ntop[i]->getNConstituents() == 1) monoJet++;
                  }
             std::cout<<monoJet<<std::endl;
            */
            //int nSearchBin = sbins.find_Binning_Index(cntCSVS, nTopCandSortedCnt, MT2, cleanMet);
            int nSearchBin = sbins.find_Binning_Index(cntCSVS, nTopCandSortedCnt, MT2, cleanMet, HT);            

            //std::vector<std::pair<double, double> > * nb0Bins = new std::vector<std::pair<double, double> >();
            //std::vector<std::pair<double, double> > * nb0NJwBins = new std::vector<std::pair<double, double> >();
            //std::vector<double> * nb0BinsNW = new std::vector<double>();

            //weights based on total N(b) yields vs. N(b) = 0 control region
            //These weights are derived from the rato of events in the N(t) = 1, 2, 3 bins after all baseline cuts except b tag between the
            //N(b) = 0 control region and each N(b) signal region using Z->nunu MC.  They account for both the combinatoric reweighting factor
            //as well as the different event yields between the control region and each signal region.
            //const double wnb01 = 3.820752e-02;//3.478840e-02;//6.26687e-2;
            //const double wnb02 = 2.946461e-03;//2.586369e-03;//5.78052e-3;
            //const double wnb03 = 1.474770e-04;//1.640077e-04;//7.08235e-4;
            //
            //// weights to apply when doing b-faking
            //const double w1b = wnb01 * weight1fakeb;
            //const double w2b = wnb02 * weight2fakeb;
            //const double w3b = wnb03 * weight3fakeb;

            if(cntCSVS == 0)
            {
                //nb0Bins->push_back(std::make_pair(find_Binning_Index(0, nTopCandSortedCnt, MT2, cleanMet), 1.0));
                //nb0Bins->emplace_back(std::pair<double, double>(sbins.find_Binning_Index(1, nTopCandSortedCnt1b, MT2_1b, cleanMet, HT), wnb01 * weight1fakeb));
                //nb0Bins->emplace_back(std::pair<double, double>(sbins.find_Binning_Index(2, nTopCandSortedCnt2b, MT2_2b, cleanMet, HT), wnb02 * weight2fakeb));
                //nb0Bins->emplace_back(std::pair<double, double>(sbins.find_Binning_Index(3, nTopCandSortedCnt3b, MT2_3b, cleanMet, HT), wnb03 * weight3fakeb));

                //nb0NJwBins->emplace_back(std::pair<double, double>(sbins.find_Binning_Index(1, nTopCandSortedCnt1b, MT2_1b, cleanMet, HT), nJet1bfakeWgt));
                //nb0NJwBins->emplace_back(std::pair<double, double>(sbins.find_Binning_Index(2, nTopCandSortedCnt2b, MT2_2b, cleanMet, HT), nJet2bfakeWgt));
                //nb0NJwBins->emplace_back(std::pair<double, double>(sbins.find_Binning_Index(3, nTopCandSortedCnt3b, MT2_3b, cleanMet, HT), nJet3bfakeWgt));

                //nb0BinsNW->emplace_back(sbins.find_Binning_Index(1, nTopCandSortedCnt1b, MT2_1b, cleanMet, HT));
                //nb0BinsNW->emplace_back(sbins.find_Binning_Index(2, nTopCandSortedCnt2b, MT2_2b, cleanMet, HT));
                //nb0BinsNW->emplace_back(sbins.find_Binning_Index(3, nTopCandSortedCnt3b, MT2_3b, cleanMet, HT));
            }

            tr.registerDerivedVar("nSearchBin", nSearchBin);
            //tr.registerDerivedVec("nb0BinsNW", nb0BinsNW);
            //tr.registerDerivedVec("nb0Bins", nb0Bins);
            //tr.registerDerivedVec("nb0NJwBins", nb0NJwBins);
            //tr.registerDerivedVar("weight1fakebComb", w1b);
            //tr.registerDerivedVar("weight2fakebComb", w2b);
            //tr.registerDerivedVar("weight3fakebComb", w3b);
        }

    public:

        GetSearchBin(std::string sb_era) : sbins(sb_era) {}

        void operator()(NTupleReader& tr)
        {
            getSearchBin(tr);
        }
    };
}

#endif
