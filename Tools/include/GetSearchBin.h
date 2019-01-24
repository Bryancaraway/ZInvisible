#ifndef GETSEARCHBIN_H
#define GETSEARCHBIN_H

#include "TypeDefinitions.h"
#include "PhotonTools.h"

#include "SusyAnaTools/Tools/NTupleReader.h"
#include "SusyAnaTools/Tools/customize.h"
#include "SusyAnaTools/Tools/searchBins.h"
#include "SusyAnaTools/Tools/SB2018.h"
#include "TopTagger/Tools/cpp/TaggerUtility.h"
#include "TopTagger/Tools/cpp/PlotUtility.h"
#include "ScaleFactors.h"
#include "ScaleFactorsttBar.h"

#include "TopTagger.h"
#include "TTModule.h"
#include "TopTaggerUtilities.h"
#include "TopTaggerResults.h"
#include "TopTagger/Tools/cpp/PlotUtility.h"

#include "TopTagger/TopTagger/interface/TopObject.h"

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
            const auto& nb                  = tr.getVar<int>("cntCSVSZinv");
            const auto& nt                  = tr.getVar<int>("nTopCandSortedCntZinv");
            const auto& njets               = tr.getVar<int>("cntNJetsPt20Eta24Zinv");
            const auto& nWs                 = tr.getVar<int>("nWs");
            const auto& nResolvedTops       = tr.getVar<int>("nResolvedTops");
            const auto& met                 = tr.getVar<data_t>("cleanMetPt");
            const auto& ht                  = tr.getVar<data_t>("HTZinv");
            const auto& bottompt_scalar_sum = tr.getVar<data_t>("ptbZinv");
            const auto& mtb                 = tr.getVar<data_t>("mtbZinv");
            const auto& softbLVec           = tr.getVec<TLorentzVector>("softbLVecZinv");
            const auto& ISRLVec             = tr.getVec<TLorentzVector>("vISRJetZinv");
            
            //const auto& bjetsLVec           = tr.getVec<TLorentzVector>("vBjsZinv");
            //const auto& jet_vec = tr->getVec<TLorentzVector>(jetVecLabel);
            
            //const auto& MT2 = tr.getVar<data_t>("best_had_brJet_MT2Zinv");
            //const auto& cleanMetPhi = tr.getVar<data_t>("cleanMetPhi");
            //const auto& nTopCandSortedCnt1b = tr.getVar<int>("nTopCandSortedCntZinv1b");
            //const auto& nTopCandSortedCnt2b = tr.getVar<int>("nTopCandSortedCntZinv2b");
            //const auto& nTopCandSortedCnt3b = tr.getVar<int>("nTopCandSortedCntZinv3b");
            //const auto& MT2_1b = tr.getVar<data_t>("best_had_brJet_MT2Zinv1b");
            //const auto& MT2_2b = tr.getVar<data_t>("best_had_brJet_MT2Zinv2b");
            //const auto& MT2_3b = tr.getVar<data_t>("best_had_brJet_MT2Zinv3b");
            //const auto& weight1fakeb = tr.getVar<data_t>("weight1fakeb");
            //const auto& weight2fakeb = tr.getVar<data_t>("weight2fakeb");
            //const auto& weight3fakeb = tr.getVar<data_t>("weight3fakeb");
            //const auto& nJet1bfakeWgt = tr.getVar<data_t>("nJet1bfakeWgt");
            //const auto& nJet2bfakeWgt = tr.getVar<data_t>("nJet2bfakeWgt");
            //const auto& nJet3bfakeWgt = tr.getVar<data_t>("nJet3bfakeWgt");
            //const auto& nTopCandSortedCnt = tr.getVar<int>("nTopCandSortedCntZinv");
            
            //------------------------------------------//
            //--- Updated Search Bins (January 2019) ---//
            //------------------------------------------//

            //================================SUS-16-049 (team_A) search bin low dm================================
            //int SB_team_A_lowdm(int njets, int nb, int nSV, float ISRpt, float bottompt_scalar_sum, float met)
            //================================search bin v2 high dm================================================
            //int SBv2_highdm(float mtb_cut, float mtb, int njets, int ntop, int nw, int nres, int nb, float met, float ht)
            
            //int cntCSVS = 0;
            //cntCSVS = AnaFunctions::countCSVS(jet_vec, tr->getVec<float>(CSVVecLabel), AnaConsts::DeepCSV.at(eraLabel).at("cutM"), AnaConsts::bTagArr, vBidxs); 
            
            int nSV = softbLVec.size();
            float ISRpt = 0.0;
            if(ISRLVec.size() == 1) ISRpt = ISRLVec.at(0).Pt();
            float mtb_cut = 175.0;

            int nSearchBinLowDM = SB_team_A_lowdm(njets, nb, nSV, ISRpt, bottompt_scalar_sum, met);
            int nSearchBinHighDM = SBv2_highdm(mtb_cut, mtb, njets, nt, nWs, nResolvedTops, nb, met, ht);

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
            //int nSearchBin = sbins.find_Binning_Index(cntCSVS, nTopCandSortedCnt, MT2, cleanMet, HT);            


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

            //if(cntCSVS == 0)
            //{
            //    //nb0Bins->push_back(std::make_pair(find_Binning_Index(0, nTopCandSortedCnt, MT2, cleanMet), 1.0));
            //    //nb0Bins->emplace_back(std::pair<double, double>(sbins.find_Binning_Index(1, nTopCandSortedCnt1b, MT2_1b, cleanMet, HT), wnb01 * weight1fakeb));
            //    //nb0Bins->emplace_back(std::pair<double, double>(sbins.find_Binning_Index(2, nTopCandSortedCnt2b, MT2_2b, cleanMet, HT), wnb02 * weight2fakeb));
            //    //nb0Bins->emplace_back(std::pair<double, double>(sbins.find_Binning_Index(3, nTopCandSortedCnt3b, MT2_3b, cleanMet, HT), wnb03 * weight3fakeb));

            //    //nb0NJwBins->emplace_back(std::pair<double, double>(sbins.find_Binning_Index(1, nTopCandSortedCnt1b, MT2_1b, cleanMet, HT), nJet1bfakeWgt));
            //    //nb0NJwBins->emplace_back(std::pair<double, double>(sbins.find_Binning_Index(2, nTopCandSortedCnt2b, MT2_2b, cleanMet, HT), nJet2bfakeWgt));
            //    //nb0NJwBins->emplace_back(std::pair<double, double>(sbins.find_Binning_Index(3, nTopCandSortedCnt3b, MT2_3b, cleanMet, HT), nJet3bfakeWgt));

            //    //nb0BinsNW->emplace_back(sbins.find_Binning_Index(1, nTopCandSortedCnt1b, MT2_1b, cleanMet, HT));
            //    //nb0BinsNW->emplace_back(sbins.find_Binning_Index(2, nTopCandSortedCnt2b, MT2_2b, cleanMet, HT));
            //    //nb0BinsNW->emplace_back(sbins.find_Binning_Index(3, nTopCandSortedCnt3b, MT2_3b, cleanMet, HT));
            //}

            tr.registerDerivedVar("nSearchBinLowDM", nSearchBinLowDM);
            tr.registerDerivedVar("nSearchBinHighDM", nSearchBinHighDM);
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
