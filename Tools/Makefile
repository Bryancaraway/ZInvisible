IDIR       = .
ODIR       = obj
SDIR       = .
RSDIR      = $(CMSSW_BASE)/src/SusyAnaTools/Tools

TTIDIR     = $(CMSSW_BASE)/src/TopTagger/TopTagger/interface
TPIDIR     = $(CMSSW_BASE)/src/TopTagger/CfgParser/include
TPTDIR     = $(CMSSW_BASE)/src
TTTDIR     = $(CMSSW_BASE)/src/TopTagger/TopTagger/test

TTSDIR     = $(CMSSW_BASE)/src/TopTagger/Tools/cpp

TTTSDIR    = $(CMSSW_BASE)/src/TopTaggerTools/Tools/src
TTTIDIR    = $(CMSSW_BASE)/src/TopTaggerTools/Tools/include

ZINVISIBLE = $(CMSSW_BASE)/src/ZInvisible

# TODO for the brave: use Makefile.in, configure.ac, and configure script to create Makefile from Makefile.in
# Then you can use sytax like this:
# ZINVISIBLE = @ZINVISIBLE@ 

CXX        = g++

CXXFLAGS  += -I. -I$(CMSSW_BASE)/src -std=c++0x -I$(LHAPDF_DATA_PATH)/../../include
## Optimization flag
CXXFLAGS += -g #-O3
## Enable the maximun warning
#CXXFLAGS += -Wall -Wextra -Weffc++ -g

PYTHONCFG = python-config

## Include ROOT
CXXFLAGS  += $(shell root-config --cflags)

CXXDEPFLAGS = -MMD -MP

LD         = g++
LDFLAGS    =

INCLUDES += $(shell $(PYTHONCFG) --includes)

# include "include" directory
INCLUDES += -I$(ZINVISIBLE)/Tools/include

CXXFLAGS += -DDOTENSORFLOWPYBIND

LIBS       = $(shell root-config --glibs) -lASImage
MT2LIB     = -L$(shell $(PYTHONCFG) --prefix)/lib $(shell $(PYTHONCFG) --libs) -L$(TTTDIR) -lTopTagger
LHAPDFLIB  = -L$(LHAPDF_DATA_PATH)/../../lib -lLHAPDF

PROGRAMS = calcEff calcEffPhoton harvestHists makePlots moneyplot scalefactorplot systplot makeTopPlots makeDataMCplots makeDYPlots makePhotonPlots simpleAnalyzer

all: mkobj PyWraps $(PROGRAMS)


mkobj:
	@mkdir -p obj

PyWraps: $(ODIR)/samplesModule.so $(ODIR)/ScaleFactors.so

$(ODIR)/samplesModule.so: $(ODIR)/samplesPyWrap.o $(ODIR)/samplesModulePyWrap.o
	$(CXX) -shared -o $@ $^

$(ODIR)/samplesPyWrap.o: $(RSDIR)/samples.cc $(RSDIR)/samples.h 
	$(CXX) --std=c++11 -c -fPIC -o $@ $<

$(ODIR)/samplesModulePyWrap.o: $(RSDIR)/samplesModule.cc
	$(CXX) --std=c++11 -c -fPIC -o $@ $<

$(ODIR)/ScaleFactors.so: $(ODIR)/ScaleFactorsPyWrap.o
	$(CXX) -shared -o $@ $^

$(ODIR)/ScaleFactorsPyWrap.o: $(SDIR)/ScaleFactors.cc $(SDIR)/ScaleFactors.h
	$(CXX) --std=c++11 -c -fPIC -o $@ $<

#special rule to make root reflex dictionary
$(ODIR)/rootdict.cc : $(SDIR)/rootdict.h
	rootcint -f $@ $^ -inlineInputHeade
	mv $(ODIR)/rootdict_rdict.pcm $(SDIR)


$(ODIR)/%.o : $(SDIR)/%.C
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(RSDIR) -I$(TTIDIR) -I$(TPIDIR) -I$(TPTDIR) $(INCLUDES) -o $@ -c $<

$(ODIR)/%.o : $(SDIR)/%.cc
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(RSDIR) -I$(TTIDIR) -I$(TPIDIR) -I$(TPTDIR) $(INCLUDES) -o $@ -c $<

$(ODIR)/%.o : $(SDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(RSDIR) -I$(TTIDIR) -I$(TPIDIR) -I$(TPTDIR) $(INCLUDES) -o $@ -c $<

$(ODIR)/%.o : $(RSDIR)/%.C
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(RSDIR) -I$(TTIDIR) -I$(TPIDIR) -I$(TPTDIR) $(INCLUDES) -o $@ -c $<

$(ODIR)/%.o : $(RSDIR)/%.cc
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(RSDIR) -I$(TTIDIR) -I$(TPIDIR) -I$(TPTDIR) $(INCLUDES) -o $@ -c $<

$(ODIR)/%.o : $(RSDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(RSDIR) -I$(TTIDIR) -I$(TPIDIR) -I$(TPTDIR) $(INCLUDES) -o $@ -c $<

$(ODIR)/%.o : $(TTSDIR)/%.C
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(RSDIR) -I$(TTIDIR) -I$(TPIDIR) -I$(TPTDIR) $(INCLUDES) -o $@ -c $<

$(ODIR)/%.o : $(TTSDIR)/%.cc
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(RSDIR) -I$(TTIDIR) -I$(TPIDIR) -I$(TPTDIR) $(INCLUDES) -o $@ -c $<

$(ODIR)/%.o : $(TTSDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(RSDIR) -I$(TTIDIR) -I$(TPIDIR) -I$(TPTDIR) $(INCLUDES) -o $@ -c $<

$(ODIR)/%.o : $(TTTSDIR)/%.C
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(RSDIR) -I$(TTIDIR) -I$(TPIDIR) -I$(TPTDIR) -I$(TTTIDIR) $(INCLUDES) -o $@ -c $<

$(ODIR)/%.o : $(TTTSDIR)/%.cc
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(RSDIR) -I$(TTIDIR) -I$(TPIDIR) -I$(TPTDIR) -I$(TTTIDIR) $(INCLUDES) -o $@ -c $<

$(ODIR)/%.o : $(TTTSDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CXXDEPFLAGS) -I$(RSDIR) -I$(TTIDIR) -I$(TPIDIR) -I$(TPTDIR) -I$(TTTIDIR) $(INCLUDES) -o $@ -c $<


calcEff: $(ODIR)/NTupleReader.o $(ODIR)/SATException.o $(ODIR)/samples.o $(ODIR)/calcEff.o $(ODIR)/PDFUncertainty.o $(ODIR)/RegisterFunctions.o $(ODIR)/Systematic.o $(ODIR)/Plotter.o $(ODIR)/MiniTupleMaker.o $(ODIR)/searchBins.o $(ODIR)/BTagCalibrationStandalone.o  $(ODIR)/TaggerUtility.o $(ODIR)/PlotUtility.o $(ODIR)/SusyUtility.o $(ODIR)/baselineDef.o $(ODIR)/customize.o $(ODIR)/ISRCorrector.o 
	$(LD) $^ $(LIBS) $(MT2LIB) $(LHAPDFLIB) -o $@

calcEffPhoton: $(ODIR)/NTupleReader.o $(ODIR)/SATException.o $(ODIR)/samples.o $(ODIR)/calcEffPhoton.o $(ODIR)/PDFUncertainty.o $(ODIR)/RegisterFunctions.o $(ODIR)/Systematic.o $(ODIR)/Plotter.o $(ODIR)/MiniTupleMaker.o $(ODIR)/searchBins.o $(ODIR)/BTagCalibrationStandalone.o  $(ODIR)/TaggerUtility.o $(ODIR)/PlotUtility.o $(ODIR)/SusyUtility.o $(ODIR)/baselineDef.o $(ODIR)/customize.o $(ODIR)/ISRCorrector.o 
	$(LD) $^ $(LIBS) $(MT2LIB) $(LHAPDFLIB) -o $@

harvestHists: $(ODIR)/harvestHists.o
	$(LD) $^ $(LIBS) -o $@

makePlots: $(ODIR)/NTupleReader.o $(ODIR)/SATException.o $(ODIR)/samples.o $(ODIR)/Plotter.o $(ODIR)/MakePlots.o $(ODIR)/MiniTupleMaker.o $(ODIR)/PDFUncertainty.o $(ODIR)/RegisterFunctions.o $(ODIR)/Systematic.o $(ODIR)/searchBins.o $(ODIR)/BTagCalibrationStandalone.o $(ODIR)/TaggerUtility.o $(ODIR)/PlotUtility.o $(ODIR)/SusyUtility.o $(ODIR)/baselineDef.o $(ODIR)/customize.o $(ODIR)/ISRCorrector.o 
	$(LD) $^ $(LIBS) $(MT2LIB) $(LHAPDFLIB) -o $@

makeTopPlots: $(ODIR)/NTupleReader.o $(ODIR)/SATException.o $(ODIR)/samples.o $(ODIR)/Plotter.o $(ODIR)/MakeTopPlots.o $(ODIR)/MiniTupleMaker.o $(ODIR)/PDFUncertainty.o $(ODIR)/RegisterFunctions.o $(ODIR)/Systematic.o $(ODIR)/searchBins.o $(ODIR)/BTagCalibrationStandalone.o $(ODIR)/TaggerUtility.o $(ODIR)/PlotUtility.o $(ODIR)/SusyUtility.o $(ODIR)/baselineDef.o $(ODIR)/customize.o $(ODIR)/ISRCorrector.o 
	$(LD) $^ $(LIBS) $(MT2LIB) $(LHAPDFLIB) -o $@

makeDYPlots: $(ODIR)/NTupleReader.o $(ODIR)/SATException.o $(ODIR)/samples.o $(ODIR)/Plotter.o $(ODIR)/MakeDYPlots.o $(ODIR)/MiniTupleMaker.o $(ODIR)/PDFUncertainty.o $(ODIR)/RegisterFunctions.o $(ODIR)/Systematic.o $(ODIR)/searchBins.o $(ODIR)/BTagCalibrationStandalone.o $(ODIR)/TaggerUtility.o $(ODIR)/PlotUtility.o $(ODIR)/SusyUtility.o $(ODIR)/baselineDef.o $(ODIR)/customize.o $(ODIR)/ISRCorrector.o 
	$(LD) $^ $(LIBS) $(MT2LIB) $(LHAPDFLIB) -o $@

makePhotonPlots: $(ODIR)/NTupleReader.o $(ODIR)/SATException.o $(ODIR)/samples.o $(ODIR)/Plotter.o $(ODIR)/MakePhotonPlots.o $(ODIR)/MiniTupleMaker.o $(ODIR)/PDFUncertainty.o $(ODIR)/RegisterFunctions.o $(ODIR)/Systematic.o $(ODIR)/searchBins.o $(ODIR)/BTagCalibrationStandalone.o $(ODIR)/TaggerUtility.o $(ODIR)/PlotUtility.o $(ODIR)/SusyUtility.o $(ODIR)/baselineDef.o $(ODIR)/customize.o $(ODIR)/ISRCorrector.o 
	$(LD) $^ $(LIBS) $(MT2LIB) $(LHAPDFLIB) -o $@

miniMakePlots: $(ODIR)/NTupleReader.o $(ODIR)/SATException.o $(ODIR)/miniTupleSamples.o $(ODIR)/Plotter.o $(ODIR)/MinitupleMakePlots.o $(ODIR)/MiniTupleMaker.o $(ODIR)/PDFUncertainty.o $(ODIR)/RegisterFunctions.o $(ODIR)/Systematic.o $(ODIR)/searchBins.o $(ODIR)/BTagCalibrationStandalone.o $(ODIR)/TaggerUtility.o $(ODIR)/PlotUtility.o $(ODIR)/SusyUtility.o $(ODIR)/baselineDef.o $(ODIR)/customize.o $(ODIR)/ISRCorrector.o 
	$(LD) $^ $(LIBS) $(MT2LIB) $(LHAPDFLIB) -o $@

makeDataMCplots: $(ODIR)/NTupleReader.o $(ODIR)/SATException.o $(ODIR)/samples.o $(ODIR)/Plotter.o $(ODIR)/MakePlots.o $(ODIR)/MiniTupleMaker.o $(ODIR)/PDFUncertainty.o $(ODIR)/RegisterFunctions.o $(ODIR)/Systematic.o $(ODIR)/searchBins.o $(ODIR)/BTagCalibrationStandalone.o $(ODIR)/TaggerUtility.o $(ODIR)/PlotUtility.o $(ODIR)/baselineDef.o $(ODIR)/SusyUtility.o $(ODIR)/customize.o $(ODIR)/ISRCorrector.o 
	$(LD) $^ $(LIBS) $(MT2LIB) $(LHAPDFLIB) -o $@

systematics: $(ODIR)/NTupleReader.o $(ODIR)/SATException.o $(ODIR)/miniTupleSamples.o $(ODIR)/Systematic.o  $(ODIR)/Plotter.o $(ODIR)/systematics.o $(ODIR)/MiniTupleMaker.o $(ODIR)/PDFUncertainty.o $(ODIR)/RegisterFunctions.o $(ODIR)/searchBins.o $(ODIR)/BTagCalibrationStandalone.o $(ODIR)/TaggerUtility.o $(ODIR)/PlotUtility.o $(ODIR)/SusyUtility.o $(ODIR)/baselineDef.o $(ODIR)/customize.o $(ODIR)/ISRCorrector.o 
	$(LD) $^ $(LIBS) $(MT2LIB) $(LHAPDFLIB) -o $@

nJetWgtSyst: $(ODIR)/NTupleReader.o $(ODIR)/SATException.o $(ODIR)/nJetWgtSyst.o $(ODIR)/miniTupleSamples.o $(ODIR)/searchBins.o $(ODIR)/baselineDef.o $(ODIR)/customize.o
	$(LD) $^ $(LIBS) $(MT2LIB) $(LHAPDFLIB) -o $@

moneyplot: $(ODIR)/moneyplot.o $(ODIR)/SATException.o $(ODIR)/searchBins.o 
	$(LD) $^ $(LIBS) -o $@

scalefactorplot: $(ODIR)/scalefactorplot.o
	$(LD) $^ $(LIBS) -o $@

systplot: $(ODIR)/systplot.o $(ODIR)/SATException.o $(ODIR)/searchBins.o 
	$(LD) $^ $(LIBS) -o $@

beffCalc: $(ODIR)/bTagEfficiencyCalc.o $(ODIR)/NTupleReader.o $(ODIR)/samples.o $(ODIR)/baselineDef.o $(ODIR)/customize.o $(ODIR)/SATException.o
	$(LD) $^ $(LIBS)  $(MT2LIB) -o $@

makeSignalHistograms: $(ODIR)/makeSignalHistograms.o $(ODIR)/NTupleReader.o $(ODIR)/SATException.o $(ODIR)/samples.o  $(ODIR)/baselineDef.o $(ODIR)/customize.o
	$(LD) $^ $(LIBS) $(MT2LIB) -o $@

makeTaggerHistograms: $(ODIR)/makeTaggerHistograms.o $(ODIR)/NTupleReader.o $(ODIR)/SATException.o $(ODIR)/samples.o  $(ODIR)/baselineDef.o $(ODIR)/customize.o $(ODIR)/ISRCorrector.o  $(ODIR)/BTagCalibrationStandalone.o $(ODIR)/TaggerUtility.o $(ODIR)/PlotUtility.o $(ODIR)/SusyUtility.o
	$(LD) $^ $(LIBS) $(MT2LIB) -o $@

simpleAnalyzer: $(ODIR)/simpleAnalyzer.o $(ODIR)/NTupleReader.o $(ODIR)/SATException.o $(ODIR)/samples.o  $(ODIR)/baselineDef.o $(ODIR)/customize.o $(ODIR)/ISRCorrector.o  $(ODIR)/TTbarCorrector.o $(ODIR)/BTagCalibrationStandalone.o $(ODIR)/TaggerUtility.o $(ODIR)/PlotUtility.o $(ODIR)/SusyUtility.o $(ODIR)/rootdict.o
	$(LD) $^ $(LIBS) $(MT2LIB) -o $@

clean:
	rm -f $(ODIR)/*.o $(ODIR)/*.so $(ODIR)/*.d $(PROGRAMS) core 

-include $(ODIR)/*.d
