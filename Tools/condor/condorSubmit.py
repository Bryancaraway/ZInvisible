#!/cvmfs/cms.cern.ch/slc6_amd64_gcc491/cms/cmssw/CMSSW_7_4_8/external/slc6_amd64_gcc491/bin/python
####!${SRT_CMSSW_RELEASE_BASE_SCRAMRTDEL}/external/${SCRAM_ARCH}/bin/python


import optparse 
import subprocess
import datetime
import subprocess
import sys
import os
from os import system, environ
import json

sys.path = [environ["CMSSW_BASE"] + "/src/SusyAnaTools/Tools/condor/",] + sys.path
from samples import SampleCollection



# main()
def main():
    
    parser = optparse.OptionParser("usage: %prog [options]\n")
    
    parser.add_option ('-n',  dest='numfile',               type='int',          default = 5,     help="number of files per job (default: 5)")
    parser.add_option ('-d',  dest='datasets',              type='string',       default = '',    help="List of datasets 'ZJetsToNuNu_2016,GJets_2016,DYJetsToLL_2016'")
    parser.add_option ('-l',  dest='dataCollections',       action='store_true', default = False, help="List all datacollections")
    parser.add_option ('-L',  dest='dataCollectionslong',   action='store_true', default = False, help="List all datacollections and sub collections")
    parser.add_option ('-r',  dest='refLumi',               type='string',       default = None,  help="Data collection to define lumi (uses default lumi if no reference data collection is defined)")
    parser.add_option ('-y',  dest='year',                  type='string',       default = None,  help="Year data or MC to analyze.")
    parser.add_option ('-c',  dest='noSubmit',              action='store_true', default = False, help="Do not submit jobs.  Only create condor_submit.txt.")
    parser.add_option ('-e',  dest='goMakeEff',             action='store_true', default = False, help="Run calcEff instead of makePlots.")
    parser.add_option ('-p',  dest='goMakeEffPhoton',       action='store_true', default = False, help="Run calcEffPhoton instead of makePlots.")
    parser.add_option ('-b',  dest='goMakeBeff',            action='store_true', default = False, help="Run beffCalc instead of makePlots.")
    parser.add_option ('-s',  dest='goMakeSigEff',          action='store_true', default = False, help="Run makeSignalHistograms instead of makePlots.")
    parser.add_option ('-t',  dest='goMakeTopPlots',        action='store_true', default = False, help="Run makeTopPlots instead of makePlots.")
    parser.add_option ('-m',  dest='goTTPlots',             action='store_true', default = False, help="Run TTPlots instead of makePlots.")
    
    options, args = parser.parse_args()
    year = options.year
    
    datasets = []
    if options.datasets:
        datasets = options.datasets.split(',')
    else:
        print "Please use -d to specify one or more datasets (separate with commas)."
        exit(1)

    # sample config files
    years = ["2016", "2017", "2018", "2018_AB", "2018_CD"]
    if year not in years:
        print "Please use -y to enter year (2016, 2017, 2018, 2018_AB, or 2018_CD)."
        exit(1)

    yearWithoutPeriod = year[0:4] 
    sampleSetsFile = "sampleSets_PostProcessed_" + yearWithoutPeriod + ".cfg"
    sampleCollectionsFile = "sampleCollections_" + yearWithoutPeriod + ".cfg"
    
    # TopTagger.cfg
    mvaFileName = ""
    #with file(environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/TopTagger.cfg") as meowttcfgFile:
    with file(environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/TopTagger_Tensorflow_"+yearWithoutPeriod+".cfg") as meowttcfgFile:
        for line in meowttcfgFile:
            line = line.split("#")[0]
            if "modelFile" in line:
                mvaFileName = line.split("=")[1].strip().strip("\"")
                break
    
    
    print "mvaFileName = {0}".format(mvaFileName)
    
    
    #here I hack in the tarball for GMP, this needs to be generalized to the other options 
    
    filestoTransferGMP = [environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/makePlots", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/CSVv2_Moriond17_B_H.csv", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/2016_trigger_eff.root", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/2017_trigger_eff.root", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/2018_trigger_eff.root", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/lepEffHists.root", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/effhists_GJets.root", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/njetWgtHists.root", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/dataMCweights.root", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/dataMCreweight.root", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/dataMCreweight_allJets.root", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/puppiCorr.root",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/allINone_bTagEff.root", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/ISRWeights.root", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/allINone_ISRJets.root", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/PileupHistograms_0121_69p2mb_pm4p6.root",
                          environ["CMSSW_BASE"] + "/src/TopTagger/TopTagger/test/libTopTagger.so",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/TopTagSF_AltTWP.root",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/TopTagger.cfg",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleSets_PostProcessed_2016.cfg",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleSets_PostProcessed_2017.cfg",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleSets_PostProcessed_2018.cfg",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleCollections_2016.cfg",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleCollections_2017.cfg",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleCollections_2018.cfg"
                         ]
    
    if mvaFileName:                 filestoTransferGMP += [environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/%(trainingFile)s"%{"trainingFile":mvaFileName}]
    
    filestoTransferGMEP = [environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/calcEffPhoton", 
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/CSVv2_Moriond17_B_H.csv", 
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/2016_trigger_eff.root", 
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/2017_trigger_eff.root", 
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/2018_trigger_eff.root", 
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/puppiCorr.root",
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/allINone_bTagEff.root", 
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/ISRWeights.root", 
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/allINone_ISRJets.root", 
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/PileupHistograms_0121_69p2mb_pm4p6.root",
                           environ["CMSSW_BASE"] + "/src/TopTagger/TopTagger/test/libTopTagger.so",
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/TopTagger.cfg",
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleSets_PostProcessed_2016.cfg",
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleSets_PostProcessed_2017.cfg",
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleSets_PostProcessed_2018.cfg",
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleCollections_2016.cfg",
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleCollections_2017.cfg",
                           environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleCollections_2018.cfg"
                          ]
    
    if mvaFileName:                 filestoTransferGMEP += [environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/%(trainingFile)s"%{"trainingFile":mvaFileName}]
    if sampleSetsFile:              filestoTransferGMEP += [environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/" + sampleSetsFile]
    if sampleCollectionsFile:       filestoTransferGMEP += [environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/" + sampleCollectionsFile]
    
    #go make plots!
    # old version for submitting from condor directory
    #submitFileGMP = """universe = vanilla
    #Executable = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goMakePlots.sh
    #Requirements = OpSys == "LINUX"&& (Arch != "DUMMY" )
    #Should_Transfer_Files = YES
    #WhenToTransferOutput = ON_EXIT
    #Transfer_Input_Files = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goMakePlots.sh,$ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/gmp.tar.gz,$ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/$ENV(CMSSW_VERSION).tar.gz
    #Output = logs/makePlots_$(Process).stdout
    #Error = logs/makePlots_$(Process).stderr
    #Log = logs/makePlots_$(Process).log
    #notify_user = ${LOGNAME}@FNAL.GOV
    #x509userproxy = $ENV(X509_USER_PROXY)
    #+maxWallTime = 2880
    #
    #"""
    # new version for submitting from specific directory for submission (use relative instead of full paths for tar files)
    # note that tabs and spaces will appear in condor_submit.txt if you have them here
    submitFileGMP = """universe = vanilla
Executable = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goMakePlots.sh
Requirements = OpSys == "LINUX"&& (Arch != "DUMMY" )
Should_Transfer_Files = YES
WhenToTransferOutput = ON_EXIT
Transfer_Input_Files = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goMakePlots.sh,gmp.tar.gz,$ENV(CMSSW_VERSION).tar.gz
Output = logs/makePlots_$(Process).stdout
Error = logs/makePlots_$(Process).stderr
Log = logs/makePlots_$(Process).log
notify_user = ${LOGNAME}@FNAL.GOV
x509userproxy = $ENV(X509_USER_PROXY)
+maxWallTime = 2880

"""
    
    #Here is the configuration for the Data/MC validation of the TopTagger 
    filestoTransferTT  = [environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/simpleAnalyzer",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/puppiCorr.root",
                          environ["CMSSW_BASE"] + "/src/SusyAnaTools/Tools/data/allINone_bTagEff.root", 
                          environ["CMSSW_BASE"] + "/src/SusyAnaTools/Tools/ISR_Root_Files/ISRWeights.root", 
                          environ["CMSSW_BASE"] + "/src/SusyAnaTools/Tools/ISR_Root_Files/allINone_ISRJets.root", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/CSVv2_Moriond17_B_H.csv", 
                          environ["CMSSW_BASE"] + "/src/SusyAnaTools/Tools/data/PileupHistograms_0121_69p2mb_pm4p6.root", 
                          environ["CMSSW_BASE"] + "/src/TopTagger/TopTagger/test/libTopTagger.so",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/TopTagger.cfg",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleSets_PostProcessed_2016.cfg",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleSets_PostProcessed_2017.cfg",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleSets_PostProcessed_2018.cfg",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleCollections_2016.cfg",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleCollections_2017.cfg",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/sampleCollections_2018.cfg"
                         ]
    
    if mvaFileName: filestoTransferTT += [environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/%(trainingFile)s"%{"trainingFile":mvaFileName}]
    
    #go make TTopTagger plots!
    submitFileTT = """universe = vanilla
    Executable = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goTTplots.sh
    Requirements = OpSys == "LINUX"&& (Arch != "DUMMY" )
    Should_Transfer_Files = YES
    WhenToTransferOutput = ON_EXIT
    Transfer_Input_Files = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goTTplots.sh,$ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/TT.tar.gz,$ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/$ENV(CMSSW_VERSION).tar.gz
    Output = logs/TT_$(Process).stdout
    Error = logs/TT_$(Process).stderr
    Log = logs/TT_$(Process).log
    notify_user = ${LOGNAME}@FNAL.GOV
    x509userproxy = $ENV(X509_USER_PROXY)
    +maxWallTime = 2880
    
    """
    
    #go make top plots!
    filestoTransferGTP = [environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/makeTopPlots",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/puppiCorr.root",
                          environ["CMSSW_BASE"] + "/src/SusyAnaTools/Tools/data/allINone_bTagEff.root", 
                          environ["CMSSW_BASE"] + "/src/SusyAnaTools/Tools/ISR_Root_Files/ISRWeights.root", 
                          environ["CMSSW_BASE"] + "/src/SusyAnaTools/Tools/ISR_Root_Files/allINone_ISRJets.root", 
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/CSVv2_Moriond17_B_H.csv", 
                          environ["CMSSW_BASE"] + "/src/SusyAnaTools/Tools/data/PileupHistograms_0121_69p2mb_pm4p6.root", 
                          environ["CMSSW_BASE"] + "/src/TopTagger/TopTagger/test/libTopTagger.so",
                          environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/TopTagger.cfg"
                          ]
    
    if mvaFileName: filestoTransferTT += [environ["CMSSW_BASE"] + "/src/ZInvisible/Tools/%(trainingFile)s"%{"trainingFile":mvaFileName}]
    
    #print filestoTransferGTP
    
    submitFileGTP = """universe = vanilla
    Executable = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goMakeTopPlots.sh
    Should_Transfer_Files = YES
    WhenToTransferOutput = ON_EXIT
    Transfer_Input_Files = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goMakePlots.sh,$ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/gtp.tar.gz,$ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/$ENV(CMSSW_VERSION).tar.gz 
    Output = logs/makePlots_$(Process).stdout
    Error = logs/makePlots_$(Process).stderr
    Log = logs/makePlots_$(Process).log
    notify_user = ${LOGNAME}@FNAL.GOV
    x509userproxy = $ENV(X509_USER_PROXY)
    +maxWallTime = 2880
    
    """
    
    #go make lepton efficiency
    submitFileGME = """universe = vanilla
    Executable = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goMakeEff.sh
    Requirements = OpSys == "LINUX"&& (Arch != "DUMMY" )
    Should_Transfer_Files = YES
    WhenToTransferOutput = ON_EXIT
    Transfer_Input_Files = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/calcEff, $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goMakeEff.sh, $ENV(CMSSW_BASE)/src/ZInvisible/Tools/zRes.root
    notify_user = ${LOGNAME}@FNAL.GOV
    x509userproxy = $ENV(X509_USER_PROXY)
    
    """
    
    #go make photon efficiency
    submitFileGMEP = """universe = vanilla
    Executable = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goMakeEffPhoton.sh
    Requirements = OpSys == "LINUX"&& (Arch != "DUMMY" )
    Should_Transfer_Files = YES
    WhenToTransferOutput = ON_EXIT
    Transfer_Input_Files = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goMakeEffPhoton.sh,$ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/gmep.tar.gz,$ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/$ENV(CMSSW_VERSION).tar.gz
    Output = logs/calcEffPhoton_$(Process).stdout
    Error = logs/calcEffPhoton_$(Process).stderr
    Log = logs/calcEffPhoton_$(Process).log
    notify_user = ${LOGNAME}@FNAL.GOV
    x509userproxy = $ENV(X509_USER_PROXY)
    +maxWallTime = 2880
    
    """
    
    #go B Efficiency calc
    submitFileGBE = """universe = vanilla
    Executable = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goMakeBeff.sh
    Requirements = OpSys == "LINUX"&& (Arch != "DUMMY" )
    Should_Transfer_Files = YES
    WhenToTransferOutput = ON_EXIT
    Transfer_Input_Files = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/beffCalc, $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goMakeBeff.sh
    Output = logs/makePlots_$(Process).stdout
    Error = logs/makePlots_$(Process).stderr
    Log = logs/makePlots_$(Process).log
    notify_user = ${LOGNAME}@FNAL.GOV
    x509userproxy = $ENV(X509_USER_PROXY)
    
    """
    
    #go Signal Efficiency
    submitFileGSE = """universe = vanilla
    Executable = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goMakeSigEff.sh
    Requirements = OpSys == "LINUX"&& (Arch != "DUMMY" )
    Should_Transfer_Files = YES
    WhenToTransferOutput = ON_EXIT
    Transfer_Input_Files = $ENV(CMSSW_BASE)/src/ZInvisible/Tools/makeSignalHistograms, $ENV(CMSSW_BASE)/src/ZInvisible/Tools/condor/goMakeSigEff.sh
    Output = logs/makePlots_$(Process).stdout
    Error = logs/makePlots_$(Process).stderr
    Log = logs/makePlots_$(Process).log
    notify_user = ${LOGNAME}@FNAL.GOV
    x509userproxy = $ENV(X509_USER_PROXY)
    
    """
    
    submitFile = ""
    exeName = ""
    
    # load samples
    sc = SampleCollection("../" + sampleSetsFile, "../" + sampleCollectionsFile)
    
    # make directory for condor submission
    now = datetime.datetime.now()
    dirName = "DataMC_%s_submission_%s" % (year, now.strftime("%Y-%m-%d_%H-%M-%S"))
    os.system("mkdir %s"%dirName)
    os.chdir(dirName)
    
    def makeExeAndFriendsTarrball(filestoTransfer, fname):
        if not options.dataCollections and not options.dataCollectionslong:
            #WORLDSWORSESOLUTIONTOAPROBLEM
            system("mkdir -p WORLDSWORSESOLUTIONTOAPROBLEM")
            for fn in filestoTransfer:
                #print "fn = {0}".format(fn)
                system("cd WORLDSWORSESOLUTIONTOAPROBLEM; ln -s %s" % fn)
            
            print "Create tarball {0}.tag.gz".format(fname)
            tarallinputs = "tar czvf %s.tar.gz WORLDSWORSESOLUTIONTOAPROBLEM --dereference" % fname
            print tarallinputs
            system(tarallinputs)
            system("rm -r WORLDSWORSESOLUTIONTOAPROBLEM")
    
    
    if not options.dataCollections and not options.dataCollectionslong:
        print "Create tarball ${CMSSW_VERSION}.tar.gz"
        system("tar --exclude-caches-all --exclude-vcs -zcf ${CMSSW_VERSION}.tar.gz -C ${CMSSW_BASE}/.. ${CMSSW_VERSION} --exclude=src --exclude=tmp")
    
    # makeExeAndFriendsTarrball() is necessary now to apply WORLDSWORSESOLUTIONTOAPROBLEM 
    
    if options.goMakeEff:
        exeName = "calcEff"
        submitFile = submitFileGME
    elif options.goMakeEffPhoton:
        exeName = "calcEffPhoton"
        submitFile = submitFileGMEP
        makeExeAndFriendsTarrball(filestoTransferGMEP, "gmep")
    elif options.goMakeBeff:
        exeName = "beffCalc"
        submitFile = submitFileGBE
    elif options.goMakeSigEff:
        exeName = "makeSignalHistograms"
        submitFile = submitFileGSE
    elif options.goMakeTopPlots:
        exeName = "makeTopPlots"
        submitFile = submitFileGTP
        makeExeAndFriendsTarrball(filestoTransferGTP, "gtp")
    elif options.goTTPlots:
        exeName = "simpleAnalyzer"
        submitFile = submitFileTT
        makeExeAndFriendsTarrball(filestoTransferTT, "TT")
    else:
        exeName = "makePlots"
        submitFile = submitFileGMP
        makeExeAndFriendsTarrball(filestoTransferGMP, "gmp")
    
    nFilesPerJob = options.numfile
    
    fileParts = [submitFile]
    
    if options.dataCollections or options.dataCollectionslong:
        scl = sc.sampleCollectionList()
        for sampleCollection in scl:
            sl = sc.sampleList(sampleCollection)
            print sampleCollection
            if options.dataCollectionslong:
                sys.stdout.write("\t")
                for sample in sl:
                    sys.stdout.write("%s  "%sample[1])
                print ""
                print ""
        exit(1)
    
    
    lumis = sc.sampleCollectionLumiList()
    lumi = sc.getFixedLumi()
    if options.refLumi != None:
        lumi = lumis[options.refLumi]
        print "Normalizing to %s pb-1" % (lumi)
   
    total_files = 0
    total_events = 0
    jobMap = {}
    for ds in datasets:
        sample_files = 0
        sample_events = 0
        ds = ds.strip()
        #print ds
        print "# --- {0} --- #".format(ds)
        if "Data" in ds:
            print "Lumi: {0}".format(sc.getFixedLumi())
        
        # s: file, n:name, e:nEvts
        for s, n, e in sc.sampleList(ds):
            # debugging
            #print "\t{0} {1} {2}".format(s, n, e)
            n_files = sum(1 for line in open(s))
            sample_files += n_files
            sample_events += e
            total_files += n_files
            total_events += e
            jobMap[n] =  n_files
            print "\t{0}: n_files={1}, n_events={2}".format(n, n_files, e)
            try:
                f = open(s)
            except IOError:
                fShort = s.split("/")[-1]
                if(os.path.isfile(fShort)):
                    os.remove(fShort)
                system("xrdcp root://cmseos.fnal.gov/$(echo %s | sed 's|/eos/uscms||') ."%s)
                print "fShort = {0}".format(fShort)
                f = open(fShort)
            if not f == None:
                count = 0
                for l in f:
                    if '.root' in l and not 'failed' in l:
                        count = count + 1
                for startFileNum in xrange(0, count, nFilesPerJob):
                    fileParts.append("Arguments = %s $ENV(CMSSW_VERSION) %i %i %f %s %s\n"%(n, nFilesPerJob, startFileNum, lumi, s, year))
                    fileParts.append("Output = logs/$(Cluster)_$(Process)_%s_%s_%i.stdout\n"%(exeName, n, startFileNum))
                    fileParts.append("Error = logs/$(Cluster)_$(Process)_%s_%s_%i.stderr\n"%(exeName, n, startFileNum))
                    fileParts.append("Log = logs/$(Cluster)_$(Process)_%s_%s_%i.log\n"%(exeName, n, startFileNum))
                    fileParts.append("Queue\n\n")
    
                f.close()
        # number of files and events in sample
        print "\t{0}: sample_files={1}, sample_events={2}".format(ds, sample_files, sample_events)
     
    # total number of files and events
    print "# --- Totals --- #"
    print "total_files={0}, total_events={1}".format(total_files, total_events)
    
    with open("nJobs.json", "w") as outfile:
        json.dump(jobMap, outfile, sort_keys=True, indent=4)
    
    fout = open("condor_submit.txt", "w")
    fout.write(''.join(fileParts))
    fout.close()
    
    if not options.noSubmit: 
        system('mkdir -p logs')
        system("echo 'condor_submit condor_submit.txt'")
        system('condor_submit condor_submit.txt')

    print "Condor submission directory: {0}".format(dirName)

if __name__ == "__main__":
    main()

