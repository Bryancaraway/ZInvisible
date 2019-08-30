########################
### Sample Analyzer ####
### Written by      ####
### Bryan Caraway   ####
########################
### Usage:          
### python validateEventSecection.py -d (TTZToLLNuNu_2017) -y (year) -n (nEvents)
######################## 

import optparse
import ROOT
from ROOT import TLorentzVector

import uproot
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path = ["CMSSW_10_2_9" + "/src/SusyAnaTools/Tools/condor/",] + sys.path
outputDir = '/eos/uscms/store/user/bcaraway/AnaSamples/'
#from samples import SampleCollection

def main():

    parser = optparse.OptionParser("usage: %prog [options]\n")
    
    parser.add_option ('-n', dest='nEvents',       type='int',          default = '-1',  help="Number of total events per sample")
    parser.add_option ('-d', dest='datasets',      type='string',       default = '',    help="List of datasets 'ZJetsToNuNu_2016,GJets_2016,DYJetsToLL_2016'")             
    parser.add_option ('-y', dest='year',          type='string',       default = None,  help="Year data or MC to analyze.")
    parser.add_option ('-f', dest='inputFile',     type='string',       default = None,  help="Use an optional file")
    parser.add_option ('-p', dest='pre',           action='store_true', default = False, help="Analyze PreProcessed version of sampleSets.")
                       
    options, args = parser.parse_args()
    year = options.year 
                       
    datasets = []
    if options.datasets:
        datasets = options.datasets.split(',')
    else:
        print("Please use -d to specify one or more datasets (separate with commas).")
        exit(1)
    
    lumiYearDict = {"2016":35.9E3, "2017":41.9E3, "2018":58.9E3, "2018_AB":20.8E3, "2018_CD":38.1E3}
    if year not in lumiYearDict:
        print("Please use -y to enter year (2016, 2017, 2018, 2018_AB, or 2018_CD).")
        exit(1)

    yearWithoutPeriod = year[0:4] 
    process_type =  "Pre" if options.pre else "Post"
    sampleSetsFile = "sampleSets_"+process_type+"Processed_" + yearWithoutPeriod + ".cfg"
    sampleCollectionsFile = "sampleCollections_" + yearWithoutPeriod + ".cfg"

    # Return datasets, with MC weights
    sampleDict = getDataWeights(datasets,sampleSetsFile,sampleCollectionsFile,lumiYearDict[year])
    doAnalyzer(sampleDict, options.nEvents, options.inputFile)

    
def getDataWeights(datasets,sampleSetsFile,sampleCollectionsFile,lumi):
    # dataset = [TTZ_2017, TTZToLLNuNu_2017, TTZToQQ_2017]
    # TTZToQQ_2017, /eos/uscms/store/user/lpcsusyhad/Stop_production/Fall17_94X_v2_NanAOD_MC/PostProcessed_15Jan2019_v2p7, TTZToQQ_2017.txt, Events, 0.5297, 553143, 196857, 1.0
    sampleDict = [] # "files" , "weight"
    with open(sampleCollectionsFile) as colFile:
        for line in colFile:
            if line.split(",")[0].strip() in datasets:
                for sample in line.split(",")[1:]:
                    datasets.append(sample.strip())
                datasets.remove(line.split(",")[0].strip())
    with open(sampleSetsFile) as setsFile:
        for line in setsFile:
            csv = line.split(",")
            if csv[0] in datasets:
                sampleDict.append(
                    {"file":csv[1].strip()+"/"+csv[2].strip(),
                     "weight":lumi*float(csv[7].strip())*float(csv[4].strip())/(float(csv[5].strip())-float(csv[6].strip()))})# lumi*k_factor*x-section/(positive_weights-negative_weights)
    return sampleDict

def doAnalyzer(sampleDict , nmax, inputFile):
    trees = []
    #weightNames    = ['puWeight', 'BTagWeight', 'ISRWeight', 'PrefireWeight']
    #eventSelection = ["nBottoms_drLeptonCleaned>1", "nResolvedTops_drLeptonCleaned>0"]
    diKnownLNuZMass   = {'11': [], '12': [], '13': [], '14': [], '15': [], '16': []}
    diUnknownLNuZMass = {'11': [], '12': [], '13': [], '14': [], '15': [], '16': []} 

    for sample in sampleDict:
        n_ = 0 # count the events 
        with open(sample["file"]) as listfile:
            for line in listfile: # open individual root files in sample.txt
                if ((nmax != -1) and (n_ >= nmax)):
                    break
                fname = line.split()[0]
                isZToLL = None ## new branch in tree
                if (inputFile): 
                    fname = inputFile
                with uproot.open(fname) as f:
                    print(fname)
                    t = f.get('Events')
                    checkEvents(t)

                    IDs     = t.array('GenPart_pdgId')
                    masses  = t.array('GenPart_mass')
                    pts     = t.array('GenPart_pt')
                    phis    = t.array('GenPart_phi')
                    etas    = t.array('GenPart_eta')
                    mothers = t.array('GenPart_genPartIdxMother')
                    status  = t.array('GenPart_status')
                    flag    = t.array('GenPart_statusFlags')

                    massDict = {'11': .511E-3, '12': 0, '13': 105.66E-3, '14': 0, '15': 1776.86E-3, '16': 0, '23': 91.1876}
                    isllnunu = ((abs(IDs) == 11) | (abs(IDs) == 13) | (abs(IDs) == 15) | (abs(IDs) == 12) | (abs(IDs) == 14) | (abs(IDs) == 16))
                    isFromZ  = ((IDs[mothers] == 23) & (isllnunu))
                    noZ      = (isFromZ.sum() == False)

                    fromZDict         = {"nZEvents" : len(isFromZ.sum()), "nZTwoDaughters": 0, "nZMassDaughters": 0, "TTLep": 0, "TTHad": 0}
                    
                    # find if T decays semileptonically or all hadronically
                    
                    IDcut = (((IDs[mothers] == 6) | (IDs[mothers] == 24) |
                              (IDs[mothers] == -6) | (IDs[mothers] == -24)) &
                             ((IDs != 24) & (IDs != -24) &
                             (IDs != 5) & (IDs != -5)))                    
                    variety = (abs(IDs[IDcut]) < 10).sum()
                    print (len(variety[variety == 4]))
                    
                    for daughters, mother in zip(IDs, mothers):
                        if(len(daughters)>0):
                            foundWPlus  = False
                            foundWMinus = False
                            for i in range(0,len(daughters)):
                                for j in range(0,len(daughters)):
                                    if ((daughters[i] != daughters[j]) and (mother[i] == mother[j]) and ((abs(daughters[i]) < 5) and (abs(daughters[j]) < 5))) :
                                        if (((daughters[mother[i]] == 24) or (daughters[mother[i]] == 6)) and (foundWPlus == False)) :
                                            foundWPlus = True
                                        if (((daughters[mother[i]] == -24) or (daughters[mother[i]] == -6)) and (foundWMinus == False)) :
                                            foundWMinus = True
                            if (foundWMinus and foundWPlus) :
                                
                                fromZDict["TTHad"] += 1
                            else :
                                fromZDict["TTLep"] += 1

                    print(fromZDict["TTHad"], fromZDict["TTLep"])
                    ## Find definite daughters of Z


                    for pair, mother, mass, pt, phi, eta in zip(IDs[isFromZ], mothers[isFromZ], masses[isFromZ], pts[isFromZ], phis[isFromZ], etas[isFromZ]):
                        if(len(pair) > 0):
                            if (str(abs(pair[0])) not in fromZDict.keys()):
                                fromZDict[str(abs(pair[0]))] = 0
                            fromZDict[str(abs(pair[0]))] += 1
                            fromZDict["nZTwoDaughters"] += 1
                            # Find Di Lep Mass
                            part1 , part2, partSum = TLorentzVector(), TLorentzVector(), TLorentzVector()
                            part1.SetPtEtaPhiM(pt[0],eta[0],phi[0],massDict[str(abs(pair[0]))])
                            part2.SetPtEtaPhiM(pt[1],eta[1],phi[1],massDict[str(abs(pair[1]))])
                            partSum = part1 + part2
                            diKnownLNuZMass[str(abs(pair[0]))].append(partSum.M())


                    ## Find best matched daughter of Z

                    otherMothersIndex = ['-1','0']
                    otherMothers     = ['-1','1','-2','2','21']
                    noZMask = []
                    counter = 0
                    print(len(IDs))
                    for ID, mother, mass, pt, phi, eta in zip(IDs[noZ], mothers[noZ], masses[noZ], pts[noZ], phis[noZ], etas[noZ]):
                        counter += 1
                        if (counter % 10000 == 0) : print(counter)
                        if ((len(ID) > 0)):
                            bestZMass   = 9999
                            bestZMassID = -1
                            for i  in range(0,len(ID)):
                                for j in range(i,len(ID)):
                                    if ((str(abs(ID[i])) in massDict) and (str(ID[i]) != '23')) :
                                        if ((ID[i] == (-1*ID[j])) and (mother[i] == mother[j]) and ((str(mother[i]) in otherMothersIndex) or (str(ID[mother[i]]) in otherMothers))) :
                                            # Find Closest to Z Di Lep Mass
                                            part1 , part2, partSum = TLorentzVector(), TLorentzVector(), TLorentzVector()
                                            part1.SetPtEtaPhiM(pt[i],eta[i],phi[i],massDict[str(abs(ID[i]))])
                                            part2.SetPtEtaPhiM(pt[j],eta[j],phi[j],massDict[str(abs(ID[j]))])
                                            partSum = part1 + part2
                                            if ((abs(partSum.M() - massDict['23']) < (abs(bestZMass - massDict['23'])))) :
                                                bestZMass = partSum.M()
                                                bestZMassID = str(abs(ID[i]))
                                    
                            if ((bestZMass != 9999) and (bestZMassID != -1)) :
                                fromZDict["nZMassDaughters"] += 1
                                fromZDict[bestZMassID] += 1
                                diUnknownLNuZMass[bestZMassID].append(bestZMass)
                                noZMask.append(True)
                            else : 
                                noZMask.append(False)
                                print ID
                    

                    for key in sorted(fromZDict.keys()):
                        print("%s: %s" % (key, fromZDict[key])),
                    n_ += len(isFromZ.sum())
                    print("")
                    x = IDs[noZ]
                    ### PRINT OUT EVENTS THAT WERENT COUNTED ###
                if (inputFile): break    
    
    # PLOT HERE
    #makeHistogram(diKnownLNuZMass,   "From Known Daughters")
    #del diKnownLNuZMass
    #makeHistogram(diUnknownLNuZMass, "From Best Matched Daughters")

def makeHistogram(mass_array,title):
    fig, particle_plots = plt.subplots( 2, 3, figsize=(16, 10))
    fig.suptitle(title)
    for ax, mass_key in zip(particle_plots.flat, mass_array.keys()):
        ax.hist(mass_array[mass_key],
                 182,
                 range = (0,182),
                 histtype = 'step',
                 label = mass_key)
        ax.plot([91.1876,91.1876], ax.get_ybound(), 'r', linewidth=2.0)
    
        ax.set_xlabel('Di{'+mass_key+'} ZMass (GeV)')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)

    plt.show()
    plt.clf()
    plt.close()

def checkEvents(t):
    
    
    event   = t.array('event')
    lumi    = t.array('luminosityBlock')

    IDs     = t.array('GenPart_pdgId')
    masses  = t.array('GenPart_mass')
    pts     = t.array('GenPart_pt')
    phis    = t.array('GenPart_phi')
    etas    = t.array('GenPart_eta')
    mothers = t.array('GenPart_genPartIdxMother')
    status  = t.array('GenPart_status')
    flag    = t.array('GenPart_statusFlags')
    
    checkEvents = {'Event': [],'Lumi': []}
    csvFile = "ttbarlep.csv"
    with open(csvFile, 'r') as f:
        for line in f:
            csvValues = line.split(",") #[1]: Lumi,[2]: Event, [3]: Run, [4]: ZtoLL, [5]: TTallHad
            checkEvents['Event'].append(int(csvValues[2]))
            checkEvents['Lumi'].append( int(csvValues[1]))
    
    print(checkEvents['Event'])

    for i,j in zip(checkEvents['Event'], checkEvents['Lumi']):
        for k, eve in enumerate(event):
            if (i == eve) and (j == lumi[k]):
                print(eve)
                print(IDs[k])
                print(mothers[k])
                print(pts[k])
                print(phis[k])
                print(etas[k])
    
                    # check if Z decays to double l
#                    IDsZllcut   = ((IDs[mothers] == 23) & ((abs(IDs) == 11) | (abs(IDs) == 13) | (abs(IDs) == 15)))
#                    IDsZnunucut = ((IDs[mothers] == 23) & ((abs(IDs) == 12) | (abs(IDs) == 14) | (abs(IDs) == 16)))
#                    isZToLL = IDsZllcut.sum().astype(bool)
#                    isZToNuNu = IDsZnunucut.sum().astype(bool)
#                    print(IDs[IDsZllcut | IDsZnunucut])
#                    for ID, mother in zip(IDs,mothers):
#                        print(ID)
#                        print(mother),
#                        print("\n")
#                    for event, moms in zip(IDs[(IDsZllcut == False) & (IDsZnunucut == False)], mothers[(IDsZllcut == False) & (IDsZnunucut == False)]):
#                        print(event)
#                        print(moms),
#                        print('\n')
#                    for f,i in zip(flag,IDs):
#
#                        for j,k in zip(f,i):
#                            print("Particle:  ",k),
#                            print('\t'),
#                            print(format(j,'016b'))
#                        exit()
#                    
                ## Store isZToLL in root file and safe to eos space ##
#                tfile   = ROOT.TFile.Open(fname,"READ")
#                outname = fname.split('/')
#                if not os.path.exists(outputDir+outname[-2]):
#                    os.makedirs(outputDir+outname[-2])
#                
#                print('Cloning file')
#                treeNames = tfile.GetListOfKeys()
#                treeCopies = []
#                for treeName in treeNames:
#                    treeCopy = treeName.ReadObj().Clone()
#                    treeCopies.append(treeCopy)
#                outfile = ROOT.TFile(outputDir+outname[-2]+'/'+outname[-1], 'RECREATE')
#                ###### Create new Branch in Events ####
#                for t in treeCopies:
#                    if ('Events' in t.GetName()):
#                        newBranch = t.Branch('isZToLL',isZToLL,'isZToLL/B')
#                    print('Writing to File')
#                    t.Write()
#
#                print('Closing File:\t'+outfile.GetName())
#                outfile.Close()
#                tfile.Close()
#                

                    
if __name__ == "__main__":
    main()
