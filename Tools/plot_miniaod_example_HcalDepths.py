# import ROOT in batch mode


import sys
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
import ROOT
from ROOT import TLorentzVector
ROOT.gROOT.SetBatch(True)
sys.argv = oldargv

from ctypes import c_uint8
###########################

##
def isAncestor(daughter,mom, targetId) :
    
    if mom.pdgId() == targetId :
        return daughter, True
    else:
        for i in xrange(0,mom.numberOfMothers()) :
            daughter, fromTarget = isAncestor(mom,mom.mother(i), targetId)
            if fromTarget :
                return daughter, True
    return None, False
##
def makeTLV(daughter):
    genpartTLV = TLorentzVector()
    genpartTLV.SetPtEtaPhiM(daughter.pt(),daughter.eta(),daughter.phi(),massDict[str(abs(daughter.pdgId()))])
    return genpartTLV
##
# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libDataFormatsFWLite.so")
ROOT.FWLiteEnabler.enable()

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events

genparts, genpartLabel         = Handle("std::vector<pat::PackedGenParticle>"), "packedGenParticles"
prunedParts, prunedPartsLabel = Handle("vector<reco::GenParticle>"),           "prunedGenParticles"

# open file (you can use 'edmFileUtil -d /store/whatever.root' to get the physical file name)
events = Events('file:StopCfg-PreProcessed_StopTuple_V2.9.0/FC3DC2F1-591D-E911-B200-44A842434739.root')
Zmass = 91.1876
massDict = {'11': .511E-3, '12': 0, '13': 105.66E-3, '14': 0, '15': 1776.86E-3, '16': 0}
listOfDaughters = []
for iev,event in enumerate(events):
    if iev >= 100: 
        break 
    daughters = []
    event.getByLabel(genpartLabel,     genparts)
    event.getByLabel(prunedPartsLabel, prunedParts)
    print "\nEvent %d: run %6d, lumi %4d, event %12d" % (iev,event.eventAuxiliary().run(), event.eventAuxiliary().luminosityBlock(),event.eventAuxiliary().event())

    # loop over genParticles in the event
    checkStatus  = True
    printDiLepNu = False
    for genpart in prunedParts.product():
        particleInfo = {"Id":genpart.pdgId(), "Status": genpart.status(), 
                        "FirstMother": 0, "LastMother": 0, "FirstDaughter": 0, "LastDaughter": 0,
                        "Pt": genpart.pt(), "Eta": genpart.eta(), "Phi": genpart.phi(), "E": genpart.energy(), "Mass": genpart.mass()}
        if(genpart.numberOfMothers()   > 0): 
            particleInfo["FirstMother"]   = genpart.mother(0).pdgId()
            particleInfo["LastMother"]    = genpart.mother(genpart.numberOfMothers()-1).pdgId()
        if(genpart.numberOfDaughters() > 0):
            particleInfo["FirstDaughter"] = genpart.daughter(0).pdgId()
            particleInfo["LastDaughter"]  = genpart.daughter(genpart.numberOfDaughters()-1).pdgId()
            
        print( "Id: %4i\tStatus: %i\tFirstMother: %4i\tLastMother: %4i\tFirstDaughter: %4i\tLastDaughter: %4i\tPt: %3.2f\tEta: %2.2f\tPhi: %2.2f\t E: %4.2f\tMass: %3.2f" 
                   % (particleInfo['Id'], particleInfo['Status'], 
                      particleInfo['FirstMother'], particleInfo['LastMother'],
                      particleInfo['FirstDaughter'], particleInfo['LastDaughter'],
                      particleInfo['Pt'], particleInfo['Eta'], particleInfo['Phi'], particleInfo['E'], particleInfo['Mass']))
        
        #if (((iev == 12) or (iev == 16) or (iev == 38) ) and (str(abs(genpart.pdgId())) in massDict.keys())):
        #    print genpart.pdgId(), genpart.status(), genpart.numberOfMothers(), genpart.mother(0).pdgId(),
        #    if (genpart.mother(1)) : print(genpart.mother(1).pdgId())
        #    else : print('')
        if (genpart.pdgId() == 23):
            for i in xrange(0, genpart.numberOfDaughters()):
                if (str(abs(genpart.daughter(i).pdgId())) in massDict.keys()):
                    checkStatus = False
                    if(printDiLepNu ): print(genpart.daughter(i).pdgId())
                    directDaughter = genpart.daughter(i)
                    daughters.append({ 'ID': directDaughter.pdgId(), 'Mother': directDaughter.mother(0).pdgId(), 'nMothers': directDaughter.numberOfMothers(),
                                       'TLV': makeTLV(directDaughter), 'Status':directDaughter.status(), 'Event': iev }) 
        if ((str(abs(genpart.pdgId())) in massDict.keys()) and (checkStatus)):#(genpart.status() == 23)): 
            directDaughter = genpart
            if(printDiLepNu ): print(genpart.pdgId())
            daughters.append({ 'ID': directDaughter.pdgId(), 'Mother': directDaughter.mother(0).pdgId(), 'nMothers': directDaughter.numberOfMothers(),
                               'TLV': makeTLV(directDaughter), 'Status':directDaughter.status(), 'Event': iev })
        #
    #
    listOfDaughters.append(daughters)

#    for genpart in genparts.product():
#        #print(genpart.pdgId())
#        if (genpart.mother(0)):
#            directDaughter, isFromTarget = isAncestor(genpart, genpart.mother(0), 23)
#            if (isFromTarget == False): 
#                directDaughter, isFromTarget = isAncestor(genpart, genpart.mother(0), 22)
#            if (isFromTarget):
#                #print(directDaughter.pdgId())
#                genpartTLV = TLorentzVector()
#                genpartTLV.SetPtEtaPhiM(directDaughter.pt(),directDaughter.eta(),directDaughter.phi(),massDict[str(abs(directDaughter.pdgId()))])
#                daughters.append({'ID': directDaughter.pdgId(), 
#                                   'TLV': genpartTLV })
#            #
#        #
#    listOfDaughters.append(daughters)
    #

for i in listOfDaughters:
    for j in range(0,len(i)):
        for k in range(j,len(i)):
            if ((i[j]['ID'] == (-1*i[k]['ID'])) and (i[j]['Mother'] == i[k]['Mother'])):
                sumTLV = TLorentzVector()
                sumTLV = i[j]['TLV'] + i[k]['TLV']
                if (abs(sumTLV.M() - Zmass) < 500):
                    print('Mass: %.2f\t Len: %i\t Event: %i\t ID: %i,%i\t Mother: %i,%i\t nMothers: %i,%i\t Status: %i,%i' 
                          % (sumTLV.M(), len(i), i[j]['Event'], i[j]['ID'], i[k]['ID'],  i[j]['Mother'], i[k]['Mother'], 
                             i[j]['nMothers'], i[k]['nMothers'], i[j]['Status'], i[k]['Status']))
            
