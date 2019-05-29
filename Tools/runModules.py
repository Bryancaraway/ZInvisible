# runModules.py

import json
from cutflowPlot import makeCutflows 
from calculateNormalizationFromLL import Normalization
from calculateShapeFromPhoton import Shape

def main():
    doPhotons = True
    useHEMVeto = True
    useAllMC = True
    verbose = False
    
    plot_dir = "more_plots"
    json_file = "run.json" 
    eras = ["2016", "2017", "2018_AB", "2018_CD"]

    N = Normalization(useAllMC, verbose)
    S = Shape(plot_dir, verbose)
    
    with open(json_file, "r") as input_file:
        runMap = json.load(input_file)
        for era in eras:
            runDir = runMap[era]
            result_file = "condor/" + runDir + "/result.root"
            makeCutflows(result_file, era, plot_dir, doPhotons, useHEMVeto)
            N.getNormAndError(result_file, era)
            S.getShape(result_file, era)

    N.makeTexFile("normalization.tex")

if __name__ == "__main__":
    main()


