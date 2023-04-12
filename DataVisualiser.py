
import matplotlib.pyplot as plt
import numpy as np
import time
from random import choices
from scipy.optimize import curve_fit
import scipy.stats
from strawberryfields.apps import plot

from Constants import *

'''helper functions for loading data'''

def get0Integral(molecule):
    integrals = np.load("data/c0Integrals.npy", allow_pickle=True).item()
    if '1e-' in molecule:
        molecule = molecule[:-5]
    return integrals[molecule]

def loadFCF(integrals, molecule, maxC):
    if 'random' in molecule:
        FCFile = "data/randomMoleculeResults/FCFactors"+str(integrals)+molecule+str(maxC)+".npy"
    else:
        FCFile = "data/FCFactors"+str(integrals)+molecule+str(maxC)+".npy"
    if molecule == "Thymine":
        return formatThymineFCF(np.load(FCFile, allow_pickle=True), maxC)
    else:        
        return np.load(FCFile, allow_pickle=True)

def formatThymineFCF(data, maxC):
    dataDictionaries = [{} for i in range(maxC)]
    for i in data:
        dataDictionaries[np.count_nonzero(i)-2].update({tuple(i[:-1]):i[-1]}) #-2 to account for the FCFactor and because FC1 is stored in row 0
    return dataDictionaries
    
def loadTimeFile(integrals, molecule, maxC, path=''):
    if '1e-' in molecule:
        timeFile = "data/randomMoleculeResults/variedEpsilon/totalClassTimes"+str(integrals)+molecule[:-5]+str(maxC)+molecule[-5:]+".txt"
    elif 'andom' in molecule:
        timeFile = "data/randomMoleculeResults/totalClassTimes"+str(integrals)+molecule+str(maxC)+".txt"
    else:
        timeFile = "data/"+path+"/totalClassTimes"+str(integrals)+molecule+str(maxC)+".txt"

    times = []
    percents = []    
    with open(timeFile, "r") as data:
        data.readline() #skip header
        for line in data:
            values = line.split(",")
            times.append(float(values[2]))
            percents.append(float(values[4]))
    return times, percents

def allNonRandomValues(exclude=[], include=list(MOLECULARDATA.keys())):  
    cs = []
    integrals = []
    molecules = []
    for molecule in list(MOLECULARDATA.keys()):
        if molecule not in exclude and molecule in include:
            if molecule == "Formic" or molecule=="negativeFormic":
                c = 7
                i = 45000
            elif molecule =="Pyrrole":
                c = 4
                i = 1000000
            elif molecule == "Thymine":
                c=5
                i = 5000000
            else:
                c = 5
                i = 1000000
            cs.append(c)
            molecules.append(molecule)
            integrals.append(i)
    return integrals, molecules, cs

def allRandomValues(exclude=[], include=RANDOMMOLECULES):
    cs = []
    integrals = []
    molecules = []
    for molecule in RANDOMMOLECULES:
        if molecule not in exclude and molecule in include:
            if "Formic" in molecule:
                c = 7
                i = 20000
            elif "Pyrrole" in molecule:
                c = 24
                i = 100000
            cs.append(c)
            molecules.append(molecule)
            integrals.append(i)
    return integrals, molecules, cs
    ''

def allValues():
    rT, rM, rC = allRandomValues()
    t, m, c = allNonRandomValues()
    return t+rT, m+rM, c+rC

def averageData(times, percents, cRanges, average, molecules, exclude=-1):
    i = 0
    ts = []
    ps = []
    cs = []
    while i < len(molecules):
        averageTs = []
        averagePs = []
        averageCs = []
        if i < len(molecules)-1:
            if molecules[i][:exclude] == molecules[i+1][:exclude]:
                averageCounter = 0
                while averageCounter<average:
                    j=i+averageCounter
                    averageTs.append(times[j])
                    averagePs.append(percents[j])
                    averageCs.append(cRanges[j])
                    averageCounter+=1
                ts.append(np.mean(averageTs,axis=0))
                ps.append(np.mean(averagePs,axis=0))
                cs.append(np.mean(averageCs,axis=0))
                i = i + averageCounter
            else:
                ts.append(times[i])
                ps.append(percents[i])
                cs.append(cRanges[i])
                i+=1
        else:
            ts.append(times[i])
            ps.append(percents[i])
            cs.append(cRanges[i])
            i+=1
    return ts, ps, cs
'''legacy graphing functions for old format data'''

def histogramMaker():
    '''LEGACY'''
    f = input("enter file name")
    times = []
    with open(f, "r") as data:
        for line in data:
            currentChar = "-"
            i=0
            while currentChar!=",":
                currentChar = line[i]
                i-=1
            i+=2
            times.append(float(line[i:]))
    count = 0
    total = 0
    for t in times:
        if t>0:
            count+=1
            total+=t
    print(total/count)
    print(count)
    print(total/len(times))
    #plt.hist(times)
    #plt.show()

def integralsVSTime(integrals, molecule, fullplot=False):
    '''LEGACY'''
    '''
    produce a plot of numintegrals against time for a given molecule
    integrals: a list of integrals (divided by 1000)
    molecule: the molecule to plot data for
    fullplot: whether to plot the search times against class number for each individual data set
    '''
    total = []
    totalSearch = []
    for i in integrals:
        value = i*1000
        f = "totalClassTimes" + str(value) + molecule + ".txt"
        times = []
        with open(f, "r") as data:
            data.readline() #skip header
            for line in data:
                time = line.split(",")[2]
                times.append(float(time))
        if fullplot:
            plt.plot([1,2,3,4,5,6,7], times)
            plt.show()
        total.append(np.sum(times))

        f = "SearchTimes" + str(value) + molecule + ".txt"
        times = []
        with open(f, "r") as data:
            data.readline() #skip header
            for line in data:
                time = line.split(",")[2]
                times.append(float(time))
        if fullplot:
            plt.plot([3,4,5,6,7], times)
            plt.show()
        totalSearch.append(np.sum(times))

    total = np.array(total)
    total = total/total[0] #scale for readability
    totalSearch = np.array(totalSearch)
    totalSearch = totalSearch/totalSearch[0]
    plt.plot(integrals, total, marker=".", label = "calculations")
    plt.plot(integrals, totalSearch, marker=".", label = "searches")
    plt.legend()
    plt.show()

'''main graphing functions'''

def numIntegralsVsPercentContribution(integrals, molecule, maxC):
    timeFile = "data/totalClassTimes"+str(integrals)+molecule+".txt"
    FCData = loadFCF(integrals, molecule, maxC)
    print("Loading complete")
    times = []
    numIntegrals = []
    percentComplete = [get0Integral(molecule)]
    with open(timeFile, "r") as data:
        data.readline() #skip header
        for line in data:
            values = line.split(",")
            times.append(float(values[2]))
            c = int(values[1])
            percentComplete.append(float(values[4]))
            numIntegrals.append(len(FCData[c-1]))#c-1 because c0 isn't stored
    deltaCompletion = []
    for i in range(1, len(percentComplete)):
        deltaCompletion.append(percentComplete[i]-percentComplete[i-1])

    fig, ax1 = plt.subplots()
    ax1.plot(np.linspace(1,maxC,maxC), np.array(times)/np.array(numIntegrals), c='b')
    ax1.set_xlabel('Class')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Time/FCF (s)', color='b')
    ax1.set_xticks(np.linspace(1,maxC,maxC))
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(np.linspace(1,maxC,maxC), deltaCompletion, color='r')
    ax2.set_ylabel('Change in Completion', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.title("Percent Completion and Time/FCF for "+str(molecule))
    plt.savefig(str(integrals) + "percentCompletionVsTimePerFCP"+molecule+str(maxC))
    plt.show()

def numIntegralsPerClass(integrals, molecule, maxC, plot=True):
    FCData = loadFCF(integrals, molecule, maxC)
    print("Loading complete")
    numIntegrals = []
    for c in range(maxC):
        numIntegrals.append(len(FCData[c]))
    if plot:
        plt.plot(np.linspace(1,maxC,maxC), np.array(numIntegrals))
        plt.xlabel("Class")
        plt.ylabel("Number of FCPs")
        plt.title("Number of FCPs computed for each class for "+molecule)
        plt.savefig(str(integrals)+"numIntegralsPerClass" + molecule + str(maxC))
        plt.show()
    else:
        return np.array(numIntegrals)

def timePerIntegral(integrals, molecule, maxC, plot=True):
    timeFile = "data/totalClassTimes"+str(integrals)+molecule+".txt"
    FCData = loadFCF(integrals, molecule, maxC)
    print("Loading complete")
    times = []
    numIntegrals = []
    with open(timeFile, "r") as data:
        data.readline() #skip header
        for line in data:
            values = line.split(",")
            times.append(float(values[2]))
            c = int(values[1])
            numIntegrals.append(len(FCData[c-1]))#c-1 because c0 isn't stored
    if plot:
        plt.plot(np.linspace(1,maxC,maxC), np.array(times)/np.array(numIntegrals))
        plt.xlabel("Class")
        plt.ylabel("Time/FCF")
        plt.title("Time/FCF for "+molecule)
        plt.savefig(str(integrals)+"timePerIntegral" + molecule + str(maxC))
        plt.show()
    else:
        return np.array(times)/np.array(numIntegrals)

def plotAllTimes(Individual=False):
    '''plot all the time/integrals for all molecules
    EXCLUDES MOLECULES WITH INPUT PARAMETERS LIKE SPARSE PYRROLE/RANDOM MOLECULES
    total: whether to plot them all on the same graph
    individual: whether to plot the graphs individually as well'''
    for molecule in MOLECULARDATA:
        if molecule == "Formic":
            c = 7
            integrals = 45000
        elif molecule =="Pyrrole":
            c = 4
            integrals = 1000000
        elif molecule == "Thymine":
            c=4
            integrals = 5000000
        else:
            c = 5
            integrals = 1000000
        time = timePerIntegral(integrals, molecule, c, False)
        if len(time)!=7:
            emptyValues = [None]*(7-c)
            time = np.append(time, emptyValues)
        plt.plot(np.linspace(1,7,7), time, label = molecule)
        if Individual:
            plt.xlabel("Class")
            plt.ylabel("Time/FCF")
            plt.title("Time/FCF for " + molecule)
            plt.show()
    if not Individual:
        plt.xlabel("Class")
        plt.ylabel("Time/FCF")
        plt.title("Time/FCF for All Complete Molecules")
        plt.legend()
        plt.savefig("AllCompleteTimes")
        plt.show()

def maximumQuantumNumber(integrals, molecule, maxC, plot=True):
    '''make a plot showing maximum quantum number for each molecule, note for random molecules I limited it'''
    '''can ignore 0 as there is always one of those'''
    FCData = loadFCF(integrals, molecule, maxC)
    print("Loading complete")
    maximumWks = np.zeros(maxC)
    maximumWkVectorSums = np.zeros(maxC)
    for c in range(maxC):
        data = list(FCData[c].keys())
        if data == []:
            maximumWks[c] = 0
            maximumWkVectorSums[c] = 0
        else:
            maximumWks[c] = np.max(data) #this finds the highest quantum number of all the oscillators considered for that C (e.g. for [(3,2), (0,4)] returns 4)
            maximumWkVectorSums[c] = max(np.sum(data, axis=1)) #this finds the sum of the biggest wk vector for that C (e.g. for [(3,2), (0,4)] returns 5)

    if plot:
        plt.bar(np.arange(maxC)+1, maximumWks)
        plt.xlabel("Class")
        plt.ylabel("Maximum Wk")
        plt.title("Maximum Quantum Number for Each FC Class For "+molecule)
        plt.show()
        plt.bar(np.arange(maxC)+1, maximumWkVectorSums)
        plt.xlabel("Class")
        plt.ylabel("Maximum Wk Vector")
        plt.title("Maximum Sum of Quantum Number Vector for Each FC Class For "+molecule)
        plt.show()
    else:
        return maximumWks, maximumWkVectorSums
    
def plotMultipleWk(integrals, molecules, maxCs, graphName = "AllMaximumQuantumNumber", average=False):
    '''
    average: whether to plot random molecules individually or averaged. if averaged the first 10 molecules provided must be two groups of 5 random molecules
    '''
    maxWks = np.empty(len(molecules), dtype=object)
    maximumSums = np.empty(len(molecules), dtype=object)
    
    for i in range(len(molecules)):
        maxWks[i], maximumSums[i] = maximumQuantumNumber(integrals[i], molecules[i], maxCs[i], plot=False)  

    if average:
        molecules = ['Average'+molecules[0][:-1], 'Average'+molecules[5][:-1]] + molecules[10:]
        maxCs = [maxCs[0], maxCs[5]] + maxCs[10:]
        maxWks = [list(np.mean(maxWks[:5])), list(np.mean(maxWks[5:10]))] + list(maxWks[10:])
        maximumSums = [list(np.mean(maximumSums[:5])), list(np.mean(maximumSums[5:10]))] + list(maximumSums[10:])
    
    for i in range(len(molecules)):
        plt.plot(np.arange(maxCs[i])+1, maxWks[i], label=molecules[i])
    plt.xlabel("Class")
    plt.ylabel("Maximum Wk")
    plt.title("Maximum Quantum Number for Each FC Class")
    plt.legend()
    plt.savefig(graphName)
    plt.show()

    for i in range(len(molecules)):
        plt.plot(np.arange(maxCs[i])+1, maximumSums[i], label = molecules[i])

    plt.xlabel("Class")
    plt.ylabel("Maximum Wk")
    plt.title("Maximum Sum of Quantum Number Vector for Each FC Class")
    plt.legend()
    plt.savefig("sum"+graphName)
    plt.show()

def percentComplete(integrals, molecule, maxC, plot=True, cumulative=False):
    '''Generate a plot showing the percentage each FC class contributes to the overall spectrum. Can be a cumulative graph if desired
    integrals: the maximum number of integrals the data set was generated for
    molecule: the molecule name for the data set
    maxC: the maximum class the data was generated up to
    plot: whether to plot the data (True) or return it (False)
    cumulative: whether to return the raw or cumulative data'''
    times, ps = loadTimeFile(integrals, molecule, maxC)
    cs = [i for i in range(maxC+1)]
    percents = [get0Integral(molecule)]
    for i in ps:
        percents.append(i)
    if not cumulative:
        for i in range(len(percents)):
            percents[i] = percents[i]-np.sum(percents[:i])
    if plot:
        plt.plot(cs, percents)
        plt.xlabel("Class")
        plt.ylabel("Percent Complete")
        plt.title("Percentage of Spectrum Complete at Each FC Class")
        plt.savefig(str(integrals) + "percentCompletionVsTimePerFCP"+molecule+str(maxC))
        plt.show()
    else:  
        return times, cs, np.array(percents)
    
def plotMultiplePercentComplete(integrals, molecules, maxCs, graphName = "AllPercentCompletes", average=0, title=False, exclude=-1, minC=0):
    '''Generate a plot showing the percentage each FC class contributes to the overall spectrum for many molecules. Can be a cumulative graph if desired
    average is the number of different molecules to average, e.g. if there are 5 random formics do average=5. 
    molecules to be averaged must be passed first to make the labelling work
    integrals: the maximum number of integrals the data set was generated for for each molecule
    molecule: the molecule names for the data sets
    maxC: the maximum class the data sets were generated up to
    graphName: the name of the file to save the data to
    average: how many molecules there are to average
    title: whether to plot a title
    exclude: how many characters to ignore at the end of a molecule's name when checking if two are equal. default is -1
    minC: the minimum class value to consider'''

    cs = [0 for i in range(len(molecules))]
    percents = np.empty(len(molecules), dtype=object)
    times = np.empty(len(molecules),  dtype=object)

    for i in range(len(molecules)):
        times[i], cs[i], percents[i] = percentComplete(integrals[i], molecules[i], maxCs[i], plot=False)
        times[i] = np.array(times[i])

    if average!=0:
        times, percents, cs = averageData(times, percents, cs, average,molecules, exclude=exclude)

    if '1e-' not in molecules[0]:
        colours = plt.cm.get_cmap('Paired')(np.linspace(0,1,len(times)))
    else:
        colours = ['#cc0000', '#ff3333', '#cc6600', '#ff9933', '#cccc00', '#ffff33', '#66cc00', '#99ff33', '#00cccc', '#33ffff','#0000cc', '#3333ff','#cc00cc', '#ff33ff']
        colours = colours[:len(times)]
    labelCounter=0
    for i in range(len(times)):
        if molecules[labelCounter] in list(FULLNAMES.keys()):
            label=FULLNAMES[molecules[labelCounter]]
        else:
            label=molecules[labelCounter]
        
        if '1e-' in molecules[labelCounter]:
            label = str(integrals[labelCounter])+' FCFs '+molecules[labelCounter][-5:]

        if 'random' in molecules[labelCounter] and average!=0:
            label = 'average'+label
            labelCounter += average
        else:
            labelCounter += 1
        plt.plot(cs[i][minC:], percents[i][minC:], label=label,c=colours[i])

    plt.xlabel("Class")
    plt.ylabel("Percent Complete")
    if title:
        plt.title("Percentage of Spectrum Complete at Each FC Class")
    if len(cs)>5:
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    else:
        plt.legend()
    plt.savefig(graphName, bbox_inches='tight')
    plt.show()
    # for i in range(len(molecules)):
    #     if molecules[i] in list(FULLNAMES.keys()):
    #         label=FULLNAMES[molecules[i]]
    #     else:
    #         label=molecules[i]
    #     plt.plot(cs[i][1:], times[i], label = label,c=colours[i])
    plt.show()

    labelCounter=0
    for i in range(len(times)):
        if molecules[labelCounter] in list(FULLNAMES.keys()):
            label=FULLNAMES[molecules[labelCounter]]
        else:
            label=molecules[labelCounter]

        if '1e-' in molecules[labelCounter]:
            label = str(integrals[labelCounter])+' FCFs '+molecules[labelCounter][-5:]

        if 'random' in molecules[labelCounter] and average!=0:
            label = 'average'+label
            labelCounter += average
        else:
            labelCounter += 1
        plt.plot(cs[i][minC:], np.cumsum(percents[i])[minC:], label = label,c=colours[i])

    plt.xlabel("Class")
    plt.ylabel("Percent Complete")
    if title:
        plt.title("Cumulative Percentage of Spectrum Complete at Each FC Class")
    if len(cs)>5:
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    else:
        plt.legend()
    plt.savefig("cumulative"+graphName, bbox_inches='tight')
    plt.show()

def spectrumPlot(integrals, molecule, maxC, returnSpectra=False, xmin=-1000, xmax=8000):
    FCData = loadFCF(integrals, molecule, maxC)
    print("Loading complete")
    zero = [0 for i in range(len(list(FCData[0].keys())[0]))]
    values = [zero+zero]
    integralZeroZero = get0Integral(molecule)
    probs = [integralZeroZero]
    for FCFactors in FCData:
        keys = list(FCFactors.keys())
        keys = np.array(keys)
        zeros = np.zeros((len(keys),len(keys[0])))
        values += list(np.hstack((keys,zeros)))
        probs += list(FCFactors.values())

    w, wp, _, _, _ = getData(molecule)
    values.append([(PLOTERRORVALUE/wp[0])]+([0]*(2*len(zero)-1)))
    probs.append(1-np.sum(probs))
    samples = choices(values, probs, k=NSAMPLES)
    #e = qchem.vibronic.energies(s, w, wp) #This doesn't work if the data is a numpy array instead of a list, so i reproduced it myself
    e = [np.dot(s[: len(s) // 2], wp) - np.dot(s[len(s) // 2 :], w) for s in samples]
    plot.spectrum(e, xmin=xmin, xmax=xmax).show()
    if returnSpectra:
        return e

def combineIndependentSpectra(integrals, partials, moleculeName, maxC):
    '''attempt to convolute TWO spectra assuming the modes are independent, i.e. P(0,0,0,0) == p(0,0)*p(0,0)
    integrals: the maximum integral cutoff used during FCP calculation
    partials: the list of partial moledcule names. Order matters
    maxC: the highest class that FCPs were calculated for
    '''
    FCData = [0 for i in range(len(partials))]
    sortedKeys = [0 for i in range(len(partials))]
    for i in range(len(partials)):
        FCData[i] = loadFCF(integrals, partials[i], maxC)
        print("Loading complete")
        FCData[i] = {x:y for d in FCData[i] for x,y in d.items()}
        #now add in the 0 integral
        newKey = tuple([0]*len(list(FCData[i].keys())[0]))
        FCData[i][newKey] = get0Integral(partials[i])
        #now sort the dictionary by value
        keys = list(FCData[i].keys())
        values = list(FCData[i].values())
        sortedIndices = np.argsort(values)
        FCData[i] = {keys[i]: values[i] for i in sortedIndices}
        sortedKeys[i] = list(FCData[i].keys()) #we need it in descending order
        sortedKeys[i].reverse()
        print('Sorting complete')

    FCData = np.array(FCData)
    fullSize = len(list(FCData[0].keys())[0]) + len(list(FCData[1].keys())[0]) #work out the size of the whole molecule
    values = []
    probs = []
    print('Maximum modes to compute = ' + str(len(FCData[0].keys())*len(FCData[1].keys())))
    start = time.time()
    i = 0
    j = 2
    while i<len(sortedKeys[0]) and j>1: #if j ==1 it means the product of the first item in the second list was negligible, so all following values will be negligible
        firstMode = sortedKeys[0][i]
        j=0
        newProb = 1
        while j<len(sortedKeys[1]) and newProb>NEGLIGIBLE: #as our list is sorted, once one probability is negligible, all others will be too
            secondMode = sortedKeys[1][j]
            newMode = np.array(list(firstMode)+list(secondMode)+[0]*fullSize)
            newProb = FCData[0][firstMode]*FCData[1][secondMode]
            values.append(newMode)
            probs.append(newProb)
            j+=1
        i+=1
    # now account for the error/missing values
    w, wp, _, _, _ = getData(moleculeName)  
    values.append([(PLOTERRORVALUE/wp[0])]+([0]*((fullSize*2)-1)))
    probs.append(1-np.sum(probs))
    print("Time taken: "+str(time.time()-start))
    # now sample and plot
    samples = choices(values, probs, k=NSAMPLES)
    #e = qchem.vibronic.energies(s, w, wp) #This doesn't work if the data is a numpy array instead of a list, so i reproduced it myself
    e = [np.dot(s[: len(s) // 2], wp) - np.dot(s[len(s) // 2 :], w) for s in samples]
    plot.spectrum(e, xmin=-1000, xmax=8000).show()
    return e

def combineSpectra(integrals, partials, moleculeName, maxC, maxCPlot = -1, returnSpectra=False):
    '''First attempt to combine two partial spectra to make a whole. Must have same maxC for both
    simplest way is to divide the probabilities proportionally (i.e. give them all equal weight) and only consider cases where all other modes are 0
    integrals: the maximum integral cutoff used during FCP calculation
    partials: the list of partial moledcule names. Order matters
    maxC: the highest class that FCPs were calculated for
    maxCPlot: the highest class to plot. if no value is passed all classes are plotted
    '''
    FCData = [0 for i in range(len(partials))]
    for i in range(len(partials)):
        FCData[i] = loadFCF(integrals, partials[i], maxC)
    print("Loading complete")

    if maxCPlot != -1:
        maxC = maxCPlot
    FCData = np.array(FCData)
    fullSize = 0
    for m in FCData:
        fullSize+=len(list(m[0].keys())[0]) #work out the size of the whole molecule
    numEvents = np.zeros(len(partials))
    for i in range(maxC):
        for j in range(len(FCData)):
            numEvents[j] += len(FCData[j][i])
    totalEvents = np.sum(numEvents)
    values = []
    probs = []
    for i in range(maxC):
        print("Calculating class " + str(i+1))
        for m in range(len(FCData)):
            FCFactors = FCData[m][i]
            keys = np.array(list(FCFactors.keys()))

            #pad the partial molecule to the size of the total molecule
            zeros = np.zeros((len(keys),fullSize-len(keys[0])))
            if m == 0:
                newValues = np.hstack((keys,zeros))
            if m == 1:
                newValues = np.hstack((zeros,keys))

            #add the t=0 modes
            zeros = np.zeros((len(keys),fullSize))
            newValues = np.hstack((newValues,zeros))
            values += list(newValues)
            # do the probabilities
            newProbs = np.array(list(FCFactors.values()))#get the FC factors
            newProbs = newProbs*numEvents[m]/totalEvents#scale the factors to account for the other partial molecules
            probs += list(newProbs)
    # now do the c0 integrals
    c0=0
    for m in range(len(partials)):
        c0+=get0Integral(partials[m])*numEvents[m]/totalEvents
    values.append(([0]*(fullSize*2)))
    probs.append(c0)

    # now account for the error/missing values
    w, wp, _, _, _ = getData(moleculeName)    
    values.append([(PLOTERRORVALUE/wp[0])]+([0]*((fullSize*2)-1)))
    probs.append(1-np.sum(probs))

    # now sample and plot
    samples = choices(values, probs, k=NSAMPLES)
    #e = qchem.vibronic.energies(s, w, wp) #This doesn't work if the data is a numpy array instead of a list, so i reproduced it myself
    e = [np.dot(s[: len(s) // 2], wp) - np.dot(s[len(s) // 2 :], w) for s in samples]
    plot.spectrum(e, xmin=-1000, xmax=8000).show()
    if returnSpectra:
        return e

def compareResults(r1, r2):
    '''Find the similarity between two results from sampling. follows histogram used in sf.plot.spectrum
    r1: the first sample list
    r2: the second sample list'''
    #compare numpy histograms with cosine similarity
    #in order to make sure the compared histograms are over the same ranges we must add the maximum and minimum to both
    r1Max = False
    r2.append(min(r1))#r1 always has the smaller minimum as it has the negative error plot value
    r1.append(min(r1))#to ensure the lengths are kept the same
    if (max(r2)>max(r1)):
        r1.append(max(r2))
        r2.append(max(r2))#to ensure the lengths are kept the same
    else:
        r2.append(max(r1))
        r1.append(max(r1))#to ensure the lengths are kept the same

    bins = int(max(r1) - min(r1)) // 5 
    h1, edges = np.histogram(r1, bins)
    h2, edges = np.histogram(r2, bins)
    #remove the length extenders (only has marginal effect but might as well)
    h1[-1]-=1
    h1[0]-=1
    h2[-1]-=1
    h2[0]-=1
    cosTheta = np.dot(h1,h2)/(np.linalg.norm(h1)*np.linalg.norm(h2))
    print('correlation coefficient: '+str(cosTheta))

def comparePartialComplete(integrals, partials, moleculeName, maxCs, method, maxCPlot=-1):
    completeSpectra = spectrumPlot(integrals[0], moleculeName, maxCs[0], True)
    if method == 'legacy':
        combinedSpectra = combineSpectra(integrals[1], partials, moleculeName, maxCs[1], maxCPlot, True)
    elif method == 'independent':
        combinedSpectra = combineIndependentSpectra(integrals[1], partials, moleculeName, maxCs[1])
    compareResults(completeSpectra, combinedSpectra)

def showDuschinsky(molecule, plot=True, title=False):
    #copied from strawberry fields website
    w, wp, Ud, delta, t = getData(molecule)
    plt.imshow(abs(Ud), cmap="viridis")
    plt.colorbar()
    plt.xlabel("Mode index")
    plt.ylabel("Mode index")
    plt.tight_layout()
    if 'random' not in molecule and title:
        plt.title("Duschinsky Matrix for "+ FULLNAMES[molecule])
    plt.savefig("Duschinsky"+molecule)
    if plot:
        plt.show()
    else:
        plt.clf()

def timePerClass(integrals, molecule, maxC, plot=True, scaled=False):
    times, percents = loadTimeFile(integrals,molecule, maxC)
    print("Loading complete")

    if plot:
        plt.plot(np.arange(maxC)+1, times)
        plt.xlabel("Class")
        plt.ylabel("Time taken")
        plt.title("Time Taken to Compute Each FC Class for "+molecule)
        plt.show()
        plt.plot(np.arange(maxC)+1, np.cumsum(times))
        plt.xlabel("Class")
        plt.ylabel("Cumulative Time Taken")
        plt.title("Cumulative Time Taken to Compute Each FC Class for "+molecule)
        plt.show()
    else:
        times = np.array(times)
        if scaled:
            times = times/times[0]
        return times, np.array(np.cumsum(times))

def multipleTimePerClass(integrals, molecules, maxCs, graphName='AllTimePerClass', average=0, scaled=False, both=True, minC=0, title=False):
    '''Generate a plot showing the time taken to compute each FC class contributes to the overall spectrum for many molecules. Can be a cumulative graph if desired
    integrals: the maximum number of integrals the data set was generated for for each molecule
    molecule: the molecule names for the data sets
    maxC: the maximum class the data sets were generated up to
    graphName: the name of the file to save the data to
    both: whether we are considering both no delta and delta'''
    times = [0 for i in range(len(molecules))]
    cumulativeTimes = np.empty(len(molecules), dtype=object)
    for i in range(len(molecules)):
        times[i], cumulativeTimes[i] = timePerClass(integrals[i], molecules[i], maxCs[i], plot=False, scaled=scaled)
    if average !=0:
        if both:
            molecules = ['Average'+molecules[0][:-1], 'Average'+molecules[5][:-1]] + molecules[10:]
            maxCs = [maxCs[0],maxCs[5]]+maxCs[10:]
            times = [list(np.mean(times[:5], axis=0)), list(np.mean(times[5:10], axis=0))] + times[10:]
            cumulativeTimes = [list(np.mean(cumulativeTimes[:5])), list(np.mean(cumulativeTimes[5:10]))] + list(cumulativeTimes[10:])
        else:
            molecules = ['Average'+molecules[0][:-1]] + molecules[5:]
            maxCs = [maxCs[0]]+maxCs[5:]
            times = [list(np.mean(times[:5], axis=0))] + times[5:]
            cumulativeTimes = [list(np.mean(cumulativeTimes[:5]))] + list(cumulativeTimes[5:])
    for i in range(len(molecules)):
        if molecules[i] in list(FULLNAMES.keys()):
            label=FULLNAMES[molecules[i]]
        else:
            label = molecules[i] 
        plt.plot((np.arange(maxCs[i])+1)[minC:], times[i][minC:], label = label)

    if scaled:
        timeString = 'Scaled Time'
    else:
        timeString = 'Time'

    plt.xlabel("Class")
    plt.ylabel(timeString + " Taken (s)")
    if title:
        plt.title(timeString + " Taken to Compute all Classes")
    plt.legend()
    plt.savefig(graphName)
    plt.show()
    for i in range(len(molecules)):
        if molecules[i] in list(FULLNAMES.keys()):
            label=FULLNAMES[molecules[i]]
        else:
            label = molecules[i] 
        plt.plot((np.arange(maxCs[i])+1)[minC:], cumulativeTimes[i][minC:], label = molecules[i])

    plt.xlabel("Class")
    plt.ylabel("Cumulative "+timeString+" Taken (s)")
    if title:
        plt.title("Cumulative "+timeString+" Taken to Compute all Classes")
    plt.legend()
    plt.savefig("cumulative"+graphName)
    plt.show()

def scatterHeatPlot(integrals, molecule, maxC, plot=True):
    times, percents = loadTimeFile(integrals,molecule,maxC)
    times = [times[0]]+times #assume the C0 takes the same time as C1 as C0 can be computed in constant time
    percents = [get0Integral(molecule)] + percents

    for i in range(len(percents)):
        percents[i] = percents[i]-np.sum(percents[:i]) #convert from cumulative to individual values)
    totalTimes = np.sum(times)
    times = np.array(times)/totalTimes

    # plt.scatter(times, percents, c= np.linspace(0,maxC, maxC+1), cmap='Reds')
    # plt.show()
    # plt.scatter(np.linspace(0,maxC, maxC+1), percents, c=times, cmap='Reds')
    # plt.show()
    # plt.scatter(np.linspace(0,maxC, maxC+1), times, c=percents, cmap='Reds')
    # plt.show()
    #plt.scatter(np.linspace(0,maxC, maxC+1), percents, times*2000, alpha=0.8, label=molecule)
    if plot:
        plt.show()
    return times, percents, np.linspace(0,maxC,maxC+1)

def multipleScatterHeatPlot(integrals, molecules, maxCs, id="", average=0, exclude=-1):
    '''average is the number of different molecules to average, e.g. if there are 5 random formics do average=5. 
    molecules to be averaged must be passed first to make the labelling work'''
    times = []
    percents = []
    cRanges = []
    for i in range(len(integrals)):
        ts, ps, cs = scatterHeatPlot(integrals[i], molecules[i], maxCs[i], plot = False)
        times.append(ts)
        percents.append(ps)
        cRanges.append(cs)

    if average!=0:
        times, percents, cRanges = averageData(times, percents, cRanges, average,molecules, exclude=exclude)
    
    labelCounter = 0
    for i in range(len(times)):
        if 'random' in molecules[labelCounter] and average!=0:
            label = 'average'+molecules[labelCounter][:-1]
            labelCounter += average
        else:
            label = molecules[labelCounter]
            labelCounter += 1
        plt.scatter(cRanges[i], percents[i], times[i]*2000, alpha=0.8, label=label)
    plt.legend(markerscale=0.25)
    plt.xlabel("Class")
    plt.ylabel("Percent Contribution to Spectra")
    plt.xticks(np.linspace(0,np.max(maxCs),np.max(maxCs)+1))
    plt.savefig("mutlipleScatterHeatPlot"+id)
    plt.show()

def totalTime(integrals, molecules, maxCs, name, linear=False, exponent=False, polynomial=False, path='', combinations=False):
    def R_squared(y, y_hat):
        y_bar = y.mean()
        ss_tot = ((y-y_bar)**2).sum()
        ss_res = ((y-y_hat)**2).sum()
        return 1 - (ss_res/ss_tot)
    times=[]
    sizes=[]
    for i in range(len(integrals)):
        #note we don't include C0 as this is computable in O(1) time
        t, p = loadTimeFile(integrals[i],molecules[i],maxCs[i], path=path)
        if len(t)>4:
            t=t[:4]     
        t=np.sum(t)
        times.append(t)
        w, _, _, _, _ = getData(molecules[i])
        sizes.append(float(len(w)))#float for fitting purposes
    if combinations:
        #manually add in the time taken to compute the combinations
        sizes.append(39)
        times.append(10879.27)
        sizes.append(24)
        times.append(6046.82)
    sizes=np.array(sizes)
    times=np.array(times)
    plt.scatter(sizes, times)
    plt.xlabel('Number of Modes')
    plt.ylabel('Time to Compute (s)')
    

    def exponential(x, a, b, c):
        return a * (2**(b * x)) + c    
    if exponent:
        popt, pcov = curve_fit(exponential, sizes, times)
        roundedPopt = [np.format_float_positional(popt[0], precision=3, unique=False, fractional=False, trim='k'), np.format_float_positional(popt[1], precision=3, unique=False, fractional=False, trim='k'), np.format_float_positional(popt[2], precision=3, unique=False, fractional=False, trim='k')]
        for i in range(len(roundedPopt)):
            if roundedPopt[i][-1]=='.':
                roundedPopt[i]=roundedPopt[i][:-1]
        exponentialRSquared = R_squared(times, exponential(sizes, *popt))
    if polynomial:
        coeffs, polynomialCov = np.polyfit(sizes, times, 2, cov=True)
        p=np.poly1d(coeffs)
        roundedPoly = [np.format_float_positional(np.asarray(p)[0], precision=3, unique=False, fractional=False, trim='k'), np.format_float_positional(np.asarray(p)[1], precision=3, unique=False, fractional=False, trim='k'), np.format_float_positional(np.asarray(p)[2], precision=3, unique=False, fractional=False, trim='k')]
        for i in range(len(roundedPoly)):
            if roundedPoly[i][-1]=='.':
                roundedPoly[i]=roundedPoly[i][:-1]
        polynomialRSquared = R_squared(times, p(sizes))

    if linear:
        slope, intercept, r_linear, p_value, std_err = scipy.stats.linregress(sizes, times)
        roundedSlope, roundedIntercept= np.format_float_positional(slope, precision=3, unique=False, fractional=False, trim='k'), np.format_float_positional(intercept, precision=3, unique=False, fractional=False, trim='k')
        if roundedSlope[-1]=='.':
            roundedSlope=roundedSlope[:-1]
        if roundedIntercept[-1]=='.':
            roundedIntercept=roundedIntercept[:-1]
        linearRSquared = r_linear**2

    sizes=np.sort(sizes)

    if exponent:
        label='f(x) = '+str(roundedPopt[0])+'(2$^{'+str(roundedPopt[1])+'x}$) '+str(roundedPopt[2]) +'; R$^2$='+str(np.round(exponentialRSquared,3))
        plt.plot(np.linspace(0,sizes[-1], 100), exponential(np.linspace(0,sizes[-1], 100), *popt),color='g', linestyle='dashed', label=label)

    if polynomial:
        label='f(x) = '+str(roundedPoly[2])+'${x^2}$ +'+str(roundedPoly[1])+'x '+str(roundedPoly[2]) +'; R$^2$='+str(np.round(polynomialRSquared, 3))
        plt.plot(np.linspace(0,sizes[-1], 100), p(np.linspace(0,sizes[-1], 100)), color='r', linestyle='dashed', label=label)

    if linear:
        label='f(x) = '+str(roundedSlope)+'x '+str(roundedIntercept) +'; R$^2$='+str(np.round(linearRSquared, 3))
        plt.plot(sizes, (slope*sizes+intercept), color='b', linestyle='dashed', label = label)
    plt.legend()
    plt.savefig(name)
    plt.show()




integrals, molecules, maxcs = allNonRandomValues(include=['Formic','Pyrrole','Thymine'])
plotMultiplePercentComplete(integrals, molecules, maxcs, graphName='main3NonRandomCompletions')
# totalTime(integrals, molecules, maxcs, name='timesCombinations', combinations=True, linear=True)
# integrals, molecules, cs = allNonRandomValues(include=["Formic", "Pyrrole", "Thymine"])
# print(integrals)
# print(cs)
# multipleScatterHeatPlot(integrals, molecules, cs, "main3")

# for i in ALLRANDOMS:
#     showDuschinsky(i, plot=False)

# integrals = [20000 for i in range(5)]+[45000]#+[100000 for i in range(10)]
# maxCs = [7 for i in range(6)]#+[24 for i in range(10)]
# multipleTimePerClass(integrals, RANDOMFORMICS[5:]+['randomFormicPrimesNoDelta0'], maxCs, graphName="averageTimesWithPrimeFormicNoDelta", average=True, scaled=False)

#averageTimesWithPrimeFormic
# comparePartialComplete([5000000, 1000000], ["smallThymine", "bigThymine"], "Thymine", [4,5], 'independent')
# comparePartialComplete(1000000, ["bigPyrrole", "smallPyrrole"], "Pyrrole", [4,5], 'independent')

# integrals = [20000 for i in range(10)]+[45000]
# maxCs = [7 for i in range(10)] + [7]
# plotMultipleWk(integrals, RANDOMFORMICS+['Formic'], maxCs, graphName="FormicAndAverageRandomFormics", average=True)

# integrals = [100000 for i in range(10)]+[1000000]
# maxCs = [24 for i in range(10)] + [4]
# # plotMultipleWk(integrals, RANDOMPYRROLES+['Pyrrole'], maxCs, graphName="PyrroleAndAverageRandomPyrroles", average=True)
# integrals = [20000 for i in range(11)]+[45000, 45000]#+[100000 for i in range(10)]
# maxCs = [7 for i in range(13)]#+[24 for i in range(10)]
# # plotMultipleWk(integrals, RANDOMFORMICS+['randomFormicPrimes0','randomFormicPrimesNoDelta0'], maxCs, graphName="averageWKsWithPrimeFormic", average=True)
# plotMultiplePercentComplete(integrals, RANDOMFORMICS+['randomFormicPrimes0','randomFormicPrimesNoDelta0',"Formic"], maxCs, graphName="averagePercentsWithAllFormics", average=False)

    # integrals = [100000 for i in range(10)]+[1000000]#+[100000 for i in range(10)]
    # maxCs = [24 for i in range(10)]+[4]#+[24 for i in range(10)]
    # # plotMultipleWk(integrals, RANDOMFORMICS+['randomFormicPrimes0','randomFormicPrimesNoDelta0'], maxCs, graphName="averageWKsWithPrimeFormic", average=True)
    # plotMultiplePercentComplete(integrals, RANDOMPYRROLES+["Pyrrole"], maxCs, graphName="averagePercentsWithAllPyrroles", average=True)

# integrals, molecules, maxcs = allNonRandomValues()
# plotMultipleWk(integrals, molecules, maxcs, graphName="NonRandomWks")



# integrals = [20000 for i in range(10)]+[100000 for i in range(10)]
# maxCs = [7 for i in range(10)]+[24 for i in range(10)]
# plotMultiplePercentComplete(integrals, RANDOMMOLECULES, maxCs, graphName="test", average=5)

# integrals, molecules, cs = allRandomValues(include=RANDOMFORMICS)
# multipleScatterHeatPlot(integrals, molecules, cs, id='nonAverageFormics', average=5)

# integrals, molecules, cs = allRandomValues(include=RANDOMFORMICS[5:])
# integrals.append(1000000)
# molecules.append("Formic0")
# cs.append(5)
# multipleTimePerClass(integrals, molecules, cs, average=0, graphName='TimePerClassFormicsNoDelta', both=False)

# molecules = []
# integrals = []
# for i in ["06", "07", "08", "09", "10", "11", "12"]:
#     for j in [20000, 60000]:
#         for k in range(5):
#             molecules.append("randomFormicNoDelta"+str(k)+'1e-'+i)
#             integrals.append(j)
# cs = [7 for i in range(len(molecules))]
# print(molecules)
# plotMultiplePercentComplete(integrals, molecules, cs, graphName='percentCompleteVariedEpsilon', average=5, exclude=-6, minC=2)

# integrals, molecules, cs = allNonRandomValues(exclude=['negativeFormic', 'bigPyrrole', 'smallPyrrole', 'bigThymine', 'smallThymine'])
# print(molecules)
# plotMultiplePercentComplete(integrals, molecules, cs, 'NonRandomCompletions')

# integrals, molecules, cs = allRandomValues(include=RANDOMFORMICS)
# i2, m2, c2 = allNonRandomValues(include=['Formic', 'Formic0'])
# integrals = integrals + i2 + [20000, 45000]
# molecules = molecules + m2 + ['randomFormicPrimes0', 'randomFormicPrimesNoDelta0']
# cs = cs + c2 +[7,7]
# plotMultiplePercentComplete(integrals, molecules, cs, 'averagePercentsWithAllFormics',average=True)
# integralsVSTime([10,15,20,25,30,35,40,50], "Formic")
# spectrumPlot(45000, 'Formic',7)
#spectrumPlot(1000000, 'bigPyrrole', 5)
#spectrumPlot(1000000, 'smallPyrrole', 5)
#spectrumPlot(50000, 'Pyrrole', 5) 
# loadFCF(5000000, "Thymine", 4)
# comparePartialComplete([1000000, 1000000], ["bigPyrrole", "smallPyrrole"], "Pyrrole", [4,5], 4)



# for molecule in MOLECULARDATA:
#     if molecule == "Formic":
#         c = 7
#         integrals = 45000
#     elif molecule =="Pyrrole":
#         c = 4
#         integrals = 1000000
#     elif molecule == "Thymine":
#         c=4
#         integrals = 5000000
#     else:
#         c = 5
#         integrals = 1000000
#     # numIntegralsPerClass(integrals, molecule, c)
# for molecule in MOLECULARDATA:
#     showDuschinsky(molecule)



# def variedEpsilonPercentComplete(epsilons, integral, graphName, maxC):
#     '''compare the effect of increasing epsilon on random molecules %complete at each class'''
#     averages = []
#     for i in epsilons:
#         percents = []
#         for j in range(5):
#             ts, ps = loadTimeFile(integral, 'randomFormicNoDelta'+str(j)+str(maxC)+'1e-'+i,maxC)
#             percents.append(ps)
#         percents = np.mean(percents, axis=0)
#         averages.append(percents)

#     cs = np.linspace(1, maxC, maxC)

#     for i in range(len(averages)):
#         plt.plot(cs, averages[i], label = '1e-'+(epsilons[i]))
#     plt.xlabel("Class")
#     plt.ylabel("Percent Complete")
#     #plt.title("Cumulative Percentage of Spectrum Complete at Each FC Class")
#     plt.legend()
#     plt.savefig(graphName)
#     plt.show()
#     print(averages)
#     print(np.cumsum(averages[0]))
#     for i in range(len(epsilons)):
#         plt.plot(cs, np.cumsum(averages[i]), label = '1e-'+(epsilons[i]))

#     plt.xlabel("Class")
#     plt.ylabel("Percent Complete")
#     plt.title("Cumulative Percentage of Spectrum Complete at Each FC Class")
#     plt.legend()
#     #plt.savefig("cumulative"+graphName)
#     plt.show()