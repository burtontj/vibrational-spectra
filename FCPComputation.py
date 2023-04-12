from sympy.utilities.iterables import multiset_permutations
import numpy as np
import scipy as sp
from random import choices
import time

import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.apps import data, qchem, plot


from Constants import *
from FrankCondonComputations import *
from DataStorage import *
from DataVisualiser import compareResults, allNonRandomValues

def update0Integrals():
    def generateData(w, wp, Ud, delta, t):
        t, U1, r, U2, alpha = qchem.vibronic.gbs_params(w, wp, Ud, delta, t)
        gaussianState = generateGaussianState(U1, r, U2, alpha)
        integralZeroZero = gaussianState.fock_prob([0 for i in range(len(w))])
        return integralZeroZero
    '''
    Update the file that contains all the c0 integrals for investigated molecules. 
    These are constant and run-independent and are computable in O(1).
    '''
    print('Updating c0 integrals')
    c0Integrals = {}
    #first do the molecules without custom arguments
    for i in list(MOLECULARDATA):
        molecule = i
        print('Beginning ' + molecule)
        w, wp, Ud, delta, t = MOLECULARDATA[molecule]()
        integralZeroZero = generateData(w, wp, Ud, delta, t)
        c0Integrals.update({molecule:integralZeroZero})
    #now do the randomly generated molecules
    for i in RANDOMMOLECULES:
        molecule=i
        print('Beginning ' + molecule)
        w, wp, Ud, delta, t = randomMolecule(molecule)
        integralZeroZero = generateData(w, wp, Ud, delta, t)
        c0Integrals.update({molecule:integralZeroZero})
    #now do the randomly generated molecules with prime frequencies
    for i in RANDOMPRIMES:
        molecule=i
        print('Beginning ' + molecule)
        w, wp, Ud, delta, t = randomMolecule(molecule)
        integralZeroZero = generateData(w, wp, Ud, delta, t)
        c0Integrals.update({molecule:integralZeroZero})
    #now do the randomly generated molecules with prime frequencies and no delta
    for i in RANDOMPRIMESNODELTA:
        molecule=i
        print('Beginning ' + molecule)
        w, wp, Ud, delta, t = randomMolecule(molecule)
        integralZeroZero = generateData(w, wp, Ud, delta, t)
        c0Integrals.update({molecule:integralZeroZero})
    np.save("data/c0Integrals", c0Integrals)

def spectrumPlot(values,probs,molecule,plotIt=True):
    w, wp, Ud, delta, t = getData(molecule)  
    t, U1, r, U2, alpha = qchem.vibronic.gbs_params(w, wp, Ud, delta, t)
    numOscillators = len(w)
    zero = [0 for i in range(numOscillators)] 

    values.append([(PLOTERRORVALUE/wp[0])]+([0]*(2*len(zero)-1)))
    probs.append(1-np.sum(probs))
    s = choices(values, probs, k=NSAMPLES)
    e = qchem.vibronic.energies(s, w, wp)
    if molecule == "Formic":
        compareResults(e, qchem.vibronic.energies(data.Formic(), w, wp))
    if plotIt:
        plot.spectrum(e, xmin=-1000, xmax=8000).show()

def updatePlotInfo(C, v, factors, values, probs, zero):
    print('C' + str(C) + ' complete')
    v += checkCompleteness(list(factors[C-1].values()))
    print('completion % = ' + str(v*100))
    for i in list(factors[C-1].keys()):
        values.append(list(i) + zero)#for the sf.energies method we must provide 2N modes (as it does not assume zero starting temperature)
    probs+=list(factors[C-1].values())
    return v, values, probs

def getKey(indices, wks, length):
    '''function to generate a key for a dictionary of excited oscillators and FCPs
    indices: list of oscillators that are excited
    wks: list of the quantum numbers of the oscillators
    length: the total number of oscillators
    e.g. getKey([1,2],[5,7],7) = (0,5,7,0,0,0,0)'''
    key = [0 for i in range(length)]
    for i in range(len(indices)):
        key[indices[i]] = wks[i]
    return tuple(key)

def generateGaussianState(U1, r, U2, alpha, t=[0]):
    t = np.array(t)
    if np.any(t!=0):
        nmodes = 2*len(U1)
    else:
        nmodes = len(U1)
    gbs = sf.Program(nmodes)
    with gbs.context as q:
        if np.any(t != 0):
            for i in range(nmodes):
                sf.ops.S2gate(t[i]) | (q[i], q[i + nmodes])
        qchem.vibronic.VibronicTransition(U1, r, U2, alpha) | q[:nmodes]
    eng = sf.Engine(backend="gaussian")
    results = eng.run(gbs)
    return results.state

def findWkMax(FCOneFactors, FCTwoFactors, epsilon1, epsilon2, numOscillators):
    #3a
    wMaxVector = np.zeros(numOscillators)
    #3b
    for k in range(numOscillators):
        e1Satisfied = False
        e2Satisfied = False
        e1NeverSatisfied = False
        maxWk = MINWK
        while ((not e1Satisfied) or (not e2Satisfied)) and not e1NeverSatisfied and maxWk<(WKLIM+1): #check whether at least one inequality is satisfied; we want the last value for which one of these is not true
            if getKey([k],[maxWk],numOscillators) in FCOneFactors:
                e1Satisfied = FCOneFactors[getKey([k],[maxWk],numOscillators)]<epsilon1#check the first inequality
                l=0
                while l < numOscillators and not e2Satisfied: #check the second inequality for all l
                    if l == k: #only check for l != k 
                        ''
                    else:
                        if not getKey([k,l],[maxWk,maxWk],numOscillators) in FCTwoFactors: #(0,1,5)===(1,0,5) as the first two values are just locations so we swap if one doesn't appear in the dictionary
                            k,l = l,k 
                            e2Satisfied = FCTwoFactors[getKey([k,l],[maxWk,maxWk],numOscillators)]<epsilon2
                            k,l = l,k
                        else:
                            e2Satisfied = FCTwoFactors[getKey([k,l],[maxWk,maxWk],numOscillators)]<epsilon2         
                    l += 1
            else:
                e1NeverSatisfied = True # check edge cases
            maxWk += WKSTEP
        maxWk -= 2*WKSTEP #account for the additional incrementations DO NOT CHANGE THIS TO -1. It is -1 for the additional incrementation, then -1 because the equalities are < not <=
        wMaxVector[k] = maxWk
    return wMaxVector.tolist()

def main(maxNumberIntegrals, molecule, maxN, plot=False, skip=False, targetPercent=1):
    print('Beginning ' + molecule)
    w, wp, Ud, delta, t = getData(molecule)  
    t, U1, r, U2, alpha = qchem.vibronic.gbs_params(w, wp, Ud, delta, t)
    numOscillators = len(w)
    zero = [0 for i in range(numOscillators)] 

    if NEGLIGIBLE != 1e-10:
        SEARCHFILE = "searchTimes"+str(maxNumberIntegrals)+str(molecule)+str(maxN)+str(NEGLIGIBLE)+".txt"
        TOTALFILE = "totalClassTimes"+str(maxNumberIntegrals)+str(molecule)+str(maxN)+str(NEGLIGIBLE)+".txt"
    else:
        SEARCHFILE = "searchTimes"+str(maxNumberIntegrals)+str(molecule)+str(maxN)+".txt"
        TOTALFILE = "totalClassTimes"+str(maxNumberIntegrals)+str(molecule)+str(maxN)+".txt"
    setupFiles(maxNumberIntegrals, molecule, SEARCHFILE, TOTALFILE)

    gaussianState = generateGaussianState(U1, r, U2, alpha)
    integralZeroZero = gaussianState.fock_prob([0 for i in range(len(w))])
    values = [zero+zero] #used for sampling from FC factors
    probs = [integralZeroZero] #used for sampling from FC factors
    v = integralZeroZero #percent spectrum is complete after each class 
    currentPercent = integralZeroZero
    print('C0 complete')
    print('completion % = ' + str(v*100))
    FCFactors = []
    FCFactors.append({}) #dictionary to allow easier searching on quantum numbers and w values

    #step 1
    start = time.time()
    for k in range(numOscillators):
        wk = MINWK
        newFactor = 1
        newKey=[]
        while newFactor > NEGLIGIBLE and wk<WKLIM and currentPercent<targetPercent:
            newFactor= computeFCFactor(1, k, wkmax=wk, numOscillators=numOscillators, gaussianState=gaussianState) #compute class 1 integrals
            currentPercent+=newFactor
            newKey = getKey([k],[wk],numOscillators)
            FCFactors[0].update({newKey:newFactor})
            wk += WKSTEP
    length = time.time()-start
    v,values,probs = updatePlotInfo(1,v,FCFactors, values, probs, zero)
    write(TOTALFILE, "a", molecule, 1, length, "---", v)
    
    #step 2 
    if currentPercent < targetPercent:    
        FCFactors.append({})
        start=time.time()
        for k in range(numOscillators-1):
            for l in range(k+1, numOscillators):
                wkl = MINWK
                newFactor = NEGLIGIBLE + 1
                newKey = []
                while newFactor > NEGLIGIBLE and wkl<WKLIM and currentPercent<targetPercent:
                    newFactor = computeFCFactor(2, k,l, wkmax=[wkl,wkl], numOscillators=numOscillators, gaussianState=gaussianState) #compute class 2 integrals 
                    currentPercent+=newFactor
                    newKey = getKey([k,l],[wkl,wkl],numOscillators)
                    FCFactors[1].update({newKey:newFactor})
                    wkl+=1
                #The above finds us the maximum wkl, now we find the permutations (e.g. 1,2, 1,3, 1,4...) (ignoring pairs as that's done above). we skip k=0 as that is done in fc1
                if currentPercent < targetPercent:
                    for i in range(1, wkl+1):
                        for j in range(1, wkl+1):
                            if i!=j:
                                newFactor = computeFCFactor(2, k,l, wkmax=[i,j], numOscillators=numOscillators, gaussianState=gaussianState)
                                currentPercent+=newFactor
                                newKey = getKey([k,l],[i,j],numOscillators)
                                FCFactors[1].update({newKey:newFactor})
        length = time.time()-start
        v,values,probs = updatePlotInfo(2,v,FCFactors, values, probs, zero)
        write(TOTALFILE, "a", molecule, 2, length, "---", v)

    
    #step 3
    for n in range(3,maxN+1):
        if currentPercent < targetPercent:
            FCFactors.append({})
            #3a
            epsilon1 = NEGLIGIBLE #value from Santoro et al.
            epsilon2 = NEGLIGIBLE #value from Santoro et al.
            numIntegrals = maxNumberIntegrals + 1
            start = time.time()
            print("Beginning search for class "+str(n))
            while numIntegrals>maxNumberIntegrals: #improve this
                #3b
                wMaxVector = findWkMax(FCFactors[0], FCFactors[1], epsilon1, epsilon2, numOscillators)
                #3c
                averageWk = np.mean(wMaxVector)
                numIntegrals = sp.special.binom(numOscillators, n) * (averageWk**n)
                epsilon1 += NEGLIGIBLE
                epsilon2 += NEGLIGIBLE
            length = time.time()-start
            print(wMaxVector)
            write(SEARCHFILE, "a", molecule, n, length)
            print('Approximate FC factors to compute: ' +str(numIntegrals))

            Continue = 'y'
            if skip:
                Continue = input("continue with calculation for n="+str(n)+"? (y/n): ")
            if Continue =='n':
                continue
            start = time.time()
            oscillators = np.zeros(numOscillators)
            for i in range(n):
                oscillators[i] = 1
            permutations = list(multiset_permutations(oscillators)) #generate all combinations of oscillators for class n (e.g. for n=2, numOscillators = 4: [1,1,0,0], [1,0,1,0]...)
            indices = [[] for i in range(len(permutations))]
            for i in range(len(indices)):
                indices[i] = (np.nonzero(np.isin(permutations[i],1)))[0].tolist() #convert the permutation list to a list of indices, equivalent to a list of oscillator numbers
            #we now have (for e.g. n = 3) each triplet of oscillators, these can either be passed to the strawberry fields to use fock_prob to sample or computed classicaly
            FCFactors[n-1], currentPercent = computeMultipleFC(n, indices, wMaxVector, numOscillators=numOscillators, gaussianState=gaussianState,  currentPercent=currentPercent, targetPercent=targetPercent)
            v,values,probs = updatePlotInfo(n,v,FCFactors, values, probs, zero)
            length = time.time()-start
            write(TOTALFILE, "a", molecule, n, length, numIntegrals, v)
       
        
    #write the FC factors to a file for future use
    print("Calculations complete")
    if not skip:
        if molecule == "Thymine": # thymine is too large to write as a dictionary so we have to convert it to a numpy array first
            newFCFactors = []
            for factorDictionary in FCFactors:
                for i in factorDictionary:
                    newvalue = np.array(i)
                    newvalue = np.append(newvalue, factorDictionary[i])
                    newFCFactors.append(newvalue)
            FCFactors = np.array(newFCFactors)
        np.save("FCFactors"+str(maxNumberIntegrals)+str(molecule)+str(maxN)+str(NEGLIGIBLE), FCFactors)
    print("Writing complete")
    #plot the spectrum
    spectrumPlot(values, probs, molecule, plotIt=plot)


main(45000, 'Formic', 3, plot=True)


