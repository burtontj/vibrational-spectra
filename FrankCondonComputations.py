import numpy as np
import itertools
import time

from Constants import *

def timeFunc(func, *params,**kwargs):
    '''
    Function to time execution of another function
    func: the function to be timed
    params: the input parameters to the timed function
    kwargs: the keyword arguments input to the timed function
    returns: a tuple of the form (result, time) where result is the output of the function and time the time taken to execute
    '''
    start = time.time()
    output = func(*params, **kwargs)
    length = time.time()-start
    return output, length

def hermitian(m):
    return np.conjugate(np.transpose(m))

def computeZeroZeroIntegral(w, wp, J, K): #take in matrices of frequencies, duschinsky matrix and displacement matrix and compute 0-0 integral
    '''not currently implemented'''
    #from: https://aip.scitation.org/doi/pdf/10.1063/1.1725748 and https://www.tandfonline.com/doi/pdf/10.1080/0026897031000109310?needAccess=true 
    #not sure whether the J matrix can be computed without q coordinates? It can (use Strawberry fields)
    n = len(wp) #mu is the number of coordinates in the column vector Q'
    gamma = np.diag(w)**2#*4*(np.pi**2)/sp.constants.h
    gammaPrime = np.diag(wp)**2#*4*(np.pi**2)/sp.constants.h
    K=(K/wp) #relation between delta and reduced frequencies: https://strawberryfields.readthedocs.io/en/stable/code/api/api/strawberryfields.apps.qchem.vibronic.gbs_params.html#strawberryfields.apps.qchem.vibronic.gbs_params 
    #split the I00 into parts to make it more readable
    termOne = (np.linalg.det(np.matmul(gamma, gammaPrime)))**(1/4)
    X = np.linalg.multi_dot([hermitian(J), gammaPrime, J])
    termTwo = (np.linalg.det(np.matmul(J, (X+gamma))))**(-1/2)
    termThreeA = (-1/2)*np.linalg.multi_dot([hermitian(K), gammaPrime, K])
    termThreeB = (1/2)*np.linalg.multi_dot([hermitian(K), hermitian(gammaPrime), J])
    invXplusGamma = np.linalg.inv(X + gamma)
    termThreeC = np.linalg.multi_dot([hermitian(J),gammaPrime,K])
    termThree = np.exp(termThreeA + np.linalg.multi_dot([termThreeB, invXplusGamma, termThreeC]))
    return termOne*termTwo*termThree*(2**((3*n-6)/2))
     

def computeFCIntegral(C, *oscillators, wk):
    '''not currently implemented'''
    return

def checkCompleteness(FCFactors):
    total = 0.0
    for i in FCFactors:
        total+=np.sum(i)
    return total

def computeFCFactor(C, *oscillators, wkmax, method = 'sf', numOscillators, gaussianState=0, currentPercent=0, targetPercent=1):
    '''FCFs are <m|u|n>^2 and I^2. can choose which to compute here. Does FCF mean the same things in both papers? If so, where does the wk come in (I think in the m matrix)?
    C: The class number being computed
    oscillators: The indices of the excited oscillators
    wkmax: the maximum quantum numbers of those oscillators
    Method: which method to compute the FC factor with: integral (classical); sf (using sf.fock_prob); or hafnian (using the Hafnian from Raul's paper)
    numOscillators: the number of oscillators in the system
    '''
    if not method in METHODS:
        raise Exception('Method of computation must be one of: ' + str(METHODS))

    if method == 'integral':
        #use the standard FCP integrals
        return abs(computeFCIntegral(C, *oscillators, wk=wkmax))**2 #compute classicaly using FC integrals
    
    elif method == 'sf':
        #use the Gaussian state
        if C == 1:  
            mVector = np.zeros(numOscillators, dtype=int)
            mVector[oscillators[0]] = wkmax #there is only one oscillator for C=1
            factor = gaussianState.fock_prob(mVector, cutoff=max(sum(mVector)+1,10))
            return factor

        elif C==2:
            mVector = np.zeros(numOscillators, dtype=int)
            mVector[oscillators[0]] = wkmax[0]
            mVector[oscillators[1]] = wkmax[1]
            factor = gaussianState.fock_prob(mVector, cutoff=max(sum(mVector)+1,10))
            return factor

        else: 
            wkValues = []
            for i in oscillators[0]:
                wkValues.append(np.arange(MINWK, wkmax[i]+1)) #copy over the wk values for our specific index
            wkPermutations = np.array(list(itertools.product(*wkValues)))
            factors = {}
            for i in range(len(wkPermutations)):
                p = wkPermutations[i]
                mVector = np.zeros(numOscillators, dtype=int) #initialise the m vector to be all 0s, we will change it for each wk 
                #this loop could probably be improved for efficiency. However, it is small and always will be (going up to class 7 it only repeats 7 times) so is not the bottleneck
                for j in range(len(p)):
                    mVector[oscillators[0][j]] = p[j] 
                factor = gaussianState.fock_prob(mVector, cutoff=max(sum(mVector)+1,10))
                factors.update({tuple(mVector):factor})
                currentPercent += factor
                if currentPercent>=targetPercent:
                    return factors, currentPercent
            return factors, currentPercent

def computeMultipleFC(C, oscillatorIndices, wMaxVector, numOscillators, method='sf', gaussianState=0, currentPercent=0, targetPercent=1):
    factors = {}
    for i in range(len(oscillatorIndices)):
        indices = oscillatorIndices[i]
        f, currentPercent = computeFCFactor(C, indices, wkmax=wMaxVector, method=method, numOscillators=numOscillators, gaussianState=gaussianState, currentPercent=currentPercent, targetPercent=targetPercent)
        factors.update(f)
        if currentPercent>=targetPercent:
            return factors, currentPercent
    return factors, currentPercent

