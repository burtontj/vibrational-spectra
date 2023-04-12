import numpy as np

from strawberryfields.ops import *
from strawberryfields.apps import data, qchem


MINWK = 1 #minimum allowed quantum number
WKSTEP = 1 #how to increment quantum number
NEGLIGIBLE = 1e-10 #minimum value for which we consider and FC integral
#EPSILONSTEP = 1e-10 #the step to increase the inequality threshholds epsilon1 and epsilon2 by if too many integrals would be computed
NSAMPLES = 20000 #Number of samples to generate the spectrum
PLOTERRORVALUE = -1000 #The value to plot 'missing' spectra to on the graph
METHODS = ['sf','hafnian','integral'] #not implemented currently
WKLIM = 3000000 #upper limit on wk values to prevent excessive computation times - setting this to something big is equivlanet to not having it because of hte maxintegrals cutoff

'''Vibrational data:'''
def formicTestParameters():
    '''Generate vibrational data for Formic acid using Strawberry Fields data'''
    formic = data.Formic()
    w = formic.w  # ground state frequencies #what happens if these are drawn from a gaussian distribution, or all the same, or all the same but different signs
    wp = formic.wp  # excited state frequencies
    Ud = formic.Ud  # Duschinsky matrix == J in chemistry paper. what happens if this is more random - lack of structure = harder? 
    delta = formic.delta  # displacement vector == K in chemistry paper. 
    t = 0  # temperature
    return w, wp, Ud, delta, t

def formic0():
    w, wp, Ud, delta, t = formicTestParameters()
    delta = 0*delta
    return w, wp, Ud, delta, t

def negativeFormic():
    # Formic but with frequencies flipped so that they are always a negative transition
    w, wp, Ud, delta, t = formicTestParameters()
    for i in range (len(wp)):
        if wp[i]-w[i] > 0:
            wp[i], w[i]= w[i],wp[i]
    return w, wp, Ud, delta, t

def pyrroleTestParameters():
    '''Vibrational data of Pyrrole from strawberry fields data set'''
    pyrrole = data.Pyrrole(0)
    Li = pyrrole.Li  # normal modes of the ground electronic state
    Lf = pyrrole.Lf  # normal modes of the excited electronic state
    ri = pyrrole.ri  # atomic coordinates of the ground electronic state
    rf = pyrrole.rf  # atomic coordinates of the excited electronic state
    w = pyrrole.wi  # vibrational frequencies of the ground electronic state
    wp = pyrrole.wf  # vibrational frequencies of the excited electronic state
    m = pyrrole.m  # atomic masses
    Ud, delta = qchem.duschinsky(Li, Lf, ri, rf, wp, m)
    t=0
    return w, wp, Ud, delta, t

def bigPyrrole():
    w, wp, Ud, delta, t = pyrroleTestParameters()
    w, wp, delta = w[:-5], wp[:-5], delta[:-5]
    Ud = Ud[:-5,:-5]
    return w, wp, Ud, delta, t

def smallPyrrole():
    w, wp, Ud, delta, t = pyrroleTestParameters()
    w, wp, delta = w[19:], wp[19:], delta[19:]
    Ud = Ud[19:,19:]
    return w, wp, Ud, delta, t

def pyrrole0():
    w, wp, Ud, delta, t = pyrroleTestParameters()
    delta = 0*delta
    return w, wp, Ud, delta, t

def diagonalPyrrole():
    w, wp, Ud, delta, t = pyrroleTestParameters()
    identity = np.eye(len(wp))
    np.fill_diagonal(identity, np.diagonal(Ud))
    Ud = identity.copy()
    return w, wp, Ud, delta, t

def sparsePyrrole(cutOff):
    '''Return Pyrrole but with all duschinsky elements below a certain cutoff.'''
    w, wp, Ud, delta, t = pyrroleTestParameters()
    originalUd = Ud.copy()
    Ud[(cutOff>abs(Ud))] = 0
    return w, wp, Ud, delta, t

def randomMolecule(name):
    data = np.load('data/randomMolecules/'+name+'.npy', allow_pickle=True)
    w, wp, Ud, delta, t = data[0], data[1], data[2], data[3], data[4]
    return w, wp, Ud, delta, t

def thymineTestParameters():
    f = open("data/clean_thymine.data", "r")
    f.readline()#skip header
    w = f.readline().strip().split(" ")
    w = np.array(w, dtype=np.float64)
    f.readline()#skip header
    wp = f.readline().strip().split(" ")
    wp = np.array(wp, dtype=np.float64)
    f.readline()#skip header
    f.readline()#skip header
    Ud=np.empty((39,39))
    for i in range(39):
        f.readline()#skip header
        row = f.readline().strip().split(" ")
        row[:] = [x for x in row if x]
        row = np.array(row, dtype=np.float64)
        Ud[i] = row
    f.readline()#skip header
    f.readline()#skip header
    delta = f.readline().strip().split(" ")
    delta[:] = [x for x in delta if x]
    delta = np.array(delta, dtype=np.float64)

    #the below gives the partial matrices and deltas
    f.readline()#skip header
    f.readline()#skip header
    Ud0 = f.readline().strip().split(" ")
    Ud0[:] = [x for x in Ud0 if x]
    Ud0 = np.array(Ud0, dtype=np.float64).reshape((13,13))
    f.readline()#skip header
    f.readline()#skip header
    delta0 = f.readline().strip().split(" ")
    delta0[:] = [x for x in delta0 if x]
    delta0 = np.array(delta0, dtype=np.float64)

    f.readline()#skip header
    f.readline()#skip header
    Ud1 = f.readline().strip().split(" ")
    Ud1[:] = [x for x in Ud1 if x]
    Ud1 = np.array(Ud1, dtype=np.float64).reshape((26,26))
    f.readline()#skip header
    f.readline()#skip header
    delta1 = f.readline().strip().split(" ")
    delta1[:] = [x for x in delta1 if x]
    delta1 = np.array(delta1, dtype=np.float64)
    f.close()
    t=0
    return w, wp, Ud, delta, t

def smallThymine():
    f = open("data/clean_thymine.data", "r")
    f.readline()#skip header
    w = f.readline().strip().split(" ")
    w = np.array(w, dtype=np.float64)
    f.readline()#skip header
    wp = f.readline().strip().split(" ")
    wp = np.array(wp, dtype=np.float64)
    f.readline()#skip header
    f.readline()#skip header
    Ud=np.empty((39,39))
    for i in range(39):
        f.readline()#skip header
        row = f.readline().strip().split(" ")
        row[:] = [x for x in row if x]
        row = np.array(row, dtype=np.float64)
        Ud[i] = row
    f.readline()#skip header
    f.readline()#skip header
    delta = f.readline().strip().split(" ")
    delta[:] = [x for x in delta if x]
    delta = np.array(delta, dtype=np.float64)

    #the below gives the partial matrices and deltas
    f.readline()#skip header
    f.readline()#skip header
    Ud0 = f.readline().strip().split(" ")
    Ud0[:] = [x for x in Ud0 if x]
    Ud0 = np.array(Ud0, dtype=np.float64).reshape((13,13))
    f.readline()#skip header
    f.readline()#skip header
    delta0 = f.readline().strip().split(" ")
    delta0[:] = [x for x in delta0 if x]
    delta0 = np.array(delta0, dtype=np.float64)

    f.readline()#skip header
    f.readline()#skip header
    Ud1 = f.readline().strip().split(" ")
    Ud1[:] = [x for x in Ud1 if x]
    Ud1 = np.array(Ud1, dtype=np.float64).reshape((26,26))
    f.readline()#skip header
    f.readline()#skip header
    delta1 = f.readline().strip().split(" ")
    delta1[:] = [x for x in delta1 if x]
    delta1 = np.array(delta1, dtype=np.float64)
    f.close()
    t=0

    return w[:-26], wp[:-26], Ud0, delta0, 0

def bigThymine():
    f = open("data/clean_thymine.data", "r")
    f.readline()#skip header
    w = f.readline().strip().split(" ")
    w = np.array(w, dtype=np.float64)
    f.readline()#skip header
    wp = f.readline().strip().split(" ")
    wp = np.array(wp, dtype=np.float64)
    f.readline()#skip header
    f.readline()#skip header
    Ud=np.empty((39,39))
    for i in range(39):
        f.readline()#skip header
        row = f.readline().strip().split(" ")
        row[:] = [x for x in row if x]
        row = np.array(row, dtype=np.float64)
        Ud[i] = row
    f.readline()#skip header
    f.readline()#skip header
    delta = f.readline().strip().split(" ")
    delta[:] = [x for x in delta if x]
    delta = np.array(delta, dtype=np.float64)

    #the below gives the partial matrices and deltas
    f.readline()#skip header
    f.readline()#skip header
    Ud0 = f.readline().strip().split(" ")
    Ud0[:] = [x for x in Ud0 if x]
    Ud0 = np.array(Ud0, dtype=np.float64).reshape((13,13))
    f.readline()#skip header
    f.readline()#skip header
    delta0 = f.readline().strip().split(" ")
    delta0[:] = [x for x in delta0 if x]
    delta0 = np.array(delta0, dtype=np.float64)

    f.readline()#skip header
    f.readline()#skip header
    Ud1 = f.readline().strip().split(" ")
    Ud1[:] = [x for x in Ud1 if x]
    Ud1 = np.array(Ud1, dtype=np.float64).reshape((26,26))
    f.readline()#skip header
    f.readline()#skip header
    delta1 = f.readline().strip().split(" ")
    delta1[:] = [x for x in delta1 if x]
    delta1 = np.array(delta1, dtype=np.float64)
    f.close()
    t=0

    return w[13:], wp[13:], Ud1, delta1, 0

def getData(molecule):
    '''
    apply the correct function to get molecular data
    '''
    if molecule in MOLECULARDATA:
        return MOLECULARDATA[molecule]()
    elif 'random' in molecule:
        return randomMolecule(molecule)
    
'''#Dictionary of non-random molecules and their loading functions'''
#DO NOT STORE SPARSE PYRROLE IN HERE, DO IT MANUALLY TO STOP BREAKING THE C0 FUNCTION
MOLECULARDATA = {"Formic":formicTestParameters, 
                 "Pyrrole":pyrroleTestParameters, 
                 "Thymine":thymineTestParameters,
                 "Formic0":formic0,
                 "bigPyrrole":bigPyrrole,
                 "smallPyrrole":smallPyrrole,
                 "Pyrrole0":pyrrole0,
                 "smallThymine":smallThymine,
                 "bigThymine":bigThymine,
                 "negativeFormic":negativeFormic}

'''A list of the names used for plotting graphs'''
FULLNAMES = {"Formic":"Formic Acid", 
            "Pyrrole":"Pyrrole", 
            "Thymine":"Thymine",
            "Formic0":"Formic delta=0",
            "bigPyrrole":"Pyrrole Group 1",
            "smallPyrrole":"Pyrrole Group 2",
            "Pyrrole0":"Pyrrole delta=0",
            "smallThymine":"Thymine Group 1",
            "bigThymine":"Thymine Group 2",
            "negativeFormic":"Formic with Negative Frequency Changes",
            "diagonalPyrrole":"Pyrrole with Only Diagonal Duschinsky Elements",
            "randomFormicPrimes0":"Prime Random Formic",
            "randomFormicPrimesNoDelta0":"Prime Random Formic delta=0",
            "AveragerandomFormicNoDelta":"Average Random Formic delta=0",
            "AveragerandomFormic":"Average Random Formic Acid"}

'''Lists of all randomly generated molecule names'''
RANDOMFORMICS = [molecule+str(i) for molecule in ["randomFormic", "randomFormicNoDelta"] for i in range(5)]
RANDOMPYRROLES = [molecule+str(i) for molecule in ["randomPyrrole", "randomPyrroleNoDelta"] for i in range(5)]
RANDOMMOLECULES = RANDOMFORMICS + RANDOMPYRROLES

PRIMEFORMICS = ["randomFormicPrimes"+str(i) for i in range(5)]
PRIMEBIGPYRROLES = ["randomBigPyrrolePrimes"+str(i) for i in range(5)]
RANDOMPRIMES = PRIMEFORMICS+PRIMEBIGPYRROLES

NODELTAPRIMESFORMIC = ["randomFormicPrimesNoDelta"+str(i) for i in range(5)]
NODELTAPRIMESPYRROLE = ["randomBigPyrrolePrimesNoDelta"+str(i) for i in range(5)]
RANDOMPRIMESNODELTA = NODELTAPRIMESFORMIC+NODELTAPRIMESPYRROLE

ALLRANDOMS = RANDOMPRIMES+RANDOMPRIMESNODELTA+RANDOMMOLECULES
