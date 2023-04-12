import numpy as np
from scipy.stats import special_ortho_group

from Constants import *

def setupFiles(maxIntegrals, molecule, searchFile, timeFile):
    f=open(timeFile, "w")
    f.write("molecule,class,time,numintegrals,percent complete\n")
    f.close()

    f=open(searchFile,"w")
    f.write("molecule,class,time\n")
    f.close()

def write(fileName, method, *data):
    '''
    function to write a row of data to a given file
    fileName: the file to write to
    method: the writing method (w overwrites, a appends)
    items: the data to write
    '''
    f=open(fileName, method)
    for item in data:
        f.write(str(item)+',')
    f.write('\n')
    f.close()

def generateMolecule(numOscillators, w=0, wp=0, delta=0, name='randomMoleculeSize'):
    '''Generates molecular data for a random molecule based on specified inputs.
    numOscillators: the size of the molecule (in modes)
    w: the initial frequencies. Defaults to a randomly created set
    wp: the final frequencies. Defaults to a randomly created set
    delta: the displacement vector. Default is no displacement
    bias: NOT IMPLEMENTED. Selects a molecule with a certain shape
    name: the name for the file the molecule will be saved in'''
    if type(w)==int:
        w = np.random.rand(numOscillators) 
    if type(wp)==int:
        wp = np.random.rand(numOscillators)
    if type(delta)==int:
        delta = np.zeros(numOscillators)
    #Ud = np.random.rand(numOscillators, numOscillators) #ORIGINALLY USED THIS uniformly random matrix
    Ud = special_ortho_group.rvs(numOscillators)
        
    data = [w, wp, Ud, delta,0] #0 for temperature
    np.save(name, data)

'''below used to make random molecules based on formic acid and big pyrrole with prime-number based frequencies'''
'''big pyrrole chosen to try and keep computation times down'''
# w, wp, ud, delta, t = bigPyrrole()
# print(wp)
# wp = np.log(np.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67]))*wp[0]
# print(wp)
# for i in range(5):
#     name='randomBigPyrrolePrimes'+str(i)
#     generateMolecule(len(w), w=w, wp=wp, delta=delta, name=name)

# w, wp, ud, delta, t = formicTestParameters()
# print(wp)
# wp = np.log(np.array([17,13,11,7,5,3,2]))*wp[0]/3 #chosen following Raul's notes and dividing by 3 to keep similar size. keeping them in descending order 
# print(wp)
# for i in range(5):
#     name='randomFormicPrimes'+str(i)
#     generateMolecule(len(w), w=w, wp=wp, delta=delta, name=name)
'''below used to make random molecules based on Pyrrole and Formic acid'''
# w, wp, ud, delta, t = pyrroleTestParameters()
# for i in range(5):
#     name='randomPyrroleNoDelta'+str(i)
#     generateMolecule(len(w), w=w, wp=wp, name=name)
#     name='randomPyrrole'+str(i)
#     generateMolecule(len(w), w=w, wp=wp, delta=delta, name=name)
# w, wp, ud, delta, t = formicTestParameters()
# for i in range(5):
#     name='randomFormicNoDelta'+str(i)
#     generateMolecule(len(w), w=w, wp=wp, name=name)
#     name='randomFormic'+str(i)
#     generateMolecule(len(w), w=w, wp=wp, delta=delta, name=name)

'''no longer used'''
# f = open("totalClassTimes10000.txt","r")
# f.readline()
# cs = []
# times = []
# for line in f:
#     cs.append(float(line.split(",")[1]))
#     times.append(float(line.split(",")[2]))
# plt.plot(cs, times)
# plt.show()

# f = open("totalClassTimes100000.txt","r")
# f.readline()
# cs = []
# times = []
# for line in f:
#     cs.append(float(line.split(",")[1]))
#     times.append(float(line.split(",")[2]))
# plt.plot(cs, times)
# plt.show()
