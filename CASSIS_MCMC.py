#============================================================================================#
# This script performs χ2 computations using the MCMC algorithm for HC5N in the LTE model.   #
# Adapted from CASSIS documentation, it is designed for execution in CASSIS Jython.          #
# For more details, visit: http://cassis.irap.omp.eu/docs/script/README.html                 #
# Note: Execution may take some time.                                                        #
#============================================================================================#

import time
import subprocess
import ScriptEnvironment
from Range import Range
from Component import Component
from FileReader import FileReader
from LineAnalysisScripting import UserInputs
from eu.omp.irap.cassis.properties import Software
from cassisStats import writeStats
from java.io import File
from Plot import Plot

#======================================================================================#
# Inputs :                                                                             #
#======================================================================================#

# Define the velocity ranges for the lines
Range.unit          = "GHz"  # Available units: GHz, km/s, MHz, cm-1, micrometer
# For Cha-MMS1:
# r1                  = Range(18.636757026174, 18.640450306727)
# r2                  = Range(21.299122714198, 21.303395931368)
# r3                  = Range(23.961512293599, 23.966304401425)
# r4                  = Range(26.623869424849, 26.629180423656)

# For Cha-C2 (frequencies hardcoded for July 30th version of data):
r1                  = Range(18.636748308614, 18.640471784918)
r2                  = Range(21.299134617819, 21.303376939183)
r3                  = Range(23.961488862777, 23.966311069794)
r4                  = Range(26.623887660063, 26.629198191843)

# Define the step size for MCMC walkers; tune to maintain acceptance rate within [0.2, 0.5]
rpp                 = 20

# Initialize source, species, and model names
sourceName          = "CHA-C2"
speciesName         = "HC5N"
myModel             = "1C_lte"
myName              = sourceName+"_"+speciesName+"_"+myModel

# Local configuration for paths to Python executable and Cassis directories
myPython            = "/Users/kahaan/anaconda3/envs/myenv/bin/python"  # print(sys.executable) to find path
cassisDataPath      = Software.getCassisPath()+"/delivery/data/"
cassisScriptsPath   = Software.getCassisPath()+"/delivery/script/examples/"
myDirInput          = cassisDataPath
myDirOutput         = cassisDataPath
# inputFile           = myDirInput+"CHA-MMS1_HC5N.lis"
inputFile           = myDirInput+"cha-c2-hc5n-july30.lis"
outputFile          = myDirOutput+myName+".dat"

#======================================================================================#
# USER INPUTS                                                                          #
#======================================================================================#

userInputs          = UserInputs(
inputFile           = inputFile,
telescope           = "dsn",
tuningRange         = [18, 28],         # Frequency range to be used (in GHz)
tuningBand          = 60,               # Width on which spectra are taken (in km/s)
aijMin              = 1.0e-7,           # Minimum value for Einstein coefficient
eup           	    = [0.0, 30.0], 
kup           	    = ["*","*"],        
template            = "Full CDMS",      # Options: "FULL CMDS", "Full JPL", "Full VASTEL" etc...
moltags             = [75503],
tmb2ta              = False,            # Whether data is main beam temperature (False) or antenna temperature (True)
isoUnique           = False,
outputFile          = outputFile,
plotTitle           = myName,
warning             = True,

# Enter here the lines and the corresponding ranges to be taken into account in the computation
# The lines should be sorted by increasing frequency.
selectedLines       = {"1": r1, "2": r2, "3": r3, "4": r4}, 

# rms of the data around each selected line (in K)
rmsLines            = {"1-4": 0.01},

# Calibration accuracy of the data around each selected line  
calLines            = {"1-4": 0.10},

continuum           = myDirInput + "continuum-0",
observing_mode      = "PSw/DBSw"  # Alternatively observing_mode = "FSw",
)

#======================================================================================#
# MODEL INPUTS                                                                         #
#======================================================================================#

# Type of models :
# LTE  : nmol, tex, fwhm, size, vlsr and iso if there are different moltags
# RADEX: nmol, collisionFile, n_collisioners, tkin, fwhm, size, vlsr and iso

#======================================================================================#
# Parameters to be defined for all components                                          #
#======================================================================================#

comp_1              = Component(
# Needed for LTE and RADEX:
nmol 		        = {'min':1.0e10, 'max':1.0e14, 'nstep':1, 'log_mode':False},
temp 		        = {'min':3.0,    'max':15.0,   'nstep':1, 'log_mode':False},
fwhm 		        = {'min':0.2,    'max':1.5,    'nstep':1, 'log_mode':False},
size 		        = {'min':10,     'max':90,     'nstep':1, 'log_mode':False},
vlsr 		        = {'min':3.0,    'max':5.0,    'nstep':1, 'log_mode':False},
iso                 = {'min':1.0,    'max':1.0,    'nstep':1, 'log_mode':False},
interacting         = True,
reducePhysicalParam = {"nmol": rpp, "temp": rpp, "fwhm": rpp, "size": rpp, "vlsr": rpp, "iso": rpp},
model               = "lte",

# Only needed for RADEX:
collisioner         = ["p_H2"],  #n_ + the collisioner (ex: n_H2, n_He, n_p_H2)
n_p_H2              = {'min':5.0e6,  'max':5.0e8,  'nstep':1,   'log_mode':False},
collisionFile       = ["a-CH3OH-pH2.dat"],  # Collisional files in the CASSIS/database/lamda directory
geometry            = "sphere"  # Alternatively, geometry = "slab"
)

# Set the initial parameters by specifying the keys and and the values
# Empty dictionary means random initial parameter
params_1            = {"nmol": 3.4e12, 
                       "fwhm": 1.0, 
                       "vlsr": 4.1, 
                       "temp": 7.0, 
                       "size": 50
}

# Set the walker and burning values
drawNumber          = 20000   # The higher this value (the longer the execution time), the wider the area visited in the space of the χ2. 
cutOff              = 1
ratioAtCutOff       = 1  # Value must be <=1

# Execution time beginning
timeStart           = time.time()

# Computation of the MCMC
userInputs.initComponentsForMCMC([comp_1, params_1]) # For multiple sources, add extra components here

# Computation of the minimum chi2
userInputs.computeChi2MinUsingMCMC(drawNumber, cutOff, ratioAtCutOff)

#=============================================================================#
# ANALYSIS OF THE RESULTS                                                     #
#=============================================================================#

# A. Plot the best model and save the corresponding spectra and config files
lineModel = userInputs.plotBestModel(moltag=userInputs.moltags[0], overSampling=3, tuningBand=userInputs.tuningBand, telescope=userInputs.telescope)
lineModel.saveConfig(File(myDirOutput+myName+".lam"))
bestPhysicalModels  = userInputs.getBestPhysicalModels()
userInputs.saveBestPhysicalModels(myDirOutput+myName+"_bestModel.lis")

# B. Compute and write the statistics of the parameters
# (for more details, print the documentation with 'print writeStats.__doc__')
writeStats(userInputs, sigmaClip = 3)

# C. Plot the acceptance rates
# (print documentation for "Plot" with "print Plot.__doc__", documentation for "FileReader" coming soon)
reader              = FileReader(userInputs.outputFile)
reader.read_file(lines_to_skip=1)
columns             = reader.columns
iterations          = columns['#']
rates               = columns['rate']
sp                  = Plot(myName)
sp.plot(x=iterations, y=rates, xlabel="Walker", ylabel="Acceptance rate", legend="Acceptance rate", lineStyle="line", lineWidth=1.5, plotType="line", lineColor = "blue")

# CASSIS Execution time ending
timeEnd             = time.time()
print("CASSIS execution time =", timeEnd - timeStart)

# D. Launch the triangle plot in python
myPythonScript      = cassisScriptsPath+"Plots_MCMC.py"
# Set the fraction of rejected walkers here !
fracOfRejWalkers    = "0.2"
trianglePlot        = [myPython+" "+myPythonScript+" "+myDirOutput+" "+myName+" "+fracOfRejWalkers]
subprocess.Popen(trianglePlot, shell=True)
#===============================================================================#