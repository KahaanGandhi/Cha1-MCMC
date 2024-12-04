# ----------------------------------------------------------------------------------
# Based on the script available at:
# McGuire, B.A., https://github.com/bmcguir2/simulate_lte
# 
# Although the spectral simulator in Loomis et al. (2021) is a lightweight version 
# of this script, utilizing the same functions and classes, the full version is 
# occasionally useful for reference due to its comprehensive documentation.
# 
# This inclusion serves as a reference only, with no direct contributions from me.
# ----------------------------------------------------------------------------------

#!/usr/bin/env python

#############################################################
#						Revision History					#
#############################################################

# reads in an spcat catalog and displays a spectrum

# 1.0 - Project start
# 2.0 - ipython overhaul and simplification
# 2.1 - fixes bug with status and S/C conversion
# 2.2 - fixes bug with recall and sum_stored
# 2.3 - patch for 13ch3oh
# 2.4 - fixes bug with catalogs not at 300 K
# 2.6 - adds autoset functionality
# 2.7 - speeds up gaussian simulations
# 2.8 - further speeds up gaussian simulations and normalizes spectral resolution
# 2.9 - adds labeling ability (Eupper and Quantum Numbers) to current plot
# 3.0 - adds residual plotting ability
# 3.1 - adds residual saving ability
# 3.3 - removes labels (broken on some machines); adds ability to print out line information
# 3.4 - adds ability to do Gaussian fitting of data in the program
# 3.5 - adds ability to mass-generate a p-list for gauss fitting
# 3.6 - adds ability to convert the observations from Jy/beam to K, or K to Jy/beam.
# 3.7 - adds ability to simulate double doublets from cavity FTMW
# 3.8 - adds ability to read in frequencies to plot just as single intensity lines; or to do it manually with a list.  changes default plotting to steps, adds ability to switch back to lines.  
# 3.9 - adds ability to plot manual catalogs with a velocity offset
# 3.91 - minor change to how plots are titled
# 4.0 - changes the simulation to perform a proper 'radiative transfer' which first calculates an opacity, and then applies the appropriate optical depth correction
# 4.1 - added a K to Jy/beam converter that wasn't there before...
# 4.2 - adds Aij to print_lines() readout
# 4.3 - fixes edge case where two lines have same frequency in catalog for print_lines()
# 4.4 - minor warning fix
# 5.0 - adds ability to do velocity stacking
# 5.1 - adds ability to correct for beam dilution
# 5.2 - adds ability to save velocity stacked spectra
# 5.3 - bug fix for noisy data without lines in peak finding
# 5.4 - minor correction to calculation of optically thick lines
# 5.5 - added ability to cut out spectra around simulated stick spectra
# 5.6 - update to autoset_limits() to not simulate where there's no data between the absolute upper and lower bounds
# 5.7 - adds flag to run simulations using the Planck scale and a beam size to convert to Jy/beam
# 6.0 - major update to Tbg handling
# 6.1 - streamlining the calculations of gaussian after the tbg updates
# 6.2 - adds sgr b2 non-thermal background option
# 6.3 - fixes bug in polynomial continuum temperature calculation
# 6.4 - added utility function for checking Tbg at a given frequency
# 6.5 - bug fix to beam dilution correction for sgr b2 non-thermal background continuum
# 6.6 - bug fix to planck conversion in the solid angle calculation
# 6.7 - added greybody continuum option
# 6.8 - update to print_lines() to show sijmu.
# 6.9 - added ability to add vibrational corrections to partition function.
# 6.10 - fixes bug when using multiple constant continuum values.  Introduces 'constant' as tbg_type for this.
# 6.13 - adds ability to filter out windows from stacking that have lines in them already at the center.
# 6.14 - more robust SNR values for stacked spectra
# 6.15 - re-enabled eta for single-dish observations
# 6.16 - added flag for interferometric, allowing beam-dilution for that
# 6.17 - more robust RMS finding in velocity stacking
# 6.18 - edge of band detection to velocity stacking
# 6.19 - better flagging of lines in velocity stacking
# 6.20 - adds load_asai shortcut
# 6.21 - more accurate vibrational partition functions
# 6.22 - more functionality to sim_params
# 6.23 - custom aliases
# 6.24 - updated sim_params for tbg
# 6.25 - new partition functions for some molecules
# 6.26 - adds velocity stack postage stamp plot functionality
# 6.27 - adds ability to make postage-stamp plots
# 6.28 - makes old restore files backwards compatible; adds additional postage functionality
# 6.29 - adds harmonic progression plotting functionality
# 6.30 - adds a sum_stored_thin() option for summing optically thin spectra, and adds ability for stacking to work on summed spectra
# 6.31 - adds ability to plot errors on postage stamp plots
# 6.32 - adds ability to add markers on range plots
# 6.33 - updates flux simulation calculation; does not consider 'beam dilution' in these cases
# 6.34 - adds numpy saving and loading for observations
# 6.35 - adds creation of matched filter plot to velocity_stack
# 6.36 - saves tau information for stacking / summing
# 6.37 - adds glow calculation from loomis
# 6.38 - update to matched filter script
# 6.39 - updates to line flagging in stacking
# 6.40 - adds ability to simulate spectra based on observations
# 6.41 - fixes bug in mf script
# 6.42 - adds option to blank lines not in central portions of stacks and mfs
# 6.43 - adds option to return max mf response within range
# 6.44 - adds thioacetaldehyde partition function; fixes edge case with glow
# 6.45 - adds ability to check beam size
# 6.46 - adds utility functions for quickly checking obs for lines
# 6.47 - adds ability to set individual ylims on postage stamp plots
# 6.48 - updates matched filter to use correlate rather than convolve
# 6.49 - propynal and thiopropynal partition functions
# 6.50 - bug fix for print_lines
# 6.51 - read a molsim formatted catalog

#############################################################
#							Preamble						#
#############################################################

import sys

#Python version check

if sys.version_info.major != 3:

	print("This code is written in Python 3.  It will not execute in Python 2.7.  Exiting (sorry).")
	
	quit()

import numpy as np
from numpy import exp as exp
import time as tm
import warnings
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import itertools
from datetime import datetime
from scipy.optimize import curve_fit
import peakutils
import math
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
import ast
from scipy import signal

matplotlib.rc('text', usetex = True)
matplotlib.rc('text.latex',preamble=r'\usepackage{cmbright}')

#matplotlib.rc('text', usetex = False)
#matplotlib.rc('text.latex',preamble='')



version = 6.51

h = 6.626*10**(-34) #Planck's constant in J*s
k = 1.381*10**(-23) #Boltzmann's constant in J/K
kcm = 0.69503476 #Boltzmann's constant in cm-1/K
ckm = 2.99792458*10**5 #speed of light in km/s
ccm = 2.99792458*10**10 #speed of light in cm/s
cm = 2.99792458*10**8 #speed of light in m/s

#############################################################
#							Warning 						#
#############################################################

print('\n--- MAJOR CHANGE STARTING IN VERSION 6.0 ---') 

print('\nThere has been a major update to the way the program treats background temperatures.  It now allows for different background temperatures at different frequencies, as well as permitting functionalized backgrounds.  As a result, the old system of simply setting Tbg = X will no longer function in Version 6.0+.  Please see the documentation for the new protocols.') 

#############################################################
#							Defaults 						#
#############################################################

auto_update = False

first_run = True

GHz = False

interferometer = False

quietflag = False #turn on to suppress warnings about how long the simulation will take.

rms = float('-inf') #rms noise.  the simulation won't simulate any lines 10x smaller than this value.

thermal = float('inf') #initial default cutoff for optically-thick lines (i.e. don't touch them unless thermal is modified.)

T = 300 #temperature for simulations.  Default is 300 K.

catalog_file = None #catalog file to load in.  Needs to be a string.

C = 1.0E13 #column density.  Units are cm-2.

vlsr = 0.0 #vlsr offset applied to simulation.  Default is 0 km/s.

ll = float('-inf') #lower limit for the simulation range.  Default is none.

ul = float('inf') #upper limit for the simulation range.  Default is none.

spec = None	#file of a laboratory or observational spectrum to load in for comparison.  Default is none.

dV = 5.0 #linewidth of the simulation.  Default is 5.0 km/s.
	
CT = 300.0 #temperature the catalog is simulated at.  Default is 300 K.

gauss = True #toggle for simulating Gaussians or a stick spectrum.  Default is True.

dish_size = 100 #for use if beam corrections are desired; given in meters

source_size = 1E20 #for use if beam corrections are desired; given in arcseconds

eta = 1.0 #beam efficiency of the telescope.  Set this option manually, with configure_telescope(), or with init_telescope(). 

npts_line = 15 #default is 15 points across each line

res_kHz = False #if res_kHz is set to True, then the resolution of the Gaussian simulation is calculated using the value for res, and units of kHz

res_kms = False #if res_kms is set to True, then the resolution of the Gaussian simulation is calculated using the value for res, and units of km/s

res = 0.01 #resolution used in Gaussian simulation if res_kHz or res_kms is set to True.

cavity_ftmw = False #if set to True, simulates doubler doublets from the cavity FTMW. 

cavity_dV = 0.13 #sets the default cavity linewidth to 0.13 km/s

cavity_split = 0.826 #sets the default doppler splitting in the cavity to 0.826 km/s in each direction.

draw_style = 'steps' #can be toggled on and off for going between drawing steps and drawing lines between points using use_steps() and use_lines()

planck = False #flag to use planck scale.  If planck = True is enabled, a synthesized beam size must also be provided using synth_beam = [bmaj,bmin] below.

synth_beam = ['bmaj','bmin'] #to be used with planck = True conversions.  Will throw an error if you don't set it in the program.

sim = {} #dictionary to hold stored simulations

lines = {} #dictionary to hold matplotlib lines

tbg = [] #to hold background temperatures

vibs = None #This is a list of vibrational *frequencies* in units of cm-1

match_obs = False #if set to true, then the simulation will use the x-coords of the observations.

two_fwhm_only = False #if set to true, will simulate only two FWHM around each line 

############ Tbg Parameters ##############

#tbg_params is to hold the actual parameters used to calculate Tbg.  

	#If it is a constant, then it can be passed an integer.  tbg_type must be 'constant' and tbg_order must be an integer 0.  These are the defaults.  Other possibilities are described below.

tbg_params = 2.7

#tbg_type can be the following:

	#'constant' is a constant value.  If there are multiple ranges, then a value must be giving for each range.

	#'poly' is a polynomial of order set by tbg_order = X, where X is the order and an integer.  Tbg_params must be a list of length = X+1.  So a first order polynomial needs two values [A,B] in the tbg_params: y = Ax + B.
	
	#'power' is a power law of the form Y = Ax^B + C.  tbg_params must be a list with three values [A,B,c]
	
	#'sgrb2' invokes a special value tbg = (10**(-1.06*np.log10(frequency/1000) + 2.3))
	
	#'greybody' is a greybody continuum, requiring parameters: 		
		#T = tbg_params[0] #in Kelvin
		#beta = tbg_params[1] 
		#tauref = tbg_params[2] 
		#taufreq = tbg_params[3] #in GHz
		#major = tbg_params[4] #major axis of beam in arcsec
		#minor = tbg_params[5] #minor axis of beam in arcsec

tbg_type = 'constant'

#tbg_range can contain a list of paired upper and lower limits, themselves a length 2 list, for the sets of parameters in tbg_params to be used within.  If ranges are defined, any bit of the simulation not in the defined range defaults to 2.7 K.  float('-inf') or float('inf') are valid range values.

tbg_range = []

#Some examples:

#To have three different frequency ranges (100000-120000 MHz, 150000-160000, and 190000-210000 MHz), all with their own tbg constants of 27, 32, and 37, the following must be input:

	#tbg_params = [27,32,37]
	#tbg_range = [[100000,120000],[150000,160000],[190000,210000]]
	
#To have a power law across the entire simulation, with Y = Ax^B + C and A = 2, B = 1.2, and C = 0:

	#tbg_params = [2,1.2,0]
	#tbg_type = 'power'

#To have three polynomials, of orders 1, 3, and 4, over the three different ranges above, you'd need:

	#tbg_params = [[1.2,5],[1.7,1.3,2,4],[2,4,-0.7,0.8,1.2]]
	#tbg_range = [[100000,120000],[150000,160000],[190000,210000]]	



##########################################

vel_stacked = [] #to hold velocity-stacked spectra
int_stacked = []

vel_sim_stacked = [] #to hold velocity-stacked simulated spectra
int_sim_stacked = []

freq_obs = [] #to hold laboratory or observational spectra
int_obs = []

freq_sim = [] #to hold simulated spectra
int_sim = [] 
int_tau = []

freq_sum = [] #to hold combined spectra
int_sum = []

freq_man = [] #to hold manual (frequency only) spectra
int_man = []

freq_resid = [] #to hold residual spectra
int_resid = []

velocity_mf = [] #to hold matched filter spectra
intensity_mf = []

current = catalog_file

obs_name = ''

colors = itertools.cycle(['#ff8172','#514829','#a73824','#7b3626','#a8ac87','#8c8e64','#974710','#d38e20','#ce9a3a','#ae7018','#ac5b14','#64350f','#b18f59','#404040','#791304','#1f2161','#171848','#3082fe','#2c5b5e','#390083','#5c65f8','#6346fa','#3c3176','#1cf6ba','#c9bcf0','#90edfc','#3fb8ee','#b89b33','#e7d17b'])
styles = itertools.cycle(['-','--','-.',':'])

#############################################################
#							Functions						#
#############################################################
	
#read_cat reads the catalog file in

def read_cat(catalog_file):

	'''
	Reads in a catalog file line by line
	'''

	my_array = []

	try:
		with open(catalog_file) as input:
	
			for line in input:
		
				my_array.append(line)	
	except TypeError:
		print('Specify a catalog file with catalog_file = \'x\'')
		return			
			
	return my_array	
	
#fix_pm fixes +/- quantum number issues

def fix_pm(qnarray):

	if '+' or '-' in qnarray:
	
		qnarray[qnarray == ''] = '0'
		qnarray[qnarray == '+'] = '1'
		qnarray[qnarray == '-'] = '2'
		
	return qnarray
	
#fix_qn fixes quantum number issues

def fix_qn(old_qn):

	'''
	fixes quantum number issues arising from the use of alphabet characters to represent numbers in spcat
	'''

	new_qn = 000
			
	if 'A' in old_qn:
		
		new_qn = 100 + int(old_qn[1])
		
	if 'B' in old_qn:
		
		new_qn = 110 + int(old_qn[1])	
		
	if 'C' in old_qn:
		
		new_qn = 120 + int(old_qn[1])		

	if 'D' in old_qn:
		
		new_qn = 130 + int(old_qn[1])
		
	if 'E' in old_qn:
		
		new_qn = 140 + int(old_qn[1])
		
	if 'F' in old_qn:
		
		new_qn = 150 + int(old_qn[1])
		
	if 'G' in old_qn:
		
		new_qn = 160 + int(old_qn[1])
		
	if 'H' in old_qn:
		
		new_qn = 170 + int(old_qn[1])				
		
	if 'I' in old_qn:
		
		new_qn = 180 + int(old_qn[1])	
		
	if 'J' in old_qn:
		
		new_qn = 190 + int(old_qn[1])
		
	if 'K' in old_qn:
		
		new_qn = 200 + int(old_qn[1])
		
	if 'L' in old_qn:
		
		new_qn = 210 + int(old_qn[1])
		
	if 'M' in old_qn:
		
		new_qn = 220 + int(old_qn[1])	
		
	if 'N' in old_qn:
		
		new_qn = 230 + int(old_qn[1])	
		
	if 'O' in old_qn:
		
		new_qn = 240 + int(old_qn[1])
		
	if 'P' in old_qn:
		
		new_qn = 250 + int(old_qn[1])
		
	if 'Q' in old_qn:
		
		new_qn = 260 + int(old_qn[1])	
		
	if 'R' in old_qn:
		
		new_qn = 270 + int(old_qn[1])
		
	if 'S' in old_qn:
		
		new_qn = 280 + int(old_qn[1])
		
	if 'T' in old_qn:
		
		new_qn = 290 + int(old_qn[1])	
		
	if 'U' in old_qn:
		
		new_qn = 300 + int(old_qn[1])	
		
	if 'V' in old_qn:
		
		new_qn = 310 + int(old_qn[1])
		
	if 'W' in old_qn:
		
		new_qn = 320 + int(old_qn[1])	
		
	if 'X' in old_qn:
		
		new_qn = 330 + int(old_qn[1])	
		
	if 'Y' in old_qn:
		
		new_qn = 340 + int(old_qn[1])	
		
	if 'Z' in old_qn:
		
		new_qn = 350 + int(old_qn[1])
		
	if 'a' in old_qn:
		
		new_qn = 100 + int(old_qn[1])
		
	if 'b' in old_qn:
		
		new_qn = 110 + int(old_qn[1])	
		
	if 'c' in old_qn:
		
		new_qn = 120 + int(old_qn[1])		

	if 'd' in old_qn:
		
		new_qn = 130 + int(old_qn[1])
		
	if 'e' in old_qn:
		
		new_qn = 140 + int(old_qn[1])
		
	if 'f' in old_qn:
		
		new_qn = 150 + int(old_qn[1])
		
	if 'g' in old_qn:
		
		new_qn = 160 + int(old_qn[1])
		
	if 'h' in old_qn:
		
		new_qn = 170 + int(old_qn[1])				
		
	if 'i' in old_qn:
		
		new_qn = 180 + int(old_qn[1])	
		
	if 'j' in old_qn:
		
		new_qn = 190 + int(old_qn[1])
		
	if 'k' in old_qn:
		
		new_qn = 200 + int(old_qn[1])
		
	if 'l' in old_qn:
		
		new_qn = 210 + int(old_qn[1])
		
	if 'm' in old_qn:
		
		new_qn = 220 + int(old_qn[1])	
		
	if 'n' in old_qn:
		
		new_qn = 230 + int(old_qn[1])	
		
	if 'o' in old_qn:
		
		new_qn = 240 + int(old_qn[1])
		
	if 'p' in old_qn:
		
		new_qn = 250 + int(old_qn[1])
		
	if 'q' in old_qn:
		
		new_qn = 260 + int(old_qn[1])	
		
	if 'r' in old_qn:
		
		new_qn = 270 + int(old_qn[1])
		
	if 's' in old_qn:
		
		new_qn = 280 + int(old_qn[1])
		
	if 't' in old_qn:
		
		new_qn = 290 + int(old_qn[1])	
		
	if 'u' in old_qn:
		
		new_qn = 300 + int(old_qn[1])	
		
	if 'v' in old_qn:
		
		new_qn = 310 + int(old_qn[1])
		
	if 'w' in old_qn:
		
		new_qn = 320 + int(old_qn[1])	
		
	if 'x' in old_qn:
		
		new_qn = 330 + int(old_qn[1])	
		
	if 'y' in old_qn:
		
		new_qn = 340 + int(old_qn[1])	
		
	if 'z' in old_qn:
		
		new_qn = 350 + int(old_qn[1])																																									
				
	#qnarray[line] = int(new_qn)
	
	return int(new_qn)			
	
# splices the catalog file appropriately, then populates a numpy array with the data

def splice_array(x):

	'''
	splices the catalog file appropriately, then populates a numpy array with the data
	'''

	frequency = np.arange(len(x),dtype=np.float)
	error = np.arange(len(x),dtype=np.float)
	logint = np.arange(len(x),dtype=np.float)
	dof = np.arange(len(x),dtype=np.int)
	elower = np.arange(len(x),dtype=np.float)
	gup = np.arange(len(x),dtype=np.int)
	tag = np.arange(len(x),dtype=np.int)
	qnformat = np.arange(len(x),dtype=np.int)
	qn1 = np.arange(len(x),dtype=object)
	qn2 = np.empty(len(x),dtype=object)
	qn3 = np.empty(len(x),dtype=object)
	qn4 = np.empty(len(x),dtype=object)
	qn5 = np.empty(len(x),dtype=object)
	qn6 = np.empty(len(x),dtype=object)
	qn7 = np.empty(len(x),dtype=object)
	qn8 = np.empty(len(x),dtype=object)
	qn9 = np.empty(len(x),dtype=object)
	qn10 = np.empty(len(x),dtype=object)
	qn11 = np.empty(len(x),dtype=object)
	qn12 = np.empty(len(x),dtype=object)

	for line in range(len(x)):
	
		frequency[line] = float(str(x[line][:13]).strip())
		error[line] = float(str(x[line][13:21]).strip())
		logint[line] = float(str(x[line][21:29]).strip())
		dof[line] = int(str(x[line][29:31]).strip())
		elower[line] = float(str(x[line][31:41]).strip())
		try:
			gup[line] = int(str(x[line][41:44]).strip()) if str(x[line][41:44]).strip() else ''
		except ValueError:
			gup[line] = fix_qn(str(x[line][41:44]))
		tag[line] = int(str(x[line][44:51]).strip())
		qnformat[line] = int(str(x[line][51:55]).strip())

		qn1[line] = str(x[line][55:57]).strip()
		qn2[line] = str(x[line][57:59]).strip()
		qn3[line] = str(x[line][59:61]).strip()
		qn4[line] = str(x[line][61:63]).strip()
		qn5[line] = str(x[line][63:65]).strip()
		qn6[line] = str(x[line][65:67]).strip()
		qn7[line] = str(x[line][67:69]).strip()
		qn8[line] = str(x[line][69:71]).strip()
		qn9[line] = str(x[line][71:73]).strip()
		qn10[line] = str(x[line][73:75]).strip()
		qn11[line] = str(x[line][75:77]).strip()
		qn12[line] = str(x[line][77:]).strip()


	if '+' in qn1 or '-' in qn1:
	
		qn1 = fix_pm(qn1)
		
	if '+' in qn2 or '-' in qn2:
	
		qn2 = fix_pm(qn2)	

	if '+' in qn3 or '-' in qn3:
	
		qn3 = fix_pm(qn3)
		
	if '+' in qn4 or '-' in qn4:
	
		qn4 = fix_pm(qn4)		
			
	if '+' in qn5 or '-' in qn5:
	
		qn5 = fix_pm(qn5)
	
	if '+' in qn6 or '-' in qn6:
	
		qn6 = fix_pm(qn6)
		
	if '+' in qn7 or '-' in qn7:
	
		qn7 = fix_pm(qn7)
		
	if '+' in qn8 or '-' in qn8:
	
		qn8 = fix_pm(qn8)
		
	if '+' in qn9 or '-' in qn9:
	
		qn9 = fix_pm(qn9)
		
	if '+' in qn10 or '-' in qn10:
	
		qn10 = fix_pm(qn10)
		
	if '+' in qn11 or '-' in qn11:
	
		qn11 = fix_pm(qn11)
		
	if '+' in qn12 or '-' in qn12:
	
		qn12 = fix_pm(qn12)														

	for line in range(len(qn1)):
	
		try:
			qn1[line] = int(qn1[line])
		except ValueError:
			qn1[line] = fix_qn(qn1[line])
			
	for line in range(len(qn2)):
	
		try:
			qn2[line] = int(qn2[line])
		except ValueError:
			qn2[line] = fix_qn(qn2[line])
			
	for line in range(len(qn3)):
	
		try:
			qn3[line] = int(qn3[line])
		except ValueError:
			qn3[line] = fix_qn(qn3[line])						
			
	for line in range(len(qn4)):
	
		try:
			qn4[line] = int(qn4[line])
		except ValueError:
			qn4[line] = fix_qn(qn4[line])

	for line in range(len(qn5)):
	
		try:
			qn5[line] = int(qn5[line])
		except ValueError:
			qn5[line] = fix_qn(qn5[line])
			
	for line in range(len(qn6)):
	
		try:
			qn6[line] = int(qn6[line])
		except ValueError:
			qn6[line] = fix_qn(qn6[line])
			
	for line in range(len(qn7)):
	
		try:
			qn7[line] = int(qn7[line])
		except ValueError:
			qn7[line] = fix_qn(qn7[line])
			
	for line in range(len(qn8)):
	
		try:
			qn8[line] = int(qn8[line])
		except ValueError:
			qn8[line] = fix_qn(qn8[line])
			
	for line in range(len(qn9)):
	
		try:
			qn9[line] = int(qn9[line])
		except ValueError:
			qn9[line] = fix_qn(qn9[line])
						
	for line in range(len(qn10)):
	
		try:
			qn10[line] = int(qn10[line])
		except ValueError:
			qn10[line] = fix_qn(qn10[line])
				
	for line in range(len(qn11)):
	
		try:
			qn11[line] = int(qn11[line])
		except ValueError:
			qn11[line] = fix_qn(qn11[line])
			
	for line in range(len(qn12)):
	
		try:
			qn12[line] = int(qn12[line])
		except ValueError:
			qn12[line] = fix_qn(qn12[line])	
																							
	return frequency,error,logint,dof,elower,gup,tag,qnformat,qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8,qn9,qn10,qn11,qn12
	
#det_qns determines how many qns represent each state

def det_qns(qnformat):

	'''
	determines how many qns represent each state
	'''
	
	qns = int(str(qnformat[0])[-1:])
	
	if qns > 6:
	
		qns = 6

	return qns

#calc_q will dynamically calculate a partition function whenever needed at a given T.  The catalog file used must have enough lines in it to fully capture the partition function, or the result will not be accurate for Q.
	
def calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,T,catalog_file,vibs):

	'''
	Dynamically calculates a partition function whenever needed at a given T.  The catalog file used must have enough lines in it to fully capture the partition function, or the result will not be accurate for Q.  This is perfectly fine for the *relative* intensities of lines for a given molecule used by this program.  However, absolute intensities between molecules are not remotely accurate.  
	'''

	Q = np.float64(0.0) #Initialize a float for the partition function
	
	if 'acetone.cat' in catalog_file.lower():
	
		Q = 2.91296*10**(-7)*T**6 - 0.00021050085*T**5 + 0.05471337*T**4 - 5.5477*T**3 + 245.28*T**2 - 2728.3*T + 16431 #Hard code for Acetone
		
	elif 'sh.cat' in catalog_file.lower():
	
		Q = 0.000000012549467*T**4 - 0.000008528126823*T**3 + 0.002288160909445*T**2 + 0.069272946237033*T + 15.357239728157400
 #Hard code for SH.  Completely unreliable below 2.735 K or above 300 K.

	elif 'nh3.cat' in catalog_file.lower():
	
		Q = 0.11044*T**1.5025 + 2.5396
			
	elif 'methanol.cat' in catalog_file.lower() or 'ch3oh.cat' in catalog_file.lower() or 'ch3oh_v0.cat' in catalog_file.lower() or 'ch3oh_v1.cat' in catalog_file.lower() or 'ch3oh_v2.cat' in catalog_file.lower() or 'ch3oh_vt.cat' in catalog_file.lower():
	
		Q = 4.83410*10**-11*T**6 - 4.04024*10**-8*T**5 + 1.27624*10**-5*T**4 - 1.83807*10**-3*T**3 + 2.05911*10**-1*T**2 + 4.39632*10**-1*T -1.25670
		
	elif '13ch3oh.cat' in catalog_file.lower():
	
		Q = 0.000050130*T**3 + 0.076540934*T**2 + 4.317920731*T - 31.876881967
		
	elif 'c2n.cat' in catalog_file.lower() or 'ccn.cat' in catalog_file.lower():
		
		Q = 1.173755*10**(-11)*T**6 - 1.324086*10**(-8)*T**5 + 5.99936*10**(-6)*T**4 - 1.40473*10**(-3)*T**3 + 0.1837397*T**2 + 7.135161*T + 22.55770
		
	elif 'ch2nh.cat' in catalog_file.lower():
	
		Q = 1.2152*T**1.4863
		
	elif '13ch3oh.cat' in catalog_file.lower() or 'c033502.cat' in catalog_file.lower():
	
		Q = 0.399272*T**1.756329
		
	elif 'aceticacid' in catalog_file.lower():
	
		Q = 0.0009051494*T**3 + 2.3370894781*T**2 - 34.5494711437*T + 1110.8534245568
		
	elif 'methylformate' in catalog_file.lower() and '13' not in catalog_file.lower():
	
		Q = 3.29808*10**-8*T**5 - 2.59463*10**-5*T**4 + 5.80410*10**-3*T**3 + 1.60794*T**2 + 95.0922*T-328.468
		
	elif 'glycolaldehyde' in catalog_file.lower() and '13' not in catalog_file.lower():
	
		Q = 0.000501*T**3 + 0.562444*T**2 + 14.005379*T + 114.004177
		
	elif 'h2ccs' in catalog_file.lower():
	
		#Q = -0.00000000328379*T**5 + 0.000002934039*T**4 -0.001093668*T**3 + 0.315766*T**2 + 12.08729*T - 66.79358
		Q = 3.5655362887*T**1.5 -8.3747644
		
	elif 'ch3nh2' in catalog_file.lower():
	
		#Q = 23.82947*T**(1.50124) #JPL DATABASE VALUE
		Q = 5.957729*T**(1.501233) #Ilyushin 2014 paper value
		
	elif 'n2h+_hfs' in catalog_file.lower():
	
		Q = 3.2539 + 4.0233*T + 1.0014E-5*T**2		
		
	#GOTHAM PARTITION FUNCTIONS BELOW
	
	elif 'hcn.cat' in catalog_file.lower() and 'hfs' not in catalog_file.lower():
	
		Q = (0.92213*T**1.0836 + 4.3068)/3
		
		if T == CT:
		
			Q = 151.16
			
		elif T > 300:
		
			print('Warning: Extrapolating Q beyond 300 K for this molecule gets progressively iffier.')
			
		elif T < 5:
		
			print('Warning: Calculations for Q below 5 K are probably off by ~30%...')		
			
	elif 'hcn_hfs.cat' in catalog_file.lower():
	
		Q = (0.92213*T**1.0836 + 4.3068)
		
		if T == CT:
		
			Q = 453.4944
			
		elif T > 300:
		
			print('Warning: Extrapolating Q beyond 300 K for this molecule gets progressively iffier.')
			
		elif T < 5:
		
			print('Warning: Calculations for Q below 5 K are probably off by ~30%...')		
			
	elif 'nh2cn' in catalog_file.lower():
	
		if T > 50:
		
			Q = 2.0081*T**1.5972 - 259.42
			
		elif T < 50:
		
			Q = 0.81*T**1.7753 + 2.7549	
			
		if T > 300:
		
			print('Warning: Extrapolating Q beyond 300 K for this molecule gets progressively iffier.')
			
		elif T < 10:
		
			print('Warning: Extrapolating Q below 10 K for this molecule gets progressively iffier.')				

	elif 'nh2cho' in catalog_file.lower():
		
		Q = 5.5769*T**1.5 - 9.2166	
			
		if T > 300:
		
			print('Warning: Extrapolating Q beyond 300 K for this molecule gets progressively iffier.')
			
		elif T < 10:
		
			print('Warning: Extrapolating Q below 10 K for this molecule gets progressively iffier.  Error in Q at 9.375K is ~8%.')								
		
	elif 'hc13n' in catalog_file.lower():
	
		Q = 194.7950692278719*T + 0.1142372881295159
	
	elif 'hc11n' in catalog_file.lower():
	
		Q = 123.2554*T + 0.1381
		
	elif 'hc9n' in catalog_file.lower() and 'hfs' not in catalog_file.lower():
	
		Q = 71.7308577*T + 0.02203968
		
	elif 'hc9n' in catalog_file.lower() and 'hfs' in catalog_file.lower(): 
	
		Q = 3*71.7308577*T + 3*0.02203968
		
	elif 'hc7n' in catalog_file.lower() and 'hfs' not in catalog_file.lower():
	
		Q = 36.94999*T + 0.1356045
		
	elif 'hc7n' in catalog_file.lower() and 'hfs' in catalog_file.lower(): 
	
		Q = 3*36.94999*T + 3*0.1356045	
		
	elif 'hc5n' in catalog_file.lower() and 'hfs' not in catalog_file.lower():
	
		Q = 15.65419*T + 0.2214
		
	elif 'hc5n' in catalog_file.lower() and 'hfs' in catalog_file.lower(): 
	
		Q = 3*15.65419*T + 0.2214
		
	elif 'hc3n' in catalog_file.lower() and 'hfs' not in catalog_file.lower():
	
		Q = 4.581898*T + 0.2833
		
	elif 'hc3n' in catalog_file.lower() and 'hfs' in catalog_file.lower(): 
	
		Q = 3*4.581898*T + 0.2833
		
	elif 'hc2nc' in catalog_file.lower() and 'hfs' not in catalog_file.lower():
	
		Q = (12.58340*T + 1.0604)/3			
		
	elif 'hc2nc' in catalog_file.lower() and 'hfs' in catalog_file.lower():
	
		Q = 12.58340*T + 1.0604		
		
	elif 'hc4nc' in catalog_file.lower() and 'hfs' not in catalog_file.lower():
	
		Q = (44.62171*T + 0.6734)/3							
		
	elif 'hc4nc' in catalog_file.lower() and 'hfs' in catalog_file.lower():
	
		Q = 44.62171*T + 0.6734
		
	elif 'hc6nc' in catalog_file.lower() and 'hfs' not in catalog_file.lower():
	
		Q = (107.3126*T + 1.2714)/3		

	elif 'hc6nc' in catalog_file.lower() and 'hfs' in catalog_file.lower():
	
		Q = 107.3126*T + 1.2714
		
	elif 'propargylcyanide' in catalog_file.lower() or 'propargyl_cyanide' in catalog_file.lower():
	
		Q = 41.542*T**1.5008 + 1.5008
	
	elif 'propynal_barros' in catalog_file.lower():
	
		Q = 4.2770042*T**1.50582693 + 5.23724877
		
	elif 'thiopropynal' in catalog_file.lower():
	
		Q = 8.44164177*T**1.50477591 + 5.91902047
	
	elif 'pyrrole' in catalog_file.lower():
	
		Q = 27.727*T**1.4752
		
		if T == CT:
		
			Q = 58328.0735
			
		elif T > 40:
		
			print('Warning: Extrapolating Q beyond 40 K for this molecule gets progressively iffier.')	
			
	elif 'cyclopentadiene' in catalog_file.lower():
	
		Q = 9.7764*T**1.5 + 3.5246
	
		if T == CT:
		
			Q = 36251.5276	

		elif T > 40:
		
			print('Warning: Extrapolating Q beyond 40 K for this molecule gets progressively iffier.')
			
	elif '1-cyano-CPD' in catalog_file.lower():
	
		Q = 101.0674*T**1.5 + 23.9985
		
		if T == CT:
		
			Q = 374492.9985	

		elif T > 40:
		
			print('Warning: Extrapolating Q beyond 40 K for this molecule gets progressively iffier.')
			
	elif '2-cyano-CPD' in catalog_file.lower():
	
		Q = 102.0047*T**1.5 + 25.442
		
		if T == CT:
		
			Q = 376227.1111	
			
		elif T > 40:
		
			print('Warning: Extrapolating Q beyond 40 K for this molecule gets progressively iffier.')							
		
	elif 'cyclopropylcyanide_hfs' in catalog_file.lower():
	
		Q = 38.199*T**1.4975					
		
	elif 'pyridine' in catalog_file.lower():
	
		Q = 50.478*T**1.4955
		
	elif '1-cyanonapthalene' in catalog_file.lower():
	
		Q = 560.39*T**1.4984
		
		if T == CT:
		
			Q = 2108153.3871	
			
		elif T > 40:
		
			print('Warning: Extrapolating Q beyond 40 K for this molecule gets progressively iffier.')			
		
	elif '2-cyanonapthalene' in catalog_file.lower():
	
		Q = 562.57*T**1.4993
		
		if T == CT:
		
			Q = 2514761.8013	
			
		elif T > 40:
		
			print('Warning: Extrapolating Q beyond 40 K for this molecule gets progressively iffier.')	
		
	elif 'furan' in catalog_file.lower():
	
		Q = 33.725*T**1.4982
		
	elif 'phenol' in catalog_file.lower():
	
		Q = 264.20*T**1.4984
		
	elif 'benzaldehyde' in catalog_file.lower():
	
		Q = 53.798*T**1.4997
		
	elif 'anisole' in catalog_file.lower():
	
		Q = 54.850*T**1.4992
		
	elif 'azulene' in catalog_file.lower():
	
		Q = 96.066*T**1.4988
		
		if T == CT:
		
			Q = 383985.5935	
			
		elif T > 40:
		
			print('Warning: Extrapolating Q beyond 40 K for this molecule gets progressively iffier.')			
		
	elif 'acenaphthene' in catalog_file.lower():
	
		Q = 160.9183316*T**1.5 + 4.987858
		
		if T == CT:
		
			Q = 713664.5114	
			
		elif T > 40:
		
			print('Warning: Extrapolating Q beyond 40 K for this molecule gets progressively iffier.')		
		
	elif 'acenaphthylene' in catalog_file.lower():
	
		Q = 150.8688*T**1.5 + 12.1748
		
		if T == CT:
		
			Q = 597794.8438	
			
		elif T > 40:
		
			print('Warning: Extrapolating Q beyond 40 K for this molecule gets progressively iffier.')	
		
	elif 'fluorene' in catalog_file.lower():
	
		Q = 219.170*T**1.5 + 4.34551
		
		if T == CT:
		
			Q = 1031274.8660	
			
		elif T > 40:
		
			print('Warning: Extrapolating Q beyond 40 K for this molecule gets progressively iffier.')	
			
	elif 'benzonitrile' in catalog_file.lower():
	
		Q = 25.896*T**1.4998 + 0.38109
		
		if T == CT:
		
			Q = 129198.481
			
		elif T > 60:
		
			print('Warning: Extrapolating Q beyond 60 K for this molecule gets progressively iffier.')
			
	elif 'cyanoketene' in catalog_file.lower():
	
		Q = 11.5469451*T**1.5
		
		if T == CT:
		
			Q = 59826.2
			
		elif T > 300:
		
			print('Warning: Extrapolating Q beyond 300 K for this molecule gets progressively iffier.')	
			
	elif 'thioaa' in catalog_file.lower():
	
		Q = -150.699838 + 36.00734443162136*T + 0.5803430978982798*T**2 - 0.0002443275581575372*T**3 + 1.058207372030315e-05*T**4 - 2.989232438909946e-08*T**5 + 2.857876456157256e-11*T**6
		
		print('Warning: This partition function *includes* the vt=1 first excited torsional state.  Extrapolation below 10 K or above 300 K is dangerous as well.')
		
	else:
	
		nstates = elower.size #Determine the number of total states in the raw cat file
	
		combined_array = np.empty(shape=(nstates,qns+1)) #Set up an array that has [J, ka, kc, Elower]

		if (qns == 1):
	
			for i in range(nstates): #Fill that array with [J, ka, kc, Elower]
		
				combined_array[i][0] = qn7[i]
				combined_array[i][1] = elower[i]

		if (qns == 2):
	
			for i in range(nstates): #Fill that array with [J, ka, kc, Elower]
		
				combined_array[i][0] = qn7[i]
				combined_array[i][1] = qn8[i]
				combined_array[i][2] = elower[i]
	
		if (qns == 3):
	
			for i in range(nstates): #Fill that array with [J, ka, kc, Elower]
		
				combined_array[i][0] = qn7[i]
				combined_array[i][1] = qn8[i]
				combined_array[i][2] = qn9[i]
				combined_array[i][3] = elower[i]
			
		if (qns == 4):
	
			for i in range(nstates): #Fill that array with [J, ka, kc, QN10, Elower]
		
				combined_array[i][0] = qn7[i]
				combined_array[i][1] = qn8[i]
				combined_array[i][2] = qn9[i]
				combined_array[i][3] = qn10[i]
				combined_array[i][4] = elower[i]			

		if (qns == 5):
	
			for i in range(nstates): #Fill that array with [J, ka, kc, QN10, QN11, Elower]
		
				combined_array[i][0] = qn7[i]
				combined_array[i][1] = qn8[i]
				combined_array[i][2] = qn9[i]
				combined_array[i][3] = qn10[i]
				combined_array[i][4] = qn11[i]
				combined_array[i][5] = elower[i]	
			
		if (qns == 6):
	
			for i in range(nstates): #Fill that array with [J, ka, kc, QN10, QN11, QN12, Elower]
		
				try:
					combined_array[i][0] = qn7[i]
				except ValueError:
					print('I choked at index {}.' .format(i))
					quit()
				combined_array[i][1] = qn8[i]
				combined_array[i][2] = qn9[i]
				combined_array[i][3] = qn10[i]
				combined_array[i][4] = qn11[i]
				combined_array[i][5] = qn12[i]
				combined_array[i][6] = elower[i]									
		
		temp = list(set(map(tuple,combined_array))) #Don't know HOW this works, but it does: sorts through the array and removes all duplicate entries, so that only a single entry remains for each set of quantum numbers.
	
		ustates = len(temp) #Number of unique lower states

	
		for i in range(ustates):
	
			J = temp[i][0] #Extract a J value from the list
			E = temp[i][qns] #Extract its corresponding energy
			
			#if 'benzonitrile.cat' in catalog_file.lower() or 'bn_global.cat' in catalog_file.lower() or 'benzonitrile' in catalog_file.lower():
			
				#Goes through and does the adjustments for the nuclear hyperfine splitting (it's being overcounted in the catalog, needs to be divide by 3), and the spin-statistic degeneracies.
			
			#	if (temp[i][1] % 2 == 0):
				
			#		 Q += (1/3)*(5/8)*(2*J+1)*exp(np.float64(-E/(kcm*T)))
					 
			#	else:
				
			#		Q += (1/3)*(3/8)*(2*J+1)*exp(np.float64(-E/(kcm*T)))					
					
			#else:
		
			Q += (2*J+1)*exp(np.float64(-E/(kcm*T))) #Add it to Q
		
		if 'h2cco' in catalog_file.lower() or 'ketene' in catalog_file.lower():
		
			Q *= 2
			
		if 'glycine' in catalog_file.lower() and 'hfs' not in catalog_file.lower():
	
			Q *= 3		
			
		if 'alanine' in catalog_file.lower() and 'hfs' not in catalog_file.lower():
		
			Q *= 3		
			
		#result = [Q,ustates] #can enable this function to check the number of states used in the calculation, but note that this will break calls to Q further down that don't go after element 0.
		
	qvib = calc_qvib(vibs,T)
	
	Q *= qvib
	
	if 'hydroxyacetone' in catalog_file.lower() and 'dihydroxyacetone' not in catalog_file.lower():
	
		Q *= 2
	
	return Q
	
		
#calc_qvib calculates the vibrational contribution to the partition function.

def calc_qvib(vibs,T):

	if vibs == None:
	
		qvib = 1

	else:
	
		qvib = 1
	
		for x in vibs:
		
			qvib_x = 0
		
			for y in range(100):
			
				qvib_x += (np.exp(-x*y/(0.695*T)))
				
			qvib *= qvib_x
			
	return qvib	


#scale_temp scales the simulated intensities to the new temperature

def scale_temp(int_sim,qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,T,CT,catalog_file):

	'''
	Converts linear intensities at one temperature to another.
	'''

	scaled_int = np.copy(int_sim)
	
	scaled_int *= 0.0
	
	Q_T = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,T,catalog_file,vibs)
	Q_CT = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,CT,catalog_file)

	
	scaled_int = int_sim * (Q_CT/Q_T) * (CT/T) * exp(-(((1/T)-(1/CT))*elower)/0.695)
	
# 	for i in range(len(scaled_int)):
# 	
# 		if catalog[1][i] > error_limit:
# 		
# 			scaled_int[i] = 0.0

	return scaled_int

#convert_int converts the intensity to not log units

def convert_int(logint):

	'''
	Converts catalog logarithmic intensity units to linear ones
	'''

	intensity = np.copy(logint)	
	
	intensity = 10**(logint)
	
	return intensity
	
#simulates Gaussian profiles after intensities are simulated.			

def sim_gaussian(int_sim,freq,linewidth):

	'''
	Simulates Gaussian profiles for lines, after the intensities have been calculated.  Tries to be smart with how large a range it simulates over, for computational resources.  Includes a thermal cutoff for optically-thick lines.
	'''
	
	freq_gauss_tmp = []
	
	x = 0
	
	if cavity_ftmw == True:
	
		linewidth = cavity_dV

	while (x < len(int_sim)):
	
		l_f = linewidth*freq[x]/ckm #get the FWHM in MHz
	
		if two_fwhm_only is True:
		
			min_f = freq[x] - 2*l_f #get the frequency 10 FWHM lower
		
			max_f = freq[x] + 2*l_f #get the frequency 10 FWHM higher		
		
		else:
	
			min_f = freq[x] - 10*l_f #get the frequency 10 FWHM lower
		
			max_f = freq[x] + 10*l_f #get the frequency 10 FWHM higher
		
		if x < len(int_sim)-2:
		
			while (freq[x+1] < max_f and x < len(int_sim)-2):
		
					x += 1
			
					max_f = freq[x] + 10*l_f #get the frequency 10 FWHM higher
					
		if match_obs is True:			
	
			l_idx = find_nearest(freq_obs,min_f)
			u_idx = find_nearest(freq_obs,max_f)
			
			freq_line = np.asarray(freq_obs[l_idx:u_idx])
			
		else:
		
			freq_line = np.arange(min_f,max_f,res) #generate a chunk of spectra at resolution res	
	
		freq_gauss_tmp.extend(freq_line)
		
		x+= 1
	
	freq_gauss_tmp.sort()
	
	freq_gauss = np.asarray(freq_gauss_tmp)
	
	int_gauss = np.copy(freq_gauss)
	
	int_gauss *= 0.0
	
	start_time = tm.time()
	
	alerted = False

	for x in range(int_sim.shape[0]):
	
		if abs(int_sim[x]) < rms/10:
		
			continue
	
		telapsed = tm.time() - start_time
		
		if telapsed > 5 and alerted == False and quietflag == False:
		
			tstep = telapsed/x
			
			ttotal = (tstep * int_sim.shape[0])/60
		
			print('\nYou have asked for a computationally-expensive simulation.  Either wait for it to finish, narrow up your frequency range by setting ll or ul, or reduce the resolution.  Use quiet() to suppress further messages.\n')
			
			if ttotal < 0.2:
			
				print('Your simulation will probably finish in: a few seconds.')
				
			elif ttotal < 1.0:
			
				print('Your simulation will probably finish in: a minute or three.')
				
			elif ttotal < 2.0:
			
				print('Your simulation will probably finish in: a few minutes.')		
				
			elif ttotal < 5.0:
			
				print('Your simulation will probably finish in: go get a cup of coffee.')
				
			elif ttotal < 10.0:
			
				print('Your simulation will probably finish in: work on something else for a while.')
				
			elif ttotal < 30.0:
			
				print('Your simulation will probably finish in: watch an episode or two on Netflix.')	
				
			else:
			
				print('Your simulation will probably finish in: press ctrl+c and set some limits or lower the resolution.')																			
																						
			
			alerted = True 
	
		l_f = linewidth*freq[x]/ckm #get the FWHM in MHz
	
		c = l_f/2.35482
		
		#if set up to simulate cavity FTMW doppler doublets, divide the intensity by half, shift each line up and down by the splitting, and make sure to use the correct fwhm.
		
		if cavity_ftmw == True:
		
			freq_l = freq[x] - (cavity_split*freq[x]/ckm)
			freq_h = freq[x] + (cavity_split*freq[x]/ckm)
			
			int_gauss += 0.5*int_sim[x]*exp(-((freq_gauss - freq_l)**2/(2*c**2)))
			int_gauss += 0.5*int_sim[x]*exp(-((freq_gauss - freq_h)**2/(2*c**2)))
						
		else:

			int_gauss += int_sim[x]*exp(-((freq_gauss - freq[x])**2/(2*c**2)))
	
	Tbg = calc_tbg(tbg_params,tbg_type,tbg_range,freq_gauss)
	
	J_T = (h*freq_gauss*10**6/k)*(np.exp(((h*freq_gauss*10**6)/(k*T))) -1)**-1
	J_Tbg = (h*freq_gauss*10**6/k)*(np.exp(((h*freq_gauss*10**6)/(k*Tbg))) -1)**-1
	
	int_gauss_tau = (J_T - J_Tbg)*(1 - np.exp(-int_gauss))/eta
	
	return(freq_gauss,int_gauss_tau)

#write_spectrum writes out the current freq and intensity to output_file

def write_spectrum(x,output_file):

	'''
	Will write out the simulation frequency and intensity for the currently active simulation ('current'), the summed spectrum from sum_stored() ('sum'), or any stored simulation 'x' to output_file (which must be given as a string).  
	'''
	
	if x == 'current':
	
		freq_tmp = freq_sim
		int_tmp = int_sim
		
	elif x == 'sum':
	
		freq_tmp = freq_sum
		int_tmp = int_sum
		
	elif x == 'residual':
	
		freq_tmp = freq_resid
		int_tmp = int_resid
		
	elif x == 'obs':
	
		freq_tmp = freq_obs
		int_tmp = int_obs
		
	elif x == 'stacked':
	
		freq_tmp = vel_stacked
		int_tmp = int_stacked
		
	elif x == 'tbg':
	
		freq_tmp = freq_sim
		int_tmp = tbg
		
	elif x == 'stacked_sim':
	
		freq_tmp = vel_sim_stacked
		int_tmp = int_sim_stacked
		
	elif x == 'mf':
	
		freq_tmp = velocity_mf
		int_tmp = intensity_mf
				
	else:
	
		try:
			freq_tmp = sim[x].freq_sim
		except KeyError:
			print('\nOops: A spectrum called {} does not exist.  Either correct the typo (type sim to see the current spectrum labels), or remember that the syntax is write_spectrum(label,outputfile)' .format(x))
			return 
		int_tmp = sim[x].int_sim

	if gauss == True:
			
		with open(output_file, 'w') as output: 
					
			output.write('{} {}\n' .format(freq_tmp[0],int_tmp[0]))
				
		with open(output_file, 'a') as output:
					
			for h in range(len(freq_tmp)):
			
				output.write('{} {}\n' .format(freq_tmp[h],int_tmp[h]))					
	
	else:
				
		with open(output_file, 'w') as output: 
					
			output.write('{} {}\n' .format(freq_tmp[0],int_tmp[0]))
				
		with open(output_file, 'a') as output:
		
			for h in range(freq_tmp.shape[0]):
					
				output.write('{} {}\n' .format(freq_tmp[h],int_tmp[h]))									

#apply_beam applies a beam dilution correction factor

def apply_beam(frequency,intensity,source_size,dish_size,synth_beam,interferometer):

	if interferometer is False:

		#create a wave to hold wavelengths, fill it to start w/ frequencies

		wavelength = np.copy(frequency)
	
		#Convert those frequencies to Hz
	
		wavelength *= 1.0E6
	
		#Convert to meters
	
		wavelength = cm/wavelength
	
		#create an array to hold beam sizes
	
		beam_size = np.copy(wavelength)
	
		#fill it with beamsizes
	
		beam_size *= 206265 * 1.22 / dish_size
	
		#create an array to hold beam dilution factors
	
		dilution_factor = np.copy(beam_size)
	
		dilution_factor = source_size**2/(beam_size**2 + source_size**2)
		
		intensity_diluted = np.copy(intensity)
	
		intensity_diluted *= dilution_factor
	
		return intensity_diluted
		
	if interferometer is True:
	
		#create a wave to hold wavelengths, fill it to start w/ frequencies

		wavelength = np.copy(frequency)
	
		#Convert those frequencies to Hz
	
		wavelength *= 1.0E6
	
		#Convert to meters
	
		wavelength = cm/wavelength
	
		#create an array to hold beam sizes
	
		beam_size = np.ones_like(wavelength)
	
		#fill it with beamsizes
	
		beam_size *= (synth_beam[0]+synth_beam[1])/2
	
		#create an array to hold beam dilution factors
	
		dilution_factor = np.copy(beam_size)
	
		dilution_factor = source_size**2/(beam_size**2 + source_size**2)
	
		intensity_diluted = np.copy(intensity)
	
		intensity_diluted *= dilution_factor
	
		return intensity_diluted
		
def get_beam(frequency,dish_size):

	#Convert those frequencies to Hz

	hz_freqs = 1.0E6*frequency

	#Convert to meters

	wavelength = cm/hz_freqs

	#create an array to hold beam sizes

	beam_size = wavelength * 206265 * 1.22 / dish_size
	
	return beam_size	

#invert_beam applies a beam dilution correction factor in the other direction (used for tbg corrections - takes an observed tbg and returns what it actually should be un-diluted)

def invert_beam(frequency,intensity,source_size,dish_size):

	#create a wave to hold wavelengths, fill it to start w/ frequencies

	wavelength = np.copy(frequency)
	
	#Convert those frequencies to Hz
	
	wavelength *= 1.0E6
	
	#Convert to meters
	
	wavelength = cm/wavelength
	
	#create an array to hold beam sizes
	
	beam_size = np.copy(wavelength)
	
	#fill it with beamsizes
	
	beam_size *= 206265 * 1.22 / dish_size
	
	#create an array to hold beam dilution factors
	
	dilution_factor = np.copy(beam_size)
	
	dilution_factor = source_size**2/(beam_size**2 + source_size**2)
	
	intensity_undiluted = np.copy(intensity)
	
	intensity_undiluted /= dilution_factor
	
	return intensity_undiluted	
	
#run_sim runs the simulation.  It's a meta routine, so that we can update later

def run_sim(freq,intensity,T,dV,C,tau_get=None):

	'''
	Runs a full simulation accounting for the currently-active T, dV, S, and vlsr values, as well as any thermal cutoff for optically-thick lines
	'''
	
	np.seterr(under='ignore')
	np.seterr(over='ignore')
	
	Q = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,T,catalog_file,vibs)
	
	Nl = C * glow * np.exp(-elower/(0.695 * T)) / Q
	
	nl_print = trim_array(Nl,frequency,ll,ul)
	
# 	print('Nl\n')
# 	print(nl_print)
# 	print('+++++++++++++++++++++\n')
	
	tau_numerator = np.asarray((ccm/(frequency * 10**6))**2 * aij * gup * Nl * (1 - np.exp(-(h * frequency * 10**6)/(k * T))),dtype=float)
	
	tau_numerator_print = trim_array(tau_numerator,frequency,ll,ul)
	
# 	print('tau_num\n')
# 	print(tau_numerator_print)
# 	print('+++++++++++++++++++++\n')
	
	tau_denominator = np.asarray(8 * np.pi * (dV * frequency * 10**6 / ckm) * glow,dtype=float)
	
	tau_denominator_print = trim_array(tau_denominator,frequency,ll,ul)
	
# 	print('tau_denom\n')
# 	print(tau_denominator_print)
# 	print('+++++++++++++++++++++\n')		

	tau =tau_numerator/tau_denominator
	
# 	idx = find_nearest(frequency,19174.0845)
# 	
# 	print('Q: {}' .format(Q))
# 	print('C: {}' .format(C))
# 	print('Glow: {}' .format(glow[idx]))
# 	print('elower: {} cm-2' .format(elower[idx]))
# 	print('T: {}' .format(T))
# 	print('ccm: {}' .format(ccm))
# 	print('Frequency: {}' .format(frequency[idx]))
# 	print('QNs: {} {} - {} {}' .format(qn1[idx],qn2[idx],qn7[idx],qn8[idx]))
# 	print('aij: {}' .format(aij[idx]))
# 	print('gup: {}' .format(gup[idx]))
# 	print('Nl: {}' .format(Nl[idx]))
# 	print('h: {}' .format(h))
# 	print('k: {}' .format(k))
# 	print('tau_numerator: {}' .format(tau_numerator[idx]))
# 	print('dV: {}' .format(dV))
# 	print('ckm: {}' .format(ckm))
# 	print('tau_denominator: {}' .format(tau_denominator[idx]))
# 	print('tau: {}' .format(tau[idx]))
		
	if tau_get is not None:
	
		idx = find_nearest(frequency,tau_get)
		
		print('The tau for the line at frequency {} is {}.' .format(frequency[idx],tau[idx]))
	
	tau_print = trim_array(tau,frequency,ll,ul)
	
# 	print('tau\n')
# 	print(tau_print)
# 	print('+++++++++++++++++++++\n')
	
	int_temp = tau
		
	int_temp = trim_array(int_temp,frequency,ll,ul)		
	
	freq_tmp = trim_array(freq,frequency,ll,ul)
	
# 	with open('tau_vlsr_{:.3f}.txt' .format(vlsr), 'w') as output:
# 	
# 		for x in range(len(freq_tmp)):
# 		
# 			output.write('{:.4f} {:.4f}\n' .format(freq_tmp[x],int_temp[x]))

#	removed because dilution should be calculated as an end step	
# 	if planck is False:
# 	
# 		int_temp = apply_beam(freq_tmp,int_temp,source_size,dish_size,synth_beam,interferometer)
		
	int_tau = np.copy(int_temp)		
	
	# with open('{}_tau_stick.txt' .format(vlsr), 'w') as output:
# 	
# 		freq_prt = np.copy(freq_tmp)
# 		
# 		freq_prt -= (-vlsr)*freq_prt/ckm
# 	
# 		for x in range(len(int_tau)):
# 		
# 			output.write('{} {}\n' .format(freq_prt[x],int_tau[x]))		
	
	if gauss == True:

		freq_sim,int_sim = sim_gaussian(int_temp,freq_tmp,dV)
		
	else:
	
		
		freq_sim = freq_tmp

		Tbg = calc_tbg(tbg_params,tbg_type,tbg_range,freq_sim)
		
		J_T = (h*freq_sim*10**6/k)*(np.exp(((h*freq_sim*10**6)/(k*T))) -1)**-1
		J_Tbg = (h*freq_sim*10**6/k)*(np.exp(((h*freq_sim*10**6)/(k*Tbg))) -1)**-1
		
		int_sim = (J_T - J_Tbg)*(1 - np.exp(-int_temp))/eta
		
	int_sim = apply_beam(freq_sim,int_sim,source_size,dish_size,synth_beam,interferometer)
	
	# with open('{}_tb_stick.txt' .format(vlsr), 'w') as output:
# 	
# 		freq_prt = np.copy(freq_sim)
# 		
# 		freq_prt -= (-vlsr)*freq_prt/ckm
# 	
# 		for x in range(len(int_sim)):
# 		
# 			output.write('{} {}\n' .format(freq_prt[x],int_sim[x]))	
				
	if planck == True:
	
		#calculate the beam solid angle, and throw an error if it hasn't been set.
		
		try:
			omega = synth_beam[0]*synth_beam[1] #the conversion below already has the volume element built in
			#omega = synth_beam[0]*synth_beam[1]*np.pi/(4*np.log(2))	
		except TypeError:
			print('You need to set a beam size to use for this conversion with synth_beam = [bmaj,bmin]')
			print('Your simulation is still in Kelvin.')
			return freq_sim,int_sim
		
		#create an array that's a copy of int_sim to work with temporarily
			
		int_jansky = np.copy(int_sim)
		
		#create a mask so we only work on non-zero values
		
		mask = int_jansky != 0
		
		#do the conversion
		
		int_jansky[mask] = (3.92E-8 * (freq_sim[mask]*1E-3)**3 *omega/ (np.exp(0.048*freq_sim[mask]*1E-3/int_sim[mask]) - 1))
		
		int_sim = int_jansky
		
	return freq_sim,int_sim,int_tau
	
#check_Q prints out Q at a given temperature x

def check_Q(x):

	Q = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,x,catalog_file,vibs)
	
	print('Q({}) = {:.0f}' .format(x,Q))
	
#get_Q returns Q at a given temperature x

def get_Q(x):

	return calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,x,catalog_file,vibs)
	
#check_Qvib prints out Qvib at a given temperature x

def check_Qvib(x):

	Qvib = calc_qvib(vibs,x)
	
	print('Q({}) = {:.5f}' .format(x,Qvib))
	
#get_Qvib returns Qvib at a given temperature x

def get_Qvib(x):

	return calc_qvib(vibs,x)
	
#check_Qrot prints out Qrot at a given temperature x

def check_Qrot(x):

	Qrot = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,x,catalog_file,vibs=None)
	
	print('Q({}) = {:.0f}' .format(x,Qrot))
	
#get_Qrot returns Qrot at a given temperature x

def get_Qrot(x):

	return calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,x,catalog_file,vibs=None)	

#trim_array trims any given input array to the specified frequency ranges

def trim_array(array,frequency,ll,ul):

	'''
	trims any given input array to the specified frequency ranges.  Ranges can be 
	'''

	if type(ll) == int or type(ll) == float:
	
		tmp_ll = [ll]
		tmp_ul = [ul]
		
	else:
	
		tmp_ll = list(ll)
		tmp_ul = list(ul)
	
	foo = 0
	
	trimmed_array = np.array([])
		
	for z in range(len(tmp_ll)):
	
		try:
			i = np.where(frequency > tmp_ll[z])[0][0] 	#get the index of the first value above the lower limit
		except IndexError:
			if frequency[-1] < tmp_ll[z]:
				continue
			else:
				i = 0									#if the catalog begins after the lower limit
			
		try:
			i2 = np.where(frequency > tmp_ul[z])[0][0]	#get the index of the first value above the upper limit	
		except IndexError:
			i2 = len(frequency)							#if the catalog ends before the upper limit is reached		
			
		if foo == 0:
		
			trimmed_array = np.copy(array[i:i2])
			foo = 1
			
		else:
		
			trimmed_array = np.append(trimmed_array,array[i:i2])
	
	return trimmed_array		
	
#mod_T changes the temperature, re-simulates, and re-plots	
	
def modT(x):

	'''
	Modifies the temperature value of the current simulation to the value given
	'''

	try:
		float(x)
	except:
		print('x needs to be a number.')
		return
		
	global T,freq_sim,int_sim,int_tau
		
	T = float(x)
		
	freq_tmp = np.copy(frequency)
	
	freq_tmp += (-vlsr)*freq_tmp/ckm		
	
	freq_sim,int_sim,int_tau = run_sim(freq_tmp,intensity,T,dV,C)
	
	clear_line('current')
	
	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)
		
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
	fig.canvas.draw()
	
	save_results('last.results')
	
#modC changes the column density, re-simulates, and re-plots

def modC(x):

	'''
	Modifies the scaling value of the current simulation to the value given
	'''

	try:
		float(x)
	except:
		print('x needs to be a number.')
		return
		
	global C,freq_sim,int_sim,int_tau
		
	C = x
		
	freq_tmp = np.copy(frequency)
	
	freq_tmp += (-vlsr)*freq_tmp/ckm		
	
	freq_sim,int_sim,int_tau = run_sim(freq_tmp,intensity,T,dV,C)
	
	clear_line('current')
	
	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)
		
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
	fig.canvas.draw()
	
	save_results('last.results')
	
#modS is a legacy command from viewspectrum.py that just calls modC, the new command.

def modS(x):

	modC(x)

#moddV changes the velocity width, re-simulates, and re-plots

def moddV(x):

	'''
	Modifies the dV value of the current simulation to the value given
	'''

	try:
		float(x)
	except:
		print('dV needs to be a number.')
		return
		
	global dV,freq_sim,int_sim,int_tau
	
	dV = x
	
	freq_tmp = np.copy(frequency)
	
	freq_tmp += (-vlsr)*freq_tmp/ckm		
		
	freq_sim,int_sim,int_tau = run_sim(freq_tmp,intensity,T,dV,C)
	
	clear_line('current')
	
	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)
		
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
	fig.canvas.draw()
	
	save_results('last.results')
	
#modVLSR changes the LSR velocity, re-simulates, and re-plots

def modVLSR(x):

	'''
	Modifies the vlsr value of the current simulation to the value given
	'''

	try:
		float(x)
	except:
		print('vlsr needs to be a number.')
		return	
		
	global vlsr,freq_sim,int_sim,frequency,int_tau
	
	vlsr = x
		
	freq_tmp = np.copy(frequency)
	
	freq_tmp += (-vlsr)*freq_tmp/ckm		
	
	freq_sim,int_sim,int_tau = run_sim(freq_tmp,intensity,T,dV,C)
	
	clear_line('current')
	
	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)
	
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
	fig.canvas.draw()
	
	save_results('last.results')		

#modV is an alias for modVLSR

def modV(vlsr):

	'''
	Modifies the vlsr value of the current simulation to the value given
	'''

	modVLSR(vlsr)	
	
#make_plot makes the plot!

def make_plot():

	'''
	Generates a plot of the currently-active molecular simulation, as well as any laboratory data or observations which are loaded.  This will *not* restore any overplots from a previously-closed plot.
	'''

	global fig,ax,line1,line2

	plt.ion()	

	fig = plt.figure()
	ax = fig.add_subplot(111)

	minorLocator = AutoMinorLocator(5)
	plt.xlabel('Frequency (MHz)')
	
	if planck is False:
	
		plt.ylabel('Intensity (K)')
		
	else:
	
		plt.ylabel('Intensity (Jy/beam)')
	
	#make the title latex friendly:
	
	title_str = obs_name.replace('_','\_')
	
	plt.title(title_str)
	plt.rcParams['axes.labelweight'] = 'bold'

	plt.locator_params(nbins=4) #Use only 4 actual numbers on the x-axis
	ax.xaxis.set_minor_locator(minorLocator) #Let the program calculate some minor ticks from that

	ax.get_xaxis().get_major_formatter().set_scientific(False) #Don't let the x-axis go into scientific notation
	ax.get_xaxis().get_major_formatter().set_useOffset(False)
	
	try:
		freq_obs[0]
		lines['obs'] = ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0,drawstyle=draw_style)
	except:
		pass

	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)	

	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
		
	fig.canvas.draw()
	
# 	if labels_flag == True:
# 	
# 		labels_on()
	
	save_results('last.results')
	
#obs_off turns off the observations
	
def obs_off():

	'''
	turns off the laboratory data or observations on the plot
	'''

	try:
		clear_line('obs')
		save_results('last.results')
		return
	except:
		print('The observations are already off.  You can turn them on with obs_on()')
		return
	
	try:
		lines['obs']
		save_results('last.results')
	except:
		print('There are no observations loaded into the program to turn off.  Load in obs with read_obs()')
		return	
			
#obs_on turns on the observations
	
def obs_on():

	'''
	turns on the laboratory data or observations on the plot
	'''	
	
	try:
		lines['obs']
		return
	except:
		pass

	try:
		lines['obs'] = 	ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			ax.legend(loc='upper right')
		fig.canvas.draw()
		save_results('last.results')
	except:
		print('There are no observations loaded into the program to turn on.  Load in obs with read_obs()')
		return
			
#read_obs reads in observations or laboratory spectra and populates freq_obs and int_obs.  Special logic for .npz files.

def read_obs(x):

	'''
	reads in observations or laboratory spectra and populates freq_obs and int_obs.  will detect a standard .ispec header from casaviewer export, and will apply a GHz flag if necessary, as well as populating the coords variable with the coordinates from the header.  Will detect a .npz file and act accordingly as well, but only if the keywords of the .npz file are freq_obs and int_obs
	'''

	global spec, coords, GHz, res, obs_name, draw_style, freq_obs,int_obs

	spec = x

	#check if this is an *.npz file and if so, load it up and slot it in
	
	if x[-3:] == 'npz':
	
		#load the file into a temporary data variable
	
		data = np.load(x)
		
		#the keywords must be freq_obs and int_obs, and we can then slot them in.
		
		
		
		freq_obs = data['freq_obs']
		int_obs = data['int_obs']

	else:

		obs = read_cat(x)
	
		#check to see if these are casa spectra
	
		if obs[0].split(':')[0] == '#title':
	
			i = 0
			j = 0
	
			while i == 0:
		
				if obs[j].split(':')[0] == '#xLabel':
		
					if obs[j].split('[')[1].strip(']\n') == 'GHz':
				
						GHz = True
				
				if obs[j].split(':')[0] == '#region (world)':
			
					coords = obs[j].split(':')[1].strip('\n')
			
				if obs[j][0] != '#':
			
					i = 1
				
				j += 1		
		
			del obs[:j+1]
	
		#global freq_obs,int_obs

		freq_obs = []
		int_obs = []

		for x in range(len(obs)):

			freq_obs.append(float(obs[x].split()[0]))
			int_obs.append(float(obs[x].split()[1].strip('\n')))
	
		freq_tmp = list(freq_obs)
		int_tmp = list(int_obs)
	
		freq_obs = [freq_tmp for freq_tmp,int_tmp in sorted(zip(freq_tmp,int_tmp))]
		int_obs = [int_tmp for freq_tmp,int_tmp in sorted(zip(freq_tmp,int_tmp))]	
		
	if GHz == True:
	
		freq_obs[:] = [x*1000.0 for x in freq_obs]
		
	res = abs(freq_obs[1]-freq_obs[0])	
	
	if res == 0.0:
	
		res = abs(freq_obs[2]-freq_obs[1])
		
	if res == 0.0:
	
		print('First three frequency data points for the observations are identical; resolution could not be automatically determined and has been set to 10 kHz by default.  Modify this by issuing res = X, where X is the desired resolution in MHz')
	
	clear_line('obs')
		
	try:		
		lines['obs'] = 	ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0, drawstyle=draw_style)
	except:
		return
		
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
	fig.canvas.draw()	
	
	if len(spec.split('.')) > 1:
		
		tmp_str = str(spec.split('.')[-1])
	
		obs_name = str(spec.strip(tmp_str).strip('.').split('/')[-1])
		
	else: 
	
		obs_name = str(spec)
		
#close closes the currently open plot

def close():

	'''
	Closes the currently open plot window.
	'''
	
	plt.close()	
	
	save_results('last.results')
	
#store saves the current simulation parameters for recall later.  *Not* saved as a Gaussian. 'x' must be entered as a string with quotes.

def store(x=None):

	'''
	saves the current simulation parameters for recall later.  *Not* saved as a Gaussian. 'x' must be entered as a string with quotes.  If used with just store(), it defaults the name to everything up to the first period in the catalog_file name.
	'''
	
	if x == None:
	
		x = '{}' .format(catalog_file.split('.')[0].strip('\n').split('/')[-1]) 
	
	sim[x] = Molecule(x,catalog_file,tag,gup,glow,dof,error,qn1,qn2,qn3,qn4,qn5,qn6,elower,eupper,qns,logint,qn7,qn8,qn9,qn10,qn11,qn12,C,dV,T,CT,vlsr,frequency,freq_sim,intensity,int_sim,int_tau,aij,sijmu,vibs)
	
	if auto_update == True:
	
		sum_stored()
		overplot_sum()
		
	if any(line for line in ax.lines if line.get_label()==x):
		overplot(x)	
	
	save_results('last.results') 
	
#recall wipes the current simulation and re-loads a previous simulation that was stored with store(). 'x' must be entered as a string with quotes. This will close the currently-open plot.

def recall(x):

	'''
	wipes the current simulation and re-loads a previous simulation that was stored with store(). 'x' must be entered as a string with quotes. This will close the currently-open plot.
	'''

	save_results('last.results')

	global elower,eupper,qns,logint,qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8,qn9,qn10,qn11,qn12,S,dV,T,vlsr,frequency,freq_sim,intensity,int_sim,current,catalog_file,sijmu,C,tag,gup,error,aij,vibs,int_tau
	
	current = sim[x].name
	elower = sim[x].elower
	eupper = sim[x].eupper
	
	qns = sim[x].qns
	logint = sim[x].logint
	gup = sim[x].gup
	tag = sim[x].tag
	error = sim[x].error
	qn1 = sim[x].qn1
	qn2 = sim[x].qn2
	qn3 = sim[x].qn3
	qn4 = sim[x].qn4
	qn5 = sim[x].qn5
	qn6 = sim[x].qn6
	qn7 = sim[x].qn7
	qn8 = sim[x].qn8
	qn9 = sim[x].qn9
	qn10 = sim[x].qn10
	qn11 = sim[x].qn11
	qn12 = sim[x].qn12
	C = sim[x].C
	dV = sim[x].dV
	T = sim[x].T
	vlsr = sim[x].vlsr
	frequency = sim[x].frequency
	freq_sim = sim[x].freq_sim
	intensity = sim[x].intensity
	int_sim = sim[x].int_sim
	catalog_file = sim[x].catalog_file
	aij = sim[x].aij
	sijmu = sim[x].sijmu
	vibs = sim[x].vibs

	try:
		clear_line('current')
	except:
		pass
		
	tmp_freq = np.copy(frequency)
	
	tmp_freq += (-vlsr)*tmp_freq/ckm
	
	Q = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,CT,catalog_file,vibs)
	
	freq_sim,int_sim,int_tau = run_sim(tmp_freq,intensity,T,dV,C)	
		
	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)	
		
	try:
		plt.get_fignums()[0]	
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			ax.legend(loc='upper right')
		fig.canvas.draw()
	except:	
		make_plot()	
		
	save_results('last.results')	
	
#overplot overplots a previously-stored simulation on the current plot in a color other than red, but does not touch the simulation active in the main program. 'x' must be entered as a string with quotes.

def overplot(x,cchoice=None,thick=1.0,line_style='default'):

	'''
	overplots a previously-stored simulation on the current plot in a color other than red, but does not touch the simulation active in the main program. 'x' must be entered as a string with quotes.  Defaults to choosing a color from a large pool of random colors.  you can specify a color by giving it a string as the second variable.  The string is either a hex color code, or one of the few matplotlib defaults like 'red' 'green' 'blue'.
	'''

	global elower,eupper,qns,logint,qn7,qn8,qn9,qn10,qn11,qn12,S,dV,T,vlsr,frequency,freq_sim,intensity,int_sim,current,fig,ax

	#store the currently-active simulation in a temporary holding cell

	freq_temp = freq_sim 
	int_temp = int_sim 
	
	#pull the old simulation out of storage
	
	freq_sim = sim[x].freq_sim
	int_sim = sim[x].int_sim

	if any(line for line in ax.lines if line.get_label()==sim[x].name):
		if cchoice == None:
			line_color = [line.get_color() for line in ax.lines if line.get_label()==sim[x].name][0]
		else:
			line_color = cchoice
		clear_line(sim[x].name)
	else:
		if cchoice == None:
			line_color = next(colors)
		else:
			line_color = cchoice			
	
	lines[sim[x].name] = ax.plot(freq_sim,int_sim,color = line_color, linestyle=line_style, label = sim[x].name,linewidth=thick)
	
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
	fig.canvas.draw()
	
	freq_sim = freq_temp
	int_sim = int_temp
	
	save_results('last.results')
		
#load_mol loads a new molecule into the system.  Make sure to store the old molecule simulation first, if you want to get it back.  The current graph will be updated with the new molecule.  Catalog file must be given as a string.  Simulation will begin with the same T, dV, S, vlsr as previous, so change those first if you want.

def load_mol(x,format='spcat',vib_states=None):

	'''
	loads a new molecule into the system.  Make sure to store the old molecule simulation first, if you want to get it back.  The current graph will be updated with the new molecule.  Catalog file must be given as a string.  Simulation will begin with the same T, dV, C, vlsr as previous, so change those first if you want.
	'''

	global frequency,logint,error,dof,gup,glow,tag,qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8,qn9,qn10,qn11,qn12,elower,eupper,intensity,qns,catalog,catalog_file,fig,current,fig,ax,freq_sim,int_sim,first_run,sijmu,gauss,aij,vibs,int_tau
	
	current = x
	
	try:
		clear_line('current')
	except:	
		pass	
	
	catalog_file = x
	vibs = vib_states
	
	if format == 'spcat':
	
		raw_array = read_cat(catalog_file)

		catalog = splice_array(raw_array)

		frequency = np.copy(catalog[0])
		error = np.copy(catalog[1])
		logint = np.copy(catalog[2])
		dof = np.copy(catalog[3])
		elower = np.asarray(catalog[4])
		gup = np.asarray(catalog[5])
		tag = np.asarray(catalog[6])
		qnformat = np.asarray(catalog[7])		
		qn1 = np.asarray(catalog[8])
		qn2 = np.asarray(catalog[9])
		qn3 = np.asarray(catalog[10])
		qn4 = np.asarray(catalog[11])
		qn5 = np.asarray(catalog[12])
		qn6 = np.asarray(catalog[13])
		qn7 = np.asarray(catalog[14])
		qn8 = np.asarray(catalog[15])
		qn9 = np.asarray(catalog[16])
		qn10 = np.asarray(catalog[17])
		qn11 = np.asarray(catalog[18])
		qn12 = np.asarray(catalog[19])

		eupper = np.copy(elower)

		eupper = elower + frequency/29979.2458

		qns = det_qns(qnformat) #figure out how many qns we have for the molecule

		intensity = convert_int(logint)
	
		tmp_freq = np.copy(frequency)
	
		tmp_freq += (-vlsr)*tmp_freq/ckm
	
		Q = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,CT,catalog_file,vibs=None)
	
		#generate a unique set of energy levels as defined by quantum numbers
	
		ustate_qns = np.vstack((qn1, qn2, qn3, qn4, qn5, qn6)).T
	
		lstate_qns = np.vstack((qn7, qn8, qn9, qn10, qn11, qn12)).T
	
		global ustate_qns_hash, lstate_qns_hash
	
		ustate_qns_hash = np.sum(ustate_qns*np.array([1,10E3,10E6,10E9,10E12,10E15]), axis=1)
	
		lstate_qns_hash = np.sum(lstate_qns*np.array([1,10E3,10E6,10E9,10E12,10E15]), axis=1)
	
		equivalency = np.equal.outer(ustate_qns_hash, lstate_qns_hash)
	
		idx = np.argmax(equivalency, axis=0)
	
		glow = gup[idx]
	
		glow[np.sum(equivalency, axis=0)==0] = 1

		#from CDMS website

		sijmu = (exp(np.float64(-(elower/0.695)/CT)) - exp(np.float64(-(eupper/0.695)/CT)))**(-1) * ((10**logint)/frequency) * (24025.120666) * Q

	
		#aij formula from CDMS.  Verfied it matched spalatalogue's values
	
		aij = 1.16395 * 10**(-20) * frequency**3 * sijmu / gup
		
	def _load_molsim(filein):
		'''
		Reads in a catalog file from molsim
		'''
		npz_dict = np.load(filein,allow_pickle=True)	
		new_dict = {}
		for x in npz_dict:
			new_dict[x] = npz_dict[x]

		return new_dict

	if format == 'molsim':
	
		cat_dict = _load_molsim(catalog_file)
		
		frequency = cat_dict['frequency']
		error = cat_dict['freq_err']
		logint = cat_dict['logint']
		dof = cat_dict['dof']
		elower = cat_dict['elow']*0.695
		gup = cat_dict['gup']
		tag = cat_dict['tag']
		qnformat = cat_dict['qnformat']	
		qn1 = cat_dict['qn1up']
		qn2 = cat_dict['qn2up']
		qn3 = cat_dict['qn3up']
		qn4 = cat_dict['qn4up']
		qn5 = cat_dict['qn5up']
		qn6 = cat_dict['qn6up']
		qn7 = cat_dict['qn1low']
		qn8 = cat_dict['qn2low']
		qn9 = cat_dict['qn3low']
		qn10 = cat_dict['qn4low']
		qn11 = cat_dict['qn5low']
		qn12 = cat_dict['qn6low']
		eupper = cat_dict['eup']*0.695
		glow = cat_dict['glow']
		sijmu = cat_dict['sijmu']
		aij = cat_dict['aij']
		qns = det_qns(qnformat)
		
		intensity = convert_int(logint)
		tmp_freq = np.copy(frequency)
		tmp_freq += (-vlsr)*tmp_freq/ckm
		Q = calc_q(qns,elower,qn7,qn8,qn9,qn10,qn11,qn12,CT,catalog_file,vibs=None)
	
	print('A value of Q({}) = {} was used to calculated the Sijmu^2 and Aij values for this species.' .format(int(CT),int(Q)))
	
	freq_sim,int_sim,int_tau=run_sim(tmp_freq,intensity,T,dV,C)		

	if first_run == True:
		make_plot()
		first_run = False
		return 
	else:
		try:
			plt.get_fignums()[0]
		except:	
			make_plot()
			return 			
		
	#if there is a plot open, we just update the current simulation
	
	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		try:
			lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)
		except ValueError:
			print('No lines exist in this catalog within the specified limits.')
			return	

	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
	fig.canvas.draw()	
		
	save_results('last.results')
	
	return
		
#clear_line removes a line labeled 'x' from the current plot window.  x must be a string

def clear_line(x):

	'''
	removes a line labeled 'x' from the current plot window.  x must be a string
	'''	
	
	try:
		line = lines.pop(x)
	except:
		return
	
	try:
		line.remove()
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			ax.legend(loc='upper right')
		fig.canvas.draw()	

	except:
		line[0].remove()
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			ax.legend(loc='upper right')
		fig.canvas.draw()	
		
#clear is an alias for clear_line

def clear(x):

	'''
	Alias for clear_line()
	'''

	clear_line(x)
	
#save_results prints out a file with all of the parameters for each molecule *which has been stored.*  It will not print the active molecule unless it has been stored. 'x' must be a string, and the output will go to there.

def save_results(x):

	'''
	save_results prints out a file that contains at the top the output of status(), followed by all of the parameters for each molecule which has been stored.  This file can then be used with restore() to restore the parameter space and keep going.
	'''

	with open(x, 'w') as output:
	
		output.write('simulate_lte.py version {}\n' .format(version))		
		output.write('saved: {}\n\n' .format(datetime.now().strftime("%m-%d-%y %H:%M:%S")))
		
		output.write('#### Active Simulation ####\n\n')
	
		output.write('molecule:\t{}\n' .format(current))
		output.write('obs:\t{}\n' .format(spec))
		output.write('T:\t{} K\n' .format(T))
		output.write('C:\t{:.2f}\n' .format(C))
		output.write('dV:\t{:.2f} km/s\n' .format(dV))
		output.write('VLSR:\t{:.2f} km/s\n' .format(vlsr))
		output.write('ll:\t{} MHz\n' .format(ll))
		output.write('ul:\t{} MHz\n' .format(ul))
		output.write('CT:\t{} K\n' .format(CT))
		output.write('vibs:\t{}\n\n' .format(vibs))
		output.write('gauss:\t{}\n' .format(gauss))
		output.write('catalog_file:\t{}\n' .format(catalog_file))
		output.write('thermal:\t{} K\n' .format(thermal))
		output.write('GHz:\t{}\n' .format(GHz))
		output.write('rms:\t{}\n\n' .format(rms))
		
	
		output.write('#### Stored Simulations ####\n\n')
		
		output.write('Molecule\tT(K)\tC\tdV\tvlsr\tCT\tvibs\tcatalog_file\n')
	
		for molecule in sim:
		
			output.write('{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\t{}\n' .format(sim[molecule].name,sim[molecule].T,sim[molecule].C,sim[molecule].dV,sim[molecule].vlsr,sim[molecule].CT,sim[molecule].vibs,sim[molecule].catalog_file))
			
		output.write('\n#### Active Graph Status ####\n\n')
		
		output.write('Label\tColor\tStyle\n')
		
		for line in ax.lines:
		
			output.write('{}\t{}\t{}\n' .format(line.get_label(),line.get_color(),line.get_linestyle()))	
	
#status prints the current status of the program and the various key variables.

def status():

	'''
	prints the current status of the program and the various key variables
	'''

	print('Current Molecule:\t {}' .format(current))
	print('Current Catalog:\t {}' .format(catalog_file))
	print('Current lab or observation: \t {}' .format(spec))
	print('T: \t {} K' .format(T))
	print('C: \t {}' .format(C))
	print('dV: \t {} km/s' .format(dV))
	print('VLSR: \t {} km/s' .format(vlsr))
	print('ll: \t {} MHz' .format(ll))
	print('ul: \t {} MHz' .format(ul))
	print('CT: \t {} K' .format(CT))
	print('gauss: \t {}' .format(gauss))
	print('thermal: \t {}' .format(thermal))
		
#sum_stored_thin creates a combined spectrum of all stored molecule simulations and stores it in freq_sum and int_sum.  This might take a while.  It's done from scratch because the frequency axes in freq_sim stored for each molecule will not necessarily be the same, so co-adding is difficult without re-gridding, which well, maybe later.	

def sum_stored_thin():

	'''
	Creates a combined spectrum of all stored molecule simulations and stores it in freq_sum and int_sum.  This might take a while.
	'''
	
	global freq_sum,int_sum
	
	#first, we need to be intelligent about how we set the limits for the sum - they should cover the entirety of the summed frequency simulations, but not more.  We can do that by adding all those together and running an find_limits function on them.
	
	total_sim_freqs = []
	
	#loop through all the stored simulations and add their frequency axes into total_sim_freqs
	
	for x in sim:
	
		total_sim_freqs.extend(sim[x].freq_sim)
	
		#total_sim_freqs = np.concatenate((total_sim_freqs,sim[x].freq_sim),axis=None)
		
	#make that a numpy array
	
	total_sim_freqs = np.asarray(total_sim_freqs)
		
	#sort that
	
	total_sim_freqs = np.sort(total_sim_freqs)
	
	#now run a limit finder
	
	sum_ll,sum_ul = find_limits(total_sim_freqs)
	
	#now we make a new frequency array running between these limits at spacing res
	
	freq_sum = []
	
	for x,y in zip(sum_ll,sum_ul):
	
		freq_chunk = np.arange(x,y,res)
	
		freq_sum = np.concatenate((freq_sum,freq_chunk),axis=None)
		
	freq_sum = np.sort(freq_sum)
		
	#now we make an intensity array
	
	int_sum = np.zeros_like(freq_sum)
	
	#loop through all the stored simulations, regrid them onto freq_sum, and add to int_sum
	
	for x in sim:
	
		int_chunk = np.interp(freq_sum,sim[x].freq_sim,sim[x].int_sim,left=np.nan,right=np.nan)
		
		int_sum += int_chunk
		
	overplot_sum()
	
#sum_stored_thick tries to do a summation using the tau values that have been stored

def sum_stored_thick():

	'''
	Creates a combined spectrum of all stored molecule simulations and stores it in freq_sum and int_sum.  This might take a while.
	'''
	
	global freq_sum,int_sum
	
	#first, we need to be intelligent about how we set the limits for the sum - they should cover the entirety of the summed frequency simulations, but not more.  We can do that by adding all those together and running an find_limits function on them.
	
	total_sim_freqs = []
	
	#loop through all the stored simulations and add their frequency axes into total_sim_freqs
	
	for x in sim:
	
		total_sim_freqs.extend(sim[x].freq_sim)
	
		#total_sim_freqs = np.concatenate((total_sim_freqs,sim[x].freq_sim),axis=None)
		
	#make that a numpy array
	
	total_sim_freqs = np.asarray(total_sim_freqs)
		
	#sort that
	
	total_sim_freqs = np.sort(total_sim_freqs)
	
	#now run a limit finder
	
	sum_ll,sum_ul = find_limits(total_sim_freqs)
	
	#now we make a new frequency array running between these limits at spacing res
	
	freq_sum = []
	
	for x,y in zip(sum_ll,sum_ul):
	
		freq_chunk = np.arange(x,y,res)
	
		freq_sum = np.concatenate((freq_sum,freq_chunk),axis=None)
		
	freq_sum = np.sort(freq_sum)
		
	#now we make an intensity array
	
	int_sum = np.zeros_like(freq_sum)
	
	#loop through all the stored simulations, regrid them onto freq_sum, and add to int_sum
	
	for x in sim:
	
		int_chunk = np.zeros_like(int_sum)
		
		freq_tmp = np.copy(sim[x].frequency)
		
		freq_trim = trim_array(freq_tmp,sim[x].frequency,ll,ul)
		
		freq_trim -= sim[x].vlsr*freq_trim/ckm
	
		for y in range(len(sim[x].int_tau)):
		
			idx = find_nearest(freq_sum,freq_trim[y])
			
			int_chunk[idx] = sim[x].int_tau[y]
		
		int_sum += int_chunk
		
	freq_sum,int_sum = sim_gaussian(int_sum,freq_sum,dV)
	
	int_sum = apply_beam(freq_sum,int_sum,source_size,dish_size,synth_beam,interferometer)
		
	overplot_sum()	

#sum_stored creates a combined spectrum of all stored molecule simulations and stores it in freq_sum and int_sum.  This might take a while.  It's done from scratch because the frequency axes in freq_sim stored for each molecule will not necessarily be the same, so co-adding is difficult without re-gridding, which well, maybe later.	

def sum_stored():

	'''
	Creates a combined spectrum of all stored molecule simulations and stores it in freq_sum and int_sum.  This might take a while.
	'''
	global freq_sum, int_sum
	
	freq_gauss_tmp = []
	
	freq_sim_total = []

	for x in sim:
	
		tmp_freq = np.copy(sim[x].frequency)
		
		tmp_freq += (-sim[x].vlsr)*tmp_freq/ckm	
		
		tmp_freq_trimmed = trim_array(tmp_freq,tmp_freq,ll,ul)
		
		freq_sim_total.extend(tmp_freq_trimmed)

	freq_sim_total.sort()
		
	y = 0
		
	while (y < len(freq_sim_total)):
	
		l_f = sim[x].dV*freq_sim_total[y]/ckm #get the FWHM in MHz

		min_f = freq_sim_total[y] - 10*l_f #get the frequency 10 FWHM lower
		max_f = freq_sim_total[y] + 10*l_f #get the frequency 10 FWHM higher
	
		if y < len(freq_sim_total)-2:
	
			while (freq_sim_total[y+1] < max_f and y < len(freq_sim_total)-2):
	
				y += 1

		freq_line = np.arange(min_f,max_f,res)

		freq_gauss_tmp.extend(freq_line)
		
		y += 1
	
	freq_gauss_tmp.sort()
	
	freq_gauss = np.asarray(freq_gauss_tmp)
	
	int_gauss = np.copy(freq_gauss)
	
	int_gauss *= 0.0
	
	del tmp_freq, tmp_freq_trimmed
		
	for x in sim:
	
		Q = calc_q(sim[x].qns,sim[x].elower,sim[x].qn7,sim[x].qn8,sim[x].qn9,sim[x].qn10,sim[x].qn11,sim[x].qn12,sim[x].T,sim[x].catalog_file,sim[x].vibs)
	
		Nl = sim[x].C * (2*sim[x].qn7 + 1) * np.exp(-sim[x].elower/(0.695 * sim[x].T)) / Q
	
		tau_numerator = np.asarray((ccm/(sim[x].frequency * 10**6))**2 * sim[x].aij * sim[x].gup * Nl * (1 - np.exp(-(h * sim[x].frequency * 10**6)/(k * sim[x].T))),dtype=float)
	
		tau_denominator = np.asarray(8 * np.pi * (sim[x].dV * sim[x].frequency * 10**6 / ckm) * (2*sim[x].qn7 + 1),dtype=float)

		tau =tau_numerator/tau_denominator
		
		int_tmp = trim_array(tau,sim[x].frequency,ll,ul)		
	
		freq_tmp = trim_array(sim[x].frequency,sim[x].frequency,ll,ul)
		
		freq_tmp += (-sim[x].vlsr)*freq_tmp/ckm	
		
		Tbg = calc_tbg(tbg_params,tbg_type,tbg_range,freq_gauss)
	
		#tmp_freq_trimmed,tmp_int_trimmed = sim_gaussian(int_tmp,freq_tmp,sim[x].dV)
	
		for y in range(len(int_tmp)):
		
			l_f = sim[x].dV*freq_tmp[y]/ckm #get the FWHM in MHz
			
			c = l_f/2.35482			
			
			#int_tmp_tau = (sim[x].T - Tbg)*(1 - np.exp(-int_tmp[y]))
			
			int_tmp_tau_gauss = int_tmp[y]*exp(-((freq_gauss - freq_tmp[y])**2/(2*c**2)))
				
			int_gauss += int_tmp_tau_gauss
			
	Tbg = calc_tbg(tbg_params,tbg_type,tbg_range,freq_gauss)		
		
	J_T = (h*freq_gauss*10**6/k)*(np.exp(((h*freq_gauss*10**6)/(k*T))) -1)**-1
	J_Tbg = (h*freq_gauss*10**6/k)*(np.exp(((h*freq_gauss*10**6)/(k*Tbg))) -1)**-1
		
	int_gauss = (J_T - J_Tbg)*(1 - np.exp(-int_gauss))					

	if planck == True:

		#calculate the beam solid angle, and throw an error if it hasn't been set.

		try:
			omega = synth_beam[0]*synth_beam[1] #conversion below already has omega in it
			#omega = synth_beam[0]*synth_beam[1]*np.pi/(4*np.log(2))	
		except TypeError:
			print('You need to set a beam size to use for this conversion with synth_beam = [bmaj,bmin]')
			print('Cannot produce an accurate summed spectrum')
			return 

		#create an array that's a copy of int_sim to work with temporarily
	
		int_jansky = np.copy(int_gauss)

		#create a mask so we only work on non-zero values

		mask = int_jansky != 0

		#do the conversion

		int_jansky[mask] = (3.92E-8 * (freq_gauss[mask]*1E-3)**3 *omega/ (np.exp(0.048*freq_gauss[mask]*1E-3/int_gauss[mask]) - 1))

		int_gauss = int_jansky			
	
	freq_sum = freq_gauss
	int_sum = int_gauss		
	
	overplot_sum()

#overplot_sum overplots the summed spectrum of all stored molecules as created by sum_stored() on the current plot, in green.

def overplot_sum():

	'''
	Overplots the summed spectrum of all stored molecules as created by sum_stored() on the current plot, in green.
	'''	
	
	line_color = '#00FF00'
	
	if any(line for line in ax.lines if line.get_label()=='sum'):
		clear_line('sum')
	
	lines['sum'] = ax.plot(freq_sum,int_sum,color = line_color, label = 'sum', gid='sum', linestyle = '-',drawstyle=draw_style,zorder=25)
	
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
	fig.canvas.draw()	
	
#restore restores the state of the program from a save file, loading all stored spectra into memory, loads the previously active simulation into current, and restores the last active graph. x is a string with the filename of the restore file. The catalog files must be present, and named to match those in the save file.

def restore(x):

	'''
	restores the state of the program from a save file, loading all stored spectra into memory, loads the previously active simulation into current, and restores the last active graph. x is a string with the filename of the restore file. The catalog files must be present, and named to match those in the save file.
	
	This procedure attempts to correct for any backward compatability issues with old versions of the program.  Usually the restore will proceed without issue and will warn the user if there were issues it corrected.  The simplest way to update a restore file is to save it with the latest version of the program after a successful load.  Most of the time, the backwards compatability issues are caused simply by missing meta-data that have been added in later version of the program.  In this case, the default values are simply used.
	'''

	global frequency,logint,qn7,qn8,qn9,qn10,qn11,qn12,elower,eupper,intensity,qns,catalog,catalog_file,fig,current,fig,ax,freq_sim,int_sim,T,dV,C,vlsr,ll,ul,CT,gauss,first_run,thermal,sim,GHz,rms,res,vibs
	
	#close the old graph
	
	#empty out the previously-stored simulations
	
	sim = {}

	#read in the save file as an array line by line
	
	restore_array = []
	
	GHz = False
	
	try:
		with open(x) as input:
			for line in input:
				restore_array.append(line)
	except TypeError:
		print('Input must be a string in \'\'.')
		return
	except FileNotFoundError:
		print('This is not the file you are looking for.')
		return
	
	#check if the file that was read it is actually a savefile of the appropriate format...
	
	#flag for whether it's an old file
	
	old = False
	
	if restore_array[0].split()[0] != 'simulate_lte.py':
		
		#check if it was made with viewspectrum.py.  If it was, it's *probably* fine...
	
		if restore_array[0].split()[0] == 'viewspectrum.py':
			print('WARNING: This file was made with a substantially older version of the program. An attempt will be made to read it and update it.  Please save it again with the new version of the program to ensure compatibility going forward.')
		
		#otherwise it's not a valid file...
		
		else:
			print('The file is not a simulate_lte.py save file, has been altered, or was created with an incompatibly older version of the program.  In any case, I can\'t read it, sorry.')
			return
			
		if float(restore_array[0].split()[2]) < 6.9:
			
			#set a flag that we don't have vib states, and warn the user
			
			old = True
		
			print('WARNING: This file was made with an older version of the program. An attempt will be made to read it and update it.  Please save it again with the new version of the program to ensure compatibility going forward.')
			
			#add a blank vibs line that it's looking for, as well as a blank line after that.
			
			restore_array.insert(14,'vibs:\tNone\n')
			restore_array.insert(15,'\n')
			
			with open('bunk.txt', 'w') as output:
			
				for line in restore_array:
				
					output.write(line)
		
			
		
	#let's grab the date and time, so we can print some nice information to the terminal
	
	restore_date = restore_array[1].split()[1]
	restore_time = restore_array[1].split()[2]	
	
	#do a version check
	
	restore_version = float(restore_array[0].split()[2])
	
	if restore_version != version:
	
		print('Warning: {} was created using v{}.  The current version is v{}.  Type help(restore) for more information.' .format(x,restore_version,str(version)))   
		
	#separate out the sections into their own arrays

	active_array = []	
	stored_array = []
	graph_array = []
	
	#figure out where each section starts
	
	active_index = restore_array.index('#### Active Simulation ####\n')
	stored_index = restore_array.index('#### Stored Simulations ####\n')
	graph_index = restore_array.index('#### Active Graph Status ####\n')
	
	for i in range(active_index+2,stored_index-1):
	
		active_array.append(restore_array[i])
		
	for i in range(stored_index+3,graph_index-1):
	
		stored_array.append(restore_array[i])
		
	for i in range(graph_index+3,len(restore_array)):
	
		graph_array.append(restore_array[i])
		
	#we'll turn the active array into a dictionary, to further avoid conflicts with older versions.
	
	active_dict = {}
	
	for x in active_array:
	
		#if it's a blank line, move on:
		
		if x == '\n':
		
			continue
	
		key = x.split('\t')[0].strip(':')
		value = x.split('\t')[1].strip().strip('\n')
		
		active_dict[key] = value
		
	#go through and clean all those values up.  Our dictionary has these keys: 
	#'molecule'
	#'obs'
	#'T'
	#'C'
	#'dV'
	#'VLSR'
	#'ll'
	#'ul'
	#'CT'
	#'vibs'
	#'gauss'
	#'catalog_file'
	#'thermal'
	#'GHz'
	#'rms'
	
	#set the Temperature
	
	active_dict['T'] = float(active_dict['T'].split()[0])
	
	#set the Column
	
	active_dict['C'] = float(active_dict['C'])
	
	#set the dV
	
	active_dict['dV'] = float(active_dict['dV'].split()[0])
	
	#set the VLSR
	
	active_dict['VLSR'] = float(active_dict['VLSR'].split()[0])	
	
	#set the CT
	
	active_dict['CT'] = float(active_dict['CT'].split()[0])
	
	#thermal is super deprecated and isn't used anymore
	
	#set the rms. if there isn't one, make one
	
	if 'rms' not in active_dict:
	
		active_dict['rms'] = float('-inf')
		print('WARNING: The restore file does not have an rms value in it.  It was probably generated with a previous version of the program. The restore can proceed, but it is recommended that you re-save the restore file with the latest version of the program.')
	
	else:
	
		active_dict['rms'] = float(active_dict['rms'])	
		
	
		
	#just to be safe, let's set the upper limits, lower limits, gaussian toggles, thermal values, and GHz flag now.
	
	#what is the rms level set at? 
	
	rms = active_dict['rms']
	
	#are we simulating Gaussians?
	
	if 'True' in active_dict['gauss']:
		gauss = True
	else:
		gauss = False
		
	if gauss == False:
		print('I just set gauss = False.')
		
	#are observations read in in GHz?
	
	if 'True' in active_dict['GHz']:
		GHz = True
	else:
		GHz = False	
	
	#set the lower limit.  requires some logical to differentiate between a single value and a list	
		
	try:
		ll = float(active_dict['ll'].split()[0].strip(' MHz\n'))
	except ValueError:
		ll = []
		tmp_str = active_dict['ll'].split(']')[0].strip(' MHz\n').strip(']').strip('[').split(',')
		for line in range(len(tmp_str)):
			ll.append(float(tmp_str[line]))
			
	#set the upper limit.  requires some logical to differentiate between a single value and a list		
	
	try:	
		ul = float(active_dict['ul'].split()[0].strip(' MHz\n'))
	except ValueError:
		ul = []
		tmp_str = active_dict['ul'].split(']')[0].strip(' MHz\n').strip(']').strip('[').split(',')
		for line in range(len(tmp_str)):
			ul.append(float(tmp_str[line]))
	
	#not using thermal anymore
	#thermal = float(active_array[12].split('\t')[1].strip(' K\n'))
	
	try:
		obs = active_dict['obs'].strip('\n')
		read_obs(obs)
	except:
		res = 0.1
		pass
	
	#OK, now time to do the hard part.  As one always should, let's start with the middle part of the whole file, and load and then store all of the simulations.
	
	for i in range(len(stored_array)):
	
		#if this is an old file, we have to insert a vibs string
		
		if old is True:
		
			tmp_array = stored_array[i].split('\t')
			
			tmp_array.insert(6,'None')
			
			stored_array[i] = '\t'.join([str(x) for x in tmp_array])	
	
		name = stored_array[i].split('\t')[0].strip('\n')
		T = float(stored_array[i].split('\t')[1])
		C = float(stored_array[i].split('\t')[2])
		dV = float(stored_array[i].split('\t')[3])
		vlsr = float(stored_array[i].split('\t')[4])
		CT = float(stored_array[i].split('\t')[5])
		

			
		if stored_array[i].split('\t')[6] == 'None':
		
			vibs = None
			
		else:
		
			tmp_str = stored_array[i].split('\t')[6]
			
			tmp_str = tmp_str.strip('[')
			tmp_str = tmp_str.strip(']').split(',')
			
			tmp_str = [x.strip().strip('\n').strip(']') for x in tmp_str]
		
			vibs = []
			
			for x in tmp_str:
			
				vibs.append(float(x))
				
			
		catalog_file = str(stored_array[i].split('\t')[7]).strip('\n').strip()
		

		first_run = True
		load_mol(catalog_file,vib_states=vibs)

# 		try:	
# 			load_mol(catalog_file)
# 		except FileNotFoundError:
# 			continue

		store(name)
		
		close()
			
	#Now we move on to loading in the currently-active molecule
	
	try:
		obs = active_dict['obs'].strip('\n')
		read_obs(obs)
	except:
		pass
	name = active_dict['molecule'].strip('\n')

	try:
		recall(name)
	except KeyError:
		catalog_file = active_dict['catalog_file'].strip('\n')
		T = active_dict['T']
		C = active_dict['C']
		dV = active_dict['dV']
		vlsr = active_dict['VLSR']
		CT = active_dict['CT']
		
		if active_dict['vibs'] == 'None':
		
			vibs = None
			
		else:
		
			tmp_str = active_dict['vibs'].split('\t')[1]
			
			tmp_str = tmp_str.strip('[')
			tmp_str = tmp_str.strip(']').split(',')
			
			tmp_str = [x.strip().strip('\n').strip(']') for x in tmp_str]
		
			vibs = []
			
			for x in tmp_str:
			
				vibs.append(float(x))
		
		current = active_dict['molecule'].strip('\n')
		name = active_dict['molecule'].split('/')[-1].strip('\n')
		
		first_run = True
		load_mol(catalog_file,vib_states=vibs)

	#And finally, overplot anything that was on the plot previously
	
	for i in range(len(graph_array)):
	
		name = graph_array[i].split('\t')[0]
		line_color = graph_array[i].split('\t')[1]
		line_style = graph_array[i].split('\t')[2].strip('\n')
		
		if name == 'obs':
		
			continue
			
		elif name == 'current':
		
			continue
			
		elif name == 'sum':
		
			sum_stored()
			overplot_sum()
			continue			
			
		else:
			
			lines[name] = ax.plot(sim[name].freq_sim,sim[name].int_sim,color = line_color, linestyle=line_style, label = name)	
			
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			ax.legend(loc='upper right')
		fig.canvas.draw()	
		
	#If we made it here, we were successful, so let's print out what we did
	
	print('Successfully restored from file {} which was saved on {} at {}.' .format(x,restore_date,restore_time))	
	
#fix_legend allows you to change the legend to meet its size needs.

def fix_legend(x,lsize):

	'''
	Modifies the legend to have x columns.  lsize can be {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'} or an int or float.
	'''

	plt.legend(ncol=x,prop={'size':lsize},loc='upper right')
	
	fig.canvas.draw()	
	
#purge permanently removes a stored simulation from memory.  Can't undo this one, folks.

def purge(x):

	'''
	permanently removes a stored simulation x from memory.  Can't undo this one, folks.
	'''
	
	try:
		del sim[x]
	except KeyError:
		print('No simulation with that key in the simulation dictionary.  Type \'sim\' to see an (ugly) list of stored simulations.')
	
#use_GHz changes the read-in default for observations from MHz to GHz.  This will just convert the read-in frequencies to MHz.

def use_GHz():

	'''
	changes the read-in default for observations from MHz to GHz.  This will just convert the read-in frequencies to MHz.
	'''

	global GHz
	
	GHz = True

#quiet suppresses warnings about computational time.  Can be used iteratively to turn it on and off.

def quiet():

	global quietflag

	if quietflag == False:
	
		quietflag = True
		
	elif quietflag == True:
	
		quietflag = False

#find_limits() automatically finds the upper and lower limits of the input array.

def find_limits(freq_arr,spacing_tolerance=100):
	
	if len(freq_arr) == 0:
	
		print('The input array has no data.')
		
		return
		
	#run through the data and try to find gaps
	#first, calculate the average spacing of the datapoints
	
	spacing = abs(freq_arr[0]-freq_arr[10])/10
	
	#find all values where the next point is more than spacing_tolerance * spacing away
	
	#initialize a list to hold these, with the first point being in there automatically
	
	values = [freq_arr[0]]
	
	for x in range(len(freq_arr)-1):
	
		if abs(freq_arr[x+1] - freq_arr[x]) > spacing_tolerance*spacing:
		
			values.append(freq_arr[x])
			values.append(freq_arr[x+1])
			
	#make sure the last point gets in there too
	
	values.append(freq_arr[-1])
			
	ll = values[0::2] 
	ul = values[1::2]
	
	return ll,ul
		
#autoset_limits() automatically sets the upper and lower limits to 25 MHz above and below the lowest limits of the loaded spectra.

def autoset_limits(spacing_tolerance=100):

	global ll,ul
	
	if len(freq_obs) == 0:
	
		print('First, load a spectrum with read_obs()')
		
	#run through the data and try to find gaps
	#first, calculate the average spacing of the datapoints
	
	spacing = abs(freq_obs[0]-freq_obs[10])/10
	
	#find all values where the next point is more than spacing_tolerance * spacing away
	
	#initialize a list to hold these, with the first point being in there automatically
	
	values = [freq_obs[0]]
	
	for x in range(len(freq_obs)-1):
	
		if abs(freq_obs[x+1] - freq_obs[x]) > spacing_tolerance*spacing:
		
			values.append(freq_obs[x])
			values.append(freq_obs[x+1])
			
	#make sure the last point gets in there too
	
	values.append(freq_obs[-1])
			
	ll = values[0::2] 
	ul = values[1::2]

	ll = [x-25 for x in ll]
	ul = [x+25 for x in ul]

#plot_residuals will make a new plot and show the residual spectrum after the total simulation is subtracted from the observations

def plot_residuals():

	#first, make a simulation that has exactly the same frequency points as the observations
	
	global freq_resid,int_resid
	
	freq_gauss = np.asarray(freq_obs)
	
	int_gauss = np.copy(freq_gauss)
	
	int_gauss *= 0.0	
	
	for x in sim:

		tmp_freq = np.copy(sim[x].frequency)
	
		tmp_freq += (-sim[x].vlsr)*tmp_freq/ckm	
	
		tmp_freq_trimmed = trim_array(tmp_freq,tmp_freq,ll,ul)
	
		Q = calc_q(sim[x].qns,sim[x].elower,sim[x].qn7,sim[x].qn8,sim[x].qn9,sim[x].qn10,sim[x].qn11,sim[x].qn12,CT,sim[x].catalog_file)

		sijmu = (exp(np.float64(-(sim[x].elower/0.695)/CT)) - exp(np.float64(-(sim[x].eupper/0.695)/CT)))**(-1) * ((10**sim[x].logint)/sim[x].frequency) * ((4.16231*10**(-5))**(-1)) * Q
	
		tmp_int = np.copy(sim[x].intensity)
		
		Tbg = calc_tbg(tbg_params,tbg_type,tbg_range,tmp_freq)
	
		Q = calc_q(sim[x].qns,sim[x].elower,sim[x].qn7,sim[x].qn8,sim[x].qn9,sim[x].qn10,sim[x].qn11,sim[x].qn12,sim[x].T,sim[x].catalog_file)
	
		numerator = (sim[x].C)*(8*3.14159**3)*(tmp_freq*1E6)*(sijmu)*(1-((exp((h*tmp_freq*1E6)/(k*sim[x].T))-1)/(exp((h*tmp_freq*1E6)/(k*Tbg))-1)))*eta

		denominator = 1.06447*sim[x].dV*Q*(exp(sim[x].eupper/(kcm*sim[x].T)))*(3*k)*1E48

		tmp_int = numerator/denominator
	
		tmp_int_trimmed = trim_array(tmp_int,tmp_freq,ll,ul)
	
		tmp_int_trimmed[tmp_int_trimmed > thermal] = thermal

		for y in range(len(tmp_int_trimmed)):
	
			if abs(tmp_int_trimmed[y]) < 0.001:
		
				continue
	
			l_f = sim[x].dV*tmp_freq_trimmed[y]/ckm #get the FWHM in MHz
		
			c = l_f/2.35482

			int_gauss += tmp_int_trimmed[y]*exp(-((freq_gauss - tmp_freq_trimmed[y])**2/(2*c**2)))
			
	int_gauss[int_gauss > thermal] = thermal
	
	freq_sum = freq_gauss
	int_sum = int_gauss		
	
	freq_resid = np.asarray(freq_sum)
	int_resid = np.asarray(int_sum)
	
	int_obs_tmp = np.asarray(int_obs)
	
	int_resid = int_obs_tmp - int_sum
	
	f, (ax1,ax2) = plt.subplots(2, sharex=True, sharey=False)
	
	minorLocator = AutoMinorLocator(5)
	plt.xlabel('Frequency (MHz)')
	ax1.set_ylabel('Intensity (K)')
	ax2.set_ylabel('Intensity (K)')
	
	plt.locator_params(nbins=4) #Use only 4 actual numbers on the x-axis
	ax1.xaxis.set_minor_locator(minorLocator) #Let the program calculate some minor ticks from that

	ax1.get_xaxis().get_major_formatter().set_scientific(False) #Don't let the x-axis go into scientific notation
	ax1.get_xaxis().get_major_formatter().set_useOffset(False)
		
	ax2.plot(freq_obs,int_obs,color = 'black',label='obs')
	ax2.plot(freq_sum,int_sum,color = 'green',label='simulation')
	ax1.set_ylim(ax2.get_ylim())
	ax1.plot(freq_resid,int_resid,color = 'red',label='residual')
	
	ax2.legend(loc='upper right')
	ax1.legend(loc='upper right')
	
	fig.canvas.draw()

#print_lines will print out the catalog info on lines that are above a certain threshold for a molecule.  The default just prints out the current lines above a standard 1 mK threshold.		

def print_lines(mol='current',thresh=float('-inf'),rest=True,mK=False,return_array = False):

	global gauss
	
	gflag = False

	if gauss == True:
	
		gauss = False
		
		gflag = True

	#prep an array to hold the output string for each line
	
	print_array = []

	#If we're working with the current simulation, re-run it to get the non-Gaussian lines
	
	if mol == 'current':
	
		name = catalog_file.split('.')[0].strip('\n').split('/')[-1]
		
		qns_length = 4*qns*2 + 15 #length of the qn string
		
		if qns_length > 15:
		
			qn_length = qns_length	#if the length of the string is going to be longer than the label, we need to pad the label.
	
		#print_array.append('Molecule: {}' .format(name))
		#print_array.append('Column Density: {:.2e} cm-2 \t Temperature: {} K \t Linewidth: {} km/s \t vlsr: {} km/s' .format(C,T,dV,vlsr))
		#print_array.append('Frequency \t Intensity (K) \t {{:<{}}} \t Eu (K) \t gJ \t log(Aij) \t Sij' .format(qn_length).format('Quantum Numbers'))
	
		#6.8 - these are now generated down below on first run.
	
		#gotta re-read-in the molecule to get back the quantum numbers
		
		raw_array = read_cat(catalog_file)

		catalog = splice_array(raw_array)
		
		gup = np.asarray(catalog[5])
		qn1 = np.asarray(catalog[8])
		qn2 = np.asarray(catalog[9])
		qn3 = np.asarray(catalog[10])
		qn4 = np.asarray(catalog[11])
		qn5 = np.asarray(catalog[12])
		qn6 = np.asarray(catalog[13])
		qn7 = np.asarray(catalog[14])
		qn8 = np.asarray(catalog[15])
		qn9 = np.asarray(catalog[16])
		qn10 = np.asarray(catalog[17])
		qn11 = np.asarray(catalog[18])
		qn12 = np.asarray(catalog[19])
			
		#run the simulation to get the intensities
	
		freq_tmp,int_tmp,int_tau = run_sim(frequency,intensity,T,dV,C)
		
		old_f = np.nan
		
		i = 0

		for x in range(len(freq_tmp)):
		
			if int_tmp[x] > thresh:
		
				y = np.where(frequency == freq_tmp[x])[0]
		
				qn_string = ''
		
				#deal with the case where multiple transitions have the same frequency		
				
				if freq_tmp[x] == old_f:
				
					i += 1
					
				else:
				
					i = 0 

				if qns == 1:
   
					qn_string = '{:>2} -> {:>2}' .format(qn1[y][i],qn7[y][i])
	   
				if qns == 2:
   
					qn_string = '{:>2} {: >3} -> {:>2} {: >3}' .format(qn1[y][i],qn2[y][i],qn7[y][i],qn8[y][i])		
	   
				if qns == 3:
   
					qn_string = '{:>2} {: >3} {: >3} -> {:>2} {: >3} {: >3}' .format(qn1[y][i],qn2[y][i],qn3[y][i],qn7[y][i],qn8[y][i],qn9[y][i])							

				if qns == 4:
   
					qn_string = '{:>2} {: >3} {: >3} {: >3} -> {:>2} {: >3} {: >3} {: >3}' .format(qn1[y][i],qn2[y][i],qn3[y][i],qn4[y][i],qn7[y][i],qn8[y][i],qn9[y][i],qn10[y][i])	
	   
				if qns == 5:
   
					qn_string = '{:>2} {: >3} {: >3} {: >3} {: >3} -> {:>2} {: >3} {: >3} {: >3} {: >3}' .format(qn1[y][i],qn2[y][i],qn3[y][i],qn4[y][i],qn5[y][i],qn7[y][i],qn8[y][i],qn9[y][i],qn10[y][i],qn11[y][i])			
	   
				if qns == 6:
   
					qn_string = '{:>2} {: >3} {: >3} {: >3} {: >3} {: >3} -> {:>2} {: >3} {: >3} {: >3} {: >3} {: >3}' .format(qn1[y][i],qn2[y][i],qn3[y][i],qn4[y][i],qn5[y][i],qn6[y][i],qn7[y][i],qn8[y][i],qn9[y][i],qn10[y][i],qn11[y][i],qn12[y][i])							
	
				#gJ = 2*qn1[y][i] + 1
				
				gu = gup[y][0]
				gl = glow[y][0]
				
				qn_string.strip()
			
				if x == 0:
				
					qn_str_len = len(qn_string)
					
					space = ' '
					
					qn_str = 'Quantum Numbers'
					
					if qn_str_len > 15:
					
						qn_str += (qn_str_len - 15) * space
						
					total_length = len(qn_str)
				
					print_array.append('Molecule: {}' .format(name))
					print_array.append('Column Density: {:.2e} cm-2\tTemperature: {} K\tLinewidth: {} km/s\tvlsr: {} km/s\n' .format(C,T,dV,vlsr))
					
					if mK == True:
					
						if planck == True:
							
							print_array.append('Frequency\tIntensity (mJy)\t{}\tEu (K)   \tgu\tgl\tlog(Aij)\tSijmu^2' .format(qn_str))
					
						else:
						
							print_array.append('Frequency\tIntensity (mK)\t{}\tEu (K)   \tgu\tgl\tlog(Aij)\tSijmu^2' .format(qn_str)) 
											
					elif planck == True:
					
						print_array.append('Frequency\tIntensity (Jy)\t{}\tEu (K)   \tgu\tgl\tlog(Aij)\tSijmu^2' .format(qn_str))
						
					else:
					
						print_array.append('Frequency\tIntensity (K)\t{}\tEu (K)   \tgu\tgl\tlog(Aij)\tSijmu^2' .format(qn_str))
					
						
			
				if rest==False:
			
					frequency_tmp_shift = frequency[y][i] - vlsr*frequency[y][i]/3E5
					
					if len(qn_string) < 15:
					
						qn_string += (len(qn_string)-15)*' '				
	
					if mK == True:
					
						print_array.append('{:.4f}\t{:<13.3f}\t{}\t{:<9.3f}\t{}\t{}\t{:.2f}    \t{:.4f}' .format(frequency_tmp_shift,int_tmp[x]*1000,qn_string,eupper[y][i]/0.695,gu,gl,np.log10(aij[y][i]),sijmu[y][i]))
						
					else:
					
						print_array.append('{:.4f}\t{:<13.3f}\t{}\t{:<9.3f}\t{}\t{}\t{:.2f}    \t{:.4f}' .format(frequency_tmp_shift,int_tmp[x],qn_string,eupper[y][i]/0.695,gu,gl,np.log10(aij[y][i]),sijmu[y][i]))
				
				else:

					if len(qn_string) < 15:
					
						qn_string += (len(qn_string)-15)*' '			

					
					if mK == True:
					
						print_array.append('{:.4f}\t{:<13.3f}\t{}\t{:<9.3f}\t{}\t{}\t{:.2f}    \t{:.4f}' .format(frequency[y][i],int_tmp[x]*1000,qn_string,eupper[y][i]/0.695,gu,gl,np.log10(aij[y][i]),sijmu[y][i]))
						
					else:
					
						print_array.append('{:.4f}\t{:<13.3f}\t{}\t{:<9.3f}\t{}\t{}\t{:.2f}    \t{:.4f}' .format(frequency[y][i],int_tmp[x],qn_string,eupper[y][i]/0.695,gu,gl,np.log10(aij[y][i]),sijmu[y][i]))
					
				old_f = freq_tmp[x]	
						
	#printing lines from storage has been disabled
			
	if return_array is False:
	
		for x in range(len(print_array)):
	
			print('{}' .format(print_array[x]))	
			
		if gflag == True:
		
			gauss = True
			
		return 				
			
	if return_array is True:
	
		if gflag == True:
		
			gauss = True	
	
		return print_array
		
	
		
#gauss_func is a model gaussian function to be used with gauss_fit below

def gauss_func(x, dT, v, dV):

	#get the FWHM in frequency space here
		
	df = dV*v/ckm
	
	#convert to c
	
	c = df/2.35482
	
	#return the Gaussian value
	
	G = dT*np.exp(- ((x-v)**2/(2*c**2)))
	
	return G

#gauss_fit does a Gaussian fit on lines in the data, specified in tuples: p = [[dT1,v1,dV1],[dT2,v2,dV2],...] where dT1,v1,dV1 etc are the initial guesses for the intensity, line center, and fwhm of the lines.  dT is in whatever units are being used in the observations, v is in whatever units are being used in the observations, and dV is in km/s.  By default, the amplitude is unconstrained, the center frequency is constrained to within 5 MHz of the guess, and the linewidth is constrained to within 20% of the guess.  These can be changed.

def gauss_fit(p_array,plot=True,dT_bound=np.inf,v_bound=5.0,dV_bound=0.2,sigma=None,return_results=False,print_results=True):

	'''
	#gauss_fit does a Gaussian fit on lines in the data, specified in tuples: p = [[dT1,v1,dV1],[dT2,v2,dV2],...] where dT1,v1,dV1 etc are the initial guesses for the intensity, line center, and fwhm of the lines.  dT is in whatever units are being used in the observations, v is in whatever units are being used in the observations, and dV is in km/s.  By default, the amplitude is unconstrained, the center frequency is constrained to within 5 MHz of the guess, and the linewidth is constrained to within 20% of the guess.  These can be changed.
	'''

	data = [freq_obs,int_obs]
	
	coeff = []
	var_matrix = []	
	err_matrix = []
	fit = np.copy(freq_sim)
	fit *= 0.0
		
	for x in range(len(p_array)):
	
		if sigma is None:
	
			temp = curve_fit(gauss_func, data[0], data[1], p0 = p_array[x], bounds=([p_array[x][0]-(p_array[x][0]*dT_bound),p_array[x][1]-v_bound,p_array[x][2]*(1-dV_bound)],[p_array[x][0]+(p_array[x][0]*dT_bound),p_array[x][1]+v_bound,p_array[x][2]*(1+dV_bound)]))
			
		else:
		
			sig_tmp = np.copy(data[0])*0.0
			sig_tmp += sigma[x]
		
			temp = curve_fit(gauss_func, data[0], data[1], p0 = p_array[x], sigma=sig_tmp, bounds=([p_array[x][0]-(p_array[x][0]*dT_bound),p_array[x][1]-v_bound,p_array[x][2]*(1-dV_bound)],[p_array[x][0]+(p_array[x][0]*dT_bound),p_array[x][1]+v_bound,p_array[x][2]*(1+dV_bound)]))
		
		coeff.append(temp[0])
		var_matrix.append(temp[1])
		err_matrix.append(np.sqrt(np.diag(temp[1])))
		
		fit += gauss_func(freq_sim, coeff[x][0], coeff[x][1], coeff[x][2])

	if plot == True:
	
		try:
			plt.get_fignums()[0]
		except:	
			make_plot()
					
		clear_line('Gauss Fit')
	
		lines['Gauss Fit'] = ax.plot(freq_sim,fit,color='cyan',label='Gauss Fit',linestyle= '-', gid='gfit', zorder = 50000)
 		
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			ax.legend(loc='upper right')
		fig.canvas.draw()		
	
	print('Gaussian Fit Results to {} Lines' .format(len(p_array)))
	print('{:<20} \t {:<10} \t {:<10}' .format('Line Center','dT', 'dV'))
	
	print_array = []
	results_array = []
		
	for x in range(len(p_array)):
	
		dT_temp = coeff[x][0]
		v_temp = coeff[x][1]
		dV_temp = coeff[x][2]
		
		dT_err = err_matrix[x][0]
		v_err = err_matrix[x][1]
		dV_err = err_matrix[x][2]
		
		results_array.append([dT_temp,dT_err,v_temp,v_err,dV_temp,dV_err])
		
		print_array.append('{:<.4f}({:<.4f}) \t {:^.3f}({:^.3f}) \t {:^.3f}({:^.3f})' .format(v_temp,v_err,dT_temp,dT_err,dV_temp,dV_err))
		
	if print_results is True:
	
		for line in print_array:
		
			print(line)
	
	if return_results is True:
			
		return results_array
		
#make_gauss_params generates a parameters list for gaussian fitting.  Takes an input txt file which is first column frequencies and second column intensities.  

def make_gauss_params(file,vlsr,dV):

	p = []

	with open(file) as input:

		for line in input:
	
			p_indiv = []
	
			freq = float(line.split()[0])
			freq -= vlsr*freq/3E5
			
			p_indiv.append(float(line.split()[1].strip('\n')))
			p_indiv.append(freq)			
			p_indiv.append(float(dV))
		
			p.append(p_indiv)
		
	return p

#jy_to_k converts the current observations from Jy/beam to K, given a beam size bmaj and bmin in arcseconds, and a center frequency in GHz.  This assumes the beam size is constant over the entire range; so if you've loaded in observations from multiple cubes that have different sizes, it's not going to be completely accurate.  It would be better to load in one cube at a time, covert it, and write it back out. 

def jy_to_k(bmaj,bmin,freq):

	'''
	#jy_to_k converts the current observations from Jy/beam to K, given a beam size bmaj and bmin in arcseconds, and a center frequency in GHz.  This assumes the beam size is constant over the entire range; so if you've loaded in observations from multiple cubes that have different sizes, it's not going to be completely accurate.  It would be better to load in one cube at a time, covert it, and write it back out
	'''

	global freq_obs,int_obs
	
	for x in range(len(int_obs)):

		int_obs[x] = 1.224*10**6 * int_obs[x] / ((freq_obs[x]/1000)**2 * bmaj * bmin)		
		
	clear_line('obs')
		
	try:		
		lines['obs'] = 	ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0,drawstyle=draw_style)
	except:
		return
		
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
	fig.canvas.draw()
	
#k_to_jy converts the current observations from K to Jy/beam, given a beam size bmaj and bmin in arcseconds, and a center frequency in GHz.  This assumes the beam size is constant over the entire range; so if you've loaded in observations from multiple cubes that have different sizes, it's not going to be completely accurate.  It would be better to load in one cube at a time, covert it, and write it back out. 

def k_to_jy(bmaj,bmin,freq,sim=False):

	'''
	#jy_to_k converts the current observations from Jy/beam to K, given a beam size bmaj and bmin in arcseconds, and a center frequency in GHz.  This assumes the beam size is constant over the entire range; so if you've loaded in observations from multiple cubes that have different sizes, it's not going to be completely accurate.  It would be better to load in one cube at a time, covert it, and write it back out
	'''
	
	if sim == False:

		global freq_obs,int_obs
	
		for x in range(len(int_obs)):

			int_obs[x] = int_obs[x] * ((freq_obs[x]/1000)**2 * bmaj * bmin) / (1.224*10**6)	
		
		clear_line('obs')
		
		try:		
			lines['obs'] = 	ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0,drawstyle=draw_style)
		except:
			return
			
	else:
	
		global freq_sim,int_sim
	
		for x in range(len(int_sim)):

			int_sim[x] = int_sim[x] * (freq**2 * bmaj * bmin) / (1.224*10**6)	
		
		clear_line('current')
		
		try:		
			lines['current'] = 	ax.plot(freq_sim,int_sim,color = 'black',label='obs',zorder=0)
		except:
			return		
		
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
	fig.canvas.draw()	

#load_freqs will plot lines that are provided not from a standard spcat catalog, but rather just a set of frequencies.  The user can specify either a manual array OR a catalog file containing a single column of frequencies (not both), as well as an optional intensity for the lines.

def load_freqs(man_freqs='',peak=1.0,vlsr=vlsr,dV=dV):

	'''
	#load_freqs will plot lines that are provided not from a standard spcat catalog, but rather just a set of frequencies.  The user can specify either a manual array OR a catalog file containing a single column of frequencies.  Optional commands are an intensity for the lines (defaults to 1.0), a vlsr offset (defaults to current vlsr), and a linewidth (defaults to current linewidth).  Modifying any of these parameters requires re-issuing the entire command.
	'''	

	global int_man,freq_man
	
	try:
		clear_line('manual spectra')
	except:	
		pass	
	
	int_tmp = []
	freq_tmp = []

	if type(man_freqs) == list:
	
		for x in range(len(man_freqs)):
		
			freq_tmp.append(man_freqs[x])
			int_tmp.append(peak)
			
	else:
	
		tmp_array = read_cat(man_freqs)
		
		for x in range(len(tmp_array)):
		
			freq_tmp.append(float(tmp_array[x].split()[0].strip()))
			int_tmp.append(peak)	
			
	freq_tmp = np.asarray(freq_tmp)
	int_tmp = np.asarray(int_tmp)
	
	freq_tmp -= vlsr*freq_tmp/ckm			
			
	if gauss == True:

		freq_man,int_man = sim_gaussian(int_tmp,freq_tmp,dV)
		
	else:

		freq_man = freq_tmp
		int_man = int_tmp		
		
	if gauss == False:

		lines['manual spectra'] = ax.vlines(freq_man,0,int_man,linestyle = '-',color = 'cyan',label='manual spectra',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['manual spectra'] = ax.plot(freq_man,int_man,color = 'cyan',label='manual spectra',drawstyle=draw_style,zorder=500)	

	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
	fig.canvas.draw()						

#use_steps and use_lines will swap between drawing lines between points or steps between points as the default for all graphs.  Will update the graph for the obs immediately; molecules will need to be resimulated/re-recalled/re-overplotted/etc.

def use_steps():

	'''
	use_steps and use_lines will swap between drawing lines between points or steps between points as the default for all graphs.  Will update the graph for the obs immediately; molecules will need to be resimulated/re-recalled/re-overplotted/etc.
	'''

	global draw_style
	
	draw_style = 'steps'
	
	clear_line('obs')
		
	try:
		freq_obs[0]
		lines['obs'] = ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0,drawstyle=draw_style)
	except:
		pass
	
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
	fig.canvas.draw()	
	
def use_lines():

	'''
	use_steps and use_lines will swap between drawing lines between points or steps between points as the default for all graphs.  Will update the graph for the obs immediately; molecules will need to be resimulated/re-recalled/re-overplotted/etc.
	'''

	global draw_style
	
	draw_style = 'default'
	
	clear_line('obs')
	
	try:
		freq_obs[0]
		lines['obs'] = ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0,drawstyle=draw_style)
	except:
		pass	
	
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
	fig.canvas.draw()		

#baseline will subtract a polynomial baseline from the spectrum of whatever order is entered as an array.  So, for example, to subtract zeroth order baseline with a static offset of 2.5, issue baseline(2.5) or baseline([2.5]).  To subtract a line of y = mx + b, issue baseline([b,m]).  To go for a larger polynomial, y = ax^3 + bx^2 + cx + d, issue baseline([d,c,b,a]).

def baseline(constants):

	'''
	baseline will subtract a polynomial baseline from the spectrum of whatever order is entered as an array.  So, for example, to subtract zeroth order baseline with a static offset of 2.5, issue baseline(2.5) or baseline([2.5]).  To subtract a line of y = mx + b, issue baseline([b,m]).  To go for a larger polynomial, y = ax^3 + bx^2 + cx + d, issue baseline([d,c,b,a]).
	'''

	global int_obs
	
	if type(constants) == int or type(constants) == float:
	
		constants = [constants]
	
	int_base = [0] * len(int_obs)
	
	for x in range(len(constants)):
	
		for y in range(len(freq_obs)):
	
			int_base[y] += constants[x]*freq_obs[y]**x
			
	for x in range(len(int_obs)):
	
		int_obs[x] -= int_base[x]
		
	clear_line('obs')
	
	try:
		lines['obs'] = ax.plot(freq_obs,int_obs,color = 'black',label='obs',zorder=0,drawstyle=draw_style)
	except:
		pass
	
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
	fig.canvas.draw()				
	
# find peaks in the intensity array more than 3 sigma (optionally adjustable) and return the indices of those peaks as well as the measured RMS	
	
def find_peaks(frequency,intensity,fwhm,sigma=3,width_tweak=1.0):
		
		'''
		find peaks in the intensity array more than 3 sigma (optionally adjustable) and return the indices of those peaks and the rms
		'''
		
		#figure out how many channels a typical line will span
		
		#calculate fwhm in MHz
		
		fwhm_MHz = fwhm*np.median(frequency)/ckm
		
		#calculate channel spacing in MHz
		
		dMHz_chan = abs(frequency[-1] - frequency[0])/len(frequency)
		 			
		#calculate the number of channels per FHWM
		
		fwhm_chan = fwhm_MHz/dMHz_chan	
		
		#define the number of channels per line, modified by width_tweak:
		
		line_chan = int(fwhm_chan * 5 * width_tweak)
				
		rms = get_rms(intensity)
			
		#now that we know the actual rms, we can find all the peaks above the threshold, and we have to calculate a real threshold, as it is a normalized intensity (percentage).  
		
		intensity_tmp = np.asarray(intensity)
		
		max = np.amax(intensity_tmp)
		min = np.amin(intensity_tmp)
		
		#calc the percentage threshold that the rms sits at
		
		rms_thres = (sigma*rms - min)/(max - min)

		peak_indices = peakutils.indexes(intensity_tmp,thres=rms_thres,min_dist=int(fwhm_chan*.5))
		
		return peak_indices,rms		

# find peaks in velocity space spectra

def find_vel_peaks(velocity,intensity,fwhm,sigma=3,width_tweak=1.0):
		
		'''
		find peaks in the intensity array more than 3 sigma (optionally adjustable) and return the indices of those peaks and the rms
		'''
		
		#figure out how many channels a typical line will span
		
		line_chan = int(dV * 5 * width_tweak)
				
		#So this is going to be a multistep iterative process to find peaks, remove them temporarily, and then find the RMS, and repeat until the RMS isn't changing.  Then we find all the peaks above the threshold sigma level.
		
		intensity_tmp = np.asarray(intensity)
		intensity_mask = np.asarray(intensity)
				
		converged = False
		
		rms = np.inf
		
		while converged == False:
					
			peak_indices_tmp = peakutils.indexes(intensity_tmp,thres=0.99)
			
			if len(peak_indices_tmp) == 0:
			
				converged = True 
				
			if len(peak_indices_tmp) < 0.5*len(intensity):
			
				converged = True
			
			#now we remove the channels that have those lines in them
			
			#first, figure out the channels to drop
			
			drops = []
			
			for x in range(len(peak_indices_tmp)):
			
					ll = peak_indices_tmp[x]-line_chan
					
					if ll < 0:
					
						ll = 0
						
					ul = peak_indices_tmp[x]+line_chan+1
					
					if ul > len(intensity_tmp):
						
						ul = len(intensity_tmp)
			
					for z in range(ll,ul):
					
						drops.append(z)
						
			#use those to make a mask			
			
			mask = np.ones(len(intensity_tmp), dtype=bool)
			
			mask[drops] = False			
			
			#mask out those peaks
			
			intensity_tmp = intensity_tmp[mask]
			
			intensity_mask = intensity_mask[mask]
			
			new_rms = np.sqrt(np.nanmean(intensity_tmp**2))
			
			#if it is the first run, set the rms to the new rms and continue
						
			if rms == np.inf:
			
				rms = new_rms
			
			#otherwise if there are no lines left above x sigma, stop
			
			elif np.amax(intensity_tmp) < new_rms*5:
			
				rms = new_rms
			
				converged = True
				
			else:
			
				rms = new_rms
				
		#now that we know the actual rms, we can find all the peaks above the threshold, and we have to calculate a real threshold, as it is a normalized intensity (percentage).  
		
		intensity_tmp = np.asarray(intensity)
		
		max = np.amax(intensity_tmp)
		min = np.amin(intensity_tmp)
		
		#calc the percentage threshold that the rms sits at
		
		rms_thres = (sigma*rms - min)/(max - min)

		peak_indices = peakutils.indexes(intensity_tmp,thres=rms_thres,min_dist=int(fwhm*.5))
		
		return peak_indices,rms	
		
# find peaks in the intensity array more than 3 sigma (optionally adjustable) and return the indices of those peaks

def find_sim_peaks(frequency,intensity,min_sep):

	'''
	finds peaks in the intensity array that are separated by at least min_sep and returns the indices of those peaks	
	'''
	
	#Here's the problem.  Min_sep is given in km/s.  Our input frequency array is not in km/s and it's not uniformly spaced in km/s.  So we first need to create a temporary uniformly spaced array in velocity, resample both frequency and intensity onto that, then find the peaks, find the corresponding frequencies, and return the indices of those in the original array.
	
	#get the max and the min frequency
	
	max_f = np.amax(frequency)
	min_f = np.amin(frequency)
	cfreq = (max_f + min_f)/2
	
	#get the finest spacing in velocity space - it will be at the highest end of the spectrum.  We assume we're working with a properly set resolution.
	
	v_res = res*ckm/max_f
	
	#get the total range in velocity space that we're spanning, setting the central frequency as v = 0.
	
	v_span = (max_f - min_f) * ckm/(cfreq)
	
	#figure out how many channels we'll need
	
	nchans = int(v_span / v_res)
	
	#make a uniformly sampled array in velocity space
	
	v_samp = np.linspace(-v_span/2,v_span/2,num=nchans,endpoint=True)
	
	#convert this to frequency space
	
	f_samp = np.copy(v_samp)
	
	f_samp = cfreq + v_samp*cfreq/ckm
	
	#interpolate the simulation onto this
	
	int_samp = np.interp(f_samp,frequency,intensity,left=0.,right=0.)
	
	#figure out how many channels min_sep is in our uniformy sampled spectrum
	
	chan_sep = min_sep/v_res
	
	#find the peaks in the resampled spectrum
	
	indices_samp  = signal.find_peaks(int_samp,distance=chan_sep)
	
	#now we find the frequencies corresponding to those indices
	
	peak_freqs = f_samp[indices_samp[0]]
	
	#now we find the closets indices for those in the original frequency spectrum
	
	indices = [find_nearest(frequency,x) for x in peak_freqs]
	
	indices = np.asarray(indices)	
	
	return indices
	
		
# plot_peaks will plot the peaks, as well as the determined rms level and the baseline mask, optionally

def plot_peaks(frequency,intensity,peak_indices,rms,freq_mask=None,int_mask=None):

	plt.ion()	

	fig = plt.figure()
	ax = fig.add_subplot(111)

	minorLocator = AutoMinorLocator(5)
	plt.xlabel('Frequency (MHz)')
	plt.ylabel('Intensity (Probably Arbitrary)')

	plt.locator_params(nbins=4) #Use only 4 actual numbers on the x-axis
	ax.xaxis.set_minor_locator(minorLocator) #Let the program calculate some minor ticks from that

	ax.get_xaxis().get_major_formatter().set_scientific(False) #Don't let the x-axis go into scientific notation
	ax.get_xaxis().get_major_formatter().set_useOffset(False)
	
	ax.plot(frequency,intensity,color='black',label='obs',zorder=0)
	
	rms_x = np.copy(frequency)
	rms_y = np.copy(frequency)
	rms_y.fill(rms)
	
	ax.plot(rms_x,rms_y,color='red',label='rms',zorder=50)
	
	if freq_mask is None:
		pass
	else:
		ax.plot(freq_mask,int_mask,color='green',label='baseline',zorder=15)
	
	freq_peaks = []
	int_peaks = []
	
	for x in range(len(peak_indices)):
	
		freq_peaks.append(frequency[peak_indices[x]])
		int_peaks.append(intensity[peak_indices[x]])
	
	ax.scatter(freq_peaks,int_peaks,color='red',marker='x',label='peaks',zorder=25)
	
	ax.legend(loc='upper right')
	fig.canvas.draw()		

def find_nearest(array,value):		

	#idx = (np.abs(array-value)).argmin()
	
	idx = np.searchsorted(array, value, side="left")
	
	if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
	
		return idx-1
		
	else:
	
		return idx
	
#velocity_stack does a velocity stacking analysis using the current ll and ul, and the current simulation, using lines that are at least 0.1 sigma or larger, with the brightest simulated line scaled to be 1 sigma.

def velocity_stack(drops =[], flag_lines=False,flag_int_thresh = 5, print_flags = False, vel_width = 40, v_res = 0.1,plot_chunks=False,blank_lines=False,blank_keep_range=None,fix_y=False,line_stats=True,pdf=False,labels_size=12,figsize=(6,4),thick=1.0,plotlabel=None,ylims=None,calc_sigma=False,label_sigma=False,plot_sigma=False,use_sum=False,sum_width_extend=3,plot_sim_chunks=False,plot_sum_range=False,plot_sim_stack=False,stack_out=None,sim_out=None,npz_out=False,mf=False,mf_out=None,mf_vmult=5.,mf_label=None,mf_pdf=False,filter_range=[-2,2],mf_ylims=None,mf_return=False,mf_return_range=[-3*dV,3*dV]):

	if flag_lines is True and blank_lines is True:
	
		print('You have set both flag_lines and blank_lines to True.  Flag_lines will supercede this and blank_lines will do nothing.  You have been warned.')
	
	freq_local = np.copy(freq_obs)
	int_local = np.copy(int_obs)

	if use_sum is False:
	
		#find the simulation indices where the peaks are
	
		peak_indices = find_sim_peaks(freq_sim,int_sim,dV)
		
		#find the frequencies and intensities corresponding to those peaks
	
		peak_freqs = freq_sim[peak_indices]
		peak_ints = int_sim[peak_indices]		
	
	#if we're using the sum, we'll use integrated flux over the summed signal instead
	
	if use_sum is True:
	
		#find the peaks
	
		peak_indices = find_sim_peaks(freq_sum,int_sum,dV*sum_width_extend)
	
		peak_freqs = freq_sum[peak_indices]
		
		for x in range(len(peak_freqs)):
		
			#print('I found a line in the sum at {:.4f} MHz' .format(peak_freqs[x]))
		
			freq_idx = find_nearest(frequency,peak_freqs[x]+vlsr*peak_freqs[x]/ckm)
		
			peak_freqs[x] = frequency[freq_idx]-vlsr*peak_freqs[x]/ckm
			
			#print('I assigned that line to catalog frequency {:.4f}, which is at {:.4f} in the vlsr shifted spectrum.' .format(frequency[freq_idx],peak_freqs[x]))
		
		peak_ints = []
		
		#now we need to sum up for a good fraction on either side.
		
		for x in peak_freqs:
		
			#get the width we're working with in frequency space
		
			freq_width = dV*sum_width_extend*x/ckm
		
			#find the index half that width down and up from the peak center
		
			tmp_ll = find_nearest(freq_sum,(x-freq_width/2))
			tmp_ul = find_nearest(freq_sum,(x+freq_width/2))
		
			#add up all the flux between those ranges and append it to peak_ints
		
			peak_ints.append(np.nansum(int_sum[tmp_ll:tmp_ul]))
	
	obs_chunks = {}
	
	for x in range(len(peak_freqs)):
	
		#get some temporary variables to hold the center frequency and peak intensity
		
		cfreq = peak_freqs[x]
		peak_int = peak_ints[x]
						
		#calculate the lower and upper frequencies corresponding to vel_width FWHM away from the center frequency.
		
		#first we find out how far up and down we need to span in frequency space
		
		if mf is True:
		
			freq_width = vel_width*dV*cfreq/ckm*mf_vmult
		
		else:
		
			freq_width = vel_width*dV*cfreq/ckm
		
		#get the lower frequency we need
		
		l_freq = cfreq - freq_width
		
		#get the upper frequency we need
		
		u_freq = cfreq + freq_width
		
		#find the indexes in the observation closest to those frequencies
		
		l_idx = find_nearest(freq_local,l_freq)
		u_idx = find_nearest(freq_local,u_freq)
		
		#if we're going to be plotting simulations, then sort those out.
		
		if use_sum is True:
		
			sim_l_idx = find_nearest(freq_sum,l_freq)
			sim_u_idx = find_nearest(freq_sum,u_freq)
			
		else:
		
			sim_l_idx = find_nearest(freq_sim,l_freq)
			sim_u_idx = find_nearest(freq_sim,u_freq)				
		
		#create and store an ObsChunk

		if use_sum is True:
	
			obs_chunks[x] = ObsChunk(np.copy(freq_local[l_idx:u_idx]),np.copy(int_local[l_idx:u_idx]),cfreq,peak_int,x,freq_sim=np.copy(freq_sum[sim_l_idx:sim_u_idx]),int_sim=np.copy(int_sum[sim_l_idx:sim_u_idx]))
			
		else:
		
			obs_chunks[x] = ObsChunk(np.copy(freq_local[l_idx:u_idx]),np.copy(int_local[l_idx:u_idx]),cfreq,peak_int,x,freq_sim=np.copy(freq_sim[sim_l_idx:sim_u_idx]),int_sim=np.copy(int_sim[sim_l_idx:sim_u_idx]))

	#for iterating convenience, we make an obs_list
	
	obs_list = []
	
	for obs in obs_chunks:
	
		obs_list.append(obs_chunks[obs])	
		
	#now we do some stupid checking
		
	for obs in obs_list:
		
		#if the flag is already tripped, just move on.  Below, as soon as a flag is tripped we move on.
		
		if obs.flag is True:
		
			continue
			
		#we'll check to see if there is any data within 0.5*dV of the line.  If not, then we're going to flag it.		
		
		#create an array that is a copy of the frequencies in the chunk
		
		diffs = np.copy(obs.frequency)
		
		#subtract the line frequency from all of those
		
		diffs -= obs.cfreq
		
		#get the absolute value
		
		diffs = abs(diffs)
		
		#then if the minimum value of that array isn't less than 0.5*dV, we flag it
		
		if np.amin(diffs) > 0.5*dV:
	
			obs.flag = True
			
			continue
			
		#make sure the len of the array isn't something dumb, like 0
		
		if len(obs.frequency) == 0:
	
			obs.flag = True
			
			continue
			
		#drop anything corresponding to the drop list
		
		if obs.tag in drops:
		
			obs.flag = True
			
			continue
			
		#if we're flagging windows that have other lines in them, do that.

		if flag_lines is True:
		
			obs.intensity[obs.intensity > flag_int_thresh*obs.rms] = np.nan
					
				
		#if we're just blanking some lines:
		
		if blank_lines is True:
		
			if blank_keep_range is None:
		
				obs.intensity[abs(obs.intensity) > flag_int_thresh*obs.rms] = np.nan
				
			else:
			
				#we need to figure out what range *not* to blank in from -x km/s to +y km/s from blank_keep_range = [-x,+y]
				
				#first, get the frequencies associated with the lower and upper bounds
				
				l_freq = obs.cfreq + blank_keep_range[0]*obs.cfreq/ckm #here, blank_keep_range[0] should be negative
				u_freq = obs.cfreq + blank_keep_range[1]*obs.cfreq/ckm	

				#now find those indices
				
				l_idx = find_nearest(obs.frequency,l_freq)
				u_idx = find_nearest(obs.frequency,u_freq)
				
				l_sim_idx = find_nearest(obs.freq_sim,l_freq)
				u_sim_idx = find_nearest(obs.freq_sim,u_freq)				
				
				#chunk out this portion to keep it safe
				
				tmp_chunk = np.copy(obs.intensity[l_idx:u_idx])
				tmp_sim_chunk = np.copy(obs.int_sim[l_sim_idx:u_sim_idx])
				
				obs.intensity[l_idx:u_idx] = np.nan
				obs.int_sim[l_sim_idx:u_sim_idx] = np.nan
				
				obs.set_rms()
				
# 				tmp_int_a = np.copy(obs.intensity)
# 				tmp_int_b = np.copy(obs.int_sim)				
				
				#blank obs.intensity
				
				obs.intensity[abs(obs.intensity) > flag_int_thresh*obs.rms] = np.nan
				obs.int_sim[abs(obs.int_sim) > 0.0] = np.nan
				
# 				tmp_int_c = np.copy(obs.intensity)
# 				tmp_int_d = np.copy(obs.int_sim)
				
				#chunk the safe range back in
				
				obs.intensity[l_idx:u_idx] = np.copy(tmp_chunk)
				obs.int_sim[l_sim_idx:u_sim_idx] = np.copy(tmp_sim_chunk)
				
# 				plt.close('freq vs int original')				
# 				plt.close('freq_sim vs int_sim original')				
# 				plt.close('freq vs int blanked')				
# 				plt.close('freq_sim vs int_sim blanked')				
# 				plt.close('freq vs int stitched')
# 				plt.close('freq_sim vs int_sim stitched')

# 				fig = plt.figure(num='freq vs int original')
# 
# 				plt.plot(obs.frequency,tmp_int_a,color='black')
# 				
# 				fig = plt.figure(num='freq_sim vs int_sim original')
# 				
# 				plt.plot(obs.freq_sim,tmp_int_b)
# 				
# 				fig = plt.figure(num='freq vs int blanked')
# 				
# 				plt.plot(obs.frequency,tmp_int_c)	
# 				
# 				fig = plt.figure(num='freq_sim vs int_sim blanked')
# 				
# 				plt.plot(obs.freq_sim,tmp_int_d)			
# 				
# 				fig = plt.figure(num='freq vs int stitched')
# 				
# 				plt.plot(obs.frequency,obs.intensity)	
# 				
# 				fig = plt.figure(num='freq_sim vs int_sim stitched')
# 				
# 				plt.plot(obs.freq_sim,obs.int_sim)										
				
# 				obs.set_rms()
				

								
# 				print(np.nanmin(obs.frequency),np.nanmax(obs.frequency))
# 				print(obs.cfreq)
# 				print('\n\n')
				
				
				

	#now we have to figure out the weights for the arrays.  We start by finding the maximum line height.
		
	max_int = max(peak_ints)
		
	#now we divide all our obs chunks by that, and then weight them down by their rms^2
		
	for obs in obs_list:
	
		if obs.flag is False:
	
			obs.weight = obs.peak_int/max_int
			obs.weight /= obs.rms**2	
			obs.int_weighted = obs.intensity * obs.weight
			obs.int_sim_weighted = obs.int_sim * obs.weight
		
	#ok, now we need to generate a velocity array to interpolate everything onto, using the specified number of FWHMs, dV, and the desired resolution.
	
	#calculate the velocity bounds
	
	if mf is True:
	
		l_vel = -vel_width*dV*mf_vmult
		u_vel = vel_width*dV*mf_vmult	
	
	else:
	
		l_vel = -vel_width*dV
		u_vel = vel_width*dV
	
	#generate the array
	
	velocity_avg = np.arange(l_vel,u_vel,v_res)
	
	#go through all the chunks and resample them, setting anything that is outside the range we asked for to be nans.
		
	for obs in obs_list:
	
		if obs.flag is False:
	
			obs.int_samp = np.interp(velocity_avg,obs.velocity,obs.int_weighted,left=np.nan,right=np.nan)			
			obs.int_sim_samp = np.interp(velocity_avg,obs.sim_velocity,obs.int_sim_weighted,left=np.nan,right=np.nan)

	#ok, now we loop through all the chunks and add them to a list, then convert to an numpy array.  We have to do the same thing w/ RMS values to allow for proper division.
					
	interped_ints = []
	interped_rms = []
	interped_sim_ints = []
	
	i = 0
	
	for obs in obs_list:
	
		if obs.flag is False:
		
			i+=1
		
			interped_ints.append(obs.int_samp)
			interped_rms.append(obs.rms)		
			interped_sim_ints.append(obs.int_sim_samp)
			
	interped_ints = np.asarray(interped_ints)
	interped_rms = np.asarray(interped_rms)
	interped_sim_ints = np.asarray(interped_sim_ints)
	
	#we're going to now need a point by point rms array, so that when we average up and ignore nans, we don't divide by extra values.
	
	rms_array = []
	
	for x in range(len(velocity_avg)):	
	
		rms_sum = 0
		
		for y in range(len(interped_rms)):
		
			if np.isnan(interped_ints[y][x]):
			
				continue
				
			else:
			
				rms_sum += interped_rms[y]**2	
				
		rms_array.append(rms_sum)
		
	rms_array = np.copy(rms_array)

	#now we add up the interped intensities, then divide that by the rms_array
	
	int_avg = np.nansum(interped_ints,axis=0)/rms_array
	
	int_sim_avg = np.nansum(interped_sim_ints,axis=0)/rms_array
	
	#drop some edge channels
	
	int_avg = int_avg[5:-5]
	int_sim_avg = int_sim_avg[5:-5]
	velocity_avg = velocity_avg[5:-5]
	
	#finally, we get the final rms, and divide out to get to snr. # We'll use the first and last 25% of the data (not anymore)

	
	rms_tmp = get_rms(int_avg)
	
	int_avg /= rms_tmp
	int_sim_avg /= rms_tmp

	#Plotting!!!!

	plt.ion()
	
	#set some defaults
	
	fontparams = {'size':labels_size, 'family':'sans-serif','sans-serif':['Helvetica']}
	
	plt.rc('font',**fontparams)
	plt.rc('mathtext', fontset='stixsans')
	
	#Plot the chunks
	
	if plot_chunks is True:
	
		#We do 16 panels per figure, so let's get the number of figures we'll have
		
		n_figs = math.ceil(len(obs_chunks)/16)
		
		#loop through and make the figures one at a time.  The stack will be figure 0.
		
		for x in range(0,n_figs+0):
		
			#make a figure
		
			plt.figure(x+1)			
			
			#now we loop over the appropriate range of chunks
			
			chunk_range = np.arange(0,16) + x*16
			
			#Generate a grid specification for a 4x4 plot
			
			gs = gridspec.GridSpec(4,4)
			
			#Now, we add subplots at the [row,column] we want in that grid
			
			ax1 = plt.subplot(gs[0,0])
			ax2 = plt.subplot(gs[0,1])
			ax3 = plt.subplot(gs[0,2])
			ax4 = plt.subplot(gs[0,3])
			ax5 = plt.subplot(gs[1,0])
			ax6 = plt.subplot(gs[1,1])
			ax7 = plt.subplot(gs[1,2])
			ax8 = plt.subplot(gs[1,3])
			ax9 = plt.subplot(gs[2,0])
			ax10 = plt.subplot(gs[2,1])
			ax11 = plt.subplot(gs[2,2])
			ax12 = plt.subplot(gs[2,3])
			ax13 = plt.subplot(gs[3,0])
			ax14 = plt.subplot(gs[3,1])
			ax15 = plt.subplot(gs[3,2])
			ax16 = plt.subplot(gs[3,3])
			
			#make a list so we can iterate over these
			
			axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16]
			
			#loop through and set the axis settings and the data to show
			
			for y in range(len(axes)):
			
				#get the current axis (cax) we're working with
				
				cax = axes[y]
				
				#get the chunk we're working with.  If we're over the limit, just be done with it.
				
				if chunk_range[y] > len(obs_list)-1:
				
					continue
				
				chunk = obs_list[chunk_range[y]]
				
				#set the color according to whether it was flagged
				
				if chunk.flag is True:
				
					color = 'red'
					
				else:
				
					color = 'black'
					
				if fix_y is not False:
				
					cax.set_ylim(fix_y[0],fix_y[1])
				
				cax.plot(chunk.frequency,chunk.intensity,color=color)
				
				if plot_sim_chunks is True:
				
					cax.plot(chunk.freq_sim,chunk.int_sim,color='green')
				
				cax.annotate('[{}]' .format(chunk.tag), xy=(0.1,0.8), xycoords='axes fraction', color=color)
				
				cax.locator_params(nbins=3) #Use only 3 actual numbers on the x-axis
				
				cax.get_xaxis().get_major_formatter().set_scientific(False) #Don't let the x-axis go into scientific notation
				cax.get_xaxis().get_major_formatter().set_useOffset(False)		
				
			plt.tight_layout()
			
			plt.show()				
										
	
	#Plot the stack
	
	nflags = 0
	nlines = 0
	
	for obs in obs_list:
	
		if obs.flag is True:
		
			nflags += 1
		
		else:
		
			nlines += 1
			
	SNR = np.nanmax(int_avg)	
	min_val = np.nanmin(int_avg)	
	
	plt.close(fig='stack')	
	
	fig = plt.figure(num='stack',figsize=figsize)
	
	ax_s = fig.add_subplot(111)

	minorLocator = AutoMinorLocator(5)
	plt.xlabel('Velocity (km/s)')
	plt.ylabel('SNR ($\sigma$)')

	plt.locator_params(nbins=4) #Use only 4 actual numbers on the x-axis
	ax_s.xaxis.set_minor_locator(minorLocator) #Let the program calculate some minor ticks from that

	ax_s.get_xaxis().get_major_formatter().set_scientific(False) #Don't let the x-axis go into scientific notation
	ax_s.get_xaxis().get_major_formatter().set_useOffset(False)
	
	if ylims != None:
	
		ax_s.set_ylim([ylims[0],ylims[1]])
	
	elif SNR < 10:
	
		ax_s.set_ylim([-4,12])
	
	else:
	
		ax_s.set_ylim([-0.1*SNR,1.3*SNR])
		
	xlower = -vel_width*dV
	xupper = vel_width*dV
	
	ax_s.set_xlim([xlower,xupper])
	
	if line_stats is True:
	
		ax_s.annotate('{} Lines Stacked\n{} Lines Rejected\nSNR: {:.1f}' .format(nlines,nflags,SNR), xy=(0.1,0.8), xycoords='axes fraction',label='legend')
		
	if plotlabel != None:
	
		align_arg = {'ha' : 'right'}
	
		ax_s.annotate(plotlabel, xy=(0.95,0.85), xycoords='axes fraction', color='black', **align_arg)						

	ax_s.plot(velocity_avg,int_avg,color='black',label='stacked',zorder=5,linewidth=thick,drawstyle='steps')
	
	if plot_sim_stack is True:
	
		ax_s.plot(velocity_avg,int_sim_avg,color='red',label='simulation',zorder=10,linewidth=thick,drawstyle='steps')
	
	#add some minor ticks
	
	ax_s.minorticks_on()
	
	ax_s.tick_params(axis='x', which='both', direction='in')
	ax_s.tick_params(axis='y', which='both', direction='in')
	
	#make sure ticks are on both mirror axes
	
	ax_s.yaxis.set_ticks_position('both')
	ax_s.xaxis.set_ticks_position('both')
	
	#load into globals
	
	global vel_stacked,int_stacked,vel_sim_stacked,int_sim_stacked
	
	vel_stacked = np.copy(velocity_avg)
	int_stacked = np.copy(int_avg)

	vel_sim_stacked = np.copy(velocity_avg)
	int_sim_stacked = np.copy(int_sim_avg)

	#write anything out as requested
	
	if stack_out is not None:

		if npz_out is False:
	
			with open(stack_out, 'w') as output:
		
				for x in range(len(vel_stacked)):
			
					output.write('{} {}\n' .format(vel_stacked[x],int_stacked[x]))
				
		else:
	
			np.savez(stack_out,vel_stacked=vel_stacked,int_stacked=int_stacked)
			
	if sim_out is not None:
	
		if npz_out is False:
		
			with open(sim_out, 'w') as output:
			
				for x in range(len(vel_stacked)):
				
					output.write('{} {}\n' .format(vel_stacked[x],int_sim_avg[x]))
					
		else:
		
			np.savez(stack_out,vel_stacked=vel_stacked,int_sim_avg=int_sim_avg)				
			
	#run a matched filter if one was requested
	
	if mf is True:
		
		int_mf = matched_filter(vel_stacked,int_stacked,int_sim_stacked,filter_range=filter_range)

		#plotting!	
	
		plt.close(fig='mf')
	
		fig = plt.figure(num='mf',figsize=figsize)
	
		ax_s = fig.add_subplot(111)

		minorLocator = AutoMinorLocator(5)
		plt.xlabel('Velocity (km/s)')
		plt.ylabel('Impulse Response ($\sigma$)')

		plt.locator_params(nbins=4) #Use only 4 actual numbers on the x-axis
		ax_s.xaxis.set_minor_locator(minorLocator) #Let the program calculate some minor ticks from that

		ax_s.get_xaxis().get_major_formatter().set_scientific(False) #Don't let the x-axis go into scientific notation
		ax_s.get_xaxis().get_major_formatter().set_useOffset(False)

		#clip down to the correct velocity range, but maintain the limits
	
		xmin = np.amin(vel_stacked)
		xmax = np.amax(vel_stacked)
	
		nchans = int(len(int_mf)/2)
		c_chan = int(len(vel_stacked)/2)
	
		new_vel = vel_stacked[c_chan-nchans:c_chan+nchans]
	
		if abs(len(new_vel) - len(int_mf)) == 1:
	
			if len(new_vel) > len(int_mf):
	
				new_vel = new_vel[:-1]
			
			else:
		
				int_mf = int_mf[:-1]
		#stats
	
		SNR = np.nanmax(int_mf)
		min_val = np.nanmin(int_mf)

		if mf_ylims is None:
		
			if SNR < 10:
	
				ax_s.set_ylim([-4,12])
	
			else:
		
				if 0.1*SNR < 4:
			
					ymin = -4
				
				else:
			
					ymin = -0.1*SNR
	
				ax_s.set_ylim([ymin,1.3*SNR])

		else:
		
			ax_s.set_ylim([mf_ylims[0],mf_ylims[1]])
			
		ax_s.set_xlim([xmin,xmax])
	
		if mf_label != None:
	
			align_arg = {'ha' : 'right'}
	
			ax_s.annotate(mf_label, xy=(0.95,0.75), xycoords='axes fraction', color='black', **align_arg)

		ax_s.plot(new_vel,int_mf,color='black',label='mf',zorder=5,linewidth=thick,drawstyle='steps')
	
		align_arg = {'ha' : 'right'}
	
		ax_s.annotate('Peak Impulse Response: {:.1f}$\sigma$' .format(SNR), xy=(0.95,0.85), xycoords='axes fraction', color='black', **align_arg)	
		
		#add some minor ticks
	
		ax_s.minorticks_on()
	
		ax_s.tick_params(axis='x', which='both', direction='in')
		ax_s.tick_params(axis='y', which='both', direction='in')
	
		#make sure ticks are on both mirror axes
	
		ax_s.yaxis.set_ticks_position('both')
		ax_s.xaxis.set_ticks_position('both')
	
		global velocity_mf,intensity_mf
	
		velocity_mf = np.copy(new_vel)
		intensity_mf = np.copy(int_mf)	
	
		if mf_out is not None:

			if npz_out is False:
	
				with open (mf_out, 'w') as output:
		
					for x in range(len(velocity_mf)):
			
						output.write('{} {}\n' .format(velocity_mf[x],intensity_mf[x]))
				
			else:
	
				np.savez(mf_out,velocity_mf=velocity_mf,intensity_mf=intensity_mf)
			
		plt.show()
	
		if mf_pdf is not False:
	
			plt.savefig(mf_pdf,format='pdf',transparent=True,bbox_inches='tight')			
	
	plt.figure('stack')
	
	#deal with calculating integrated sigmas now.  If we aren't going to do it, we're just done.  clean up and return
	
	if calc_sigma is False:
	
		plt.show()
		
		if pdf is not False:
	
			plt.savefig(pdf,format='pdf',transparent=True,bbox_inches='tight')
			
		if mf_return is True:
	
			mf_l_idx = find_nearest(velocity_mf,mf_return_range[0])
			mf_u_idx = find_nearest(velocity_mf,mf_return_range[1])
		
			print('Yes')
	
			return np.nanmax(intensity_mf[mf_l_idx:mf_u_idx])		
			
		else:	
	
			return
		
	#Otherwise, we find the indices for the given velocity range.  Ranges are specified in the calc_sigma = [0,1,2,3], where 0 and 1 are the lower and upper limits to the line you want to integrate, and 2 and 3 are the lower and upper limits for the region the noise is taken from.
	
	line_l_idx = find_nearest(velocity_avg,calc_sigma[0])
	line_u_idx = find_nearest(velocity_avg,calc_sigma[1])
	
	noise_l_idx = find_nearest(velocity_avg,calc_sigma[2])
	noise_u_idx = find_nearest(velocity_avg,calc_sigma[3])
	
	#get sigma from the noise
	
	sigma = get_rms(int_avg[noise_l_idx:noise_u_idx])
	
	#calculate the snr
			
	snr = np.sum(int_avg[line_l_idx:line_u_idx])/(sigma*np.sqrt(len(int_avg[line_l_idx:line_u_idx])))
	
	#if we aren't adding this to the graph, print the result
	
	if label_sigma is False:
	
		print('The SNR of the line integrated from {} - {} km/s is {} sigma.' .format(calc_sigma[0],calc_sigma[1],snr))
		
	#if we are labeling it, then do that	
		
	if label_sigma is True:
	
		align_arg = {'ha' : 'right'}
	
		ax_s.annotate('Int. SNR: {:.1f}$\sigma$' .format(snr), xy=(0.95,0.75), xycoords='axes fraction', color='black', **align_arg)	
	
	#if we're plotting the range we integrated over, do that	
		
	if plot_sigma is True:
	
		plt.axvspan(calc_sigma[0], calc_sigma[1], alpha=0.05, color='blue',zorder=0)
		plt.axvline(x=calc_sigma[0],alpha=0.2,color='blue',zorder=0)
		plt.axvline(x=calc_sigma[1],alpha=0.2,color='blue',zorder=0)				
	
	plt.show()
	
	if pdf is not False:
	
		plt.savefig(pdf,format='pdf',transparent=True,bbox_inches='tight')	
		
	if mf_return is True:
	
		mf_l_idx = find_nearest(velocity_mf,mf_return_range[0])
		mf_u_idx = find_nearest(velocity_mf,mf_return_range[1])
		
		print('Yes')
	
		return np.nanmax(intensity_mf[mf_l_idx:mf_u_idx])
	
	else:		

		return

#matched_filter pushes a matched filter through the given sets of stuff and returns the result

def matched_filter(x_obs,y_obs,y_filter,filter_range=[-2,2]):

	#we only want the central channels in the stack over a (user-defined) velocity range
	
	l_idx = find_nearest(x_obs,filter_range[0])
	u_idx = find_nearest(x_obs,filter_range[1])	
	
	int_mf = np.correlate(y_obs,y_filter[l_idx:u_idx],mode='valid')
	
	#blank out the central channels for doing the rms
	
	len_mf = len(int_mf)
	
	int_mf_tmp = np.copy(int_mf)
	
	int_mf_tmp[int(0.40*len_mf):int(0.60*len_mf)] = np.nan
	
	mf_rms = get_rms(int_mf_tmp)
	
	int_mf /= mf_rms	

	return int_mf

def cut_spectra(write=False,outputfile='cut.txt',n_fwhm=30):

	if gauss == True:
	
		print('cut_spectra() does not yet work with gaussian simulated spectra, only stick spectra.  Please resimulate with gauss=False and try again.')
		
		return

	freq_cut = []
	int_cut = []

	for x in freq_sim:

		i = (np.abs(freq_obs - x)).argmin()
	
		if abs(freq_obs[i]-x) < 1:
	
			#get resolution in MHz near point
		
			res_tmp = abs((freq_obs[i]-freq_obs[i+10])/10)
		
			#get resolution in km/s near point
		
			vel_res_tmp = abs(res_tmp*ckm/freq_obs[i])
		
			#calculate number of points equivalent to <<n_fwhm>> * FHWM
		
			pts_tmp = int(n_fwhm*dV/vel_res_tmp)
		
			#find the indexes at +/- 30 FWHM
		
			i_low = i-pts_tmp
		
			i_high = i+pts_tmp
		
			for x in range(i_low,i_high):
			
				freq_cut.append(freq_obs[x])
				int_cut.append(int_obs[x])
			
	if len(freq_cut) == 0:
	
		print('Something is wrong - no cuts were made.  Either there\'s no matching data or the script is broken.')	
			
	if write==False:
	
		return freq_cut,int_cut
		
	if write==True:
	
		with open(outputfile,'w') as output:
		
			for x in range(len(freq_cut)):
			
				output.write('{} {}\n' .format(freq_cut[x],int_cut[x]))
				
		return

#calc_tbg generates the background temperature wave.  The user should never call this directly.  A meta function to update tbg and re-run the simulation is provided as update()

def calc_tbg(tbg_params,tbg_type,tbg_range,frequencies):
		
	'''
	calc_tbg generates the background temperature wave.  The user should never call this directly.  A meta function to update tbg and re-run the simulation is provided as update().  If ranges are defined, any chunk that is not in the defined range defaults to 2.7 K.
	'''
	
	#figure out how many ranges we're dealing with
	
	n_ranges = int(len(tbg_range))
	
	#initialize a numpy array for tbg that is the same length as the requested array of covered frequencies, detecting the single value edge case
	
	if len(frequencies) == 1:
	
		tbg = np.asarray([0.])
		
	else:
	
		tbg = np.zeros_like(frequencies)

	#first, make sure to get tbg_params and tbg into list form if it's a single integer or float
	
	if type(tbg_params) == int or type(tbg_params) == float:
		
		tbg_params = [tbg_params]
			
	#Will need to run several different possible scenarios here
	
	if tbg_type == 'greybody':
	
		T = tbg_params[0]
		beta = tbg_params[1]
		tauref = tbg_params[2]
		taufreq = tbg_params[3]
		major = tbg_params[4]
		minor = tbg_params[5]
		
		#get tbg in Jy first
		
		#assume the major and minor axes were given in arcseconds, and calculate the solid angle of the beam
		
		omega = np.radians(major/3600.)*np.radians(minor/3600.)*np.pi/(4*np.log(2))
		
		tau = np.zeros_like(frequencies)
		
		tau = tauref * pow((frequencies*1e6/(taufreq*1e9)), beta)
		
		T_Jy_tmp = np.zeros_like(frequencies)
		
		T_Jy_tmp = omega * 1e23 * (1-np.exp(-tau)) *  2*h*pow(frequencies*1e6,3) / pow(cm,2) / np.expm1(h*frequencies*1e6/(k*T))
		
		#now to get it into Kelvin
		
		tmp_tbg = np.copy(T_Jy_tmp)
		
		with open('bunk.txt','w') as output:
		
			for x in range(len(frequencies)):
			
				output.write('{} {}\n' .format(frequencies[x],tmp_tbg[x]))
		
		omega = synth_beam[0]*synth_beam[1]

		#do the conversion

		tbg = (3.92E-8 * (frequencies*1E-3)**3 *omega/ (np.exp(0.048*frequencies*1E-3/tbg) - 1))

		tbg[tbg < 2.7] = 2.7
		
		return tbg 					
	
	if tbg_type == 'constant':
	
		#if there's no range specified...
	
		if n_ranges == 0:

			tmp_tbg = np.full_like(frequencies,tbg_params)
			
			tbg = tmp_tbg
	
			return tbg		
		
		else:
		
			#before we can add to the tbg array, we need to find the indices we'll be wanting to work with for each range.  So.
			
			for i in range(n_ranges):
			
				#first, get the ll and ul for the range in question.
			
				ll = tbg_range[i][0]
				ul = tbg_range[i][1]
				
				#next, let's get the indexes i_low and i_high this range covers in our frequencies.
				
				try:
					i_low = np.where(frequencies > ll)[0][0]
				except IndexError:
				
					#first, check and make sure this range is actually in the simulation.  If the ll is higher than the highest frequency in frequencies, just move on
					
					if frequencies[-1] < ll:
						continue
						
					#otherwise, if the simulation starts after the lower limit, then we just take the first point in frequencies
					
					else:
						i_low = 0
						
				try:
					i_high = np.where(frequencies > ul)[0][0]
				except IndexError:
				
					#If we can't find a point in frequencies that's above the upper limit, then the last point in the simulation is the upper limit index
					i_high = len(frequencies)
					
				#now that we have the indices i_low and i_high we are going to want to apply our tbg to, we can do as above
				
				#first, lets just get the constants for this particular range
					
				value = tbg_params[i]
				
				if type(value) == float or type(value) == int:
				
					value = [value]
				
				#now we cycle through the orders again
				
				for x in range(len(value)):
				
					#create a temporary array to handle what is going to be added to tbg for this order
					
					tmp_tbg = np.full_like(frequencies,value)
					
					tbg[i_low:i_high] += tmp_tbg[i_low:i_high]
					
			tbg[tbg == 0] = 2.7
			
			return tbg			
	
	
	if tbg_type == 'poly':
	
		#need to reverse sort that list so that the ordering is correct, because it's specified by the user as : y = Ax^2 + Bx + C as [A,B,C], but the script below wants it in the order [C, B, A].
		
		#tbg_params can be a list of lists, so need to invert each one inside it
		
		tbg_params_tmp = []
		
		for x in tbg_params:
		
			tbg_params_tmp.append(x[::-1])
	
		#if there's no range specified...
	
		if n_ranges == 0:

			#we'll cycle through each order individually

			for x in range(len(tbg_params_tmp)):
				
				#create a temporary array to handle what is going to be added to tbg for this order
				
				tmp_tbg = np.zeros_like(frequencies)
				
				tmp_tbg = tbg_params_tmp[x]*frequencies**x
				
				tbg += tmp_tbg
				
			tbg[tbg == 0] = 2.7
		
			return tbg
		
		else:
		
			#before we can add to the tbg array, we need to find the indices we'll be wanting to work with for each range.  So.
			
			for i in range(n_ranges):
			
				#first, get the ll and ul for the range in question.
			
				ll = tbg_range[i][0]
				ul = tbg_range[i][1]
				
				#next, let's get the indexes i_low and i_high this range covers in our frequencies.
				
				try:
					i_low = np.where(frequencies > ll)[0][0]
				except IndexError:
				
					#first, check and make sure this range is actually in the simulation.  If the ll is higher than the highest frequency in frequencies, just move on
					
					if frequencies[-1] < ll:
						continue
						
					#otherwise, if the simulation starts after the lower limit, then we just take the first point in frequencies
					
					else:
						i_low = 0
						
				try:
					i_high = np.where(frequencies > ul)[0][0]
				except IndexError:
				
					#If we can't find a point in frequencies that's above the upper limit, then the last point in the simulation is the upper limit index
					i_high = len(frequencies)					
						
				#now that we have the indices i_low and i_high we are going to want to apply our tbg to, we can do as above
				
				#first, lets just get the constants for this particular range
					
				constants = tbg_params_tmp[i]
				
				if type(constants) == float or type(constants) == int:
				
					constants = [constants]
				
				#now we cycle through the orders again
				
				for x in range(len(constants)):
				
					#create a temporary array to handle what is going to be added to tbg for this order
					
					tmp_tbg = np.zeros_like(frequencies)
					
					tmp_tbg = constants[x]*frequencies**x
					
					tbg[i_low:i_high] += tmp_tbg[i_low:i_high]		
					
			tbg[tbg == 0] = 2.7
			
			return tbg			
						
	elif tbg_type == 'power':
	
		#if there's no range specified...
	
		if n_ranges == 0 or n_ranges == 1:

			#create a temporary array to handle what is going to be added to tbg for this order
				
			tmp_tbg = np.zeros_like(frequencies)
		
			tmp_tbg = tbg_params[0]*frequencies**tbg_params[1] + tbg_params[2]
		
			tbg += tmp_tbg
		
			tbg[tbg == 0] = 2.7

			return tbg
		
		else:
		
			#before we can add to the tbg array, we need to find the indices we'll be wanting to work with for each range.  So.
			
			for i in range(n_ranges):
			
				#first, get the ll and ul for the range in question.
			
				ll = tbg_range[i][0]
				ul = tbg_range[i][1]
				
				#next, let's get the indexes i_low and i_high this range covers in our freq_sim.
				
				try:
					i_low = np.where(frequencies > ll)[0][0]
				except IndexError:
				
					#first, check and make sure this range is actually in the simulation.  If the ll is higher than the highest frequency in frequencies, just move on
					
					if frequencies[-1] < ll:
						continue
						
					#otherwise, if the simulation starts after the lower limit, then we just take the first point in frequencies
					
					else:
						i_low = 0
						
				try:
					i_high = np.where(frequencies > ul)[0][0]
				except IndexError:
				
					#If we can't find a point in the simulation that's above the upper limit, then the last point in the simulation is the upper limit index
					
					i_high = len(frequencies)					
						
				#now that we have the indices i_low and i_high we are going to want to apply our tbg to, we can do as above
				
				#first, lets just get the constants for this particular range
				
				constants = tbg_params[i]
			
				tmp_tbg = np.zeros_like(frequencies)
	
		
				tmp_tbg = constants[0]*frequencies**constants[1] + constants[2]				
					
				tbg[i_low:i_high] += tmp_tbg[i_low:i_high]			
					
			tbg[tbg == 0] = 2.7
			
			return tbg					
	
	elif tbg_type == 'sgrb2':
	
		tmp_tbg = np.zeros_like(frequencies)
		
		tmp_tbg = (10**(-1.06*np.log10(frequencies/1000) + 2.3))
		
		#fix the beam dilution to the continuum, using the known source size and dish size
		
		tbg = invert_beam(frequencies,tmp_tbg,20,100)
		
		return tbg
	
	else:
	
		print('Your Tbg calls are not set properly. This is likely because you have tbg_type set to something other than poly or power. Tbg has been defaulted to the CMB value of 2.7 K across your entire simulation.  Please see the Tbg documentation if this is not what you desire.')
		
		tbg = np.zeros_like(frequencies)
		
		tbg += 2.7
		
		return tbg
		
#check_tbg checks the value of Tbg at a given frequency

def check_tbg(frequency):

	freq_tmp = np.asarray([np.float64(frequency),np.float64(frequency)])
	
	tbg_tmp = calc_tbg(tbg_params,tbg_type,tbg_range,freq_tmp)
	
	return tbg_tmp[0]
				
				
#update is a general call to just re-run the simulation, if the user has modified any generalized variables themselves like Tbg stuff, or updated vlsr or dV, etc, without using mod functions.

def update():

	'''
	A general call to just re-run the simulation, if the user has modified any generalized variables themselves like Tbg stuff, or updated vlsr or dV, etc, without using mod functions.
	'''

	global freq_sim,int_sim,int_tau
		
	freq_tmp = np.copy(frequency)
	
	freq_tmp += (-vlsr)*freq_tmp/ckm		
	
	freq_sim,int_sim,int_tau = run_sim(freq_tmp,intensity,T,dV,C)
	
	clear_line('current')
	
	if gauss == False:

		lines['current'] = ax.vlines(freq_sim,0,int_sim,linestyle = '-',color = 'red',label='current',zorder=500) #Plot sticks from TA down to 0 at each point in freq.

	else:

		lines['current'] = ax.plot(freq_sim,int_sim,color = 'red',label='current',drawstyle=draw_style,zorder=500)
		
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		ax.legend(loc='upper right')
	fig.canvas.draw()
	
	save_results('last.results')

#reset_tbg just resets all the tbg parameters to the defaults and calls update()

def reset_tbg():

	global tbg_params,tbg_type,tbg_range
	
	tbg_params = 2.7
	tbg_type = 'poly'
	tbg_range = []
	update()

#get the rms of the spectrum, at least a good guess at it if it isn't line-confusion limited


def get_rms(intensity):

	tmp_int = np.copy(intensity)
	
	x = np.nanmax(tmp_int)
	
	rms = np.sqrt(np.nanmean(np.square(tmp_int)))
	
	while x > 3*rms:
	
		for chan in np.where(tmp_int > 3*rms)[0]:
			tmp_int[chan] = np.nan
			
		rms = np.sqrt(np.nanmean(np.square(tmp_int)))
		
		x = np.nanmax(tmp_int)

	return	rms


#get the rms from the observations over a specified chunk in frequency space

def get_obs_rms(ll,ul):

	l_idx = find_nearest(freq_obs,ll)
	u_idx = find_nearest(freq_obs,ul)

	tmp_i = int_obs[l_idx:u_idx]

	return get_rms(tmp_i)
	
#get the peak simulated value in a region

def get_sim_peak(ll,ul,absorption=False):

	#get the indices
	
	l_idx = find_nearest(freq_sim,ll)
	u_idx = find_nearest(freq_sim,ul)

	tmp_i = int_sim[l_idx:u_idx]
	
	if (l_idx == 0 and u_idx == 0):
	
		tmp_i = int_sim[0]
		
	else:
	
		tmp_i = int_sim[l_idx:u_idx]
	
	if absorption is True:
	
		return np.abs(np.amin(tmp_i))
		
	else:
	
		return np.amax(tmp_i)

#writes out the current simulation parameters - most useful for upper limits analyses

def write_sim_params(outfile=None,notes=None,rms=False,lines=False):

	if outfile is None:
	
		outfile = current.split('/')[-1].split('.')[0] + '.sim_params'
		
	peak_idx = np.argmax(np.copy(int_sim))	
		
	Q = get_Q(T)
	Qrot = get_Qrot(T)
	Qvib = get_Qvib(T)
		
	with open(outfile, 'w') as output:
	
		output.write('Catalog File:\t{}\n' .format(current))
		output.write('Spectrum File:\t{}\n' .format(spec))
		output.write('Column Density:\t{:.2e} cm-2\n' .format(C))
		output.write('Tex:\t\t\t{} K\n' .format(T))
		output.write('Tbg:\t\t\t{:.2f} K (@ {:.2f} MHz)\n' .format(check_tbg(freq_sim[peak_idx]),freq_sim[peak_idx]))
		output.write('dV:\t\t\t\t{:.2f} km/s\n' .format(dV))
		output.write('vlsr:\t\t\t{:.2f} km/s\n' .format(vlsr))
		output.write('Q({})\t\t\t{}\n' .format(T,int(Q)))
		output.write('Qrot({})\t\t{}\n' .format(T,int(Qrot)))
		output.write('Qvib({})\t\t{:.5f}\n' .format(T,Qvib))
		if vibs is not None:
			output.write('Vib Freqs:\t\t{}\n' .format(vibs))
		if planck is False:
			output.write('Dish Size:\t\t{} m\n' .format(dish_size))
		if planck is True:
			output.write('Synth Beam:\t\t{} arcsec\n' .format(synth_beam))
		output.write('Source Size:\t{} arcsec\n' .format(source_size))
		
		if rms is True:
		
			output.write('RMS in Range:\t{:.2f} mK\n' .format(get_obs_rms(ll,ul)*1000))
		
		if lines is True:
		
			output.write('\n\n++++++Simulated Lines++++++\n\n')
		
			print_array = print_lines(mol='current',thresh=float('-inf'),rest=True,mK=True,return_array = True)
			
			for x in range(len(print_array)):
	
				output.write('{}\n' .format(print_array[x]))	
		
			output.write('\n')
		
		
		if notes is not None:
		
			output.write('\n++++++Notes++++++\n\n')
		
			output.write('{}' .format(notes))
	
	return

#makes a postage stamp plot of the observations and current simulation (optional).  Yes, I used PP as an abbreviation.  Sue me.

def make_postage_plot(PP):

	#If the user specified both GHz = True and velocity = True, yell at them.
	
	if PP.GHz is True and PP.velocity is True:
	
		print('CRITICAL ERROR: You cannot specify both GHz = True and velocity = True.  Please choose only one.  Cannot continue.')
		
		return
		
	#close the old figure if there is one
	
	plt.close(fig=PP.fig_num)	

	#initialize a figure
	
	plt.ion()
	
	#if the user specified a size, use that
	
	if PP.figsize != None:
	
		figsize = PP.figsize
		
	#otherwise default to a 3:2 ratio	
		
	else:
	
		figsize = (12,8)
	
	fig = plt.figure(num=PP.fig_num,figsize=figsize)
	
	#set some defaults
	
	fontparams = {'size':PP.labels_size, 'family':'sans-serif','sans-serif':['Helvetica']}
	
	plt.rc('font',**fontparams)
	plt.rc('mathtext', fontset='stixsans')
	
	#get the number of figures total
	
	nfigs = len(PP.lines)	
	
	#figure out how many rows and columns we will have to make the grid.  Multistep logic.
	
	#if the user specified both, we use those and move on
	
	if PP.nrows != None and PP.ncols != None:
	
		nrows = PP.nrows
		ncols = PP.ncols
		
	#if only nrows is given...	
		
	elif PP.nrows != None:
	
		nrows = PP.nrows
		
		#get ncols, rounding up
		
		ncols = np.ceil(nfits/nrows)
		
	#if only ncols is given...
	
	elif PP.ncols != None:
	
		ncols = PP.ncols
		
		#get nrows, rounding up
		
		nrows = np.ceil(nfits/nrows)
	
	#Last, if the user has not specified either, it's easy, we just make a square.
	
	else:
	
		#get the square root of the number of figures and round up.  So a 5 spectrum plot ends up as a 3x3 grid.  User can fix as they want.
	
		nrows = int(np.ceil(np.sqrt(nfigs)))
		ncols = int(np.ceil(np.sqrt(nfigs)))
		
			
	#but we warn the users if they don't have enough!
	
	if nrows*ncols < nfigs:
	
		print('CRITICAL ERROR: The number of rows and columns you specified is not enough to make all the stamps you specified.\n')
		print('You asked for a {}x{} grid, which is {} stamps, but there are {} stamps given.  Cannot continue.' .format(nrows,ncols,nfigs))
		
		return	
		
	#make a gridspec for that size
	
	gs = gridspec.GridSpec(nrows,ncols)
	
	#if we're in velocity space, make everything tight because we won't have as many labels
	
	if PP.velocity is True:
	
		gs.update(hspace=0.1)
		gs.update(wspace=0.1)
		
	else:
	
		gs.update(hspace=0.2)
		gs.update(wspace=0.1)		
	
	#make subplots to add, using x as the dictionary key, and we'll make sure to index that the same as the figs list
	
	gs_dict = {}
	
	x = 0
	
	while x < len(PP.lines):
	
		for y in range(nrows):
		
			for z in range(ncols):
			
				gs_dict[x] = plt.subplot(gs[y,z])
				
				x += 1
				
	#get a handle on the number of subplots
	
	nplots = nrows*ncols
				
	#now loop through the list of postage stamps (PS)
	
	for x in range(nplots):
	
		#get the current axis (cax) we're working with
	
		cax = gs_dict[x]
		
		if x > len(PP.lines) - 1:
		
			cax.set_axis_off()
			
			continue
		
		#get the current postage stamp we're dealing with
		
		PS = PP.lines[x]
		
		#if y limits are set, then we use those
		
		if PS.ylims != None:
		
			cax.set_ylim(PS.ylims[0],PS.ylims[1])
		
		elif PP.ylims != None:
		
			cax.set_ylim(PP.ylims[0],PP.ylims[1])
					
			
		#set the local cfreq, and if indicated, augment it by the vlsr, so we're shifted to center
		
		if PP.vlsr != None:
		
			cfreq = PS.cfreq - PS.cfreq*PP.vlsr/ckm
			
		else:
		
			cfreq = PS.cfreq
					
		#ok, now need to chunk out the data we're working with.  Start with the observations.
		
		if PP.obs is True:
		
			#get the lower limit that's nwidths away from the center frequency
			
			ll = cfreq - PP.nwidths*dV*cfreq/ckm
			ul = cfreq + PP.nwidths*dV*cfreq/ckm
			
			#find the closest values in the observations for these
			
			l_idx = find_nearest(freq_obs,ll)
			u_idx = find_nearest(freq_obs,ul)
			
			#chunk everything out based on that
			
			freq_obs_tmp = np.copy(freq_obs[l_idx:u_idx])
			int_obs_tmp = np.copy(int_obs[l_idx:u_idx])
			
			#if we are resampling, do that now
			
			if PP.v_res is not None:
			
				#calculate the new resolution in MHz
				
				df = PP.v_res*cfreq/ckm
				
				#generate a new frequency array
				
				new_freq = np.arange(ll,ul,df)
				
				#interpolate the intensity data onto this
				
				new_int = np.interp(new_freq,freq_obs_tmp,int_obs_tmp,left=np.nan,right=np.nan)
				
				#set these back in
				
				freq_obs_tmp = np.copy(new_freq)
				int_obs_tmp = np.copy(new_int)		
				
						
			#dealing with alternative plotting and then add everything to the plot, using specified values
			
			if PP.GHz is True:
		
				freq_obs_tmp *= 1000
				
			if PP.milli is True:
			
				int_obs_tmp *= 1000	
				
			if PP.velocity is True:
		
				velocity_obs_tmp = np.zeros_like(freq_obs_tmp)
			
				velocity_obs_tmp += (freq_obs_tmp - cfreq)*ckm/cfreq 	
				
				cax.plot(velocity_obs_tmp,int_obs_tmp,color=PP.obs_color,drawstyle=PP.obs_draw,zorder=1,linewidth=PP.obs_thick)				
			
			else:
			
				cax.plot(freq_obs_tmp,int_obs_tmp,color=PP.obs_color,drawstyle=PP.obs_draw,zorder=1,linewidth=PP.obs_thick)		
			
			
		#Next the simulation
		
		if PP.sim is True:
				
			#find the closest values in the simulation for these
			
			l_idx = find_nearest(freq_sim,ll)
			u_idx = find_nearest(freq_sim,ul)
			
			#chunk everything out based on that
			
			freq_sim_tmp = np.copy(freq_sim[l_idx:u_idx])
			int_sim_tmp = np.copy(int_sim[l_idx:u_idx])	
			
			#if we are resampling, do that now
			
			if PP.v_res is not None:
			
				#calculate the new resolution in MHz
				
				df = PP.v_res*cfreq/ckm
				
				#generate a new frequency array
				
				new_freq = np.arange(ll,ul,df)
				
				#interpolate the intensity data onto this
				
				new_int = np.interp(new_freq,freq_sim_tmp,int_sim_tmp,left=np.nan,right=np.nan)
				
				#set these back in
				
				freq_sim_tmp = np.copy(new_freq)
				int_sim_tmp = np.copy(new_int)		
						
		#dealing with alternative plotting and then add everything to the plot, using specified values
				
			if PP.GHz is True:

				freq_sim_tmp *= 1000
				
			if PP.milli is True:
			
				int_sim_tmp *= 1000
				
			max_int = np.nanmax(int_sim_tmp)
			
				
				
			if PP.velocity is True:			
			
				velocity_sim_tmp = np.zeros_like(freq_sim_tmp)
			
				velocity_sim_tmp += (freq_sim_tmp - cfreq)*ckm/cfreq
				
				cax.plot(velocity_sim_tmp,int_sim_tmp,color=PP.sim_color,drawstyle=PP.sim_draw,zorder=2,linewidth=PP.sim_thick)	
				
				if PP.plot_error is True:
				
					cax.errorbar(0,max_int/2,xerr=PS.vel_error,color=PP.error_bar_color,zorder=3,linewidth=PP.error_bar_thick,fmt='none',capsize=PP.error_cap_size,alpha=0.5,capthick=PP.error_bar_thick)	
				
			else:
			
				cax.plot(freq_sim_tmp,int_sim_tmp,color=PP.sim_color,drawstyle=PP.sim_draw,zorder=2,linewidth=PP.sim_thick)
				
				if PP.plot_error is True:
				
					cax.errorbar(PS.cfreq,max_int/2,xerr=PS.error,color=PP.error_bar_color,zorder=3,linewidth=PP.error_bar_thick,fmt='none',capsize=PP.error_cap_size,alpha=0.5,capthick=PP.error_bar_thick)
		
		#loop over all the stored simulations we're being asked to plot
	
		for z1,z2,z3 in zip(PP.stored,PP.stored_thick,PP.stored_color):
	
			#find the closest values in the simulation for these
	
			l_idx = find_nearest(sim[z1].freq_sim,ll)
			u_idx = find_nearest(sim[z1].freq_sim,ul)	
		
			#chunk everything out based on that
	
			freq_sim_tmp = np.copy(sim[z1].freq_sim[l_idx:u_idx])
			int_sim_tmp = np.copy(sim[z1].int_sim[l_idx:u_idx])
		
			#if we are resampling, do that now
	
			if PP.v_res is not None:
	
				#calculate the new resolution in MHz
		
				df = PP.v_res*cfreq/ckm
		
				#generate a new frequency array
		
				new_freq = np.arange(ll,ul,df)
		
				#interpolate the intensity data onto this
		
				new_int = np.interp(new_freq,freq_sim_tmp,int_sim_tmp,left=np.nan,right=np.nan)
		
				#set these back in
		
				freq_sim_tmp = np.copy(new_freq)
				int_sim_tmp = np.copy(new_int)
	
			#dealing with alternative plotting and then add everything to the plot, using specified values
				
			if PP.GHz is True:

				freq_sim_tmp *= 1000
				
			if PP.milli is True:
			
				int_sim_tmp *= 1000
				
			max_int = np.nanmax(int_sim_tmp)
			
				
			if PP.velocity is True:			
			
				velocity_sim_tmp = np.zeros_like(freq_sim_tmp)
			
				velocity_sim_tmp += (freq_sim_tmp - cfreq)*ckm/cfreq
				
				cax.plot(velocity_sim_tmp,int_sim_tmp,color=z3,drawstyle=PP.sim_draw,zorder=2,linewidth=z2)	
							
			else:
			
				cax.plot(freq_sim_tmp,int_sim_tmp,color=z3,drawstyle=PP.sim_draw,zorder=2,linewidth=z2)
		
		#Next the sum
		
		if PP.sum is True:
				
			#find the closest values in the simulation for these
			
			l_idx = find_nearest(freq_sum,ll)
			u_idx = find_nearest(freq_sum,ul)
			
			#chunk everything out based on that
			
			freq_sim_tmp = np.copy(freq_sum[l_idx:u_idx])
			int_sim_tmp = np.copy(int_sum[l_idx:u_idx])	
			
			#if we are resampling, do that now
			
			if PP.v_res is not None:
			
				#calculate the new resolution in MHz
				
				df = PP.v_res*cfreq/ckm
				
				#generate a new frequency array
				
				new_freq = np.arange(ll,ul,df)
				
				#interpolate the intensity data onto this
				
				new_int = np.interp(new_freq,freq_sim_tmp,int_sim_tmp,left=np.nan,right=np.nan)
				
				#set these back in
				
				freq_sim_tmp = np.copy(new_freq)
				int_sim_tmp = np.copy(new_int)		
						
		#dealing with alternative plotting and then add everything to the plot, using specified values
				
			if PP.GHz is True:

				freq_sim_tmp *= 1000
				
			if PP.milli is True:
			
				int_sim_tmp *= 1000
				
			max_int = np.nanmax(int_sim_tmp)
			
				
				
			if PP.velocity is True:			
			
				velocity_sim_tmp = np.zeros_like(freq_sim_tmp)
			
				velocity_sim_tmp += (freq_sim_tmp - cfreq)*ckm/cfreq
				
				cax.plot(velocity_sim_tmp,int_sim_tmp,color=PP.sum_color,drawstyle=PP.sum_style,zorder=2,linewidth=PP.sum_thick)	
				
			else:
			
				cax.plot(freq_sim_tmp,int_sim_tmp,color=PP.sum_color,drawstyle=PP.sum_style,zorder=2,linewidth=PP.sum_thick)
						
		#Annotate the thing, if one has been provided
		
		if PS.label != None:
		
			if PS.box is False:
		
				cax.annotate('{}' .format(PS.label), xy=(0.05,0.93), xycoords='axes fraction', color='black',va='top')
				
			else:
			
				#first we define a white box for the label
	
				bbox_props = dict(boxstyle='square', fc='white', lw=0)
				
				cax.annotate('{}' .format(PS.label), xy=(0.05,0.93), xycoords='axes fraction', color='black', bbox = bbox_props,va='top')
			
		#Add an annotation for the restfrequency used for the velocity calculation if we're in velocity space
		
		if PP.velocity is True:
		
			if PP.GHz is True:
			
				units = 'GHz'
				
			else:
			
				units = 'MHz'
		
			cfreq_label = '{:.1f} {}' .format(PS.cfreq,units)
			
			align_arg = {'ha' : 'right'}
		
			if PS.box is False:
			
				cax.annotate(cfreq_label, xy=(0.95,0.93), xycoords='axes fraction', color='black', **align_arg, va='top' )
				
			else:
			
				bbox_props = dict(boxstyle='square', fc='white', lw=0)
				
				cax.annotate(cfreq_label, xy=(0.95,0.93), xycoords='axes fraction', color='black', **align_arg, bbox = bbox_props, va='top')								
			
		#set the number of x-ticks
				
		cax.locator_params(axis='x', tight=True, nbins=PP.xticks)
			
		#set the number of y-ticks		
		
		cax.locator_params(axis='y', tight=True, nbins=PP.yticks)
			
		#Don't let either axis go into scientific notation or have some sort of offset if we did things automagically above
		
		if PP.velocity is True:
		
			cax.get_xaxis().get_major_formatter().set_scientific(False)
			cax.get_xaxis().get_major_formatter().set_useOffset(False)	
		
		#add some minor ticks
		
		cax.minorticks_on()
		
		cax.tick_params(axis='x', which='minor', direction='in')
		cax.tick_params(axis='y', which='minor', direction='in')
		
		#make sure ticks are on both mirror axes
		
		cax.yaxis.set_ticks_position('both')
		cax.xaxis.set_ticks_position('both')
		
		#if we're in velocity space, turn off the excess labels.  We'll turn them back on later.
		
		if PP.velocity is True:
				
			cax.tick_params(direction='in',labelbottom=False,labelleft=False)
			
		#if we're in frequency space, just turn off the left labels
		
		else:
		
			cax.tick_params(direction='in',labelleft=False)	
		

	#deal with the x and y labels
	
	#if we're only labeling the lower left, then we do that and move on.
	
	if PP.lower_left_only is True:
	
		#first figuring out what the lower left plot is
	
		lower_left = nrows*ncols - (ncols)
	
		cax = gs_dict[lower_left]
	
		#x-axis logic
	
		if PP.GHz is True:
	
			cax.set_xlabel('Frequency (GHz)')
		
		elif PP.velocity is True:
	
			cax.set_xlabel('Velocity (km s$^{-1}$)')
		
		else:
	
			cax.set_xlabel('Frequency (MHz)')
		
		#y-axis logic
	
		if PP.milli is True:
	
			if planck is True:
		
				cax.set_ylabel('mJy beam$^{-1}$')
			
			else:
		
				cax.set_ylabel('T$_{\mbox{A}}$* (mK)')
			
		else:
	
			if planck is True:
		
				cax.set_ylabel('Jy beam$^{-1}$')
			
			else:
		
				cax.set_ylabel('T$_{\mbox{A}}$* (K)')	
			
		cax.tick_params(direction='in',labelbottom=True,labelleft=True)		
		
	#now labeling bottom row and left column only	
		
	else:
			
		#first figuring out what the lower left plot is
	
		lower_left = nrows*ncols - (ncols)
		
		#now iterating over all the ones in the bottom row
		
		for x in range(lower_left,nfigs):
		
			cax = gs_dict[x]	
			
			#x-axis logic
	
			if PP.GHz is True:
	
				cax.set_xlabel('Frequency (GHz)')
		
			elif PP.velocity is True:
	
				cax.set_xlabel('Velocity (km s$^{-1}$)')
		
			else:
	
				cax.set_xlabel('Frequency (MHz)')
				
			cax.tick_params(direction='in',labelbottom=True)
			
		#Now we figure out the column indexes.  They'll be 0, then 0+ncols, then 0+2*ncols, up to 0+(ncols*nrows)	
		
		l_idx = [0]
		
		if nrows > 1:
		
			for x in range(1,nrows):
			
				l_idx.append(ncols*x)
				
		#iterate over those indexes and set the labels
		
		for idx in l_idx:
		
			cax = gs_dict[idx]
			
			#y-axis logic
	
			if PP.milli is True:
	
				if planck is True:
		
					cax.set_ylabel('mJy beam$^{-1}$')
			
				else:
		
					cax.set_ylabel('T$_{\mbox{A}}$* (mK)')
			
			else:
	
				if planck is True:
		
					cax.set_ylabel('Jy beam$^{-1}$')
			
				else:
		
					cax.set_ylabel('T$_{\mbox{A}}$* (K)')	
			
			cax.tick_params(direction='in',labelleft=True)		
							
	#set the title, if one exists
	
	if PP.title != None:
	
		plt.title(PP.title)		
	
	plt.show()					
	
	if PP.pdf is not False:
	
		plt.savefig(PP.pdf,format='pdf',transparent=True,bbox_inches='tight')

	return

#makes a set of plots covering a specified total range in frequency space.

def make_range_plot(RP):

	#close the old figure if there is one
	
	plt.close(fig='Range Plot')
	
	#initialize a figure
	
	plt.ion()
	
	#if the user specified a size, use that
	
	if RP.figsize != None:
	
		figsize = RP.figsize
		
	#otherwise default to a 3:2 ratio	
		
	else:
	
		figsize = (12,8)
	
	fig = plt.figure(num='Range Plot',figsize=figsize)	
	
	#set some defaults
	
	fontparams = {'size':RP.labels_size, 'family':'sans-serif','sans-serif':['Helvetica']}
	
	plt.rc('font',**fontparams)
	plt.rc('mathtext', fontset='stixsans')
	
	#get the number of figures total.  We divide the total range covered by the range of each chunk, then round up to the nearest integer.
	
	nfigs = int(np.ceil(np.abs(RP.full_range[1]-RP.full_range[0])/RP.chunk_range))

	#figure out how many rows and columns we will have to make the grid.  Multistep logic.
	
	#if the user specified both, we use those and move on
	
	if RP.nrows != None and RP.ncols != None:
	
		nrows = PP.nrows
		ncols = PP.ncols
		
	#if only nrows is given...	
		
	elif RP.nrows != None:
	
		nrows = RP.nrows
		
		#get ncols, rounding up
		
		ncols = RP.ceil(nfits/nrows)
		
	#if only ncols is given...
	
	elif RP.ncols != None:
	
		ncols = RP.ncols
		
		#get nrows, rounding up
		
		nrows = np.ceil(nfits/nrows)
	
	#Last, if the user has not specified either, we just set ncols to 1 and move on.
	
	else:
	
		nrows = nfigs
		ncols = 1
					
	#but we warn the users if they don't have enough!
	
	if nrows*ncols < nfigs:
	
		print('CRITICAL ERROR: The number of rows and columns you specified is not enough to make all the stamps you specified.\n')
		print('You asked for a {}x{} grid, which is {} stamps, but there are {} stamps given.  Cannot continue.' .format(nrows,ncols,nfigs))
		
		return	
		
	#make a gridspec for that size and tighten it up
	
	gs = gridspec.GridSpec(nrows,ncols)
	
	gs.update(hspace=0.2)
	gs.update(wspace=0.1)	
	
	#make subplots to add, using x as the dictionary key, and we'll make sure to index that the same as the figs list
	
	gs_dict = {}
	
	x = 0
	
	while x < nfigs:
	
		for y in range(nrows):
		
			for z in range(ncols):
			
				gs_dict[x] = plt.subplot(gs[y,z])
				
				x += 1	
				
	#get a handle on the number of subplots
	
	nplots = nrows*ncols	
	
	#now loop through them, get the simulations we're working with, and plot them up
	
	for x in range(nplots):
	
		#get the current axis (cax) we're working with
	
		cax = gs_dict[x]
		
		#if y limits are set, then we use those
		
		if RP.ylims != None:
		
			cax.set_ylim(RP.ylims[0],RP.ylims[1])
			
		#now we set the x limits.  First the lower limit, which is the lower of the tuple from the full_range, plus the chunk_range if we've moved up past the first (x=0).
		
		plt_ll = RP.full_range[0] + RP.chunk_range*x	
		
		#now the upper limit.
		
		plt_ul = plt_ll + RP.chunk_range		
		
		#now set those in
		
		cax.set_xlim(plt_ll,plt_ul)
		
		#ok, now need to chunk out the data we're working with.  Start with the observations.
		
		if RP.obs is True:
			
			#we need to be more careful about our limits here, in case the upper end of full_range doesn't extend the full width of the final subplot
			
			#lower limit should always be fine
			
			ll = plt_ll
			
			#set the upper limit, but check if it's above the specified full_range, and if it is, bring it back down
			
			if plt_ul > RP.full_range[1]:
			
				ul = RP.full_range[1]
				
			else:
			
				ul = plt_ul
			
			l_idx = find_nearest(freq_obs,ll)
			u_idx = find_nearest(freq_obs,ul)
			
			#chunk everything out based on that
			
			freq_obs_tmp = np.copy(freq_obs[l_idx:u_idx])
			int_obs_tmp = np.copy(int_obs[l_idx:u_idx])	
						
			#dealing with alternative plotting and then add everything to the plot, using specified values
			
			if RP.GHz is True:
		
				freq_obs_tmp *= 1000
				
			if RP.milli is True:
			
				int_obs_tmp *= 1000	

			cax.plot(freq_obs_tmp,int_obs_tmp,color=RP.obs_color,drawstyle=RP.obs_draw,zorder=1,linewidth=RP.obs_thick)	
			
		#Next we loop through all the simulations. If there aren't any, then we just keep going.  We'll set up an iteration index so we can reference the list of colors
		
		i = 0
		
		if len(RP.sims) != 0:
		
			#loop through all the simulations
			
			for z in RP.sims:
			
				#check if this is the current simulation and if so, proceed accordingly
				
				if z == 'current':
				
					#find the closest values in the simulation for these
			
					l_idx = find_nearest(freq_sim,ll)
					u_idx = find_nearest(freq_sim,ul)
			
					#chunk everything out based on that
			
					freq_sim_tmp = np.copy(freq_sim[l_idx:u_idx])
					int_sim_tmp = np.copy(int_sim[l_idx:u_idx])
					
				#check if it's the summed simulation
				
				elif z == 'sum':
					
					#find the closest values in the simulation for these
			
					l_idx = find_nearest(freq_sum,ll)
					u_idx = find_nearest(freq_sum,ul)
			
					#chunk everything out based on that
			
					freq_sim_tmp = np.copy(freq_sum[l_idx:u_idx])
					int_sim_tmp = np.copy(int_sum[l_idx:u_idx])				
				
					
				#otherwise we have to dig it out of the archive, and we'll first check if it is even in there, continuing if not
				
				elif z not in sim:
				
					print("WARNING: There is no simulation labeled '{}' stored in memory.  Skipping." .format(x))
					
					continue
				
				else:
			
					#find the closest values in the simulation for these
			
					l_idx = find_nearest(sim[z].freq_sim,ll)
					u_idx = find_nearest(sim[z].freq_sim,ul)
			
					#chunk everything out based on that
			
					freq_sim_tmp = np.copy(sim[z].freq_sim[l_idx:u_idx])
					int_sim_tmp = np.copy(sim[z].int_sim[l_idx:u_idx])	
		
				#dealing with alternative plotting and then add everything to the plot, using specified values
			
				if RP.GHz is True:
		
					freq_sim_tmp *= 1000
				
				if RP.milli is True:
			
					int_sim_tmp *= 1000
				
				#plot it!
				
				zorder = i+3
				
				if z == 'sum':
				
					zorder = 2
			
				cax.plot(freq_sim_tmp,int_sim_tmp,color=RP.sim_colors[i],drawstyle=RP.sim_draw,zorder=zorder,linewidth=RP.sim_thicks[i])
			
				i += 1
			
		#add any markers
		
		if RP.markers is not None:
		
			#loop through all the markers, ignore any not in range, plot those that are.
			
			for mkr in RP.markers:
			
				if mkr.x > plt_ll and mkr.x < plt_ul:
				
					cax.plot((mkr.x+mkr.x_off),(mkr.y+mkr.y_off),marker=mkr.fmt,color=mkr.color,linewidth=mkr.linewidth,markersize=mkr.size)
		
		#add a legend to the plot, we'll have to iterate through the molecules and colors.
	
		#first we define a white box for the label
	
		bbox_props = dict(boxstyle='square', fc='white', lw=0)
		
		#get an index so we can move things down and get colors
		
		j = 0
	
		for y in RP.labels:
		
			#if we are only labeling the top plot, only do this if x is 0.
			
			if RP.label_top_only is True:
			
				if x != 0:
				
					continue
	
			cax.annotate('{}' .format(y), xy=(0.02,(0.95-RP.labels_shift-.05*(j*RP.labels_spacing))), xycoords='axes fraction', color=RP.sim_colors[j], bbox = bbox_props,zorder=100+j)
			
			j += 1
			
		#set the number of x-ticks
				
		cax.locator_params(axis='x', tight=True, nbins=RP.xticks)
			
		#set the number of y-ticks		
		
		cax.locator_params(axis='y', tight=True, nbins=RP.yticks)
			
		#Don't let either axis go into scientific notation or have some sort of offset if we did things automagically above
		
		cax.get_xaxis().get_major_formatter().set_scientific(False)
		cax.get_xaxis().get_major_formatter().set_useOffset(False)	
		
		#add some minor ticks
		
		cax.minorticks_on()
		
		cax.tick_params(axis='x', which='both', direction='in')
		cax.tick_params(axis='y', which='both', direction='in')
		
		#make sure ticks are on both mirror axes
		
		cax.yaxis.set_ticks_position('both')
		cax.xaxis.set_ticks_position('both')
		
		#ok, axis labeling
		
		if RP.GHz is True:
		
			xlabel = 'Frequency (GHz)'
			
		else:
		
			xlabel = 'Frequency (MHz)'
		
		if RP.label_bottom_only is True:
		
			cax = gs_dict[nfigs-1]
			
			cax.set_xlabel(xlabel)
				
		else:
		
			for x in gs_dict:
			
				cax = gs_dict[x]
						
				cax.set_xlabel(xlabel)
				
		#y-axis logic
	
		if RP.milli is True:
	
			if planck is True:
		
				cax.set_ylabel('mJy beam$^{-1}$')
			
			else:
		
				cax.set_ylabel('T$_{\mbox{A}}$* (mK)')
			
		else:
	
			if planck is True:
		
				cax.set_ylabel('Jy beam$^{-1}$')
			
			else:
		
				cax.set_ylabel('T$_{\mbox{A}}$* (K)')	
			
		cax.tick_params(direction='in',labelbottom=True,labelleft=True)	
		
	#set the title, if one exists
	
	if RP.title != None:
	
		plt.title(RP.title)		
	
	plt.show()					
	
	if RP.pdf is not False:
	
		plt.savefig(RP.pdf,format='pdf',transparent=True,bbox_inches='tight')

	return		

#makes a set of harmonic plots given a list of central frequencies and a range of frequency on either side of those frequencies.

def make_harmonic_plot(HP):

	#close the old figure if there is one
	
	plt.close(fig='Harmonic Plot')
	
	#initialize a figure
	
	plt.ion()
	
	#if the user specified a size, use that
	
	if HP.figsize != None:
	
		figsize = HP.figsize
		
	#otherwise default to a 3:2 ratio	
		
	else:
	
		figsize = (12,8)
	
	fig = plt.figure(num='Harmonic Plot',figsize=figsize)	
	
	#set some defaults
	
	fontparams = {'size':HP.labels_size, 'family':'sans-serif','sans-serif':['Helvetica']}
	
	plt.rc('font',**fontparams)
	plt.rc('mathtext', fontset='stixsans')
	
	#get the number of sub-figures total, which is just the number of center frequencies we'll use.
	
	nfigs = len(HP.cfreqs)
	nrows = nfigs
	ncols = 1
	
	#make a gridspec for that size and tighten it up
	
	gs = gridspec.GridSpec(nrows,ncols)
	
	gs.update(hspace=0.2)
	gs.update(wspace=0.1)	
	
	#make subplots to add, using x as the dictionary key, and we'll make sure to index that the same as the figs list
	
	gs_dict = {}
	
	for x in range(nfigs):
	
		gs_dict[x] = plt.subplot(gs[x,0])
		
	#ok, now just loop through and plot things up appropriately.
	
	for x in range(nfigs):
	
		#get the current axis (cax) we're working with
	
		cax = gs_dict[x]
		
		#if y limits are set, then we use those
		
		if HP.ylims != None:
		
			cax.set_ylim(HP.ylims[0],HP.ylims[1])
			
		#now we set the xlimits
		
		plt_ll = HP.cfreqs[x] - HP.chunk_range
		plt_ul = HP.cfreqs[x] + HP.chunk_range
		
		cax.set_xlim(-HP.chunk_range,HP.chunk_range)

		#ok, now need to chunk out the data we're working with.  Start with the observations.
		
		if HP.obs is True:
			
			l_idx = find_nearest(freq_obs,plt_ll)
			u_idx = find_nearest(freq_obs,plt_ul)
			
			#chunk everything out based on that
			
			freq_obs_tmp = np.copy(freq_obs[l_idx:u_idx])
			int_obs_tmp = np.copy(int_obs[l_idx:u_idx])	
						
			#dealing with alternative plotting and then add everything to the plot, using specified values
			
			if HP.GHz is True:
		
				freq_obs_tmp *= 1000
				
			if HP.milli is True:
			
				int_obs_tmp *= 1000	
				
			#now, regrid everything so that the cfreq = 0
			
			freq_obs_tmp -= HP.cfreqs[x]	

			cax.plot(freq_obs_tmp,int_obs_tmp,color=HP.obs_color,drawstyle=HP.obs_draw,zorder=1,linewidth=HP.obs_thick)	
			
		#Next we loop through all the simulations. If there aren't any, then we just keep going.  We'll set up an iteration index so we can reference the list of colors
		
		i = 0
		
		if len(HP.sims) != 0:
		
			#loop through all the simulations
			
			for z in HP.sims:
			
				#check if this is the current simulation and if so, proceed accordingly
				
				if z == 'current':
				
					#find the closest values in the simulation for these
			
					l_idx = find_nearest(freq_sim,plt_ll)
					u_idx = find_nearest(freq_sim,plt_ul)
			
					#chunk everything out based on that
			
					freq_sim_tmp = np.copy(freq_sim[l_idx:u_idx])
					int_sim_tmp = np.copy(int_sim[l_idx:u_idx])
					
				#check if it's the summed simulation
				
				elif z == 'sum':
					
					#find the closest values in the simulation for these
			
					l_idx = find_nearest(freq_sum,plt_ll)
					u_idx = find_nearest(freq_sum,plt_ul)
			
					#chunk everything out based on that
			
					freq_sim_tmp = np.copy(freq_sum[l_idx:u_idx])
					int_sim_tmp = np.copy(int_sum[l_idx:u_idx])				
				
					
				#otherwise we have to dig it out of the archive, and we'll first check if it is even in there, continuing if not
				
				elif z not in sim:
				
					print("WARNING: There is no simulation labeled '{}' stored in memory.  Skipping." .format(x))
					
					continue
				
				else:
			
					#find the closest values in the simulation for these
			
					l_idx = find_nearest(sim[z].freq_sim,plt_ll)
					u_idx = find_nearest(sim[z].freq_sim,plt_ul)
			
					#chunk everything out based on that
			
					freq_sim_tmp = np.copy(sim[z].freq_sim[l_idx:u_idx])
					int_sim_tmp = np.copy(sim[z].int_sim[l_idx:u_idx])	
		
				#dealing with alternative plotting and then add everything to the plot, using specified values
			
				if HP.GHz is True:
		
					freq_sim_tmp *= 1000
				
				if HP.milli is True:
			
					int_sim_tmp *= 1000
					
				#now, regrid everything so that the cfreq = 0
			
				freq_sim_tmp -= HP.cfreqs[x]					
				
				#plot it!
				
				zorder = i+3
				
				if z == 'sum':
				
					zorder = 2
			
				cax.plot(freq_sim_tmp,int_sim_tmp,color=HP.sim_colors[i],drawstyle=HP.sim_draw,zorder=zorder,linewidth=HP.sim_thicks[i])
			
				i += 1
						
		#set the number of x-ticks
				
		cax.locator_params(axis='x', tight=True, nbins=HP.xticks)
			
		#set the number of y-ticks		
		
		cax.locator_params(axis='y', tight=True, nbins=HP.yticks)
			
		#Don't let either axis go into scientific notation or have some sort of offset if we did things automagically above
		
		cax.get_xaxis().get_major_formatter().set_scientific(False)
		cax.get_xaxis().get_major_formatter().set_useOffset(False)	
		
		#add some minor ticks
		
		cax.minorticks_on()
		
		cax.tick_params(axis='x', which='both', direction='in')
		cax.tick_params(axis='y', which='both', direction='in')
		
		#make sure ticks are on both mirror axes
		
		cax.yaxis.set_ticks_position('both')
		cax.xaxis.set_ticks_position('both')	
				
		#y-axis logic
		
		if x == nfigs-1:
	
			if HP.milli is True:
	
				if planck is True:
		
					cax.set_ylabel('mJy beam$^{-1}$')
			
				else:
		
					cax.set_ylabel('T$_{\mbox{A}}$* (mK)')
			
			else:
	
				if planck is True:
		
					cax.set_ylabel('Jy beam$^{-1}$')
			
				else:
		
					cax.set_ylabel('T$_{\mbox{A}}$* (K)')	
				
		#set the bottom labels off unless we're on the last plot, we'll turn the bottom bottom ones on later.
		
		if x == nfigs-1:
		
			if HP.GHz is True:
		
				xlabel = 'Frequency (GHz)'
			
			else:
		
				xlabel = 'Delta Frequency (MHz)'
				
			cax.set_xlabel(xlabel)
			
			cax.tick_params(direction='in',labelbottom=True,labelleft=True)	
			
		else:
		
			cax.tick_params(direction='in',labelbottom=False,labelleft=True)
	
		#add a legend to the plot, we'll have to iterate through the molecules and colors.

		#first we define a white box for the label

		bbox_props = dict(boxstyle='square', fc='white', lw=0)

		cax.annotate('{:.2f}' .format(HP.cfreqs[x]), xy=(1.,0.5), xycoords='axes fraction', bbox = bbox_props,zorder=100)
		
	#set the title, if one exists

	if HP.title != None:

		plt.title(HP.title)		

	plt.show()					

	if HP.pdf is not False:

		plt.savefig(HP.pdf,format='pdf',transparent=True,bbox_inches='tight')

	return				

def get_brandon_tau(tau_freq):
	
	tmp_freq = np.copy(frequency)
	
	tmp_freq += (-vlsr)*tmp_freq/ckm

	run_sim(tmp_freq,intensity,T,dV,C,tau_get=tau_freq)
	
	return									

#saves the current observations to a numpy file of a specified name

def write_npz_spec(file):

	np.savez(file,freq_obs=freq_obs,int_obs=int_obs)
	
	return

#set_ulim_c attempts to automatically set the upper limit column density based on the rms and peak simulated intensity within the given limits that default to ll and ul

def set_ulim_c(x1,x2,level=None,absorption=False):

	global C
	
	modC(C)
	
	if level is None:
	
		modC(C*get_obs_rms(x1,x2)/get_sim_peak(x1,x2,absorption=absorption))
		
	else:
	
		modC(C*level/get_sim_peak(x1,x2,absorption=absorption))
	
	print('C: {:.2e}' .format(C))
	
	return

#finds the highest SNR line (or lines) in a simulation for use in getting upper limits - super experimental

def find_best_ulim(sep=dV,n=1,search_n=100,rms_spread=10,print_results=False,auto_limits=True):

	if auto_limits is True:
	
		autoset_limits()

	#find the indices of the peaks in the simulation

	peak_idx = find_sim_peaks(freq_sim,np.absolute(int_sim),sep)
	
	#get the absolute value of the intensity and frequency values there
	
	peak_ints = np.array([abs(int_sim[x]) for x in peak_idx])	
	peak_freqs = np.array([freq_sim[x] for x in peak_idx])
	
	#sort them based on intensity
	
	sort_idx = peak_ints.argsort()[::-1]
	
	peak_ints = peak_ints[sort_idx]
	peak_freqs = peak_freqs[sort_idx]
	peak_idx = peak_idx[sort_idx]
	peak_rms = np.copy(peak_ints)*0.
	peak_SNR = np.copy(peak_ints)*0.
	
	#Now go through and calculate RMS values at the top search_n*n points, looking rms_spread * the FWHM on either side of the line for the RMS
	
	search_range = min([len(peak_freqs),n*search_n])
	
	for i in range(search_range):
	
		dV_f = dV*peak_freqs[i]/ckm
		
		ll_i = peak_freqs[i] - rms_spread*dV_f
		ul_i = peak_freqs[i] + rms_spread*dV_f
		
		rms = get_obs_rms(ll_i,ul_i)
		
		#if the rms is NaN because there's no data...
		
		if math.isnan(rms) is True:
		
			peak_rms[i] = np.nan
			SNR = 0
			peak_SNR[i] = SNR
			
		else:
		
			peak_rms[i] = rms		
			SNR = peak_ints[i]/rms		
			peak_SNR[i] = SNR
		
	#Now find the maximum RMS values and re-sort by those
	
	trimmed_ints = peak_ints[:n*search_n]
	trimmed_freqs = peak_freqs[:n*search_n]
	trimmed_idx = peak_idx[:n*search_n]
	trimmed_rms = peak_rms[:n*search_n]
	trimmed_SNR = peak_SNR[:n*search_n]
	
	sort_idx = trimmed_SNR.argsort()[::-1]
	
	trimmed_ints = trimmed_ints[sort_idx]
	trimmed_freqs = trimmed_freqs[sort_idx]
	trimmed_idx = trimmed_idx[sort_idx]
	trimmed_rms = trimmed_rms[sort_idx]
	trimmed_SNR = trimmed_SNR[sort_idx]
	
	#print out things if that was asked for
	
	if print_results is True:
	
		print('Frequency\tIntensity\tRMS\tSNR\n')
		
		for i in range(n):
		
			print('{:.4f}\t{:.4f}\t{:.4f}\t{:.1f}\n' .format(trimmed_freqs[i],trimmed_ints[i],trimmed_rms[i],trimmed_SNR[i]))
	
	return trimmed_freqs[:n]
		
#generate an upper limit report based on the best ulim automatically

def autoset_ulim_c(rms_spread=10,print_results=True,make_pp=True,print_best_line=True,absorption=False,auto_limits=True):

	global ll,ul
	
	if auto_limits is True:
	
		autoset_limits()
	
	best_freq = find_best_ulim(auto_limits=auto_limits)[0]
	
	dV_f = dV*best_freq/ckm
	
	ll = float(best_freq - rms_spread*dV_f)
	ul = float(best_freq + rms_spread*dV_f)	
	
	set_ulim_c(ll,ul,absorption=absorption)
	
	if auto_limits is True:
	
		autoset_limits()
		
	modC(C)
	
	best_freq = find_best_ulim(auto_limits=auto_limits)[0]
	dV_f = dV*best_freq/ckm
	
	ll = float(best_freq - rms_spread*dV_f)
	ul = float(best_freq + rms_spread*dV_f)	
		
	set_ulim_c(ll,ul,absorption=absorption)	
	set_ulim_c(ll,ul,absorption=absorption)
	
	if print_results is True:
	
		print('Frequency: {:.2f}' .format(best_freq))
		
	if make_pp is True:
	
		PS = PostageStamp(best_freq)
		PP = PostagePlot([PS])

		make_postage_plot(PP)
		
	if print_best_line is True:
	
		print_lines(mK=True)
	
	return	

#subtract a simulation from and observation and return a quality metric

def get_subtraction(obsx,obsy,simx,simy,ll,ul,return_sim=False):

	#trim the observations down to size
	
	trim_obsy = trim_array(obsy,obsx,ll,ul)
	trim_obsx = trim_array(obsx,obsx,ll,ul)
	
	#interpolate the simulation onto the observations, to make sure there's no extras
	
	interped_obs = np.interp(simx,trim_obsx,trim_obsy,left=np.nan,right=np.nan)
	
	total_spec = interped_obs - simy
	
	total = np.sum(np.abs(total_spec))
	
	if return_sim is True:
	
		sub_sim = trim_obsy - interped_sim
		
		return total, trim_obsx, sub_sim
		
	else:

		return total
		
#do a quick and dirty peak find routine for the input observations, optionally writing out the results

def find_obs_peaks(outfile=None,sigma=5,start_chan=0,end_chan=None,chanstep=500,print_results=False,return_results=False):

	line_freqs = []
	line_ints = []
	rms_level = []

	done = False

	llpt = 0
	ulpt = chanstep

	while done is False:

		peak_indices, tmp_rms = find_peaks(freq_obs[llpt:ulpt],int_obs[llpt:ulpt],0.3,sigma=sigma)
	
		for x in peak_indices:
	
			line_freqs.append(freq_obs[x+llpt])
			line_ints.append(int_obs[x+llpt])
			rms_level.append(tmp_rms)	
	
		llpt += chanstep
		ulpt += chanstep
		
		if end_chan is None:
	
			if llpt > len(freq_obs):
	
				done = True
		
			if ulpt > len(freq_obs):
	
				done = True

		else:
		
			if llpt > end_chan:
	
				done = True
		
			if ulpt > end_chan:
	
				done = True		
				
	if print_results is True:
	
		print('Spectrum contains {} peaks above {}sigma.\n' .format(len(line_freqs),sigma))

	if outfile is not None:
	
		if type(outfile) is not str:
		
			print('Output file must be a string.')
			
			if return_results is True:
			
				return line_freqs,line_ints,rms_level
	
		with open(outfile, 'w') as output:

			for x in range(len(line_freqs)):
	
				output.write('{} {} {}\n' .format(line_freqs[x],line_ints[x],rms_level[x]))	

	if return_results is True:
	
		return line_freqs,line_ints,rms_level
	
#do a quick and dirty find routine on bright channels for the input observations, optionally writing out the results

def find_obs_brights(outfile=None,sigma=5,start_chan=0,end_chan=None,chanstep=500,print_results=False,return_results=False):

	i = 0

	done = False

	bright_freq = []
	bright_int = []

	llpt = 0
	ulpt = chanstep

	while done is False:

		tmp_rms = get_rms(int_obs[llpt:ulpt])
	
		i += len(np.where(int_obs[llpt:ulpt] > 5*tmp_rms)[0])
	
		for chan in np.where(int_obs[llpt:ulpt] > 5*tmp_rms)[0]:
	
			bright_freq.append(freq_obs[chan+llpt])
			bright_int.append(int_obs[chan+llpt])
	
		llpt += chanstep
		ulpt += chanstep
		
		if end_chan is None:
	
			if llpt > len(freq_obs):
	
				done = True
		
			if ulpt > len(freq_obs):
	
				done = True

		else:
		
			if llpt > end_chan:
	
				done = True
		
			if ulpt > end_chan:
	
				done = True			
				
	if print_results is True:
	
		print ('I found {} channels > 5 sigma out of {} total.' .format(i,len(int_obs)))
		print ('That is a filling factor of {:.2e}.' .format(i/len(int_obs)))
		print ('Alternatively, that is one bright channel for every {:.1f} dark ones.' .format(len(int_obs)/i))	

	if outfile is not None:
	
		if type(outfile) is not str:
		
			print('Output file must be a string.')
			
			if return_results is True:
			
				return bright_freq,bright_int
	
		with open(outfile, 'w') as output:

			for x in range(len(bright_freq)):
	
				output.write('{} {}\n' .format(bright_freq[x],bright_int[x]))	
	
	if return_results is True:	
	
		return bright_freq,bright_int


#############################################################
#						Custom Aliases	   					#
#############################################################

def mod10():

	modC(C*10)
	
	return
	
def mod2():

	modC(C*2)
	
	return
	
def mod12(): #this is NOT modC(C*12)

	modC(C*1.2)
	
	return
	
def mod_10():

	modC(C/10)
	
	return
	
def mod_2():

	modC(C/2)
	
	return
	
def mod_12(): #this is NOT modC(C/12)

	modC(C/1.2)
	
	return

#############################################################
#				Custom Loading for Common Sources			#
#############################################################	

#Change these paths to the local files on your machine.  

#load_mm1() loads the default MM1 pointing position extraction and parameters.

def load_mm1():

	global tbg_range,tbg_type,tbg_params,planck,synth_beam,T,dV,vlsr,C

	read_obs('/Users/Brett/Dropbox/Observational_Data/NGC6334I/alma/mm1/i/ngc6334i_MM1_i.txt')
	
	tbg_range = [[130000,132500],[143500,146000],[251000,252500],[266000,266600],[270400,271000],[279000,283000],[290000,295000],[302400,306100],[336000,340000],[348000,352000],[635000,690000],[698400,706000],[873500,881500],[890000,898000]]

	tbg_type = 'constant'

	tbg_params = [11.25,11.25,27.4,27.4,27.4,26.94,28.16,35.0,31.28,31.28,43.0,41.38,35.9,35.9]

	planck = True

	synth_beam = [0.26,0.26]

	T = 135

	dV = 3.2

	vlsr = -7

	C = 1E17

	autoset_limits()
	
#load_tmc1() loads GOTHAM.

def load_tmc1():

	global T,dV,vlsr,source_size,res,tbg_type

	read_obs('/Users/Brett/Dropbox/TMC1/tmc_all_gbt.npz')
	
	T = 8

	dV = 0.15

	vlsr = 5.82
	
	source_size = 30

	autoset_limits()
	
	res *= 2
	
	tbg_type = 'constant'
	
#load_tmc1_II() loads GOTHAM phase II.

def load_tmc1_II():

	global T,dV,vlsr,source_size,res,tbg_type

	read_obs('/Users/Brett/Dropbox/Observations/GBT/GOTHAM_Reduction/spectra/tmc1_spectra.npz')
	
	T = 8

	dV = 0.15

	vlsr = 5.82
	
	source_size = 30

	autoset_limits()
	
	res *= 2
	
	tbg_type = 'constant'	
	
#load_primos_cold() loads PRIMOS for absorption

def load_primos_cold():

	global T,dV,vlsr,source_size,res,tbg_type,dish_size

	read_obs('/Users/Brett/Dropbox/PRIMOS_DATA/blanked_primos_old.npz')
	
	T = 5

	dV = 9

	vlsr = 0
	
	source_size = 20
	dish_size = 100

	autoset_limits()
	
	tbg_type = 'sgrb2'	
	
#load_primos_hot() loads PRIMOS for compact emission

def load_primos_hot():

	global T,dV,vlsr,source_size,res,tbg_type,dish_size

	read_obs('/Users/Brett/Dropbox/PRIMOS_DATA/blanked_primos_old.npz')
	
	T = 80

	dV = 9

	vlsr = 0
	
	source_size = 5
	dish_size = 100

	autoset_limits()
	
	tbg_type = 'sgrb2'		
	
#load_tmc1() loads the default MM1 pointing position extraction and parameters.

def load_asai(source):

	global T,dV,vlsr,source_size,res,dish_size,tbg_type,tbg_range,tbg_params
	
	dish_size = 30
	source_size = 1E20
	tbg_range = []
	tbg_type = 'constant'
	tbg_params = [2.7]
	vlsr = 0.
	
	if source.lower() == 'barnard1' or source.lower() == 'b1':
	
		read_obs('/Users/Brett/Dropbox/Observations/ASAI/Barnard1/b1_concat_nodups.npz')
	
		T = 10

		dV = 0.8
		
		source_size = 1E20
	
	elif source.lower() == 'iras4a':
	
		read_obs('/Users/Brett/Dropbox/Observations/ASAI/IRAS4A/iras4a_concat_nodups.npz')
	
		T = 21

		dV = 5	
		
		source_size = 1E20

	elif source.lower() == 'l1157b1':
	
		read_obs('/Users/Brett/Dropbox/Observations/ASAI/L1157B1/l1157b1_concat_nodups.npz')
	
		T = 60

		dV = 8	
		
		source_size = 1E20	

	elif source.lower() == 'l1157mm':
	
		read_obs('/Users/Brett/Dropbox/Observations/ASAI/L1157mm/l1157mm_concat_nodups.npz')
	
		T = 60

		dV = 3
		
		source_size = 1E20
		
	elif source.lower() == 'l1448r2':
	
		read_obs('/Users/Brett/Dropbox/Observations/ASAI/L1448R2/l1448r2_concat_nodups.npz')
	
		T = 60

		dV = 8	
		
		source_size = 1E20
		
	elif source.lower() == 'l1527':
	
		read_obs('/Users/Brett/Dropbox/Observations/ASAI/L1527/l1527_concat_nodups.npz')
	
		T = 12

		dV = 0.5
		
		source_size = 1E20
		
	elif source.lower() == 'l1544':
	
		read_obs('/Users/Brett/Dropbox/Observations/ASAI/l1544/l1544_concat_nodups.npz')
	
		T = 10

		dV = 0.5
		
		source_size = 1E20
		
	elif source.lower() == 'svs13a':
	
		read_obs('/Users/Brett/Dropbox/Observations/ASAI/SVS13A/svs13a_concat_nodups.npz')
	
		T = 19

		dV = 3.0
		
		source_size = 0.3
		
	elif source.lower() == 'tmc1':
	
		read_obs('/Users/Brett/Dropbox/Observations/ASAI/TMC1/tmc1_concat_nodups.npz')
	
		T = 7

		dV = 0.3
		
		source_size = 1E20
		
	else:
	
		print('Source name not recognized.  Choose from: Barnard1, IRAS4A, L1157B1, L1157mm, L1448R2, L1527, L1544, SVS13A, TMC1')									
	
		return
	
	autoset_limits()
	
	return
	
#load_hexos() loads up a hexos data set, given a ton of options

def load_hexos(source,c=False,band=0,cr=False,hc=False):

	global tbg_type,tbg_params,tbg_range,dV,T,source_size,dish_size,vlsr

	sources = ['sgrb2','orionkl']
	
	if source == 'sgrb2n':
	
		source = 'sgrb2'
	
	path = '/Users/Brett/Dropbox/HEXOS_Data/formatted/'
	
	if c is True:
	
		cont_str = 'c' 
		
	else:
	
		cont_str = 'nc'
		
	if cr is True:
	
		cr_str = 'cr'
		
	if hc is True:
	
		hr_str = 'hc'
	
	if source.lower() not in sources:
	
		print("Only 'sgrb2' and 'orionkl' are valid specifiers for hexos quickload.")
		
		return
		
	dish_size = 3.5
	
	vlsr = 0.0
		
	if source.lower() == 'sgrb2':
	
		T = 280.
		
		dV = 8.
		
		source_size = 2.3
	
		path += 'sgr_hexos/' + cont_str + '/'
			
		if band == 0:
		
			path += 'hexos_sgrb2n_' + cont_str + '.npz'
			
		else:
		
			path += 'band' + band + '_sgrb2n_' + cont_str + '.tap.a.npz'
			
		tbg_type = 'poly'
	
		#First range is from 479600 - 1280200 and is y = 1.65327E-5*x - 3.10799 (inc. 2.7 CMB added in)
	
		range1 = [1.65327E-5,-3.10799]
	
		#Second range is from 1425500 - 1535200 and is y = 0*x + 16.19 (inc. 2.7 CMB added in)
	
		range2 = [0,16.19]
		
		#Third range is from 1573600 - 1907150 and is y = -7.03292E-6*x + 28.1471 (inc. 2.7 CMB added in)
	
		range3 = [-7.03292E-6,28.1471]		
	
		tbg_params = [range1,range2,range3]	
	
		tbg_range = [[479600,1280200],[1425500,1535200],[1573600,1907150]]				
		
	elif source.lower() == 'orionkl':
	
		dV = 6.5
		
		T = 200.
		
		source_size = 10.
	
		path += 'orionkl_hexos/' + cont_str + '/'
			
		if cr is False and hc is False:
		
			if band == 0:
			
				path += 'hexos_orionkl_' + cont_str + '.npz'
				
			else:
		
				path += 'band' + band + '_orionkl_' + cont_str + '.tap.a.npz'
			
			tbg_type = 'power'
			
			tbg_params = [8.2279E-14,2.3395,2.5501] #Power law fit to the non-continuum subtracted baseline, with 2.7 K added back in for the CMB.
			
			tbg_range = [[470000,1296000]]
			
		elif cr is True:
		
			if band == 0:
			
				path += 'cr/' + 'hexos_orionkl_' + cont_str + '_cr.npz'
				
			else:
		
				path += 'cr/' + 'band' + band + 'cr_orionkl_' + cont_str + '.tmb.a.npz'
			
			tbg_type = 'poly'
		
			#First range is from 1425900 - 1535100 and is y = 0.000012027*x - 0.8076 (inc. 2.7 CMB added in)
		
			range1 = [0.000012027,-0.8076]
		
			#Second range is from 1573300 - 1798247.5 and is y = 0.0000168957*x - 5.7911 (inc. 2.7 CMB added in)
		
			range2 = [0.0000168957,-5.7911]
		
			#Third range is from 1798247.5 - 1906600 and is y = 0.0000138264*x - 1.3533 (inc. 2.7 CMB added in)
		
			range3 = [0.0000138264,-1.3533]
		
			tbg_params = [range1,range2,range3]	
		
			tbg_range = [[1425900,1535100],[1573300,1798247.5],[1798247.5,1906600]]		
					
		elif hc is True:
		
			if band == 0:
			
				path += 'hc/' + 'hexos_orionkl_' + cont_str + '_hc.npz'
				
			else:
		
				path += 'hc/' + 'band' + band + 'hc_orionkl_' + cont_str + '.tmb.a.npz'
				
			tbg_type = 'poly'
		
			#First range is from 1425900 - 1702847.27 and is y = 0.0000181332*x - 9.86691 (inc. 2.7 CMB added in)
		
			range1 = [0.0000181332,-9.86691]
		
			#Second range is from 1702847.27 - 1798247.5 and is y = 0.000019472*x - 12.9821 (inc. 2.7 CMB added in)
		
			range2 = [0.000019472,-12.9821]
		
			tbg_params = [range1,range2]	
		
			tbg_range = [[1425900,1702847.27],[1702847.27,1798247.5]]			
			
	read_obs(path)
	
	autoset_limits()
		
	return
	

#load_belloche() loads up the IRAM 30m survey of Sgr B2N

def load_belloche():

	global T, dish_size, tbg_params, source_size, dV, vlsr, constant

	read_obs('/Users/Brett/Dropbox/PRIMOS_DATA/belloche_sgrb2.npz')
	autoset_limits()
	
	T = 120
	dish_size = 30
	
	tbg_type = 'constant'
	tbg_params = [5.2]
	tbg_range = []
	
	source_size = 2.2
	
	dV = 5.0
	
	vlsr = 0.0

	return

#meta function to print all the possible load options

def print_quickloads():

	quickloads = [
	
		'load_mm1()',
		'load_tmc1()',
		'load_primos_cold()',
		'load_primos_hot()',
		"load_asai('b1')",
		"load_asai('iras4a')",
		"load_asai('l1157b1')",
		"load_asai('l1157mm')",
		"load_asai('l1448r2')",
		"load_asai('l1527')",
		"load_asai('l1544')",
		"load_asai('svs13a')",
		"load_asai('tmc1')",
		'load_hexos(source,c=False,band=0,cr=False,hc=False)'
		'load_belloche()'
	
	]		
	
	for x in quickloads:
	
		print(x)
		
	return
					

#############################################################
#							Classes for Storing Results		#
#############################################################	

class Molecule(object):

	def __init__(self,name,catalog_file,tag,gup,glow,dof,error,qn1,qn2,qn3,qn4,qn5,qn6,elower,eupper,qns,logint,qn7,qn8,qn9,qn10,qn11,qn12,C,dV,T,CT,vlsr,frequency,freq_sim,intensity,int_sim,int_tau,aij,sijmu,vibs):
	
		self.name = name
		self.catalog_file = catalog_file
		self.tag = tag
		self.gup = gup
		self.glow = glow
		self.dof = dof
		self.error = error
		self.qn1 = qn1
		self.qn2 = qn2
		self.qn3 = qn3
		self.qn4 = qn4
		self.qn5 = qn5
		self.qn6 = qn6
		self.elower = elower
		self.eupper = eupper
		self.qns = qns
		self.logint = logint
		self.qn7 = qn7
		self.qn8 = qn8
		self.qn9 = qn9
		self.qn10 = qn10
		self.qn11 = qn11
		self.qn12 = qn12
		self.C = C
		self.dV = dV
		self.T = T
		self.CT = CT
		self.vlsr = vlsr
		self.frequency = frequency
		self.freq_sim = freq_sim
		self.intensity = intensity
		self.int_sim = int_sim
		self.int_tau = int_tau
		self.aij = aij
		self.sijmu = sijmu
		self.vibs = vibs
		
class ObsChunk(object):

	def __init__(self,frequency,intensity,cfreq,peak_int,tag,freq_sim=None,int_sim=None):
	
		self.frequency = frequency
		self.intensity = intensity
		self.velocity = None
		self.int_samp = None
		self.int_weighted = None
		self.cfreq = cfreq
		self.peak_int = peak_int
		self.rms = None
		self.flag = False
		self.weight = None
		self.tag = tag
		self.freq_sim = freq_sim
		self.int_sim = int_sim
		self.int_sim_samp = None
		self.int_sim_weighted = None
		self.sim_velocity = None
		
		self.set_flag()
		self.set_velocity()
		self.set_sim_velocity()
		
		self.set_rms()
		
		
		return			
		
	def set_flag(self):
	
		if len(self.frequency) < 2:
		
			self.flag = True		
			
		return	

	def set_velocity(self):
	
		if self.flag is True:
		
			return
	
		#make an array the same length as frequency
	
		velocity = np.zeros_like(self.frequency)
		
		velocity += (self.frequency - self.cfreq)*ckm/self.cfreq
		
		self.velocity = velocity
		
		return
		
	def set_sim_velocity(self):
	
		if self.flag is True:
		
			return
	
		#make an array the same length as frequency
	
		sim_velocity = np.zeros_like(self.freq_sim)
		
		sim_velocity += (self.freq_sim - self.cfreq)*ckm/self.cfreq
		
		self.sim_velocity = sim_velocity
		
		return		
		
	def set_rms(self):
	
		if self.flag is True:
		
			return
			
		self.rms = get_rms(self.intensity)
		
		return
		
class PostagePlot(object):

	def __init__(self,lines,nwidths=40,ylims=None,velocity=False,pdf=False,obs=True,sim=True,nrows=None,ncols=None,obs_color='Black',sim_color='Red',obs_draw='steps',sim_draw='steps',title=None,GHz=False,xticks=3,yticks=3,xlabel=None,ylabel=None,milli=False,vlsr=vlsr,v_res=None,figsize=None,labels_size=18,lower_left_only=False,obs_thick=1.0,sim_thick=1.0,error_bar_color='blue',error_bar_thick=1.0,plot_error=False,error_cap_size=2.,stored=[],stored_thick=[],stored_color=[],sum=False,sum_color='lime',sum_thick=1.,sum_style='steps',fig_num='Stamps'):
	
		self.lines = lines
		self.nwidths = nwidths
		self.ylims = ylims
		self.velocity = velocity
		self.pdf = pdf
		self.obs = obs
		self.sim = sim
		self.nrows = nrows
		self.ncols = ncols
		self.obs_color = obs_color
		self.sim_color = sim_color
		self.obs_draw = obs_draw
		self.sim_draw = sim_draw
		self.title = title
		self.GHz = GHz
		self.xticks = xticks
		self.yticks = yticks
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.milli = milli
		self.vlsr = vlsr
		self.v_res = v_res
		self.figsize = figsize
		self.labels_size = labels_size
		self.lower_left_only = lower_left_only
		self.obs_thick = obs_thick
		self.sim_thick = sim_thick
		self.error_bar_color = error_bar_color
		self.error_bar_thick = error_bar_thick
		self.error_cap_size = error_cap_size
		self.plot_error = plot_error
		self.stored = stored
		self.stored_thick = stored_thick
		self.stored_color = stored_color
		self.sum = sum
		self.sum_color = sum_color
		self.sum_thick = sum_thick
		self.sum_style = sum_style
		self.fig_num = fig_num
		
		return
		
class PostageStamp(object):

	def __init__(self,cfreq,error=None,label=None,box=False,ylims=None):
	
		self.cfreq = cfreq
		self.label = label
		self.error = error
		self.box = box
		self.vel_error = None
		self.ylims=ylims
		
		self.set_vel_error()
	
		return

	def set_vel_error(self):
	
		if self.error is None:
		
			return
			
		self.vel_error = self.error * ckm / self.cfreq
		
		return

class RangePlot(object):

	def __init__(self,full_range,chunk_range,ylims=None,pdf=False,obs=True,sims=['current'],nrows=None,ncols=None,obs_color='Black',sim_colors=['Red'],obs_draw='steps',sim_draw='steps',title=None,GHz=False,xticks=3,yticks=3,xlabel=None,ylabel=None,milli=False,figsize=None,labels_size=18,label_bottom_only=False,obs_thick=1.0,sim_thicks=[1.0],labels=[],labels_spacing=1,labels_shift=0,label_top_only=True,use_markers=False,markers=None):
	
		self.full_range = full_range
		self.chunk_range = chunk_range
		self.ylims = ylims
		self.pdf = pdf
		self.obs = obs
		self.sims = sims
		self.nrows = nrows
		self.ncols = ncols
		self.obs_color = obs_color
		self.sim_colors = sim_colors
		self.obs_draw = obs_draw
		self.sim_draw = sim_draw
		self.title = title
		self.GHz = GHz
		self.xticks = xticks
		self.yticks = yticks
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.milli = milli
		self.figsize = figsize
		self.labels_size = labels_size
		self.label_bottom_only = label_bottom_only
		self.obs_thick = obs_thick
		self.sim_thicks = sim_thicks
		self.labels = labels
		self.labels_spacing = labels_spacing
		self.labels_shift = labels_shift
		self.label_top_only = label_top_only
		self.markers = markers
				
		return
		
class RangeMarker(object):

	def __init__(self,x,y,x_off=0.,y_off=0.,fmt='x',color='black',linewidth='1.',size='3.'):
	
		self.x = x
		self.y = y
		self.x_off = x_off
		self.y_off = y_off
		self.fmt = fmt
		self.color = color
		self.linewidth = linewidth
		self.size = size
		
		return	

class HarmonicPlot(object):

	def __init__(self,cfreqs,chunk_range,ylims=None,pdf=False,obs=True,sims=[],obs_color='Black',sim_colors=[],obs_draw='steps',sim_draw='steps',title=None,GHz=False,xticks=5,yticks=5,xlabel=None,ylabel=None,milli=False,figsize=None,labels_size=18,obs_thick=1.0,sim_thicks=[1.0]):
	
		self.cfreqs = cfreqs
		self.chunk_range = chunk_range
		self.ylims = ylims
		self.pdf = pdf
		self.obs = obs
		self.sims = sims
		self.obs_color = obs_color
		self.sim_colors = sim_colors
		self.obs_draw = obs_draw
		self.sim_draw = sim_draw
		self.title = title
		self.GHz = GHz
		self.xticks = xticks
		self.yticks = yticks
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.milli = milli
		self.figsize = figsize
		self.labels_size = labels_size
		self.obs_thick = obs_thick
		self.sim_thicks = sim_thicks
				
		return	


				
#############################################################
#							Run Program						#
#############################################################