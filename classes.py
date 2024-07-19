# ----------------------------------------------------------------------------------
# Based on methodologies described in:
# Loomis, R.A. et al., Nat Astron 5, 188–196 (2021), DOI: 10.1038/s41550-020-01261-4
# Extends prior scripts for spectral simulation and MCMC inference.
# ----------------------------------------------------------------------------------

import numpy as np
from numpy import exp as exp
import time as tm

from constants import *
from functions import *

# Classes used in simulate_lte

class MolCat(object):

	def __init__(self, name, catalog_file, format='spcat', CT=300):
		self.name = name
		self.catalog_file = catalog_file
		self.format = format
		self.CT = CT
		self.tag = None
		self.gup = None
		self.glow = None
		self.dof = None
		self.error = None
		self.qns = None
		self.qn1 = None
		self.qn2 = None
		self.qn3 = None
		self.qn4 = None
		self.qn5 = None
		self.qn6 = None
		self.qn7 = None
		self.qn8 = None
		self.qn9 = None
		self.qn10 = None
		self.qn11 = None
		self.qn12 = None
		self.elower = None
		self.eupper = None
		self.logint = None
		self.frequency = None
		self.intensity = None
		self.aij = None
		self.sijmu = None

		self.read_catalog()

		return


	# Reads in a catalog file and calculates all parameters
	def read_catalog(self):
		raw_array = []
		try:
			with open(self.catalog_file) as input:
				for line in input:
					raw_array.append(line)	

		except TypeError:
			print('Specify a catalog file with catalog_file = \'x\'')
			return	

		catalog = self.splice_array(raw_array)

		self.frequency = np.copy(catalog[0])
		self.error = np.copy(catalog[1])
		self.logint = np.copy(catalog[2])
		self.dof = np.copy(catalog[3])
		self.elower = np.asarray(catalog[4])
		self.gup = np.asarray(catalog[5])
		self.tag = np.asarray(catalog[6])
		self.qnformat = np.asarray(catalog[7])		
		self.qn1 = np.asarray(catalog[8])
		self.qn2 = np.asarray(catalog[9])
		self.qn3 = np.asarray(catalog[10])
		self.qn4 = np.asarray(catalog[11])
		self.qn5 = np.asarray(catalog[12])
		self.qn6 = np.asarray(catalog[13])
		self.qn7 = np.asarray(catalog[14])
		self.qn8 = np.asarray(catalog[15])
		self.qn9 = np.asarray(catalog[16])
		self.qn10 = np.asarray(catalog[17])
		self.qn11 = np.asarray(catalog[18])
		self.qn12 = np.asarray(catalog[19])

		self.eupper = np.copy(self.elower) + self.frequency/29979.2458
		self.det_qns() 
		self.convert_int()

		Q_CT = calc_q(self, self.CT)
		self.sijmu = (exp(np.float64(-(self.elower/0.695)/self.CT)) - exp(np.float64(-(self.eupper/0.695)/self.CT)))**(-1) * ((10**self.logint)/self.frequency) * ((4.16231*10**(-5))**(-1)) * Q_CT
	
		#aij formula from CDMS.  Verfied it matched spalatalogue's values
		self.aij = 1.16395 * 10**(-20) * self.frequency**3 * self.sijmu / self.gup

		ustate_qns = np.vstack((self.qn1, self.qn2, self.qn3, self.qn4, self.qn5, self.qn6)).T
		lstate_qns = np.vstack((self.qn7, self.qn8, self.qn9, self.qn10, self.qn11, self.qn12)).T

		ustate_qns_hash = np.sum(ustate_qns*np.array([1,10,100,1000,10000,100000]), axis=1)
		lstate_qns_hash = np.sum(lstate_qns*np.array([1,10,100,1000,10000,100000]), axis=1)
		
		equivalency = np.equal.outer(ustate_qns_hash, lstate_qns_hash)
		idx = np.argmax(equivalency, axis=0)
	
		self.glow = self.gup[idx]
		self.glow[np.sum(equivalency, axis=0)==0] = 1

		return


	# Determines how many qns represent each state
	def det_qns(self):
		self.qns = int(str(self.qnformat[0])[-1:])
	
		if self.qns > 6:
			self.qns = 6

		return


	# Converts catalog logarithmic intensity units to linear ones
	def convert_int(self):
		self.intensity = 10**(np.copy(self.logint))
		return


	# Splices the catalog file appropriately, then populates a numpy array with the data
	def splice_array(self, x):
		frequency = np.arange(len(x),dtype=float)
		error = np.arange(len(x),dtype=float)
		logint = np.arange(len(x),dtype=float)
		dof = np.arange(len(x),dtype=int)
		elower = np.arange(len(x),dtype=float)
		gup = np.arange(len(x),dtype=int)
		tag = np.arange(len(x),dtype=int)
		qnformat = np.arange(len(x),dtype=int)
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
				fix_qn(gup,line,str(x[line][41:44]))
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
				fix_qn(qn1,line,qn1[line])
			
		for line in range(len(qn2)):
			try:
				qn2[line] = int(qn2[line])
			except ValueError:
				fix_qn(qn2,line,qn2[line])
			
		for line in range(len(qn3)):
			try:
				qn3[line] = int(qn3[line])
			except ValueError:
				fix_qn(qn3,line,qn3[line])						
			
		for line in range(len(qn4)):
			try:
				qn4[line] = int(qn4[line])
			except ValueError:
				fix_qn(qn4,line,qn4[line])

		for line in range(len(qn5)):
			try:
				qn5[line] = int(qn5[line])
			except ValueError:
				fix_qn(qn5,line,qn5[line])
			
		for line in range(len(qn6)):
			try:
				qn6[line] = int(qn6[line])
			except ValueError:
				fix_qn(qn6,line,qn6[line])
			
		for line in range(len(qn7)):
			try:
				qn7[line] = int(qn7[line])
			except ValueError:
				fix_qn(qn7,line,qn7[line])
			
		for line in range(len(qn8)):
			try:
				qn8[line] = int(qn8[line])
			except ValueError:
				fix_qn(qn8,line,qn8[line])
			
		for line in range(len(qn9)):
			try:
				qn9[line] = int(qn9[line])
			except ValueError:
				fix_qn(qn9,line,qn9[line])
						
		for line in range(len(qn10)):
			try:
				qn10[line] = int(qn10[line])
			except ValueError:
				fix_qn(qn10,line,qn10[line])
				
		for line in range(len(qn11)):
			try:
				qn11[line] = int(qn11[line])
			except ValueError:
				fix_qn(qn11,line,qn11[line])
			
		for line in range(len(qn12)):
			try:
				qn12[line] = int(qn12[line])
			except ValueError:
				fix_qn(qn12,line,qn12[line])	
																							
		return frequency,error,logint,dof,elower,gup,tag,qnformat,qn1,qn2,qn3,qn4,qn5,qn6,qn7,qn8,qn9,qn10,qn11,qn12





class MolSim(object):

	def __init__(self, name, catalog, obs_params, vlsr, C, dV, T, gauss=True, ll=None, ul=None, mol_obs=None, res=None):
		self.name = name
		self.catalog = catalog
		self.obs_params = obs_params

		if mol_obs:
			self.ll = []
			self.ul = []
			self.res = []

			for observation in mol_obs:
				self.ll.extend([observation.freq_obs.min()])
				self.ul.extend([observation.freq_obs.max()])
				self.res.extend([observation.res])

		else:
			self.ll = ll
			self.ul = ul
			self.res = res

		self.nchunks = len(self.ll)

		# vlsr, C, dV, and T are all lists (for some number of source components)
		self.vlsr = vlsr
		self.ncomponents = len(vlsr)
		self.C = C
		self.dV = dV
		self.T = T

		self.gauss = gauss
		self.freq_sim = None
		self.int_sim = None
		self.tau_sim = None
		self.Q = None
		self.run_sim()

		return


	# Run a full simulation accounting for the currently-active T, dV, S, and vlsr values, as well as any thermal cutoff for optically-thick lines
	def run_sim(self):
		np.seterr(under='ignore')
		np.seterr(over='ignore')

		self.int_sim = []
		self.freq_sim = []
		self.tau_sim = []

		for i in np.arange(self.ncomponents):	
			int_sim_comp = []
			tau_sim_comp = []
			self.Q = calc_q(self.catalog, self.T[i])
	
			Nl = self.C[i] * self.catalog.glow * np.exp(-self.catalog.elower/(0.695 * self.T[i])) / self.Q
	
			tau_numerator = np.asarray((ccm/(self.catalog.frequency * 10**6))**2 * self.catalog.aij * self.catalog.gup * Nl * (1 - np.exp(-(h * self.catalog.frequency * 10**6)/(k * self.T[i]))),dtype=float)

			tau_denominator = np.asarray(8 * np.pi * (self.dV[i] * self.catalog.frequency * 10**6 / ckm) * self.catalog.glow,dtype=float)
			tau = tau_numerator/tau_denominator

			for j in np.arange(self.nchunks):	
				int_temp = tau
				int_temp = trim_array(int_temp, self.catalog.frequency, [self.ll[j]], [self.ul[j]])	

				tau_sim_comp.extend(int_temp)

				freq_temp = np.copy(self.catalog.frequency)

				freq_temp = trim_array(freq_temp, self.catalog.frequency, [self.ll[j]], [self.ul[j]])

				if self.gauss == True:
					freq_sim_tmp, int_sim_tmp = sim_gaussian(self, int_temp, freq_temp, comp_idx=i, chunk_idx=j)
	
				else:
					freq_sim_tmp = freq_temp
	
					J_T = (h*freq_sim_tmp*10**6/k)*(np.exp(((h*freq_sim_tmp*10**6)/(k*self.T[i]))) -1)**-1
					J_Tbg = (h*freq_sim_tmp*10**6/k)*(np.exp(((h*freq_sim_tmp*10**6)/(k*self.obs_params.Tbg))) -1)**-1
	
					int_sim_tmp = (J_T - J_Tbg)*(1 - np.exp(-int_temp))

				int_sim_tmp = apply_beam(freq_sim_tmp, int_sim_tmp, self.obs_params.source_size, self.obs_params.dish_size)

				freq_shift = (-self.vlsr[i])*freq_sim_tmp/ckm
				freq_obs = freq_sim_tmp + freq_shift

				if self.gauss == True:
					int_sim_comp.extend(np.interp(freq_sim_tmp, freq_obs, int_sim_tmp))

				else:
					int_sim_comp.extend(int_sim_tmp)

				if i ==0:
					self.freq_sim.extend(freq_sim_tmp)

			self.int_sim.append(int_sim_comp)
			self.tau_sim.append(tau_sim_comp)

		self.int_sim = np.array(self.int_sim).sum(axis=0)
		self.tau_sim = np.array(self.tau_sim).sum(axis=0)

		return


	# Fit the current simulation to the specified observation
	# def fit_obs(self, mol_obs):
	# 	self.chain = lm_fit(self, mol_obs)

	# 	return




class MolObs(object):

	def __init__(self, name, obs_file, rms=None):
		self.name = name
		self.obs_file = obs_file
		self.freq_obs = None
		self.int_obs = None
		self.res = None
		self.rms = rms
		self.read_obs(self.obs_file)

		return

	# Read in observations or laboratory spectra.  Detects a standard .ispec header from casaviewer export and applies a GHz flag if necessary
	def read_obs(self, obs_file):
		obs = []

		try:
			with open(obs_file) as input:
				for line in input:
					obs.append(line)	
		except TypeError:
			print('Specify an observation file file with obs_file = \'x\'')
			return		
	
		GHz = False

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
	
		freq_obs = []
		int_obs = []

		for i in range(len(obs)):
			freq_obs.append(float(obs[i].split()[0]))
			int_obs.append(float(obs[i].split()[1].strip('\n')))

		obs_array = np.array((freq_obs, int_obs))
		obs_array_sorted = obs_array[:,obs_array[0].argsort()]

		self.freq_obs = obs_array_sorted[0]
		self.int_obs = obs_array_sorted[1]
		
		if GHz == True:
			self.freq_obs *= 1000.
		
		self.res = abs(self.freq_obs[1]-self.freq_obs[0])	
	
		if self.res == 0.0:
			self.res = abs(self.freq_obs[2]-self.freq_obs[1])
		
		if self.res == 0.0:
			print('First three frequency data points for the observations are identical; resolution could not be automatically determined and has been set to 10 kHz by default.')
			self.res = 0.01	

		if not self.rms:
			self.rms = np.std(self.int_obs)
			self.rms = np.std(self.int_obs < self.rms*4)

		return




class ObsParams(object):

	def __init__(self, name, Tbg=2.7, dish_size=100, source_size=1e20, eta=1.0):
		self.name = name
		self.Tbg = Tbg
		self.dish_size = dish_size
		self.source_size = source_size
		self.eta = eta

		return
