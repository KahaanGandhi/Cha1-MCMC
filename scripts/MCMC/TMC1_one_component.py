#------------------------------------------------------------------------------------------------
# Developer: Kahaan Gandhi
# Based on methodologies described in:
# Loomis, R.A. et al., Nat Astron 5, 188–196 (2021), DOI: 10.1038/s41550-020-01261-4
# Simplifying assumption that TMC-1 GOTHAM spectra has one velocity component.

# Note: This script is outdated and may require updates to align with recent repository changes.
#------------------------------------------------------------------------------------------------

import numpy as np
import emcee
import matplotlib.pyplot as plt
import os
import sys
from multiprocessing import Pool
from numba import njit
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '..'))
spectral_simulator_dir = os.path.join(base_dir, 'spectral_simulator')
sys.path.insert(0, base_dir)
sys.path.insert(0, spectral_simulator_dir)

from spectral_simulator.classes import *
from spectral_simulator.constants import *


# Calculates local RMS noise in a given spectrum by iteratively masking outliers. 3.5σ default, 6σ for weaker species. 
def calc_noise_std(intensity, threshold=3.5):
    dummy_ints = np.copy(intensity)
    noise = np.copy(intensity)
    dummy_mean = np.nanmean(dummy_ints)
    dummy_std = np.nanstd(dummy_ints)

    # Repeat 3 times to make sure to avoid any interloping lines
    for _ in range(3):
        for chan in np.where(dummy_ints-dummy_mean < (-dummy_std*threshold))[0]:
            noise[chan-10:chan+10] = np.nan
        for chan in np.where(dummy_ints-dummy_mean > (dummy_std*threshold))[0]:
            noise[chan-10:chan+10] = np.nan
        noise_mean = np.nanmean(noise)
        noise_std = np.nanstd(np.real(noise))
        
    return noise_mean, noise_std

    # for chan in np.where(dummy_ints-dummy_mean < (-dummy_std*threshold))[0]:
    #     noise[chan-10:chan+10] = np.nan
    # for chan in np.where(dummy_ints-dummy_mean > (dummy_std*threshold))[0]:
    #     noise[chan-10:chan+10] = np.nan
    # noise_mean = np.nanmean(noise)
    # noise_std = np.nanstd(np.real(noise))

    # for chan in np.where(dummy_ints-noise_mean < (-noise_std*threshold))[0]:
    #     noise[chan-10:chan+10] = np.nan
    # for chan in np.where(dummy_ints-noise_mean > (noise_std*threshold))[0]:
    #     noise[chan-10:chan+10] = np.nan
    # noise_mean = np.nanmean(noise)
    # noise_std = np.nanstd(np.real(noise))

    # for chan in np.where(dummy_ints-noise_mean < (-noise_std*threshold))[0]:
    #     noise[chan-10:chan+10] = np.nan
    # for chan in np.where(dummy_ints-noise_mean > (noise_std*threshold))[0]:
    #     noise[chan-10:chan+10] = np.nan
    # noise_mean = np.nanmean(noise)
    # noise_std = np.nanstd(np.real(noise))


# Reads in the data, returns the data which has coverage of a given species (from simulated intensities)
def read_file(filename, restfreqs, int_sim, shift=0.0, GHz=False, plot=False, block_interlopers=True):
    data = np.load(filename, allow_pickle=True)

    # Unpack data arrays
    freqs = data[0]
    intensity = data[1]
    if GHz:
        freqs *= 1000.

    relevant_freqs = np.zeros(freqs.shape)
    relevant_intensity = np.zeros(intensity.shape)
    relevant_yerrs = np.zeros(freqs.shape)
    covered_trans = []

    # Iterate through rest frequencies to identify their corresponding spectral lines
    for i, rf in enumerate(restfreqs):
        thresh = 0.05                                           # Set a threshold as 5% of the peak intensity...
        if int_sim[i] > thresh*np.max(int_sim):                 # find significant simulated intensities...
            vel = (rf - freqs)/rf*300000 + shift                # calculate velocity shift for each frequency...
            locs = np.where((vel<(.3+6.)) & (vel>(-.3+5.6)))    # and filter for a velocity range

            if locs[0].size != 0:
                noise_mean, noise_std = calc_noise_std(intensity[locs])
                if block_interlopers and (np.max(intensity[locs]) > 6*noise_std):
                    print(f"{rf:10.4f} MHz  |  Interloping line detected.")
                    if plot:
                        plt.plot(freqs[locs], intensity[locs])
                        plt.show()
                else:
                    # Mark the transition and store relavent data
                    covered_trans.append(i)
                    print(f"{rf:10.4f} MHz  |  Line found. ")
                    relevant_freqs[locs] = freqs[locs]
                    relevant_intensity[locs] = intensity[locs]
                    relevant_yerrs[locs] = np.sqrt(noise_std**2 + (intensity[locs]*0.1)**2)
                if plot:
                    plt.plot(freqs[locs], intensity[locs])
                    plt.show()
            else:
                print(f"{rf:10.4f} MHz  |  No data. ")
    
    # Filter out zero entries to return a sparse, small spectrum
    mask = relevant_freqs > 0
    relevant_freqs = relevant_freqs[mask]
    relevant_intensity = relevant_intensity[mask]
    relevant_yerrs = relevant_yerrs[mask]
    
    return(relevant_freqs, relevant_intensity, relevant_yerrs, covered_trans)


# Simulate molecular spectral emission lines for a set of observational parameters
def predict_intensities(source_size, Ncol, Tex, dV, mol_cat):
    obs_params = ObsParams("test", source_size=source_size)
    sim = MolSim("mol sim", mol_cat, obs_params, [0.0], [Ncol], [dV], [Tex], ll=[7000], ul=[30000], gauss=False)
    freq_sim = sim.freq_sim
    int_sim = sim.int_sim
    tau_sim = sim.tau_sim
    
    return freq_sim, int_sim, tau_sim

# Apply a beam dilution correction factor to intensity data
@njit(fastmath=True)
def apply_beam(frequency, intensity, source_size, dish_size):
    # Convert frequencies to wavelenths (cm) 
    wavelength = cm/(frequency*1e6)
    
    # Compute beam size with diffraction-limited resolution formula, assuming each component is centered in the beam
    beam_size = wavelength * 206265 * 1.22 / dish_size  # 206265 converts radians to arcseconds
    
    # Compute beam dilution factor as the ratio of beam area to source area (overlapping circles)
    dilution_factor = source_size**2/(beam_size**2 + source_size**2)
    
    # Apply the beam dilution factor to the intensities
    intensity_diluted = intensity*dilution_factor
    
    return intensity_diluted


# Construct a composite model of molecular line emissions
@njit(fastmath=True)
def make_model(freqs1, ints1, ss1, datagrid0, datagrid1, vlsr1, dV, Tex):
    curr_model = np.zeros(datagrid1.shape)
    model1 = np.zeros(datagrid1.shape)
    nlines = freqs1.shape[0]

    # Compute Gaussian profiles for each source and sum them
    for i in range(nlines):
        vel_grid = (freqs1[i]-datagrid0)/freqs1[i]*ckm  # Convert frequency shifts to velocity space
        mask = np.abs(vel_grid-5.8) < dV*10
        
        # Gaussian profiles for the intensity at each frequency point
        model1[mask] += ints1[i]*np.exp(-0.5*((vel_grid[mask] - vlsr1)/(dV/2.355))**2.)

    # Apply the Planck function for thermal radiation, adjusted for background cosmic temperature (2.7 K)
    J_T = (h*datagrid0*10**6/k)*(np.exp(((h*datagrid0*10**6)/(k*Tex))) -1)**-1
    J_Tbg = (h*datagrid0*10**6/k)*(np.exp(((h*datagrid0*10**6)/(k*2.7))) -1)**-1
    
    # Apply the beam dilution correction to each model
    model1 = apply_beam(datagrid0, (J_T - J_Tbg)*(1 - np.exp(-model1)), ss1, 100)

    # Sum all the individual models to create the composite model
    curr_model = model1

    return curr_model


# Log likelihood for MCMC, evaluates how well a set of model parameters fit the observed data
def lnlike(theta, datagrid, mol_cat):
    tot_lnlike = 0.
    yerrs = datagrid[2]
    line_ids = datagrid[3]
    source_size1, Ncol1, Tex, vlsr1, dV = theta

    # Simulate spectral lines for each compononent using current parameter values
    freqs1, ints1, taus1 = predict_intensities(source_size1, Ncol1, Tex, dV, mol_cat)

    # Select relavent data indices from the predicted spectra
    freqs1 = np.array(freqs1)[line_ids]

    taus1 = np.array(taus1)[line_ids]

    ints1 = np.array(ints1)[line_ids]

    # Construct composite molecular line emission model
    curr_model = make_model(freqs1, taus1, source_size1, datagrid[0], datagrid[1], vlsr1, dV, Tex)
    inv_sigma2 = 1.0/(yerrs**2)  # Inverse of variance
    
    # Compute negative log-likelihood as sum of squared differences between observed and simulated spectra, weighted by inverse variance
    tot_lnlike = np.sum((datagrid[1] - curr_model)**2*inv_sigma2 - np.log(inv_sigma2))
    
    return -0.5*tot_lnlike


# Apply physical priors (e.g. positivity constraints) and limits. For TMC-1, impose sequential order on velocities
def is_within_bounds(theta):
    source_size1, Ncol1, Tex, vlsr1, dV = theta
    
    return (
        0. < source_size1 < 200 and
        0. < Ncol1 < 10**16. and
        0. < vlsr1 < 10. and
        dV < 0.7 and  # TODO: change to 0.7, potentially...
        2.7 < Tex
    )


# Log-prior probability for MCMC, ensuring that a set of model parameters falls within physical and statistical constraints
def lnprior(theta, prior_stds, prior_means):
    # Unpack parameters and prior distributions
    source_size1, Ncol1, Tex, vlsr1, dV = theta
    s0, s4, s8, s9, s13 = prior_stds
    m0, m4, m8, m9, m13 = prior_means

    # Adjust standard deviations for velocity-related parameters to be less restrictive (relaxing vlsr and dV)
    s9 = m13*0.8
    s13 = m13*0.3

    # Zero likelihood if any parameters are out of bounds to prevent walker from accepting illegal step
    if not is_within_bounds(theta):
        return -np.inf
    
    # Compute log-priors for each parameter, assuming a Gaussian distribution centered at prior mean
    p0 = np.log(1.0/(np.sqrt(2*np.pi)*s0))-0.5*(source_size1-m0)**2/s0**2

    # We don't use column density, since this won't hold consistent for HC9N <-> linear species
    p8 = np.log(1.0/(np.sqrt(2*np.pi)*s8))-0.5*(Tex-m8)**2/s8**2
    p9 = np.log(1.0/(np.sqrt(2*np.pi)*s9))-0.5*(vlsr1-m9)**2/s9**2

    p13 = np.log(1.0/(np.sqrt(2*np.pi)*s13))-0.5*(dV-m13)**2/s13**2

    return p0 + p8 + p9 + p13


# Log-probability for MCMC, evaluating model parameters with both prior distribution and observed fit
def lnprob(theta, datagrid, mol_cat, prior_stds, prior_means):
    lp = lnprior(theta, prior_stds, prior_means)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, datagrid, mol_cat)


# Conduct Markov Chain Monte Carlo (MCMC) inference using emcee's ensemble sampler
def fit_multi_gaussian(datafile, fit_folder, catalogue, nruns, mol_name, prior_path, restart=True, template_run=False):
    print(f"Fitting column densities for {mol_name}. Restart = {restart}.")
    ndim, nwalkers = 5, 128
    if not os.path.exists(datafile):
        raise FileNotFoundError(f"The data file {datafile} could not be found.")
    datagrid = np.load(datafile, allow_pickle=True)
    mol_cat = MolCat(mol_name, catalogue)

    # Choose initial parameters and perturbations based on the run type
    if template_run:
        # Hardcoded values specific for template species like HC9N or Benzonitrile
        # Initial values and standard deviations from Loomis et al. (2021)
        initial = np.array([56, 2.15e13, 6.7, 5.790, 0.5]) # HC9N
        prior_means = initial
        prior_stds = np.array([6.5, 0.215e13, 0.1, 0.0035, 0.002])  # HC9N
        print(f"Using hardcoded priors for a template run of {mol_name}.")
    else:
        # Load priors from previous chain data or specified path
        if not os.path.exists(prior_path):
            raise FileNotFoundError(f"The prior path {prior_path} could not be found.")
        psamples = np.load(prior_path).T
        print(f"Dimensions of samples loaded from chain: {psamples.shape}")
                
        prior_means = np.mean(np.percentile(psamples, 50, axis=1), axis=1)
        percentile_16 = np.mean(np.percentile(psamples, 16, axis=1), axis=1)
        percentile_84 = np.mean(np.percentile(psamples, 84, axis=1), axis=1)
        prior_stds = np.abs((percentile_16 + percentile_84 - 2 * prior_means) / 2.)
        
        if prior_means.shape == (14,) and prior_stds.shape == (14,) and prior_means.ndim == 1 and prior_stds.ndim == 1:
            print("Priors are correctly shaped as 1-dimensional arrays with 14 elements each.")
        else:
            raise ValueError("Error: prior_means and prior_stds should be 1-dimensional arrays with 14 elements each.")
        
        if restart:
            # Restart with predefined initial values from domain-specific sources
            # initial = np.array([42.8, 24.3, 47.9, 21.5, 5.8e13, 9.5e13, 4.e13, 1.06e14, 7.7, 5.603, 5.745, 5.873, 6.024, 0.1568])
            
            # HC9N telescope and source properties, but with a perturbed version of HC11N colummn dentisities
            initial = np.array([37, 25, 56, 22, 0.73e11, 2.60e11, 0.36e11, 4.12e11, 6.7, 5.624, 5.790, 5.910, 6.033, 0.117])
            print("Offsetting HC11N Ncol values, and using HC9N telescope and source properties.")
            initial += np.array([0, 0, 0, 0, 1.e10, -1.e10, 1.e10, -1.e10, 0, 0, 0, 0, 0, 0])
            print("Restarting with hardcoded initial values from the TMC-1 MCMC codebase.")
        else:
            # Continue from the existing chain data for this molecule
            chain_data = np.load(os.path.join(fit_folder, mol_name, "chain.npy"))[:,-200:,:].reshape(-1, ndim).T
            initial = np.median(chain_data, axis=1)
            print(f"Continuing from existing chain data of {mol_name}.")

    # Initialize walkers with a small perturbation relative to the prior standard deviations
    perturbation = np.array([1.e-1, 1.e11, 1.e-3, 1.e-3, 1.e-3])
    pos = [initial + perturbation * np.random.randn(ndim) for _ in range(nwalkers)]

    # Ensure all initial positions are valid, ideally in a tight ball (as per emcee documentation)
    if len(pos) != nwalkers:
        print(f"Warning: Only initialized {len(pos)} valid walkers out of the requested {nwalkers}.")
    else:
        print(f"Successfully initialized {len(pos)} valid walkers.")
    
    # Set up the sampler with a multiprocessing pool
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(datagrid, mol_cat, prior_stds, prior_means), pool=pool)

        # Perform affine invariant MCMC sampling with Gelman-Rubin convergence
        for i in tqdm(range(nruns), desc=f"MCMC Sampling for {mol_name}"):
            sampler.run_mcmc(pos, 1)
            file_name = os.path.join(fit_folder, mol_name, "chain_one.npy")
            np.save(file_name, sampler.chain)
            pos = sampler.chain[:, -1, :]

    return


def init_setup(fit_folder, cat_folder, data_path, mol_name, block_interlopers):
    print(f"Running setup for: {mol_name}, block interlopers = {block_interlopers}.")
    try:
        os.mkdir(os.path.join(fit_folder, mol_name))
    except FileExistsError:
        pass

    catfile = os.path.join(cat_folder, f"{mol_name}.cat")
    if not os.path.exists(catfile):
        raise FileNotFoundError(f"No catalog file found at {catfile}.")

    # Initialize molecular simulation components
    mol_cat = MolCat(mol_name, catfile)
    obs_params = ObsParams("init", source_size=40)
    sim = MolSim(f"{mol_name} sim 8K", mol_cat, obs_params, [0.0], [7.e11], [0.37], [8.], ll=[7000], ul=[30000], gauss=False)
    freq_sim = np.array(sim.freq_sim)
    int_sim = np.array(sim.int_sim)
    
    # Read and process spectral data
    print(f"Reading in data from {data_path}")
    freqs_gotham, ints_gotham, yerrs_gotham, covered_trans_gotham = read_file(data_path, freq_sim, int_sim, block_interlopers=block_interlopers, plot=False)
    covered_trans_gotham = np.array(covered_trans_gotham, dtype=int)
    
    # Assemble data grid for MCMC fitting
    datagrid = [freqs_gotham, ints_gotham, yerrs_gotham, covered_trans_gotham]
    datagrid = np.array(datagrid, dtype=object)
    datafile_path = os.path.join(fit_folder, mol_name, "all_" + mol_name + "_lines_GOTHAM_freq_space.npy")
    
    print(f"Saving data to: {datafile_path}")
    for i, item in enumerate(datagrid):
        print(f"Datagrid element {i}  |  Type: {type(item)}  |  Shape: {item.shape if isinstance(item, np.ndarray) else 'N/A'}")
        
    np.save(datafile_path, datagrid, allow_pickle=True)

    return datafile_path, catfile


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    input_dict = {
        'mol_name': 'hc7n_hfs',
        'fit_folder': os.path.join(BASE_DIR, 'GOTHAM_fit_results'),
        'cat_folder': os.path.join(BASE_DIR, 'CDMS_catalog'),
        'data_path': os.path.join(BASE_DIR, 'GOTHAM_data', 'hc7n_hfs_chunks.npy'),
        'block_interlopers': True,
        'nruns': 10000,
        'restart': True,
        'prior_path': os.path.join(BASE_DIR, 'GOTHAM_fit_results', 'hc9n_hfs', 'chain_one.npy'),
        'template_run': True
    }

    datafile, catalogue = init_setup(
        fit_folder=input_dict['fit_folder'],
        cat_folder=input_dict['cat_folder'],
        data_path=input_dict['data_path'],
        mol_name=input_dict['mol_name'],
        block_interlopers=input_dict['block_interlopers']
    )

    fit_multi_gaussian(
        datafile=datafile,
        fit_folder=input_dict['fit_folder'],
        catalogue=catalogue,
        nruns=input_dict['nruns'],
        mol_name=input_dict['mol_name'],
        prior_path=input_dict['prior_path'],
        restart=input_dict['restart'],
        template_run=input_dict['template_run']
    )
    
    param_labels = [
        'Source Size #1 [″]', 
        'Ncol #1 [cm⁻²]', 
        'Tex [K]',
        'vlsr #1 [km s⁻¹]', 
        'dV [km s⁻¹]'
    ]

    # Verify that chain file path matches where data was saved
    CHAIN_PATH = os.path.join(input_dict['fit_folder'], input_dict['mol_name'], "chain_one.npy")
    if os.path.exists(CHAIN_PATH):
        plot_results(CHAIN_PATH, param_labels)
    else:
        print(f"Chain file not found at {CHAIN_PATH}.")