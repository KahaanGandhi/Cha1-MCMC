# ----------------------------------------------------------------------------------
# Developer: Kahaan Gandhi
# Based on methodologies described in:
# Loomis, R.A. et al., Nat Astron 5, 188–196 (2021), DOI: 10.1038/s41550-020-01261-4
# Extends prior scripts for spectral simulation and MCMC inference.
# ----------------------------------------------------------------------------------

import emcee
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from numba import njit
from tqdm import tqdm
from classes import *
from constants import *

# TODO: 4.1 vlsr, 1.5 linewdith for, c2 --> mms1 inrease column density by a factor of two

# TODO: modify ObsParams / MolSim to reflect DSN DSS-43 (keeping beam correction in mind)
# TODO: convert +/- 10 channel iterative mask to frequency space (GOTHAM <> DSN), then reapply corrected channel masking
# TODO: check on mK vs. K, fix in preprocessing step if needed
# TODO: add scientific notation to corner plots
# TODO: try simulating best fit parameters w/ LTE CASSIS
# TODO: implement other run times (non template path, non restart)
# TODO: verify vlsr for each source, and mask_radius

# Calculates local RMS noise in a given spectrum by iteratively masking outliers. 3.5σ default, 6σ for weaker species. 
def calc_noise_std(intensity, threshold=3.5):
    dummy_ints = np.copy(intensity)
    noise = np.copy(intensity)
    dummy_mean = np.nanmean(dummy_ints)
    dummy_std = np.nanstd(dummy_ints)

    # Repeat 3 times to make sure to avoid any interloping lines
    for _ in range(3):
        mask_radius = 3  # Channel range to mask adjacents values (GOTHAM applied +/- 10 channel mask for 1.4kHz frequency resolution)
        for chan in np.where(dummy_ints - dummy_mean < (-dummy_std * threshold))[0]:
            noise[chan - mask_radius : chan + mask_radius] = np.nan
        for chan in np.where(dummy_ints - dummy_mean > (dummy_std * threshold))[0]:
            noise[chan - mask_radius : chan + mask_radius] = np.nan
        noise_mean = np.nanmean(noise)
        noise_std = np.nanstd(np.real(noise))

    return noise_mean, noise_std


# Reads in the data, returns the data which has coverage of a given species (from simulated intensities)
def read_file(filename, restfreqs, int_sim, shift=0.0, GHz=False, plot=False, block_interlopers=True):
    data = np.load(filename, allow_pickle=True)

    # Unpack data arrays
    # TODO: apply mK (DSS-43) <--> K (GOTHAM) conversion?
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
        thresh = 0.05                                                  # Set a threshold as 5% of the peak intensity...
        if int_sim[i] > thresh * np.max(int_sim):                      # find significant simulated intensities...
            vel = (rf - freqs) / rf * 300000 + shift                   # calculate velocity shift for each frequency...
            locs = np.where((vel < (1. + 4.1)) & (vel > (-1. + 4.1)))  # and filter for a velocity range

            if locs[0].size != 0:
                noise_mean, noise_std = calc_noise_std(intensity[locs])
                if block_interlopers and (np.max(intensity[locs]) > 3.5 * noise_std): # 3.5σ threshold, 6σ for weaker species
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
                    relevant_yerrs[locs] = np.sqrt(noise_std ** 2 + (intensity[locs] * 0.1) ** 2)
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
    
    # TODO: take a look at our masked spectrum here, with plots...
    # plt.scatter(relevant_freqs, relevant_intensity)
    # plt.plot(relevant_freqs, relevant_intensity)
    # plt.show()
    # exit()
    
    return(relevant_freqs, relevant_intensity, relevant_yerrs, covered_trans)


# Simulate molecular spectral emission lines for a set of observational parameters
def predict_intensities(source_size, Ncol, Tex, dV, mol_cat):
    obs_params = ObsParams("test", source_size=source_size)
    # TODO: 0.0 or 4.1?
    sim = MolSim("mol sim", mol_cat, obs_params, [0.0], [Ncol], [dV], [Tex], ll=[18000], ul=[27000], gauss=False)
    freq_sim = sim.freq_sim
    int_sim = sim.int_sim
    tau_sim = sim.tau_sim
    
    return freq_sim, int_sim, tau_sim


# Apply a beam dilution correction factor to intensity data
@njit(fastmath=True)
def apply_beam(frequency, intensity, source_size, dish_size):
    # Convert frequencies to wavelenths (cm) 
    wavelength = cm / (frequency * 1e6)
    
    # Compute beam size with diffraction-limited resolution formula, assuming each component is centered in the beam
    beam_size = wavelength * 206265 * 1.22 / dish_size  # 206265 converts radians to arcseconds
    
    # Compute beam dilution factor as the ratio of beam area to source area (overlapping circles)
    dilution_factor = source_size ** 2 / (beam_size ** 2 + source_size ** 2)
    
    # Apply the beam dilution factor to the intensities
    intensity_diluted = intensity * dilution_factor
    
    return intensity_diluted


# Construct a model of molecular line emissions (can sum over multiple sources to create composite model)
@njit(fastmath=True)
def make_model(freqs, intensities, source_size, datagrid_freq, datagrid_vel, vlsr, dV, Tex):
    model = np.zeros(datagrid_vel.shape)
    num_lines = freqs.shape[0]

    # Compute Gaussian profiles for each line and sum them
    for i in range(num_lines):
        velocity_grid = (freqs[i] - datagrid_freq) / freqs[i] * ckm  # Convert frequency shifts to velocity space
        mask = np.abs(velocity_grid - 4.1) < dV * 10
        
        # Gaussian profile for the intensity at each frequency point
        model[mask] += intensities[i] * np.exp(-0.5 * ((velocity_grid[mask] - vlsr) / (dV / 2.355))**2.)

    # Apply the Planck function for thermal radiation, adjusted for the background cosmic temperature (2.7 K)
    J_T = (h * datagrid_freq * 10**6 / k) * (np.exp((h * datagrid_freq * 10**6) / (k * Tex)) - 1)**-1
    J_Tbg = (h * datagrid_freq * 10**6 / k) * (np.exp((h * datagrid_freq * 10**6) / (k * 2.7)) - 1)**-1
    
    # Apply the beam dilution correction to the model
    model = apply_beam(datagrid_freq, (J_T - J_Tbg) * (1 - np.exp(-model)), source_size, 70)

    return model


# Log likelihood for MCMC, evaluates how well a set of model parameters fit the observed data
def lnlike(theta, datagrid, mol_cat):
    tot_lnlike = 0.
    yerrs = datagrid[2]
    line_indices = datagrid[3]
    source_size, Ncol, Tex, vlsr, dV = theta

    # Simulate spectral lines for each compononent using current parameter values
    freqs, ints, taus = predict_intensities(source_size, Ncol, Tex, dV, mol_cat)

    # Select relavent data indices from the predicted spectra
    freqs = np.array(freqs)[line_indices]
    taus = np.array(taus)[line_indices]
    ints = np.array(ints)[line_indices]

    # Construct composite molecular line emission model
    curr_model = make_model(freqs, taus, source_size, datagrid[0], datagrid[1], vlsr, dV, Tex)
    inv_sigma2 = 1.0 / (yerrs ** 2)  # Inverse of variance
    
    # Compute negative log-likelihood as sum of squared differences between observed and simulated spectra, weighted by inverse variance
    tot_lnlike = np.sum((datagrid[1] - curr_model) ** 2 * inv_sigma2 - np.log(inv_sigma2))
    
    return -0.5 * tot_lnlike


# Apply physical priors (e.g. positivity constraints) and limits. For TMC-1, impose sequential order on velocities
def is_within_bounds(theta):
    source_size, Ncol, Tex, vlsr, dV = theta
    
    return (
        0. < source_size < 200. and
        0. < Ncol < 10**16. and
        0. < vlsr < 10. and
        0. < dV < 2. and
        2.7 < Tex < 20.
    )


# Log-prior probability for MCMC, ensuring that a set of model parameters falls within physical and statistical constraints
def lnprior(theta, prior_stds, prior_means):
    # Unpack model parameters and their corresponding prior distributions
    source_size, Ncol, Tex, vlsr, dV = theta
    std_source_size, std_Ncol, std_Tex, std_vlsr, std_dV = prior_stds
    mean_source_size, mean_Ncol, mean_Tex, mean_vlsr, mean_dV = prior_means

    # Adjust standard deviations for velocity-related parameters to be less restrictive
    std_vlsr = mean_dV * 0.8
    std_dV = mean_dV * 0.3

    # Return negative infinity if parameters are out of allowed bounds
    if not is_within_bounds(theta):
        return -np.inf
    
    # Calculate log-prior probabilities assuming Gaussian distributions
    log_prior_source_size = np.log(1.0 / (np.sqrt(2 * np.pi) * std_source_size)) - 0.5 * ((source_size - mean_source_size) ** 2 / std_source_size ** 2)
    log_prior_Tex = np.log(1.0 / (np.sqrt(2 * np.pi) * std_Tex)) - 0.5 * ((Tex - mean_Tex) ** 2 / std_Tex ** 2)
    log_prior_vlsr = np.log(1.0 / (np.sqrt(2 * np.pi) * std_vlsr)) - 0.5 * ((vlsr - mean_vlsr) ** 2 / std_vlsr ** 2)
    log_prior_dV = np.log(1.0 / (np.sqrt(2 * np.pi) * std_dV)) - 0.5 * ((dV - mean_dV) ** 2 / std_dV ** 2)

    return log_prior_source_size + log_prior_Tex + log_prior_vlsr + log_prior_dV


# Log-probability for MCMC, evaluating model parameters with both prior distribution and observed fit
def lnprob(theta, datagrid, mol_cat, prior_stds, prior_means):
    lp = lnprior(theta, prior_stds, prior_means)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, datagrid, mol_cat)


def init_setup(fit_folder, cat_folder, data_path, mol_name, block_interlopers):
    print(f"Running setup for: {mol_name}, block interlopers = {block_interlopers}.")
    catfile = os.path.join(cat_folder, f"{mol_name}.cat")

    try:
        os.mkdir(fit_folder)
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(fit_folder, mol_name))
    except FileExistsError:
        pass
    if not os.path.exists(catfile):
        raise FileNotFoundError(f"No catalog file found at {catfile}.")

    # Initialize molecular simulation components
    mol_cat = MolCat(mol_name, catfile)
    obs_params = ObsParams("init", dish_size=70)
    
    # TODO: try vlsr=[4.1], and other observational parameters
    sim = MolSim(f"{mol_name} sim 8K", mol_cat, obs_params, vlsr=[0.0], C=[4.5e12], dV=[0.7575], T=[7.1], ll=[18000], ul=[27000], gauss=False)
    freq_sim = np.array(sim.freq_sim)
    int_sim = np.array(sim.int_sim)
    
    # Read and process spectral data
    print(f"Reading in data from: {data_path}")
    freqs_DSN, ints_DSN, yerrs_DSN, covered_trans_DSN = read_file(data_path, freq_sim, int_sim, block_interlopers=block_interlopers, plot=False)
    covered_trans_DSN = np.array(covered_trans_DSN, dtype=int)
    
    # Assemble data grid for MCMC fitting
    datagrid = [freqs_DSN, ints_DSN, yerrs_DSN, covered_trans_DSN]
    datagrid = np.array(datagrid, dtype=object)
    datafile_path = os.path.join(fit_folder, mol_name, "all_" + mol_name + "_lines_DSN_freq_space.npy")
    
    print(f"Saving data to: {datafile_path}")
    for i, element in enumerate(datagrid):
        element_type = type(element).__name__
        element_shape = element.shape if isinstance(element, np.ndarray) else 'N/A'
        print(f"Reduced Spectrum DataGrid | Index: {i} | Type: {element_type} | Shape: {element_shape}")
    np.save(datafile_path, datagrid, allow_pickle=True)

    return datafile_path, catfile


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
        # Hardcoded values specific for template species like HC5N
        # Source size, Ncol, Tex, vlsr, dV
        # 10 K, 3.4e12
        initial = np.array([48, 4.5e12, 7.1, 4.1, 0.7575])
        # initial = np.array([60 , 4.5e12, 14.0, 3.95, 0.7575]) 
        prior_means = initial
        prior_stds = np.array([6.5, 1.6e12, 0.8, 0.06, 0.22])
        print(f"Using hardcoded priors for a template run of {mol_name}.")
    else:
        print("This type of run is not set up for HC5N yet, aborting simulation.")
        exit()
    
    # Initialize walkers with a small perturbation relative to the prior standard deviations
    perturbation = np.array([1.e-1, 1.e11, 1.e-3, 1.e-3, 1.e-3])
    pos = [initial + perturbation * np.random.randn(ndim) for _ in range(nwalkers)]
    
    # Set up the sampler with a multiprocessing pool
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(datagrid, mol_cat, prior_stds, prior_means), pool=pool)

        # Perform affine invariant MCMC sampling with Gelman-Rubin convergence
        for _ in tqdm(range(nruns), desc=f"MCMC Sampling for {mol_name}"):
            sampler.run_mcmc(pos, 1)
            file_name = os.path.join(fit_folder, mol_name, "chain.npy")
            np.save(file_name, sampler.chain)
            pos = sampler.chain[:, -1, :]

    return


if __name__ == "__main__":
    BASE_DIR = os.getcwd()

    input_dict = {
            'mol_name': 'hc5n_hfs',
            'fit_folder': os.path.join(BASE_DIR, 'DSN_fit_results'),
            'cat_folder': os.path.join(BASE_DIR, 'GOTHAM_catalogs'),
            'data_path': os.path.join(BASE_DIR, 'DSN_data', 'MMS1_hc5n_hfs_chunks.npy'),
            'block_interlopers': True,
            'nruns': 4000,
            'restart': True,
            'prior_path': os.path.join(BASE_DIR, 'DSN_fit_results', 'hc5n_hfs', 'chain.npy'),
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
    CHAIN_PATH = os.path.join(input_dict['fit_folder'], input_dict['mol_name'], "chain.npy")
    if os.path.exists(CHAIN_PATH):
        plot_results(CHAIN_PATH, param_labels)
    else:
        print(f"Chain file not found at {CHAIN_PATH}.")