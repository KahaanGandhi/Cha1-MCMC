#-----------------------------------------------------------------------------------
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


# Calculates local RMS noise in a given spectrum by iteratively masking outliers. 3.5σ default, 6σ for weaker species. 
def calc_noise_std(intensity, threshold=3.5):
    dummy_ints = np.copy(intensity)
    noise = np.copy(intensity)
    dummy_mean = np.nanmean(dummy_ints)
    dummy_std = np.nanstd(dummy_ints)

    # Repeat 3 times to make sure to avoid any interloping lines
    for _ in range(3):
        mask_radius = 3  # Channel range to mask adjacents values
        for chan in np.where(dummy_ints - dummy_mean < (-dummy_std * threshold))[0]:
            noise[chan - mask_radius : chan + mask_radius] = np.nan
        for chan in np.where(dummy_ints - dummy_mean > (dummy_std * threshold))[0]:
            noise[chan - mask_radius : chan + mask_radius] = np.nan
        noise_mean = np.nanmean(noise)
        noise_std = np.nanstd(np.real(noise))

    return noise_mean, noise_std


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


# Apply physical priors (e.g. positivity constraints) and limits. For multiple sources, impose sequential order on velocities
def is_within_bounds(theta):
    source_size, Ncol, Tex, vlsr, dV = theta
    
    return (
        30. < source_size < 90. and
        10**8. < Ncol < 10**14. and
        3. < vlsr < 5.5 and
        0.35 < dV < 1.5 and
        3.4 < Tex < 12.
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

    # Vary weight to incenvitize/punish exploration from previously succesful observational parameters
    return 1.0 * (log_prior_source_size + log_prior_Tex + log_prior_vlsr + log_prior_dV)


# Log-probability for MCMC, evaluating model parameters with both prior distribution and observed fit
def lnprob(theta, datagrid, mol_cat, prior_stds, prior_means):
    lp = lnprior(theta, prior_stds, prior_means)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, datagrid, mol_cat)


# Simulate molecular spectral emission lines for a set of observational parameters
def predict_intensities(source_size, Ncol, Tex, dV, mol_cat):
    obs_params = ObsParams("test", source_size=source_size)
    sim = MolSim("mol sim", mol_cat, obs_params, [4.33], [Ncol], [dV], [Tex], ll=[18000], ul=[25000], gauss=False)
    freq_sim = sim.freq_sim
    int_sim = sim.int_sim
    tau_sim = sim.tau_sim
    
    return freq_sim, int_sim, tau_sim


# Construct a model of molecular line emissions (can sum over multiple sources to create composite model)
@njit(fastmath=True)
def make_model(freqs, intensities, source_size, datagrid_freq, datagrid_ints, vlsr, dV, Tex):    
    model = np.zeros(datagrid_ints.shape)
    num_lines = freqs.shape[0]

    # Compute Gaussian profiles for each line and sum them (can use vlsr in place of fixed value)
    for i in range(num_lines):
        velocity_grid = (freqs[i] - datagrid_freq) / freqs[i] * ckm  + 4.33  # Convert frequency shifts to velocity space
        mask = np.abs(velocity_grid - 4.33) < dV * 10
        
        # Gaussian profile for the intensity at each frequency point
        model[mask] += intensities[i] * np.exp(-0.5 * ((velocity_grid[mask] - vlsr) / (dV / 2.355))**2.)

    # Apply the Planck function for thermal radiation, adjusted for the background cosmic temperature (2.7 K)
    J_T = (h * datagrid_freq * 10**6 / k) * (np.exp((h * datagrid_freq * 10**6) / (k * Tex)) - 1)**-1
    J_Tbg = (h * datagrid_freq * 10**6 / k) * (np.exp((h * datagrid_freq * 10**6) / (k * 2.7)) - 1)**-1

    # Apply the beam dilution correction to the model
    model = apply_beam(datagrid_freq, (J_T - J_Tbg) * (1 - np.exp(-model)), source_size, 70)

    return model


# Reads in the data, returns the data which has coverage of a given species (from simulated intensities)
def read_file(filename, restfreqs, int_sim, shift=4.33, GHz=False, plot=False, block_interlopers=True):
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
        thresh = 0.05                                                     # Set a threshold as 5% of the peak intensity...
        if int_sim[i] > thresh * np.max(int_sim):                         # find significant simulated intensities...
            vel = (rf - freqs) / rf * ckm + shift                         # calculate velocity shift for each frequency...
            locs = np.where((vel < (4.33 + 1.5)) & (vel > (4.33 - 1.5)))  # and filter for a velocity range that captures the line features

            if locs[0].size != 0:
                noise_mean, noise_std = calc_noise_std(intensity[locs])
                if block_interlopers and (np.max(intensity[locs]) > 3.5 * noise_std): # 3.5σ threshold, 6σ for weaker species
                    print(f"{GRAY}{rf:10.4f} MHz  |  Interloping line detected.{RESET}")
                    if plot:
                        plt.plot(freqs[locs], intensity[locs])
                        plt.show()
                else:
                    # Mark the transition and store relavent data
                    covered_trans.append(i)
                    print(f"{GRAY}{rf:10.4f} MHz  |  Line found.{RESET}")
                    relevant_freqs[locs] = freqs[locs]
                    relevant_intensity[locs] = intensity[locs]
                    relevant_yerrs[locs] = np.sqrt(noise_std ** 2 + (intensity[locs] * 0.1) ** 2)
                if plot:
                    plt.plot(freqs[locs], intensity[locs])
                    plt.show()
            else:
                print(f"{GRAY}{rf:10.4f} MHz  |  No data.{RESET}")

    # Filter out zero entries to return a sparse, small spectrum
    mask = relevant_freqs > 0
    relevant_freqs = relevant_freqs[mask]
    relevant_intensity = relevant_intensity[mask]
    relevant_yerrs = relevant_yerrs[mask]

    # A hardcoded shift correction for inproperly corrected DSN data...
    # relevant_freqs -= 0.367

    return(relevant_freqs, relevant_intensity, relevant_yerrs, covered_trans)


def init_setup(fit_folder, cat_folder, data_path, mol_name, block_interlopers):
    print(f"\n{CYAN}Running setup for: {mol_name}, block interlopers = {block_interlopers}.{RESET}")
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
        raise FileNotFoundError(f"{RED}No catalog file found at {catfile}.{RESET}")

    # Initialize molecular simulation components
    mol_cat = MolCat(mol_name, catfile)
    obs_params = ObsParams("init", dish_size=70)
    sim = MolSim(f"{mol_name} sim 8K", mol_cat, obs_params, vlsr=[4.33], C=[3.4e12], dV=[0.89], T=[7.0], ll=[18000], ul=[25000], gauss=False)
    freq_sim = np.array(sim.freq_sim)
    int_sim = np.array(sim.int_sim)
    
    # Read and process spectral data
    print(f"{CYAN}Reading in spectral data from: {data_path}{RESET}")
    freqs_DSN, ints_DSN, yerrs_DSN, covered_trans_DSN = read_file(data_path, freq_sim, int_sim, block_interlopers=block_interlopers, plot=False)
    covered_trans_DSN = np.array(covered_trans_DSN, dtype=int)
    
    # Assemble data grid for MCMC fitting
    datagrid = [freqs_DSN, ints_DSN, yerrs_DSN, covered_trans_DSN]
    datagrid = np.array(datagrid, dtype=object)
    datafile_path = os.path.join(fit_folder, mol_name, "all_" + mol_name + "_lines_DSN_freq_space.npy")
    print(f"\n{CYAN}Saving reduced spectrum to: {datafile_path}{RESET}")

    for i, element in enumerate(datagrid):
        element_type = type(element).__name__
        element_shape = element.shape if isinstance(element, np.ndarray) else 'N/A'
        print(f"{GRAY}Reduced Spectrum Datagrid | Index: {i} | Type: {element_type} | Shape: {element_shape}{RESET}")
    np.save(datafile_path, datagrid, allow_pickle=True)
    
    return datafile_path, catfile


# Conduct Markov Chain Monte Carlo (MCMC) inference using emcee's ensemble sampler
def fit_multi_gaussian(datafile, fit_folder, catalogue, nruns, mol_name, prior_path, restart=True, template_run=False, parallelize=True):
    print(f"{CYAN}Fitting column densities for {mol_name}. Restart = {restart}.{RESET}")
    ndim, nwalkers = 5, 128
    if not os.path.exists(datafile):
        raise FileNotFoundError(f"{RED}The data file {datafile} could not be found.{RESET}")
    datagrid = np.load(datafile, allow_pickle=True)
    mol_cat = MolCat(mol_name, catalogue)

    # Choose initial parameters and perturbations based on the run type
    if template_run:
        # Hardcoded values specific for template species like HC5N (Source size, Ncol, Tex, vlsr, dV)
        initial = np.array([48, 3.4e12, 8.0, 4.3, 0.7575])
        prior_means = initial
        prior_stds = np.array([6.5, 0.34e12, 3.0, 0.06, 0.22])
        print(f"{GRAY}Using hardcoded priors and initial positions for a template run of {mol_name}.{RESET}")
    else:
        # Load priors from previous chain data
        if not os.path.exists(prior_path):
            raise FileNotFoundError(f"{RED}The prior path {prior_path} could not be found.{RESET}")
        print(f"{GRAY}Loading previous chain data from: {prior_path}{RESET}")
        psamples = np.load(prior_path).T
        print(f"{GRAY}Dimensions of samples loaded from chain: {psamples.shape}{RESET}")
        
        prior_means = np.mean(np.percentile(psamples, 50, axis=1), axis=1)
        percentile_16 = np.mean(np.percentile(psamples, 16, axis=1), axis=1)
        percentile_84 = np.mean(np.percentile(psamples, 84, axis=1), axis=1)
        prior_stds = np.abs((percentile_16 + percentile_84 - 2 * prior_means) / 2.)
        print(f"{GRAY}Loading priors from chain.{RESET}")
        
        if prior_means.shape == (5,) and prior_stds.shape == (5,) and prior_means.ndim == 1 and prior_stds.ndim == 1:
            print(f"{GRAY}Priors are correctly shaped as 1-dimensional arrays with 5 elements each.{RESET}")
        else:
            raise ValueError(f"{RED}Error: priors should be 1-dimensional arrays with 5 elements each.{RESET}")
        
        if restart:
            initial = np.array([48, 3.4e12, 11.0, 4.3, 0.7575])
            print(f"{GRAY}Using hardcoded initial positions.{RESET}")
        else:
            chain_data = np.load(os.path.join(fit_folder, "hc5n_hfs", "chain_template.npy"))[:,-200:,:].reshape(-1, ndim).T
            initial = np.median(chain_data, axis=1)
            print(f"{GRAY}Loading initial positions from chain.{RESET}")
    
    # Initialize walkers with a small perturbation relative to the prior standard deviations
    perturbation = np.array([1.e-1, 0.1 * prior_means[1], 1.e-3, 1.e-3, 1.e-3])
    pos = [initial + perturbation * np.random.randn(ndim) for _ in range(nwalkers)]
    print()
    
    if template_run and mol_name == "hc5n_hfs":
        file_name = os.path.join(fit_folder, mol_name, "chain_template.npy")
    elif template_run:
        print(f"{RED}Template run selected with incorrect template species. Proceeding with non-template run.{RESET}")
        file_name = os.path.join(fit_folder, mol_name, "chain.npy")
    else:
        file_name = os.path.join(fit_folder, mol_name, "chain.npy")

    # Perform affine invariant MCMC sampling with Gelman-rubin convergence
    if parallelize:
        with Pool() as pool:
            # Initialize the sampler with parallelization
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(datagrid, mol_cat, prior_stds, prior_means), pool=pool)
            
            for _ in tqdm(range(nruns), desc=f"MCMC Sampling for {mol_name}", colour='white'):
                sampler.run_mcmc(pos, 1)
                np.save(file_name, sampler.chain)
                pos = sampler.chain[:, -1, :]
        return
    else:
        # Initialize the sampler without parallelization (ideal for debugging)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(datagrid, mol_cat, prior_stds, prior_means))
    
        for _ in tqdm(range(nruns), desc=f"MCMC Sampling for {mol_name}", colour='white'):
            sampler.run_mcmc(pos, 1)
            np.save(file_name, sampler.chain)
            pos = sampler.chain[:, -1, :]


if __name__ == "__main__":
    BASE_DIR = os.getcwd()

    data_paths = {
        # 'hc5n_hfs': os.path.join(BASE_DIR, 'DSN_data', 'cha_c2_hc5n_example.npy'),
        # 'hc7n_hfs': os.path.join(BASE_DIR, 'DSN_data', 'cha_c2_hc7n_example.npy'),
        'hc5n_hfs': os.path.join(BASE_DIR, 'DSN_data', 'cha_mms1_hc5n_example.npy'),
        'hc7n_hfs': os.path.join(BASE_DIR, 'DSN_data', 'cha_mms1_hc7n_example.npy'),
        # TODO: run pipeline for HC5N in MMS1 and update result here...
        # Add more molecules and paths as needed... 
    }

    config = {
        # These settings are usually changed based on the specific run
        'mol_name': 'hc7n_hfs',
        'nruns': 10000,
        'template_run': False,
        'restart': False,

        # These settings can usually remain fixed
        'fit_folder': os.path.join(BASE_DIR, 'DSN_fit_results'),
        'cat_folder': os.path.join(BASE_DIR, 'CDMS_catalog'),
        'data_path': None,
        'block_interlopers': True,
        'prior_path': os.path.join(BASE_DIR, 'DSN_fit_results', 'hc5n_hfs', 'chain_template.npy'),
        'parallelize': True,
    }
        
    try:
        config['data_path'] = data_paths.get(config['mol_name'])
    except KeyError:
        print(f"{RED}Molecule {config['mol_name']} not found in data_paths. Verify molecule name or update data_paths dictionary.{RESET}")
        
    datafile, catalogue = init_setup(
        fit_folder=config['fit_folder'],
        cat_folder=config['cat_folder'],
        data_path=config['data_path'],
        mol_name=config['mol_name'],
        block_interlopers=config['block_interlopers'],
    )

    fit_multi_gaussian(
        datafile=datafile,
        fit_folder=config['fit_folder'],
        catalogue=catalogue,
        nruns=config['nruns'],
        mol_name=config['mol_name'],
        prior_path=config['prior_path'],
        restart=config['restart'],
        template_run=config['template_run'],
        parallelize=config['parallelize'],
    )

    param_labels = [
        'Source Size [″]',
        'Nᴄᴏʟ [cm⁻²]',
        'Tᴇx [K]',
        'ᴠʟsʀ [km s⁻¹]',
        'dV [km s⁻¹]'
    ]
    
    # Verify that chain file path matches where data was saved
    if config['template_run'] and config['mol_name'] == "hc5n_hfs":
        chain_path = os.path.join(config['fit_folder'], config['mol_name'], "chain_template.npy")
    else:
        chain_path = os.path.join(config['fit_folder'], config['mol_name'], "chain.npy")
    if os.path.exists(chain_path):
        plot_results(chain_path, param_labels, include_trace=False)
    else:
        print(f"{RED}Chain file not found at {chain_path}. Exiting.{RESET}")