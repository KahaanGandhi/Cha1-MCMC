#------------------------------------------------------------------------------------------
# Developer: Kahaan Gandhi
# MCMC inference to fit spectral models to observed data, based on methods in:
# Loomis, R.A. et al., Nat Astron 5, 188–196 (2021), DOI: 10.1038/s41550-020-01261-4.
#
# Source size is highly covariant with column density; in cases where source size can 
# be estimated through other means, fixing it can better constrain column density. 
# Allows for customizable MCMC sampling and observational parameters (see main function).
#------------------------------------------------------------------------------------------

import emcee
import os
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from numba import njit
from tqdm import tqdm
from tabulate import tabulate

from spectral_simulator.classes import *
from spectral_simulator.functions import *
from spectral_simulator.constants import *

# Functions optimized with @njit (Numba doesn't compile methods with 'self', so attributes are passed as arguments)
# Apply a beam dilution correction factor to intensity data
@njit(fastmath=True)
def apply_beam_numba(frequency, intensity, source_size, dish_size):
    wavelength = cm / (frequency * 1e6)
    
    # Compute beam size with diffraction-limited resolution formula
    beam_size = wavelength * 206265 * 1.22 / dish_size  # 206265 converts radians to arcseconds
    dilution_factor = source_size ** 2 / (beam_size ** 2 + source_size ** 2)
    intensity_diluted = intensity * dilution_factor
    return intensity_diluted

# Construct a model of molecular line emissions (can sum over multiple sources to create a composite model)
@njit(fastmath=True)
def make_model_numba(freqs, intensities, source_size, datagrid_freq, datagrid_ints, vlsr, dV, Tex, aligned_velocity, dish_size):
    model = np.zeros(datagrid_ints.shape)
    num_lines = freqs.shape[0]

    # Compute Gaussian profiles for each spectral line and sum them
    for i in range(num_lines):
        velocity_grid = (freqs[i] - datagrid_freq) / freqs[i] * ckm + aligned_velocity
        mask = np.abs(velocity_grid - aligned_velocity) < dV * 10
        model[mask] += intensities[i] * np.exp(-0.5 * ((velocity_grid[mask] - vlsr) / (dV / 2.355)) ** 2)

    # Apply the Planck function for thermal radiation, adjusted for the background cosmic temperature (2.7 K)
    J_T = (h * datagrid_freq * 1e6 / k) * (np.exp((h * datagrid_freq * 1e6) / (k * Tex)) - 1) ** -1
    J_Tbg = (h * datagrid_freq * 1e6 / k) * (np.exp((h * datagrid_freq * 1e6) / (k * 2.7)) - 1) ** -1

    # Apply the beam dilution correction to the molecular emission model
    model = apply_beam_numba(datagrid_freq, (J_T - J_Tbg) * (1 - np.exp(-model)), source_size, dish_size)
    return model

class SpectralFitMCMC:
    def __init__(self, config):
        self.config            = config
        self.mol_name          = self.config['mol_name']
        self.template_run      = self.config['template_run']
        self.fit_folder        = self.config['fit_folder']
        self.cat_folder        = self.config['cat_folder']
        self.data_path         = self.config['data_paths'].get(self.mol_name)
        self.prior_path        = self.config['prior_path']
        self.block_interlopers = self.config['block_interlopers']
        self.lower_limit       = self.config['lower_limit']
        self.upper_limit       = self.config['upper_limit']
        self.aligned_velocity  = self.config['aligned_velocity']
        self.dish_size         = self.config['dish_size']
        self.nwalkers          = self.config['nwalkers']
        self.nruns             = self.config['nruns']
        self.template_means    = self.config['template_means']
        self.template_stds     = self.config['template_stds']
        self.parallelize       = self.config['parallelize']
        self.fixed_source_size = self.config['fixed_source_size']
        self.bounds            = self.config['bounds']

        # Adjust parameter labels and dimensions based on whether source size is fixed or not
        if isinstance(self.fixed_source_size, (float, int)):
            self.source_size = self.fixed_source_size
            self.ndim = 4
            self.param_labels = ['Ncol [cm⁻²]', 'Tex [K]', 'vlsr [km s⁻¹]', 'dV [km s⁻¹]']
        else:
            self.source_size = None
            self.ndim = 5
            self.param_labels = ['Source Size [″]', 'Ncol [cm⁻²]', 'Tex [K]', 'vlsr [km s⁻¹]', 'dV [km s⁻¹]']

    # Call standalone @njit-optimized function with required attributes
    def make_model(self, freqs, intensities, datagrid_freq, datagrid_ints, vlsr, dV, Tex, source_size):
        return make_model_numba(freqs=freqs, intensities=intensities, source_size=source_size,
                                datagrid_freq=datagrid_freq, datagrid_ints=datagrid_ints, vlsr=vlsr, dV=dV, Tex=Tex,
                                aligned_velocity=self.aligned_velocity, dish_size=self.dish_size)

    def apply_beam(self, frequency, intensity, source_size):
        return apply_beam_numba(frequency, intensity, source_size, self.dish_size)

    # Calculates local RMS noise in a given spectrum by iteratively masking outliers. 3.5σ default, 6σ for weaker species.
    def calc_noise_std(self, intensity, threshold=3.5):
        dummy_ints = np.copy(intensity)
        noise = np.copy(intensity)
        dummy_mean = np.nanmean(dummy_ints)
        dummy_std = np.nanstd(dummy_ints)

        # Repeat 3 times to make sure to avoid any interloping lines
        for _ in range(3):
            mask_radius = 3  # Channel range to mask adjacent values
            for chan in np.where(dummy_ints - dummy_mean < (-dummy_std * threshold))[0]:
                noise[chan - mask_radius: chan + mask_radius] = np.nan
            for chan in np.where(dummy_ints - dummy_mean > (dummy_std * threshold))[0]:
                noise[chan - mask_radius: chan + mask_radius] = np.nan
            noise_mean = np.nanmean(noise)
            noise_std = np.nanstd(np.real(noise))

        return noise_mean, noise_std

    # Log likelihood for MCMC, evaluates how well a set of model parameters fit the observed data
    def lnlike(self, theta, datagrid, mol_cat):
        tot_lnlike = 0.0
        yerrs = datagrid[2]
        line_indices = datagrid[3]

        # Unpack model parameters
        if self.source_size is not None:
            Ncol, Tex, vlsr, dV = theta
            source_size = self.source_size
        else:
            source_size, Ncol, Tex, vlsr, dV = theta

        # Simulate spectral lines for each component using current parameter values
        freqs, ints, taus = self.predict_intensities(Ncol=Ncol, Tex=Tex, dV=dV, mol_cat=mol_cat, source_size=source_size)
        freqs = np.array(freqs)[line_indices]
        taus = np.array(taus)[line_indices]
        ints = np.array(ints)[line_indices]

        # Construct composite molecular line emission model
        curr_model = self.make_model(freqs=freqs, intensities=taus, datagrid_freq=datagrid[0],
                                     datagrid_ints=datagrid[1], vlsr=vlsr, dV=dV, Tex=Tex, source_size=source_size)
        inv_sigma2 = 1.0 / (yerrs ** 2)

        # Compute negative log-likelihood as sum of squared differences between observed and simulated spectra, weighted by inverse variance
        tot_lnlike = np.sum((datagrid[1] - curr_model) ** 2 * inv_sigma2 - np.log(inv_sigma2))

        return -0.5 * tot_lnlike

    # Check if a set of model parameters falls within priors
    def is_within_bounds(self, theta):
        if self.source_size is not None:
            Ncol, Tex, vlsr, dV = theta
            bounds = self.bounds

            return (
                bounds['Ncol'][0] < Ncol < bounds['Ncol'][1] and
                bounds['Tex'][0] < Tex < bounds['Tex'][1] and
                bounds['vlsr'][0] < vlsr < bounds['vlsr'][1] and
                bounds['dV'][0] < dV < bounds['dV'][1]
            )
        else:
            source_size, Ncol, Tex, vlsr, dV = theta
            bounds = self.bounds

            return (
                bounds['source_size'][0] < source_size < bounds['source_size'][1] and
                bounds['Ncol'][0] < Ncol < bounds['Ncol'][1] and
                bounds['Tex'][0] < Tex < bounds['Tex'][1] and
                bounds['vlsr'][0] < vlsr < bounds['vlsr'][1] and
                bounds['dV'][0] < dV < bounds['dV'][1]
            )

    # Log-prior probability for MCMC, ensuring that a set of model parameters falls within physical and statistical constraints
    def lnprior(self, theta, prior_stds, prior_means, weight=1.0):
        if self.source_size is not None:
            Ncol, Tex, vlsr, dV = theta
            std_Ncol, std_Tex, std_vlsr, std_dV = prior_stds
            mean_Ncol, mean_Tex, mean_vlsr, mean_dV = prior_means

            # Adjust standard deviations for velocity-related parameters to be less restrictive
            std_vlsr = mean_dV * 0.8
            std_dV = mean_dV * 0.3

            # Return negative infinity if parameters are out of allowed bounds
            if not self.is_within_bounds(theta):
                return -np.inf

            # Calculate log-prior probabilities assuming Gaussian distributions
            log_prior_Ncol = 0  # Flat prior for Ncol
            log_prior_Tex = np.log(1.0 / (np.sqrt(2 * np.pi) * std_Tex)) - 0.5 * ((Tex - mean_Tex) ** 2 / std_Tex ** 2)
            log_prior_vlsr = np.log(1.0 / (np.sqrt(2 * np.pi) * std_vlsr)) - 0.5 * ((vlsr - mean_vlsr) ** 2 / std_vlsr ** 2)
            log_prior_dV = np.log(1.0 / (np.sqrt(2 * np.pi) * std_dV)) - 0.5 * ((dV - mean_dV) ** 2 / std_dV ** 2)

            # Weight incentivizes/punishes exploration from previously successful observational parameters
            return weight * (log_prior_Tex + log_prior_vlsr + log_prior_dV + log_prior_Ncol)
        else:
            source_size, Ncol, Tex, vlsr, dV = theta
            std_source_size, std_Ncol, std_Tex, std_vlsr, std_dV = prior_stds
            mean_source_size, mean_Ncol, mean_Tex, mean_vlsr, mean_dV = prior_means

            # Adjust standard deviations for velocity-related parameters to be less restrictive
            std_vlsr = mean_dV * 0.8
            std_dV = mean_dV * 0.3

            # Return negative infinity if parameters are out of allowed bounds
            if not self.is_within_bounds(theta):
                return -np.inf

            # Calculate log-prior probabilities assuming Gaussian distributions
            log_prior_source_size = np.log(1.0 / (np.sqrt(2 * np.pi) * std_source_size)) - 0.5 * ((source_size - mean_source_size) ** 2 / std_source_size ** 2)
            log_prior_Ncol = 0  # Flat prior for Ncol
            log_prior_Tex = np.log(1.0 / (np.sqrt(2 * np.pi) * std_Tex)) - 0.5 * ((Tex - mean_Tex) ** 2 / std_Tex ** 2)
            log_prior_vlsr = np.log(1.0 / (np.sqrt(2 * np.pi) * std_vlsr)) - 0.5 * ((vlsr - mean_vlsr) ** 2 / std_vlsr ** 2)
            log_prior_dV = np.log(1.0 / (np.sqrt(2 * np.pi) * std_dV)) - 0.5 * ((dV - mean_dV) ** 2 / std_dV ** 2)

            # Weight incentivizes/punishes exploration from previously successful observational parameters
            return weight * (log_prior_source_size + log_prior_Tex + log_prior_vlsr + log_prior_dV + log_prior_Ncol)

    # Log-probability for MCMC, evaluating model parameters with both prior distribution and observed fit
    def lnprob(self, theta, datagrid, mol_cat, prior_stds, prior_means):
        lp = self.lnprior(theta, prior_stds, prior_means)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta, datagrid, mol_cat)

    # Simulate molecular spectral emission lines for a set of observational parameters
    def predict_intensities(self, Ncol, Tex, dV, mol_cat, source_size):
        obs_params = ObsParams("test", source_size=source_size)
        sim = MolSim("mol sim", mol_cat, obs_params, vlsr=[self.aligned_velocity], C=[Ncol], dV=[dV], T=[Tex],
                     ll=[self.lower_limit], ul=[self.upper_limit], gauss=False)
        return sim.freq_sim, sim.int_sim, sim.tau_sim

    # Reads in the data, returns the data which has coverage of a given species (from simulated intensities)
    def read_file(self, filename, restfreqs, int_sim, shift=None, GHz=False, plot=False, block_interlopers=True):
        data = np.load(filename, allow_pickle=True)

        # Unpack data arrays
        freqs = data[0]
        intensity = data[1]
        if GHz:
            freqs *= 1000.0

        relevant_freqs = np.zeros(freqs.shape)
        relevant_intensity = np.zeros(intensity.shape)
        relevant_yerrs = np.zeros(freqs.shape)
        covered_trans = []

        # Iterate through rest frequencies to identify their corresponding spectral lines
        for i, rf in enumerate(restfreqs):
            thresh = 0.05  # Set a threshold as 5% of the peak intensity...
            if int_sim[i] > thresh * np.max(int_sim):  # find significant simulated intensities...
                vel = (rf - freqs) / rf * ckm + (shift if shift else self.aligned_velocity)  # calculate velocity shift for each frequency...
                locs = np.where((vel < (self.aligned_velocity + 1.5)) & (vel > (self.aligned_velocity - 1.5)))  # and filter for a velocity range that captures the line features

                if locs[0].size != 0:
                    noise_mean, noise_std = self.calc_noise_std(intensity[locs])
                    if block_interlopers and (np.max(intensity[locs]) > 3.5 * noise_std):  # 3.5σ threshold, 6σ for weaker species
                        print(f"{GRAY}{rf:10.4f} MHz  |  Interloping line detected.{RESET}")
                        if plot:
                            plt.plot(freqs[locs], intensity[locs])
                            plt.show()
                    else:
                        # Mark the transition and store relevant data
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

        return relevant_freqs, relevant_intensity, relevant_yerrs, covered_trans

    def init_setup(self):
        print(f"\n{CYAN}Running setup for: {self.mol_name}, block interlopers = {self.block_interlopers}.{RESET}")
        catfile_path = os.path.join(self.cat_folder, f"{self.mol_name}.cat")

        try:
            os.mkdir(self.fit_folder)
        except FileExistsError:
            pass
        try:
            os.mkdir(os.path.join(self.fit_folder, self.mol_name))
        except FileExistsError:
            pass
        if not os.path.exists(catfile_path):
            raise FileNotFoundError(f"{RED}No catalog file found at {catfile_path}.{RESET}")

        # Initialize molecular simulation components
        source_size = self.source_size if self.source_size is not None else self.template_means[0]
        obs_params = ObsParams("init", dish_size=self.dish_size, source_size=source_size)
        mol_cat = MolCat(self.mol_name, catfile_path)
        sim = MolSim(f"{self.mol_name} sim 8K", mol_cat, obs_params, vlsr=[self.aligned_velocity], C=[3.4e12],
                     dV=[0.89], T=[7.0], ll=[self.lower_limit], ul=[self.upper_limit], gauss=False)
        freq_sim = np.array(sim.freq_sim)
        int_sim = np.array(sim.int_sim)

        # Read and process spectral data
        print(f"{CYAN}Reading in spectral data from: {self.data_path}{RESET}")
        freqs_DSN, ints_DSN, yerrs_DSN, covered_trans_DSN = self.read_file(self.data_path, freq_sim, int_sim,
                                                                            block_interlopers=self.block_interlopers,
                                                                            plot=False)
        covered_trans_DSN = np.array(covered_trans_DSN, dtype=int)

        # Assemble data grid for MCMC fitting
        datagrid = np.array([freqs_DSN, ints_DSN, yerrs_DSN, covered_trans_DSN], dtype=object)
        datafile_path = os.path.join(self.fit_folder, self.mol_name, "all_" + self.mol_name + "_lines_DSN_freq_space.npy")
        print(f"\n{CYAN}Saving reduced spectrum to: {datafile_path}{RESET}")
        np.save(datafile_path, datagrid, allow_pickle=True)

        return datafile_path, catfile_path

    # Conduct Markov Chain Monte Carlo (MCMC) inference using emcee's ensemble sampler
    def fit_multi_gaussian(self, datafile, catalogue):
        print(f"{CYAN}Fitting column densities for {self.mol_name}.{RESET}")
        ndim = self.ndim
        if not os.path.exists(datafile):
            raise FileNotFoundError(f"{RED}The data file {datafile} could not be found.{RESET}")
        datagrid = np.load(datafile, allow_pickle=True)
        mol_cat = MolCat("mol", catalogue)

        # Choose initial parameters and perturbations based on run type
        if self.template_run:
            # Use the configurable template means and standard deviations
            initial = self.template_means
            prior_means = initial
            prior_stds = self.template_stds
            print(f"{GRAY}Using template priors and initial positions for {self.mol_name}.{RESET}")
            file_name = os.path.join(self.fit_folder, self.mol_name, "chain_template.npy")
        else:
            # Load priors from previous chain data
            if not os.path.exists(self.prior_path):
                raise FileNotFoundError(f"{RED}The prior path {self.prior_path} could not be found.{RESET}")
            print(f"{GRAY}Loading previous chain data from: {self.prior_path}{RESET}")

            # Load and reshape prior samples
            psamples = np.load(self.prior_path).T
            print(f"{GRAY}Dimensions of samples loaded from chain: {psamples.shape}{RESET}")

            # Compute prior means and standard deviations
            prior_means = np.mean(np.percentile(psamples, 50, axis=1), axis=1)
            percentile_16 = np.percentile(psamples, 16, axis=1).mean(axis=1)
            percentile_84 = np.percentile(psamples, 84, axis=1).mean(axis=1)
            prior_stds = np.abs((percentile_16 - prior_means + percentile_84 - prior_means) / 2.0)

            file_name = os.path.join(self.fit_folder, self.mol_name, "chain.npy")

            # Validate the shape of priors
            if prior_means.shape == (ndim,) and prior_stds.shape == (ndim,) and prior_means.ndim == 1 and prior_stds.ndim == 1:
                print(f"{GRAY}Priors are correctly shaped as 1-dimensional arrays with {ndim} elements each.{RESET}")
            else:
                raise ValueError(f"{RED}Priors should be 1-dimensional arrays with {ndim} elements each.{RESET}")


            # Load initial positions from the previous chain for non-template runs
            chain_data = np.load(self.prior_path)[:, -200:, :].reshape(-1, ndim).T
            initial = np.median(chain_data, axis=1)
            print(f"{GRAY}Loading initial positions from chain.{RESET}")

            # psamples = np.load(self.prior_path)
            # if psamples.shape[-1] != ndim:
            #     raise ValueError(f"{RED}The prior chain has {psamples.shape[-1]} dimensions, but {ndim} dimensions are required.{RESET}")
            # psamples = psamples.reshape(-1, ndim).T
            # print(f"{GRAY}Dimensions of samples loaded from chain: {psamples.shape}{RESET}")
            # prior_means = np.percentile(psamples, 50, axis=1)
            # percentile_16 = np.percentile(psamples, 16, axis=1)
            # percentile_84 = np.percentile(psamples, 84, axis=1)
            # prior_stds = (percentile_84 - percentile_16) / 2.0

        # Initialize walkers with a small perturbation relative to the prior standard deviations
        if self.source_size is not None:
            perturbation = np.array([0.1 * prior_means[0], 1.e-3, 1.e-3, 1.e-3])
        else:
            perturbation = np.array([1.e-1, 0.1 * prior_means[1], 1.e-3, 1.e-3, 1.e-3])
        pos = [initial + perturbation * np.random.randn(ndim) for _ in range(self.nwalkers)]
        print()

        # Perform affine invariant MCMC sampling
        if self.parallelize:
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.lnprob,
                                                args=(datagrid, mol_cat, prior_stds, prior_means), pool=pool)
                for _ in tqdm(range(self.nruns), desc=f"MCMC Sampling for {self.mol_name}", colour='white'):
                    sampler.run_mcmc(pos, 1)
                    np.save(file_name, sampler.chain)
                    pos = sampler.chain[:, -1, :]
            return sampler.chain
        else:
            # Initialize the sampler without parallelization (ideal for debugging)
            sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.lnprob,
                                            args=(datagrid, mol_cat, prior_stds, prior_means))
            for _ in tqdm(range(self.nruns), desc=f"MCMC Sampling for {self.mol_name}", colour='white'):
                sampler.run_mcmc(pos, 1)
                np.save(file_name, sampler.chain)
                pos = sampler.chain[:, -1, :]
            return sampler.chain

    def run(self):
        datafile_path, catalogue_path = self.init_setup()
        chain = self.fit_multi_gaussian(datafile_path, catalogue_path)

        # Verify that chain file path matches where data was saved
        if self.template_run:
            chain_path = os.path.join(self.fit_folder, self.mol_name, "chain_template.npy")
        else:
            chain_path = os.path.join(self.fit_folder, self.mol_name, "chain.npy")

        if os.path.exists(chain_path):
            # Call the custom plotting function
            plot_results(chain_path, self.param_labels, include_trace=False)
        else:
            print(f"{RED}Chain file not found at {chain_path}. Exiting.{RESET}")


if __name__ == "__main__":

    config = {
        # Frequently adjusted for specific MCMC runs
        'mol_name':          'hc5n_hfs',    # Molecule name, as named in CDMS_catalog
        'template_run':      True,          # True for template species; hardcoded initial positions for first run
        'nruns':             10000,         # MCMC iterations; higher values improve convergence
        'nwalkers':          128,           # Number of walkers; more walkers explore parameter space better

        # Physical priors (e.g. positivity constraints and limits)
        'bounds': {
            'source_size':   [30.0, 90.0],             # Source size in arcseconds
            'Ncol':          [10 ** 8.0, 10 ** 14.0],  # Column density (cm⁻²)
            'Tex':           [3.4, 12.0],              # Excitation temperature (K), avoid values below CMB (2.7 K)
            'vlsr':          [3.0, 5.5],               # Source velocity (km/s); for multiple sources, force sequential ordering
            'dV':            [0.35, 1.5],              # Line width (km/s)
        },

        # Priors for means (μ) and standard deviations (σ) of template species
        # Order of parameters: [source_size, Ncol, Tex, vlsr, dV]
        'template_means':    np.array([46.91, 3.4e10, 8.0, 4.3, 0.7575]),
        'template_stds':     np.array([6.5, 0.34e10, 3.0, 0.06, 0.22]),

        # Observation-specific settings for spectra
        'dish_size':         70,            # Telescope dish diameter (m)
        'lower_limit':       18000,         # Lower frequency limit (MHz)
        'upper_limit':       25000,         # Upper frequency limit (MHz)
        'aligned_velocity':  4.33,          # Velocity for spectral alignment (km/s)
        'fixed_source_size': None,          # Set to a numerical value to fix source size (4 free parameters), None or False for 5 free parameters

        # Usually unchanged unless paths or setup are modified
        'block_interlopers': True,          # Recommended True to block interloping lines
        'parallelize':       True,          # True for multiprocessing (faster); False for easier debugging
        'fit_folder':        os.path.join(os.getcwd(), 'DSN_fit_results'),
        'cat_folder':        os.path.join(os.getcwd(), 'CDMS_catalog'),
        'prior_path':        os.path.join(os.getcwd(), 'DSN_fit_results', 'hc5n_hfs', 'chain_template.npy'),
        'data_paths': {
            'hc5n_hfs':      os.path.join(os.getcwd(), 'DSN_data', 'cha_mms1_hc5n_example.npy'),
            'hc7n_hfs':      os.path.join(os.getcwd(), 'DSN_data', 'cha_mms1_hc7n_example.npy'),
            # Add more paths here...
        },
    }

    # Adjust template means and stds if fixed_source_size is specified
    if isinstance(config['fixed_source_size'], (float, int)):
        # Remove source_size from template_means and template_stds
        config['template_means'] = config['template_means'][1:]
        config['template_stds'] = config['template_stds'][1:]

    model = SpectralFitMCMC(config)
    model.run()