import numpy as np
import emcee
import matplotlib.pyplot as plt
import multiprocessing
import os
from classes import *
from constants import *
from numba import njit
from tqdm import tqdm


# Calculates local RMS noise in a given spectrum by iteratively masking outliers. 3.5σ default, 6σ for weaker species. 
def calc_noise_std(intensity, threshold=3.5):
    dummy_ints = np.copy(intensity)
    noise = np.copy(intensity)
    dummy_mean = np.nanmean(dummy_ints)
    dummy_std = np.nanstd(dummy_ints)

    # Repeat 3 times to make sure to avoid any interloping lines
    for chan in np.where(dummy_ints-dummy_mean < (-dummy_std*threshold))[0]:
        noise[chan-10:chan+10] = np.nan
    for chan in np.where(dummy_ints-dummy_mean > (dummy_std*threshold))[0]:
        noise[chan-10:chan+10] = np.nan
    noise_mean = np.nanmean(noise)
    noise_std = np.nanstd(np.real(noise))

    for chan in np.where(dummy_ints-noise_mean < (-noise_std*threshold))[0]:
        noise[chan-10:chan+10] = np.nan
    for chan in np.where(dummy_ints-noise_mean > (noise_std*threshold))[0]:
        noise[chan-10:chan+10] = np.nan
    noise_mean = np.nanmean(noise)
    noise_std = np.nanstd(np.real(noise))

    for chan in np.where(dummy_ints-noise_mean < (-noise_std*threshold))[0]:
        noise[chan-10:chan+10] = np.nan
    for chan in np.where(dummy_ints-noise_mean > (noise_std*threshold))[0]:
        noise[chan-10:chan+10] = np.nan
    noise_mean = np.nanmean(noise)
    noise_std = np.nanstd(np.real(noise))

    return noise_mean, noise_std


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
def make_model(freqs1, freqs2, freqs3, freqs4, ints1, ints2, ints3, ints4, ss1, ss2, ss3, ss4, datagrid0, datagrid1, vlsr1, vlsr2, vlsr3, vlsr4, dV, Tex):
    curr_model = np.zeros(datagrid1.shape)
    model1 = np.zeros(datagrid1.shape)
    model2 = np.zeros(datagrid1.shape)
    model3 = np.zeros(datagrid1.shape)
    model4 = np.zeros(datagrid1.shape)
    nlines = freqs1.shape[0]

    # Compute Gaussian profiles for each source and sum them
    for i in range(nlines):
        vel_grid = (freqs1[i]-datagrid0)/freqs1[i]*ckm  # Convert frequency shifts to velocity space
        mask = np.abs(vel_grid-5.8) < dV*10
        
        # Gaussian profiles for the intensity at each frequency point
        model1[mask] += ints1[i]*np.exp(-0.5*((vel_grid[mask] - vlsr1)/(dV/2.355))**2.)
        model2[mask] += ints2[i]*np.exp(-0.5*((vel_grid[mask] - vlsr2)/(dV/2.355))**2.)
        model3[mask] += ints3[i]*np.exp(-0.5*((vel_grid[mask] - vlsr3)/(dV/2.355))**2.)
        model4[mask] += ints4[i]*np.exp(-0.5*((vel_grid[mask] - vlsr4)/(dV/2.355))**2.)

    # Apply the Planck function for thermal radiation, adjusted for background cosmic temperature (2.7 K)
    J_T = (h*datagrid0*10**6/k)*(np.exp(((h*datagrid0*10**6)/(k*Tex))) -1)**-1
    J_Tbg = (h*datagrid0*10**6/k)*(np.exp(((h*datagrid0*10**6)/(k*2.7))) -1)**-1
    
    # Apply the beam dilution correction to each model
    model1 = apply_beam(datagrid0, (J_T - J_Tbg)*(1 - np.exp(-model1)), ss1, 100)
    model2 = apply_beam(datagrid0, (J_T - J_Tbg)*(1 - np.exp(-model2)), ss2, 100)
    model3 = apply_beam(datagrid0, (J_T - J_Tbg)*(1 - np.exp(-model3)), ss3, 100)
    model4 = apply_beam(datagrid0, (J_T - J_Tbg)*(1 - np.exp(-model4)), ss4, 100)

    # Sum all the individual models to create the composite model
    curr_model = model1 + model2 + model3 + model4

    return curr_model



def lnlike(theta, datagrid, mol_cat):
    tot_lnlike = 0.
    yerrs = datagrid[2]
    line_ids = datagrid[3]
    source_size1, source_size2, source_size3, source_size4, Ncol1, Ncol2, Ncol3, Ncol4, Tex, vlsr1, vlsr2, vlsr3, vlsr4, dV = theta

    freqs1, ints1, taus1 = predict_intensities(source_size1, Ncol1, Tex, dV, mol_cat)
    freqs2, ints2, taus2 = predict_intensities(source_size2, Ncol2, Tex, dV, mol_cat)
    freqs3, ints3, taus3 = predict_intensities(source_size3, Ncol3, Tex, dV, mol_cat)
    freqs4, ints4, taus4 = predict_intensities(source_size4, Ncol4, Tex, dV, mol_cat)

    freqs1 = np.array(freqs1)[line_ids]
    freqs2 = np.array(freqs2)[line_ids]
    freqs3 = np.array(freqs3)[line_ids]
    freqs4 = np.array(freqs4)[line_ids]

    taus1 = np.array(taus1)[line_ids]
    taus2 = np.array(taus2)[line_ids]
    taus3 = np.array(taus3)[line_ids]
    taus4 = np.array(taus4)[line_ids]

    ints1 = np.array(ints1)[line_ids]
    ints2 = np.array(ints2)[line_ids]
    ints3 = np.array(ints3)[line_ids]
    ints4 = np.array(ints4)[line_ids]

    curr_model = make_model(freqs1, freqs2, freqs3, freqs4, taus1, taus2, taus3, taus4, source_size1, source_size2, source_size3, source_size4, datagrid[0], datagrid[1], vlsr1, vlsr2, vlsr3, vlsr4, dV, Tex)

    inv_sigma2 = 1.0/(yerrs**2)
    tot_lnlike = np.sum((datagrid[1] - curr_model)**2*inv_sigma2 - np.log(inv_sigma2))
    return -0.5*tot_lnlike



def lnprior(theta, prior_stds, prior_means):
    source_size1, source_size2, source_size3, source_size4, Ncol1, Ncol2, Ncol3, Ncol4, Tex, vlsr1, vlsr2, vlsr3, vlsr4, dV = theta
    s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13 = prior_stds
    m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13 = prior_means

    # in several cases the standard deviations on some of the parameters are too restrictive (e.g. vlsr and dV). Relaxing slightly
    s9 = m13*0.8
    s10 = m13*0.8
    s11 = m13*0.8
    s12 = m13*0.8
    s13 = m13*0.3

    # set custom priors and limits here
    if (0. < source_size1 < 200) and (0. < source_size2 < 200) and (0. < source_size3 < 200) and (0. < source_size4 < 200) and (0. < Ncol1 < 10**16.) and (0. < Ncol2 < 10**16.) and (0. < Ncol3 < 10**16.) and (0. < Ncol4 < 10**16.) and (vlsr1 < (vlsr2-0.05)) and (vlsr2 < (vlsr3-0.05)) and (vlsr3 < (vlsr4-0.05)) and (vlsr2 < (vlsr1+0.3)) and (vlsr3 < (vlsr2+0.3)) and (vlsr4 < (vlsr3+0.3)) and dV < 0.3:
        
        p0 = np.log(1.0/(np.sqrt(2*np.pi)*s0))-0.5*(source_size1-m0)**2/s0**2
        p1 = np.log(1.0/(np.sqrt(2*np.pi)*s1))-0.5*(source_size2-m1)**2/s1**2
        p2 = np.log(1.0/(np.sqrt(2*np.pi)*s2))-0.5*(source_size3-m2)**2/s2**2
        p3 = np.log(1.0/(np.sqrt(2*np.pi)*s3))-0.5*(source_size4-m3)**2/s3**2

        p8 = np.log(1.0/(np.sqrt(2*np.pi)*s8))-0.5*(Tex-m8)**2/s8**2

        p9 = np.log(1.0/(np.sqrt(2*np.pi)*s9))-0.5*(vlsr1-m9)**2/s9**2
        p10 = np.log(1.0/(np.sqrt(2*np.pi)*s10))-0.5*(vlsr2-m10)**2/s10**2
        p11 = np.log(1.0/(np.sqrt(2*np.pi)*s11))-0.5*(vlsr3-m11)**2/s11**2
        p12 = np.log(1.0/(np.sqrt(2*np.pi)*s12))-0.5*(vlsr4-m12)**2/s12**2

        p13 = np.log(1.0/(np.sqrt(2*np.pi)*s13))-0.5*(dV-m13)**2/s13**2

        return p0 + p1 + p2 + p3 + p8 + p9 + p10 + p11 + p12 + p13

    return -np.inf

def lnprob(theta, datagrid, mol_cat, prior_stds, prior_means):
    lp = lnprior(theta, prior_stds, prior_means)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, datagrid, mol_cat)


def fit_multi_gaussian(datafile, fit_folder, catalogue, nruns, mol_name, prior_path, restart=True):
    print(f"Fitting column densities for {mol_name}. restart = {restart}.")

    ndim, nwalkers = 14, 128
    datagrid = np.load(datafile, allow_pickle=True)
    mol_cat = MolCat(mol_name, catalogue) 

    # Define initial parameter values for the MCMC walkers
    # initial = np.array([99, 65, 265, 262, 1.98e12, 6.22e12, 2.92e12, 4.88e12, 6.1, 5.595, 5.764, 5.886, 6.017, 0.121]) # BENZONITRILE
    initial = np.array([42.8, 24.3, 47.9, 21.5, 5.8e13, 9.5e13, 4.e13, 1.06e14, 7.7, 5.603, 5.745, 5.873, 6.024, 0.1568]) # LOOMIS
    # initial = np.array([37, 25, 56, 22, 2.47e12, 11.19e12, 2.20e12, 5.64e12, 6.7, 5.624, 5.790, 5.910, 6.033, 0.117])  # HC9N

    if restart:
        print("Initializing tight ball of walkers around hardcoded values.")
        perturbation = np.array([1.e-1, 1.e-1, 1.e-1, 1.e-1, 1.e10, 1.e10, 1.e10, 1.e10, 1.e-3, 1.e-3, 1.e-3, 1.e-3, 1.e-3, 1.e-3])
        pos = [initial + perturbation * np.random.randn(ndim) for i in range(nwalkers)]
    else:
        print(f"Initializing tight ball of walkers using prior chain values of {mol_name}.")
        chain_data = np.load(os.path.join(fit_folder, mol_name, "chain.npy"))[:,-200:,:].reshape(-1, ndim).T
        initial = np.median(chain_data, axis=1)
        perturbation = np.array([1.e-1, 1.e-1, 1.e-1, 1.e-1, 1.e10, 1.e10, 1.e10, 1.e10, 1.e-3, 1.e-3, 1.e-3, 1.e-3, 1.e-3, 1.e-3])
        pos = [initial + perturbation * np.random.randn(ndim) for i in range(nwalkers)]

    # Load and process prior samples
    psamples = np.load(prior_path).T
    psamples_flat = psamples.reshape(ndim, -1)
    
    # Compute prior means and standard deviations
    prior_means = np.percentile(psamples_flat, 50, axis=1)
    lower_percentiles = np.percentile(psamples_flat, 16, axis=1)
    upper_percentiles = np.percentile(psamples_flat, 84, axis=1)
    prior_stds = (np.abs(lower_percentiles - prior_means) + np.abs(upper_percentiles - prior_means)) / 2


    def is_within_bounds(params):
        source_size1, source_size2, source_size3, source_size4, Ncol1, Ncol2, Ncol3, Ncol4, Tex, vlsr1, vlsr2, vlsr3, vlsr4, dV = params
        if not (0. < source_size1 < 200):
            return False, f"source_size1 out of bounds: {source_size1}"
        if not (0. < source_size2 < 200):
            return False, f"source_size2 out of bounds: {source_size2}"
        if not (0. < source_size3 < 200):
            return False, f"source_size3 out of bounds: {source_size3}"
        if not (0. < source_size4 < 200):
            return False, f"source_size4 out of bounds: {source_size4}"
        if not (0. < Ncol1 < 10**16):
            return False, f"Ncol1 out of bounds: {Ncol1}"
        if not (0. < Ncol2 < 10**16):
            return False, f"Ncol2 out of bounds: {Ncol2}"
        if not (0. < Ncol3 < 10**16):
            return False, f"Ncol3 out of bounds: {Ncol3}"
        if not (0. < Ncol4 < 10**16):
            return False, f"Ncol4 out of bounds: {Ncol4}"
        if not (vlsr1 < (vlsr2 - 0.05)):
            return False, f"vlsr1 not less than vlsr2 - 0.05: {vlsr1} >= {vlsr2 - 0.05}"
        if not (vlsr2 < (vlsr3 - 0.05)):
            return False, f"vlsr2 not less than vlsr3 - 0.05: {vlsr2} >= {vlsr3 - 0.05}"
        if not (vlsr3 < (vlsr4 - 0.05)):
            return False, f"vlsr3 not less than vlsr4 - 0.05: {vlsr3} >= {vlsr4 - 0.05}"
        if not (vlsr2 < (vlsr1 + 0.3)):
            return False, f"vlsr2 not less than vlsr1 + 0.3: {vlsr2} >= {vlsr1 + 0.3}"
        if not (vlsr3 < (vlsr2 + 0.3)):
            return False, f"vlsr3 not less than vlsr2 + 0.3: {vlsr3} >= {vlsr2 + 0.3}"
        if not (vlsr4 < (vlsr3 + 0.3)):
            return False, f"vlsr4 not less than vlsr3 + 0.3: {vlsr4} >= {vlsr3 + 0.3}"
        if not (dV < 0.3):
            return False, f"dV out of bounds: {dV}"
        return True, "Within bounds"

    # Check how many walkers are within the bounds and print their positions
    valid_walkers_info = [is_within_bounds(walker) for walker in pos]
    num_valid_walkers = sum(1 for valid, _ in valid_walkers_info if valid)
    print(f"Number of walkers within bounds: {num_valid_walkers}/{nwalkers}")
    for i, (walker, (valid, info)) in enumerate(zip(pos, valid_walkers_info)):
        if not valid:
            print(f"Walker {i}: {walker}, Valid: {valid}, Info: {info}")
    
    ncores = multiprocessing.cpu_count()

    # Set up the sampler with a multiprocessing pool
    with multiprocessing.Pool(ncores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(datagrid, mol_cat, prior_stds, prior_means), pool=pool)

        for i in tqdm(range(nruns), desc=f"MCMC Sampling for {mol_name}"):
            sampler.run_mcmc(pos, 1)
            file_name = os.path.join(fit_folder, mol_name, "chain.npy")
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

    mol_cat = MolCat(mol_name, catfile)
    obs_params = ObsParams("init", source_size=40)
    sim = MolSim(f"{mol_name} sim 8K", mol_cat, obs_params, [0.0], [7.e11], [0.37], [8.], ll=[7000], ul=[30000], gauss=False)
    freq_sim = np.array(sim.freq_sim)
    int_sim = np.array(sim.int_sim)

    print(f"Reading in data from {data_path}")
    freqs_gotham, ints_gotham, yerrs_gotham, covered_trans_gotham = read_file(data_path, freq_sim, int_sim, block_interlopers=block_interlopers, plot=False)
    covered_trans_gotham = np.array(covered_trans_gotham, dtype=int)
    datagrid = [freqs_gotham, ints_gotham, yerrs_gotham, covered_trans_gotham]
    datagrid = np.array(datagrid, dtype=object)
    datafile_path = os.path.join(fit_folder, mol_name, "all_" + mol_name + "_lines_GOTHAM_freq_space.npy")
    print("Saving data to: " + datafile_path)
    for i, item in enumerate(datagrid):
        print(f"Datagrid element {i}  |  Type: {type(item)}  |  Shape: {item.shape if isinstance(item, np.ndarray) else 'N/A'}")

    np.save(datafile_path, datagrid, allow_pickle=True)

    return datafile_path, catfile

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Define the new relative paths
    input_dict = {
        'mol_name': 'hc11n',
        'fit_folder': os.path.join(BASE_DIR, 'fit_results'),
        'cat_folder': os.path.join(BASE_DIR, 'GOTHAM-catalogs'),
        'data_path': os.path.join(BASE_DIR, 'GOTHAM-data', 'hc11n_chunks.npy'),
        'block_interlopers': True,
        'nruns': 10000,
        'restart': False,
        'prior_path': os.path.join(BASE_DIR, 'fit_results', 'hc9n_hfs', 'chain.npy'),
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
        restart=input_dict['restart']
    )
    
    param_labels = [
        "Source Size 1", "Source Size 2", "Source Size 3", "Source Size 4",
        "Column Density 1", "Column Density 2", "Column Density 3", "Column Density 4",
        "Excitation Temperature", 
        "Velocity LSR 1", "Velocity LSR 2", "Velocity LSR 3", "Velocity LSR 4",
        "Delta V"
    ]

    # Verify that chain file path matches where data was saved
    CHAIN_PATH = os.path.join(input_dict['fit_folder'], input_dict['mol_name'], "chain.npy")
    if os.path.exists(CHAIN_PATH):
        plot_results(CHAIN_PATH, param_labels)
    else:
        print(f"Chain file not found at {CHAIN_PATH}.")
        