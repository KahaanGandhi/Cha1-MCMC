import numpy as np
from numpy import exp as exp
import time as tm
import corner
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from constants import *


# Calculate a partition function at a given T.  The catalog used must have enough lines in it to fully capture the partition function, or the result will not be accurate for Q.
def calc_q(catalog, T):
    Q = 0.
    
    if 'n2h+_hfs.cat' in catalog.catalog_file.lower():
        Q = -3.13420474e-08*T**3 + 3.28722820e-05*T**2 + 4.01951955e+00*T + 3.32018827e+00

    elif 'acetone.cat' in catalog.catalog_file.lower():
        Q = 2.91296*10**(-7)*T**6 - 0.00021050085*T**5 + 0.05471337*T**4 - 5.5477*T**3 + 245.28*T**2 - 2728.3*T + 16431 #Hard code for Acetone
        
    elif 'sh.cat' in catalog.catalog_file.lower():
        Q = 0.000000012549467*T**4 - 0.000008528126823*T**3 + 0.002288160909445*T**2 + 0.069272946237033*T + 15.357239728157400
 #Hard code for SH.  Completely unreliable below 2.735 K or above 300 K.

    elif 'h2s.cat' in catalog.catalog_file.lower():
        Q = -0.000004859941547*T**3 + 0.005498622332982*T**2 + 0.507648423477309*T - 1.764494755639740 #Hard code for H2S.  Completely unreliable below 2.735 K or above 300 K.
    
    elif 'hcn.cat' in catalog.catalog_file.lower():
        Q = -1.64946939*10**-9*T**4 + 4.62476813*10**-6*T**3 - 1.15188755*10**-3*T**2 + 1.48629408*T + .386550361
        
    elif 'methanol.cat' in catalog.catalog_file.lower() or 'ch3oh.cat' in catalog.catalog_file.lower() or 'ch3oh_v0.cat' in catalog.catalog_file.lower() or 'ch3oh_v1.cat' in catalog.catalog_file.lower() or 'ch3oh_v2.cat' in catalog.catalog_file.lower() or 'ch3oh_vt.cat' in catalog.catalog_file.lower():
        Q = 4.83410*10**-11*T**6 - 4.04024*10**-8*T**5 + 1.27624*10**-5*T**4 - 1.83807*10**-3*T**3 + 2.05911*10**-1*T**2 + 4.39632*10**-1*T -1.25670
        
    elif '13methanol.cat' in catalog.catalog_file.lower() or '13ch3oh.cat' in catalog.catalog_file.lower():
        Q = 0.000050130*T**3 + 0.076540934*T**2 + 4.317920731*T - 31.876881967
        
    elif 'c2n.cat' in catalog.catalog_file.lower() or 'ccn.cat' in catalog.catalog_file.lower():
        Q = 1.173755*10**(-11)*T**6 - 1.324086*10**(-8)*T**5 + 5.99936*10**(-6)*T**4 - 1.40473*10**(-3)*T**3 + 0.1837397*T**2 + 7.135161*T + 22.55770
        
    elif 'ch2nh.cat' in catalog.catalog_file.lower():
        Q = 1.2152*T**1.4863
        
    elif '13ch3oh.cat' in catalog.catalog_file.lower() or 'c033502.cat' in catalog.catalog_file.lower():
        Q = 0.399272*T**1.756329


    # GOTHAM RELEVANT PARTITION FUNCTIONS
    # Cyanopolyynes and isocyanides
    elif 'hc3n' in catalog.catalog_file.lower() and 'hfs' not in catalog.catalog_file.lower():
        Q = 4.581898*T + 0.2833
    elif 'hc3n' in catalog.catalog_file.lower() and 'hfs' in catalog.catalog_file.lower():
        Q = 3*(4.581898*T + 0.2833)

    elif 'hc2nc_hfs' in catalog.catalog_file.lower():
        Q = 12.58340*T + 1.0604


    elif 'hc5n' in catalog.catalog_file.lower() and 'hfs' not in catalog.catalog_file.lower():
        Q = 15.65419*T + 0.2214
    elif 'hc5n' in catalog.catalog_file.lower() and 'hfs' in catalog.catalog_file.lower():
        Q = 3*(15.65419*T + 0.2214)

    elif 'hc4nc' in catalog.catalog_file.lower() and 'hfs' not in catalog.catalog_file.lower():
        Q = (44.62171*T + 0.6734)/3                            
    elif 'hc4nc' in catalog.catalog_file.lower() and 'hfs' in catalog.catalog_file.lower():
        Q = 44.62171*T + 0.6734

    elif 'hc7n' in catalog.catalog_file.lower() and 'hfs' not in catalog.catalog_file.lower():
        Q = 36.94999*T + 0.1356
    elif 'hc7n' in catalog.catalog_file.lower() and 'hfs' in catalog.catalog_file.lower():
        Q = 3*(36.94999*T + 0.1356)

    elif 'hc6nc' in catalog.catalog_file.lower() and 'hfs' not in catalog.catalog_file.lower():
        Q = (107.3126*T + 1.2714)/3        
    elif 'hc6nc' in catalog.catalog_file.lower() and 'hfs' in catalog.catalog_file.lower():
        Q = 107.3126*T + 1.2714

    elif 'hc9n' in catalog.catalog_file.lower() and 'hfs' not in catalog.catalog_file.lower():
        Q = 71.7308577*T + 0.02203968
    elif 'hc9n' in catalog.catalog_file.lower() and 'hfs' in catalog.catalog_file.lower(): 
        Q = 3*(71.7308577*T + 0.02203968)

    elif 'hc11n.cat' in catalog.catalog_file.lower() and 'hfs' not in catalog.catalog_file.lower():
        Q = 123.2554*T + 0.1381
    elif 'hc11n' in catalog.catalog_file.lower() and 'hfs' in catalog.catalog_file.lower(): 
        Q = 3*(123.2554*T + 0.1381)


    # OTHER GOTHAM
    elif 'propargylcyanide' in catalog.catalog_file.lower():
        Q = 41.542*T**1.5008

    elif 'pyrrole' in catalog.catalog_file.lower():
        Q = 27.727*T**1.4752

    elif 'cyclopropylcyanide_hfs' in catalog.catalog_file.lower():
        Q = 38.199*T**1.4975

    elif 'pyridine' in catalog.catalog_file.lower():
        Q = 50.478*T**1.4955

    elif '1-cyanonaphthalene' in catalog.catalog_file.lower():
        Q = 560.39*T**1.4984

    elif '2-cyanonaphthalene' in catalog.catalog_file.lower():
        Q = 562.57*T**1.4993

    elif 'furan' in catalog.catalog_file.lower():
        Q = 33.725*T**1.4982

    elif 'phenol' in catalog.catalog_file.lower():
        Q = 264.20*T**1.4984

    elif 'benzaldehyde' in catalog.catalog_file.lower():
        Q = 53.798*T**1.4997

    elif 'anisole' in catalog.catalog_file.lower():
        Q = 54.850*T**1.4992

    elif 'azulene' in catalog.catalog_file.lower():
        Q = 96.066*T**1.4988

    elif 'acenaphthene' in catalog.catalog_file.lower():
        Q = 161.29*T**1.4994

    elif 'acenapthylene' in catalog.catalog_file.lower():
        Q = 151.58*T**1.4988

    elif 'fluorene' in catalog.catalog_file.lower():
        Q = 219.51*T**1.4996

    elif 'benzonitrile' in catalog.catalog_file.lower():
        Q = 25.896*T**1.4998 + 0.38109

        if T > 60:
            print('Warning: Extrapolating Q beyond 60 K for this molecule gets progressively iffier.')


    else:
        nstates = catalog.elower.size #Determine the number of total states in the raw cat file
        combined_array = np.empty(shape=(nstates,catalog.qns+1)) #Set up an array that has [J, ka, kc, Elower]

        if (catalog.qns == 1):
            for i in range(nstates): #Fill that array with [J, ka, kc, Elower]
                combined_array[i][0] = catalog.qn7[i]
                combined_array[i][1] = catalog.elower[i]

        if (catalog.qns == 2):
            for i in range(nstates): #Fill that array with [J, ka, kc, Elower]
                combined_array[i][0] = catalog.qn7[i]
                combined_array[i][1] = catalog.qn8[i]
                combined_array[i][2] = catalog.elower[i]
    
        if (catalog.qns == 3):
            for i in range(nstates): #Fill that array with [J, ka, kc, Elower]
                combined_array[i][0] = catalog.qn7[i]
                combined_array[i][1] = catalog.qn8[i]
                combined_array[i][2] = catalog.qn9[i]
                combined_array[i][3] = catalog.elower[i]
            
        if (catalog.qns == 4):
            for i in range(nstates): #Fill that array with [J, ka, kc, QN10, Elower]
                combined_array[i][0] = catalog.qn7[i]
                combined_array[i][1] = catalog.qn8[i]
                combined_array[i][2] = catalog.qn9[i]
                combined_array[i][3] = catalog.qn10[i]
                combined_array[i][4] = catalog.elower[i]            

        if (catalog.qns == 5):
            for i in range(nstates): #Fill that array with [J, ka, kc, QN10, QN11, Elower]
                combined_array[i][0] = catalog.qn7[i]
                combined_array[i][1] = catalog.qn8[i]
                combined_array[i][2] = catalog.qn9[i]
                combined_array[i][3] = catalog.qn10[i]
                combined_array[i][4] = catalog.qn11[i]
                combined_array[i][5] = catalog.elower[i]    
            
        if (catalog.qns == 6):
            for i in range(nstates): #Fill that array with [J, ka, kc, QN10, QN11, QN12, Elower]
                try:
                    combined_array[i][0] = catalog.qn7[i]
                except ValueError:
                    print('I choked at index {}.' .format(i))
                    quit()
                combined_array[i][1] = catalog.qn8[i]
                combined_array[i][2] = catalog.qn9[i]
                combined_array[i][3] = catalog.qn10[i]
                combined_array[i][4] = catalog.qn11[i]
                combined_array[i][5] = catalog.qn12[i]
                combined_array[i][6] = catalog.elower[i]                                    
        
        temp = list(set(map(tuple,combined_array))) #Don't know HOW this works, but it does: sorts through the array and removes all duplicate entries, so that only a single entry remains for each set of quantum numbers.
        ustates = len(temp) #Number of unique lower states
    
        for i in range(ustates):
            J = temp[i][0] #Extract a J value from the list
            E = temp[i][catalog.qns] #Extract its corresponding energy

            Q += (2*J+1)*exp(np.float64(-E/(kcm*T))) #Add it to Q

    return Q



# Fixes +/- quantum number issues
def fix_pm(qnarray):
    if '+' or '-' in qnarray:
        qnarray[qnarray == ''] = '0'
        qnarray[qnarray == '+'] = '1'
        qnarray[qnarray == '-'] = '2'
    return qnarray



# Fixes quantum number issues arising from the use of alphabet characters to represent numbers in spcat
def fix_qn(qnarray,line,old_qn):
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
                
    qnarray[line] = int(new_qn)        

    return




# Trim any given input array to the specified frequency ranges
def trim_array(array,frequency,ll,ul):
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
            i = np.where(frequency > tmp_ll[z])[0][0]     #get the index of the first value above the lower limit
        except IndexError:
            if frequency[-1] < tmp_ll[z]:
                continue
            else:
                i = 0                                    #if the catalog begins after the lower limit
            
        try:
            i2 = np.where(frequency > tmp_ul[z])[0][0]    #get the index of the first value above the upper limit    
        except IndexError:
            i2 = len(frequency)                            #if the catalog ends before the upper limit is reached        
            
        if foo == 0:
            trimmed_array = np.copy(array[i:i2])
            foo = 1
            
        else:
            trimmed_array = np.append(trimmed_array,array[i:i2])
    
    return trimmed_array





# Simulates Gaussian profiles for lines, after the intensities have been calculated.  Tries to be smart with how large a range it simulates over, for computational resources.  Includes a thermal cutoff for optically-thick lines.            
def sim_gaussian(molsim, int_sim, freq, comp_idx, chunk_idx):
    freq_gauss_tmp = []
    x = 0

    while (x < len(int_sim)):
        l_f = molsim.dV[comp_idx]*freq[x]/ckm #get the FWHM in MHz
        min_f = freq[x] - 10*l_f #get the frequency 10 FWHM lower
        max_f = freq[x] + 10*l_f #get the frequency 10 FWHM higher
        
        if x < len(int_sim)-2:
            while (freq[x+1] < max_f and x < len(int_sim)-2):
                    x += 1
                    max_f = freq[x] + 10*l_f #get the frequency 10 FWHM higher
    
        freq_line = np.arange(min_f, max_f, molsim.res[chunk_idx]) #generate a chunk of spectra at resolution res
        freq_gauss_tmp.extend(freq_line)
        x+= 1
    
    freq_gauss_tmp.sort()
    freq_gauss = np.asarray(freq_gauss_tmp)
    
    int_gauss = np.copy(freq_gauss)
    int_gauss *= 0.0
    
    start_time = tm.time()
    alerted = False

    for x in range(int_sim.shape[0]):
        #if abs(int_sim[x]) < rms/10:
        #    continue
    
        telapsed = tm.time() - start_time
        
        if telapsed > 5 and alerted == False:
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
    
        l_f = molsim.dV[comp_idx]*freq[x]/ckm #get the FWHM in MHz
        c = l_f/2.35482

        J_T = (h*freq_gauss*10**6/k)*(np.exp(((h*freq_gauss*10**6)/(k*molsim.T[comp_idx]))) -1)**-1
        J_Tbg = (h*freq_gauss*10**6/k)*(np.exp(((h*freq_gauss*10**6)/(k*molsim.obs_params.Tbg))) -1)**-1
        int_gauss += int_sim[x]*exp(-((freq_gauss - freq[x])**2/(2*c**2)))
    
    J_T = (h*freq_gauss*10**6/k)*(np.exp(((h*freq_gauss*10**6)/(k*molsim.T[comp_idx]))) -1)**-1
    J_Tbg = (h*freq_gauss*10**6/k)*(np.exp(((h*freq_gauss*10**6)/(k*molsim.obs_params.Tbg))) -1)**-1
    
    #int_gauss_tau = (J_T - J_Tbg)*(1 - np.exp(-int_gauss))
    int_gauss_tau = int_gauss
    
    # regrid
    freq_sim = np.arange(molsim.ll[chunk_idx], molsim.ul[chunk_idx]+1e-8, molsim.res[chunk_idx])
    int_sim = np.interp(freq_sim, freq_gauss, int_gauss_tau)

    return(freq_sim,int_sim)


# Apply a beam dilution correction factor
def apply_beam(frequency, intensity, source_size, dish_size):
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

def plot_results(chain_path, param_labels):
    
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "DejaVu Sans"
    })
    
    # Load the MCMC chain
    chain = np.load(chain_path)
    
    # Remove burn-in (first 20% of steps)
    burn_in = int(0.2 * chain.shape[1])
    chain = chain[:, burn_in:, :]
    
    # Reshape the chain to (nwalkers*nsteps, ndim)
    samples = chain.reshape((-1, chain.shape[-1]))
    
    # Generating corner plot
    fig = corner.corner(samples, labels=param_labels, quantiles=[0.16, 0.5, 0.84], truths=[None]*len(param_labels), show_titles=True, title_kwargs={"fontsize": 12})
    fig.savefig(f"{chain_path[:-4]}_corner.png")  # Save the figure

    # Plotting trace plots
    fig, axes = pl.subplots(nrows=chain.shape[2], figsize=(10, 2 * chain.shape[2]))
    for i in range(chain.shape[2]):
        ax = axes[i]
        ax.plot(chain[:, :, i].T, color="k", alpha=0.3)  # Transpose for proper plotting
        ax.set_title(f'Parameter {i+1}: {param_labels[i]}')
    axes[-1].set_xlabel("Step Number")
    plt.tight_layout()
    fig.savefig(f"{chain_path[:-4]}_trace.png")  # Save the figure

    format_value = lambda x: f"{x:.2e}" if abs(x) < 1e-3 or abs(x) > 1e3 else f"{x:.5f}"
    print("\nParameter Estimates:")
    for i, label in enumerate(param_labels):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        median = format_value(mcmc[1])
        lower = format_value(q[0])
        upper = format_value(q[1])
        print(f'{label}: {median} [-{lower} +{upper}]')