## Installation Instructions

Ensure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your machine before proceeding with the setup.

First, clone this repository to your local machine:

```bash
git clone https://github.com/KahaanGandhi/Cha1-MCMC.git
cd Cha1-MCMC
```
Create and activate a new Conda environment with the appropriate Python version (e.g., 3.9):

```bash
conda create --name cha1-mcmc-env python=3.9
conda activate cha1-mcmc-env
```
Install the dependencies listed in the requirements.txt file:
```bash
pip install -r requirements.txt
```
If you encounter any issues with specific packages that aren't available, install them manually:
```bash
conda install <package-name>
```

Alternatively, if you prefer to use [MacPorts](https://guide.macports.org/chunked/installing.macports.html), run the following:
```bash
# Clone this repository
git clone https://github.com/KahaanGandhi/Cha1-MCMC.git
cd Cha1-MCMC

# Install Python 3.9 using MacPorts
sudo port install python39
sudo port select --set python python39

# Install pip for Python 3.9
sudo port install py39-pip

# Install the dependencies from the requirements.txt file
pip-3.9 install -r requirements.txt
```

Again, if you encounter any issues with specific packages, install them manually:
```bash
sudo port install <package-name>
```

## Running Instructions

### Step 1: Preparing the Data

- Verify your data is in the correct `.npy` format, containing frequency and intensity arrays for the molecule of interest. Refer to `DSN_pipeline.ipynb`  for guidance on reformatting common file formats.
- Add your data to the `data_paths` section of the configuration in `MCMC_inference.py`, following the format:
```python
'molecule_name': os.path.join(os.getcwd(), 'your_data_folder', 'your_data_file.npy'),
```

### Step 2: Configuring the MCMC Run

- Open the `MCMC_inference.py` file and locate the `config` dictionary at the bottom of the script.
- Adjust parameters like `dish_size`, `lower_limit`, `upper_limit`, or `aligned_velocity` to match your telescope and observations. The configuration is currently set up for DSS-43 observations of Chamaeleon I, and should be adjusted accordingly for different telescopes or sources. Verify that your molecule of interest has rotational transitions that fall within the specified frequency range.

- You can also increase `nwalkers` or `nruns` to explore more of the parameter space during MCMC sampling.

### Step 3: Running the Initial MCMC

- For your initial run, set ```template_run``` to ```True``` to use hardcoded initial values specific to the template species.
- Cyanopolyynes like HC<sub>5</sub>N tend to share source properties, so it is recommended to first obtain a fit for shorter linear molecules, as their transitions are more easily detectable above noise levels.
- Run the script from the main directory:
```bash
cd /path/to/Cha1-MCMC
python MCMC_inference.py
```

For MacPorts users, remember to explictly call the correct Python version when running scripts:
```bash
python3.9 MCMC_inference.py
```

### Step 4: Refining the Fit

- After the initial run, set `template_run` to `False` to load priors from the previous run into the analysis.
- You can rerun the script to refine the fit:
```bash
python MCMC_inference.py
```
- If desired, you can redo the template run (perhaps with more walkers or steps for a more thorough exploration of the parameter space), which will overwrite the previous template run results.
