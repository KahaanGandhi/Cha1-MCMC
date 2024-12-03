Spectroscopic observations enable studies of the chemical composition of star-forming regions based on their distinct molecular emission and absorption spectra. The difficulty of detecting complex organic molecules, which probe chemical complexity, scales with their size — emission from heavy molecules is distributed over many lines, often too weak to detect individually. To address this, MCMC algorithms fit spectral models to sparse spectra, enabling molecular detection through multi-transition radio observations.

## Installation Instructions

Ensure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your machine before proceeding. First, clone this repository to your local machine:

```bash
git clone https://github.com/KahaanGandhi/Cha1-MCMC.git
cd Cha1-MCMC
```
Create and activate a new Conda environment:

```bash
conda create --name cha1-mcmc-env python=3.9
conda activate cha1-mcmc-env
```
Install the dependencies listed in the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Running Instructions

### Step 1: Preparing the Data

- Verify your data is in the correct `.npy` format, containing frequency and intensity arrays for the molecule of interest. To reformat common file formats, refer to `notebooks/DSN_pipeline.ipynb`.
- Add your data to `data_paths` in `inference.py`, following the format:
```python
'molecule_name': os.path.join(os.getcwd(), 'your_data_folder', 'your_data_file.npy'),
```

### Step 2: Configuring the MCMC Run

- Open the `inference.py` file and locate the `config` dictionary at the bottom of the script.
- **Observation parameters**: Adjust parameters like `dish_size`, `lower_limit`, and `upper_limit` to match your telescope and source. Ensure that your molecule has rotational transitions that fall in the specified frequency range. The current configuration is for DSS-43 observations of Chamaeleon I. 
- **Source size and column density**:
  - Source size is highly covariant with column density; in cases where source size can be estimated through other means, fixing it can better constrain column density. If unknown, it will be a free parameter.
  - Column density is initialized via Maximum Likelihood Estimation (MLE) by default, supporting both fixed and variable source sizes.
- **Exploring sample space**: Increase `nwalkers` for broader exploration, especially if the parameter space is large or multimodal. Increase `nruns` (number of MCMC steps) to improve convergence and precision. As a rule of thumb, more walkers help with complex models, while more steps refine results in well-defined spaces. Look for signs like poor convergence or incomplete exploration to decide which to adjust.

### Step 3: Running the Initial MCMC

- For your first run, set ```template_run``` to ```True``` to use hardcoded initial values specific to the template species.
- Cyanopolyynes tend to share source properties, so it is recommended to first obtain a fit for shorter linear molecules, as their transitions are more easily detectable above noise levels.
- Run the script from the main directory:
```bash
cd /path/to/Cha1-MCMC
python inference.py
```

### Step 4: Refining the Fit

- After the initial run, set `template_run` to `False` to load posteriors from the previous run as priors.
- Rerun the script to refine the fit:
```bash
python inference.py
```
- You can redo the template run if needed; this will overwrite the previous results.
- Once satisfied, set `prior_path` to the saved location to use it for future non-template runs. You can keep multiple template runs, so check that `prior_path` points to the correct chain for each case.

If you have any questions or feedback, feel free to reach out via email at [kahaan@gmail.com](mailto:kahaan@gmail.com). A BibTeX citation will be included alongside a forthcoming publication; please cite the software if you found it helpful with your work.