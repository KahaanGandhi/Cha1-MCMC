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
With the environment activated and the dependencies installed, navigate to the main directory and run the MCMC_inference.py script:
```bash
cd /path/to/Cha1-MCMC
python MCMC_inference.py
```
This will start the MCMC inference based on the parameters set in the script.






