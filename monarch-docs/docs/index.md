<img src="monarch_logo.png" alt="drawing" width="200"/>

MONARCH is an open-source Python package to rapidly model cardiac physiology and remodeling. It is developed by the BEAT Lab at UCI directed by Pim Oomen. MONARCH stands for Model for Open-source Numerical simulations of Arrhythmia, Remodeling, Cardiac mechanics, and Hypertrophy.


## Installation
### Virtual environment
It is recommended to install MONARCH in a virtual environment. To create a virtual environment using Conda, first install a distribution of [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://conda.io/miniconda.html). After installation, open a terminal window to create a new virtual environment named (for example) `monarch` that uses `Python 3.12`:
```
conda create -n monarch python=3.12
conda activate monarch
```
### Installing using pip (not yet available)

MONARCH can be installed using pip:
```
pip install monarch-beatlab
```
### Installing from source
Download the latest source code from Github. From a terminal window in the directory where you want to download and install the MONARCH source code, run:
```
git clone https://github.com/BeatLabUCI/monarch.git
```
Then build and install MONARCH from the top level of the source tree:
```
pip install .
```

### Checking your installation
To check that MONARCH has been installed correctly, open a Python interpreter (with the correct virtual environment if applicable) and run:
```
import monarch
```

## Using Monarch
We include several Jupyter notebooks to demonstrate how to use MONARCH. These notebooks can be found in the `examples` directory.