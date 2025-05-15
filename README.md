<img src="monarch_logo.png" alt="drawing" width="200"/>

MONARCH is an open-source Python package to rapidly model cardiac physiology and remodeling. It is developed by the BEAT Lab at UCI directed by Pim Oomen. MONARCH stands for Model for Open-source Numerical simulations of Arrhythmia, Remodeling, Cardiac mechanics, and Hypertrophy.


## Installation
### Virtual environment
It is recommended to install MONARCH in a virtual environment. To create a virtual environment using Conda, first install a distribution of [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://conda.io/miniconda.html). After installation, open a terminal window to create a new virtual environment named (for example) `monarch` that uses `Python 3.12`:
```
conda create -n monarch python=3.12
conda activate monarch
```
### Installing from source
Download the latest source code from Github. From a terminal window (make sure to activate your virtual environment) in the directory where you want to download and install the MONARCH source code, run:
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

## Using the Monarch GUI
The monarch GUI requires the packages "mkdocs" and "voila". Install them locally on your machine using 
the commands "pip install voila" and "pip install mkdocs". After that, for using the Monarch graphical user interface, open a terminal and cd into monarch-docs, 
then run the command "mkdocs serve". A local link should pop up in the terminal. This link is the development version of the website for Monarch's documentation.
This includes the growth and beats demo notebooks, as well as a link to the interactive graphs.
After clicking on that link, you should be led to the web browser.
Next, navigate back to your IDE and open another terminal tab and cd into monarch-docs, then docs. 
Then, run the command "voila monarch_starter_interactive.ipynb" 
Now, the GUI should be fully functional!