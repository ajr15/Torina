# Torina
An AI app for chemistry
=======

To create a conda environment with all of the dependencies. Write the followig commands
```
conda create -n Torina python=3.7
conda install -y -n Torina -c rdkit rdkit 
conda install -y -n Torina -c openbabel openbabel
conda install -y -n Torina -c conda-forge scikit-learn
conda activate Torina
pip install tensorflow
```
## OPTIONAL PACKAGES
```
conda install -y -n Torina dask
conda install -y -n Torina -c conda-forge deepchem
```
>>>>>>> dev
