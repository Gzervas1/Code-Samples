# Code-Samples
This repository contains coding samples from a variety of projects throughout my MSc studies at the University of Amsterdam.

## Scipy Model Fitting
Fitting models on synthetic dark matter detection data, first for instrumental calibration purposes and then for parameter constraints.
Calculation of maximum likelihood estimates and resulting errors using different methods, chi-squared estimation, and visualization.
Self-contained in a jupyter-notebook along with explanations taken at each step. Data are provided in the folder.

## Scipy Statistical Tests
Data filtering and correlation checking using Pearson's-r for a sample of galaxies, based on their physical properties. 
Linear regression to quantify correlations, student's t-tests for checking statistical integrity.
Bootstrapping for reconstructing the distributions of parameters of interest for further statistical tests.
Provided in the form of a jupyter-notebook with the associated data included in the folder.

## Dark Matter Project
2-month-long collaborative coding project. Credit also to my fellow students and our co-ordinators Dr. Shin'ichiro Ando and Dr. Oscar Macias for
their feedback and encouragement. 
The project is focused on three main areas:
1) Constructing templates for each component of the Galactic center responsible for the emission of gamma rays.
2) Convolution with Cherenkov Telescope Array (CTA) instrumental properties for the creation of synthetic Dark matter data.
3) Custom chi-squared minimization code used for reconstructing the dark matter particle properties associated with gamma-ray emission.


The code written in python can be found in the files functions.py and groupX_functions.py,
with the associated descriptions therein.

The data used can be found under the folders dataA,B,C,X  with readme files providing descriptions of their meaning.

Finally, under the folders figsA,B,C,X, there are sample figures produced using the code.


For the code to run successfully, the installation of the python module HDMSpectra (https://github.com/nickrodd/HDMSpectra) 
and iminuit package (https://iminuit.readthedocs.io/en/stable/) is required.
