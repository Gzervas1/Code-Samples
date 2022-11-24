----
energiesL_0407.txt
energiesH_0407.txt
----
Lower and upper edge of energy bins, whose (logarithmic) center is given by 'data/0331/energies_0331.txt'.



-----
Edisp_0407.txt
-----
This takes into account the energy dispersion at the detector: i.e., the PDF P(E_R|E) as in Eq. (1) of draft_0329.pdf. 
This is two-dimensional array of the shape of (300,500). The first axis is for migration \mu = E_R/E, and the second axis is for true energy E. 
Note that it is normalized such that \int d\mu P(\mu|E) = 1.



-----
migration_Edisp_0407.txt
-----
One dimensional array (of length 300) for migration \mu = E_R/E.



-----
energy_Edisp_0407.txt
-----
One dimensional array (of length 500) for true gamma-ray energy E in units of TeV.



-----
Jmap_0407.txt
-----
A 2d array of J factor in units of [GeV^2 cm^-5].



-----
dNdE_Wino_3TeV_0407.txt
-----
Gamma-ray spectrum per annihilation dN_{\gamma,ann}/dE in Eq. (2) of draft_0329.pdf, in the case of Wino dark matter with the mass of 3 TeV. 
Note that you have to manually add the line component on top of this (to be discussed later). Units: TeV for E and TeV^-1 for dN_{\gamma,ann}/dE.
