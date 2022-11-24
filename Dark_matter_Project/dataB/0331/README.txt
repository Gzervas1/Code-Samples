-----
detector_background_0331.txt
-----
Cosmic ray background at the detector. See file header for more info.



-----
effective_area_0331.txt
-----
Effective area of CTA. See file header for more info.



-----
energies_0331.txt
-----
Energies of each bin in units of TeV. There are 47 bins and for both "total_ics.npy" and "total_gce.npy"



-----
total_ics.npy
total_gce.npy
-----
Two background components: inverse-Compton scattering (ics) and gas-correlated emission (gce). Units are [1/TeV/cm^2/s/sr]. 
Files contain a three-dimiensional array of the size of (47,20,20). First axis of 47 components corresponds to different energy bins (as in energies_0331.txt), 
and the second and third axes correspond to Galactic longitude l and latitude b. 
Both l and b goes between -5 to 5 degrees around the Galactic center (l,b)=(0,0). Pixel size is therefore 0.5x0.5 deg^2.




-----
fermi_bubbles_min_0331.npy
fermi_bubbles_max_0331.npy
-----
The file structure and units are the same as 'total_ics.npy' and 'total_gce.npy'. These are for the 'Fermi bubbles' component. 
There are 'min' and 'max' models which will bracket the current uncertainty of this component.



-----
intensity_wino_0331.txt
-----
The file contains 20x20 two-dimensional array, whose inputs are gamma-ray intensity in units of [cm^-2 s^-1 sr^-1]. 
Here we are assuming a single energy bin around the wino mass of 3 TeV and annihilation into gamma-ray line component. 
The annihilation cross section is <\sigma v>_{line} = 2e-26 cm^3/s. These two dimensions corresponds to (l,b) where the center is the Galactic center; 
the same as other data arrays.
