from preface import *
import pandas as pd
import time
import scipy.special
from scipy import integrate, interpolate
import math
from pathlib import Path 
import inspect
from HDMSpectra import HDMSpectra 
from scipy.interpolate import interp1d
import scipy.optimize as opt
from scipy.linalg import inv
#import pygtc
from scipy.integrate import simps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
from iminuit import Minuit, describe


def groupA_data_path(dirandfile):
    src_file_path = inspect.getfile(lambda: None)
    base_path = Path(src_file_path).parent
    path = (base_path / "dataA" / dirandfile).resolve()

    return path

def groupA_r(l,psi):
    return np.sqrt(D_GC**2 + l**2 - 2*D_GC*l*np.cos(psi))

def groupA_rho_NFW(l,psi): 
    """
    Calculates NFW DM density of the milky way
    Input: line of sight [b.u.] and angle from Galactic Centre [rad]
    Output: density [b.u.]
    """
    rho0  = 0.28*GeV/cm**3
    rhos  = rho0*D_GC/rs*(D_GC/rs+1.)**2
    r = groupA_r(l,psi)
    return rhos/((r/rs)/(r/rs+1.)**2)

def groupA_rho_CORED(l,psi):
    """
    Calculates Cored DM density of the milky way
    Input: line of sight [b.u.] and angle from Galactic Centre [rad]
    Output: density [b.u.]
    """
    rc = 1.*kpc
    c = 12.2 # halo consentration parameter
    r = groupA_r(l,psi) #####
    f = np.tanh(r/rc)
    M = 1.11e12 * Msun
    gc = 1/(np.log(1+c)-c/(1+c))
    M_nfw = M*gc*(np.log(1+r/rs)-(r/rs)*(1+r/rs)**(-1))
    rho_nfw = groupA_rho_NFW(l,psi)
    return f*rho_nfw+((1-f**2)/(4*np.pi*r**2*rc))*M_nfw

def groupA_J(psi,flavour = "NFW",nPts = 1000): 
    """
    Calculates J factor (integral rho^2 dl)
    Input: angle from Galactic Centre [rad] and profile
           nPts = number of datapoints in the line of sight
    Output: J factor [b.u.]
    """
    smax = D_GC*np.cos(psi) + np.sqrt(Rvir**2 - D_GC**2*(np.sin(psi))**2)
    s_array = np.linspace(0, smax, num = nPts)
    if flavour =="NFW":
        integrand = groupA_rho_NFW(s_array,psi)**2
    elif flavour =="CORED":
        integrand = groupA_rho_CORED(s_array,psi)**2
    J = integrate.simps(integrand,x=s_array, axis = 0)
    return J

def groupA_psi(l, b):
    """
    Calculates longitude and lattiude to psi
    """
    psi = np.arccos(-np.cos(l+np.pi)*np.sin(b+np.pi/2))
    return psi

def groupA_psigrid(lmin = -5, lmax = 5, bmin = -5, bmax = 5, dl = 0.5, db = 0.5):
    """
    Make a grid
    """
    l = np.arange(lmin+dl/2., lmax, dl)
    b = np.arange(bmin+dl/2., bmax, db)
    L, B = np.meshgrid(l, b)
    psi = groupA_psi(L*deg, B*deg)
    return psi

def groupA_plot_Jline(range_start, range_finish,flavour = "NFW"): 
    """
    Calculates the line inetgral for a given range of psi [rad] and plots
    Input: lower value of psi and upper value of psi
    Output: Plot of J for a given range of psi [rad]
    """
    psi = np.linspace(range_start, range_finish, num=1000)
    J = groupA_J(psi,flavour,1000) 
    
    fig, ax = plt.subplots(1, 1, figsize = (7, 7))

    ax.plot(psi/deg, J/(GeV**2/cm**5))
    ax.set_yscale('log')
    
    ax.set_xlabel(r'$\psi$ (rad)')
    ax.set_ylabel(r'$J$ (GeV$^2$/cm$^5$)')
    
    ax.tick_params(axis='y', which='minor', labelsize=10, length = 0, width = 1)
    plt.show()

def groupA_plot_Jgrid(flavour = "NFW"):
    """
    Plots the J in a 3d plot with l on the x-axis and b on the y-axis
    """
    dl, db = 0.5, 0.5
    lmin, lmax, bmin, bmax = -5, 5, -5, 5
    psi = groupA_psigrid(lmin, lmax, bmin, bmax, dl, db)
    J = groupA_J(psi, flavour,1000)
    J = J/(GeV**2/cm**5)
    c = plt.imshow(J, interpolation ='nearest', extent = (lmin, lmax, bmin, bmax))
    cbar = plt.colorbar(c)
    plt.xlabel('l (deg)')
    plt.ylabel('b (deg)')
    cbar.set_label(r'$J$ (GeV$^2$/cm$^5$)')
    plt.show()

def groupA_dNdE(mDM):
    """
    Provides lists of dN/dE for dark matter --> (W^+ W^-) (WT), (Z Z) (ZT)--> gamma, (γ γ) --> gamma, (b,b) --> gamma,(t t) --> gamma and (h h) --> gamma .
    WT (ZT) = (WL(ZL)+WR(ZR))/2. Z boson = id 23, W boson id = 24 , γ id = 22
    Input: dark matter mass
    Output: saved lists of dN/dE for all types of decay with corresponding energies
    """
    x = np.logspace(-6.,0,440)
    E_list = x*mDM
    finalstate = 22 #photons = 22

    df_dNdE = pd.DataFrame()
    df_dNdE['E'] = E_list
    df_dNdE['WT'] = (HDMSpectra.spec(finalstate, 1924, x, mDM/GeV, groupA_data_path('HDMSpectra.hdf5'), annihilation=True)/mDM+                         HDMSpectra.spec(finalstate, 2924, x, mDM, groupA_data_path('HDMSpectra.hdf5'), annihilation=True)/mDM)/2
    df_dNdE['ZT'] = (HDMSpectra.spec(finalstate, 1923, x, mDM/GeV, groupA_data_path('HDMSpectra.hdf5'), annihilation=True)/mDM+                         HDMSpectra.spec(finalstate, 2923, x, mDM, groupA_data_path('HDMSpectra.hdf5'), annihilation=True)/mDM)/2
    df_dNdE['gamma'] = (HDMSpectra.spec(finalstate, 1922, x, mDM/GeV, groupA_data_path('HDMSpectra.hdf5'), annihilation=True)/mDM+                         HDMSpectra.spec(finalstate, 2922, x, mDM, groupA_data_path('HDMSpectra.hdf5'), annihilation=True)/mDM)/2
    df_dNdE['bb'] = (HDMSpectra.spec(finalstate, 'b', x, mDM/GeV, groupA_data_path('HDMSpectra.hdf5'), annihilation=True)/mDM)
    df_dNdE['tt'] = (HDMSpectra.spec(finalstate, 't', x, mDM/GeV, groupA_data_path('HDMSpectra.hdf5'), annihilation=True)/mDM)
    df_dNdE['hh'] = (HDMSpectra.spec(finalstate, 'h', x, mDM/GeV, groupA_data_path('HDMSpectra.hdf5'), annihilation=True)/mDM)
    return df_dNdE

def groupA_dNdE_interpolating(E, E_list, dNdE_list, m_DM):
    """
    Linear interpolation of E using the lists of E and dNdE
    Input: E_nu , dark matter mass and kind of decay (all in base units)
    Output: dN/dE [b.u.]
    """
    dNdE = np.interp(E, E_list, dNdE_list)
    #return 0 if E is outside the values of the list
    dNdE = dNdE*np.heaviside(m_DM-E,0.)
    return dNdE

def groupA_crosssections(mDM):
    """
    Get cros ssection from sigmav_wino and interpolates cross section based on mass
    Only works for masses 2500 < m < 3200 GeV !!
    """
    #sigmav = np.genfromtxt(groupA_data_path('0416/sigmav_wino.txt')) #
    sigmav = np.genfromtxt('dataX/cross.txt')
    sv_WW = np.interp(mDM, sigmav[:,0]*GeV, sigmav[:,1]*cm**3/s)
    sv_Zg = np.interp(mDM, sigmav[:,0]*GeV, sigmav[:,2]*cm**3/s)
    sv_gg = np.interp(mDM, sigmav[:,0]*GeV, sigmav[:,3]*cm**3/s)
    sv_ZZ = np.interp(mDM, sigmav[:,0]*GeV, sigmav[:,4]*cm**3/s)
    return sv_WW, sv_Zg, sv_gg, sv_ZZ

def groupA_contcomp(mDM):
    """
    Computing the continuum and line components. Both have to be "smeared" with the energy resoloution of the CTA.
    """
    sv_WW, sv_Zg, sv_gg, sv_ZZ = groupA_crosssections(mDM)

    df_dNdE = groupA_dNdE(mDM)
    energies = df_dNdE['E'].values

    contComp = (sv_WW*df_dNdE['WT'] + sv_gg*df_dNdE['gamma'] + sv_ZZ*df_dNdE['ZT']).values

    return contComp, energies

def groupA_smear1D(phi_map, energy_central, p_m, p_E, pdf):
    E_rec  = energy_central
    E_true = np.logspace(np.log10(0.01),3,1000)*TeV
    fint_Edisp = interpolate.interp1d(np.log(p_E), pdf, axis=-1, bounds_error = False, fill_value=0.)
    Edisp2 = fint_Edisp(np.log(E_true)) # migration probabilites values for E_true.
    # Edisp2[i, j] gives probability of E_true[j] migrating to \mu[i]
    Edisp3 = np.empty((np.alen(E_rec), np.alen(E_true)))
    # Edisp3 is interpolating Edisp2 to the E_rec values using Jacobian.
    # Note that Edisp2 is a function of migration, whereas Edisp3 is a function of E_rec.
    # fint_Edisp2 interpolates Edisp2 from specific values of \mu to all values of \mu.
    for i in np.arange(np.alen(E_true)): # looping over true energy values
        fint_Edisp2 = interpolate.interp1d(p_m, Edisp2[:,i], bounds_error=False, fill_value=0.)
        migration   = E_rec/E_true[i]
        Edisp3[:,i] = fint_Edisp2(migration)/E_true[i]

    fint_phi = interpolate.interp1d(np.log(energy_central), phi_map, axis=0, bounds_error = False, fill_value=0.)
    k = fint_phi(np.log(E_true))
    q = integrate.simps(k*Edisp3*E_true,x=np.log(E_true),axis=1)
    return q

def groupA_phi(energy, psi, mDM, flavour = "NFW"):
    """
    Computing gamma-ray intensity phi(E,psi) as shown in eqn (2) of the draft paper
    (needed to compute the final intensity map). 
    Input : the energy and mass of the DM particle, angle psi
    Output: phi
    """
    # Read in the inputs required for the function groupB_smear (from the shared "data"-folder)
    p_E = np.loadtxt(groupA_data_path('0407/energy_Edisp_0407.txt'))*TeV
    p_m = np.loadtxt(groupA_data_path('0407/migration_Edisp_0407.txt'))
    p = np.loadtxt(groupA_data_path('0407/Edisp_0407.txt'))
    
    # Calculate continuum component with corresponding energies and smear them
    contComp, energies = groupA_contcomp(mDM)
    contCompSMEAR = groupA_smear1D(contComp, energies, p_m, p_E, p)

    # Calculate line component with smearing
    fint_Edisp = interpolate.interp1d(np.log(p_E),p,axis=-1,bounds_error=False,fill_value=0.)
    Edisp2 = fint_Edisp(np.log(mDM))/mDM
    fint_Edisp2 = interpolate.interp1d(p_m,Edisp2,bounds_error=False,fill_value=0.)
    sv_WW, sv_Zg, sv_gg, sv_ZZ = groupA_crosssections(mDM)
    lineCompSMEAR = (sv_gg+sv_Zg/2.)*2.*fint_Edisp2(energies/mDM)

    # Calculate value of smeared continuum and line component at specific energy
    contCompSMEARvalue = groupA_dNdE_interpolating(energy, energies, contCompSMEAR, mDM)
    lineCompSMEARvalue = groupA_dNdE_interpolating(energy, energies, lineCompSMEAR, mDM)
    
    CompSMEARvalue = contCompSMEARvalue + lineCompSMEARvalue
    
    # Calculate phi
    preFactor = 1/(8*np.pi*mDM**2)
    phi = preFactor* (CompSMEARvalue.reshape(-1,1,1))*groupA_J(psi, flavour,nPts = 10000)
    
    return phi

def groupA_CMB(energy):
    """
    calculate cosmic ray bacground. 
    Input: energy
    """
    ElistL = np.loadtxt('dataA/0407/EnergiesL_0407.txt')*TeV
    ElistH = np.loadtxt('dataA/0407/EnergiesH_0407.txt')*TeV
    # get the index where the grid is
    index = np.where(energy > ElistL)[0][-1]
    # function wants exposure time to return counts, we want intensity so divide agian by the exposure time. the value is thus
    # irrelevant
    
    exposuretime = 500*hour
    cosmicraybackground = groupC_background_pixelcounts_3D(exposuretime,0.25*deg**2)[index]/exposuretime

    return cosmicraybackground

def groupA_plot_phigrid(energy = 100*GeV, mDM = 3000*GeV,flavour = "NFW"):
    dl, db = 0.5, 0.5
    lmin, lmax, bmin, bmax = -5, 5, -5, 5
    psigrid = groupA_psigrid(lmin, lmax, bmin, bmax, dl, db)

    phigrid = groupA_phi(energy, psigrid, mDM,flavour)/(GeV**2/cm**5)
    background = groupA_CMB(energy)
    c = plt.imshow(phigrid[0]+background, interpolation ='nearest', extent = (lmin, lmax, bmin, bmax))
    cbar = plt.colorbar(c)
    plt.xlabel('l (deg)')
    plt.ylabel('b (deg)')
    cbar.set_label(r'Intensity [b.u]')
    plt.show()

def groupA_N_pix(psi, mDM, flavour = "NFW",nPts = 5000):
    
    T = 500*hour
    Omega_pix = 0.25*deg**2

    energiesL = np.loadtxt(groupA_data_path('0407/energiesL_0407.txt'))*TeV
    energiesH = np.loadtxt(groupA_data_path('0407/energiesH_0407.txt'))*TeV
    
    energies = np.logspace(np.log10(energiesL),np.log10(energiesH),nPts)
    A_eff = groupC_Aeff(energies)
    energies_4d = np.expand_dims(energies,axis=(2,3))
    A_eff_4d = np.expand_dims(A_eff,axis=(2,3))
    
    energies0 = np.logspace(np.log10(energiesL[0]),np.log10(energiesH[-1]),nPts)
    phi0 = groupA_phi(energies0,psi,mDM,flavour)+1.e-100/TeV/cm**2/s/sr
    fint_lnphi = interpolate.interp1d(np.log(energies0),np.log(phi0),axis=0,
                                      bounds_error=False,fill_value=-np.inf)
    phi = np.exp(fint_lnphi(np.log(energies)))
    N_pix = T*Omega_pix*integrate.simps(A_eff_4d*phi*energies_4d,
                                        x=np.log(energies_4d),axis=0)
    return N_pix

def groupA_plot_Npixgrid(b):
    """
    Plots the J in a 3d plot with l on the x-axis and b on the y-axis
    """
    dl, db = 0.5, 0.5
    lmin, lmax, bmin, bmax = -5, 5, -5, 5
    psi = groupA_psigrid(lmin, lmax, bmin, bmax, dl, db)
    
    energiesL = np.loadtxt(groupA_data_path('0407/energiesL_0407.txt'))
    energiesH = np.loadtxt(groupA_data_path('0407/energiesH_0407.txt'))
    
    c = plt.imshow(N_pix_gridpsi[b], interpolation ='nearest', extent = (lmin, lmax, bmin, bmax))
    cbar = plt.colorbar(c)
    plt.xlabel('l (deg)')
    plt.ylabel('b (deg)')
    cbar.set_label(r'$N_{pix}$')
    
    plt.title(f'${round(energiesL[b], 3)} < E_R < {round(energiesH[b], 3)}$ TeV')
    
    plt.show()

def groupA_plot_totalN():
    energiesC = np.loadtxt(groupA_data_path('0331/energies_0331.txt'))
    plt.plot(energiesC, np.sum(N_pix_gridpsi, axis = (1,2)))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1)
    plt.xlabel('$E_R$')
    plt.ylabel('$N_{pix}$')
    plt.show()



def groupB_data_path(dirandfile):
    import inspect
    src_file_path = inspect.getfile(lambda: None)
    base_path = Path(src_file_path).parent
    path = (base_path / "dataB" / dirandfile).resolve()

    return path

def groupB_intensity_map(imap, map_number,energy_central=None):
    '''Returns an intensity map for a specific energy bin'''
    if energy_central is None:
        energy_central = np.loadtxt(groupB_data_path('0331/energies_0331.txt')) * TeV

    plt.imshow(imap[map_number]/(1/TeV/cm**2/s/sr), extent=[-5,5,-5,5])
    plt.suptitle('E={0:.3g} TeV'.format(energy_central[map_number]/TeV))
    plt.xlabel('$l(^o)$')
    plt.ylabel('b($^o$)')
    plt.colorbar()

def groupB_counts_map(intensity, energy_bins=None, e_area_2=None):
    '''Input: Intensity maps cubed, respective energy bins, CTA effective area
       Output: Count maps cubed.'''
    if energy_bins is None:
        energy_H = np.loadtxt(groupB_data_path('0407/energiesH_0407.txt')) * TeV
        energy_L = np.loadtxt(groupB_data_path('0407/energiesL_0407.txt')) * TeV
        energy_bins = energy_H - energy_L
    if e_area_2 is None:
        eff_area_rec_energy = np.loadtxt(groupB_data_path('0331/effective_area_0331.txt'), usecols=0) * TeV
        eff_area = np.loadtxt(groupB_data_path('0331/effective_area_0331.txt'), usecols=1) * m ** 2
        energy_central = np.loadtxt(groupB_data_path('0331/energies_0331.txt')) * TeV
        e_area_2 = np.interp(energy_central, eff_area_rec_energy, eff_area)

    T = 500*hour
    pixel = (0.5*deg)**2*sr
    c_map = (np.transpose(np.transpose(intensity[0:])*energy_bins*e_area_2[0:]))*T*pixel

    return c_map

def groupB_smear(phi_map, energy_central=None, p_m=None, p_E=None, pdf=None):
    
    '''Returns an intensity map taking into account the energy dispersion.'''

    if energy_central is None:
        energy_central = np.loadtxt(groupB_data_path('0331/energies_0331.txt')) * TeV
    if p_E is None:
        p_E = np.loadtxt(groupB_data_path('0407/energy_Edisp_0407.txt')) * TeV
    if p_m is None:
        p_m = np.loadtxt(groupB_data_path('0407/migration_Edisp_0407.txt'))
    if pdf is None:
        pdf = np.loadtxt(groupB_data_path('0407/Edisp_0407.txt'))

    E_rec = energy_central
    E_true = np.logspace(np.log10(0.01), 3, 1000)*TeV
    fint_Edisp = interpolate.interp1d(np.log(p_E), pdf, axis=-1, bounds_error=False, fill_value=0.)
    Edisp2 = fint_Edisp(np.log(E_true))  # migration probabilites values for E_true.
    Edisp3 = np.empty((np.alen(E_rec), np.alen(E_true)))
    '''
    Edisp2[i, j] gives probability of E_true[j] migrating to \mu[i]
    Edisp3 is interpolating Edisp2 to the E_rec values using Jacobian. 
    Note that Edisp2 is a function of migration, whereas Edisp3 is a function of E_rec.
    fint_Edisp2 interpolates Edisp2 from specific values of \mu to all values of \mu.
    '''
    for i in np.arange(np.alen(E_true)):  # looping over true energy values
        fint_Edisp2 = interpolate.interp1d(p_m, Edisp2[:, i], bounds_error=False, fill_value=0.)
        migration = E_rec/E_true[i]
        Edisp3[:, i] = fint_Edisp2(migration)/E_true[i]

    E_true_3d = E_true.reshape(-1, 1, 1)
    Edisp_4d = np.expand_dims(Edisp3, (2, 3))
    fint_phi = interpolate.interp1d(np.log(energy_central), phi_map, axis=0, bounds_error=False, fill_value=0.)
    k = fint_phi(np.log(E_true))
    q = simps(k*Edisp_4d*E_true_3d, x=np.log(E_true), axis=1)

    return q

def groupB_smear1D(phi_map, energies=None, p_m=None, p_E=None, pdf=None):

    if energies is None:
        energies = np.loadtxt(groupB_data_path('0331/energies_0331.txt')) * TeV
    if p_E is None:
        p_E = np.loadtxt(groupB_data_path('0407/energy_Edisp_0407.txt')) * TeV
    if p_m is None:
        p_m = np.loadtxt(groupB_data_path('0407/migration_Edisp_0407.txt'))
    if pdf is None:
        pdf = np.loadtxt(groupB_data_path('0407/Edisp_0407.txt'))

    E_true = np.logspace(np.log10(0.01), 3, 1000)*TeV
    fint_Edisp = interpolate.interp1d(np.log(p_E), pdf, axis=-1, bounds_error=False, fill_value=0.)
    Edisp2 = fint_Edisp(np.log(E_true))  # migration probabilites values for E_true.
    Edisp3 = np.empty((np.alen(energies), np.alen(E_true)))
    '''
    Edisp2[i, j] gives probability of E_true[j] migrating to \mu[i]
    Edisp3 is interpolating Edisp2 to the E_rec values using Jacobian. 
    Note that Edisp2 is a function of migration, whereas Edisp3 is a function of E_rec.
    fint_Edisp2 interpolates Edisp2 from specific values of \mu to all values of \mu.
    '''
    for i in np.arange(np.alen(E_true)):  # looping over true energy values
        fint_Edisp2 = interpolate.interp1d(p_m, Edisp2[:, i], bounds_error=False, fill_value=0.)
        migration = energies/E_true[i]
        Edisp3[:, i] = fint_Edisp2(migration)/E_true[i]
   
    fint_phi = interpolate.interp1d(np.log(energies), phi_map, axis=0, bounds_error=False, fill_value=0.)
    k = fint_phi(np.log(E_true))
    q = simps(k*Edisp3*E_true, x=np.log(E_true), axis=1)
    return q

def groupB_create_Edisp3(energies=None, p_m=None, p_E=None, pdf=None):

    if energies is None:
        energies = np.loadtxt(groupB_data_path('0331/energies_0331.txt')) * TeV
    if p_E is None:
        p_E = np.loadtxt(groupB_data_path('0407/energy_Edisp_0407.txt')) * TeV
    if p_m is None:
        p_m = np.loadtxt(groupB_data_path('0407/migration_Edisp_0407.txt'))
    if pdf is None:
        pdf = np.loadtxt(groupB_data_path('0407/Edisp_0407.txt'))

    E_true = np.logspace(np.log10(0.01), 3, 1000)*TeV
    fint_Edisp = interpolate.interp1d(np.log(p_E), pdf, axis=-1, bounds_error=False, fill_value=0.)
    Edisp2 = fint_Edisp(np.log(E_true)) # migration probabilites values for E_true.
    # Edisp2[i, j] gives probability of E_true[j] migrating to \mu[i]
    Edisp3 = np.empty((np.alen(energies), np.alen(E_true)))
    # Edisp3 is interpolating Edisp2 to the E_rec values using Jacobian.
    # Note that Edisp2 is a function of migration, whereas Edisp3 is a function of E_rec.
    # fint_Edisp2 interpolates Edisp2 from specific values of \mu to all values of \mu.
    for i in np.arange(np.alen(E_true)):  # looping over true energy values
        fint_Edisp2 = interpolate.interp1d(p_m, Edisp2[:, i], bounds_error=False, fill_value=0.)
        migration = energies/E_true[i]
        Edisp3[:, i] = fint_Edisp2(migration)/E_true[i]
    return Edisp3

def groupB_N_pix(psi, mDM, flavour="NFW", nPts=1000):
    """
    Borrowed from groupA for a moment
    """

    T = 500 * hour
    Omega_pix = 0.5 * deg ** 2

    energiesH = np.loadtxt(groupB_data_path('0407/energiesH_0407.txt')) * TeV
    energiesL = np.loadtxt(groupB_data_path('0407/energiesL_0407.txt')) * TeV

    bins = len(energiesL)
    N_pix = ([])
    for b in range(bins):
        energies = np.linspace(energiesL[b], energiesH[b], nPts)
        A_eff = groupC_Aeff(energies)
        phi = groupA_phi(energies, psi, mDM)
        N_pix.append(T * Omega_pix * simps((A_eff * phi.T).T, x=energies, axis=0))
    N_pix = np.array(N_pix)
    return N_pix

def groupB_total_counts_map(DM_mass, flavour, background):
    '''Inputs: Wino mass in GeV, DM profile, background='min' or 'max' (for Fermi bubbles)
       Output: Count map cubed with DM signal (continuous+line components) + Background (Fermi bubbles, GCE, IC, CR)'''

    print('Running groupB_total_counts_map')

    energy_central = np.loadtxt(groupB_data_path('0331/energies_0331.txt')) * TeV
    fermi_bub_max = np.load(groupB_data_path('0331/fermi_bubbles_max_0331.npy')) / TeV / cm ** 2 / s / sr
    fermi_bub_min = np.load(groupB_data_path('0331/fermi_bubbles_min_0331.npy')) / TeV / cm ** 2 / s / sr
    gce = np.load(groupB_data_path('0331/total_gce_0331.npy')) / TeV / cm ** 2 / s / sr
    ics = np.load(groupB_data_path('0331/total_ics_0331.npy')) / TeV / cm ** 2 / s / sr
    p_E = np.loadtxt(groupB_data_path('0407/energy_Edisp_0407.txt')) * TeV
    p_m = np.loadtxt(groupB_data_path('0407/migration_Edisp_0407.txt'))
    p = np.loadtxt(groupB_data_path('0407/Edisp_0407.txt'))

    print('Compute CR Counts')
    cr_counts = groupC_background_pixelcounts_3D(500 * hour, (0.5 * deg) ** 2)
    print(cr_counts.shape)

    print('Compute BKG')
    if background == "min":
        bkg_min_map = groupB_counts_map(groupB_smear(fermi_bub_min + gce + ics, energy_central, p_m, p_E, p))
        #bkg_min_map = groupB_smear(groupB_counts_map(fermi_bub_min+gce+ics))
        bkg = bkg_min_map + cr_counts
    elif background == "max":
        bkg_max_map = groupB_counts_map(groupB_smear(fermi_bub_max + gce + ics, energy_central, p_m, p_E, p))
        #bkg_max_map = groupB_smear(groupB_counts_map(fermi_bub_max+gce+ics))
        bkg = bkg_max_map + cr_counts
    else:
        bkg = np.zeros((47, 20, 20))

    print(bkg.shape)

    print('Compute DM (Mass %i)' % DM_mass)
    counts_map_DM = groupA_N_pix(groupA_psigrid(), DM_mass, flavour)
    print(counts_map_DM.shape)

    final_counts = bkg + counts_map_DM

    return final_counts

########################
#   PLOTS FUNCTIONS    #
########################

def groupB_count_plot(imap, energy_central=None):
    '''Input: intensity map cubed, energy bins (centers)
       Output: plot showing total counts per energy bin.'''
    if energy_central is None:
        energy_central = np.loadtxt(groupB_data_path('0331/energies_0331.txt')) * TeV

    counts_map = groupB_counts_map(imap)
    counts = np.sum(counts_map, axis=(1, 2))
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel('Energy (TeV)')
    plt.ylabel('Counts')
    plt.ylim((1, 1e7))
    plt.xlim((1e-2, 4*1e2))
    plt.plot(energy_central/TeV, counts)

def groupB_plot_effective_area(energy_central=None, rec_energy=None, eff_area=None, savename=None):

    if energy_central is None:
        energy_central = np.loadtxt(groupB_data_path('0331/energies_0331.txt')) * TeV
    if rec_energy is None:
        rec_energy = np.loadtxt(groupB_data_path('0331/effective_area_0331.txt'), usecols=0) * TeV
    if eff_area is None:
        eff_area = np.loadtxt(groupB_data_path('0331/effective_area_0331.txt'), usecols=1) * m ** 2

    plt.figure(0)
    plt.tight_layout()
    plt.scatter(rec_energy/TeV, eff_area/m**2)
    e_area_2 = np.interp(energy_central,rec_energy, eff_area)
    plt.plot(energy_central/TeV, e_area_2/m**2,color='red')

    plt.xlabel('Energy (TeV)')
    plt.ylabel('Effective Area (m)$^{2}$')
    plt.suptitle('Effective Area(E)', fontsize=20)
    plt.legend(["Data","Interpolation"])

    plt.figure(1)
    plt.tight_layout()
    plt.scatter(rec_energy/TeV, eff_area/m**2)
    plt.plot(energy_central/TeV, e_area_2/m**2,color='red')

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel('Energy (TeV)')
    plt.ylabel('Effective Area (m)$^{2}$')
    plt.suptitle('Effective Area(E)', fontsize=20)
    plt.legend(["Data", "Interpolation"])

    if savename:
        plt.savefig(savename)

def groupB_plot_six_maps(_bkg, title='Title', _energy_central=None, which=None, savename=None):
    #Intensity map of the background in different energy areas (different component dominates in different energies).
    if which is None:
        which = [5, 12, 19, 26, 31, 38]
    if _energy_central is None:
        _energy_central = np.loadtxt(groupB_data_path('0331/energies_0331.txt')) * TeV

    fig, axs = plt.subplots(2, 3, figsize=(14, 10))
    axs = axs.ravel()
    fig.suptitle(title, fontsize=25)
    for i, counter in enumerate(which):
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        pcm = axs[i].imshow(_bkg[counter]/(1/TeV/cm**2/s/sr), extent=[-5, 5, -5, 5])
        axs[i].set_xlabel('$l(^o)$')
        axs[i].set_ylabel('b($^o$)')
        axs[i].set_title('E={0:.3g} TeV'.format(_energy_central[counter]/TeV))
        fig.colorbar(pcm, cax=cax, shrink=0.9)
    plt.tight_layout()
    if savename:
        fig.savefig(savename)
    else:
        plt.show()


# Plot rec energy for 40 values of true energy
def groupB_plot_rec_per_true(Edisp3=None, energy=None, savename=None):
    # PDFs of true energy values to be measured with one of the possible recunstracted energy values.
    if Edisp3 is None:
        Edisp3 = groupB_create_Edisp3()
    if energy is None:
        energy = np.loadtxt(groupB_data_path('0331/energies_0331.txt')) * TeV

    fig, axs = plt.subplots(nrows=4, ncols=10, sharex='col', sharey='row', figsize=(30, 15))
    axs = axs.ravel()
    fig.suptitle('Reconstructed Energy per True Energy', fontsize=20)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    E_rec = energy
    E_true = np.logspace(np.log10(0.01), 3, 1000) * TeV
    a = np.sum(Edisp3, axis=0)
    # last 95 true energy values have 0 probabillity.
    for i in range(1000):
        if a[i] != 0:
            Edisp3[:, i] = Edisp3[:, i] / a[i]

    k = 0
    for i in range(0, 900, 20):
        axs[k].plot(E_rec / TeV, Edisp3[:, i])
        axs[k].set_xscale('log')
        axs[k].annotate('%.2f TeV' % (E_true[i]/TeV), xy=(0.01, 0.92), xycoords='axes fraction')
        axs[k].axvline(x=E_true[i]/TeV, color='red')

        #axs[k].xlabel('E_reconstracted (TeV)')
        #axs[k].ylabel('Probabillity')
        #axs[k].suptitle('E_true={0:.3g} TeV'.format(E_true[i + j] / TeV))
        k += 1
        if k>39:
            break
    plt.xlabel("Reconstructed Energy")
    if savename:
        fig.savefig(savename)
    else:
        plt.show()


# Plot true energy for the 47 values of rec energy
def groupB_plot_true_per_rec(Edisp3=None, energy=None, savename=None):
    # PDFs of true energy values to be measured with one of the possible recunstracted energy values.
    if Edisp3 is None:
        Edisp3 = groupB_create_Edisp3()
    if energy is None:
        energy = np.loadtxt(groupB_data_path('0331/energies_0331.txt')) * TeV

    fig, axs = plt.subplots(nrows=5, ncols=10, sharex='col', sharey='row', figsize=(30, 20))
    axs = axs.ravel()
    fig.suptitle('True Energy vper Reconstructed Energy', fontsize=20)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # fig.text(0.5, 0.04, 'common X', ha='center')
    # fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')

    E_rec = energy
    E_true = np.logspace(np.log10(0.01), 3, 1000) * TeV

    a = np.sum(Edisp3, axis=1)
    a[0] = 1
    a[1] = 1
    a[46] = 1

    k = 0
    for i in range(47):
        axs[k].plot(E_true / TeV, Edisp3[i, :] / a[i])
        #axs[k].xlim(1e-2, 5e2)
        axs[k].axis(xmin=1e-2, xmax=5e2)
        axs[k].set_xscale('log')
        axs[k].annotate('%.2f TeV' % (E_rec[i]/TeV), xy=(0.01, 0.92), xycoords='axes fraction')
        axs[k].axvline(x=E_rec[i]/TeV, color='red', linewidth=0.4)

        # plt.xscale("log")
        # plt.figure(i)
        # plt.xlabel('E_True (TeV)')
        # plt.ylabel('Probabillity')
        # plt.suptitle('E_rec={0:.3g} TeV'.format(E_rec[i] / TeV))
        k += 1
    plt.xlabel("True Energy")
    if savename:
        fig.savefig(savename)
    else:
        plt.show()


# Plot smeared counts per energy per bkg component
def groupB_plot_smear_count_vs_energy(bkgs=None, energies=None, p_m=None, p_E=None, pdf=None, savename=None):

    if bkgs is None:
        fermi_bub_max = np.load(groupB_data_path('0331/fermi_bubbles_max_0331.npy')) / TeV / cm ** 2 / s / sr
        fermi_bub_min = np.load(groupB_data_path('0331/fermi_bubbles_min_0331.npy')) / TeV / cm ** 2 / s / sr
        gce = np.load(groupB_data_path('0331/total_gce_0331.npy')) / TeV / cm ** 2 / s / sr
        ics = np.load(groupB_data_path('0331/total_ics_0331.npy')) / TeV / cm ** 2 / s / sr
        bkgs = [[fermi_bub_max, fermi_bub_min, gce, ics],
                ["FB MAX", "FB MIN", "GCE", "ICS"]]

    if energies is None:
        energies = np.loadtxt(groupB_data_path('0331/energies_0331.txt')) * TeV
    if p_E is None:
        p_E = np.loadtxt(groupB_data_path('0407/energy_Edisp_0407.txt')) * TeV
    if p_m is None:
        p_m = np.loadtxt(groupB_data_path('0407/migration_Edisp_0407.txt'))
    if pdf is None:
        pdf = np.loadtxt(groupB_data_path('0407/Edisp_0407.txt'))

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.title('Background componets:counts per energy', fontsize=25)
    for i, ibkg in enumerate(bkgs[0]):
        color = next(ax._get_lines.prop_cycler)['color']
        groupB_count_plot(ibkg)
        ax.lines[-1].set_color(color)
        ax.lines[-1].set_linestyle('--')
        groupB_count_plot(groupB_smear(ibkg, energies, p_m, p_E, pdf))
        ax.lines[-1].set_color(color)
        ax.lines[-1].set_label(bkgs[1][i])

    plt.rc('legend', fontsize=10)
    plt.legend()
    if savename:
        plt.savefig(savename)
    else:
        plt.show()


def plot_hist_etrue(N, h):

    energy_central = np.loadtxt(groupB_data_path('0331/energies_0331.txt')) * TeV
    energy_L = np.loadtxt(groupB_data_path('0407/energiesL_0407.txt')) * TeV

    E_rec = energy_central
    E_true = np.logspace(np.log10(0.01), 3, 1000)*TeV
    Edisp3 = np.empty((np.alen(E_rec), np.alen(E_true)))

    a = np.sum(Edisp3, axis=0)
    for i in range(1000):
        if a[i] != 0:
            Edisp3[:, i] = Edisp3[:, i] / a[i]
    u = np.zeros(N)
    n = 0
    o = 0
    a = (Edisp3[:, h] * N).astype(int)

    for k in a:
        o += 1
        for j in range(k):
            u[j + n] = energy_central[o] / TeV
        n += k
    w = np.ones(N) * (1 / N)
    plt.figure(figsize=(12, 10))
    plt.hist(u, bins=energy_L / TeV, weights=w)  # ,energy_L/TeV)
    plt.xscale("log")
    # plt.figure(i)

    plt.xlabel('E_reconstracted (TeV)')
    plt.ylabel('Probabillity')
    plt.suptitle('E_true={0:.3g} TeV'.format(E_true[h] / TeV))



    
#### GROUP C

def groupC_data_path(dirandfile):
    base_path = Path(__file__).parent
    path = (base_path / "../GroupX/dataC" / dirandfile).resolve()
    return path

def groupC_Aeff(energy):
    """
    Input: energy [GeV]
    Output: effective area [GeV]
    """
    effective_areaTXT = np.loadtxt(groupC_data_path('effective_area_0331.txt'))

    #effective_areaTXT = np.loadtxt('dataC/effective_area_0331.txt', dtype='float', delimiter=" ")
    reconstructed_energy_EffArea = effective_areaTXT[:,0]*TeV
    effective_area = effective_areaTXT[:,1]*m**2
    f =interp1d(np.log(reconstructed_energy_EffArea),np.log(effective_area), kind='cubic',bounds_error=False,fill_value=-np.inf)
    
    
    return np.exp(f(np.log(energy)))



def groupC_background_bincounts_3D(Emin_array,Emax_array,Ngrid=1000):
    """
    Function calculating the background intensity counts per bin
    Input: Lists with minimum and maximum values of energy bins, possibly value of N grid
    Output: Background intensity per bin (array)  
    """
    
#     detector_background=np.loadtxt('dataC/detector_background_0331.txt')
    detector_background = np.loadtxt(groupC_data_path('detector_background_0331.txt'))


    reconstructed_energy=detector_background[:,0]*TeV  #in TeV

    flux=detector_background[:,1]*1./TeV/s/sr   #in /TeV/s/sr
    
    f =interp1d(np.log(reconstructed_energy),np.log(flux), kind='linear',bounds_error=False,fill_value = -np.inf) 
    
    energy_range=np.logspace(np.log10(Emin_array),np.log10(Emax_array),num=Ngrid)        #array corresponding to energy values along the bin 
    counts_per_energy=np.exp(f(np.log(energy_range)))  #counts corresponding to each energy value 

    bincounts=integrate.simps(counts_per_energy, x=energy_range,axis=0)   #integrated counts over the whole array(bin)
    
    return bincounts



def groupC_background_pixelcounts_3D(exposure_time,pixel_size):
    """
    Function that counts the number of background counts
    Input: exposure time, pixel size
    Output: Background counts per energy bin per pixel
    Format: energy bins x 20 x 20
    """
    
#     EnergyLow = np.loadtxt('dataC/EnergiesL_0407.txt')*TeV
#     EnergyHigh = np.loadtxt('dataC/EnergiesH_0407.txt')*TeV

    energy_H = np.loadtxt(groupC_data_path('energiesH_0407.txt')) * TeV
    energy_L = np.loadtxt(groupC_data_path('energiesL_0407.txt')) * TeV

    background = groupC_background_bincounts_3D(energy_L,energy_H)
    count = background*exposure_time*pixel_size
    
    ones = np.linspace(1,1,20).reshape(1,20,1)
    ones2 = ones.reshape(1,1,20)
    count = count.reshape(-1,1,1)
    counts = count*ones*ones2

    return counts

def groupC_cross_sections(winomass,winotype):
    """
    Function that calculates the cross-section 
    Input: Dark matter mass and wino type ('WT', 'ZT' or 'gamma')
    Output: Cross-section
    Needs adjusting to also accept bb, hh, tt!!!
    """
    
    #data = np.loadtxt(groupC_data_path('sigmav_wino.txt'))
    data = np.loadtxt('dataX/cross.txt')
    df_sigma = pd.DataFrame()
    
    df_sigma['mass'] = data[:,0]*GeV
    df_sigma['WT'] = data[:,1]*cm**3/s
    df_sigma['ZT'] = data[:,2]*cm**3/s
    df_sigma['gamma'] = data[:,3]*cm**3/s
    
    Mass_list = df_sigma['mass']
    Sigma_list = df_sigma[winotype]
    
    f =interp1d(np.log(Mass_list),np.log(Sigma_list), kind='linear',bounds_error=False,fill_value=-np.inf)
    g = interp1d(Mass_list,Sigma_list,kind='linear',bounds_error=False,fill_value=0)
    
    return g(winomass)


def groupC_wino_intensity_3D(Emin_array,Emax_array,winotype,Ngrid = 1000):
    """
    Function that calculates the wino intensity map
    Input: Lists with minimum and maximum values of energy bins, possibly value of N grid
    Output: 20x20 array of wino intensity per bin
    """
    
    Jmap=np.txt(groupC_data_path('Jmap_0407.txt'))*GeV**2/cm**5
    data_dnde=groupA_dNdE(3*TeV)
    E = data_dnde['E']*TeV
    dNdE = (data_dnde[winotype]+1.*10**(-100))/TeV
    f =interp1d(np.log(E),np.log(dNdE),kind='linear', bounds_error=False,fill_value=-np.inf)

    energy_range=np.logspace(np.log10(Emin_array),np.log10(Emax_array),num=Ngrid)
    g = groupC_Aeff(energy_range)
    h = g*np.exp(f(np.log(energy_range)))
    h_per_bin = integrate.simps(h,x=energy_range,axis=0)
    h_per_bin = h_per_bin.reshape(-1,1,1)
    
    cross_section = groupC_cross_sections(3*TeV,winotype)
    constant=cross_section/(8*np.pi*((3*TeV)**2))
    dnde_per_bin=h_per_bin*Jmap*constant 
    
    return dnde_per_bin

def groupC_loaddata(mass, Jmap, bin_by_bin=False, m_ref=TeV, sv_ref=1.e-25*cm**3/s, N_ref=1., fb='min'):
    """
    Function that loads the different signal maps for a specific mass
    Input: Dark matter mass
    Output: Masked signal count maps
    Needs adjusting to also compute bb, hh, tt!!!
    """
    exposure_time=500*hour
    pixel_size=0.25*deg**2    
    
    b_i = groupC_background_pixelcounts_3D(exposure_time,pixel_size)
    gce_1_i = groupB_counts_map(groupB_smear(groupC_gce('1')))
    gce_2_i = groupB_counts_map(groupB_smear(groupC_gce('2')))
    gce_3_i = groupB_counts_map(groupB_smear(groupC_gce('3')))
    gce_4_i = groupB_counts_map(groupB_smear(groupC_gce('4')))
    #ic_i = np.load(groupC_data_path('ics_c_map_new.npy'))
    #fb_i = np.load(groupC_data_path('fermi_c_map_min_new.npy'))
    ic_i = groupB_counts_map(groupB_smear(np.load('dataB/0331/total_ics_0331.npy')*1./TeV/cm**2/s/sr))
    if fb=='min':
        fb_i = groupB_counts_map(groupB_smear(np.load('dataB/0331/fermi_bubbles_min_0331.npy')*1./TeV/cm**2/s/sr))
    elif fb=='max':
        fb_i = groupB_counts_map(groupB_smear(np.load('dataB/0331/fermi_bubbles_max_0331.npy')*1./TeV/cm**2/s/sr))



    if bin_by_bin==False:
        sv_WW,sv_Zg,sv_gg,sv_ZZ = groupA_crosssections(mass)
        sv_ref2 = sv_ref*(sv_gg/(sv_gg+sv_Zg/2.))
        s_ph_i = groupC_groupA_N_pix(mass,exposure_time,pixel_size,'gamma',Jmap,sv_ref=sv_ref2,sv_line_ref=sv_ref)
        s_WT_i = groupC_groupA_N_pix(mass,exposure_time,pixel_size,'WT',Jmap,sv_ref=sv_ref)
        s_ZT_i = groupC_groupA_N_pix(mass,exposure_time,pixel_size,'ZT',Jmap,sv_ref=sv_ref)
        s_bb_i = groupC_groupA_N_pix(mass,exposure_time,pixel_size,'bb',Jmap,sv_ref=sv_ref)
        s_hh_i = groupC_groupA_N_pix(mass,exposure_time,pixel_size,'hh',Jmap,sv_ref=sv_ref)
        s_tt_i = groupC_groupA_N_pix(mass,exposure_time,pixel_size,'tt',Jmap,sv_ref=sv_ref)
        parameters = np.array([s_ph_i,s_WT_i,s_ZT_i,s_bb_i,s_hh_i,s_tt_i,b_i,gce_1_i,gce_2_i,gce_3_i,gce_4_i,ic_i,fb_i],dtype='float64')
    #else:
    #    sv_WW,sv_Zg,sv_gg,sv_ZZ = groupA_crosssections(mass)
    #    sv_line = sv_gg+sv_Zg/2.
    #    s_ph_i = groupC_groupA_N_pix(mass,exposure_time,pixel_size,'gamma',Jmap,nPts=1000,sv_ref=sv_gg,sv_line_ref=sv_line)
    #    s_WT_i = groupC_groupA_N_pix(mass,exposure_time,pixel_size,'WT',Jmap,nPts=1000,sv_ref=sv_WW)
    #    s_ZT_i = groupC_groupA_N_pix(mass,exposure_time,pixel_size,'ZT',Jmap,nPts=1000,sv_ref=sv_ZZ)
    #    parameters = np.array([s_ph_i+s_WT_i+s_ZT_i,b_i,gce_1_i,gce_2_i,gce_3_i,gce_4_i,ic_i,fb_i])
    else:
        phi_i = sv_ref/(8.*np.pi*m_ref**2)*N_ref*Jmap
        energy_central = np.loadtxt('dataB/0331/energies_0331.txt')*TeV
        s_i = phi_i*(groupC_Aeff(energy_central).reshape(-1,1,1))*exposure_time*pixel_size
        parameters = np.array([s_i,b_i,gce_1_i,gce_2_i,gce_3_i,gce_4_i,ic_i,fb_i],dtype='float64')
        
    maskedparameters = groupC_masking(parameters)
    
    #parameters = [s_ph_i,s_WT_i,s_ZT_i,b_i,gce_1_i,gce_2_i,gce_3_i,gce_4_i,ic_i,fb_i]
    
    return maskedparameters



def groupC_groupA_phi(energy, mDM, winotype, Jmap, flavour="NFW",
                      sv_ref=1.e-25*cm**3/s, sv_line_ref=1.e-25*cm**3/s):
    """
    Computing gamma-ray intensity phi(E,psi) as shown in eqn (2) of the draft paper
    (needed to compute the final intensity map). 
    Input : the energy and mass of the DM particle, angle psi
    Output: phi
    """

    # Read in the inputs required for the function groupB_smear (from the shared "data"-folder)
    p_E = np.loadtxt(groupA_data_path('0407/energy_Edisp_0407.txt'))*TeV
    p_m = np.loadtxt(groupA_data_path('0407/migration_Edisp_0407.txt'))
    p = np.loadtxt(groupA_data_path('0407/Edisp_0407.txt'))
    
    # Calculate continuum component with corresponding energies and smear them
    f_dNdE = groupA_dNdE(mDM)
    energies = f_dNdE['E']
    contComp = f_dNdE[winotype]
    contCompSMEAR = sv_ref*groupA_smear1D(contComp, energies, p_m, p_E, p)
    contCompSMEARvalue = groupA_dNdE_interpolating(energy, energies, contCompSMEAR, mDM)
    CompSMEARvalue = contCompSMEARvalue

    # Calculate line component with smearing
    if winotype=='gamma':
        fint_Edisp = interpolate.interp1d(np.log(p_E),p,axis=-1,bounds_error=False,fill_value=0.)
        Edisp2 = fint_Edisp(np.log(mDM))/mDM
        fint_Edisp2 = interpolate.interp1d(p_m,Edisp2,bounds_error=False,fill_value=0.)
        lineCompSMEAR = sv_line_ref*2.*fint_Edisp2(energies/mDM)
        lineCompSMEARvalue = groupA_dNdE_interpolating(energy, energies, lineCompSMEAR, mDM)
        CompSMEARvalue = contCompSMEARvalue + lineCompSMEARvalue

    # Calculate phi
    preFactor = 1/(8*np.pi*mDM**2)
    #phi = preFactor* (CompSMEARvalue.reshape(-1,1,1))*groupA_J(psi, flavour,nPts = 2000)
    phi = preFactor* (CompSMEARvalue.reshape(-1,1,1))*Jmap
    
    return phi



def groupC_groupA_N_pix(mDM,T,Omega_pix,winotype,Jmap,flavour="NFW",nPts=5000,
                        sv_ref=1.e-25*cm**3/s,sv_line_ref=1.e-25*cm**3/s):
    
    energy_L = np.loadtxt(groupA_data_path('0407/energiesL_0407.txt'))*TeV
    energy_H = np.loadtxt(groupA_data_path('0407/energiesH_0407.txt'))*TeV
    
    energies = np.logspace(np.log10(energy_L),np.log10(energy_H),nPts)
    A_eff = groupC_Aeff(energies)
    energies_4d = np.expand_dims(energies,axis=(2,3))
    A_eff_4d = np.expand_dims(A_eff,axis=(2,3))
    
    energies0 = np.logspace(np.log10(energy_L[0]),np.log10(energy_H[-1]),nPts)
    phi0 = groupC_groupA_phi(energies0,mDM,winotype,Jmap,flavour,sv_ref,sv_line_ref)+1.e-100/TeV/cm**2/s/sr
    fint_lnphi = interpolate.interp1d(np.log(energies0),np.log(phi0),axis=0,
                                      bounds_error=False,fill_value=-np.inf)
    phi = np.exp(fint_lnphi(np.log(energies)))
    N_pix = T*Omega_pix*integrate.simps(A_eff_4d*phi*energies_4d,
                                        x=np.log(energies_4d),axis=0)
    return N_pix



def GROUPB_counts_map(intensity):
    energy_H = np.loadtxt(groupC_data_path('energiesH_0407.txt'))*TeV
    energy_L = np.loadtxt(groupC_data_path('energiesL_0407.txt'))*TeV
    energy_bins=energy_H-energy_L
    
    eff_area_rec_energy = np.loadtxt(groupC_data_path('effective_area_0331.txt'), usecols=0) * TeV
    eff_area = np.loadtxt(groupC_data_path('effective_area_0331.txt'), usecols=1) * m ** 2
    energy_central = np.loadtxt(groupC_data_path('energies_0331.txt')) * TeV
    e_area_2 = np.interp(energy_central, eff_area_rec_energy, eff_area)
    
        
    T=500*hour             
    pixel=(0.5*deg)**2      
    c_map = (np.transpose(np.transpose(intensity[0:])*energy_bins*e_area_2[0:]))*T*pixel
    return c_map


def groupC_mockdata_10D(s_ph_i,s_WT_i,s_ZT_i,b_i,gce_1_i,gce_2_i,gce_3_i,gce_4_i,ic_i,fb_i):
    """
    Function that generates mock data
    Input: Signal contributions
    Output: randomized mock data based on the sum of all 10 parameters
    Format: energy bins x 20 x 20
    """
    lam = s_ph_i + s_WT_i + s_ZT_i + b_i + gce_1_i+ gce_2_i+ gce_3_i+ gce_4_i + ic_i + fb_i
    mock = np.random.poisson(lam)
    
    return(mock)

def groupC_mockdata(modelparameters):
    """
    Temporary function: generates mock data for specific modelparameters 
    Input: mass
    Output: masked mock data 
    Format: 47x20x20
    """
    
    s_ph_i,s_WT_i,s_ZT_i,b_i,gce_1_i,gce_2_i,gce_3_i,gce_4_i,ic_i,fb_i=modelparameters
    
    mockdata = groupC_mockdata_10D(s_ph_i,s_WT_i,s_ZT_i,b_i,gce_1_i,gce_2_i,gce_3_i,gce_4_i,ic_i,fb_i)
    maskedmock = groupC_masking(mockdata)
    
    return maskedmock

def groupC_modelparameters(mass,Jmap):
    """
    Temporary function, call only once
    """
    
    exposure_time = 500*hour
    pixel_size = 0.25*deg**2
    
    sv_WW, sv_Zg, sv_gg, sv_ZZ = groupA_crosssections(mass)
    sv_line = sv_gg+sv_Zg/2.
    
    s_ph_i = groupC_groupA_N_pix(mass,exposure_time,pixel_size,'gamma',Jmap,nPts=1000,sv_ref=sv_gg,sv_line_ref=sv_line)
    s_WT_i = groupC_groupA_N_pix(mass,exposure_time,pixel_size,'WT',Jmap,nPts=1000,sv_ref=sv_WW)
    s_ZT_i = groupC_groupA_N_pix(mass,exposure_time,pixel_size,'ZT',Jmap,nPts=1000,sv_ref=sv_ZZ)
    
    #s_ph_i = groupC_winodata_3D(mass,exposure_time,pixel_size,'gamma')
    #s_WT_i = groupC_winodata_3D(mass,exposure_time,pixel_size,'WT')
    #s_ZT_i = groupC_winodata_3D(mass,exposure_time,pixel_size,'ZT')
    
    b_i = groupC_background_pixelcounts_3D(exposure_time,pixel_size)
    gce_1_i = groupB_smear(groupB_counts_map(groupC_gce('1')))
    gce_2_i = groupB_smear(groupB_counts_map(groupC_gce('2')))
    gce_3_i = groupB_smear(groupB_counts_map(groupC_gce('3')))
    gce_4_i = groupB_smear(groupB_counts_map(groupC_gce('4')))
    ic_i = np.load(groupC_data_path('ics_c_map_new.npy'))
    fb_i = np.load(groupC_data_path('fermi_c_map_min_new.npy'))
    
    return np.array([s_ph_i,s_WT_i,s_ZT_i,b_i,gce_1_i,gce_2_i,gce_3_i,gce_4_i,ic_i,fb_i])

def groupC_analytical(mockdata,data):
    """
    Function that analytically solves for the best fit values of normalization for dark matter signals: 
    photon, WT and ZT, the background, and contributions of: gas-correlated emmision {1,2,3,4}, inverse compton and Fermi Bubble
    Output: Array with best fit values for N_s_ph, N_s_WT, N_s_ZT, N_b, N_gce{1,2,3,4} , N_ic and N_fb 
    """
    energybins = mockdata.shape[0]
    
    vector = np.array(data).reshape(-1,1,energybins,20,20)
    transpose = vector.reshape(1,-1,energybins,20,20)
    
    matrix = vector*transpose #shape = parameters x parameters x 47 x 20 x 20
    matrix = np.array(np.sum(np.where(mockdata!=0,matrix/(mockdata+1.e-100),0),axis=(2,3,4))) #shape parameters x parameters 
    
    inverse = inv(matrix)
    columnvector = np.sum(vector,axis = (2,3,4))
    result = np.array(inverse.dot(columnvector))
    
    return result

def groupC_numerical(mockdata, signalmaps):
    
    def groupC_chi_squared2(c1,c2,c3,c4,c5,c6,c7,c8):
        """
        Function that calculates chi squared for a set of best fit parameters
        Input: Data set, theoretical signal maps and best fit normalizations
        Output: Chi-squared
        """

        s_DM   = signalmaps[0,:,:]
        b_i    = signalmaps[1,:,:]
        gce1_i = signalmaps[2,:,:]
        gce2_i = signalmaps[3,:,:]
        gce3_i = signalmaps[4,:,:]
        gce4_i = signalmaps[5,:,:]
        ic_i   = signalmaps[6,:,:]
        fb_i   = signalmaps[7,:,:]

        numerator_sqrt_i = mockdata-c1*s_DM-c2*b_i-c3*gce1_i-c4*gce2_i-c5*gce3_i-c6*gce4_i-c7*ic_i-c8*fb_i
        numerator_i = numerator_sqrt_i**2
        chi_squared = np.sum(np.where(mockdata!=0,numerator_i/(mockdata+1.e-100),0),axis=(-1,-2))

        return chi_squared
    
    minima = Minuit(groupC_chi_squared2,c1=1.,c2=1.,c3=1.,c4=1.,c5=1.,c6=1.,c7=1.,c8=1.) # starting values for (a,b)=(1,1)
    minima.errordef = 1 # 0.5 for loglikelihood and change to 1 for least squares!!!
    minima.limits = [(0,1e5),(0,100),(0,100),(0,100),(0,100),(0,100),(0,100),(0,100)]#limit_a = (0.,100.) and limit_b = (0.1,10.)
    minima.fixed['c1'] = False # vary the parameter c1
    minima.fixed['c2'] = False # vary the parameter c2
    minima.fixed['c3'] = False # vary the parameter c3
    minima.fixed['c4'] = False # vary the parameter c4
    minima.fixed['c5'] = False # vary the parameter c5
    minima.fixed['c6'] = False # vary the parameter c6
    minima.fixed['c7'] = False # vary the parameter c7
    minima.fixed['c8'] = False # vary the parameter c8
    minima.params

    migrad = minima.migrad()
    #minos = minima.minos()
    
    return np.array(minima.values)


def groupC_chi_squared(mockdata,signalmaps,normalizations):
    """
    Function that calculates chi squared for a set of best fit parameters
    Input: Data set, theoretical signal maps and best fit normalizations
    Output: Chi-squared
    """
    
    numerator_sqrt_i = mockdata - sum(signalmaps*(normalizations.reshape(-1,1,1,1)))
    numerator_i = numerator_sqrt_i**2
    
    def g(numer):
        return np.array(np.sum(np.where(mockdata!=0,numer/(mockdata+1.e-100),0)))
    
    fraction = g(numerator_i)
    
    chi_squared = np.sum(fraction)
    
    return chi_squared



def groupC_chi_squared2(c1,c2,c3,c4,c5,c6,c7,c8):
    """
    Function that calculates chi squared for a set of best fit parameters
    Input: Data set, theoretical signal maps and best fit normalizations
    Output: Chi-squared
    """

    s_DM,b_i,gce1_i,gce2_i,gce3_i,gce4_i,ic_i,fb_i = signalmaps

    numerator_sqrt_i = mockdata - c1*s_DM - c2*b_i - c3*gce1_i - c4*gce2_i - c5*gce3_i - c6*gce4_i - c7*ic_i - c8*fb_i
    numerator_i = numerator_sqrt_i**2
    
    def g(numer):
        return np.array(np.sum(np.where(mockdata!=0,numer/(mockdata+1.e-100),0)))
    
    fraction = g(numerator_i)
    
    chi_squared = np.sum(fraction)
    
    return chi_squared


def groupC_mass_chi_squared(mockdata,signalmaps):
    """
    Function that finds the best fit normalizations for a specific mass and calculates corresponding chi-squared
    Input: Mock data, signal count maps for specific mass
    Output: Chi-squared and best fit normalizations
    """
    
    
    normalizations = groupC_numerical(mockdata,signalmaps)
    
    chi_squared = groupC_chi_squared(mockdata,signalmaps,normalizations)
    
    return chi_squared,normalizations


def groupC_mass_scan(signalmaps,masslist,mockdata):
    """
    Function that computes the best fit values for both mass and normalizations
    Input: Masses to scan, mock data exposure time and pixel size
    Output: Best fit values
    """

    masses = len(masslist)
    reshaper = np.linspace(1,1,masses).reshape(-1,1,1,1)
    mockdata = mockdata*reshaper
    
    # masses (x signals) x 47 x 20 x 20
    chi_squared_list = list(map(lambda x,y: groupC_mass_chi_squared(x,y),mockdata,signalmaps))
    swapped = np.swapaxes(np.array(chi_squared_list),0,1)
    
    chi_squared = (swapped[0]).tolist()
    normalizations = (swapped[1]).tolist()  
    
#     plt.plot(masslist,chi_squared)
#     plt.show()
    
    global_minimum = min(chi_squared)
    min_index = chi_squared.index(global_minimum)
    bestfitmass = masslist[min_index]

    
    return bestfitmass, normalizations[min_index]



def groupC_gce(ring):
    """
    Function that returns gce intensity in a specific ring
    Input: ring ('1','2','3' or '4')
    Output: 47x20x20 map
    """
    
    gceringmap = np.load(groupC_data_path('gce_ring_maps_0419.npy'))*1./TeV/cm**2/s/sr
    
    if ring == '1':
        gce = gceringmap[0]
    if ring == '2':
        gce = gceringmap[1]
    if ring == '3':
        gce = gceringmap[2]
    if ring == '4':
        gce = gceringmap[3]    
    
    return gce


def groupC_iteration(masslist, signalmaps,modelparameters,counter):
    """
    Function that calculates best fit parameters for one iteration
    Input: masslist, signal contributions
    Output: Best fit parameters for one iteration
    """

    mockdata = groupC_mockdata(modelparameters) #will be getting this from group B, use temporary function
    
    winomass,normalizations= groupC_mass_scan(signalmaps,masslist,mockdata)
    
    if counter%50==0: 
        print(counter)
        
    elif counter == 1:
        print(counter)
    
    return winomass,normalizations



def groupC_iterations_10D(iterations):
    """
    Function that iterates the previously called function
    Input: number of iterations
    Output: List of best fit values for all 11 parameters
    """
    iterate = np.linspace(1,1,iterations).reshape(-1,1,1,1,1)
    counter = np.linspace(1,iterations,num=iterations)
    
    mass = 2.7*TeV 
    psi = groupA_psigrid()
    Jmap = groupA_J(psi, flavour='NFW',nPts = 2000)
    
    modelparameters = groupC_modelparameters(mass,Jmap)*iterate
    modelparameters = modelparameters*iterate
    
    iterate2 = iterate.reshape(-1,1)
    iterate3 = iterate.reshape(-1,1,1,1,1,1)
    masslist=np.array([2500,2600,2700,2800,2900,3000,3100,3200])
    
    length = len(masslist)
    ones = np.ones(length).reshape(-1,1,1)
    Jmaps = Jmap*ones
    
    signalmaps = list(map(lambda x,y: groupC_loaddata(x,y),masslist,Jmaps))
    signalmaps = signalmaps*iterate3 #iterations x masses x signals x 47x20x20
    
    bestfit = list(map(lambda x,y,z,w: groupC_iteration(x,y,z,w),masslist*iterate2,signalmaps,modelparameters,counter))
    bestfit = np.array([np.array(bestfit)[:,1:][i].tolist() for i in range(len(np.array(bestfit)[:,1]))])
    
    return bestfit



def groupC_iterations_binbybin(iteration, winomass=3*TeV):

    psi = groupA_psigrid()
    Jmap = groupA_J(psi,flavour='NFW',nPts=2000)
    data_template = np.sum(groupC_modelparameters(winomass,Jmap),axis=0)
    m_ref  = TeV 
    sv_ref = 1.e-25*cm**3/s
    N_ref  = 1.
    signalmaps = groupC_loaddata(None,Jmap,bin_by_bin=True,m_ref=m_ref,sv_ref=sv_ref,N_ref=N_ref)
    bestfit = np.empty((np.alen(energy_central),iteration,len(signalmaps)))
    for i in np.arange(np.alen(energy_central)):
        data_template_new = data_template[i,:,:]*mask_2d
        signalmapsnew = signalmaps[:,i,:,:]
        mockdata = np.random.poisson(data_template_new*np.ones((iteration,1,1)))
        signalmapsnew = signalmapsnew*np.ones((iteration,1,1,1))
        bestfit[i] = np.array(list(map(lambda x,y: groupC_numerical(x,y),mockdata,signalmapsnew)))
        bestfit[i,:,0] *= sv_ref*N_ref/m_ref**2/(energy_H[i]-energy_L[i])
        print('Finished energy bin %s: %.1f GeV'%(i+1,energy_central[i]/GeV))
        print('Median best fit: %.3e cm^3/s/TeV\n'%(np.percentile(bestfit[i,:,0],50)*energy_central[i]**2/(cm**3/s/TeV)))
    return bestfit

def groupC_coeff_wino_binbybin(mDM):
    contComp, energies = groupA_contcomp(mDM)
    p_E = np.loadtxt(groupA_data_path('0407/energy_Edisp_0407.txt'))*TeV
    p_m = np.loadtxt(groupA_data_path('0407/migration_Edisp_0407.txt'))
    p = np.loadtxt(groupA_data_path('0407/Edisp_0407.txt'))
    energy_central = np.loadtxt('dataB/0331/energies_0331.txt')*TeV
    energy_H = np.loadtxt('dataB/0407/energiesH_0407.txt')*TeV
    energy_L = np.loadtxt('dataB/0407/energiesL_0407.txt')*TeV

    contCompSMEAR = groupA_smear1D(contComp, energies, p_m, p_E, p)

    fint_Edisp = interpolate.interp1d(np.log(p_E),p,axis=-1,bounds_error=False,fill_value=0.)
    Edisp2 = fint_Edisp(np.log(mDM))/mDM
    fint_Edisp2 = interpolate.interp1d(p_m,Edisp2,bounds_error=False,fill_value=0.)
    sv_WW, sv_Zg, sv_gg, sv_ZZ = groupA_crosssections(mDM)
    lineCompSMEAR = (sv_gg+sv_Zg/2.)*2.*fint_Edisp2(energies/mDM)

    #energy = np.logspace(np.log10(energy_L),np.log10(energy_H),1001)
    #contCompSMEARvalue = groupA_dNdE_interpolating(energy, energies, contCompSMEAR, mDM)
    #lineCompSMEARvalue = groupA_dNdE_interpolating(energy, energies, lineCompSMEAR, mDM)
    #CompSMEARvalue = contCompSMEARvalue + lineCompSMEARvalue
    #integral = simps((groupC_Aeff(energy)+1e-100*cm**2)*CompSMEARvalue*energy,x=np.log(energy),axis=0)
    #integral = integral/simps((groupC_Aeff(energy)+1e-100*cm**2)*energy,x=np.log(energy),axis=0)
    #integral /= mDM**2

    energy = np.logspace(np.log10(energy_L[0]),np.log10(energy_H[-1]),1001)
    contCompSMEARvalue = groupA_dNdE_interpolating(energy, energies, contCompSMEAR, mDM)
    lineCompSMEARvalue = groupA_dNdE_interpolating(energy, energies, lineCompSMEAR, mDM)
    CompSMEARvalue = contCompSMEARvalue + lineCompSMEARvalue

    return CompSMEARvalue/mDM**2, energy


def groupC_normalizations_plot(data):
    
    energy_central = np.loadtxt(groupC_data_path('energies_0331.txt')) 
    energies = energy_central
    bins = len(energies)
    
    data = np.array(data)
    data = data.reshape(bins,-1,8)
    N_DM = data[:,:,0]
    
    stdlist = []
    meanlist = []
    
    
    for energy in range(bins):
        std = np.std(N_DM[energy,:])
        stdlist.append(std)
        
        mean = np.mean(N_DM[energy,:])
        meanlist.append(mean)
        
    meanlist = np.array(meanlist)
    stdlist = np.array(stdlist)
    
    plt.xscale('log')
    plt.xlabel('Energy (TeV)')
    plt.ylabel('Dark matter normalization')
    plt.ylim(bottom=0,top=8)
    plt.vlines(2.7,0,8,'k')
    plt.xlim((min(energies),5))
    plt.plot(energies,meanlist,'k-')
    plt.fill_between(energies,meanlist+stdlist,meanlist-stdlist,alpha=0.6)
    plt.savefig('DM_normalization.png')
    plt.show()
    return 


def groupC_matrix_scatter_plot(array,names):
    """
    Function that makes a triangular scatterplot
    Input: data, labels
    Output: Triangular plot
    """
    h_array=[]
    ndims=len(array[0])
    [h_array.append(array[:,i]) for i in range(ndims)]

    fig, axes = plt.subplots(ndims,ndims,figsize=(20,15))
    fig.subplots_adjust(wspace=0.35,hspace=0.5)
    plt.rc('xtick',labelsize=8)
    plt.rc('ytick',labelsize=8)
    
    for i in range(ndims): 
        for j in range(ndims): 
            if i == j:
                axes[i,j].hist(h_array[i])
                
            elif i > j:
                axes[i,j].scatter(h_array[j],h_array[i],s=1)
                
            else:
                axes[i,j].axis('off')
            if j == 0:
                if i == j:
                    axes[i,j].set_ylabel('counts',fontsize=14)
                else:
                    axes[i,j].set_ylabel(names[i],fontsize=14)
            if i == ndims-1:
                axes[i,j].set_xlabel(names[j],fontsize=14)
   # plt.savefig('scatterplot.png')            
    plt.show()
    return


def groupC_plot(data,names,iterations):
    """
    Function that calls the scatterplot function and makes a contourplot of data filtered of large outliers
    Input: data, labels, number of iterations
    Output: triangular scatter- and contourplots
    """
    
    mean = np.mean(data[:,:,0],axis=1)
    std = np.std(data[:,:,0],axis=1)
    sigma = 5.
    
    cond = np.ones((iterations,),dtype=bool)
    for i in range(len(data[0])): cond *= (data[:,i,0]>mean[i]-sigma*std[i])*(data[:,i,0]<mean[i]+sigma*std[i]) 
        
    GTC = pygtc.plotGTC(chains=data[:,:,0][cond],paramNames=names,plotName='fullGTC.pdf')
    groupC_matrix_scatter_plot(data,names)
    return



def groupC_masking(array):
    """
    Input: ?x20x20 count 3D array
    Output: Masked array as per the specified mask
    """
    
    def masking(array):
        mask_2d=np.loadtxt(groupC_data_path('mask_0414.txt'))
        array_masked = array*mask_2d
        
        return array_masked

    array_mask = list(map(lambda x: masking(x), array))
    
    return(np.array(array_mask))





    