# This module handles the projection of a catalog on a specific grid
# Adapted from the DESC WLMASSMAP code.
# Comes from https://github.com/LSSTDESC/SkySim5000_IA_infusion/blob/main/MapMaking.ipynb

import numpy as np
import healpy as hp


def eq2ang(ra, dec):
    """
    convert equatorial ra,dec in degrees to angular theta, phi in radians
    parameters

    ----------
    ra: scalar or array
        Right ascension in degrees
    dec: scalar or array
        Declination in degrees
    returns
    -------
    theta,phi: tuple
        theta = pi/2-dec*D2R # in [0,pi]
        phi   = ra*D2R       # in [0,2*pi]
    """
    dec = dec*np.pi/180
    ra  = ra*np.pi/180
    theta = np.pi/2 - dec
    phi  = ra
    return theta, phi

def project_healpix(catalog, nside, hp_type='RING'):
    """
    Adds a HEALpix pixel index to all galaxies in the catalog

    Parameters
    ----------
    catalog: table
        Input shape catalog

    nside: int
        HEALpix nside parameter

    hp_type: string
        HEALpix pixel order ('RING', 'NESTED')

    Returns
    -------
    catalog: table
        Output shape catalog with pixel index column
        nest will be either True or False depending on input
    """
    theta, phi = eq2ang(catalog['ra'], catalog['dec'])
    catalog['pixel_index'] = hp.ang2pix(nside, theta, phi,
                                        nest=(hp_type=='NESTED'))
    return catalog

def bin_shear_map(catalog, nx=None, ny=None, npix=None, sigtype=None, seed_id=None):
    """
    Computes the shear map by binning the catalog according to pixel_index.
    Either nx,ny or npix must be provided.
    
    Parameters
    ----------
    catalog: table
        Input shape catalog with pixel_index column

    nx,ny: int, optional
        Number of pixels of a 2d flat map
    
    npix: int, optional
        Number of pixels of a spherical map (or other 1D pixelating scheme)

    sigtype: type of observations 
        GG_noisefree    = cosmic shear noise-free
        GG_IA_noisefree = cosmic shear noise-free + IA
        GG_IA_noisy     = cosmic shear + IA + shapenoise
        pureIA          = pure IA 
        
    Returns
    -------
    gmap: ndarray
        Shear maps

    nmap: ndarray
        Number of galaxies per pixels
    """
    assert (npix is not None) or ((nx is not None) and (ny is not None))
    assert (sigtype is not None)

    # Bin the shear catalog
    if npix is None:
        npix = nx*ny

    if sigtype == 'GG_noisefree':
        str1 = 'g1';str2 = 'g2'
    if sigtype == 'GG_IA_noisy':
        str1 = 'e_obs_1';str2 = 'e_obs_2'
    if sigtype == 'GG_IA_noisefree':
        str1 = 'e_obs_no_noise_1';str2 = 'e_obs_no_noise_2'
    if sigtype == 'pureIA':
        str1 = 'e_IA_TATT_1';str2 = 'e_IA_TATT_2'
    if sigtype == 'HACC':
        str1 = 'gamma1';str2 = 'gamma2'
    if sigtype == 'KiDS':
        str1 = 'e1';str2 = 'e2'; strw = 'weight'

        
    #HACC/OuterRim:
    g1map = np.bincount(catalog['pixel_index'],
                        weights=catalog[str1],
                        minlength=npix)
    g2map = np.bincount(catalog['pixel_index'],
                        weights=catalog[str2],
                        minlength=npix)
    Nmap  = np.bincount(catalog['pixel_index'], minlength=npix)

    # Normalize by number of galaxies
    nz_ind = Nmap > 0
    g1map[nz_ind] /= Nmap[nz_ind]
    g2map[nz_ind] /= Nmap[nz_ind]

    gmap = np.stack([g1map,g2map], axis=0)

    return gmap, Nmap