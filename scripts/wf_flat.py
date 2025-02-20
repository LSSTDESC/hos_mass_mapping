import numpy as np
from typing import List
import scipy.interpolate
import pyccl

from scripts.ks_flat import ks93inv, ks93


def compute_s_from_theory(cosmo_params: List[float], z: np.ndarray, nz: np.ndarray, box_width_pix: int, box_size_rad: float, pixel_size_rad: float) -> np.ndarray:
    """
    Returns the S covariance matrix as generated from theory. Note: the result is not FFT-shifted.

    Parameters
    ----------
    z: array
        Number density z sample points
    nz: array
        Number density n(z) values
    box_width_pix: int
        The number of pixels on one side of the box
    box_size_rad: float
        The size of the box in radians

    Returns
    -------
    S_theory: 2D array
    """

    # get the spatial frequencies
    k = np.fft.fftfreq(box_width_pix)
    k1, k2 = np.meshgrid(k, k)

    # get the corresponding ell values for the pixels
    kk = np.sqrt(k1**2 + k2**2)
    ll = kk * 2 * np.pi * box_width_pix / box_size_rad    

    # compute the theory
    Omega_m, sigma8, w0, n_s, Omega_b, h = cosmo_params
    Omega_c = Omega_m - Omega_b
    # baryonic correction parameters from p.10 https://arxiv.org/pdf/2103.05582.pdf
    extra_parameters = {"camb": {"HMCode_A_baryon": 3.13,
                            "HMCode_eta_baryon": 0.603}}

    cosmo = pyccl.Cosmology(Omega_c = Omega_c, Omega_b = Omega_b, sigma8 = sigma8, h = h, n_s = n_s, w0 = w0, extra_parameters = extra_parameters)
    tracer = pyccl.WeakLensingTracer(cosmo, (z, nz))
    ell_theory = np.arange(10000)
    cl_theory = pyccl.angular_cl(cosmo, tracer, tracer, ell_theory)

    # convert with the right scaling down to the box
    cl_scaled = (box_width_pix / pixel_size_rad)**2 * cl_theory

    # interpolate to the right ell values
    theory_interpolator = scipy.interpolate.UnivariateSpline(ell_theory, cl_scaled)
    S_theory = theory_interpolator(ll)
    return S_theory


def spin_wiener_filter(data_q: np.ndarray, data_u: np.ndarray, ncov_diag_Q: np.ndarray, ncov_diag_U: np.ndarray, input_ps_map_E: np.ndarray, input_ps_map_B: np.ndarray, iterations=10):
    """
    Wiener filter Elsner-Wandelt messenger field adapted for spin-2 fields (CMB polarization or galaxy weak lensing)

    Parameters
    ----------
    data_q : Q square image data (e.g. gamma1)
    data_u : U square image data (e.g. gamma2)
    ncov_diag_Q : Q noise variance per pixel (assumed uncorrelated)
    ncov_diag_U : U noise variance per pixel (assumed uncorrelated)
    input_ps_map_E : 1D power P(k) for E-mode signal power spectrum evaluated 2D components k1,k2 as a square image
    input_ps_map_B : 1D power P(k) for B-mode signal power spectrum evaluated 2D components k1,k2 as a square image
    iterations : number of iterations

    Returns
    -------
    s_q,s_u : Wiener filtered q and u signals
    """
    tcov_diag = np.min(np.array([ncov_diag_Q, ncov_diag_U]))
    scov_ft_E = np.fft.fftshift(input_ps_map_E)
    scov_ft_B = np.fft.fftshift(input_ps_map_B)
    s_q = np.zeros(data_q.shape)
    s_u = np.zeros(data_q.shape)

    for i in np.arange(iterations):
        # in Q, U representation
        t_Q  = (tcov_diag/ncov_diag_Q)*data_q + ((ncov_diag_Q-tcov_diag)/ncov_diag_Q) * s_q
        t_U  = (tcov_diag/ncov_diag_U)*data_u + ((ncov_diag_U-tcov_diag)/ncov_diag_U) * s_u
        # in E, B representation
        t_E, t_B = ks93(t_Q,t_U)
        s_E  = (scov_ft_E/(scov_ft_E+tcov_diag))*np.fft.fft2(t_E)
        s_B  = (scov_ft_B/(scov_ft_B+tcov_diag))*np.fft.fft2(t_B)
        s_E = np.fft.ifft2(s_E)
        s_B = np.fft.ifft2(s_B)
        # in Q, U representation
        s_q, s_u = ks93inv(s_E,s_B)

    return s_q, s_u



