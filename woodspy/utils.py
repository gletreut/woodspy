#!/bin/python3
'''
Created on 2024-06-14

@author: Guillaume Le Treut - guillaume.letreut@gmail.com
'''

#---------------------------------------------------------------------------
# imports
#---------------------------------------------------------------------------
from pathlib import Path
import logging
logging.basicConfig(format='%(asctime)s [%(filename)s:%(lineno)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

import h5py
import numpy as np

#---------------------------------------------------------------------------
# FFT
#---------------------------------------------------------------------------
def compute_freq_mesh(shape):
    """
    compute frequency meshes.
    Returns a list of matrices with shape `shape`, [K1, K2, ..., Kndim], where ndim is len(shape).
    Ki[j1,j2,j3,...,jndim] = ji/shape[i]
    """
    ndim = len(shape)

    # store fourier frequencies ranges
    kranges = []
    for i in range(ndim):
        krange = np.fft.fftfreq(shape[i])
        kranges.append(krange)

    # build meshgrid of fourier frequencies
    Ks = [None for i in range(ndim)]
    return np.array(np.meshgrid(*kranges, indexing='ij'))

def compute_nabla_tilde(shape, a=1., reverse=False):
    """
    compute the nabla "vector".
      * shape: vector giving the size in each dimension of an input field.
      * a: lattice site size. (length unit).
    """

    ndim = len(shape)
    newshape = [ndim] + list(shape)
    nabla_tilde = np.zeros(newshape, dtype=np.complex_)

    # build meshgrid of fourier frequencies
    Ks = compute_freq_mesh(shape)

    # fill-in nabla_tilde vector
    for d in range(ndim):
        # nabla_tilde[d] = 2*1.j*np.exp(1.j*np.pi*Ks[d])*np.sin(np.pi*Ks[d]) / a
        nabla_tilde[d] = 2*np.pi*1.j*Ks[d] / a

    if reverse:
        nabla_tilde = - np.conjugate(nabla_tilde)

    return nabla_tilde

def compute_laplacian_tilde(shape, a=1.):
    """
    compute the laplacian field.
      * shape: vector giving the size in each dimension of an input field.
      * a: lattice site size. (length unit).
    """

    # build meshgrid of fourier frequencies
    Ks = compute_freq_mesh(shape)

    # return -4.*np.sum(np.sin(np.pi*Ks)**2, axis=0)
    return -(2.*np.pi)**2 * np.sum(Ks**2, axis=0)

def compute_fft(phi, start=1):
    """
    Compute the fft for the input field phi.
    Assumes that the dimension d<start are not to be Fourier transformed.
    """
    xp = get_array_module(phi)

    shape = phi.shape
    ndim = len(shape)

    return xp.fft.fftn(phi, axes=range(start, ndim))

def compute_ifft(phi_tilde, start=1):
    """
    Compute the ifft for the input field phi_tilde.
    Assumes that the dimension d<start are not to be Fourier transformed.
    Assumes that the returned field must be real-valued.
    """
    xp = get_array_module(phi_tilde)

    shape = phi_tilde.shape
    ndim = len(shape)

    return xp.real(xp.fft.ifftn(phi_tilde, axes=range(start, ndim)))
