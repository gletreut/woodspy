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

import numpy as np
import opt_einsum as oe

#---------------------------------------------------------------------------
# Gaussian chain
#---------------------------------------------------------------------------
class GaussianChain:
  def __init__(self,b):
    self.b = b

  def solve_FK_gaussian(self, S, X, U, Qinit=None):
    '''
    Solve the Fokker-Planck equation associated to a Gaussian chain.
    INPUT:
      * S [Ns-array]: contour length values at which
      * X [dxM^d-array]: list of mesh coordinate values for each dimension: X[a] returns the coordinates for dimension a, evaluated on the mesh
      * U [M^d-array]: potential function evaluated on the mesh
      * Qinit [M^d-array]: domain for initial monomer, if None then 1s everywhere.
    OUTPUT:
      * Qs [NsxM^d-array]: propagator evaluated at successive contour lengths

    The FK equation considered is:
      dq/dt = b^2 / 6 \nabla^2 q - U(x) q

    Reference:
      G. Fredickson. The Equilibrium theory of inhomogeneous polymers. p 112. Algorithm 3.1
    '''
    from woodspy.utils import compute_laplacian_tilde, compute_fft, compute_ifft
    # contour length increment
    ds = S[1]-S[0]
    # spatial discretization
    ndim = len(X)
    dx = []
    for d in range(ndim):
      Xd = X[d]
      i0 = np.zeros(ndim, dtype=np.int_)
      i1 = i0 + np.int_(np.arange(ndim, dtype=np.int_) == d)
      dx.append(Xd[tuple(i1)]-Xd[tuple(i0)])

    # initialization
    if Qinit is None:
      Q = np.ones(list(X.shape[1:]))    # the normalization is not very important
                              # but propto concentration of 1st monomer
    else:
      Q = Qinit.copy()
    Qs = [Q.copy()]

    # integration by strang splitting of the FK equation
    ## potential step evolution operator
    evol_U = np.exp(-U*0.5*ds)
    ## diffusive step evolution operator (in Fourier space)
    lap_tilde = compute_laplacian_tilde(Q.shape, a=dx)
    evol_D = np.exp(self.b**2/6. * lap_tilde * ds)

    ## integration loop
    for i in range(len(S)-1):
      Q1 = np.einsum('...,...->...', Q, evol_U)
      Q1_tilde = compute_fft(Q1, start=0)
      Q2_tilde = np.einsum('...,...->...', Q1_tilde, evol_D)
      Q2 = compute_ifft(Q2_tilde, start=0)
      Q = np.einsum('...,...->...', Q2, evol_U)
      Qs.append(Q.copy())
    ## end for loop

    return np.array(Qs)
