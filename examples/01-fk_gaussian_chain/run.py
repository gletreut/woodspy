#!/bin/python3
'''
Created on 2024-06-14

@author: Guillaume Le Treut - guillaume.letreut@gmail.com

@description:
Compute the density of a Gaussian chain in an external potential by integration of the FK equation.

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
# functions
#---------------------------------------------------------------------------
def potential(x,eps=1.):
  '''
  INPUT:
    * x [dxM1x...xMd array]: mesh with dimension d >= 1
  '''
  xshape = x.shape
  ndim = x.shape[0]

  center = np.arange(ndim)+1
  center = center.reshape([ndim]+[1]*ndim)

  return eps*np.einsum('d...->...', (x-center)**2)

def solve_FK_gaussian(S, X, U, Qinit=None, domain=None, b=1.):
  '''
  Solve the Fokker-Planck equation associated to a Gaussian chain.
  INPUT:
    * S [Ns-array]: contour length values at which
    *
  '''
  # initializations
  ds = S[1]-S[0]
  # compute dx
  ndim = len(X)
  dx = []
  for d in range(ndim):
    Xd = X[d]
    i0 = np.zeros(ndim, dtype=np.int_)
    i1 = i0 + np.int_(np.arange(ndim, dtype=np.int_) == d)
    dx.append(Xd[tuple(i1)]-Xd[tuple(i0)])

  Qs = []
  if Qinit is None:
    Q = np.ones(X.shape)    # the normalization is not very important

#---------------------------------------------------------------------------
# parameters
#---------------------------------------------------------------------------
L = 10.
b = 1.
Ns = 2**8
xlim = np.array([[-1.5*L, 1.5*L],[-1.5*L, 1.5*L]])
Nx = np.array([2**5,2**5])
seed = 123
idump = 1

#-------------------------------------------------------------------
# script
#-------------------------------------------------------------------

if __name__ == "__main__":
  # output folder
  outdir = Path(__file__).parent
  resfile = outdir / 'results.hd5'
  logging.info("Run results in file {:s}".format(str(resfile)))

  # choose the true action values from a Gaussian distribution
  rng = np.random.default_rng(seed=seed)
  # with h5py.File(resfile, 'w') as fout:
    # dset = fout.create_dataset("true/qvalue", data=Q_true)

  # compute some variables
  ds = L / (Ns-1)
  dx = np.diff(xlim, axis=1)[:,0]/(Nx-1)
  logging.info("ds = {:g}".format(ds))
  logging.info("dx = {:s}".format(np.array2string(dx, precision=8)))

  # define the geometry
  xvals = [np.linspace(*xl, n) for (xl,n) in zip(xlim,Nx)]
  mesh = np.array(np.meshgrid(*xvals, indexing='ij'))     # 1st dimension: index of the coordinate expressed in this array
                                                          # other dimensions: value of the coordinate for the given dimension at the grid points

  '''
  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax = fig.gca()
  xmesh,ymesh = mesh
  ax.plot(xmesh.ravel(), ymesh.ravel(), 'o')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  plt.show()
  #'''

  # compute the potential
  U = potential(mesh)
  '''
  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax = fig.gca()
  xmesh,ymesh = mesh
  ax.contourf(xmesh, ymesh, U)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  plt.show()
  #'''

  # compute the propagator at the specified contour points
  S = np.linspace(0., L, Ns)
  Q = solve_FK_gaussian(S, mesh, U, Qinit=None, domain=None, b=b)

  logging.info("Normal exit!")



