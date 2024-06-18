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
import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.colors as mco
import matplotlib.cm as mcm
from woodspy.polymer.models import GaussianChain
from woodspy.polymer.scft import prop2density
#---------------------------------------------------------------------------
# functions
#---------------------------------------------------------------------------
def potential(x,eps=1.,x0=None):
  '''
  INPUT:
    * x [dxM1x...xMd array]: mesh with dimension d >= 1
  '''
  xshape = x.shape
  ndim = x.shape[0]

  if x0 is None:
    x0 = np.zeros([ndim]+[1]*ndim)

  return eps*np.einsum('d...->...', (x-x0)**2)

def nu_exp(d):
  if d == 1:
    return 1.
  elif d == 2:
    return 0.75
  elif d == 3:
    return 0.6
  else:
    logging.error("Nu exponent not implemnted for d = {:d}".format(d))
    raise ValueError

def gamma_exp(d):
  if d == 1:
    return 1.
  elif d == 2:
    return 4/3.
  elif d == 3:
    return 7/6.
  else:
    logging.error("Nu exponent not implemnted for d = {:d}".format(d))
    raise ValueError

#---------------------------------------------------------------------------
# parameters
#---------------------------------------------------------------------------
b = 1.
seed = 123
# idump = 1

'''
# 1D
L = 80.
Ns = 2**8
xlim = np.array([[-1.5*L, 1.5*L]])
Nx = np.array([2**16])

#'''
# 2D
L = 20.
Ns = 2**8
xlim = np.array([[-1.5*L, 1.5*L],[-1.5*L, 1.5*L]])
Nx = np.array([2**7,2**7])
#'''

'''
# 3D
L = 10.
Ns = 2**8
xlim = np.array([[-1.5*L, 1.5*L],[-1.5*L, 1.5*L],[-1.5*L, 1.5*L]])
Nx = np.array([2**5,2**5,2**5])
#'''

# general parameters for figures
lw = 0.5
ms = 2
ext = '.png'
dpi = 300

#-------------------------------------------------------------------
# script
#-------------------------------------------------------------------

if __name__ == "__main__":
  #------ initializations ------------------------------------------
  # output folder
  outdir = Path(__file__).parent
  # resfile = outdir / 'results.hd5'
  # logging.info("Run results in file {:s}".format(str(resfile)))

  # ouput directory for plots
  figdir = outdir / 'figures'
  if not figdir.is_dir():
    figdir.mkdir()
  logging.info("Figures in directory {:s}".format(str(figdir)))

  # choose the true action values from a Gaussian distribution
  rng = np.random.default_rng(seed=seed)
  # with h5py.File(resfile, 'w') as fout:
    # dset = fout.create_dataset("true/qvalue", data=Q_true)

  # define several variables
  ds = L / (Ns-1)
  dx = np.diff(xlim, axis=1)[:,0]/(Nx-1)
  logging.info("ds = {:g}".format(ds))
  logging.info("dx = {:s}".format(np.array2string(dx, precision=8)))

  # define the contour points
  S = np.linspace(0., L, Ns)

  # define the geometry
  xvals = [np.linspace(*xl, n) for (xl,n) in zip(xlim,Nx)]
  mesh = np.array(np.meshgrid(*xvals, indexing='ij'))     # 1st dimension: index of the coordinate expressed in this array
                                                          # other dimensions: value of the coordinate for the given dimension at the grid points

  # define the domain for the first monomer
  imid = [n//2 for n in Nx]
  xmid = mesh[:,*imid].reshape([len(mesh)] + [1]*len(mesh))
  Qinit = np.zeros(Nx, dtype=np.float_)
  Qinit[*imid] = 1.

  #'''
  if len(Nx) == 2:
    fig = plt.figure(facecolor='w')
    ax = fig.gca()
    xmesh,ymesh = mesh
    ax.plot(xmesh.ravel(), ymesh.ravel(), 'o', ms=ms, mfc='b', mew=0)

    idx = Qinit == 1.
    ax.plot(xmesh[idx].ravel(), ymesh[idx].ravel(), 'ro', ms=ms)
    ax.set_xlabel('x', fontsize='medium')
    ax.set_ylabel('y', fontsize='medium')

    fname = 'initial_condition'
    fpath = figdir / (fname + ext)
    fig.savefig(fpath, bbox_inches='tight', pad_inches=0, dpi=dpi)
    # plt.show()
    fig.clf()
    plt.close('all')
  #'''

  # chain model
  chain = GaussianChain(b)

  #------ free Gaussian chain attached to the origin ---------------
  logging.info("Starting calculations for free chain.")
  Qs_left = chain.solve_FK_gaussian(S, mesh, np.zeros(mesh[0].shape), Qinit=Qinit)
  Qs_right = chain.solve_FK_gaussian(S, mesh, np.zeros(mesh[0].shape), Qinit=None)

  rho_free = prop2density(Qs_left, Qs_right)
  rho_free = rho_free / np.sum(rho_free) * L

  preturn_free = Qs_left[:,*imid] / np.sum(Qs_left, axis=tuple(np.arange(1,len(Qs_left.shape))))

  #------ Gaussian chain in an external potential ------------------
  logging.info("Starting calculations for chain in external potential.")

  # compute the potential
  U = potential(mesh, eps=0.1, x0=xmid)

  #'''
  fname = 'potential'
  fpath = figdir / (fname + ext)
  ## 1D
  if len(Nx) == 1:
    fig = plt.figure(facecolor='w')
    ax = fig.gca()

    xmesh = mesh[0]
    ax.plot(xmesh, U)
    ax.set_xlabel('x', fontsize='medium')
    ax.set_ylabel('U(x)', fontsize='medium')

    fig.savefig(fpath, bbox_inches='tight', pad_inches=0, dpi=dpi)
    # plt.show()
    fig.clf()
    plt.close('all')

  ## 2D
  elif len(Nx) == 2:
    fig = plt.figure(facecolor='w')
    ax = fig.gca()
    xmesh,ymesh = mesh
    mapl = ax.contourf(xmesh, ymesh, U)
    plt.colorbar(mapl)

    ax.set_xlabel('x', fontsize='medium')
    ax.set_ylabel('y', fontsize='medium')
    fig.savefig(fpath, bbox_inches='tight', pad_inches=0, dpi=dpi)
    # plt.show()
    fig.clf()
    plt.close('all')
  #'''

  Qs_left = chain.solve_FK_gaussian(S, mesh, U, Qinit=Qinit)
  Qs_right = chain.solve_FK_gaussian(S, mesh, U, Qinit=None)

  rho_potential = prop2density(Qs_left, Qs_right)
  rho_potential = rho_potential / np.sum(rho_potential) * L

  #------ Gaussian chain with self excluded volume -----------------
  logging.info("Starting self-consistent calculations for chain with excluded volume.")
  # initial guess for the density
  rho_old = np.ones(Nx, dtype=np.float_)
  rho_old = rho_old / np.sum(rho_old) * L

  # define self-consistent potential
  U = 1.*rho_old

  rho_list = [rho_old]
  rnorm_old = np.linalg.norm(rho_old)
  etol = 1.0e-6
  itermax = 50
  for it in range(itermax):
    Qs_left = chain.solve_FK_gaussian(S, mesh, U, Qinit=Qinit)
    Qs_right = chain.solve_FK_gaussian(S, mesh, U, Qinit=None)

    rho_new = prop2density(Qs_left, Qs_right)
    rho_new = rho_new / np.sum(rho_new) * L

    # stop criterion
    rnorm_new = np.linalg.norm(rho_new)
    erel = 2*np.linalg.norm(rho_old - rho_new)/(rnorm_old + rnorm_new)
    logging.info("erel = {:g}".format(erel))

    # updates
    rho_list.append(rho_new)
    rho_old = rho_new.copy()
    rnorm_old = rnorm_new
    U = 1.*rho_old

    # stop criterion
    if (erel < etol):
      logging.info("Stop criterion reached in SCFT.")
      break
    elif it == itermax -1:
      logging.warning("Maximum iteration reached in SCFT!")
  # end for-loop
  rho_scft = rho_list[-1].copy()
  preturn_scft = Qs_left[:,*imid] / np.sum(Qs_left, axis=tuple(np.arange(1,len(Qs_left.shape))))

  #''' show SCFT convergence
  X = np.sqrt(np.sum((mesh - xmid)**2, axis=0)).ravel()
  idx = np.argsort(X)
  fig = plt.figure()
  ax = fig.gca()

  norm = mco.Normalize(vmin=0, vmax=len(rho_list)-1)
  cm = mcm.rainbow
  for i,rho in enumerate(rho_list):
    Y = rho.ravel()
    ax.plot(X[idx], Y[idx], '-', lw=lw, color=cm(norm(i)))

  ax.set_xlim(0.,10.*np.sqrt(L)/6.)
  ax.set_ylim(0.,None)
  ax.set_xlabel('r', fontsize='medium')
  ax.set_ylabel('rho(r)', fontsize='medium')

  fname = 'scft_convergence'
  fpath = figdir / (fname + ext)
  fig.savefig(fpath, bbox_inches='tight', pad_inches=0, dpi=dpi)
  # plt.show()
  fig.clf()
  plt.close('all')
  #'''

  #------ compare the concentration profiles -----------------------
  #'''
  fig = plt.figure(facecolor='w')
  ax = fig.gca()
  X = np.sqrt(np.sum((mesh - xmid)**2, axis=0)).ravel()
  idx = np.argsort(X)
  fig = plt.figure()
  ax = fig.gca()

  labels = ['free', 'potential', 'scft']
  rho_list = [rho_free, rho_potential, rho_scft]
  for rho, label in zip(rho_list, labels):
    Y = rho.ravel()

    ax.plot(X[idx], Y[idx], '-', lw=lw, label=label)

  ax.set_xlabel('r', fontsize='medium')
  ax.set_ylabel('rho(r)', fontsize='medium')
  ax.legend(loc='best', fontsize='medium')
  ax.set_xlim(0.,10.*np.sqrt(L)/6.)
  ax.set_ylim(0.,None)

  fname = 'concentration_profiles'
  fpath = figdir / (fname + ext)
  fig.savefig(fpath, bbox_inches='tight', pad_inches=0, dpi=dpi)
  # plt.show()
  fig.clf()
  plt.close('all')
  #'''

  #------ compare the probability of looping -----------------------
  #'''
  fig = plt.figure()
  ax = fig.gca()

  labels = ['free', 'scft']
  preturn_list = [preturn_free, preturn_scft]
  for preturn, label in zip(preturn_list, labels):

    ax.plot(S, preturn, '-', lw=lw, label=label)

  ndim = len(Nx)
  sexp = ndim/2.
  i = Ns//2
  ax.plot(S[i:], S[i:]**(-sexp)*1.1*preturn_free[i]/S[i]**(-sexp), 'k-', lw=lw)
  nu = nu_exp(ndim)
  gamma = gamma_exp(ndim)
  sexp = nu*ndim + gamma - 1.
  i = Ns//2
  ax.plot(S[i:], S[i:]**(-sexp)*0.9*preturn_scft[i]/S[i]**(-sexp), 'k-', lw=lw)
  ax.set_xlabel('s', fontsize='medium')
  ax.set_ylabel('p_N(0)', fontsize='medium')
  ax.legend(loc='best', fontsize='medium')
  # ax.set_xlim(0.,None)
  # ax.set_ylim(0.,None)
  ax.set_xscale('log')
  ax.set_yscale('log')

  fname = 'proba_loop'
  fpath = figdir / (fname + ext)
  fig.savefig(fpath, bbox_inches='tight', pad_inches=0, dpi=dpi)
  # plt.show()
  fig.clf()
  plt.close('all')
  #'''

  logging.info("Normal exit!")



