#!/bin/python3
'''
Created on 2024-06-14

@author: Guillaume Le Treut - guillaume.letreut@gmail.com

@description:
Methods useful for self-consistent field calculations.
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
# Calculations of averages and operators
#---------------------------------------------------------------------------
def prop2density(Qs_left, Qs_right):
  '''
  Compute the density function from the array of propagators
  INPUT:
    * Qs_left [NsxM^d-array]: propagator evaluated at successive contour lengths (left branch)
    * Qs_right [NsxM^d-array]: propagator evaluated at successive contour lengths (right branch)
    * L [float]: chain length
  OUTPUT:
    * rho [M^d-array]: density function
  '''
  if (Qs_left.shape != Qs_right.shape):
    log.error("Left and right propagator dimensions much match!")
    raise ValueError

  Ns = len(Qs_left)
  grid_shape = Qs_left.shape[1:]

  rho = np.zeros(grid_shape)

  for i in range(Ns):
    rho += Qs_left[i]*Qs_right[Ns-1-i]

  rho = rho / (np.sum(rho)) * Ns
  return rho

