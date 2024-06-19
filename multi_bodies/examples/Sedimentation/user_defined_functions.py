'''
Simple example of a flagellated bacteria. 
'''
from __future__ import division, print_function
import numexpr as ne
import multi_bodies_functions
from multi_bodies_functions import *


def bodies_external_force_torque_new(bodies, r_vectors, *args, **kwargs):
  '''
  This function returns the external force-torques acting on the bodies.
  It returns an array with shape (2*len(bodies), 3)
  
  In this is example we just set it to zero.
  '''
  force_torque = np.zeros((2*len(bodies), 3))
  for k, b in enumerate(bodies):
    force_torque[k,2] = b.mg
    
  return force_torque
multi_bodies_functions.bodies_external_force_torque = bodies_external_force_torque_new


