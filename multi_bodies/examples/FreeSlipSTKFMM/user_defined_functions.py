'''
Use this module to override forces interactions defined in 
multi_body_functions.py. See an example in the file
RigidMultiblobsWall/multi_bodies/examples/user_defined_functions.py



In this module we override the default blob-blob, blob-wall and
body-body interactions used by the code. To use this implementation 
copy this file to 
RigidMultiblobsWall/multi_bodies/user_defined_functions.py


This module defines (and override) the slip function:

  def set_slip_by_ID_new(body)

and it defines the new slip function slip_extensile_rod, 
see below.
'''
from __future__ import division, print_function
import numpy as np
import scipy.special as scsp
import math
import multi_bodies_functions
from multi_bodies_functions import *
import general_application_utils as utils
# Try to import numba
try:
  from numba import njit, prange
except ImportError:
  print('numba not found')

# Try to import the visit_writer (boost implementation)
try:
  import visit.visit_writer as visit_writer
except ImportError:
  pass
# Try to import stkfmm library
try:
  import PySTKFMM
except ImportError:
  pass



def set_slip_by_ID_new(body, slip, *args, **kwargs):
  '''
  This functions assing a slip function to each
  body depending on his ID. The ID of a structure
  is the name of the clones file (without .clones)
  given in the input file.
  As an example we give a default function which sets the
  slip to zero and a function for active rods with an
  slip along its axis. The user can create new functions
  for other kind of active bodies.
  '''
  body.function_slip = partial(flow_resolved, *args, **kwargs)
  return
multi_bodies_functions.set_slip_by_ID = set_slip_by_ID_new


def flow_resolved(body, *args, **kwargs):
  '''
  Adds the background flow.
  '''
  # Get blobs vectors 
  r_configuration = body.get_r_vectors()
  
  # Flow along x, gradiend along z
  return  flow_resolved_coord(r_configuration, *args, **kwargs)


def flow_resolved_coord(r, *args, **kwargs):
  '''
  Adds the background flow.
  '''
  # Get slip options, offset start, offset end and speed
  gamma_dot = kwargs.get('gamma_dot')
  Lz = kwargs.get('Lz')
  N = r.size // 3
  
  # Flow along x, gradiend along z
  background_flow = np.zeros((N, 3))
  background_flow[:,0] = -gamma_dot * np.cos(2 * np.pi * r[:,2] / Lz)
    
  return background_flow.reshape((N, 3))
