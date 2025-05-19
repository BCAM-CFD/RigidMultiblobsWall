'''
Simple example of a flagellated bacteria. 
'''
from __future__ import division, print_function
import numexpr as ne
import multi_bodies_functions
from multi_bodies_functions import *


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


def flow_resolved(body, t, *args, **kwargs):
  '''
  Adds the background flow.
  '''
  # Get blobs vectors 
  r_configuration = body.get_r_vectors()
  
  # Flow along x, gradiend along z
  return  flow_resolved_coord(r_configuration, body.location, t, *args, **kwargs)


def flow_resolved_coord(r, q, t, *args, **kwargs):
  '''
  Use Poisseuille flow.

  IMPORTANT: edit the variables flow_magnitude and radius_effect to the desired values.
  '''
  # Set slip options
  flow_magnitude = kwargs.get('flow_magnitude')
  omega_0 = kwargs.get('omega_0')
  omega_f = kwargs.get('omega_f')
  delta = kwargs.get('delta')
  t_f = kwargs.get('t_f')

  # Flow along x, gradiend along z
  N = r.size // 3  
  background_flow = np.zeros((N, 3))
   
  # time-dependent flow magnitude
  flow = flow_magnitude * np.sin(t_f * omega_0 / np.log(omega_f / omega_0) * ((omega_f / omega_0)**(t / t_f) - 1))
  if t / t_f < 0.5 * delta:
    chirp = 0.5 + 0.5 * np.cos(2 * np.pi / delta * (t / t_f - 0.5 * delta))
  elif t / t_f < 1 - 0.5 * delta:
    chirp = 1
  else:
    chirp = 0.5 + 0.5 * np.cos(2 * np.pi / delta * (t / t_f - 1 + 0.5 * delta))

  # Set background flow along z-axis
  background_flow[:,0] = chirp * flow * (r[:,2] - q[2])
    
  return background_flow

