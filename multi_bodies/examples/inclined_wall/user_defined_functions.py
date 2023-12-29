'''
We defined here the tilt gravity force
'''
import numpy as np
import multi_bodies_functions
from multi_bodies_functions import *



# Override blob_external_forces
def blob_external_forces_new(r_vectors, *args, **kwargs):
  '''
  This function compute the external force acting on a
  single blob. It returns an array with shape (3).

  In this example we add tilted gravity and a repulsion with the wall;
  the interaction with the wall is derived from the potential

  U(z) = U0 + U0 * (a-z)/b   if z<a
  U(z) = U0 * exp(-(z-a)/b)  iz z>=a

  with
  e = repulsion_strength_wall
  a = blob_radius
  h = distance to the wall
  b = debye_length_wall
  '''
  N = r_vectors.size // 3
  f = np.zeros((N, 3))

  # Get parameters from arguments
  tilt_angle = kwargs.get('tilt_angle')
  blob_mass = kwargs.get('blob_mass')
  blob_radius = kwargs.get('blob_radius')
  g = kwargs.get('g')
  repulsion_strength_wall = kwargs.get('repulsion_strength_wall')
  debye_length_wall = kwargs.get('debye_length_wall')
  # Add gravity
  f[:,0] = -g * blob_mass * np.sin(tilt_angle)
  f[:,2] = -g * blob_mass * np.cos(tilt_angle)

  # Add wall interaction
  h = r_vectors[:,2]
  lr_mask = h > blob_radius
  sr_mask = h <= blob_radius
  f[lr_mask,2] += (repulsion_strength_wall / debye_length_wall) * np.exp(-(h[lr_mask]-blob_radius)/debye_length_wall)
  f[sr_mask,2] += (repulsion_strength_wall / debye_length_wall)

  return f
multi_bodies_functions.blob_external_forces = blob_external_forces_new




