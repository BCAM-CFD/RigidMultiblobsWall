import numpy as np
import multi_bodies_functions
from multi_bodies_functions import *


# Override blob_external_force 
def blob_external_forces_new(r_vectors, *args, **kwargs):
  '''
  This function compute the external force acting on a
  single blob. It returns an array with shape (3).
  
  The interaction with the wall is derived from the potential

  U(d) = U0 * exp(-(d-a)/b)

  U(z) = U0 + U0 * (a-z)/b   if z<a
  U(z) = U0 * exp(-(z-a)/b)  iz z>=a
 
  with 
  e = repulsion_strength_wall
  a = blob_radius
  h = distance to the wall
  b = debye_length_wall

  WARNING, before using set the position of the walls.
  '''
  # Set walls
  Lx_minus = 0
  Lx_plus = 20
  Ly_minus = 0
  Ly_plus = 20
  Lz_minus = 0
  Lz_plus = 20

  # Get parameters from arguments
  blob_radius = kwargs.get('blob_radius')
  repulsion_strength_wall = kwargs.get('repulsion_strength_wall')
  debye_length_wall = kwargs.get('debye_length_wall')

  # Init zero force
  f = np.zeros_like(r_vectors)

  # Add interactions with Lx_minus wall
  h = r_vectors[:,0] - Lx_minus
  lr_mask = h > blob_radius
  sr_mask = h <= blob_radius
  f[lr_mask,0] += (repulsion_strength_wall / debye_length_wall) * np.exp(-(h[lr_mask]-blob_radius)/debye_length_wall)
  f[sr_mask,0] += (repulsion_strength_wall / debye_length_wall)

  # Add interactions with Lx_plus wall
  h = Lx_plus - r_vectors[:,0]
  lr_mask = h > blob_radius
  sr_mask = h <= blob_radius
  f[lr_mask,0] -= (repulsion_strength_wall / debye_length_wall) * np.exp(-(h[lr_mask]-blob_radius)/debye_length_wall)
  f[sr_mask,0] -= (repulsion_strength_wall / debye_length_wall)


  # Add interactions with Ly_minus wall
  h = r_vectors[:,1] - Ly_minus
  lr_mask = h > blob_radius
  sr_mask = h <= blob_radius
  f[lr_mask,1] += (repulsion_strength_wall / debye_length_wall) * np.exp(-(h[lr_mask]-blob_radius)/debye_length_wall)
  f[sr_mask,1] += (repulsion_strength_wall / debye_length_wall)

  # Add interactions with Ly_plus wall
  h = Ly_plus - r_vectors[:,1]
  lr_mask = h > blob_radius
  sr_mask = h <= blob_radius
  f[lr_mask,1] -= (repulsion_strength_wall / debye_length_wall) * np.exp(-(h[lr_mask]-blob_radius)/debye_length_wall)
  f[sr_mask,1] -= (repulsion_strength_wall / debye_length_wall)

  # Add interactions with Lz_minus wall
  h = r_vectors[:,2] - Lz_minus
  lr_mask = h > blob_radius
  sr_mask = h <= blob_radius
  f[lr_mask,2] += (repulsion_strength_wall / debye_length_wall) * np.exp(-(h[lr_mask]-blob_radius)/debye_length_wall)
  f[sr_mask,2] += (repulsion_strength_wall / debye_length_wall)

  # Add interactions with Lz_plus wall
  h = Lz_plus - r_vectors[:,2]
  lr_mask = h > blob_radius
  sr_mask = h <= blob_radius
  f[lr_mask,2] -= (repulsion_strength_wall / debye_length_wall) * np.exp(-(h[lr_mask]-blob_radius)/debye_length_wall)
  f[sr_mask,2] -= (repulsion_strength_wall / debye_length_wall)

  return f
multi_bodies_functions.blob_external_forces = blob_external_forces_new

