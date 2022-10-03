'''
Simple example of a structure with ghost blobs.
'''
from __future__ import division, print_function
import numexpr as ne
from mobility import mobility as mob
import multi_bodies_functions
from multi_bodies_functions import *

def set_slip_by_ID_new(body, slip, *args, **kwargs):
  '''
  This function assign a slip function to each body.
  If the body has an associated slip file the function
  "active_body_slip" is assigned (see function below).
  Otherwise the slip is set to zero.

  This function can be override to assign other slip
  functions based on the body ID, (ID of a structure
  is the name of the clones file (without .clones)).
  See the example in
  "examples/pair_active_rods/".
  '''
  if slip is not None:
    active_body_slip_partial = partial(active_body_slip_new, slip = slip)
    body.function_slip = active_body_slip_partial
  else:
    body.function_slip = default_zero_blobs
  return
multi_bodies_functions.set_slip_by_ID = set_slip_by_ID_new


def active_body_slip_new(body, slip):
  '''
  This function set the slip read from the *.slip file to the
  blobs. The slip on the file is given in the body reference 
  configuration (quaternion = (1,0,0,0)) therefore this
  function rotates the slip to the current body orientation.
  
  This function can be used, for example, to model active rods
  that propel along their axes. 
  '''
  # Get rotation matrix
  rotation_matrix = body.orientation.rotation_matrix()

  # Rotate  slip on each blob
  slip_rotated = np.empty((body.Nblobs, 3))
  for i in range(body.Nblobs):
    slip_rotated[i] = np.dot(rotation_matrix, slip[i])
  return slip_rotated
multi_bodies_functions.active_body_slip = active_body_slip_new


def get_ghost_blobs_r_vectors(bodies):
  '''
  Return coordinates of all the blobs with shape (Nblobs, 3).
  '''
  r_vectors = [] 
  for b in bodies:
    if hasattr(b, 'ghost_reference'):
      r_vectors.append(b.location + np.dot(b.ghost_reference, b.orientation.rotation_matrix().T))
  return np.vstack(r_vectors) if len(r_vectors) > 0 else np.zeros(0)


def get_ghost_blobs_forces(bodies):
  '''
  Return coordinates of all the blobs with shape (Nblobs, 3).
  '''
  blob_forces = [] 
  for b in bodies:
    if hasattr(b, 'ghost_reference_forces'):
      blob_forces.append(np.dot(b.ghost_reference_forces, b.orientation.rotation_matrix().T))
  return np.vstack(blob_forces) if len(blob_forces) > 0 else np.zeros(0)


def get_blobs_radius(bodies):
  radius_blobs = []
  for k, b in enumerate(bodies):
    radius_blobs.append(b.blobs_radius)
  radius_blobs = np.concatenate(radius_blobs, axis=0)
  return np.vstack(radius_blobs).flatten()


def get_ghosts_blobs_radius(bodies):
  radius_blobs = []
  for k, b in enumerate(bodies):
    if hasattr(b, 'ghost_blobs_radius'):
      radius_blobs.append(b.ghost_blobs_radius)
  radius_blobs = np.concatenate(radius_blobs, axis=0)    
  return np.vstack(radius_blobs).flatten() if len(radius_blobs) > 0 else np.zeros(0)
  

def calc_slip_new(bodies, Nblobs, *args, **kwargs):
  '''
  Function to calculate the slip in all the blobs.
  '''
  slip = np.zeros((Nblobs, 3))
  
  # Add prescribed slip 
  offset = 0
  for b in bodies:
    slip[offset:offset+b.Nblobs] = b.calc_slip()
    offset += b.Nblobs

  # Get ghost blobs forces
  blob_forces_ghosts = get_ghost_blobs_forces(bodies)
 
  # Calc flow perturbations
  if blob_forces_ghosts.size > 0:
    # Get ghost and real blobs
    r_vectors = get_blobs_r_vectors(bodies, Nblobs)
    r_vectors_ghosts = get_ghost_blobs_r_vectors(bodies)
    
    # Get blob radius
    radius_blobs = get_blobs_radius(bodies)
    radius_blobs_ghost = get_ghosts_blobs_radius(bodies)
    eta = kwargs.get('eta')

    slip = slip.flatten()
    if kwargs.get('implementation').find('no_wall') == -1:
      slip -= mob.single_wall_mobility_trans_times_force_source_target_numba(r_vectors_ghosts, r_vectors, blob_forces_ghosts,
                                                                             radius_blobs_ghost, radius_blobs, eta,
                                                                             periodic_length=kwargs.get('periodic_length'))
    else:
      slip -= mob.no_wall_mobility_trans_times_force_source_target_numba(r_vectors_ghosts, r_vectors, blob_forces_ghosts,
                                                                         radius_blobs_ghost, radius_blobs, eta,
                                                                         periodic_length=kwargs.get('periodic_length'))
  return slip
multi_bodies_functions.calc_slip = calc_slip_new


def bodies_external_force_torque_new(bodies, r_vectors, *args, **kwargs):
  '''
  This function returns the external force-torques acting on the bodies.
  It returns an array with shape (2*len(bodies), 3)
  '''
  force_torque = np.zeros((2*len(bodies), 3))

  for k, b in enumerate(bodies):
    if hasattr(b, 'ghost_force_torque'):
      force_torque[2*k] = -np.dot(b.ghost_force_torque[0], b.orientation.rotation_matrix().T)
      force_torque[2*k + 1] = -np.dot(b.ghost_force_torque[1], b.orientation.rotation_matrix().T)
  return force_torque
multi_bodies_functions.bodies_external_force_torque = bodies_external_force_torque_new

