''' Fluid Mobilities near a wall, from Swan and Brady's paper.'''

import numpy as np
import scipy.sparse
import scipy.spatial as scsp
import sys
import time
import imp
import general_application_utils as utils

# If pycuda is installed import mobility_pycuda
try:
  imp.find_module('pycuda')
  found_pycuda = True
except ImportError:
  found_pycuda = False
if found_pycuda:
  try:
    import pycuda.autoinit
    autoinit_pycuda = True
  except:
    autoinit_pycuda = False
  if autoinit_pycuda:
    try:
      from . import mobility_pycuda
    except ImportError:
      from .mobility import mobility_pycuda

# If numba is installed import mobility_numba
try: 
  imp.find_module('numba')
  found_numba = True
except ImportError:
  found_numba = False
if found_numba:
  from numba import njit, prange 
  try:
    from . import mobility_numba
  except ImportError:
    import mobility_numba

# Try to import the mobility fmm implementation
try:
  import mobility_fmm as fmm
except ImportError:
  pass
try:
  import mobility_cpp
except ImportError:
  try:
    from . import mobility_cpp
  except ImportError:
    pass

# Try to import stkfmm library
try:
  import PySTKFMM
except ImportError:
  pass
  

def shift_heights(r_vectors, blob_radius, *args, **kwargs):
  '''
  Return an array with the blobs' height

  z_effective = maximum(z, blob_radius)

  This function is used to compute positive
  definite mobilites for blobs close to the wall.
  '''
  r_effective = np.copy(r_vectors)
  r_effective[r_vectors[:,2] <= blob_radius, 2] = blob_radius

  return r_effective


def damping_matrix_B(r_vectors, blob_radius, *args, **kwargs):
  '''
  Return sparse diagonal matrix with components
  B_ii = 1.0               if z_i >= blob_radius
  B_ii = z_i / blob_radius if z_i < blob_radius

  It is used to compute positive definite mobilities
  close to the wall.
  '''
  B = np.ones(r_vectors.size)
  overlap = False
  for k, r in enumerate(r_vectors):
    if r[2] < blob_radius:
      B[k*3]     = r[2] / blob_radius
      B[k*3 + 1] = r[2] / blob_radius
      B[k*3 + 2] = r[2] / blob_radius
      overlap = True
  return (scipy.sparse.dia_matrix((B, 0), shape=(B.size, B.size)), overlap)


def shift_heights_different_radius(r_vectors, blob_radius, *args, **kwargs):
  '''
  Return an array with the blobs' height

  z_effective = maximum(z, blob_radius)

  This function is used to compute positive
  definite mobilites for blobs close to the wall.
  '''
  r_effective = np.copy(r_vectors)
  for k, r in enumerate(r_effective):
    r[2] = r[2] if r[2] > blob_radius[k] else blob_radius[k]
  return r_effective


def damping_matrix_B_different_radius(r_vectors, blob_radius, *args, **kwargs):
  '''
  Return sparse diagonal matrix with components
  B_ii = 1.0               if z_i >= blob_radius
  B_ii = z_i / blob_radius if z_i < blob_radius

  It is used to compute positive definite mobilities
  close to the wall.
  '''
  B = np.ones(r_vectors.size)
  overlap = False
  for k, r in enumerate(r_vectors):
    if r[2] < blob_radius[k]:
      B[k*3]     = r[2] / blob_radius[k]
      B[k*3 + 1] = r[2] / blob_radius[k]
      B[k*3 + 2] = r[2] / blob_radius[k]
      overlap = True
  return (scipy.sparse.dia_matrix((B, 0), shape=(B.size, B.size)), overlap)


def image_singular_stokeslet(r_vectors, eta, a, *args, **kwargs):
  ''' Calculate the image system for the singular stokeslet (M above).'''
  fluid_mobility = np.array([
      np.zeros(3*len(r_vectors)) for _ in range(3*len(r_vectors))])
  # Loop through particle interactions
  for j in range(len(r_vectors)):
    for k in range(len(r_vectors)):
      if j != k:  #  do particle interaction
        r_particles = r_vectors[j] - r_vectors[k]
        r_norm = np.linalg.norm(r_particles)
        wall_dist = r_vectors[k][2]
        r_reflect = r_vectors[j] - (r_vectors[k] - 2.*np.array([0., 0., wall_dist]))
        r_ref_norm = np.linalg.norm(r_reflect)
        # Loop through components.
        for l in range(3):
          for m in range(3):
            # Two stokeslets, one with negative force at image.
            fluid_mobility[j*3 + l][k*3 + m] = (
              ((l == m)*1./r_norm + r_particles[l]*r_particles[m]/(r_norm**3) -
               ((l == m)*1./r_ref_norm + r_reflect[l]*r_reflect[m]/(r_ref_norm**3)))/
              (8.*np.pi))
        # Add doublet and dipole contribution.
        fluid_mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] += (
          doublet_and_dipole(r_reflect, wall_dist))

      else:
        # j == k
        fluid_mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] = 1./(6*np.pi*eta*a)*np.identity(3)
  return fluid_mobility

def stokes_doublet(r, *args, **kwargs):
  ''' Calculate stokes doublet from direction, strength, and r. '''
  r_norm = np.linalg.norm(r)
  e3 = np.array([0., 0., 1.])
  doublet = (np.outer(r, e3) + np.dot(r, e3)*np.identity(3) -
             np.outer(e3, r) - 3.*np.dot(e3, r)*np.outer(r, r)/(r_norm**2))
  # Negate the first two columns for the correct forcing.
  doublet[:, 0:2] = -1.*doublet[:, 0:2]
  doublet = doublet/(8*np.pi*(r_norm**3))
  return doublet

def potential_dipole(r, *args, **kwargs):
  ''' Calculate potential dipole. '''
  r_norm = np.linalg.norm(r)
  dipole = np.identity(3) - 3.*np.outer(r, r)/(r_norm**2)
  # Negate the first two columns for the correct forcing.
  dipole[:, 0:2] = -1.*dipole[:, 0:2]
  dipole = dipole/(4.*np.pi*(r_norm**3))
  return dipole


def doublet_and_dipole(r, h, *args, **kwargs):
  '''
  Just keep the pieces of the potential dipole and the doublet
  that we need for the image system.  No point in calculating terms that will cancel.
  This function includes the prefactors of 2H and H**2.
  Seems to be significantly faster.
  '''
  r_norm = np.linalg.norm(r)
  e3 = np.array([0., 0., 1.])
  doublet_and_dipole = 2.*h*(np.outer(r, e3) - np.outer(e3, r))/(8.*np.pi*(r_norm**3))
  doublet_and_dipole[:, 0:2] = -1.*doublet_and_dipole[:, 0:2]
  return doublet_and_dipole


def single_wall_mobility_trans_times_force_cpp(r_vectors, force, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level by the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses mobility_cpp, a separate extention from mobility_ext that must be built.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = mobility_cpp.damping_matrix_B(r_vectors, a)

  # Compute B * force
  if overlap is True:
    force = B.dot(force.flatten())
  # Compute M_tilde * B * force
  velocities = mobility_cpp.single_wall_mobility_trans_times_force(r_vectors_effective, force, eta, a, L)
  # Compute B.T * M * B * vector
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities


def single_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level by the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses pycuda.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    force = B.dot(force.flatten())
  # Compute M_tilde * B * force
  velocities = mobility_pycuda.single_wall_mobility_trans_times_force_pycuda(r_vectors_effective, force, eta, a, *args, **kwargs)
  # Compute B.T * M * B * vector
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities

def in_plane_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level by the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function makes use of pycuda.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    force = B.dot(force)
  # Compute M_tilde * B * force
  velocities = mobility_pycuda.in_plane_mobility_trans_times_force_pycuda(r_vectors_effective, force, eta, a, *args, **kwargs)
  # Compute B.T * M * B * vector
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities


def no_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses pycuda.
  '''
  vel = mobility_pycuda.no_wall_mobility_trans_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs)
  return vel


def single_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.
  
  This function uses pycuda.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    force = B.dot(force.flatten())
  # Compute M_tilde * B * force
  rot = mobility_pycuda.single_wall_mobility_rot_times_force_pycuda(r_vectors_effective, force, eta, a, *args, **kwargs)
  # Compute B.T * M * B * force
  if overlap is True:
    rot = B.dot(rot)
  return rot


def no_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function uses pycuda.
  '''
  rot = mobility_pycuda.no_wall_mobility_rot_times_force_pycuda(r_vectors, force, eta, a, *args, **kwargs)
  return rot


def single_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses pycuda.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * vector
  if overlap is True:
    torque = B.dot(torque.flatten())
  # Compute M_tilde * B * torque
  rot = mobility_pycuda.single_wall_mobility_rot_times_torque_pycuda(r_vectors_effective, torque, eta, a, *args, **kwargs)
  # Compute B.T * M * B * torque
  if overlap is True:
    rot = B.dot(rot)
  return rot


def no_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function uses pycuda.
  '''
  rot = mobility_pycuda.no_wall_mobility_rot_times_torque_pycuda(r_vectors, torque, eta, a, *args, **kwargs)
  return rot


def single_wall_mobility_trans_times_force_torque_pycuda(r_vectors, force, torque, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses pycuda.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force, B * torque
  if overlap is True:
    force = B.dot(force)
    torque = B.dot(torque)
  # Compute M_tilde * B * (force + torque)
  velocities = mobility_pycuda.single_wall_mobility_trans_times_force_torque_pycuda(r_vectors_effective, force, torque, eta, a, *args, **kwargs)
  # Compute B.T * M * B * (force + torque)
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities


def no_wall_mobility_trans_times_force_torque_pycuda(r_vectors, force, torque, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function uses pycuda.
  '''
  velocities = mobility_pycuda.no_wall_mobility_trans_times_force_torque_pycuda(r_vectors, force, torque, eta, a, *args, **kwargs)
  return velocities


def single_wall_mobility_trans_times_torque_pycuda(r_vectors, torque, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses pycuda.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * torque
  if overlap is True:
    torque = B.dot(torque.flatten())
  # Compute M_tilde * B * torque
  velocities = mobility_pycuda.single_wall_mobility_trans_times_torque_pycuda(r_vectors_effective, torque, eta, a, *args, **kwargs)
  # Compute B.T * M * B * torque
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities

def in_plane_mobility_trans_times_torque_pycuda(r_vectors, torque, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function makes use of pycuda.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * torque
  if overlap is True:
    torque = B.dot(torque)
  # Compute M_tilde * B * torque
  velocities = mobility_pycuda.in_plane_mobility_trans_times_torque_pycuda(r_vectors_effective, torque, eta, a, *args, **kwargs)
  # Compute B.T * M * B * torque
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities


def no_wall_mobility_trans_times_torque_pycuda(r_vectors, force, eta, a, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level to the force
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses pycuda.
  '''
  vel = mobility_pycuda.no_wall_mobility_trans_times_torque_pycuda(r_vectors, force, eta, a, *args, **kwargs)
  return vel


def single_wall_mobility_trans_times_force_source_target_pycuda(source, target, force, radius_source, radius_target, eta, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level by the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses pycuda.
  '''
  # Compute effective heights
  x = shift_heights_different_radius(target, radius_target)
  y = shift_heights_different_radius(source, radius_source)

  # Compute dumping matrices
  B_target, overlap_target = damping_matrix_B_different_radius(target, radius_target, *args, **kwargs)
  B_source, overlap_source = damping_matrix_B_different_radius(source, radius_source, *args, **kwargs)

  # Compute B * force
  if overlap_source is True:
    force = B_source.dot(force.flatten())

  # Compute M_tilde * B * force
  velocities = mobility_pycuda.single_wall_mobility_trans_times_force_source_target_pycuda(y, x, force, radius_source, radius_target, eta, *args, **kwargs)

  # Compute B.T * M * B * vector
  if overlap_target is True:
    velocities = B_target.dot(velocities)
  return velocities


def single_wall_mobility_trans_times_force_source_target_numba(source, target, force, radius_source, radius_target, eta, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level by the force
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses numba.
  '''
  # Get domain size for Pseudo-PBC
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  
  # Compute effective heights
  x = shift_heights_different_radius(target, radius_target)
  y = shift_heights_different_radius(source, radius_source)

  # Compute dumping matrices
  B_target, overlap_target = damping_matrix_B_different_radius(target, radius_target, *args, **kwargs)
  B_source, overlap_source = damping_matrix_B_different_radius(source, radius_source, *args, **kwargs)

  # Compute B * force
  if overlap_source is True:
    force = B_source.dot(force.flatten())

  # Compute M_tilde * B * force
  velocities = mobility_numba.mobility_trans_times_force_source_target_numba(y, x, force, radius_source, radius_target, eta, L=L, wall=1)

  # Compute B.T * M * B * vector
  if overlap_target is True:
    velocities = B_target.dot(velocities)
  return velocities


def no_wall_mobility_trans_times_force_source_target_numba(source, target, force, radius_source, radius_target, eta, *args, **kwargs):
  '''
  Returns the product of the mobility at the blob level by the force
  on the blobs.
  Mobility for particles in unbounded domain.  

  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses numba.
  '''
  # Get domain size for Pseudo-PBC
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))

  # Compute M_tilde * B * force
  velocities = mobility_numba.mobility_trans_times_force_source_target_numba(source, target, force, radius_source, radius_target, eta, L=L, wall=0)

  return velocities


def single_wall_fluid_mobility_loops(r_vectors, eta, a, *args, **kwargs):
  ''' 
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the
  Blake paper point particle result.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  num_particles = len(r_vectors_effective)
  # We add the corrections from the appendix of the paper to the unbounded mobility.
  fluid_mobility = rotne_prager_tensor_loops(r_vectors_effective, eta, a)
  for j in range(num_particles):
    for k in range(j+1, num_particles):
      # Here notation is based on appendix C of the Swan and Brady paper:
      # 'Simulation of hydrodynamically interacting particles near a no-slip
      # boundary.'
      h = r_vectors_effective[k][2]
      R = (r_vectors_effective[j] - (r_vectors_effective[k] - 2.*np.array([0., 0., h])))/a
      R_norm = np.linalg.norm(R)
      e = R/R_norm
      e_3 = np.array([0., 0., e[2]])
      h_hat = h/(a*R[2])
      # Taken from Appendix C expression for M_UF
      fluid_mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] += (1./(6.*np.pi*eta*a))*(
        -0.25*(3.*(1. - 6.*h_hat*(1. - h_hat)*e[2]**2)/R_norm
               - 6.*(1. - 5.*e[2]**2)/(R_norm**3)
               + 10.*(1. - 7.*e[2]**2)/(R_norm**5))*np.outer(e, e)
         - (0.25*(3.*(1. + 2.*h_hat*(1. - h_hat)*e[2]**2)/R_norm
                  + 2.*(1. - 3.*e[2]**2)/(R_norm**3)
                  - 2.*(1. - 5.*e[2]**2)/(R_norm**5)))*np.identity(3)
         + 0.5*(3.*h_hat*(1. - 6.*(1. - h_hat)*e[2]**2)/R_norm
                - 6.*(1. - 5.*e[2]**2)/(R_norm**3)
                + 10.*(2. - 7.*e[2]**2)/(R_norm**5))*np.outer(e, e_3)
         + 0.5*(3.*h_hat/R_norm - 10./(R_norm**5))*np.outer(e_3, e)
         - (3.*(h_hat**2)*(e[2]**2)/R_norm
            + 3.*(e[2]**2)/(R_norm**3)
            + (2. - 15.*e[2]**2)/(R_norm**5))*np.outer(e_3, e_3)/(e[2]**2))

      fluid_mobility[(k*3):(k*3 + 3), (j*3):(j*3 + 3)] = (
        fluid_mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)].T)

  for j in range(len(r_vectors_effective)):
    # Diagonal blocks, self mobility.
    h = r_vectors_effective[j][2]/a
    for l in range(3):
      fluid_mobility[j*3 + l][j*3 + l] += (1./(6.*np.pi*eta*a))*(
        (l != 2)*(-1./16.)*(9./h - 2./(h**3) + 1./(h**5))
        + (l == 2)*(-1./8.)*(9./h - 4./(h**3) + 1./(h**5)))

  # Compute M = B^T * M_tilde * B
  if overlap is True:
    return B.dot( (B.dot(fluid_mobility.T)).T )
  else:
    return fluid_mobility


def rotne_prager_tensor_loops(r_vectors, eta, a, *args, **kwargs):
  ''' 
  Calculate free rotne prager tensor for particles at locations given by
  r_vectors (list of 3 dimensional locations) of radius a.
  '''
  num_particles = len(r_vectors)
  fluid_mobility = np.array([np.zeros(3*num_particles) for _ in range(3*num_particles)])
  for j in range(num_particles):
    for k in range(num_particles):
      if j != k:
        # Particle interaction, rotne prager.
        r = r_vectors[j] - r_vectors[k]
        r_norm = np.linalg.norm(r)
        if r_norm > 2.*a:
          # Constants for far RPY tensor, taken from OverdampedIB paper.
          C1 = 3.*a/(4.*r_norm) + (a**3)/(2.*r_norm**3)
          C2 = 3.*a/(4.*r_norm) - (3.*a**3)/(2.*r_norm**3)

        elif r_norm <= 2.*a:
          # This is for the close interaction,
          # Call C3 -> C1 and C4 -> C2
          C1 = 1 - 9.*r_norm/(32.*a)
          C2 = 3*r_norm/(32.*a)
        fluid_mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] = (1./(6.*np.pi*eta*a)*(
          C1*np.identity(3) + C2*np.outer(r, r)/(np.maximum(r_norm, np.finfo(float).eps)**2)))

      elif j == k:
        # j == k, diagonal block.
        fluid_mobility[(j*3):(j*3 + 3), (k*3):(k*3 + 3)] = ((1./(6.*np.pi*eta*a)) * np.identity(3))
  return fluid_mobility


def single_wall_fluid_mobility_product(r_vectors, vector, eta, a, *args, **kwargs):
  '''
  WARNING: pseudo-PBC are not implemented for this function.

  Product (Mobility * vector). Mobility for particles near a wall.
  This uses the expression from the Swan and Brady paper for a finite
  size particle, as opposed to the Blake paper point particle result.

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.
  '''
  mobility = single_wall_fluid_mobility(np.reshape(r_vectors, (r_vectors.size // 3, 3)), eta, a)
  velocities = np.dot(mobility, vector)
  return velocities


def no_wall_fluid_mobility_product(r_vectors, vector, eta, a, *args, **kwargs):
  '''
  WARNING: pseudo-PBC are not implemented for this function.

  Product (Mobility * vector). Mobility for particles in an unbounded domain.
  This uses the standard Rotne-Prager-Yamakawa expression.
  '''
  mobility = rotne_prager_tensor(np.reshape(r_vectors, (r_vectors.size // 3, 3)), eta, a)
  velocities = np.dot(mobility, vector)
  return velocities


def single_wall_self_mobility_with_rotation(location, eta, a, *args, **kwargs):
  '''
  Self mobility for a single sphere of radius a with translation rotation
  coupling.  Returns the 6x6 matrix taking force and torque to
  velocity and angular velocity.
  This expression is taken from Swan and Brady's paper:
  '''
  h = location[2]/a
  fluid_mobility = (1./(6.*np.pi*eta*a))*np.identity(3)
  zero_matrix = np.zeros([3, 3])
  fluid_mobility = np.concatenate([fluid_mobility, zero_matrix])
  zero_matrix = np.zeros([6, 3])
  fluid_mobility = np.concatenate([fluid_mobility, zero_matrix], axis=1)
  # First the translation-translation block.
  for l in range(3):
    for m in range(3):
      fluid_mobility[l][m] += (1./(6.*np.pi*eta*a))*(
        (l == m)*(l != 2)*(-1./16.)*(9./h - 2./(h**3) + 1./(h**5))
        + (l == m)*(l == 2)*(-1./8.)*(9./h - 4./(h**3) + 1./(h**5)))
  # Translation-Rotation blocks.
  for l in range(3):
    for m in range(3):
      fluid_mobility[3 + l][m] += (1./(6.*np.pi*eta*a*a))*((3./32.)*
                                     (h**(-4))*epsilon_tensor(2, l, m))
      fluid_mobility[m][3 + l] += fluid_mobility[3 + l][m]

  # Rotation-Rotation block.
  for l in range(3):
    for m in range(3):
      fluid_mobility[3 + l][3 + m] += (
        (1./(8.*np.pi*eta*(a**3)))*(l == m) - ((1./(6.*np.pi*eta*(a**3)))*(
                                      (15./64.)*(h**(-3))*(l == m)*(l != 2)
                                      + (3./32.)*(h**(-3))*(m == 2)*(l == 2))))
  return fluid_mobility


def fmm_single_wall_stokeslet(r_vectors, force, eta, a, *args, **kwargs):
  '''
  WARNING: pseudo-PBC are not implemented for this function.

  Compute the Stokeslet interaction plus self mobility
  II/(6*pi*eta*a) in the presence of a wall at z=0.
  It uses the fmm implemented in the library stfmm3d.
  Must compile mobility_fmm.f90 before this will work
  (see Makefile).

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    force = B.dot(force)
  # Compute M_tilde * B * vector
  num_particles = r_vectors.size // 3
  ier = 0
  iprec = 5
  r_vectors_fortran = np.copy(r_vectors_effective.T, order='F')
  force_fortran = np.copy(np.reshape(force, (num_particles, 3)).T, order='F')
  u_fortran = np.empty_like(r_vectors_fortran, order='F')
  fmm.fmm_stokeslet_half(r_vectors_fortran, force_fortran, u_fortran, ier, iprec, a, eta, num_particles)
  # Compute B.T * M * B * force
  if overlap is True:
    return B.dot(np.reshape(u_fortran.T, u_fortran.size))
  else:
    return np.reshape(u_fortran.T, u_fortran.size)


def fmm_rpy(r_vectors, force, eta, a, *args, **kwargs):
  '''
  WARNING: pseudo-PBC are not implemented for this function.

  Compute the Stokes interaction using the Rotner-Prager
  tensor. Here there is no wall.
  It uses the fmm implemented in the library rpyfmm.
  Must compile mobility_fmm.f90 before this will work
  (see Makefile).
  '''
  num_particles = r_vectors.size // 3
  ier = 0
  iprec = 1
  r_vectors_fortran = np.copy(r_vectors.T, order='F')
  force_fortran = np.copy(np.reshape(force, (num_particles, 3)).T, order='F')
  u_fortran = np.empty_like(r_vectors_fortran, order='F')
  fmm.fmm_rpy(r_vectors_fortran, force_fortran, u_fortran, ier, iprec, a, eta, num_particles)
  return np.reshape(u_fortran.T, u_fortran.size)


def mobility_vector_product_source_target_one_wall(source, target, force, radius_source, radius_target, eta, *args, **kwargs):
  '''
  WARNING: pseudo-PBC are not implemented for this function.

  Compute velocity of targets of radius radius_target due
  to forces on sources of radius source_target in half-space.

  That is, compute the matrix vector product
  velocities_target = M_tt * forces_sources
  where M_tt has dimensions (target, source)
  '''
  # Compute effective heights
  x = shift_heights_different_radius(target, radius_target)
  y = shift_heights_different_radius(source, radius_source)

  # Compute dumping matrices
  B_target, overlap_target = damping_matrix_B_different_radius(target, radius_target, *args, **kwargs)
  B_source, overlap_source = damping_matrix_B_different_radius(source, radius_source, *args, **kwargs)

  # Compute B * vector
  if overlap_source is True:
    force = B_source.dot(force.flatten())

  # Compute unbounded contribution
  force = np.reshape(force, (force.size // 3, 3))
  velocity = mobility_vector_product_source_target_unbounded(y, x, force, radius_source, radius_target, eta, *args, **kwargs)
  y_image = np.copy(y)
  y_image[:,2] = -y[:,2]

  # Compute wall correction
  prefactor = 1.0 / (8 * np.pi * eta)
  b2 = radius_target**2
  a2 = radius_source**2
  I = np.eye(3)
  J = np.zeros((3,3))
  J[2,2] = 1.0
  P = np.eye(3)
  P[2,2] = -1.0
  delta_3 = np.zeros(3)
  delta_3[2] = 1.0
  # Loop over targets
  for i, r_target in enumerate(x):
    # Distance between target and image sources
    r_source_to_target = r_target - y_image
    x3 = np.zeros(3)
    x3[2] = r_target[2]
    # Loop over sources
    for j, r in enumerate(r_source_to_target):
      y3 = np.zeros(3)
      y3[2] = y[j,2]
      r2 = np.dot(r,r)
      r_norm  = np.sqrt(r2)
      r3 = r_norm * r2
      r5 = r3 * r2
      r7 = r5 * r2
      r9 = r7 * r2
      RR = np.outer(r,r)
      R3 = delta_3 * r[2]

      # Compute 3x3 block mobility
      Mij = ((1+(b2[i]+a2[j])/(3.0*r2)) * I + (1-(b2[i]+a2[j])/r2) * RR / r2) / r_norm
      Mij += 2*(-J/r_norm - np.outer(r,x3)/r3 - np.outer(y3,r)/r3 + x3[2]*y3[2]*(I/r3 - 3*RR/r5))
      Mij += (2*b2[i]/3.0) * (-J/r3 + 3*np.outer(r,R3)/r5 - y3[2]*(3*R3[2]*I/r5 + 3*np.outer(delta_3,r)/r5 + 3*np.outer(r,delta_3)/r5 - 15*R3[2]*RR/r7))
      Mij += (2*a2[j]/3.0) * (-J/r3 + 3*np.outer(R3,r)/r5 - x3[2]*(3*R3[2]*I/r5 + 3*np.outer(delta_3,r)/r5 + 3*np.outer(r,delta_3)/r5 - 15*R3[2]*RR/r7))
      Mij += (2*b2[i]*a2[j]/3.0) * (-I/r5 + 5*R3[2]*R3[2]*I/r7 - J/r5 + 5*np.outer(R3,r)/r7 - J/r5 + 5*np.outer(r,R3)/r7 + 5*np.outer(R3,r)/r7 + 5*RR/r7 + 5*np.outer(r,R3)/r7 -35 * R3[2]*R3[2]*RR/r9)
      Mij = -prefactor * np.dot(Mij, P)
      velocity[i] += np.dot(Mij, force[j])

  # Compute B.T * M * B * vector
  if overlap_target is True:
    velocity = B_target.dot(np.reshape(velocity, velocity.size))

  return velocity


def mobility_vector_product_source_target_unbounded(source, target, force, radius_source, radius_target, eta, *args, **kwargs):
  '''
  WARNING: pseudo-PBC are not implemented for this function.

  Compute velocity of targets of radius radius_target due
  to forces on sources of radius source_targer in unbounded domain.

  That is, compute the matrix vector product
  velocities_target = M_tt * forces_sources
  where M_tt has dimensions (target, source)

  See Reference P. J. Zuk et al. J. Fluid Mech. (2014), vol. 741, R5, doi:10.1017/jfm.2013.668
  '''
  force = np.reshape(force, (force.size // 3, 3))
  velocity = np.zeros((target.size // 3, 3))
  prefactor = 1.0 / (8 * np.pi * eta)
  b2 = radius_target**2
  a2 = radius_source**2
  # Loop over targets
  for i, r_target in enumerate(target):
    # Distance between target and sources
    r_source_to_target = r_target - source
    # Loop over sources
    for j, r in enumerate(r_source_to_target):
      r2 = np.dot(r,r)
      r_norm  = np.sqrt(r2)
      # Compute 3x3 block mobility
      if r_norm >= (radius_target[i] + radius_source[j]):
        Mij = (1 + (b2[i]+a2[j]) / (3 * r2)) * np.eye(3) + (1 - (b2[i]+a2[j]) / r2) * np.outer(r,r) / r2
        Mij = (prefactor / r_norm) * Mij
      elif r_norm > np.absolute(radius_target[i]-radius_source[j]):
        r3 = r_norm * r2
        Mij = ((16*(radius_target[i]+radius_source[j])*r3 - ((radius_target[i]-radius_source[j])**2 + 3*r2)**2) / (32*r3)) * np.eye(3) +\
            ((3*((radius_target[i]-radius_source[j])**2-r2)**2) / (32*r3)) * np.outer(r,r) / r2
        Mij = Mij / (6 * np.pi * eta * radius_target[i] * radius_source[j])
      else:
        largest_radius = radius_target[i] if radius_target[i] > radius_source[j] else radius_source[j]
        Mij = (1.0 / (6 * np.pi * eta * largest_radius)) * np.eye(3)
      velocity[i] += np.dot(Mij, force[j])

  return velocity


def epsilon_tensor(i, j, k):
  '''
  Epsilon tensor (cross product).  Only works for arguments
  between 0 and 2.
  '''
  if j == ((i + 1) % 3) and k == ((j+1) % 3):
    return 1.
  elif i == ((j + 1) % 3) and j == ((k + 1) % 3):
    return -1.
  else:
    return 0.

def rotne_prager_tensor_cpp(r_vectors, eta, a, *args, **kwargs):
  ''' 
  Calculate free rotne prager tensor for particles at locations given by
  r_vectors of radius a.
  '''
  return mobility_cpp.rotne_prager_tensor(r_vectors, eta, a)

def rotne_prager_tensor(r_vectors, eta, a, *args, **kwargs):
  ''' 
  Calculate free rotne prager tensor for particles at locations given by
  r_vectors of radius a.
  '''
  # Extract variables
  r_vectors = r_vectors.reshape((r_vectors.size // 3, 3))
  x = r_vectors[:,0]
  y = r_vectors[:,1]
  z = r_vectors[:,2]
  
  # Compute distances between blobs
  dx = x - x[:, None]
  dy = y - y[:, None]
  dz = z - z[:, None]
  dr = np.sqrt(dx**2 + dy**2 + dz**2)

  # Compute scalar functions f(r) and g(r)
  factor = 1.0 / (6.0 * np.pi * eta)
  fr = np.zeros_like(dr)
  gr = np.zeros_like(dr)
  sel = dr > 2.0 * a
  nsel = np.logical_not(sel)
  sel_zero = dr == 0.
  nsel[sel_zero] = False

  fr[sel] = factor * (0.75 / dr[sel] + a**2 / (2.0 * dr[sel]**3))
  gr[sel] = factor * (0.75 / dr[sel]**3 - 1.5 * a**2 / dr[sel]**5)

  fr[sel_zero] = (factor / a)
  fr[nsel] = factor * (1.0 / a - 0.28125 * dr[nsel] / a**2)
  gr[nsel] = factor * (3.0 / (32.0 * a**2 * dr[nsel]))

  # Build mobility matrix of size 3N \times 3N
  M = np.zeros((r_vectors.size, r_vectors.size))
  M[0::3, 0::3] = fr + gr * dx * dx
  M[0::3, 1::3] =      gr * dx * dy
  M[0::3, 2::3] =      gr * dx * dz

  M[1::3, 0::3] =      gr * dy * dx
  M[1::3, 1::3] = fr + gr * dy * dy
  M[1::3, 2::3] =      gr * dy * dz

  M[2::3, 0::3] =      gr * dz * dx
  M[2::3, 1::3] =      gr * dz * dy
  M[2::3, 2::3] = fr + gr * dz * dz
  return M

def single_wall_fluid_mobility_cpp(r_vectors, eta, a):
  return mobility_cpp.single_wall_fluid_mobility(r_vectors, eta, a)

def single_wall_fluid_mobility(r_vectors, eta, a, *args, **kwargs):
  ''' 
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.
  '''
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B_damp, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  num_particles = len(r_vectors_effective)
  # We add the corrections from the appendix of the paper to the unbounded mobility.
  if 'mobility_cpp' in sys.modules:
    fluid_mobility = mobility_cpp.rotne_prager_tensor(r_vectors_effective, eta, a)
  else:
    fluid_mobility = rotne_prager_tensor(r_vectors_effective, eta, a)


  # Extract variables
  r_vectors_effective = r_vectors_effective.reshape((r_vectors_effective.size // 3, 3))
  N = r_vectors.size // 3
  x = r_vectors_effective[:,0]
  y = r_vectors_effective[:,1]
  z = r_vectors_effective[:,2]
  
  # Compute distances between blobs
  dx = (x[:, None] - x) / a
  dy = (y[:, None] - y) / a
  dz = (z[:, None] + z) / a
  dr = np.sqrt(dx**2 + dy**2 + dz**2)
  h_hat = z[:, None] / (a * dz)
  h = z / a
  ex = dx / dr
  ey = dy / dr
  ez = dz / dr

  # Compute scalar functions, the mobility is
  # M = A*delta_ij + B*e_i*e_j + C*e_i*delta_3j + D*delta_i3*e_j + E*delta_i3*delta_3j 
  factor = 1.0 / (6.0 * np.pi * eta * a)  
  sel = dr > 1e-12 * a
  rows, columns = np.diag_indices(N)
  sel[rows, columns] = False

  # Allocate memory
  M = np.zeros((r_vectors.size, r_vectors.size)) 
  rows, columns = np.diag_indices(r_vectors.size) 
  
  A = np.zeros_like(dr)
  B = np.zeros_like(dr)
  C = np.zeros_like(dr)
  D = np.zeros_like(dr)
  E = np.zeros_like(dr)


  # Self-mobility terms
  A_vec = -0.0625 * (9.0 / h - 2.0 / h**3 + 1.0 / h**5)
  E_vec = -A_vec - 0.125 * (9.0 / h - 4.0 / h**3 + 1.0 / h**5)
  M[rows[0::3], columns[0::3]] += A_vec
  M[rows[1::3], columns[1::3]] += A_vec
  M[rows[2::3], columns[2::3]] += A_vec + E_vec

  # Particle-particle terms
  A[sel] = -0.25 * (3.0 * (1.0 + 2*h_hat[sel]*(1.0-h_hat[sel])*ez[sel]**2) / dr[sel] + 
                    2.0*(1-3.0*ez[sel]**2) / dr[sel]**3 - 2.0*(1-5.0*ez[sel]**2) / dr[sel]**5)

  B[sel] = -0.25 * (3.0 * (1.0 - 6.0*h_hat[sel]*(1.0-h_hat[sel])*ez[sel]**2) / dr[sel] - 
                    6.0 * (1.0 - 5.0*ez[sel]**2) / dr[sel]**3 + 10.0 * (1.0 - 7.0*ez[sel]**2) / dr[sel]**5)

  C[sel] = 0.5 * ez[sel] * (3.0 * h_hat[sel]*(1.0 - 6.0*(1.0-h_hat[sel])*ez[sel]**2) / dr[sel] - 
                            6.0 * (1.0 - 5.0*ez[sel]**2) / dr[sel]**3 + 10.0 * (2.0 - 7.0*ez[sel]**2) / dr[sel]**5)

  D[sel] = 0.5 * ez[sel] * (3.0 * h_hat[sel] / dr[sel] - 10.0 / dr[sel]**5)

  E[sel] = -(3.0 * h_hat[sel]**2 * ez[sel]**2 / dr[sel] + 3.0 * ez[sel]**2 / dr[sel]**3 + (2.0 - 15.0*ez[sel]**2) / dr[sel]**5)
  
  # Build mobility matrix of size 3N \times 3N
  M[0::3, 0::3] += A + B * ex * ex 
  M[0::3, 1::3] +=     B * ex * ey
  M[0::3, 2::3] +=     B * ex * ez + C.T * ex

  M[1::3, 0::3] +=     B * ey * ex
  M[1::3, 1::3] += A + B * ey * ey
  M[1::3, 2::3] +=     B * ey * ez + C.T * ey 

  M[2::3, 0::3] +=     B * ez * ex            + D.T * ex 
  M[2::3, 1::3] +=     B * ez * ey            + D.T * ey 
  M[2::3, 2::3] += A + B * ez * ez + C.T * ez + D.T * ez + E.T

  M *= factor
  M += fluid_mobility

  # Compute M = B^T * M_tilde * B
  if overlap is True:
    return B_damp.dot( (B_damp.dot(M.T)).T )
  else:
    return M


def no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  vel = mobility_numba.no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, L)
  return vel


def single_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level by the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
   
  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes. 

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    force = B.dot(force.flatten())
  # Compute M_tilde * B * force
  velocities = mobility_numba.single_wall_mobility_trans_times_force_numba(r_vectors_effective, force, eta, a, L)
  # Compute B.T * M * B * vector
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities


def in_plane_mobility_trans_times_force_numba(r_vectors, force, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level by the force 
  on the blobs.
  Mobility for particles near a wall, fixed in a plane.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
   
  If a component of periodic_length is larger than zero the
  space is assume to be pseudo-periodic in that direction. In that case
  the code will compute the interactions M*f between particles in
  the minimal image convection and also in the first neighbor boxes. 

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    force = B.dot(force.flatten())
  # Compute M_tilde * B * force
  velocities = mobility_numba.in_plane_mobility_trans_times_force_numba(r_vectors_effective, force, eta, a, L)
  # Compute B.T * M * B * vector
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities


def no_wall_mobility_trans_times_torque_numba(r_vectors, torque, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs. Mobility for particles in an unbounded domain, it uses
  the standard RPY tensor.  
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  vel = mobility_numba.no_wall_mobility_trans_times_torque_numba(r_vectors, torque, eta, a, L)
  return vel


def single_wall_mobility_trans_times_torque_numba(r_vectors, torque, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs. Mobility for particles on top of an infinite wall.
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    torque = B.dot(torque.flatten())
  # Compute M_tilde * B * force
  velocities = mobility_numba.single_wall_mobility_trans_times_torque_numba(r_vectors_effective, torque, eta, a, L)
  # Compute B.T * M * B * vector
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities

def in_plane_mobility_trans_times_torque_numba(r_vectors, torque, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs. Mobility for particles on top of an infinite wall.
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    torque = B.dot(torque.flatten())
  # Compute M_tilde * B * force
  velocities = mobility_numba.in_plane_mobility_trans_times_torque_numba(r_vectors_effective, torque, eta, a, L)
  # Compute B.T * M * B * vector
  if overlap is True:
    velocities = B.dot(velocities)
  return velocities


def no_wall_mobility_rot_times_force_numba(r_vectors, force, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility rotational-translation at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  rot = mobility_numba.no_wall_mobility_rot_times_force_numba(r_vectors, force, eta, a, L)
  return rot


def single_wall_mobility_rot_times_force_numba(r_vectors, force, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 

  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.
  
  This function uses pycuda.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * force
  if overlap is True:
    force = B.dot(force.flatten())
  # Compute M_tilde * B * force
  rot = mobility_numba.single_wall_mobility_rot_times_force_numba(r_vectors_effective, force, eta, a, L)
  # Compute B.T * M * B * force
  if overlap is True:
    rot = B.dot(rot)
  return rot


def no_wall_mobility_rot_times_torque_numba(r_vectors, torque, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  rot = mobility_numba.no_wall_mobility_rot_times_torque_numba(r_vectors, torque, eta, a, L)
  return rot


def single_wall_mobility_rot_times_torque_numba(r_vectors, torque, eta, a, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs.
  Mobility for particles near a wall.  This uses the expression from
  the Swan and Brady paper for a finite size particle, as opposed to the 
  Blake paper point particle result. 
  
  For blobs overlaping the wall we use
  Compute M = B^T * M_tilde(z_effective) * B.

  This function uses pycuda.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  # Get effective height
  r_vectors_effective = shift_heights(r_vectors, a)
  # Compute damping matrix B
  B, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
  # Compute B * vector
  if overlap is True:
    torque = B.dot(torque.flatten())
  # Compute M_tilde * B * torque
  rot = mobility_numba.single_wall_mobility_rot_times_torque_numba(r_vectors_effective, torque, eta, a, L)
  # Compute B.T * M * B * torque
  if overlap is True:
    rot = B.dot(rot)
  return rot



def no_wall_pressure_Stokeslet_numba(source, target, force, *args, **kwargs):
  ''' 
  Returns the pressure created by Stokeslets located at source in the positions
  of the targets. The space is unbounded.
  
  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  p = mobility_numba.no_wall_pressure_Stokeslet_numba(source, target, force, L)
  return p


def single_wall_pressure_Stokeslet_numba(source, target, force, *args, **kwargs):
  ''' 
  Returns the pressure created by Stokeslets located at source in the positions
  of the targets. Stokeslets above an infinite no-slip wall.

  This function uses numba.
  '''
  L = kwargs.get('periodic_length', np.array([0.0, 0.0, 0.0]))
  p = mobility_numba.single_wall_pressure_Stokeslet_numba(source, target, force, L)
  return p


def mobility_radii_trans_times_force(r_vectors, force, eta, a, radius_blobs, function, *args, **kwargs): 
  '''
  Mobility vector product M*f with blobs with different radii.
  function should provide the appropiate implementation (python, numba, pycuda, above a wall or unbounded...).
  '''
  return function(r_vectors, r_vectors, force, radius_blobs, radius_blobs, eta, *args, **kwargs)


def no_wall_double_layer_source_target_numba(source, target, normals, vector, weights, *args, **kwargs):
  '''
  Returns the product of the second layer operator with a vector.
  The diagonal terms are set to zero.

  This function uses numba.
  '''
  # Compute M_tilde * B * force
  velocities = mobility_numba.double_layer_source_target_numba(source, target, normals, vector, weights)

  return velocities


@njit(parallel=True, fastmath=True)
def no_wall_mobility_trans_times_force_overlap_correction_numba(r_vectors, force, eta, a, list_of_neighbors, offsets, L=np.array([0., 0., 0.])):
  ''' 
  Returns the blob-blob overlap correction for unbound fluids using the
  RPY mobility. It subtract the uncorrected value for r<2*a and it adds
  the corrected value.

  This function uses numba.
  '''
  # Variables
  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape(N, 3)
  force = force.reshape(N, 3)
  u = np.zeros((N, 3))
  fourOverThree = 4.0 / 3.0
  inva = 1.0 / a
  norm_fact_f = 1.0 / (8.0 * np.pi * eta * a)

  Lx = L[0]
  Ly = L[1]
  Lz = L[2]
  
  rx_vec = np.copy(r_vectors[:,0])
  ry_vec = np.copy(r_vectors[:,1])
  rz_vec = np.copy(r_vectors[:,2])
  fx_vec = np.copy(force[:,0])
  fy_vec = np.copy(force[:,1])
  fz_vec = np.copy(force[:,2])
  
  # Loop over image boxes and then over particles
  for i in prange(N):
    rxi = rx_vec[i]
    ryi = ry_vec[i]
    rzi = rz_vec[i]
    ux = 0
    uy = 0
    uz = 0
    for k in range(offsets[i+1] - offsets[i]):
      j = list_of_neighbors[offsets[i] + k]
      if i == j:
        continue
      # Compute vector between particles i and j
      rx = rxi - rx_vec[j]
      ry = ryi - ry_vec[j]
      rz = rzi - rz_vec[j]

      # PBC
      if Lx > 0:
        rx = rx - int(rx / Lx + 0.5 * (int(rx>0) - int(rx<0))) * Lx
      if Ly > 0:
        ry = ry - int(ry / Ly + 0.5 * (int(ry>0) - int(ry<0))) * Ly
      if Lz > 0:
        rz = rz - int(rz / Lz + 0.5 * (int(rz>0) - int(rz<0))) * Lz
      
      # Normalize distance with hydrodynamic radius
      rx = rx * inva 
      ry = ry * inva
      rz = rz * inva
      r2 = rx*rx + ry*ry + rz*rz
      r = np.sqrt(r2)
        
      # TODO: We should not divide by zero 
      invr = 1.0 / r
      invr2 = invr * invr
        
      if r > 2:
        Mxx = 0
        Mxy = 0
        Mxz = 0
        Myy = 0
        Myz = 0
        Mzz = 0
      else:
        c1 = fourOverThree * (1.0 - 0.28125 * r) # 9/32 = 0.28125
        c2 = fourOverThree * 0.09375 * invr      # 3/32 = 0.09375
        Mxx = c1 + c2 * rx*rx 
        Mxy =      c2 * rx*ry 
        Mxz =      c2 * rx*rz 
        Myy = c1 + c2 * ry*ry 
        Myz =      c2 * ry*rz 
        Mzz = c1 + c2 * rz*rz 
        c1 = 1.0 + 2.0 / (3.0 * r2)
        c2 = (1.0 - 2.0 * invr2) * invr2
        Mxx -= (c1 + c2*rx*rx) * invr
        Mxy -= (     c2*rx*ry) * invr
        Mxz -= (     c2*rx*rz) * invr
        Myy -= (c1 + c2*ry*ry) * invr
        Myz -= (     c2*ry*rz) * invr
        Mzz -= (c1 + c2*rz*rz) * invr                     
      Myx = Mxy
      Mzx = Mxz
      Mzy = Myz
	  
      # 2. Compute product M_ij * F_j           
      ux += (Mxx * fx_vec[j] + Mxy * fy_vec[j] + Mxz * fz_vec[j]) 
      uy += (Myx * fx_vec[j] + Myy * fy_vec[j] + Myz * fz_vec[j]) 
      uz += (Mzx * fx_vec[j] + Mzy * fy_vec[j] + Mzz * fz_vec[j]) 
    u[i,0] = ux * norm_fact_f
    u[i,1] = uy * norm_fact_f
    u[i,2] = uz * norm_fact_f          
  return u.flatten()


@utils.static_var('r_vectors_old', [])
@utils.static_var('list_of_neighbors', [])
@utils.static_var('offsets', [])
def mobility_trans_times_force_stkfmm(r, force, eta, a, rpy_fmm=None, L=np.array([0.,0.,0.]), wall=False, radius_blobs=None, *args, **kwargs):
  ''' 
  Returns the product of the mobility at the blob level to the force 
  on the blobs. Mobility for particles in an unbounded, semiperiodic or
  periodic domain. It uses the standard RPY tensor.
  
  This function uses the stkfmm library.
  '''
  def project_to_periodic_image(r, L):
    '''
    Project a vector r to the minimal image representation
    of size L=(Lx, Ly, Lz) and with a corner at (0,0,0). If 
    any dimension of L is equal or smaller than zero the 
    box is assumed to be infinite in that direction.
    
    If one dimension is not periodic shift all coordinates by min(r[:,i]) value.
    '''
    if L is not None:
      for i in range(3):
        if(L[i] > 0):
          r[:,i] = r[:,i] - (r[:,i] // L[i]) * L[i]
          # Careful with float precision
          sel = r[:,i] >= L[i]
          r[sel,i] = L[i] - 1e-13
        else:
          ri_min =  np.min(r[:,i])
          if ri_min < 0:
            r[:,i] -= ri_min

        # Careful with float precision
        sel = r[:,i] < 0
        r[sel,i] = 0
    return r

  # Prepare coordinates
  N = r.size // 3
  r_vectors = np.copy(r)
  r_vectors = project_to_periodic_image(r_vectors, L)
  if radius_blobs is None:
    radius_blobs = np.ones(N) * a
  a_max = np.max(radius_blobs)

  if wall:
    # Compute damping matrix B
    B_damp, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
    # Get effective height
    r_vectors = shift_heights(r_vectors, a)

    if overlap is True:
      force = B_damp.dot(force.flatten())

  # Set tree if necessary
  build_tree = True
  if len(mobility_trans_times_force_stkfmm.list_of_neighbors) > 0:
    if np.array_equal(mobility_trans_times_force_stkfmm.r_vectors_old, r_vectors):
      build_tree = False
      list_of_neighbors = mobility_trans_times_force_stkfmm.list_of_neighbors
      offsets = mobility_trans_times_force_stkfmm.offsets
  if build_tree:
    # Build tree in STKFMM
    if L[0] > 0:
      x_min = 0
      Lx_pvfmm = L[0]
      Lx_cKDTree = L[0]
    else:
      x_min = np.min(r_vectors[:,0])
      Lx_pvfmm = (np.max(r_vectors[:,0]) * 1.01 - x_min)
      Lx_cKDTree = (np.max(r_vectors[:,0]) * 1.01 - x_min) * 10
    if L[1] > 0:
      y_min = 0
      Ly_pvfmm = L[1]
      Ly_cKDTree = L[1]
    else:
      y_min = np.min(r_vectors[:,1])
      Ly_pvfmm = (np.max(r_vectors[:,1]) * 1.01 - y_min)
      Ly_cKDTree = (np.max(r_vectors[:,1]) * 1.01 - y_min) * 10
    if L[2] > 0:
      z_min = 0
      Lz_pvfmm = L[2]
      Lz_cKDTree = L[2]
    else:
      z_min = np.min(r_vectors[:,2])
      z_min = 0
      Lz_pvfmm = (np.max(r_vectors[:,2]) * 1.01 - z_min)
      Lz_cKDTree = (np.max(r_vectors[:,2]) * 1.01 - z_min) * 10

    # Set box size for pvfmm
    if L[0] > 0 or L[1] > 0 or L[2] > 0:
      L_box = np.max(L)
    else:
      L_box = np.max([Lx_pvfmm, Ly_pvfmm, 2 * Lz_pvfmm])

    # Buid FMM tree
    rpy_fmm.set_box(np.array([x_min, y_min, z_min]), L_box)
    rpy_fmm.set_points(r_vectors, r_vectors, np.zeros(0))
    rpy_fmm.setup_tree(PySTKFMM.KERNEL.RPY)

    # Build tree in python and neighbors lists
    d_max = 2 * a_max
    if L[0] > 0 or L[1] > 0 or L[2] > 0:
      boxsize = np.array([Lx_cKDTree, Ly_cKDTree, Lz_cKDTree])      
    else:
      boxsize = None
    tree = scsp.cKDTree(r_vectors, boxsize = boxsize)
    pairs = tree.query_ball_tree(tree, d_max)
    offsets = np.zeros(len(pairs)+1, dtype=int)
    for i in range(len(pairs)):
      offsets[i+1] = offsets[i] + len(pairs[i])
    list_of_neighbors = np.concatenate(pairs).ravel()
    mobility_trans_times_force_stkfmm.offsets = np.copy(offsets)
    mobility_trans_times_force_stkfmm.list_of_neighbors = np.copy(list_of_neighbors)
    mobility_trans_times_force_stkfmm.r_vectors_old = np.copy(r_vectors)

  # Set force with right format (single layer potential)
  trg_value = np.zeros((N, 6))
  src_SL_value = np.zeros((N, 4))
  src_SL_value[:,0:3] = np.copy(force.reshape((N, 3)))
  src_SL_value[:,3] = radius_blobs
    
  # Evaluate fmm; format p = trg_value[:,0], v = trg_value[:,1:4], Lap = trg_value[:,4:]
  rpy_fmm.clear_fmm(PySTKFMM.KERNEL.RPY)
  rpy_fmm.evaluate_fmm(PySTKFMM.KERNEL.RPY, src_SL_value, trg_value, np.zeros(0))
  comm = kwargs.get('comm')
  comm.Barrier()

  # Compute RPY mobility 
  # 1. Self mobility 
  vel = (1.0 / (6.0 * np.pi * eta * radius_blobs[:,None])) * force.reshape((N,3)) 
  # 2. Stokeslet 
  vel += trg_value[:,0:3] / (eta) 
  # 3. Laplacian 
  vel += (radius_blobs[:,None]**2 / (6.0 * eta)) * trg_value[:,3:] 
  # 4. Double Laplacian 
  #    it is zero with PBC 
  # 5. Add blob-blob overlap correction 
  # v_overlap = no_wall_mobility_trans_times_force_overlap_correction_numba(r_vectors, force, eta, a, list_of_neighbors, offsets, L=L, radius_blobs=radius_blobs) 
  v_overlap = no_wall_mobility_trans_times_force_overlap_correction_numba(r_vectors, force, eta, a, list_of_neighbors, offsets, L=L) 
  vel += v_overlap.reshape((N, 3)) 
  
  if wall:
    if overlap is True:
      # print('vel = ', vel.shape)
      vel = B_damp.dot(vel.flatten())
  
  return vel.flatten()


@utils.static_var('r_source_old', [])
@utils.static_var('r_target_old', [])
@utils.static_var('list_of_neighbors', [])
@utils.static_var('offsets', [])
def fluid_velocity_stkfmm(r_source, r_target, force, eta, a, rpy_fmm=None, L=np.array([0.,0.,0.]), wall=False, *args, **kwargs):
  ''' 
  Velocity of the fluid computed as

  v(x) = (1 + a**2/6 * Laplacian_y) * G(x,y) * F(y)   if r > a.
  v(x) = F(y) / (6 * pi * eta * a) if r <= a.

  with r = |x - y|
  
  This function uses the stkfmm library.
  '''
  def project_to_periodic_image(r1, r2, L):
    '''
    Project a vector r to the minimal image representation
    of size L=(Lx, Ly, Lz) and with a corner at (0,0,0). If 
    any dimension of L is equal or smaller than zero the 
    box is assumed to be infinite in that direction.
    
    If one dimension is not periodic shift all coordinates by min(r[:,i]) value.
    '''
    if L is not None:
      for i in range(3):
        if(L[i] > 0):
          r1[:,i] = r1[:,i] - (r1[:,i] // L[i]) * L[i]
          r2[:,i] = r2[:,i] - (r2[:,i] // L[i]) * L[i]
        else:
          ri_min =  min(np.min(r1[:,i]), np.min(r2[:,i]))
          if ri_min < 0:
            r1[:,i] -= ri_min
            r2[:,i] -= ri_min
    return r1, r2

  # Prepare coordinates
  N_source = r_source.size // 3
  N_target = r_target.size // 3
  r_source, r_target = project_to_periodic_image(np.copy(r_source), np.copy(r_target), L)

  if wall:
    # Compute damping matrix B
    B_damp, overlap = damping_matrix_B(r_vectors, a, *args, **kwargs)
    # Get effective height
    r_vectors = shift_heights(r_vectors, a)

    if overlap is True:
      force = B_damp.dot(force.flatten())

  # Set tree if necessary
  build_tree = True
  if len(fluid_velocity_stkfmm.list_of_neighbors) > 0:
    if np.array_equal(fluid_velocity_stkfmm.r_source_old, r_source) and np.array_equal(fluid_velocity_stkfmm.r_target_old, r_target):
      build_tree = False
      list_of_neighbors = fluid_velocity_stkfmm.list_of_neighbors
      offsets = fluid_velocity_stkfmm.offsets
  if build_tree:
    # Build tree in STKFMM
    if L[0] > 0:
      x_min = 0
      Lx_pvfmm = L[0]
      Lx_cKDTree = L[0]
    else:
      x_min = min(np.min(r_source[:,0]), np.min(r_target[:,0]))
      Lx_pvfmm = max(np.max(r_source[:,0]), np.max(r_target[:,0])) * 1.01 - x_min
      Lx_cKDTree = (max(np.max(r_source[:,0]), np.max(r_target[:,0])) * 1.01 - x_min) * 10
    if L[1] > 0:
      y_min = 0
      Ly_pvfmm = L[1]
      Ly_cKDTree = L[1]
    else:
      y_min = min(np.min(r_source[:,1]), np.min(r_target[:,1]))
      Ly_pvfmm = max(np.max(r_source[:,1]), np.max(r_target[:,1])) * 1.01 - y_min            
      Ly_cKDTree = (max(np.max(r_source[:,1]), np.max(r_target[:,1])) * 1.01 - y_min) * 10
    if L[2] > 0:
      z_min = 0
      Lz_pvfmm = L[2]
      Lz_cKDTree = L[2]
    else:
      z_min = 0
      Lz_pvfmm = max(np.max(r_source[:,2]), np.max(r_target[:,2])) * 1.01 - z_min
      Lz_cKDTree = (max(np.max(r_source[:,2]), np.max(r_target[:,2])) * 1.01 - z_min) * 10

    # Set box size for pvfmm
    if L[0] > 0 or L[1] > 0 or L[2] > 0:
      L_box = np.max(L)
    else:
      L_box = np.max([Lx_pvfmm, Ly_pvfmm, Lz_pvfmm])

    # Buid FMM tree
    rpy_fmm.set_box(np.array([x_min, y_min, z_min]), L_box)
    rpy_fmm.set_points(r_source, r_target, np.zeros(0))
    rpy_fmm.setup_tree(PySTKFMM.KERNEL.RPY)

    # Build tree in python and neighbors lists
    d_max = a
    if L[0] > 0 or L[1] > 0 or L[2] > 0:
      boxsize = np.array([Lx_cKDTree, Ly_cKDTree, Lz_cKDTree])      
    else:
      boxsize = None
    tree_source = scsp.cKDTree(r_source, boxsize = boxsize)
    tree_target = scsp.cKDTree(r_target, boxsize = boxsize)
    # pairs = tree_source.query_ball_tree(tree_target, d_max)
    pairs = tree_target.query_ball_tree(tree_source, d_max)
    offsets = np.zeros(len(pairs)+1, dtype=int)
    for i in range(len(pairs)):
      offsets[i+1] = offsets[i] + len(pairs[i])
    list_of_neighbors = np.concatenate(pairs).ravel().astype(int)
    fluid_velocity_stkfmm.offsets = np.copy(offsets)
    fluid_velocity_stkfmm.list_of_neighbors = np.copy(list_of_neighbors)
    fluid_velocity_stkfmm.r_source_old = np.copy(r_source)
    fluid_velocity_stkfmm.r_target_old = np.copy(r_target)

  # Set force with right format (single layer potential)
  trg_value = np.zeros((N_target, 6))
  src_SL_value = np.zeros((N_source, 4))
  src_SL_value[:,0:3] = np.copy(force.reshape((N_source, 3)))
  src_SL_value[:,3] = a
  if L[0] > 0 and L[2] == 0:
    # Neutral charge for PX and PXY
    src_SL_value[:,0:3] -= np.average(src_SL_value[:,0:3], axis=0)
    
  # Evaluate fmm; format p = trg_value[:,0], v = trg_value[:,1:4], Lap = trg_value[:,4:]
  rpy_fmm.clear_fmm(PySTKFMM.KERNEL.RPY)
  rpy_fmm.evaluate_fmm(PySTKFMM.KERNEL.RPY, src_SL_value, trg_value, np.zeros(0))
  comm = kwargs.get('comm')
  comm.Barrier()
 
  # Compute RPY mobility 
  # 1. Stokeslet 
  vel = trg_value[:,0:3] / (eta)
  if list_of_neighbors.size > 0:
    v_overlap = fluid_velocity_overlap_correction_numba(r_source, r_target, force, eta, a, list_of_neighbors, offsets, L=L) 
    vel += v_overlap.reshape((N_target, 3))    
  return vel.flatten()


@njit(parallel=True, fastmath=True)
def fluid_velocity_overlap_correction_numba(r_source, r_target, force, eta, a, list_of_neighbors, offsets, L=np.array([0., 0., 0.])):
  ''' 
  Returns the blob-blob overlap correction for unbound fluids using the
  RPY mobility. It subtract the uncorrected value for r<2*a and it adds
  the corrected value.

  This function uses numba.
  '''
  # Variables
  N_source = r_source.size // 3
  r_source = r_source.reshape(N_source, 3)
  N_target = r_target.size // 3
  r_taget = r_target.reshape(N_target, 3) 
  force = force.reshape(N_source, 3)
  u = np.zeros((N_target, 3))
  fourOverThree = 4.0 / 3.0
  inva = 1.0 / a
  norm_fact_f = 1.0 / (8.0 * np.pi * eta * a)

  Lx = L[0]
  Ly = L[1]
  Lz = L[2]
  
  rx_src = np.copy(r_source[:,0])
  ry_src = np.copy(r_source[:,1])
  rz_src = np.copy(r_source[:,2])
  rx_trg = np.copy(r_target[:,0])
  ry_trg = np.copy(r_target[:,1])
  rz_trg = np.copy(r_target[:,2])
  fx_vec = np.copy(force[:,0])
  fy_vec = np.copy(force[:,1])
  fz_vec = np.copy(force[:,2])
  
  # Loop over image boxes and then over particles
  for i in prange(N_target):
    rxi = rx_trg[i]
    ryi = ry_trg[i]
    rzi = rz_trg[i]
    ux = 0
    uy = 0
    uz = 0
    for k in range(offsets[i+1] - offsets[i]):
      j = list_of_neighbors[offsets[i] + k]
      # Compute vector between particles i and j
      rx = rxi - rx_src[j]
      ry = ryi - ry_src[j]
      rz = rzi - rz_src[j]

      # PBC
      if Lx > 0:
        rx = rx - int(rx / Lx + 0.5 * (int(rx>0) - int(rx<0))) * Lx
      if Ly > 0:
        ry = ry - int(ry / Ly + 0.5 * (int(ry>0) - int(ry<0))) * Ly
      if Lz > 0:
        rz = rz - int(rz / Lz + 0.5 * (int(rz>0) - int(rz<0))) * Lz
      
      # Normalize distance with hydrodynamic radius
      rx = rx * inva 
      ry = ry * inva
      rz = rz * inva
      r2 = rx*rx + ry*ry + rz*rz
      r = np.sqrt(r2)
        
      # TODO: We should not divide by zero 
      invr = 1.0 / r
      invr2 = invr * invr
        
      if r > 1:
        Mxx = 0
        Mxy = 0
        Mxz = 0
        Myy = 0
        Myz = 0
        Mzz = 0
      if r == 0:
        pass
      else:
        Mxx = fourOverThree 
        Myy = fourOverThree 
        Mzz = fourOverThree 
        c1 = 1.0 + 1.0 / (3.0 * r2)
        c2 = (1.0 - 1.0 * invr2) * invr2
        Mxx -=  (c1 + c2*rx*rx) * invr
        Mxy  = -(     c2*rx*ry) * invr
        Mxz  = -(     c2*rx*rz) * invr
        Myy -=  (c1 + c2*ry*ry) * invr
        Myz  = -(     c2*ry*rz) * invr
        Mzz -=  (c1 + c2*rz*rz) * invr
      Myx = Mxy
      Mzx = Mxz
      Mzy = Myz
	  
      # 2. Compute product M_ij * F_j           
      ux += (Mxx * fx_vec[j] + Mxy * fy_vec[j] + Mxz * fz_vec[j]) 
      uy += (Myx * fx_vec[j] + Myy * fy_vec[j] + Myz * fz_vec[j]) 
      uz += (Mzx * fx_vec[j] + Mzy * fy_vec[j] + Mzz * fz_vec[j]) 
    u[i,0] = ux * norm_fact_f
    u[i,1] = uy * norm_fact_f
    u[i,2] = uz * norm_fact_f          
  return u.flatten()


@utils.static_var('r_vectors_old', [])
def double_layer_stkfmm(r, normals, field, weights, PVel, L=np.zeros(3), *args, **kwargs):
  '''
  Stokes double layer.
  '''
  
  def project_to_periodic_image(r, L):
    '''
    Project a vector r to the minimal image representation
    of size L=(Lx, Ly, Lz) and with a corner at (0,0,0). If 
    any dimension of L is equal or smaller than zero the 
    box is assumed to be infinite in that direction.
    
    If one dimension is not periodic shift all coordinates by min(r[:,i]) value.
    '''
    if L is not None:
      for i in range(3):
        if(L[i] > 0):
          r[:,i] = r[:,i] - (r[:,i] // L[i]) * L[i]
        else:
          ri_min =  np.min(r[:,i])
          if ri_min < 0:
            r[:,i] -= ri_min
    return r
  
  # Prepare coordinates
  N = r.size // 3
  r_vectors = np.copy(r)
  r_vectors = project_to_periodic_image(r_vectors, L)
 
  # Set tree if necessary
  build_tree = True
  if len(double_layer_stkfmm.r_vectors_old) > 0:
    if np.array_equal(double_layer_stkfmm.r_vectors_old, r_vectors):
      double_layer_stkfmm.r_vectors_old = np.copy(r_vectors)
      build_tree = False
  if build_tree:
    # Build tree in STKFMM
    if L[0] > 0:
      x_min = 0
      Lx_pvfmm = L[0]
      Lx_cKDTree = L[0]
    else:
      x_min = np.min(r_vectors[:,0])
      Lx_pvfmm = (np.max(r_vectors[:,0]) * 1.01 - x_min)
      Lx_cKDTree = (np.max(r_vectors[:,0]) * 1.01 - x_min) * 10
    if L[1] > 0:
      y_min = 0
      Ly_pvfmm = L[1]
      Ly_cKDTree = L[1]
    else:
      y_min = np.min(r_vectors[:,1])
      Ly_pvfmm = (np.max(r_vectors[:,1]) * 1.01 - y_min)
      Ly_cKDTree = (np.max(r_vectors[:,1]) * 1.01 - y_min) * 10
    if L[2] > 0:
      z_min = 0
      Lz_pvfmm = L[2]
      Lz_cKDTree = L[2]
    else:
      z_min = np.min(r_vectors[:,2])
      z_min = 0
      Lz_pvfmm = (np.max(r_vectors[:,2]) * 1.01 - z_min)
      Lz_cKDTree = (np.max(r_vectors[:,2]) * 1.01 - z_min) * 10

    # Set box size for pvfmm
    if L[0] > 0 or L[1] > 0 or L[2] > 0:
      L_box = np.max(L)
    else:
      L_box = np.max([Lx_pvfmm, Ly_pvfmm, 2 * Lz_pvfmm])
    
    # Set box size for pvfmm
    if L[0] > 0 or L[1] > 0 or L[2] > 0:
      L_box = np.max(L)
    else:
      L_box = np.max([Lx_pvfmm, Ly_pvfmm, 2 * Lz_pvfmm])

    # Buid FMM tree
    PVel.set_box(np.array([x_min, y_min, z_min]), L_box)
    PVel.set_points(np.zeros(0), r_vectors, r_vectors)
    PVel.setup_tree(PySTKFMM.KERNEL.PVel)
    
  # Set double layer
  trg_value = np.zeros((N, 4))
  src_DL_value = np.einsum('bi,bj,b->bij', normals, field, weights).reshape((N, 9))
  if np.any(L > 0):
    trace = np.average(src_DL_value[:,0] + src_DL_value[:,4] + src_DL_value[:,8])
    src_DL_value[:,0] -= trace / 3
    src_DL_value[:,4] -= trace / 3
    src_DL_value[:,8] -= trace / 3

  # Evaluate fmm; format c = trg_value[:,0], grad_c = trg_value[:,1:4]
  PVel.clear_fmm(PySTKFMM.KERNEL.PVel)
  PVel.evaluate_fmm(PySTKFMM.KERNEL.PVel, np.zeros(0), trg_value, src_DL_value)  
  comm = kwargs.get('comm')
  comm.Barrier()

  # Return velocity
  u = 2 * trg_value[:,1:4]
  return u


