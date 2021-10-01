'''
In this module the user can define functions that modified the
code multi_blobs.py. For example, functions to define the
blobs-blobs interactions, the forces and torques on the rigid
bodies or the slip on the blobs.
'''
import numpy as np
import sys
import imp
import os.path
from functools import partial
import scipy.spatial as spatial

# Try to import numba
try:
  from numba import njit, prange
except ImportError:
  print('numba not found')

import general_application_utils as utils
from quaternion_integrator.quaternion import Quaternion

# If numba is installed import forces_numba
try: 
  imp.find_module('numba')
  found_numba = True
except ImportError:
  found_numba = False
if found_numba:
  import forces_numba
try:
  import forces_cpp
except ImportError:
  pass


def project_to_periodic_image(r, L):
  '''
  Project a vector r to the minimal image representation
  centered around (0,0,0) and of size L=(Lx, Ly, Lz). If 
  any dimension of L is equal or smaller than zero the 
  box is assumed to be infinite in that direction.
  '''
  if L is not None:
    for i in range(3):
      if(L[i] > 0):
        r[i] = r[i] - int(r[i] / L[i] + 0.5 * (int(r[i]>0) - int(r[i]<0))) * L[i]
  return r


def put_r_vecs_in_periodic_box(r_vecs, L):
  for r_vec in r_vecs:
    for i in range(3):
      if L[i] > 0:
        while r_vec[i] < 0:
          r_vec[i] += L[i]
        while r_vec[i] > L[i]:
          r_vec[i] -= L[i]


def default_zero_r_vectors(r_vectors, *args, **kwargs):
  return np.zeros((r_vectors.size // 3, 3))


def default_zero_blobs(body, *args, **kwargs):
  ''' 
  Return a zero array of shape (body.Nblobs, 3)
  '''
  return np.zeros((body.Nblobs, 3))


def default_zero_bodies(bodies, *args, **kwargs):
  ''' 
  Return a zero array of shape (2*len(bodies), 3)
  '''
  return np.zeros((2*len(bodies), 3))
  

def set_slip_by_ID(body, slip):
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
    active_body_slip_partial = partial(active_body_slip, slip = slip)
    body.function_slip = active_body_slip_partial
  else:
    body.function_slip = default_zero_blobs
  return


def active_body_slip(body, slip):
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


def blob_external_forces(r_vectors, *args, **kwargs):
  '''
  This function compute the external force acting on a
  single blob. It returns an array with shape (3).

  In this example we add gravity and a repulsion with the wall;
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
  blob_mass = kwargs.get('blob_mass')
  blob_radius = kwargs.get('blob_radius')
  g = kwargs.get('g')
  repulsion_strength_wall = kwargs.get('repulsion_strength_wall')
  debye_length_wall = kwargs.get('debye_length_wall')
  # Add gravity
  f[:,2] = -g * blob_mass

  # Add wall interaction
  h = r_vectors[:,2]
  lr_mask = h > blob_radius
  sr_mask = h <= blob_radius
  f[lr_mask,2] += (repulsion_strength_wall / debye_length_wall) * np.exp(-(h[lr_mask]-blob_radius)/debye_length_wall)
  f[sr_mask,2] += (repulsion_strength_wall / debye_length_wall)

  return f


def calc_one_blob_forces(r_vectors, *args, **kwargs):
  '''
  Compute one-blob forces. It returns an array with shape (Nblobs, 3).
  '''
  r_vectors = np.reshape(r_vectors, (r_vectors.size // 3, 3)) 
  return blob_external_forces(r_vectors, *args, **kwargs)   


def set_blob_blob_forces(implementation):
  '''
  Set the function to compute the blob-blob forces
  to the right function.
  The implementation in pycuda is much faster than the
  one in C++, which is much faster than the one python; 
  To use the pycuda implementation is necessary to have 
  installed pycuda and a GPU with CUDA capabilities. To
  use the C++ implementation the user has to compile 
  the file blob_blob_forces_ext.cc.   
  '''
  if implementation == 'None':
    return default_zero_r_vectors
  elif implementation == 'python':
    return calc_blob_blob_forces_python
  elif implementation == 'numba':
    return forces_numba.calc_blob_blob_forces_numba
  elif implementation == 'tree_numba':
    return forces_numba.calc_blob_blob_forces_tree_numba


def blob_blob_force(r, *args, **kwargs):
  '''
  This function compute the force between two blobs
  with vector between blob centers r.

  In this example the force is derived from the potential
  
  U(r) = U0 + U0 * (2*a-r)/b   if z<2*a
  U(r) = U0 * exp(-(r-2*a)/b)  iz z>=2*a
  
  with
  eps = potential strength
  r_norm = distance between blobs
  b = Debye length
  a = blob_radius
  '''
  # Get parameters from arguments
  L = kwargs.get('periodic_length')
  a = kwargs.get('blob_radius')
  eps = kwargs.get('repulsion_strength') 
  b = kwargs.get('debye_length')

  # Compute force
  project_to_periodic_image(r, L)
  r_norm = np.linalg.norm(r)
  
  if r_norm > 2.0 * a:
    return -((eps / b) * np.exp(-(r_norm- 2.0 * a) / b) / np.maximum(r_norm, np.finfo(float).eps)) * r 
  else:
    return -((eps / b) / np.maximum(r_norm, np.finfo(float).eps)) * r;
  
  return force_torque
  

def calc_blob_blob_forces_python(r_vectors, *args, **kwargs):
  '''
  This function computes the blob-blob forces and returns
  an array with shape (Nblobs, 3).
  '''
  Nblobs = r_vectors.size // 3
  force_blobs = np.zeros((Nblobs, 3))

  # Double loop over blobs to compute forces
  for i in range(Nblobs-1):
    for j in range(i+1, Nblobs):
      # Compute vector from j to u
      r = r_vectors[j] - r_vectors[i]
      force = blob_blob_force(r, *args, **kwargs)
      force_blobs[i] += force
      force_blobs[j] -= force

  return force_blobs


def calc_blob_blob_forces_boost(r_vectors, *args, **kwargs):
  '''
  Call a boost function to compute the blob-blob forces.
  '''
  # Get parameters from arguments
  L = kwargs.get('periodic_length')
  eps = kwargs.get('repulsion_strength')
  b = kwargs.get('debye_length')  
  blob_radius = kwargs.get('blob_radius')  

  number_of_blobs = r_vectors.size // 3
  r_vectors = np.reshape(r_vectors, (number_of_blobs, 3))
  forces = np.empty(r_vectors.size)
  if L is None:
    L = -1.0*np.ones(3)

  forces_ext.calc_blob_blob_forces(r_vectors, forces, eps, b, blob_radius, number_of_blobs, L)
  return np.reshape(forces, (number_of_blobs, 3))


def body_body_force_torque(r, quaternion_i, quaternion_j, *args, **kwargs):
  '''
  This function compute the force between two bodies
  with vector between locations r.
  In this example the torque is zero and the force 
  is derived from a Yukawa potential
  
  U = eps * exp(-r_norm / b) / r_norm
  
  with
  eps = potential strength
  r_norm = distance between bodies' location
  b = Debye length
  '''
  force_torque = np.zeros((2, 3)) 
  return force_torque


def calc_body_body_forces_torques_python(bodies, r_vectors, *args, **kwargs):
  '''
  This function computes the body-body forces and torques and returns
  an array with shape (2*Nblobs, 3).
  '''
  Nbodies = len(bodies)
  force_torque_bodies = np.zeros((2*len(bodies), 3))
  
  # Double loop over bodies to compute forces
  for i in range(Nbodies-1):
    for j in range(i+1, Nbodies):
      # Compute vector from j to u
      r = bodies[j].location - bodies[i].location
      force_torque = body_body_force_torque(r, bodies[i].orientation, bodies[j].orientation, *args, **kwargs)
      # Add forces
      force_torque_bodies[2*i] += force_torque[0]
      force_torque_bodies[2*j] -= force_torque[0]
      # Add torques
      force_torque_bodies[2*i+1] += force_torque[1]
      force_torque_bodies[2*j+1] -= force_torque[1]

  return force_torque_bodies


def force_torque_calculator_sort_by_bodies(bodies, r_vectors, *args, **kwargs):
  '''
  Return the forces and torque in each body with
  format [f_1, t_1, f_2, t_2, ...] and shape (2*Nbodies, 3),
  where f_i and t_i are the force and torque on the body i.
  '''
  # Create auxiliar variables
  Nblobs = r_vectors.size // 3
  force_torque_bodies = np.zeros((2*len(bodies), 3))
  force_blobs = np.zeros((Nblobs, 3))
  blob_mass = 1.0
  blob_radius = bodies[0].blob_radius

  # Compute one-blob forces (same function for all blobs)
  force_blobs += calc_one_blob_forces(r_vectors, blob_radius = blob_radius, blob_mass = blob_mass, *args, **kwargs)

  # Compute blob-blob forces (same function for all pair of blobs)
  force_blobs += calc_blob_blob_forces(r_vectors, blob_radius = blob_radius, *args, **kwargs)  

  # Compute body force-torque forces from blob forces
  offset = 0
  for k, b in enumerate(bodies):
    # Add force to the body
    force_torque_bodies[2*k:(2*k+1)] += sum(force_blobs[offset:(offset+b.Nblobs)])
    # Add torque to the body
    R = b.calc_rot_matrix()  
    force_torque_bodies[2*k+1:2*k+2] += np.dot(R.T, np.reshape(force_blobs[offset:(offset+b.Nblobs)], 3*b.Nblobs))
    offset += b.Nblobs

  # Add one-body external force-torque
  force_torque_bodies += bodies_external_force_torque(bodies, r_vectors, blob_radius = blob_radius, *args, **kwargs)

  # Add body-body forces (same for all pair of bodies)
  force_torque_bodies += calc_body_body_forces_torques(bodies, r_vectors, *args, **kwargs)
  return force_torque_bodies


def preprocess(bodies, *args, **kwargs):
  '''
  This function is call at the start of the schemes.
  The default version do nothing, it should be modify by
  the user if he wants to change the schemes.
  '''
  return

def postprocess(bodies, *args, **kwargs):
  '''
  This function is call at the end of the schemes but
  before checking if the postions are a valid configuration.
  The default version do nothing, it should be modify by
  the user if he wants to change the schemes.
  '''
  return


def bodies_external_force_torque(bodies, r_vectors, *args, **kwargs):
  '''
  This function returns the external force-torques acting on the bodies.
  It returns an array with shape (2*len(bodies), 3)
  
  
  The force is zero the torque is:

  T = mu \times B

  mu = define in the body frame of reference and rotate to the
       lab frame of reference.
  B = R_B * B0 * (cos(omega*time), sin(omega*time), 0)
  R_B = rotation matrix associated with a quaternion_B.

  '''
  # Get parameters
  force_torque = np.zeros((2*len(bodies), 3))
  mu = kwargs.get('mu')
  B0 = kwargs.get('B0')
  omega = kwargs.get('omega')
  quaternion_B = kwargs.get('quaternion_B')
  step = kwargs.get('step')
  dt = kwargs.get('dt')
  time = step * dt
  harmonic_confinement = kwargs.get('harmonic_confinement')
  harmonic_confinement_plane = kwargs.get('harmonic_confinement_plane')

  # Rotate magnetic field
  R_B = quaternion_B.rotation_matrix()
  B = B0 * np.array([np.cos(omega * time), np.sin(omega * time), 0.0])
  B = np.dot(R_B, B)
  
  for k, b in enumerate(bodies):
    # Rotate magnetic dipole
    rotation_matrix = b.orientation.rotation_matrix()
    mu_body = np.dot(rotation_matrix, mu)

    # Compute torque
    force_torque[2*k+1] = np.cross(mu_body, B)

    # Add harmonic potential
    force_torque[2*k,2] = -harmonic_confinement * b.k * (b.location[2] - harmonic_confinement_plane)

    # Add gravity
    force_torque[2*k,2] += -b.mg

    # Add wall repulsion
    h = b.location[2]
    if h > b.R:
      force_torque[2*k,2] += (b.repulsion_strength_wall / b.debye_length_wall) * np.exp(-(h - b.R) / b.debye_length_wall)
    else:
      force_torque[2*k,2] += (b.repulsion_strength_wall / b.debye_length_wall)       
  return force_torque


def set_body_body_forces_torques(implementation, *args, **kwargs):
  '''
  Set the function to compute the body-body forces
  to the right function. 
  '''
  if implementation == 'None':
    return default_zero_bodies
  elif implementation == 'python':
    return calc_body_body_forces_torques_python
  elif implementation == 'numba':
    return partial(calc_body_body_forces_torques_numba, L=kwargs.get('L'))  
  elif implementation == 'stkfmm':
    # STKFMM parameters
    mult_order = kwargs.get('stkfmm_mult_order')
    max_pts = 512
    pbc_string = kwargs.get('stkfmm_pbc')
    if pbc_string == 'None':
      pbc = PySTKFMM.PAXIS.NONE
    elif pbc_string == 'PX':
      pbc = PySTKFMM.PAXIS.PX
    elif pbc_string == 'PXY':
      pbc = PySTKFMM.PAXIS.PXY
    elif pbc_string == 'PXYZ':
      pbc = PySTKFMM.PAXIS.PXYZ

    # u, lapu kernel (4->6)
    kernel = PySTKFMM.KERNEL.LapPGradGrad
    kernel_index = PySTKFMM.KERNEL(kernel)
  
    # Setup FMM
    stkfmm = PySTKFMM.Stk3DFMM(mult_order, max_pts, pbc, kernel_index)
    stkfmm.showActiveKernels()
    kdimSL, kdimDL, kdimTrg = stkfmm.getKernelDimension(kernel)
    print('kdimSL, kdimDL, kdimTrg = ', kdimSL, kdimDL, kdimTrg)

    return partial(calc_body_body_forces_torques_stkfmm, 
                   stkfmm=stkfmm, 
                   mu=kwargs.get('mu'), 
                   vacuum_permeability=kwargs.get('vacuum_permeability'), 
                   L=kwargs.get('L'), 
                   comm=kwargs.get('comm'))  


def calc_body_body_forces_torques_numba(bodies, r_vectors, *args, **kwargs):
  '''
  This function computes the body-body forces and torques and returns
  an array with shape (2*Nblobs, 3).
  '''
  Nbodies = len(bodies)
  force_torque_bodies = np.zeros((len(bodies), 6))
  mu = kwargs.get('mu')
  vacuum_permeability = kwargs.get('vacuum_permeability')
  dipole_dipole = kwargs.get('dipole_dipole')
  L = kwargs.get('L')
  
  # Extract body locations and dipoles
  r_bodies = np.zeros((len(bodies), 3))
  dipoles = np.zeros((len(bodies), 3))
  R = np.zeros(len(bodies))
  repulsion_strength = np.zeros(len(bodies))
  debye_length = np.zeros(len(bodies))
  for i, b in enumerate(bodies):
    r_bodies[i] = b.location
    R[i] = b.R
    repulsion_strength[i] = b.repulsion_strength
    debye_length[i] = b.debye_length
    dipoles[i] = np.dot(b.orientation.rotation_matrix(), mu)
  
  # Compute forces and torques
  if dipole_dipole == 'True':
    force, torque = body_body_force_torque_numba_fast(r_bodies, dipoles, vacuum_permeability)
  elif dipole_dipole == 'isotropic':
    force, torque = body_body_force_torque_numba_isotropic(r_bodies, dipoles, vacuum_permeability)
  else:
    force = np.zeros((Nbodies, 3))
    torque = np.zeros((Nbodies, 3))
  force_torque_bodies[:,0:3] = force
  force_torque_bodies[:,3:6] = torque

  # Calc steric body-body force
  if np.max(repulsion_strength) > 0:
    force = calc_body_body_forces_tree_numba(r_bodies, L, repulsion_strength, debye_length, R)
    force_torque_bodies[:, 0:3] += force 
  return force_torque_bodies.reshape((2*len(bodies),3))


@njit(parallel=True, fastmath=True)
def body_body_force_torque_numba_fast(r_bodies, dipoles, vacuum_permeability):
  '''
  This function compute the force between N bodies
  with locations r and dipoles dipoles.
  '''
  N = r_bodies.size // 3
  force = np.zeros_like(r_bodies)
  torque = np.zeros_like(r_bodies)
  mx = np.copy(dipoles[:,0])
  my = np.copy(dipoles[:,1])
  mz = np.copy(dipoles[:,2])
  rx = np.copy(r_bodies[:,0])
  ry = np.copy(r_bodies[:,1])
  rz = np.copy(r_bodies[:,2])

  # Loop over bodies
  for i in prange(N):
    mxi = mx[i]
    myi = my[i]
    mzi = mz[i]
    rxi = rx[i]
    ryi = ry[i]
    rzi = rz[i]
    for j in range(N):
      if i == j:
        continue
      mxj = mx[j]
      myj = my[j]
      mzj = mz[j]
      
      # Distance between bodies
      rxij = rxi - rx[j]
      ryij = ryi - ry[j]
      rzij = rzi - rz[j]
      r2 = (rxij*rxij + ryij*ryij + rzij*rzij)
      r = np.sqrt(r2)
      r3_inv = 1.0 / (r * r2)
      r4_inv = 1.0 / (r2 * r2)
      rxij_hat = rxij / r
      ryij_hat = ryij / r
      rzij_hat = rzij / r

      #if r > 2.4:
      #  continue

      # Compute force
      Ai = mxi * rxij_hat + myi * ryij_hat + mzi * rzij_hat
      Aj = mxj * rxij_hat + myj * ryij_hat + mzj * rzij_hat
      mimj = mxi * mxj + myi * myj + mzi * mzj
      force[i,0] += (mxi * Aj + mxj * Ai + rxij_hat * mimj - 5 * rxij_hat * Ai * Aj) * r4_inv
      force[i,1] += (myi * Aj + myj * Ai + ryij_hat * mimj - 5 * ryij_hat * Ai * Aj) * r4_inv
      force[i,2] += (mzi * Aj + mzj * Ai + rzij_hat * mimj - 5 * rzij_hat * Ai * Aj) * r4_inv

      # Compute torque
      torque[i,0] += (3*Aj * (myi * rzij_hat - mzi*ryij_hat) - (myi * mzj - mzi*myj)) * r3_inv
      torque[i,1] += (3*Aj * (mzi * rxij_hat - mxi*rzij_hat) - (mzi * mxj - mxi*mzj)) * r3_inv
      torque[i,2] += (3*Aj * (mxi * ryij_hat - myi*rxij_hat) - (mxi * myj - myi*mxj)) * r3_inv

  # Multiply by prefactors
  force *= (0.75 * vacuum_permeability / np.pi)
  torque *= (0.25 * vacuum_permeability / np.pi)

  # Return 
  return force, torque


@njit(parallel=True, fastmath=True)
def body_body_force_torque_numba_isotropic(r_bodies, dipoles, vacuum_permeability):
  '''
  This function compute the force between N bodies
  with locations r and isotropic dipoles dipoles.
  '''
  N = r_bodies.size // 3
  force = np.zeros_like(r_bodies)
  torque = np.zeros_like(r_bodies)
  mx = np.copy(dipoles[:,0])
  my = np.copy(dipoles[:,1])
  mz = np.copy(dipoles[:,2])
  rx = np.copy(r_bodies[:,0])
  ry = np.copy(r_bodies[:,1])
  rz = np.copy(r_bodies[:,2])

  # Loop over bodies
  for i in prange(N):
    mi = np.sqrt(mx[i] * mx[i] + my[i] * my[i] + mz[i] * mz[i])
    rxi = rx[i]
    ryi = ry[i]
    rzi = rz[i]
    for j in range(N):
      if i == j:
        continue
      mj = np.sqrt(mx[j] * mx[j] + my[j] * my[j] + mz[j] * mz[j])
      
      # Distance between bodies
      rxij = rxi - rx[j]
      ryij = ryi - ry[j]
      rzij = rzi - rz[j]
      r2 = (rxij*rxij + ryij*ryij + rzij*rzij)
      r = np.sqrt(r2)
      r5_inv = 1.0 / (r * r2 * r2)

      # Compute force
      force[i,0] -= mi * mj * rxij * r5_inv
      force[i,1] -= mi * mj * ryij * r5_inv
      force[i,2] -= mi * mj * rzij * r5_inv

  # Multiply by prefactors
  force *= (0.125 * 3 * vacuum_permeability / np.pi)

  # Return 
  return force, torque
  

@utils.static_var('r_vectors_old', [])
@utils.static_var('list_of_neighbors', [])
@utils.static_var('offsets', [])
def calc_body_body_forces_torques_stkfmm(bodies, r_vectors, *args, **kwargs):
  ''' 
  This function computes the body-body forces and torques and returns
  an array with shape (2*Nblobs, 3).
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
          r[:,i] -= np.min(r[:,i])
    return r

  # Prepare coordinates
  Nbodies = len(bodies)
  mu = kwargs.get('mu')
  vacuum_permeability = kwargs.get('vacuum_permeability')
  L = kwargs.get('L')
  stkfmm = kwargs.get('stkfmm')
  
  # Extract body locations and dipoles
  r_bodies = np.zeros((len(bodies), 3))
  dipoles = np.zeros((len(bodies), 3))
  R = np.zeros(len(bodies))
  repulsion_strength = np.zeros(len(bodies))
  debye_length = np.zeros(len(bodies))
  for i, b in enumerate(bodies):
    r_bodies[i] = np.copy(b.location)
    dipoles[i] = np.dot(b.orientation.rotation_matrix(), mu)
    R[i] = b.R
    repulsion_strength[i] = b.repulsion_strength
    debye_length[i] = b.debye_length
  r_bodies = project_to_periodic_image(r_bodies, L)

  # Set tree if necessary
  build_tree = True
  if len(calc_body_body_forces_torques_stkfmm.list_of_neighbors) > 0:
    if np.array_equal(calc_body_body_forces_torques_stkfmm.r_bodies_old, r_bodies):
      build_tree = False
  if build_tree:
    # Build tree in STKFMM
    if L[0] > 0:
      x_min = 0
      x_max = L[0]
    else:
      x_min = np.min(r_bodies[:,0])
      x_max = np.max(r_bodies[:,0]) + 1e-10
    if L[1] > 0:
      y_min = 0
      y_max = L[1]
    else:
      y_min = np.min(r_bodies[:,1])
      y_max = np.max(r_bodies[:,1]) + 1e-10
    if L[2] > 0:
      z_min = 0
      z_max = L[2]
    else:
      z_min = np.min(r_bodies[:,2])
      z_max = np.max(r_bodies[:,2]) + 1e-10

    # Build FMM tree
    stkfmm.setBox(np.array([x_min, y_min, z_min]), np.max(np.concatenate((L, np.array([x_max-x_min, y_max-y_min, z_max-z_min])))))
    stkfmm.setPoints(0, np.zeros(0), Nbodies, r_bodies, Nbodies, r_bodies)
    stkfmm.setupTree(PySTKFMM.KERNEL.LapPGradGrad)

    # Copy for next call
    calc_body_body_forces_torques_stkfmm.r_bodies_old = np.copy(r_bodies)

  # Set force with right format (single layer potential)
  trg_value = np.zeros((Nbodies, 10))
  src_DL_value = np.zeros((Nbodies, 3))
  # Note, the minus signin the line below is necessary to reconcile the definition 
  # of the double layer potential and the magnetic field created by a dipole mu
  src_DL_value[:,0:3] = -np.copy(dipoles.reshape((Nbodies, 3))) 
    
  # Evaluate fmm; format phi = trg_value[:,0], grad(phi) = trg_value[:,1:4]
  stkfmm.clearFMM(PySTKFMM.KERNEL.LapPGradGrad)
  stkfmm.evaluateFMM(PySTKFMM.KERNEL.LapPGradGrad, 0, np.zeros(0), Nbodies, trg_value, Nbodies, src_DL_value)
  comm = kwargs.get('comm')
  comm.Barrier()

  # Compute force and torque
  force_torque = np.zeros((Nbodies, 6))
  for i in range(Nbodies):
    force_torque[i, 0] = trg_value[i,4] * dipoles[i,0] + trg_value[i,5] * dipoles[i,1] + trg_value[i,6] * dipoles[i,2]
    force_torque[i, 1] = trg_value[i,5] * dipoles[i,0] + trg_value[i,7] * dipoles[i,1] + trg_value[i,8] * dipoles[i,2]
    force_torque[i, 2] = trg_value[i,6] * dipoles[i,0] + trg_value[i,8] * dipoles[i,1] + trg_value[i,9] * dipoles[i,2]
    force_torque[i, 3:6] = np.cross(dipoles[i], trg_value[i, 1:4])
  force_torque *= vacuum_permeability

  # Calc steric body-body force
  if np.max(repulsion_strength) > 0:
    force = calc_body_body_forces_tree_numba(r_bodies, L, repulsion_strength, debye_length, R)
    force_torque[:, 0:3] += force
  return force_torque.reshape((2*Nbodies, 3))


@njit(parallel=True, fastmath=True)
def body_body_force_tree_numba(r_vectors, L, eps, b, a, list_of_neighbors, offsets):
  '''
  This function compute the force between two blobs
  with vector between blob centers r.

  In this example the force is derived from the potential
  
  U(r) = U0 + U0 * (2*a-r)/b   if z<2*a
  U(r) = U0 * exp(-(r-2*a)/b)  iz z>=2*a
  
  with
  eps = potential strength
  r_norm = distance between blobs
  b = Debye length
  a = blob_radius
  '''

  N = r_vectors.size // 3
  r_vectors = r_vectors.reshape((N, 3))
  force = np.zeros((N, 3))

  # Copy arrays
  Lx = L[0]
  Ly = L[1]
  Lz = L[2]
  rx_vec = np.copy(r_vectors[:,0])
  ry_vec = np.copy(r_vectors[:,1])
  rz_vec = np.copy(r_vectors[:,2])

  for i in prange(N):
    for k in range(offsets[i+1] - offsets[i]):
      j = list_of_neighbors[offsets[i] + k]
      if i == j:
        continue
      rx = rx_vec[j] - rx_vec[i]
      ry = ry_vec[j] - ry_vec[i]
      rz = rz_vec[j] - rz_vec[i]

      # If PBC use minimum distance
      if Lx > 0:
        rx -= int(rx / Lx + 0.5 * (int(rx>0) - int(rx<0))) * Lx
      if Ly > 0:
        ry -= int(ry / Ly + 0.5 * (int(ry>0) - int(ry<0))) * Ly
      if Lz > 0:
        rz -= int(rz / Lz + 0.5 * (int(rz>0) - int(rz<0))) * Lz

      # Parameters
      a_ij = 0.5 * (a[i] + a[j])
      b_ij = 0.5 * (b[i] + b[j])
      eps_ij = np.sqrt(eps[i] * eps[j])
        
      # Compute force
      r_norm = np.sqrt(rx*rx + ry*ry + rz*rz)
      if r_norm > 2*a_ij:
        f0 = -((eps_ij / b_ij) * np.exp(-(r_norm - 2.0*a_ij) / b_ij) / r_norm)
      else:
        f0 = -((eps_ij / b_ij) / np.maximum(r_norm, 1e-25))
      force[i, 0] += f0 * rx
      force[i, 1] += f0 * ry
      force[i, 2] += f0 * rz
  return force


def calc_body_body_forces_tree_numba(r_vectors, L, eps, b, a):
  '''
  This function computes the blob-blob forces and returns
  an array with shape (Nblobs, 3).
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
          r[:,i] -= np.min(r[:,i])
    return r

  # Get parameters from arguments
  d_max = 2 * np.max(a) + 30 * np.max(b)
  r_copy = r_vectors

  # Build tree and find neighbors
  build_tree = True
  if build_tree:
    if L[0] > 0 or L[1] > 0 or L[2] > 0:
      r_copy = np.copy(r_vectors)
      r_copy = project_to_periodic_image(r_copy, L)
      if L[0] > 0:
        Lx = L[0]
      else:
        x_min = np.min(r_copy[:,0])
        x_max = np.max(r_copy[:,0])
        Lx = (x_max - x_min) * 10
      if L[1] > 0:
        Ly = L[1]
      else:
        y_min = np.min(r_copy[:,1])
        y_max = np.max(r_copy[:,1])
        Ly = (y_max - y_min) * 10
      if L[2] > 0:
        Lz = L[2]
      else:
        z_min = np.min(r_copy[:,2])
        z_max = np.max(r_copy[:,2])
        Lz = (z_max - z_min) * 10
      boxsize = np.array([Lx, Ly, Lz])
    else:
      boxsize = None
    tree = scspatial.cKDTree(r_copy, boxsize=boxsize)
    pairs = tree.query_ball_tree(tree, d_max)
    offsets = np.zeros(len(pairs)+1, dtype=int)
    for i in range(len(pairs)):
      offsets[i+1] = offsets[i] + len(pairs[i])
    list_of_neighbors = np.concatenate(pairs).ravel()
  
  # Compute forces
  force_bodies = body_body_force_tree_numba(r_copy, L, eps, b, a, list_of_neighbors, offsets)
  return force_bodies


def slip_rod_resolved(body, *args, **kwargs):
  '''
  Apply shear flow along x-axis.
  flow_x = shear_rate * z 
  '''

  # Get slip options
  shear_rate = kwargs.get('shear_rate')

  # Applied shear
  r_configuration = body.get_r_vectors()
  slip = np.zeros((body.Nblobs, 3))
  slip[:,0] -= shear_rate * r_configuration[:,2]
  return slip


# Override force interactions by user defined functions.
# This only override the functions implemented in python.
# If user_defined_functions is empty or does not exists
# this import does nothing.
user_defined_functions_found = False
if os.path.isfile('user_defined_functions.py'):
  user_defined_functions_found = True
if user_defined_functions_found:
  import user_defined_functions

