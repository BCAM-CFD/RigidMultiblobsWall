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
import scipy.spatial as scspatial
import math
from mobility import mobility as mob
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


def bodies_external_force_torque_new(bodies, r_vectors, *args, **kwargs):
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
  # zeros = np.zeros(3)
  
  for k, b in enumerate(bodies):
    # Rotate magnetic dipole
    rotation_matrix = b.orientation.rotation_matrix()
    mu_body = np.dot(rotation_matrix, mu)
    # r_dipoles, mu_body = b.get_dipoles()
     
    # Compute torque
    force_torque[2*k+1] = np.cross(mu_body, B)
    # force_torque[2*k+1] = b.sum_dipoles(zeros, np.cross(mu_body, B), r_dipoles)[3:6]
 
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
multi_bodies_functions.bodies_external_force_torque = bodies_external_force_torque_new


def set_body_body_forces_torques_new(implementation, *args, **kwargs):
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
multi_bodies_functions.set_body_body_forces_torques = set_body_body_forces_torques_new


def calc_body_body_forces_torques_numba(bodies, r_vectors, *args, **kwargs):
  '''
  This function computes the body-body forces and torques and returns
  an array with shape (2*Nblobs, 3).
  '''
  Nbodies = len(bodies)
  force_torque_bodies = np.zeros((len(bodies), 6))
  vacuum_permeability = kwargs.get('vacuum_permeability')
  dipole_dipole = kwargs.get('dipole_dipole')
  L = kwargs.get('L')
  
  # Extract body locations and dipoles
  r_dipoles = np.zeros((len(bodies), bodies[0].dipoles_r.size // 3, 3))
  dipoles = np.zeros((len(bodies), bodies[0].dipoles.size // 3, 3))
  r_bodies = np.zeros((len(bodies), 3))
  R = np.zeros(len(bodies))
  repulsion_strength = np.zeros(len(bodies))
  debye_length = np.zeros(len(bodies))
  for i, b in enumerate(bodies):
    r, mu = b.get_dipoles()
    r_dipoles[i,:,:] = r
    dipoles[i,:,:] = mu
    r_bodies[i] = b.location
    R[i] = b.R
    repulsion_strength[i] = b.repulsion_strength
    debye_length[i] = b.debye_length
  r_dipoles = r_dipoles.reshape((r_dipoles.size // 3, 3))
  dipoles = dipoles.reshape((dipoles.size // 3, 3))
  
  # Compute forces and torques
  if dipole_dipole == 'True':
    # force, torque = body_body_force_torque_numba_fast(r_bodies, dipoles, vacuum_permeability)
    force, torque = body_body_force_torque_numba_fast(r_dipoles, dipoles, vacuum_permeability)
  elif dipole_dipole == 'isotropic':
    force, torque = body_body_force_torque_numba_isotropic(r_dipoles, dipoles, vacuum_permeability)
  else:
    force = np.zeros((Nbodies, 3))
    torque = np.zeros((Nbodies, 3))

  # Collect dipole forces-torques
  num_dipoles_body = bodies[0].dipoles_r.size // 3
  for k, b in enumerate(bodies):
    force_torque_bodies[k,0:6] = b.sum_dipoles(force[num_dipoles_body * k : num_dipoles_body * (k+1)],
                                               torque[num_dipoles_body * k : num_dipoles_body * (k+1)],
                                               r_dipoles[num_dipoles_body * k : num_dipoles_body * (k+1)])
    
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
  if 'rod_resolved_shear' in body.ID:
    body.function_slip = partial(slip_rod_resolved, *args, **kwargs)
  elif 'shear_mode' in body.ID:
    body.function_slip = partial(slip_shear_mode, *args, **kwargs)
  else:
    body.function_slip = default_zero_blobs    
  return
multi_bodies_functions.set_slip_by_ID = set_slip_by_ID_new


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


def slip_shear_mode(body, *args, **kwargs):
  '''
  Apply shear mode.
  '''
  shear_rate = kwargs.get('shear_rate')
  shear_mode = kwargs.get('shear_mode')
  r_configuration = body.get_r_vectors()
  slip = np.zeros((body.Nblobs, 3))
  
  if shear_mode == 1:
    slip[:,0] -= shear_rate * r_configuration[:,0]
    slip[:,1] += shear_rate * r_configuration[:,1]
  elif shear_mode == 2:
    slip[:,0] -= shear_rate * r_configuration[:,1]

  return slip 


def plot_velocity_field_circle(r_vectors, lambda_blobs, blob_radius, eta, circle_radius, p, output, radius_source=None, frame_body=None, *args, **kwargs):
  '''
  This function plots the velocity field to a circle.
  The grid is defined in the body frame of reference of body "frame_body".
  If frame_body < 0 the grid is defined in the laboratory frame of reference.
  '''
  # Create circle
  t = np.arange(p)
  grid_coor_ref = np.zeros((p, 3))
  grid_coor_ref[:,0] = circle_radius * np.cos(2 * np.pi * t / p)
  grid_coor_ref[:,1] = circle_radius * np.sin(2 * np.pi * t / p)
  

  # Transform grid to the body frame of reference
  if frame_body is not None:
    grid_coor = utils.get_vectors_frame_body(grid_coor_ref, body=frame_body)
  else:
    grid_coor = grid_coor_ref
    
  # Set radius of blobs (= a) and grid nodes (= 0)
  if radius_source is None:
    radius_source = np.ones(r_vectors.size // 3) * blob_radius
  radius_target = np.zeros(grid_coor.size // 3) 
  
  # Compute velocity field 
  mobility_vector_prod_implementation = kwargs.get('mobility_vector_prod_implementation')
  if mobility_vector_prod_implementation == 'python':
    grid_velocity = mob.mobility_vector_product_source_target_one_wall(r_vectors, 
                                                                       grid_coor, 
                                                                       lambda_blobs, 
                                                                       radius_source, 
                                                                       radius_target, 
                                                                       eta, 
                                                                       *args, 
                                                                       **kwargs) 
  elif mobility_vector_prod_implementation == 'C++':
    grid_velocity = mob.boosted_mobility_vector_product_source_target(r_vectors, 
                                                                      grid_coor, 
                                                                      lambda_blobs, 
                                                                      radius_source, 
                                                                      radius_target, 
                                                                      eta, 
                                                                      *args, 
                                                                      **kwargs)
  elif mobility_vector_prod_implementation == 'numba_no_wall':
    grid_velocity = mob.no_wall_mobility_trans_times_force_source_target_numba(r_vectors, 
                                                                               grid_coor, 
                                                                               lambda_blobs, 
                                                                               radius_source, 
                                                                               radius_target, 
                                                                               eta, 
                                                                               *args, 
                                                                               **kwargs) 
  else:
    grid_velocity = mob.single_wall_mobility_trans_times_force_source_target_pycuda(r_vectors, 
                                                                                    grid_coor, 
                                                                                    lambda_blobs, 
                                                                                    radius_source, 
                                                                                    radius_target, 
                                                                                    eta, 
                                                                                    *args, 
                                                                                    **kwargs)
    
  # Tranform velocity to the body frame of reference
  if frame_body is not None:
    grid_velocity = utils.get_vectors_frame_body(grid_velocity, body=frame_body, translate=False, transpose=True)
 
  # Write velocity field.
  header = 'R=' + str(circle_radius) + ', p=' + str(p) + ', N=' + str(p) + ', centered body=' + str(frame_body) + ', 7 Columns: grid point (x,y,z), quadrature weight, velocity (vx,vy,vz)'
  result = np.zeros((grid_coor.shape[0], 7))
  result[:,0:3] = grid_coor_ref
  result[:,3] = 2 * np.pi * circle_radius / p
  grid_velocity = grid_velocity.reshape((grid_velocity.size // 3, 3)) 
  result[:,4:] = grid_velocity
  np.savetxt(output, result, header=header) 
  return
multi_bodies_functions.plot_velocity_field_circle = plot_velocity_field_circle
