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
  
  for k, b in enumerate(bodies):
    # Rotate magnetic dipole
    rotation_matrix = b.orientation.rotation_matrix()
    mu_body = np.dot(rotation_matrix, mu)

    # Compute torque
    force_torque[2*k+1] = np.cross(mu_body, B)

    # Add harmonic potential
    force_torque[2*k,2] = -harmonic_confinement * (b.location[2] - harmonic_confinement_plane)
    
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
    return calc_body_body_forces_torques_numba
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
  mu = kwargs.get('mu')
  vacuum_permeability = kwargs.get('vacuum_permeability')
  dipole_dipole = kwargs.get('dipole_dipole')
  
  # Extract body locations and dipoles
  r_bodies = np.zeros((len(bodies), 3))
  dipoles = np.zeros((len(bodies), 3))
  for i, b in enumerate(bodies):
    r_bodies[i] = b.location
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
  return force_torque_bodies.reshape((2*len(bodies),3))


@njit(parallel=True, fastmath=True)
def body_body_force_torque_numba(r_bodies, dipoles, vacuum_permeability):
  '''
  This function compute the force between N bodies
  with locations r and dipoles dipoles.
  '''
  N = r_bodies.size // 3
  force = np.zeros_like(r_bodies)
  torque = np.zeros_like(r_bodies)

  # Loop over bodies
  for i in prange(N):
    mi = dipoles[i]
    for j in range(N):
      if i == j:
        continue
      mj = dipoles[j]
      
      # Distance between bodies
      rij = r_bodies[i] - r_bodies[j]
      r = np.sqrt(rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2])
      rij_hat = rij / r

      #if r > 2.4:
      #  continue

      # Compute force
      Ai = np.dot(mi, rij_hat)
      Aj = np.dot(mj, rij_hat)
      force[i] += (mi * Aj + mj * Ai + rij_hat * np.dot(mi,mj) - 5 * rij_hat * Ai * Aj) / r**4
      # force[i] += -(1e-04 / r**4) * rij_hat

      # Compute torque
      torque[i,0] += (3*Aj * (mi[1] * rij_hat[2] - mi[2]*rij_hat[1]) - (mi[1] * mj[2] - mi[2]*mj[1])) / r**3
      torque[i,1] += (3*Aj * (mi[2] * rij_hat[0] - mi[0]*rij_hat[2]) - (mi[2] * mj[0] - mi[0]*mj[2])) / r**3
      torque[i,2] += (3*Aj * (mi[0] * rij_hat[1] - mi[1]*rij_hat[0]) - (mi[0] * mj[1] - mi[1]*mj[0])) / r**3

  # Multiply by prefactors
  force *= 0.75 * vacuum_permeability / np.pi
  torque *= 0.25 * vacuum_permeability / np.pi 

  # Return 
  return force, torque


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
  force *= (0.125 * vacuum_permeability / np.pi)

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
  for i, b in enumerate(bodies):
    r_bodies[i] = np.copy(b.location)
    dipoles[i] = np.dot(b.orientation.rotation_matrix(), mu)
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
  return force_torque.reshape((2*Nbodies, 3))



