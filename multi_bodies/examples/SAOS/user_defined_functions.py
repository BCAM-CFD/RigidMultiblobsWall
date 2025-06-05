'''
Simple example of a flagellated bacteria. 
'''
from __future__ import division, print_function
import numexpr as ne
import multi_bodies_functions
from multi_bodies_functions import *

# Try to import numba
try:
  from numba import njit, prange
except ImportError:
  print('numba not found')


def bodies_external_force_torque_new(bodies, r_vectors, *args, **kwargs):
  '''
  This function returns the external force-torques acting on the bodies.
  It returns an array with shape (2*len(bodies), 3)
  
  The force is zero the torque is:

  T = mu \times B

  mu = define in the body frame of reference and rotate to the
       lab frame of reference.
  B = R_B * (B0_x * cos(omega_x*time + phi_x), B0_y * cos(omega_y*time + phi_y), B0_z * cos(omega_z*time + phi_z))
  R_B = rotation matrix associated with a quaternion_B.

  '''
  # Get parameters
  force_torque = np.zeros((2*len(bodies), 3))
  mu = kwargs.get('mu')
  B0 = kwargs.get('B0')
  omega = kwargs.get('omega')
  phi = kwargs.get('phi')
  quaternion_B = kwargs.get('quaternion_B')
  step = kwargs.get('step')
  dt = kwargs.get('dt')
  time = step * dt


  # Rotate magnetic field
  R_B = quaternion_B.rotation_matrix()
  B = B0 * np.cos(omega * time + phi)
  B = np.dot(R_B, B)

  for k, b in enumerate(bodies):
    # Rotate magnetic dipole
    rotation_matrix = b.orientation.rotation_matrix()
    mu_body = np.dot(rotation_matrix, mu)
     
    # Compute torque
    force_torque[2*k+1] = np.cross(mu_body, B)
      
  return force_torque
# multi_bodies_functions.bodies_external_force_torque = bodies_external_force_torque_new


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
# multi_bodies_functions.set_body_body_forces_torques = set_body_body_forces_torques_new


def calc_body_body_forces_torques_numba(bodies, r_vectors, *args, **kwargs):
  '''
  This function computes the body-body forces and torques and returns
  an array with shape (2*Nblobs, 3).
  '''
  Nbodies = len(bodies)
  force_torque_bodies = np.zeros((len(bodies), 6))
  vacuum_permeability = kwargs.get('vacuum_permeability')
  dipole_dipole = kwargs.get('dipole_dipole')
  mu = kwargs.get('mu')
  
  # Extract body locations and dipoles
  dipoles = np.zeros((len(bodies), 3))
  r_bodies = np.zeros((len(bodies), 3))

  for i, b in enumerate(bodies):
    # Rotate magnetic dipole
    rotation_matrix = b.orientation.rotation_matrix()
    dipoles[i] = np.dot(rotation_matrix, mu)
    r_bodies[i] = b.location
  
  # Compute forces and torques
  if dipole_dipole == 'True':
    force, torque = body_body_force_torque_numba_fast(r_bodies, dipoles, vacuum_permeability)
  elif dipole_dipole == 'isotropic':
    force, torque = body_body_force_torque_numba_isotropic(r_bodies, dipoles, vacuum_permeability)
  else:
    force = np.zeros((Nbodies, 3))
    torque = np.zeros((Nbodies, 3))

  # Collect dipole forces-torques
  force_torque_bodies[:,0:3] = force
  force_torque_bodies[:,3:6] = torque
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
  return  flow_resolved_coord_constant(r_configuration, body.location, t, *args, **kwargs)


def flow_resolved_coord(r, q, t, *args, **kwargs):
  '''
  Use Poisseuille flow.

  IMPORTANT: edit the variables flow_magnitude and radius_effect to the desired values.
  '''
  # Set slip options
  shear_0 = kwargs.get('shear_0')
  omega_0 = kwargs.get('omega_0')
  omega_f = kwargs.get('omega_f')
  delta = kwargs.get('delta')
  t_f = kwargs.get('t_f')

  # Flow along x, gradiend along z
  N = r.size // 3  
  background_flow = np.zeros((N, 3))
   
  # time-dependent shear and shear rate
  K = t_f * omega_0 / np.log(omega_f / omega_0)
  shear = shear_0 * np.sin(K * ((omega_f / omega_0)**(t / t_f) - 1))
  shear_rate = (shear_0 * K * np.log(omega_f / omega_0) / t_f) * np.cos(K * ((omega_f / omega_0)**(t / t_f) - 1)) * (omega_f / omega_0)**(t / t_f)

  # Chirp and chirp rate
  if t / t_f < 0.5 * delta:
    chirp = 0.5 + 0.5 * np.cos(2 * np.pi / delta * (t / t_f - 0.5 * delta))
    chirp_rate = -(np.pi / (delta * t_f)) * np.sin(2 * np.pi / delta * (t / t_f - 0.5 * delta))
  elif t / t_f < 1 - 0.5 * delta:
    chirp = 1
    chirp_rate = 0
  else:
    chirp = 0.5 + 0.5 * np.cos(2 * np.pi / delta * (t / t_f - 1 + 0.5 * delta))
    chirp_rate = -(np.pi / (delta * t_f)) * np.sin(2 * np.pi / delta * (t / t_f - 1 + 0.5 * delta))

  # Set background flow along z-axis
  background_flow[:,0] = (chirp * shear_rate + chirp_rate * shear) * (r[:,2] - q[2])
    
  return background_flow


def flow_resolved_coord_constant(r, q, t, *args, **kwargs):
  '''
  Use Poisseuille flow.

  IMPORTANT: edit the variables flow_magnitude and radius_effect to the desired values.
  '''
  # Set slip options
  shear_0 = kwargs.get('shear_0')
  periodic_length = kwargs.get('periodic_length')

  # Flow along x, gradiend along z
  N = r.size // 3  
  background_flow = np.zeros((N, 3))
   
  # Set background flow along z-axis
  background_flow[:,0] = shear_0 * (r[:,2] - q[2])
  
  return background_flow

