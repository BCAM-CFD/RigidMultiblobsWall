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
import multi_bodies_functions
from multi_bodies_functions import *
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
  omega_perp = kwargs.get('omega_perp')
  step = kwargs.get('step')
  dt = kwargs.get('dt')
  time = step * dt

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
    # force_torque[2*k+1] = mu

  return force_torque
multi_bodies_functions.bodies_external_force_torque = bodies_external_force_torque_new


def set_body_body_forces_torques_new(implementation):
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
  
  # Extract body locations and dipoles
  r_bodies = np.zeros((len(bodies), 3))
  dipoles = np.zeros((len(bodies), 3))
  for i, b in enumerate(bodies):
    r_bodies[i] = b.location
    dipoles[i] = np.dot(b.orientation.rotation_matrix(), mu)
  
  # Compute forces and torques
  force, torque = body_body_force_torque_numba(r_bodies, dipoles, vacuum_permeability)
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

      # Compute force
      Ai = np.dot(mi, rij_hat)
      Aj = np.dot(mj, rij_hat)
      force[i] += (mi * Aj + mj * Ai + rij_hat * np.dot(mi,mj) - 5 * rij_hat * Ai * Aj) / r**4

      # Compute torque
      torque[i,0] += (3*Aj * (mi[1] * rij_hat[2] - mi[2]*rij_hat[1]) - (mi[1] * mj[2] - mi[2]*mj[1])) / r**3
      torque[i,1] += (3*Aj * (mi[2] * rij_hat[0] - mi[0]*rij_hat[2]) - (mi[2] * mj[0] - mi[0]*mj[2])) / r**3
      torque[i,2] += (3*Aj * (mi[0] * rij_hat[1] - mi[1]*rij_hat[0]) - (mi[0] * mj[1] - mi[1]*mj[0])) / r**3

  # Multiply by prefactors
  force *= 0.75 * vacuum_permeability / np.pi
  torque *= 0.25 * vacuum_permeability / np.pi 

  # Return 
  return force, torque
  

@utils.static_var('counter', [])
@utils.static_var('grid_coor', [])
@utils.static_var('stress_avg', [])
@utils.static_var('stress_deviation', [])
def save_stress_field(mesh, r_vectors_blobs, force_blobs, blob_radius, step, save_stress_step, output):
  if len(save_stress_field.counter) == 0:
    save_stress_field.counter.append(0)
    print('Initializing')

    # Prepare grid values
    grid = np.reshape(mesh, (3,3)).T
    grid_length = grid[1] - grid[0]
    grid_points = np.array(grid[2], dtype=np.int32)
    num_points = grid_points[0] * grid_points[1] * grid_points[2]

    # Set grid coordinates
    dx_grid = grid_length / grid_points
    grid_x = np.array([grid[0,0] + dx_grid[0] * (x+0.5) for x in range(grid_points[0])])
    grid_y = np.array([grid[0,1] + dx_grid[1] * (x+0.5) for x in range(grid_points[1])])
    grid_z = np.array([grid[0,2] + dx_grid[2] * (x+0.5) for x in range(grid_points[2])])
    # Be aware, x is the fast axis.
    zz, yy, xx = np.meshgrid(grid_z, grid_y, grid_x, indexing = 'ij')
    grid_coor = np.zeros((num_points, 3))
    grid_coor[:,0] = np.reshape(xx, xx.size)
    grid_coor[:,1] = np.reshape(yy, yy.size)
    grid_coor[:,2] = np.reshape(zz, zz.size)
    
    # Create stress 
    stress_avg = np.zeros((num_points, 9))
    stress_deviation = np.zeros((num_points, 9))

    save_stress_field.grid_coor = grid_coor
    save_stress_field.stress_avg = stress_avg
    save_stress_field.stress_deviation = stress_deviation
  
  # Get stored variables
  num_points = save_stress_field.grid_coor.size // 3
  counter = save_stress_field.counter[0]
  stress_avg = save_stress_field.stress_avg
  stress_deviation = save_stress_field.stress_deviation

  # Compute stress field
  stress_field = np.random.randn(num_points, 9)

  # Save stress
  save_stress_field.stress_deviation += counter * (stress_field - stress_avg)**2 / (counter + 1)
  save_stress_field.stress_avg += (stress_field - stress_avg) / (counter + 1)
  # print('save_stress_field = ', save_stress_field.counter[0], save_stress_field.stress_avg[0])

  # Update counter
  save_stress_field.counter[0] = save_stress_field.counter[0] + 1 

  # Save stress tensor to vtk fields
  if step % save_stress_step == 0:
    # Get stress and Compute stress variance
    stress_field = save_stress_field.stress_avg
    stress_variance = save_stress_field.stress_deviation / np.maximum(1.0, (save_stress_field.counter[0] - 1))
    
    # Prepare grid values
    grid = np.reshape(mesh, (3,3)).T
    grid_length = grid[1] - grid[0]
    grid_points = np.array(grid[2], dtype=np.int32)

    # Set grid coordinates
    dx_grid = grid_length / grid_points
    grid_x = np.array([grid[0,0] + dx_grid[0] * (x+0.5) for x in range(grid_points[0])])
    grid_y = np.array([grid[0,1] + dx_grid[1] * (x+0.5) for x in range(grid_points[1])])
    grid_z = np.array([grid[0,2] + dx_grid[2] * (x+0.5) for x in range(grid_points[2])])
    grid_x = grid_x - dx_grid[0] * 0.5
    grid_y = grid_y - dx_grid[1] * 0.5
    grid_z = grid_z - dx_grid[2] * 0.5
    grid_x = np.concatenate([grid_x, [grid[1,0]]])
    grid_y = np.concatenate([grid_y, [grid[1,1]]])
    grid_z = np.concatenate([grid_z, [grid[1,2]]])

    # Prepara data for VTK writer 
    variables = [stress_avg[:,0], stress_avg[:,1], stress_avg[:,2], stress_avg[:,3], stress_avg[:,4], stress_avg[:,5], stress_avg[:,6], stress_avg[:,7], stress_avg[:,8]]
    dims = np.array([grid_points[0]+1, grid_points[1]+1, grid_points[2]+1], dtype=np.int32) 
    nvars = 9
    vardims =   np.array([1,1,1,1,1,1,1,1,1], dtype=np.int32)
    centering = np.array([0,0,0,0,0,0,0,0,0], dtype=np.int32)
    varnames = ['stress_XX\0', 'stress_XY\0', 'stress_XZ\0', 'stress_YX\0', 'stress_YY\0', 'stress_YZ\0', 'stress_ZX\0', 'stress_ZY\0', 'stress_ZZ\0']
    name = output + '.stress_field.vtk'

    # Write velocity field
    visit_writer.boost_write_rectilinear_mesh(name,      # File's name
                                              0,         # 0=ASCII,  1=Binary
                                              dims,      # {mx, my, mz}
                                              grid_x,    # xmesh
                                              grid_y,    # ymesh
                                              grid_z,    # zmesh
                                              nvars,     # Number of variables
                                              vardims,   # Size of each variable, 1=scalar, velocity=3*scalars
                                              centering, # Write to cell centers of corners
                                              varnames,  # Variables' names
                                              variables) # Variables
    
    print('***********************************************************\n\n')
  return 
multi_bodies_functions.save_stress_field = save_stress_field
