'''
This modules solves the mobility or the resistance problem for one
configuration of a multibody supensions and it can save some data like
the velocities or forces on the bodies, the mobility of a body or
the mobility of the blobs.
'''
from __future__ import division, print_function
import argparse
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla
import subprocess
from functools import partial
import sys
import time

# Find project functions
found_functions = False
path_to_append = ''
while found_functions is False:
  try:
    import multi_bodies_functions
    import multi_bodies
    from mobility import mobility as mob
    from quaternion_integrator.quaternion import Quaternion
    from quaternion_integrator.quaternion_integrator_multi_bodies import QuaternionIntegrator
    from body import body 
    from read_input import read_input
    from read_input import read_vertex_file
    from read_input import read_clones_file
    import general_application_utils as utils
    found_functions = True
  except ImportError:
    path_to_append += '../'
    print('searching functions in path ', path_to_append)
    sys.path.append(path_to_append)
    if len(path_to_append) > 21:
      print('\nProjected functions not found. Edit path in multi_bodies_utilities.py')
      sys.exit()

# Try to import the visit_writer (boost implementation)
try:
  import visit.visit_writer as visit_writer
except ImportError:
  pass


# Callback generator
def make_callback():
  closure_variables = dict(counter=0, residuals=[]) 
  def callback(residuals):
    closure_variables["counter"] += 1
    closure_variables["residuals"].append(residuals)
    print(closure_variables["counter"], residuals)
  return callback

def read_config(name):
  '''
  Read config and store in an array of shape (num_frames, num_bodies, 7).
  '''

  # Read number of lines and bodies
  N = 0
  try:
    with open(name, 'r') as f_handle:
      num_lines = 0
      N = 0
      for line in f_handle:
        if num_lines == 0:
          N = int(line)
        num_lines += 1
  except OSError:
    return np.array([])

  # Set array
  num_frames = num_lines // (N + 1) 
  x = np.zeros((num_frames, N, 7))

  # Read config
  with open(name, 'r') as f_handle:
    for k, line in enumerate(f_handle):
      if (k % (N+1)) == 0:
        continue
      else:
        data = np.fromstring(line, sep=' ')
        frame = k // (N+1)      
        i = k - 1 - (k // (N+1)) * (N+1)
        if frame >= num_frames:
          break
        x[frame, i] = np.copy(data)

  # Return config
  return x

def plot_velocity_field(grid, r_vectors_blobs, lambda_blobs, blob_radius, eta, output, tracer_radius, *args, **kwargs):
  '''
  This function plots the velocity field to a grid. 
  '''
  # Prepare grid values
  grid = np.reshape(grid, (3,3)).T
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

  # Set radius of blobs (= a) and grid nodes (= 0)
  radius_source = np.ones(r_vectors_blobs.size // 3) * blob_radius 
  radius_target = np.ones(grid_coor.size // 3) * tracer_radius

  # Compute velocity field 
  mobility_vector_prod_implementation = kwargs.get('mobility_vector_prod_implementation')
  if mobility_vector_prod_implementation == 'python':
    grid_velocity = mob.mobility_vector_product_source_target_one_wall(r_vectors_blobs, 
                                                                       grid_coor, 
                                                                       lambda_blobs, 
                                                                       radius_source, 
                                                                       radius_target, 
                                                                       eta, 
                                                                       *args, 
                                                                       **kwargs) 
  elif mobility_vector_prod_implementation == 'C++':
    grid_velocity = mob.boosted_mobility_vector_product_source_target(r_vectors_blobs, 
                                                                      grid_coor, 
                                                                      lambda_blobs, 
                                                                      radius_source, 
                                                                      radius_target, 
                                                                      eta, 
                                                                      *args, 
                                                                      **kwargs)
  elif mobility_vector_prod_implementation == 'numba':
    grid_velocity = mob.mobility_vector_product_source_target_one_wall_numba(r_vectors_blobs, 
                                                                             grid_coor, 
                                                                             lambda_blobs, 
                                                                             radius_source, 
                                                                             radius_target, 
                                                                             eta, 
                                                                             *args, 
                                                                             **kwargs) 
  else:
    grid_velocity = mob.single_wall_mobility_trans_times_force_source_target_pycuda(r_vectors_blobs, 
                                                                                    grid_coor, 
                                                                                    lambda_blobs, 
                                                                                    radius_source, 
                                                                                    radius_target, 
                                                                                    eta, 
                                                                                    *args, 
                                                                                    **kwargs) 
  
  # Prepara data for VTK writer 
  variables = [np.reshape(grid_velocity, grid_velocity.size)] 
  dims = np.array([grid_points[0]+1, grid_points[1]+1, grid_points[2]+1], dtype=np.int32) 
  nvars = 1
  vardims = np.array([3])
  centering = np.array([0])
  varnames = ['velocity\0']
  name = output + '.velocity_field.vtk'
  grid_x = grid_x - dx_grid[0] * 0.5
  grid_y = grid_y - dx_grid[1] * 0.5
  grid_z = grid_z - dx_grid[2] * 0.5
  grid_x = np.concatenate([grid_x, [grid[1,0]]])
  grid_y = np.concatenate([grid_y, [grid[1,1]]])
  grid_z = np.concatenate([grid_z, [grid[1,2]]])

  

  # Write velocity field
  visit_writer.boost_write_rectilinear_mesh(name,      # File's name
                                            0,         # 0=ASCII,  1=Binary
                                            dims,      # {mx, my, mz}
                                            grid_x,     # xmesh
                                            grid_y,     # ymesh
                                            grid_z,     # zmesh
                                            nvars,     # Number of variables
                                            vardims,   # Size of each variable, 1=scalar, velocity=3*scalars
                                            centering, # Write to cell centers of corners
                                            varnames,  # Variables' names
                                            variables) # Variables
  return


if __name__ ==  '__main__':
  # Get command line arguments
  parser = argparse.ArgumentParser(description='Solve the mobility or resistance problem'
                                   'for a multi-body suspension and save some data.')
  parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', 
                      help='name of the input file')
  parser.add_argument('--print-residual', action='store_true', help='print gmres and lanczos residuals')
  args=parser.parse_args()
  input_file = args.input_file

  # Read input file
  read = read_input.ReadInput(input_file)

  # Copy input file to output
  subprocess.call(["cp", input_file, read.output_name + '.inputfile'])

  # Create rigid bodies
  bodies = []
  body_types = []
  body_names = []
  for ID, structure in enumerate(read.structures):
    print('Creating structures = ', structure[1])
    struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
    num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])
    # Read slip file if it exists
    slip = None
    if(len(structure) > 2):
      slip = read_slip_file.read_slip_file(structure[2])
    body_types.append(num_bodies_struct)
    body_names.append(read.structures_ID[ID])
    # Creat each body of tyoe structure
    for i in range(num_bodies_struct):
      b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, read.blob_radius)
      b.mobility_blobs = multi_bodies.set_mobility_blobs(read.mobility_blobs_implementation)
      b.ID = read.structures_ID[ID]
      multi_bodies_functions.set_slip_by_ID(b, slip)
      # Add body mass
      if ID < len(read.mg_bodies):
        b.mg = read.mg_bodies[ID]
      else:
        b.mg = 0.0
      if ID < len(read.k_bodies):
        b.k = read.k_bodies[ID]
      else:
        b.k = 1.0
      if ID < len(read.R_bodies):
        b.R = read.R_bodies[ID]
      else:
        b.R = 1.0
      if ID < len(read.repulsion_strength_wall_bodies):
        b.repulsion_strength_wall = read.repulsion_strength_wall_bodies[ID]
      else:
        b.repulsion_strength_wall = 0
      if ID < len(read.debye_length_wall_bodies):
        b.debye_length_wall = read.debye_length_wall_bodies[ID]
      else:
        b.debye_length_wall = 1
      if ID < len(read.repulsion_strength_bodies):
        b.repulsion_strength = read.repulsion_strength_bodies[ID]
      else:
        b.repulsion_strength = 0
      if ID < len(read.debye_length_bodies):
        b.debye_length = read.debye_length_bodies[ID]
      else:
        b.debye_length = 1
      # Append bodies to total bodies list
      bodies.append(b)
  bodies = np.array(bodies)

  # Set some more variables
  num_of_body_types = len(body_types)
  num_bodies = bodies.size
  Nblobs = sum([x.Nblobs for x in bodies])
  multi_bodies.mobility_vector_prod = multi_bodies.set_mobility_vector_prod(read.mobility_vector_prod_implementation)
  multi_bodies_functions.calc_blob_blob_forces = multi_bodies_functions.set_blob_blob_forces(read.blob_blob_force_implementation)
  multi_bodies_functions.calc_body_body_forces_torques = multi_bodies_functions.set_body_body_forces_torques(read.body_body_force_torque_implementation)
  multi_bodies.mobility_blobs = multi_bodies.set_mobility_blobs(read.mobility_blobs_implementation)

  # Write bodies information
  with open(read.output_name + '.bodies_info', 'w') as f:
    f.write('num_of_body_types  ' + str(num_of_body_types) + '\n')
    f.write('body_names         ' + str(body_names) + '\n')
    f.write('body_types         ' + str(body_types) + '\n')
    f.write('num_bodies         ' + str(num_bodies) + '\n')
    f.write('num_blobs          ' + str(Nblobs) + '\n')

  # Calculate slip on blobs
  if multi_bodies.calc_slip is not None:
    slip = multi_bodies.calc_slip(bodies, Nblobs)
  else:
    slip = np.zeros((Nblobs, 3))

  # Read forces file
  if read.force_file is not None:
    force_torque = np.loadtxt(read.force_file)
    force_torque = np.reshape(force_torque, (2*num_bodies, 3))
  else:
    force_torque = np.zeros((2*num_bodies, 3))
    
  # Read velocity file
  if read.velocity_file is not None:
    velocity = np.loadtxt(read.velocity_file)
    velocity = np.reshape(velocity, (2*num_bodies, 3))
  else:
    velocity = np.zeros((2*num_bodies, 3))
    

  # If scheme == mobility solve mobility problem
  if read.scheme == 'mobility':
    start_time = time.time()  
    # Get blobs coordinates
    r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, Nblobs)

    # Use the code to compute force-torques on bodies if a file was not given
    if read.force_file is None:
      multi_bodies_functions.calc_body_body_forces_torques = multi_bodies_functions.set_body_body_forces_torques(read.body_body_force_torque_implementation,
                                                                                                                 stkfmm_mult_order=read.stkfmm_mult_order, 
                                                                                                                 stkfmm_pbc=read.stkfmm_pbc,
                                                                                                                 L=read.periodic_length,
                                                                                                                 comm=None,
                                                                                                                 mu=read.mu,
                                                                                                                 vacuum_permeability=read.vacuum_permeability)
      force_torque = multi_bodies_functions.force_torque_calculator_sort_by_bodies(bodies,
                                                                                   r_vectors_blobs,
                                                                                   g = read.g, 
                                                                                   repulsion_strength_wall = read.repulsion_strength_wall, 
                                                                                   debye_length_wall = read.debye_length_wall, 
                                                                                   repulsion_strength = read.repulsion_strength, 
                                                                                   debye_length = read.debye_length, 
                                                                                   mu = read.mu,
                                                                                   B0 = read.B0,
                                                                                   omega = read.omega,
                                                                                   quaternion_B = Quaternion(read.quaternion_B / np.linalg.norm(read.quaternion_B)),
                                                                                   omega_perp = read.omega_perp,
                                                                                   vacuum_permeability = read.vacuum_permeability,
                                                                                   periodic_length = read.periodic_length,
                                                                                   step = 0, 
                                                                                   dt = read.dt, 
                                                                                   domain = read.domain,
                                                                                   harmonic_confinement = read.harmonic_confinement,
                                                                                   harmonic_confinement_plane = read.harmonic_confinement_plane,
                                                                                   dipole_dipole = read.dipole_dipole) 

    # Set right hand side
    System_size = Nblobs * 3 + num_bodies * 6
    RHS = np.reshape(np.concatenate([slip, -force_torque]), (System_size))       
    
    # Set linear operators 
    linear_operator_partial = partial(multi_bodies.linear_operator_rigid, bodies=bodies, r_vectors=r_vectors_blobs, eta=read.eta, a=read.blob_radius)
    A = spla.LinearOperator((System_size, System_size), matvec = linear_operator_partial, dtype='float64')

    # Set preconditioner
    mobility_inv_blobs = []
    mobility_bodies = np.empty((len(bodies), 6, 6))
    # Loop over bodies
    for k, b in enumerate(bodies):
      # 1. Compute blobs mobility and invert it
      M = b.calc_mobility_blobs(read.eta, read.blob_radius)
      M_inv = np.linalg.inv(M)
      mobility_inv_blobs.append(M_inv)
      # 2. Compute body mobility
      N = b.calc_mobility_body(read.eta, read.blob_radius, M_inv = M_inv)
      mobility_bodies[k] = N

    # 4. Pack preconditioner
    PC_partial = partial(multi_bodies.block_diagonal_preconditioner, bodies=bodies, mobility_bodies=mobility_bodies, \
                           mobility_inv_blobs=mobility_inv_blobs, Nblobs=Nblobs)
    PC = spla.LinearOperator((System_size, System_size), matvec = PC_partial, dtype='float64')

    # Solve preconditioned linear system # callback=make_callback()
    (sol_precond, info_precond) = utils.gmres(A, RHS, tol=read.solver_tolerance, M=PC, maxiter=1000, restart=60) 
    
    # Extract velocities and constraint forces on blobs
    velocity = np.reshape(sol_precond[3*Nblobs: 3*Nblobs + 6*num_bodies], (num_bodies, 6))
    lambda_blobs = np.reshape(sol_precond[0: 3*Nblobs], (Nblobs, 3))

    # Save velocity
    name = read.output_name + '.velocity.dat'
    np.savetxt(name, velocity, delimiter='  ')
    print('Time to solve mobility problem =', time.time() - start_time )

    # Compute force-torques on bodies
    force = np.reshape(multi_bodies.K_matrix_T_vector_prod(bodies, lambda_blobs, Nblobs), (num_bodies, 6))
    
    # Save force
    name = read.output_name + '.force.dat'
    np.savetxt(name, force, delimiter='  ')

    # Plot velocity field
    if read.plot_velocity_field.size > 1: 
      print('plot_velocity_field')
      plot_velocity_field(read.plot_velocity_field, r_vectors_blobs, lambda_blobs, read.blob_radius, read.eta, read.output_name, read.tracer_radius,
                          mobility_vector_prod_implementation = read.mobility_vector_prod_implementation)
      
  # If scheme == resistance solve resistance problem 
  elif read.scheme == 'resistance': 
    start_time = time.time() 
    # Get blobs coordinates 
    r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, Nblobs) 
    
    # Calculate block-diagonal matrix K
    K = multi_bodies.calc_K_matrix(bodies, Nblobs)

    # Set right hand side
    slip += multi_bodies.K_matrix_vector_prod(bodies, velocity, Nblobs) 
    RHS = np.reshape(slip, slip.size)
    
    # Calculate mobility (M) at the blob level
    mobility_blobs = multi_bodies.mobility_blobs(r_vectors_blobs, read.eta, read.blob_radius)

    # Compute constraint forces 
    force_blobs = np.linalg.solve(mobility_blobs, RHS)

    # Compute force-torques on bodies
    force = np.reshape(multi_bodies.K_matrix_T_vector_prod(bodies, force_blobs, Nblobs), (num_bodies, 6))
    
    # Save force
    name = read.output_name + '.force.dat'
    np.savetxt(name, force, delimiter='  ')
    print('Time to solve resistance problem =', time.time() - start_time  )

    # Plot velocity field
    if read.plot_velocity_field.size > 1: 
      print('plot_velocity_field')
      lambda_blobs = np.reshape(force_blobs, (Nblobs, 3))
      plot_velocity_field(read.plot_velocity_field, r_vectors_blobs, lambda_blobs, read.blob_radius, read.eta, read.output_name, read.tracer_radius,
                          mobility_vector_prod_implementation = read.mobility_vector_prod_implementation)
  
  elif read.scheme == 'body_mobility': 
    start_time = time.time()
    r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, Nblobs)
    mobility_blobs = multi_bodies.mobility_blobs(r_vectors_blobs, read.eta, read.blob_radius)
    resistance_blobs = np.linalg.inv(mobility_blobs)
    K = multi_bodies.calc_K_matrix(bodies, Nblobs)
    resistance_bodies = np.dot(K.T, np.dot(resistance_blobs, K))
    mobility_bodies = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)))
    name = read.output_name + '.body_mobility.dat'
    np.savetxt(name, mobility_bodies, delimiter='  ')
    print('Time to compute body mobility =', time.time() - start_time)
    
  elif (read.scheme == 'plot_velocity_field' and False):
    print('plot_velocity_field')
    # Compute slip 

    # Compute forces

    # Solve mobility problem

    # Compute velocity field

  elif read.scheme == 'resistance_avg':
    start_time = time.time()  
    N_orientations = 1000
    N_distances = 200
    d_min = 1.3856
    d_max = 13.856
    d = (d_max - d_min) / (N_distances - 1)
    force_avg = np.zeros((N_distances, 13))
    force_std = np.zeros((N_distances, 13))
  
    for i_d in range(N_distances):
      # Update body location
      print('i_d = ', i_d)
      bodies[1].location[0] = d_min + d * i_d
      force_std[i_d, 0] = d_min + d * i_d
      force_avg[i_d, 0] = d_min + d * i_d

      # Loop over orientations
      for i_o in range(N_orientations):
        if False:
          theta_0 = np.random.normal(0, 1, 4)
          theta_1 = np.random.normal(0, 1, 4)
          bodies[0].orientation = Quaternion(theta_0 / np.linalg.norm(theta_0))
          bodies[1].orientation = Quaternion(theta_1 / np.linalg.norm(theta_1))
        else:
          omega_vec = np.random.normal(0, 1, 3)
          omega_vec[0:2] = 0
          quaternion_dt = Quaternion.from_rotation(omega_vec)
          bodies[0].orientation = quaternion_dt * bodies[0].orientation
          bodies[1].orientation = quaternion_dt * bodies[1].orientation
          
        # Get blobs coordinates 
        r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, Nblobs) 
    
        # Calculate block-diagonal matrix K
        K = multi_bodies.calc_K_matrix(bodies, Nblobs)

        # Set right hand side
        slip = multi_bodies.K_matrix_vector_prod(bodies, velocity, Nblobs) 
        RHS = np.reshape(slip, slip.size)
    
        # Calculate mobility (M) at the blob level
        mobility_blobs = multi_bodies.mobility_blobs(r_vectors_blobs, read.eta, read.blob_radius)

        # Compute constraint forces 
        force_blobs = np.linalg.solve(mobility_blobs, RHS)

        # Compute force-torques on bodies
        force_torque = np.reshape(multi_bodies.K_matrix_T_vector_prod(bodies, force_blobs, Nblobs), (num_bodies, 6)).flatten()
        
        # Compute averge
        force_std[i_d, 1:] += i_o * (force_torque - force_avg[i_d, 1:])**2 / (i_o + 1)
        force_avg[i_d, 1:] += (force_torque - force_avg[i_d, 1:]) / (i_o + 1)
           
    # Save force
    name = read.output_name + '.force_vs_distance.dat'
    result = np.zeros((N_distances, 25))
    result[:,0] = force_avg[:,0]
    result[:,1:13] = force_avg[:,1:]
    result[:,13:25] = np.sqrt(force_std[:,1:] / np.maximum(1, N_orientations - 1))
    
    np.savetxt(name, result, delimiter='  ', header='Columns: distance, force-torque (12 numbers), standard deviation (12 numbers)')
    print('Time to solve resistance problem =', time.time() - start_time  )


  elif read.scheme == 'body_mobility_avg':
    start_time = time.time()  
    N_orientations = 1000
    N_distances = 200
    d_min = 1.3856
    d_max = 13.856
    d = (d_max - d_min) / (N_distances - 1)
    force_avg = np.zeros((N_distances, 7))
    force_std = np.zeros((N_distances, 7))
    omega_ext = np.zeros(6)
    omega_ext[2::3] = 2 * np.pi * 10
  
    for i_d in range(N_distances):
      # Update body location
      print('i_d = ', i_d)
      bodies[1].location[0] = d_min + d * i_d
      force_std[i_d, 0] = d_min + d * i_d
      force_avg[i_d, 0] = d_min + d * i_d

      # Loop over orientations
      for i_o in range(N_orientations):
        if False:
          theta_0 = np.random.normal(0, 1, 4)
          theta_1 = np.random.normal(0, 1, 4)
          bodies[0].orientation = Quaternion(theta_0 / np.linalg.norm(theta_0))
          bodies[1].orientation = Quaternion(theta_1 / np.linalg.norm(theta_1))
        else:
          omega_vec = np.random.normal(0, 1, 3)
          omega_vec[0:2] = 0
          quaternion_dt = Quaternion.from_rotation(omega_vec)
          bodies[0].orientation = quaternion_dt * bodies[0].orientation
          bodies[1].orientation = quaternion_dt * bodies[1].orientation

        r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, Nblobs)
        mobility_blobs = multi_bodies.mobility_blobs(r_vectors_blobs, read.eta, read.blob_radius)
        resistance_blobs = np.linalg.inv(mobility_blobs)
        K = multi_bodies.calc_K_matrix(bodies, Nblobs)
        resistance_bodies = np.dot(K.T, np.dot(resistance_blobs, K))
        mobility_bodies = np.linalg.pinv(np.dot(K.T, np.dot(resistance_blobs, K)))

        M_rr = np.zeros((6, 6))
        M_rr[0:3, 0:3] = mobility_bodies[3:6, 3:6]
        M_rr[0:3, 3:6] = mobility_bodies[3:6, 9:12]
        M_rr[3:6, 0:3] = mobility_bodies[9:12, 3:6]
        M_rr[3:6, 3:6] = mobility_bodies[9:12, 9:12]

        R_tr = np.zeros((6,6))
        R_tr[0:3, 0:3] = resistance_bodies[0:3, 3:6]
        R_tr[0:3, 3:6] = resistance_bodies[0:3, 9:12]
        R_tr[3:6, 0:3] = resistance_bodies[6:9, 3:6]
        R_tr[3:6, 3:6] = resistance_bodies[6:9, 9:12]

        # Force as f = - R_tr * omega        
        force = -np.dot(R_tr, omega_ext)
      
        # Compute averge
        force_std[i_d, 1:] += i_o * (force - force_avg[i_d, 1:])**2 / (i_o + 1)
        force_avg[i_d, 1:] += (force - force_avg[i_d, 1:]) / (i_o + 1)
           
    # Save force
    name = read.output_name + '.force_vs_distance.dat'
    result = np.zeros((N_distances, 25))
    result[:,0] = force_avg[:,0]
    result[:,1:7] = force_avg[:,1:]
    result[:,7:13] = np.sqrt(force_std[:,1:] / np.maximum(1, N_orientations - 1))
    
    np.savetxt(name, result, delimiter='  ', header='Columns: distance, force (6 numbers), standard deviation (6 numbers)')
    print('Time to solve resistance problem =', time.time() - start_time  )

  elif read.scheme == 'body_mobility_trajectory':
    start_time = time.time()
    # Set parameters
    name_0 = '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run2000/run2001/run2001.41.0.0.superellipsoid.0.dat'
    name_1 = '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run2000/run2001/run2001.41.0.0.superellipsoid.1.dat'
    omega = 2 * np.pi * 10
    B0 = 1e+03
    dt = 0.0004
    N_steps = 500
    N_offset = 2
    R = (3.0 / (4 * np.pi))**(1.0 / 3.0) * 1.6
    zeta_0 = 6 * np.pi * read.eta * R
    # Use torque type: trajectory, constant_torque, constant_angular_velocity
    torque_mode = 'trajectory_v2'

    # Load trajectories
    x_0 = np.loadtxt(name_0)
    x_1 = np.loadtxt(name_1)
    force_trajectory = np.zeros((N_steps, 7))
  
    for i in range(N_offset, N_steps + N_offset):
      # Update body location
      print('i = ', i)
      bodies[0].location = x_0[i,1:4]
      bodies[1].location = x_1[i,1:4]
      bodies[0].orientation = Quaternion(x_0[i,4:])
      bodies[1].orientation = Quaternion(x_1[i,4:])

      # Compute mobility and resistance
      r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, Nblobs)
      mobility_blobs = multi_bodies.mobility_blobs(r_vectors_blobs, read.eta, read.blob_radius)
      resistance_blobs = np.linalg.inv(mobility_blobs)
      K = multi_bodies.calc_K_matrix(bodies, Nblobs)
      resistance_bodies = np.dot(K.T, np.dot(resistance_blobs, K))
      mobility_bodies = np.linalg.pinv(resistance_bodies)

      # Extract submatrices
      M_rr = np.zeros((6, 6))
      M_rr[0:3, 0:3] = mobility_bodies[3:6, 3:6]
      M_rr[0:3, 3:6] = mobility_bodies[3:6, 9:12]
      M_rr[3:6, 0:3] = mobility_bodies[9:12, 3:6]
      M_rr[3:6, 3:6] = mobility_bodies[9:12, 9:12]

      R_tr = np.zeros((6,6))
      R_tr[0:3, 0:3] = resistance_bodies[0:3, 3:6]
      R_tr[0:3, 3:6] = resistance_bodies[0:3, 9:12]
      R_tr[3:6, 0:3] = resistance_bodies[6:9, 3:6]
      R_tr[3:6, 3:6] = resistance_bodies[6:9, 9:12]

      R_tt = np.zeros((6, 6))
      R_tt[0:3, 0:3] = resistance_bodies[0:3, 0:3]
      R_tt[0:3, 3:6] = resistance_bodies[0:3, 6:9]
      R_tt[3:6, 0:3] = resistance_bodies[6:9, 0:3]
      R_tt[3:6, 3:6] = resistance_bodies[6:9, 6:9]

      # Compute torque
      torque = np.zeros((2, 3))
      t = dt * i 
      B = B0 * np.array([np.cos(omega * t), np.sin(omega * t), 0.0])
      for k, b in enumerate(bodies):
        rotation_matrix = b.orientation.rotation_matrix()
        mu_body = np.dot(rotation_matrix, read.mu)
        torque[k] = np.cross(mu_body, B)

      # Force as f = - R_tr * omega
      if torque_mode == 'constant_angular_velocity':
        omega_vec = np.zeros(6)
        omega_vec[2] = omega
        omega_vec[5] = omega
        force = -np.dot(R_tr, omega_vec)
      elif torque_mode == 'trajectory':
        force = -np.dot(R_tr, np.dot(M_rr, torque.flatten()))
      elif torque_mode == 'constant_angular_velocity_v2':
        omega_vec = np.zeros(6)
        omega_vec[2] = omega
        omega_vec[5] = omega
        force = -zeta_0 * np.dot(np.linalg.inv(R_tt), np.dot(R_tr, omega_vec))
      elif torque_mode == 'trajectory_v2':
        force = -zeta_0 * np.dot(np.linalg.inv(R_tt), np.dot(R_tr, np.dot(M_rr, torque.flatten())))
      
      # Decompose force and save
      r = bodies[1].location - bodies[0].location
      r[2] = 0
      r_perp = np.zeros(3)
      r_perp[0] = r[1]
      r_perp[1] = -r[0]
      r_perp = r_perp / np.linalg.norm(r_perp)
      r_hat = r / np.linalg.norm(r)
      z = np.zeros(3)
      z[2] = 1
      force_longitudinal_0 = np.dot(r_hat, force[0:3])
      force_longitudinal_1 = np.dot(r_hat, force[3:6])
      force_transverse_0 = np.dot(r_perp, force[0:3])
      force_transverse_1 = np.dot(r_perp, force[3:6])
      force_vertical_0 = np.dot(z, force[0:3])
      force_vertical_1 = np.dot(z, force[3:6])
      force_trajectory[i - N_offset, 0] = t - N_offset * dt
      force_trajectory[i - N_offset, 1] = force_longitudinal_0
      force_trajectory[i - N_offset, 2] = force_transverse_0
      force_trajectory[i - N_offset, 3] = force_vertical_0
      force_trajectory[i - N_offset, 4] = force_longitudinal_1
      force_trajectory[i - N_offset, 5] = force_transverse_1
      force_trajectory[i - N_offset, 6] = force_vertical_1
           
    # Save force
    name = read.output_name + '.force_vs_time.dat'
    np.savetxt(name, force_trajectory, delimiter='  ', header='Columns: time, force longitudinal, transverse, vertical, viz second particle')
    print('Time to solve resistance problem =', time.time() - start_time  )

  elif read.scheme == 'body_mobility_trajectory_many_bodies':
    start_time = time.time()
    # Set parameters
    name = '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run2000/run2000/run2000.40.3.0.shell.config'
    x = read_config(name)
    omega = 2 * np.pi * 10
    B0 = 1e+03
    dt = 0.0004
    N_steps = 500
    N_offset = 117
    R = (3.0 / (4 * np.pi))**(1.0 / 3.0) * 1.6
    zeta_0 = 6 * np.pi * read.eta * R

    # Use torque type: trajectory, constant_angular_velocity
    torque_mode = 'constant_angular_velocity'

    # Load trajectories
    force_trajectory = np.zeros((N_steps, 7))
  
    for i in range(N_offset, N_steps + N_offset):
      # Update body location
      print('i = ', i)
      for j in range(len(bodies)):
        bodies[j].location = x[i,j, 0:3]
        bodies[j].orientation = Quaternion(x[i,j, 3:])

      # Compute mobility and resistance
      r_vectors_blobs = multi_bodies.get_blobs_r_vectors(bodies, Nblobs)
      mobility_blobs = multi_bodies.mobility_blobs(r_vectors_blobs, read.eta, read.blob_radius)
      resistance_blobs = np.linalg.inv(mobility_blobs)
      K = multi_bodies.calc_K_matrix(bodies, Nblobs)
      resistance_bodies = np.dot(K.T, np.dot(resistance_blobs, K))
      mobility_bodies = np.linalg.pinv(resistance_bodies)

      # Compute torque
      torque = np.zeros((len(bodies), 3))
      t = dt * i 
      B = B0 * np.array([np.cos(omega * t), np.sin(omega * t), 0.0])
      for k, b in enumerate(bodies):
        if k < 2:
          rotation_matrix = b.orientation.rotation_matrix()
          mu_body = np.dot(rotation_matrix, read.mu)
          torque[k] = np.cross(mu_body, B)
      torque = torque.flatten()

      if torque_mode == 'constant_angular_velocity':
        # Force as f = - R_tr * omega
        velocity = np.zeros(6 * len(bodies))
        #velocity[5::6] = omega
        velocity[5] = omega
        velocity[11] = omega
        force_torque = -np.dot(resistance_bodies, velocity)
        force = np.zeros(3 * len(bodies))
        force[0::3] = force_torque[0::6]
        force[1::3] = force_torque[1::6]
        force[2::3] = force_torque[2::6]       
      elif torque_mode == 'trajectory':
        # Force as f = - R_tr * M_rr * torque
        force_torque = np.zeros(6 * len(bodies))
        force_torque[3::6] = torque[0::3]
        force_torque[4::6] = torque[1::3]
        force_torque[5::6] = torque[2::3]
        velocity = np.dot(mobility_bodies, force_torque)
        # Make linear velocity zero
        velocity[0::6] = 0
        velocity[1::6] = 0
        velocity[2::6] = 0
        force_torque = -np.dot(resistance_bodies, velocity)
        force = np.zeros(3 * len(bodies))
        force[0::3] = force_torque[0::6]
        force[1::3] = force_torque[1::6]
        force[2::3] = force_torque[2::6]       
      
      # Decompose force and save
      r = bodies[1].location - bodies[0].location
      r[2] = 0
      r_perp = np.zeros(3)
      r_perp[0] = r[1]
      r_perp[1] = -r[0]
      r_perp = r_perp / np.linalg.norm(r_perp)
      r_hat = r / np.linalg.norm(r)
      z = np.zeros(3)
      z[2] = 1
      force_longitudinal_0 = np.dot(r_hat, force[0:3])
      force_longitudinal_1 = np.dot(r_hat, force[3:6])
      force_transverse_0 = np.dot(r_perp, force[0:3])
      force_transverse_1 = np.dot(r_perp, force[3:6])
      force_vertical_0 = np.dot(z, force[0:3])
      force_vertical_1 = np.dot(z, force[3:6])
      force_trajectory[i - N_offset, 0] = t - N_offset * dt
      force_trajectory[i - N_offset, 1] = force_longitudinal_0
      force_trajectory[i - N_offset, 2] = force_transverse_0
      force_trajectory[i - N_offset, 3] = force_vertical_0
      force_trajectory[i - N_offset, 4] = force_longitudinal_1
      force_trajectory[i - N_offset, 5] = force_transverse_1
      force_trajectory[i - N_offset, 6] = force_vertical_1
           
    # Save force
    name = read.output_name + '.force_vs_time.dat'
    np.savetxt(name, force_trajectory, delimiter='  ', header='Columns: time, force longitudinal, transverse, vertical, viz second particle')
    print('Time to solve resistance problem =', time.time() - start_time  )
   

      


    
  print('\n\n\n# End')




