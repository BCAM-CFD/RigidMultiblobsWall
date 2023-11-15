import argparse
import numpy as np
import scipy.linalg
import scipy.spatial as spatial
import scipy.sparse.linalg as spla
import subprocess
try:
  import pickle as cpickle
except:
  try:
    import cpickle
  except:
    import _pickle as cpickle
from functools import partial
import sys
import time
import copy
import scipy.sparse as sp
from sksparse.cholmod import cholesky
#import matplotlib.pyplot as plt

# Find project functions
found_functions = False
path_to_append = ''
sys.path.append('../../')
while found_functions is False:
    try:
      from quaternion_integrator.quaternion import Quaternion
      from Lub_Solver import Lub_Solver as LS
      from stochastic_forcing import stochastic_forcing as stochastic
      from mobility import mobility as mb
      from body import body
      from read_input import read_input
      from read_input import read_vertex_file
      from read_input import read_clones_file
      from read_input import read_slip_file
      import general_application_utils
      import multi_bodies_functions

      found_functions = True
    except ImportError:
        path_to_append += '../'
        print('searching functions in path ', path_to_append)
        sys.path.append(path_to_append)
        if len(path_to_append) > 21:
            print('\nProjected functions not found. Edit path in multi_bodies.py')
            sys.exit()

if __name__ == '__main__':
    # Get command line arguments
    parser = argparse.ArgumentParser(description='Run a multi-body simulation and save trajectory.')
    parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', help='name of the input file')
    parser.add_argument('--print-residual', action='store_true', help='print gmres and lanczos residuals')
    args = parser.parse_args()
    input_file = args.input_file

    # Read input file
    read = read_input.ReadInput(input_file)

    # Set some variables for the simulation
    eta = read.eta
    a = read.blob_radius
    output_name = read.output_name
    structures = read.structures
    structures_ID = read.structures_ID
    
    # Copy input file to output
    subprocess.call(["cp", input_file, output_name + '.inputfile'])

    # Set random generator state
    if read.random_state is not None:
      with open(read.random_state, 'rb') as f:
        np.random.set_state(cpickle.load(f))
    elif read.seed is not None:
      np.random.seed(int(read.seed))
    
    # Save random generator state
    with open(output_name + '.random_state', 'wb') as f:
      cpickle.dump(np.random.get_state(), f)

    # Create rigid bodies
    bodies = []
    body_types = []
    body_names = []
    for ID, structure in enumerate(structures):
      print('Creating structures = ', structure[1])
      # Read vertex and clones files
      struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
      num_bodies_struct, struct_locations, struct_orientations = read_clones_file.read_clones_file(structure[1])
      # Read slip file if it exists
      slip = None
      if(len(structure) > 2):
        slip = read_slip_file.read_slip_file(structure[2])
      body_types.append(num_bodies_struct)
      body_names.append(structures_ID[ID])
      # Create each body of type structure
      for i in range(num_bodies_struct):
        b = body.Body(struct_locations[i], struct_orientations[i], struct_ref_config, a)
        b.ID = structures_ID[ID]
        # Calculate body length for the RFD
        if i == 0:
          b.calc_body_length()
        else:
          b.body_length = bodies[-1].body_length
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
    num_particles = len(bodies)
    Nblobs = sum([x.Nblobs for x in bodies])
    cutoff = read.Lub_Cut         
    n_steps = read.n_steps 
    n_save = read.n_save
    initial_step = read.initial_step
    dt = read.dt 

    # Create integrator
    LSolv = LS(bodies,
               a,
               eta,cutoff,
               read.periodic_length,
               firm_delta=read.firm_delta,
               domain=read.domain,
               mobility_vector_prod_implementation=read.mobility_vector_prod_implementation)
    LSolv.dt = dt
    LSolv.kT = read.kT
    LSolv.tolerance = read.solver_tolerance
    LSolv.print_residual = args.print_residual
    LSolv.output_name = read.output_name
    LSolv.n_save = n_save
    
    multi_bodies_functions.calc_blob_blob_forces = multi_bodies_functions.set_blob_blob_forces(read.blob_blob_force_implementation)
    multi_bodies_functions.calc_body_body_forces_torques = multi_bodies_functions.set_body_body_forces_torques(read.body_body_force_torque_implementation)   
    FT_calc = partial(multi_bodies_functions.force_torque_calculator_sort_by_bodies, 
                      g = read.g, 
                      firm_delta = read.firm_delta,
                      repulsion_strength_wall = read.repulsion_strength_wall,
                      debye_length_wall = read.debye_length_wall,
                      repulsion_strength = read.repulsion_strength, 
                      debye_length = read.debye_length, 
                      periodic_length = read.periodic_length,
                      eta = eta,
                      a = a,
                      mu = read.mu,
                      B0 = read.B0,
                      omega = read.omega,
                      quaternion_B = Quaternion(read.quaternion_B / np.linalg.norm(read.quaternion_B)),
                      omega_perp = read.omega_perp,
                      vacuum_permeability = read.vacuum_permeability,
                      omega_one_roller = read.omega_one_roller,
                      harmonic_confinement = read.harmonic_confinement,
                      harmonic_confinement_plane = read.harmonic_confinement_plane, 
                      dipole_dipole = read.dipole_dipole)  
    
    if True:
      # Set lubrication matrices in sparse csc format for the class members
      t0 = time.time()
      LSolv.Set_R_Mats()
      dt1 = time.time() - t0
      print(("Make R mats time : %s" %dt1))

    # Open config files
    if True:
      output_files = []
      buffering = max(1, min(body_types) * n_steps // n_save // 200)
      ID_loop = read.structures_ID + read.articulated_ID
      for i, ID in enumerate(ID_loop):
        name = output_name + '.' + ID + '.config'
        output_files.append(open(name, 'w', buffering=buffering))

    # Time loop
    start_time = time.time()
    for n in range(initial_step, n_steps): 
      # Save data if...
      if (n % n_save) == 0 and n >= 0:
        elapsed_time = time.time() - start_time
        print('Integrator = ', read.scheme, ', step = ', n, ',  wallclock time = ', time.time() - start_time)
        body_offset = 0
        for i, f_ID in enumerate(output_files):
          f_ID.write(str(body_types[i]) + '\n')
          for j in range(body_types[i]):
            orientation = bodies[body_offset + j].orientation.entries
            f_ID.write('%s %s %s %s %s %s %s\n' % (bodies[body_offset + j].location[0],
                                                   bodies[body_offset + j].location[1],
                                                   bodies[body_offset + j].location[2],
                                                   orientation[0],
                                                   orientation[1],
                                                   orientation[2],
                                                   orientation[3]))
          body_offset += body_types[i]

      # Advance time step
      LSolv.Update_Bodies_Trap(FT_calc, step=n)
     



    # Save data if...
    if ((n+1) % n_save) == 0 and n >= 0:
      elapsed_time = time.time() - start_time
      print('Integrator = ', read.scheme, ', step = ', n+1, ',  wallclock time = ', time.time() - start_time)
      body_offset = 0
      for i, f_ID in enumerate(output_files):
        f_ID.write(str(body_types[i]) + '\n')
        for j in range(body_types[i]):
          orientation = bodies[body_offset + j].orientation.entries
          f_ID.write('%s %s %s %s %s %s %s\n' % (bodies[body_offset + j].location[0],
                                                 bodies[body_offset + j].location[1],
                                                 bodies[body_offset + j].location[2],
                                                 orientation[0],
                                                 orientation[1],
                                                 orientation[2],
                                                 orientation[3]))
        body_offset += body_types[i]
