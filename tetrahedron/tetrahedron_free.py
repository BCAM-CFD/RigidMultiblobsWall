'''
A free tetrahedron is allowed to diffuse in a domain with a single
wall (below the tetrahedron) in the presence of gravity and a quadratic potential
repelling from the wall.  This script bins the heights of the vertices
of the tetrahedron for comparison to equilibrium.
'''
import sys
import os
import numpy as np
import math
import cPickle
import time
import argparse
import cProfile, StringIO, pstats
import logging

import tetrahedron as tdn
from fluids import mobility as mb
from quaternion_integrator.quaternion import Quaternion
from quaternion_integrator.quaternion_integrator import QuaternionIntegrator
from utils import static_var
from utils import StreamToLogger


ETA = 1.0   # Fluid viscosity.
A = 0.5     # Particle Radius.
H = 3.5     # Initial Distance to wall.
KT = 0.2    # Temperature

# Masses of particles. g = 1.
M1 = 0.005
M2 = 0.015
M3 = 0.01
M4 = 0.03

# Repulsion strength and cutoff.  
# Must be strong enough to prevent particles from passing 
# through the wall
REPULSION_STRENGTH = 2.0
DEBYE_LENGTH = 0.25  # This is the Debye length for Yukawa, TODO: rename.

def free_tetrahedron_mobility(location, orientation):
  ''' 
  Wrapper for torque mobility that takes a quaternion and location for
  use with quaternion_integrator. 
  '''
  r_vectors = get_free_r_vectors(location[0], orientation[0])
  return force_and_torque_mobility(r_vectors, location[0])


def force_and_torque_mobility(r_vectors, location):
  '''
  Calculate the mobility: (torque, force) -> (angular velocity, velocity) at position 
  In this case, position has orientation and location data, each of length 1.
  The mobility is equal to the inverse of: 
    [ J^T M^-1 J,   J^T M^-1 R ]
    [ R^T M^-1 J,   R^T M^-1 R ]
  where R is 3N x 3 (9 x 3) Rx = r cross x and J is a 3N x 3 matrix with 
  each 3x3 block being the identity.
  r is the distance from the fixed vertex of the tetrahedron to
  each other vertex (a length 3N vector).
  M (3N x 3N) is the finite size single wall mobility taken from the
  Swan and Brady paper:
   "Simulation of hydrodynamically interacting particles near a no-slip
    boundary."
  Here location is the dereferenced list with 3 entries.
  '''  
  mobility = mb.boosted_single_wall_fluid_mobility(r_vectors, ETA, A)
  rotation_matrix = calc_free_rot_matrix(r_vectors, location)
  J = np.concatenate([np.identity(3), np.identity(3), np.identity(3)])
  J_rot_combined = np.concatenate([J, rotation_matrix], axis=1)
  total_mobility = np.linalg.inv(np.dot(J_rot_combined.T,
                                        np.dot(np.linalg.inv(mobility),
                                               J_rot_combined)))
  return total_mobility


def get_free_r_vectors(location, quaternion):
  ''' Calculate r_i from a given quaternion. 
  The initial configuration is hard coded here but can be changed by
  considering an initial quaternion not equal to the identity rotation.
  initial configuration (top down view, the top vertex is fixed at the origin):

                         O r_1 = (0, 2/sqrt(3), -(2 sqrt(2))/3)
                        / \
                       /   \
                      /     \
                     /   O(location, everything else relative to this location)
                    /          \
                   /            \
               -> O--------------O  r_3 = (1, -1/sqrt(3),-(2 sqrt(2))/3)
             /
           r_2 = (-1, -1/sqrt(3),-(2 sqrt(2))/3)

  Each side of the tetrahedron has length 2.
  location is a 3-dimensional list giving the location of the "top" vertex.
  quaternion is a quaternion representing the tetrahedron orientation.
  '''
  initial_r1 = np.array([0., 2./np.sqrt(3.), -2.*np.sqrt(2.)/np.sqrt(3.)])
  initial_r2 = np.array([-1., -1./np.sqrt(3.), -2.*np.sqrt(2.)/np.sqrt(3.)])
  initial_r3 = np.array([1., -1./np.sqrt(3.), -2.*np.sqrt(2.)/np.sqrt(3.)])
  
  rotation_matrix = quaternion.rotation_matrix()

  r1 = np.dot(rotation_matrix, initial_r1) + np.array(location)
  r2 = np.dot(rotation_matrix, initial_r2) + np.array(location)
  r3 = np.dot(rotation_matrix, initial_r3) + np.array(location)
  
  return [r1, r2, r3]


def get_free_center_of_mass(location, orientation):
    '''
    Find the center of mass given the location of the top vertex, 
    and the orientation given as a quaternion.  This assumes masses are
    M1, M2, M3, M4 for r1, r2, r3, location respectively.
    '''
    r_vectors = get_free_r_vectors(location, orientation)
    center_of_mass = (np.array(r_vectors[0])*M1 + np.array(r_vectors[1])*M2 + 
                      np.array(r_vectors[2])*M3 + np.array(location)*M4)
    center_of_mass = center_of_mass/(M1 + M2 + M3 + M4)
    return center_of_mass


def calc_free_rot_matrix(r_vectors, location):
  ''' 
  Calculate rotation matrix (r cross) based on the free tetrahedron.
  In this case, the R vectors point from the "top" vertex to the others.
  '''
  rot_matrix = None
  for k in range(len(r_vectors)):
    # Current r cross x matrix block.
    adjusted_r_vector = r_vectors[k] - location
    block = np.array(
        [[0.0, adjusted_r_vector[2], -1.*adjusted_r_vector[1]],
        [-1.*adjusted_r_vector[2], 0.0, adjusted_r_vector[0]],
        [adjusted_r_vector[1], -1.*adjusted_r_vector[0], 0.0]])

    if rot_matrix is None:
      rot_matrix = block
    else:
      rot_matrix = np.concatenate([rot_matrix, block], axis=0)

  return rot_matrix


def free_gravity_torque_calculator(location, orientation):
  ''' 
  Calculate torque based on location, given as a length 1 list of 
  a 3-vector, and orientation, given as a length
  1 list of quaternions (1 quaternion).  This assumes the masses
  of particles 1, 2, and 3 are M1, M2, M3, and M4 respectively.
  '''
  r_vectors = get_free_r_vectors(location[0], orientation[0])
  R = calc_free_rot_matrix(r_vectors, location[0])
  # Gravity.
  g = np.array([0., 0., -1.*M1, 
                0., 0., -1.*M2,
                0., 0., -1.*M3])
  # Add repulsion from wall.
  for k in range(3):
    h = r_vectors[k][2]
    g[3*k + 2] += (REPULSION_STRENGTH*((h - A)/DEBYE_LENGTH + 1)*
                   np.exp(-1.*(h - A)/DEBYE_LENGTH)/((h - A)**2))
  return np.dot(R.T, g)


def free_gravity_force_calculator(location, orientation):
  '''
  Calculate force on tetrahedron given it's location and
  orientation.  
  args: 
  location:   list of length 1, only entry is a list of
              length 3 with coordinates of tetrahedon "top" vertex.
  orientation: list of length 1, only entry is a quaternion with the 
               tetrahedron orientation
  '''
  r_vectors = get_free_r_vectors(location[0], orientation[0])
  # Add repulsion of 'top' vertex, at location.
  h = location[0][2]
  repulsion_force = np.array([0., 0., 
                     (REPULSION_STRENGTH*((h - A)/DEBYE_LENGTH + 1)*
                     np.exp(-1.*(h - A)/DEBYE_LENGTH)/((h - A)**2))])
  # Add repulsion of other particles:
  for k in range(3):
    h = r_vectors[k][2]
    repulsion_force[2] += (REPULSION_STRENGTH*((h - A)/DEBYE_LENGTH + 1)*
                           np.exp(-1.*(h - A)/DEBYE_LENGTH)/((h - A)**2))
  gravity_force = np.array([0., 0., -1.*(M1 + M2 + M3 + M4)])
  return repulsion_force + gravity_force

@static_var('max_index', 0)
def bin_free_particle_heights(location, orientation, bin_width, 
                              height_histogram):
  '''Bin heights of the free particle based on a location and an orientaiton.'''
  r_vectors = get_free_r_vectors(location, orientation)
  for k in range(3):
    # Bin each particle height.
    idx = (int(math.floor((r_vectors[k][2])/bin_width)))
    if idx < len(height_histogram[k]):
      height_histogram[k][idx] += 1
    else:
      if idx > bin_free_particle_heights.max_index:
        bin_free_particle_heights.max_index = idx
        print "New maximum Index  %d is beyond histogram length " % idx
      

@static_var('samples', 0)  
@static_var('accepts', 0)  
def generate_free_equilibrium_sample():
  '''
  Generate an equilibrium sample of location and orientation, according
  to the distribution exp(-\beta U(heights)).
  Do this by generating a uniform quaternion and exponential location, 
  then accept/rejecting with the appropriate probability.
  
  DEPRECATED AND SHOULD NOT BE USED!!!
  This uses the old potential still, not
  the Yukawa potential.  Use the mcmc version below (which is faster anyway)
  '''
  progress_logger = logging.getLogger('Progress Logger')
  max_gibbs_term = 0.
  while True:
    generate_free_equilibrium_sample.samples += 1
    # First generate a uniform quaternion on the 4-sphere.
    theta = np.random.normal(0., 1., 4)
    theta = Quaternion(theta/np.linalg.norm(theta))
    # For location, set x and y to 0, since these are not affected
    # by the potential at all. Generate z coordinate as an exponential 
    # random variable.
    phi = np.random.uniform(0.0, 1.0)
    M = M1 + M2 + M3 + M4
    z_coord = -1.*np.log(phi)/M
    location = [0., 0., z_coord]
    r_vectors = get_free_r_vectors(location, theta)
    if ((r_vectors[0][2] > 0) and
        (r_vectors[1][2] > 0) and
        (r_vectors[2][2] > 0)):
      # Potential minus (M1 + M2 + M3 + M4)*z_coord because that part of the
      # distribution is handled by the exponential variable.
      U = (M1*(r_vectors[0][2] - z_coord) + M2*(r_vectors[1][2] - z_coord) +
           M3*(r_vectors[2][2] - z_coord))
      if z_coord < DEBYE_LENGTH:
        U += 0.5*REPULSION_STRENGTH*(DEBYE_LENGTH - z_coord)**2
      for k in range(3):
        if r_vectors[k][2] < DEBYE_LENGTH:
          U += 0.5*REPULSION_STRENGTH*(DEBYE_LENGTH - r_vectors[k][2])**2
      # Normalize so acceptance probability < 1.  The un-normalized probability
      # is definitely below exp(2M), but in fact it can never reach this because not
      # all particles can be 2 above location. Here 1.8 is determined 
      # experimentally to give more accepts without giving an acceptance 'probability' 
      # above 1 (at least not often).
      normalization_constant = np.exp(1.8*(M1 + M2 + M3)/KT)
      gibbs_term = np.exp(-1.*U/KT)
      if gibbs_term > max_gibbs_term:
        max_gibbs_term = gibbs_term
      accept_prob = gibbs_term/normalization_constant
      if accept_prob > 1:
        progress_logger.warning('Acceptance probability > 1.')
        progress_logger.warning('accept_prob = %f' % accept_prob)
        progress_logger.warning('z_coord is %f' % z_coord)
      if np.random.uniform() < accept_prob:
        generate_free_equilibrium_sample.accepts += 1
        return [location, theta]

@static_var('samples', 0)  
@static_var('accepts', 0)  
@static_var('dt', 0.15)
@static_var('last_trial', 0)
def generate_free_equilibrium_sample_mcmc(current_sample):
  '''
  Generate an equilibrium sample of location and orientation, according
  to the distribution exp(-\beta U(heights)) by using MCMC.
  '''
  rho = 0.98  # Adaptive parameter for dt.
  generate_free_equilibrium_sample_mcmc.samples += 1
  location = current_sample[0]
  orientation = current_sample[1]
  # Tune this dt parameter to try to achieve acceptance rate of ~50%.
  if generate_free_equilibrium_sample_mcmc.last_trial == 1:
    generate_free_equilibrium_sample_mcmc.dt /= rho
  else:
    generate_free_equilibrium_sample_mcmc.dt *= rho
  dt = generate_free_equilibrium_sample_mcmc.dt

  # Take a step using Metropolis.
  omega = np.random.normal(0., 1., 3)
  velocity = np.random.normal(0., 1., 3)
  orientation_increment = Quaternion.from_rotation(omega*dt)
  new_orientation = orientation_increment*orientation
  new_location = location + velocity*dt
  accept_probability = (gibbs_boltzmann_distribution(new_location,
                                                     new_orientation)/
                        gibbs_boltzmann_distribution(location,
                                                     orientation))
  if np.random.uniform() < accept_probability:
    generate_free_equilibrium_sample_mcmc.accepts += 1
    generate_free_equilibrium_sample_mcmc.last_trial = 1
    return [new_location, new_orientation]
  else:
    generate_free_equilibrium_sample_mcmc.last_trial = 0
    return [location, orientation]
                          
@static_var('low_rejections', 0)
def gibbs_boltzmann_distribution(location, orientation):
  '''
  Evaluate the equilibrium distribution at a given location 
  and orientation.
  '''
  r_vectors = get_free_r_vectors(location, orientation)
  if ((r_vectors[0][2] < 0.5) or
      (r_vectors[1][2] < 0.5) or
      (r_vectors[2][2] < 0.5) or
      location[2] < 0.5):
    gibbs_boltzmann_distribution.low_rejections += 1
    return 0.0
  # Calculate potential.
  U = (M1*(r_vectors[0][2]) + M2*(r_vectors[1][2]) +
       M3*(r_vectors[2][2]) + M4*(location[2]))
  U += (REPULSION_STRENGTH*np.exp(-1.*(location[2] - A)/DEBYE_LENGTH)/
        (location[2] - A))
  for k in range(3):
    h = r_vectors[k][2]
    U += (REPULSION_STRENGTH*np.exp(-1.*(h - A)/DEBYE_LENGTH)/
          (h - A))
  return np.exp(-1.*U/KT)


def check_particles_above_wall(location, orientation):
  ''' 
  Check that the particles at position aren't below the wall.
  This ads some buffer room to also reject moves that would end up
  pushing the particles very far from the wall.
  '''
  r_vectors = get_free_r_vectors(location[0], orientation[0])
  if location[0][2] < (A + 0.1):
    return False
  for k in range(3):
    if r_vectors[k][2] < (A + 0.1): 
      return False
  return True



if __name__ == '__main__':
  # Get command line arguments.
  parser = argparse.ArgumentParser(description='Run Simulation of free '
                                   'tetrahedron with Fixman, EM, and RFD '
                                   'schemes, and bin the resulting '
                                   'height distribution.  Tetrahedron is '
                                   'affected by gravity, and repulsed from '
                                   'the wall gently.')
  parser.add_argument('-dt', dest='dt', type=float,
                      help='Timestep to use for runs.')
  parser.add_argument('-N', dest='n_steps', type=int,
                      help='Number of steps to take for runs.')
  parser.add_argument('--data-name', dest='data_name', type=str,
                      default='',
                      help='Optional name added to the end of the '
                      'data file.  Useful for multiple runs '
                      '(--data_name=run-1).')
  parser.add_argument('--profile', dest='profile', type=bool, default=False,
                      help='True or False: Do we profile this run or not.')

  args=parser.parse_args()
  if args.profile:
    pr = cProfile.Profile()
    pr.enable()

  # Get command line parameters
  dt = args.dt
  n_steps = args.n_steps
  print_increment = max(int(n_steps/20.), 1)

  # Set up logging.
  # Make directory for logs if it doesn't exist.
  if not os.path.isdir(os.path.join(os.getcwd(), 'logs')):
    os.mkdir(os.path.join(os.getcwd(), 'logs'))

  log_filename = './logs/free-tetrahedron-dt-%f-N-%d-%s.log' % (
    dt, n_steps, args.data_name)
  progress_logger = logging.getLogger('Progress Logger')
  progress_logger.setLevel(logging.INFO)
  # Add the log message handler to the logger
  logging.basicConfig(filename=log_filename,
                      level=logging.INFO,
                      filemode='w')
  sl = StreamToLogger(progress_logger, logging.INFO)
  sys.stdout = sl
  sl = StreamToLogger(progress_logger, logging.ERROR)
  sys.stderr = sl

  # Script to run the various integrators on the quaternion.
  initial_location = [[0., 0., H]]
  initial_orientation = [Quaternion([1., 0., 0., 0.])]
  fixman_integrator = QuaternionIntegrator(free_tetrahedron_mobility,
                                           initial_orientation, 
                                           free_gravity_torque_calculator, 
                                           has_location=True,
                                           initial_location=initial_location,
                                           force_calculator=
                                           free_gravity_force_calculator)
  fixman_integrator.kT = KT
  fixman_integrator.check_function = check_particles_above_wall
  rfd_integrator = QuaternionIntegrator(free_tetrahedron_mobility,
                                        initial_orientation, 
                                        free_gravity_torque_calculator, 
                                        has_location=True,
                                        initial_location=initial_location,
                                        force_calculator=
                                        free_gravity_force_calculator)
  rfd_integrator.kT = KT
  rfd_integrator.check_function = check_particles_above_wall

  sample = [initial_location[0], initial_orientation[0]]
  # For now hard code bin width.  Number of bins is equal to 6./bin_width.
  # Here we allow for a large range because the tetrahedron is free to drift away 
  # from the wall a bit.
  bin_width = 1./5.
  fixman_heights = np.array([np.zeros(int(40./bin_width)) for _ in range(3)])
  rfd_heights = np.array([np.zeros(int(40./bin_width)) for _ in range(3)])
  equilibrium_heights = np.array([np.zeros(int(40./bin_width)) for _ in range(3)])

  start_time = time.time()
  for k in range(n_steps):
    # Fixman step and bin result.
    fixman_integrator.fixman_time_step(dt)
    bin_free_particle_heights(fixman_integrator.location[0],
                              fixman_integrator.orientation[0], 
                              bin_width, 
                              fixman_heights)

    # RFD step and bin result.
    rfd_integrator.rfd_time_step(dt)
    bin_free_particle_heights(rfd_integrator.location[0],
                              rfd_integrator.orientation[0], 
                              bin_width, 
                              rfd_heights)

    # Bin equilibrium sample.
    sample = generate_free_equilibrium_sample_mcmc(sample)
    bin_free_particle_heights(sample[0], 
                              sample[1],
                              bin_width, 
                              equilibrium_heights)

    if k % print_increment == 0:
      elapsed_time = time.time() - start_time
      if elapsed_time < 60.:
        progress_logger.info('At step: %d. Time Taken: %.2f Seconds' % 
                             (k, float(elapsed_time)))
        if k > 0:
          progress_logger.info('Estimated Total time required: %.2f Seconds.' %
                               (elapsed_time*float(n_steps)/float(k)))
      else:
        progress_logger.info('At step: %d. Time Taken: %.2f Minutes.' %
                             (k, (float(elapsed_time)/60.)))
        if k > 0:
          progress_logger.info('Estimated Total time required: %.2f Minutes.' %
                               (elapsed_time*float(n_steps)/float(k)/60.))

  elapsed_time = time.time() - start_time
  if elapsed_time > 60:
    progress_logger.info('Finished timestepping. Total Time: %.2f minutes.' % 
                         (float(elapsed_time)/60.))
  else:
    progress_logger.info('Finished timestepping. Total Time: %.2f seconds.' % 
                         float(elapsed_time))

  progress_logger.info('Fixman Rejection rate: %s' % 
                       (float(fixman_integrator.rejections)/
                        float(fixman_integrator.rejections + n_steps)))
  progress_logger.info('RFD Rejection rate: %s' % 
                       (float(rfd_integrator.rejections)/
                        float(rfd_integrator.rejections + n_steps)))
  progress_logger.info('Equilibrium Acceptance Rate: %s' % 
                       (float(generate_free_equilibrium_sample_mcmc.accepts)/
                       float(generate_free_equilibrium_sample_mcmc.samples)))



  # Gather data to save.
  heights = [fixman_heights/(n_steps*bin_width),
             rfd_heights/(n_steps*bin_width),
             equilibrium_heights/(n_steps*bin_width)]

  height_data = dict()
  # Save parameters just in case they're useful in the future.
  # TODO: Make sure you check all parameters when plotting to avoid
  # issues there.
  height_data['params'] = {'A': A, 'ETA': ETA, 'H': H, 'M1': M1, 'M2': M2, 
                           'M3': M3}
  height_data['heights'] = heights
  fixman_lengths = max([len(fixman_heights[k]) 
                        for k in range(len(fixman_heights))])

  height_data['buckets'] = (bin_width*np.array(range(fixman_lengths))
                            + 0.5*bin_width)
  height_data['names'] = ['Fixman', 'RFD', 'Gibbs-Boltzmann']

  # Make directory for data if it doesn't exist.
  if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
    os.mkdir(os.path.join(os.getcwd(), 'data'))

  # Optional name for data provided
  if len(args.data_name) > 0:
    data_name = './data/free-tetrahedron-dt-%g-N-%d-%s.pkl' % (
      dt, n_steps, args.data_name)
  else:
    data_name = './data/free-tetrahedron-dt-%g-N-%d.pkl' % (dt, n_steps)

  with open(data_name, 'wb') as f:
    cPickle.dump(height_data, f)
  
  if args.profile:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
