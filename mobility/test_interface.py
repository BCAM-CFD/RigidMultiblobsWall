import numpy as np
import scipy.linalg as scla
import sys
sys.path.append('../')
sys.path.append('./')
import mobility as mob

if __name__ == '__main__':
  # Set parameters
  eta_ratio = 1e-01
  eta = 7.0
  a = 1.3
  N = 1000
  phi = 0.25
  L = (4 * np.pi * N / (3 * phi))**(1.0/3.0) * a
  print('L = ', L)
  print('N = ', N)
  print('\n\n')

  # Generate random config
  r_vectors = np.random.rand(N, 3) * L 
  force = np.random.randn(N, 3)
  if N < 7:
    print('r_vectors = ', r_vectors)
    print('force = ', force)
    print('\n\n')

  # Compute velocity interface
  M = mob.fluid_interface_mobility_cpp(r_vectors, eta, a, eta_ratio)
  vel_dense = np.dot(M, force.flatten())
  vel = mob.fluid_interface_mobility_trans_times_force_cpp(r_vectors, force, eta, a, eta_ratio)
  
  
  # Compute velocity wall
  vel_wall = mob.single_wall_mobility_trans_times_force_cpp(r_vectors, force, eta, a)

  # Compute differences
  diff_interface = np.linalg.norm(vel - vel_dense)
  diff_wall = np.linalg.norm(vel - vel_wall)
  print('diff interface relative = ', diff_interface / np.linalg.norm(vel_dense))
  print('diff interface          = ', diff_interface)
  print('diff wall relative      = ', diff_wall / np.linalg.norm(vel_wall))
  print('diff wall               = ', diff_wall)
  print('\n\n')

  if N < 7:
    # Print velocities
    np.set_printoptions(precision=12)
    print('vel_dense = \n', vel_dense)
    print('vel       = \n', vel)
    print('vel_wall  = \n', vel_wall)

  # Compute cho
  if True:
    np.set_printoptions(precision=5)
    if N < 7:
      print('M = \n', M)
    L, lower = scla.cho_factor(M)
