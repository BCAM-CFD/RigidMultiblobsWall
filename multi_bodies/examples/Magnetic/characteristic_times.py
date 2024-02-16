'''
Utility to print some characteristic times of the problem.

The parameters units should be consistent, for example, we give
length, time and mass in units of microns, seconds and miligrams.
'''
import numpy as np


if __name__ == '__main__':
  # Set physical parameters
  vacuum_permeability = 1.25663706143592e+06
  external_magnetic_field = 1e+03
  dipole = 1e-03
  eta = 1e-03
  colloid_radius = 1.0
  omega_magnetic = 2 * np.pi  

  # Set numerical parameters
  repulsion_strength = 0.1
  debye_length = 0.243553056072
  displacement_length = 0.243553056072

  
  # Compute maximum force and torque magnitude
  force_magnetic = 3 * vacuum_permeability * dipole**2 / (4 * np.pi * colloid_radius**4)
  torque_magnetic = external_magnetic_field * dipole
  force_steric = repulsion_strength / debye_length

  # Compute mobilities
  mu_tt = 1.0 / (6 * np.pi * eta * colloid_radius)
  mu_rr = 1.0 / (8 * np.pi * eta * colloid_radius**3)

  # Compute critical frequency that colloid can follow
  omega_c = mu_rr * torque_magnetic

  # Compute time scales
  time_rotation = 2 * np.pi / omega_magnetic
  time_rotation_c = 2 * np.pi / omega_c  
  time_force_magnetic = displacement_length / (mu_tt * force_magnetic)
  time_force_steric = displacement_length / (mu_tt * force_steric)

  # Print results
  print('force_magnetic      = ', force_magnetic)  
  print('force_steric        = ', force_steric, '\n')
  print('omega_magnetic      = ', omega_magnetic)
  print('omega_c             = ', omega_c, '\n')
  print('time_rotation       = ', time_rotation)
  print('time_rotation_c     = ', time_rotation_c)  
  print('time_force_magnetic = ', time_force_magnetic)
  print('time_force_steric   = ', time_force_steric)

  
