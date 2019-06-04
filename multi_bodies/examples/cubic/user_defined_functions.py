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
  force_torque = np.zeros((2*len(bodies), 3))
  mu = kwargs.get('mu')
  B0 = kwargs.get('B0')
  omega = kwargs.get('omega')
  quaternion_B = kwargs.get('quaternion_B')
  step = kwargs.get('step')
  dt = kwargs.get('dt')
  time = step * dt

  #print('mu           = ', mu)
  #print('B0           = ', B0)
  #print('omega        = ', omega)
  #print('quaternion_B = ', quaternion_B)
  #print('step         = ', step)
  #print('dt           = ', dt)
  #print('time         = ', time)  

  R_B = quaternion_B.rotation_matrix()
  B = B0 * np.array([np.cos(omega * time), np.sin(omega * time), 0.0])
  B = np.dot(R_B, B)

  #print('R_B = \n', R_B)
  #print('B = ', B)
  
  for k, b in enumerate(bodies):
    rotation_matrix = b.orientation.rotation_matrix()
    mu_body = np.dot(rotation_matrix, mu)
    #print('rotation_matrix = \n', rotation_matrix)
    #print('mu_body = ', mu_body)
    force_torque[2*k+1] = np.cross(mu_body, B)

  return force_torque
multi_bodies_functions.bodies_external_force_torque = bodies_external_force_torque_new





