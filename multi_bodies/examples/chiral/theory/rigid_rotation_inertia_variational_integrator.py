'''
Code to simulate the rotation of a rigid body with inertia.
We use a variational integrator from Lee2005.
'''
import numpy as np
import scipy.optimize as scop
import time
import sys
sys.path.append('../../../../')
from body import body
from quaternion_integrator.quaternion import Quaternion


def torque_constnat(T0, omega_B, t, phi, R, dipole_0):
  return np.array([0., 0., T0])

def torque_dipole(T0, omega_B, t, phi, R, dipole_0):
  dipole = np.dot(R, dipole_0)
  angle = w*t + phi
  return T0 * np.cross(dipole, np.array([-np.sin(angle), np.cos(angle), 0]))

def skew_matrix(x):
  return np.array([0, -x[2], x[1],
                   x[2], 0, -x[0],
                   -x[1], x[0], 0]).reshape(3,3)


def quaternion_from_R(R):
  '''
  Using the Shepperd's method.
  '''
  # Vector of solutions
  u = np.zeros((4,4))
  
  # Option 1
  norm2_1 = 1 + R[0,0] + R[1,1] + R[2,2]
  norm = np.sqrt(norm2_1) if norm2_1 > 1e-06 else 0
  norm_inv = 1.0 / norm if norm2_1 > 1e-06 else 0
  u[0] = 0.5 * np.array([norm, (R[2,1] - R[1,2]) * norm_inv, (R[0,2] - R[2,0]) * norm_inv, (R[1,0] - R[0,1]) * norm_inv])

  # Option 2
  norm2_2 = 1 + R[0,0] - R[1,1] - R[2,2]
  norm = np.sqrt(norm2_2) if norm2_2 > 1e-06 else 0  
  norm_inv = 1.0 / norm if norm2_2 > 1e-06 else 0
  u[1] = 0.5 * np.array([(R[2,1] - R[1,2]) * norm_inv, norm, (R[0,1] + R[1,0]) * norm_inv, (R[2,0] + R[0,2]) * norm_inv])
  
  # Option 3
  norm2_3 = 1 - R[0,0] + R[1,1] - R[2,2]
  norm = np.sqrt(norm2_3) if norm2_3 > 1e-06 else 0  
  norm_inv = 1.0 / norm if norm2_3 > 1e-06 else 0
  u[2] = 0.5 * np.array([(R[0,2] - R[2,0]) * norm_inv, (R[0,1] + R[1,0]) * norm_inv, norm, (R[1,2] + R[2,1]) * norm_inv])

  # Option 4
  norm2_4 = 1 - R[0,0] - R[1,1] + R[2,2]
  norm = np.sqrt(norm2_4) if norm2_4 > 1e-06 else 0  
  norm_inv = 1.0 / norm if norm2_4 > 1e-06 else 0
  u[3] = 0.5 * np.array([(R[1,0] - R[0,1]) * norm_inv, (R[2,0] + R[0,2]) * norm_inv, (R[2,1] + R[1,2]) * norm_inv, norm])

  # Choose one
  direction = np.array([R[0,0] + R[1,1] + R[2,2], R[0,0], R[1,1], R[2,2]])
  projections = np.einsum('ij,j->i', u, direction)
  index = np.argsort(np.abs(projections))[-1]

  if True:
    print('norm2_1 = ', norm2_1)
    print('norm2_2 = ', norm2_2)
    print('norm2_3 = ', norm2_3)
    print('norm2_4 = ', norm2_4)
    print('direc   = ', direction)
    print('proj    = ', projections)
    print('index   = ', index)
    print('u[i]    = ', u[index])
    print('|u|     = ', np.linalg.norm(u, axis=1))
    print('u       = \n', u)
  
  return u[index]
  
if __name__ == '__main__':
  # Set parameters
  start = time.time()
  output_name = '/home/fbalboa/simulations/RigidMultiblobsWall/chiral/data/run3000/run3060/run3060.2.0.2'
  Jxx = 2
  Jyy = 0.5
  Jzz = 1
  perturbation = 0
  angular_momentum = np.array([0.01, 0.03, 1]) 
  phi = 0
  T0 = 0
  omega_z = 1
  omega_B = 0
  dipole_0 = np.array([0, 1, 0])
  torque_type = 'constant'
  dt = 2 * np.pi * 0.0025
  tol = 1e-12
  num_steps = 8000
  n_save = 1
  verbose = 1
  
  # Set tensor of inertia and initial freq
  J = np.zeros((3,3))
  J[0,0] = Jxx
  J[1,1] = Jyy
  J[2,2] = Jzz
  J_inv = np.linalg.inv(J)
  Jd = np.trace(J) * np.eye(3) - J

  # Set torque function
  if torque_type == 'constant':
    torque_func = torque_constnat

  # Define body axes
  e_body = np.eye(3)

  # Save inputs to file
  name = output_name + '.info'
  with open(name, 'w') as f_handle:
    f_handle.write('Jxx      = ' + str(Jxx) + '\n')
    f_handle.write('Jyy      = ' + str(Jyy) + '\n')
    f_handle.write('Jzz      = ' + str(Jzz) + '\n')    
    f_handle.write('num_step = ' + str(num_steps) + '\n')
    f_handle.write('n_save   = ' + str(n_save) + '\n')
    f_handle.write('dt       = ' + str(dt) + '\n')    
    f_handle.write('perturbation = ' + str(perturbation) + '\n')

  # Open files
  if True:
    name_omega = output_name + '.omega'
    name_angular_momentum = output_name + '.angular_momentum'
    name_quaternion = output_name + '.quaternion'
    name_axes = output_name + '.axes_body'
    name_e1_xyz = output_name + '.e1.xyz'
    name_e2_xyz = output_name + '.e2.xyz'
    name_e3_xyz = output_name + '.e3.xyz'  
    file_omega = open(name_omega, 'w')
    file_angular_momentum = open(name_angular_momentum, 'w')    
    file_quaternion = open(name_quaternion, 'w')
    file_axes = open(name_axes, 'w')
    file_e1_xyz = open(name_e1_xyz, 'w')
    file_e2_xyz = open(name_e2_xyz, 'w')
    file_e3_xyz = open(name_e3_xyz, 'w')

  def residual_fk(x, J, rhs):
    x_norm = np.linalg.norm(x) if np.linalg.norm(x) > 0 else 1e-12
    Jx = np.dot(J, x)
    return (np.sin(x_norm) / x_norm) * Jx + ((1 - np.cos(x_norm)) / x_norm**2) * np.cross(x, Jx) - rhs
  
  # Time loop
  R = np.eye(3)  
  for step in range(num_steps):
    if step >= 0 and (step % n_save) == 0:
      # Save data
      t = step * dt
      print('step = ', step, ', time = ', t, ', wall clock time = ', time.time() - start)
      omega = np.dot(R, np.dot(J_inv, angular_momentum))
      file_omega.write('%s %s %s %s\n' % (t, omega[0], omega[1], omega[2]))
      L = np.dot(R, angular_momentum)
      file_angular_momentum.write('%s %s %s %s\n' % (t, L[0], L[1], L[2]))
      e = np.dot(R, e_body)      
      file_axes.write('%s %s %s %s %s %s %s %s %s %s\n' % (t, e[0,0], e[1,0], e[2,0], e[0,1], e[1,1], e[2,1], e[0,2], e[1,1], e[2,2]))
      file_e1_xyz.write('1\n#\n')
      file_e1_xyz.write('O 0 0 0 %s %s %s \n' % (e[0,0], e[1,0], e[2,0]))
      file_e2_xyz.write('1\n#\n')
      file_e2_xyz.write('O 0 0 0 %s %s %s \n' % (e[0,1], e[1,1], e[2,1]))
      file_e3_xyz.write('1\n#\n')
      file_e3_xyz.write('O 0 0 0 %s %s %s \n' % (e[0,2], e[1,2], e[2,2]))
      q = quaternion_from_R(R)
      if np.any(np.isnan(q)):
        print('q = ', q)
        sys.exit()
      file_quaternion.write('%s %s %s %s %s\n' % (t, q[0], q[1], q[2],q[3]))
      q_norm = np.linalg.norm(q)
      if abs(q_norm - 1) > 1e-06:
        print('q_norm     = ', q_norm)
        print('q_norm - 1 = ', q_norm - 1)  
        sys.exit()

      
    # Compute RHS for nonlinear equation
    torque_0 = torque_func(T0, omega_B, t, phi, R, dipole_0)
    rhs = dt * angular_momentum + 0.5 * dt**2 * torque_0

    # Solve nonlinear system for fk
    result = scop.least_squares(residual_fk,
                                np.zeros(3),
                                verbose=(2 if verbose else 0),
                                ftol=tol,
                                xtol=tol*1e-03,
                                gtol=None,
                                jac='2-point',
                                x_scale=1,
                                max_nfev = 100,
                                kwargs={'J':J, 'rhs':rhs})
    if verbose:
      print('fk    = ', result.x)
      print('res   = ', result.fun)
      print('|res| = ', np.linalg.norm(result.fun, ord=np.inf))     
    
    # Build matrix Fk
    fk = result.x
    fk_norm = np.linalg.norm(fk) if np.linalg.norm(fk) > 0 else 1e-12
    fk_skew = skew_matrix(fk)
    Fk = np.eye(3) + np.sin(fk_norm) / fk_norm * fk_skew + (1 - np.cos(fk_norm)) / fk_norm**2 * np.dot(fk_skew, fk_skew)

    # Update rotation matrix
    R = np.dot(R, Fk)
    if verbose:
      print('|R*R.T - I| = ', np.linalg.norm(np.dot(R, R.T) - np.eye(3)))
      print('\n\n')
      
    # Update momentum
    torque_1 = torque_func(T0, omega_B, t, phi, R, dipole_0)
    angular_momentum = np.dot(Fk.T, angular_momentum + 0.5 * dt * torque_0) + 0.5 * dt * torque_1
    


    

  # Save last time step
  step += 1
  if step >= 0 and (step % n_save) == 0:
    # Save data
    t = step * dt
    print('step = ', step, ', time = ', t, ', wall clock time = ', time.time() - start)
    omega = np.dot(R, np.dot(J_inv, angular_momentum))
    file_omega.write('%s %s %s %s\n' % (t, omega[0], omega[1], omega[2]))
    L = np.dot(R, angular_momentum)
    file_angular_momentum.write('%s %s %s %s\n' % (t, L[0], L[1], L[2]))
    e = np.dot(R, e_body)      
    file_axes.write('%s %s %s %s %s %s %s %s %s %s\n' % (t, e[0,0], e[1,0], e[2,0], e[0,1], e[1,1], e[2,1], e[0,2], e[1,1], e[2,2]))
    file_e1_xyz.write('1\n#\n')
    file_e1_xyz.write('O 0 0 0 %s %s %s \n' % (e[0,0], e[1,0], e[2,0]))
    file_e2_xyz.write('1\n#\n')
    file_e2_xyz.write('O 0 0 0 %s %s %s \n' % (e[0,1], e[1,1], e[2,1]))
    file_e3_xyz.write('1\n#\n')
    file_e3_xyz.write('O 0 0 0 %s %s %s \n' % (e[0,2], e[1,2], e[2,2]))
    q = quaternion_from_R(R)
    file_quaternion.write('%s %s %s %s %s\n' % (t, q[0], q[1], q[2],q[3]))
    q_norm = np.linalg.norm(q)    
    if abs(q_norm - 1) > 1e-06:
      print('q_norm = ', q_norm)
      sys.exit()
