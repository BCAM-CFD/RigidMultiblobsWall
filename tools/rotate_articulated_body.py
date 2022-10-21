import numpy as np
import sys
sys.path.append('../../../')

from quaternion_integrator.quaternion import Quaternion
from body import body


if __name__ == '__main__':
  # Set parameters
  name = sys.argv[1]
  skiprows = 1
  axis = np.array([0, 1, 0])
  angle = np.pi / 2
  center_of_rotation = np.array([0, 0, 3])

  # Read config
  x = np.loadtxt(name, skiprows=skiprows)
  
  # Create rotation quaternion
  q = np.array([np.cos(angle / 2.0), np.sin(angle / 2.0) * axis[0], np.sin(angle / 2.0) * axis[1], np.sin(angle / 2.0) * axis[2]])
  q_norm = np.linalg.norm(q)
  theta = Quaternion(q / q_norm)
  R = theta.rotation_matrix()
  
  # Loop over bodies and rotate
  x_new = np.zeros_like(x)
  for i, xi in enumerate(x):
    x_new[i,0:3] = center_of_rotation + np.dot(R, (xi[0:3] - center_of_rotation)) 
    q_x = Quaternion(xi[3:])
    x_new[i, 3:] = (theta * q_x).entries


  # Print new configuration
  print(x_new.size // 7)
  np.savetxt(sys.stdout, x_new)
  
  
  
  
