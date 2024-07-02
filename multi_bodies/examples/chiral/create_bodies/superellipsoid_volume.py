'''
Compute the volume of a superellipsoid.

Surface:
(|x/A|**r + |y/B|**r)**(t/r) + |z/C|**t = 1


'''
import numpy as np
import scipy.special as scspecial
import sys


if __name__ == '__main__':
  print('# Start')
  # Set parameters
  A = 0.74  
  B = 0.74  
  C = 0.74 
  t = 5
  r = 5

  # More parameters
  epsilon = 2.0 / t
  abc = np.power(A * B * C, 1.0 / 3.0)
  
  # Compute volume
  V = (2.0 / 3.0) * A * B * C * (4.0 / (r*t)) * scspecial.beta(1.0 / r, 1.0 / r) * scspecial.beta(2.0 / t, 1.0 / t)
  V_v2 = 2 * abc**3 * epsilon**2                * scspecial.beta(epsilon/2, epsilon+1) * scspecial.beta(epsilon/2, epsilon/2+1)

  print('A    = ', A)
  print('B    = ', B)
  print('C    = ', C)
  print('t    = ', t)
  print('r    = ', r)
  print('eps  = ', epsilon)
  print('abc  = ', abc)
  print(' ')
  print('V    = ', V)
  print('V_v2 = ', V_v2)



