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
  A = 1
  B = 1
  C = 1
  t = 3.9
  r = 3.9
  
  # Compute volume
  V = (2.0 / 3.0) * A * B * C * (4.0 / (r*t)) * scspecial.beta(1.0 / r, 1.0 / r) * scspecial.beta(2.0 / t, 1.0 / t)

  print('V = ', V)


