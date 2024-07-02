'''
Create a body discretization with a pseudo-Monte-Carlo method.
The code needs a parametrization of the body surface F(u,v) = 0.
'''
import numpy as np
import scipy.optimize as scop
import scipy.special as scspecial
import sys
from functools import partial
from numba import njit, prange


def cartesian_superellipsoid(uv, A, B, C, m, t):
  N = uv.size // 2
  uv = uv.reshape((N,2))
  u = uv[:,0]
  v = uv[:,1]
  x = A * np.sign(np.cos(v)) * np.abs(np.cos(v))**(2.0/t) * np.sign(np.cos(u)) * np.abs(np.cos(u))**(2.0/m) 
  y = B * np.sign(np.cos(v)) * np.abs(np.cos(v))**(2.0/t) * np.sign(np.sin(u)) * np.abs(np.sin(u))**(2.0/m) 
  z = C * np.sign(np.sin(v)) * np.abs(np.sin(v))**(2.0/t) 
  return x, y, z

@njit(parallel=False)
def energy(r, xi, yi, zi, i):
  x = r[:,0]
  y = r[:,1]
  z = r[:,2]
  U = np.zeros(1)
  for j in range(x.size):
    if j == i:
      continue
    xij = xi - x[j]
    yij = yi - y[j]
    zij = zi - z[j]
    rij = np.sqrt(xij*xij + yij*yij + zij*zij)
    U = U + 1.0 / rij**6
   
  return U

if __name__ == '__main__':
  print('# Start')
  # Set parameters
  A = 0.74
  B = 0.74
  C = 0.74
  t = 5
  m = 5
  N_face = 3

  # Number of blobs
  N = N_face**3 - (N_face - 2)**3

  # Compute volume
  V = (2.0 / 3.0) * A * B * C * (4.0 / (m*t)) * scspecial.beta(1.0 / m, 1.0 / m) * scspecial.beta(2.0 / t, 1.0 / t)

  # create cube
  r_vectors = np.zeros((N, 3))
  counter = 0
  for ix in range(N_face):
    for iy in range(N_face):
      for iz in range(N_face):
        if ix==0 or ix==(N_face-1) or iy==0 or iy==(N_face-1) or iz==0 or iz==(N_face-1):
          r_vectors[counter] = np.array([2 * ix / (N_face-1) - 1, 2 * iy / (N_face-1) - 1, 2 * iz / (N_face-1) - 1])
          counter += 1

  # Set initial configuration
  uv = np.zeros((N, 2))
  x = r_vectors[:,0]
  y = r_vectors[:,1]
  z = r_vectors[:,2]
  theta = np.arctan2(np.sqrt(x**2 + y**2), z)
  phi = np.arctan2(y, x)
  uv[:, 0] = theta + np.random.randn(N) * 0.1
  uv[:, 0] = phi + np.random.randn(N) * 0.1

  def residual(uv):
    x, y, z = cartesian_superellipsoid(uv, A, B, C, m, t)
    r = np.zeros((N,3))
    r[:,0] = x
    r[:,1] = y
    r[:,2] = z
    return (r - r_vectors).flatten()

  # Call nonlinear solver
  result = scop.least_squares(residual,
                              uv.flatten(),
                              verbose=2,
                              ftol=1-10,
                              xtol=1e-10,
                              gtol=None,
                              method='dogbox',
                              x_scale='jac',
                              max_nfev = r_vectors.size * 10)

  x, y, z = cartesian_superellipsoid(result.x, A, B, C, m, t)
  r = np.zeros((N,3))
  r[:,0] = x
  r[:,1] = y
  r[:,2] = z
       
  # Save last configuration in vertex format
  name = 'kk.' + str(N) + '.vertex'
  with open(name, 'w') as f_handle:
    f_handle.write('# A      = ' + str(A) + '\n')
    f_handle.write('# B      = ' + str(B) + '\n')
    f_handle.write('# C      = ' + str(C) + '\n')
    f_handle.write('# t      = ' + str(t) + '\n')
    f_handle.write('# m      = ' + str(m) + '\n')
    f_handle.write('# N_face = ' + str(N_face) + '\n')
    f_handle.write('# Volume = ' + str(V) + '\n')
    f_handle.write('# \n')
    f_handle.write('# \n')
    
    f_handle.write(str(N) + '\n')    
    np.savetxt(f_handle, r)

  # Save last configuration in vertex xyz
  name = 'kk.' + str(N) + '.xyz'
  f_handle = open(name, 'w')
  f_handle.write(str(N) + '\n#\n')
  for i in range(N):
    f_handle.write('C %s %s %s\n' % (r[i,0], r[i,1], r[i,2]))


  # Save last configuration in vertex xyz
  name = 'kk.xyz'
  f_handle = open(name, 'w')
  f_handle.write(str(N) + '\n#\n')
  for i in range(N):
    f_handle.write('C %s %s %s\n' % (r[i,0], r[i,1], r[i,2]))
    

