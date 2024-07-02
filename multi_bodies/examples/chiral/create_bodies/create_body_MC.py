'''
Create a body discretization with a pseudo-Monte-Carlo method.
The code needs a parametrization of the body surface F(u,v) = 0.
'''
import numpy as np
import scipy.optimize as scop
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
  Nu = 12
  Nv = 10
  N = Nu * Nv
  A = 1.0
  B = 1.0
  C = 1.0
  t = 5
  m = 5
  noise = 0.1 / N
  kT = 1.0
  n_steps = 100000
  n_save = 100
  dt_min = 1e-01

  # Set initial configuration
  uv = np.zeros((N, 2))
  u = np.linspace(-np.pi, np.pi, Nu) 
  v = np.linspace(-np.pi/2.0, np.pi/2.0, Nv) 
  uu, vv = np.meshgrid(u, v)
  uv[:,0] = uu.flatten()
  uv[:,1] = vv.flatten()
  x, y, z = cartesian_superellipsoid(uv, A, B, C, m, t)
  r = np.zeros((N,3))
  r[:,0] = x
  r[:,1] = y
  r[:,2] = z

  f_handle = open('kk.'+ str(N) + '.xyz', 'w')
  for step in range(n_steps):
    # Save configuration
    if (step % n_save) == 0:
      print('step = ', step)
      result = np.zeros((N, 4))
      result[:,0] = 0
      result[:,1] = r[:,0]
      result[:,2] = r[:,1]
      result[:,3] = r[:,2]
      f_handle.write(str(N) + '\n' + '# \n')
      np.savetxt(f_handle, result)

    kT_eff = kT / np.sqrt(step + 1)
    
    # Update configuration with Monte Carlo
    for i in range(N):
      U = energy(r, r[i,0], r[i,1], r[i,2], i)
      uv_test = uv[i] + np.random.randn(2) * (noise)
      x, y, z = cartesian_superellipsoid(uv_test, A, B, C, m, t)
      U_test = energy(r, x, y, z, i)

      if np.exp((U_test - U) / kT_eff) < np.random.rand(1)[0]:
        uv[i] = uv_test
        r[i,0] = x
        r[i,1] = y
        r[i,2] = z

        
  # Save last configuration
  if (step+1 % n_save) == 0:
    result = np.zeros((N, 4))
    result[:,0] = 0
    result[:,1] = r[:,0]
    result[:,2] = r[:,1]
    result[:,3] = r[:,2]
    f_handle.write(str(N) + '\n' + '# \n')
    np.savetxt(f_handle, result)
  f_handle.close()

  # Save last configuration in vertex formar
  np.savetxt('kk.' + str(N) + '.vertex', result[:,1:], header=str(N))
  
