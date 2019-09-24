from __future__ import division, print_function
import numpy as np
import sys
import stkfmm
sys.path.append('..')
import mobility as mob
from general_application_utils import timer 
from mpi4py import MPI
from numba import njit, prange

try:
  import mobility_fmm as fmm
  fortran_fmm_found = True
except ImportError:
  fortran_fmm_found = False
fortran_fmm_found = False
print('fortran_fmm_found = ', fortran_fmm_found)


@njit(parallel=True, fastmath=True)
def distance(r_vectors):
  x = np.copy(r_vectors[:,0])
  y = np.copy(r_vectors[:,1])
  z = np.copy(r_vectors[:,2])
  N = x.size
  distance2 = np.ones(N) * 1e+99
  for i in prange(N):
    for j in range(N):
      if i == j:
        continue
      xij = x[i] - x[j]
      yij = y[i] - y[j]
      zij = z[i] - z[j]
      d2 = xij*xij + yij*yij + zij*zij
      if d2 < distance2[i]:
        distance2[i] = d2
  d2_min = np.min(distance2)
  return np.sqrt(d2_min)



if __name__ == '__main__':
  print('# Start')
  # Set parameters
  N = 131072
  a = 1e-0
  phi = 1e-03
  L = np.power(4*np.pi * N / (3 * phi), 1.0/3.0) * a
  eta = 1.0
  mult_order = 8
  max_pts = 512
  N_max = 262144

  print('L = ', L)

  # Create random blobs
  r_vectors = np.random.rand(N, 3) * L
  forces = np.random.randn(N, 3)
  #forces[:,:] = 0
  #forces[1,:] = 1.0
  #r_vectors[:,:] = 0
  #r_vectors[1,0] = L 

  # Compute velocities with numba (no wall)
  if N <= N_max:
    v_numba = mob.no_wall_mobility_trans_times_force_numba(r_vectors, forces, eta, a)
  timer('numba')
  if N <= N_max:
    v_numba = mob.no_wall_mobility_trans_times_force_numba(r_vectors, forces, eta, a)
  timer('numba')  

  # Compute velocities with stkfmm
  timer('stkfmm_create_fmm')
  fmm_PVelLaplacian = None
  fmm_PVelLaplacian = stkfmm.STKFMM(mult_order, max_pts, stkfmm.PAXIS.NONE, stkfmm.KERNEL.PVelLaplacian)
  timer('stkfmm_create_fmm')
  timer('stkfmm_setting_tree')
  v_stkfmm_tree = mob.no_wall_mobility_trans_times_force_stkfmm(r_vectors, 
                                                                forces, 
                                                                eta, 
                                                                a, 
                                                                fmm_PVelLaplacian=fmm_PVelLaplacian,
                                                                set_tree=True)
  timer('stkfmm_setting_tree')
  timer('stkfmm')
  v_stkfmm = mob.no_wall_mobility_trans_times_force_stkfmm(r_vectors, 
                                                           forces, 
                                                           eta, 
                                                           a,
                                                           fmm_PVelLaplacian=fmm_PVelLaplacian,
                                                           set_tree=False)
  timer('stkfmm')

  # Compute velocities with rpyfmm
  if fortran_fmm_found:
    timer('rpyfmm')
    v_rpyfmm = mob.fmm_rpy(r_vectors, forces, eta, a)
    timer('rpyfmm')


  print('\n\n')
  # Compute distance between points
  if N <= N_max:
    dr_min = distance(r_vectors)
    print('dr_min = ', dr_min)

  # Compute errors
  if N <= N_max:
    diff_tree = v_stkfmm_tree - v_numba
    diff = v_stkfmm - v_numba
    print('relative L2 error (1) = ', np.linalg.norm(diff_tree) / np.linalg.norm(v_numba))
    print('Linf error        (1) = ', np.linalg.norm(diff_tree.flatten(), ord=np.inf))
    print('relative L2 error (2) = ', np.linalg.norm(diff) / np.linalg.norm(v_numba))
    print('Linf error        (2) = ', np.linalg.norm(diff.flatten(), ord=np.inf))
    if fortran_fmm_found:
      print('fortran_rpy_fmm errors:')
      diff = v_rpyfmm - v_numba
      diff = v_rpyfmm - v_numba
      print('relative L2 error (3) = ', np.linalg.norm(diff) / np.linalg.norm(v_numba))
      print('Linf error        (3) = ', np.linalg.norm(diff.flatten(), ord=np.inf))   
  else:
    diff = v_stkfmm_tree - v_stkfmm
    print('relative L2 error (fmm) = ', np.linalg.norm(diff) / np.linalg.norm(v_stkfmm_tree))
    print('Linf error        (fmm) = ', np.linalg.norm(diff.flatten(), ord=np.inf))
    if fortran_fmm_found:
      print('fortran_rpy_fmm errors:')
      diff = v_rpyfmm - v_vtkfmm
      diff = v_rpyfmm - v_vtkfmm
      print('relative L2 error (3) = ', np.linalg.norm(diff) / np.linalg.norm(v_numba))
      print('Linf error        (3) = ', np.linalg.norm(diff.flatten(), ord=np.inf))   
  print('\n\n')




  
  if N < 6:
    N_min = N
  else:
    N_min = 6
  print('diff = \n', diff[0:3*N_min])
  if N <= N_max:
    print('v_numba = \n', v_numba[0:3*N_min])
  print('v_stkfmm = \n', v_stkfmm[0:3*N_min])
 
  timer(' ', print_all=True)
  print('# End')
