from __future__ import division, print_function
import numpy as np
import sys
from functools import partial
try:
  from mpi4py import MPI
except ImportError:
  print('problem with MPI')

from numba import njit, prange

sys.path.append('/workspace/scratch/users/fbalboa/sfw/FMM2/STKFMM-lib-gnu/lib64/python/')
import PySTKFMM

sys.path.append('..')
from general_application_utils import timer
import mobility as mob


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
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  print('rank = ', rank)
  comm.Barrier()
  print('rank = ', rank)
    
  N = 1000
  a = 1e-01
  phi = 1e-03
  L_scalar = np.power(4*np.pi * N / (3 * phi), 1.0/3.0) * a
  eta = 1.0
  mult_order = 10
  max_pts = 128
  wall = True
  pbc = PySTKFMM.PAXIS.NONE
  L = np.array([0., 0., 0.])
  print('L_scalar = ', L_scalar)
  print('L        = ', L)

  # Create random blobs
  r_vectors = np.random.rand(N, 3) * L_scalar
  if wall:
    r_vectors[:,2] *= 0.25
  force = np.random.randn(N, 3)

  # Setup FMM
  kernel = PySTKFMM.KERNEL.RPY
  if wall:
    timer('built_fmm')
    rpy_fmm = PySTKFMM.StkWallFMM(mult_order, max_pts, pbc, kernel)
    mobility_trans_times_force_stkfmm_partial = partial(mob.mobility_trans_times_force_stkfmm,
                                                        rpy_fmm=rpy_fmm,
                                                        L=L,
                                                        wall=wall,
                                                        comm=comm)
    timer('built_fmm')
  else:
    timer('built_fmm')
    rpy_fmm = PySTKFMM.Stk3DFMM(mult_order, max_pts, pbc, kernel)
    mobility_trans_times_force_stkfmm_partial = partial(mob.mobility_trans_times_force_stkfmm,
                                                        rpy_fmm=rpy_fmm,
                                                        L=L,
                                                        wall=wall,
                                                        comm=comm)
    timer('built_fmm')
  SLdim, DLdim, Trgdim = rpy_fmm.get_kernel_dimension(kernel)
  print('SLdim, DLdim, Trgdim = ', SLdim, DLdim, Trgdim)

  # Compute velocities
  if wall:
    u_numba = mob.single_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a)
    u_stkfmm = mobility_trans_times_force_stkfmm_partial(r_vectors, force, eta, a, rpy_fmm=rpy_fmm, L=L)
    timer('numba')
    u_numba = mob.single_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a)
    timer('numba')
    timer('stkfmm')
    u_stkfmm = mobility_trans_times_force_stkfmm_partial(r_vectors, force, eta, a, rpy_fmm=rpy_fmm, L=L)
    timer('stkfmm')
  else:
    u_numba = mob.no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a)
    u_stkfmm = mobility_trans_times_force_stkfmm_partial(r_vectors, force, eta, a, rpy_fmm=rpy_fmm, L=L)
    timer('numba')
    u_numba = mob.no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a)
    timer('numba')
    timer('stkfmm')
    u_stkfmm = mobility_trans_times_force_stkfmm_partial(r_vectors, force, eta, a, rpy_fmm=rpy_fmm, L=L)
    timer('stkfmm')

  # Compute errors
  diff = u_stkfmm - u_numba
  L2 = np.linalg.norm(diff)
  print('L2          = ', L2)
  print('L2 relative = ', L2 / np.linalg.norm(u_numba))

  if N < 6:
    print('\n\n\n')
    print('r_vectors = \n', r_vectors)
    print('u_numba   = \n', u_numba)
    print('u_stkfmm  = \n', u_stkfmm)
  else:
    u_stkfmm = u_stkfmm.reshape((u_stkfmm.size // 3, 3))
    u_numba = u_numba.reshape((u_numba.size // 3, 3))
    diff = np.linalg.norm(u_stkfmm - u_numba, axis=1)
    sel = diff > 1e-06
    indices = np.arange(N, dtype=int)
    print('Print large differences:')
    print('indices  = ', indices[sel])
    print('diff     = ', diff[sel])
    print('u_numba  = ', u_numba[sel])
    print('u_stkfmm = ', u_stkfmm[sel])
    
  timer(' ', print_all=True)
  print('# End')
