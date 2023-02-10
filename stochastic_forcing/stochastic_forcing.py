'''
Module to compute the stochastic forcing (sqrt(2*k_B*T*dt)*M^{1/2}*z) with several algorithms.
'''

import numpy as np

def stochastic_forcing_eig(mobility, factor = 1.0, z = None):
  '''
  Compute the stochastic forcing (factor * M^{1/2} * z) using
  eigenvalue decomposition. If M=V*S*V.T the noise is
  w = V*S^{1/2}*z. 
  This functions is more expensive that
  using a Cholesky factorization but it works for non-positive
  definite matrix (e.g. mobility of a 1D rod).
  
  Input:
  mobility = the mobility matrix. You can pass it like a list of lists or
             a list of numpy arrays.
  factor = the prefactor, in general something like sqrt(2*k_B*T*dt)
  z = (Optional) the random vector z
  
  Output:
  stochastic_forcing = (factor * M^{1/2} * z)
  '''
  
  # Compute eigenvalues and eigenvectors 
  eig_values, eig_vectors = np.linalg.eigh(mobility)

  # Compute the square root of positive eigenvalues and set to zero otherwise
  eig_values_sqrt = np.array([np.sqrt(x) if x > 0 else 0 for x in eig_values])

  # Multiply by random vector with zero mean and unit variance
  if z is None:
    eig_values_sqrt *= np.random.normal(0.0, 1.0, len(mobility))
  else:
    eig_values_sqrt *= z

  # Compute stochastic forcing
  stochastic_forcing = factor * np.dot(eig_vectors, eig_values_sqrt)
      
  return stochastic_forcing


def stochastic_forcing_eig_symm(mobility, factor = 1.0, z = None):
  '''
  Compute the stochastic forcing (factor * M^{1/2} * z) using
  eigenvalue decomposition. If M=V*S*V.T the noise is
  w = V*S^{1/2}*V.T*z. 
  This functions is more expensive that
  using a Cholesky factorization but it works for non-positive
  definite matrix (e.g. mobility of a 1D rod).
  
  Input:
  mobility = the mobility matrix. You can pass it like a list of lists or
             a list of numpy arrays.
  factor = the prefactor, in general something like sqrt(2*k_B*T*dt)
  z = (Optional) the random vector z
  
  Output:
  stochastic_forcing = (factor * M^{1/2} * z)
  '''
  
  # Compute eigenvalues and eigenvectors 
  eig_values, eig_vectors = np.linalg.eigh(mobility)

  # Compute the square root of positive eigenvalues and set to zero otherwise
  eig_values_sqrt = np.array([np.sqrt(x) if x > 0 else 0 for x in eig_values])

  # Multiply V^T by random vector with zero mean and unit variance
  if z is None:
    w = np.dot(eig_vectors.T, np.random.normal(0.0, 1.0, len(mobility)))
  else:
    w = np.dot(eig_vectors.T, z)

  # Multiply by square root of eigenvalues
  w *= eig_values_sqrt
  
  # Compute stochastic forcing
  stochastic_forcing = factor * np.dot(eig_vectors, w)

  return stochastic_forcing



def stochastic_forcing_cholesky(mobility, factor = 1.0, z = None):
  '''
  Compute the stochastic forcing (factor * M^{1/2} * z) using
  Cholesky decomposition. 
  
  Input:
  mobility = the positive-definite mobility matrix. You can 
             pass it like a list of lists or a list of numpy arrays.
  factor = the prefactor, in general something like sqrt(2*k_B*T*dt)
  z = (Optional) the random vector z
  
  Output:
  stochastic_forcing = (factor * M^{1/2} * z)
  '''

  # Compute Cholesky factorization
  mobility_half = np.linalg.cholesky(mobility)

  # Compute random vectore with mean zero and variance 1
  if z is None:
    z = np.random.normal(0.0, 1.0, len(mobility))

  # Compute stochastic forcing
  stochastic_forcing = factor * np.dot(mobility_half, z)
      
  return stochastic_forcing

def stochastic_forcing_lanczos(factor = 1.0, 
                               tolerance = 1e-06, 
                               max_iter = 1000, 
                               dim = None, 
                               mobility = None, 
                               mobility_mult = None,
                               L_mult = None,
                               z = None,
                               print_residual = False):
  '''
  Compute the stochastic forcing (factor * M^{1/2} * z) using
  the Lanczos algorithm, see Krylov subspace methods for 
  computing hydrodynamic interactions in Brownian dynamics simulations, 
  T. Ando et al. The Journal of Chemical Physics 137, 064106 (2012) 
  doi: 10.1063/1.4742347 and subsequent papers by Yousef Saad.

  The noise generated by this function should be the same 
  (to within tolerance) than the noise generated by the function
  stochastic_forcing_eig_symm.  
  
  Input: 
  factor = the prefactor, in general something like sqrt(2*k_B*T*dt)
  tolerance = the tolerance to determine convergence.
  max_iter = maximum number of iterations allowed.
  dim = The dimension of the noise. If it is not passed dim = len(z).
  print_residual = (Optional, default False) If True, it prints the iteration number and the 
                   residual every iteration.
  z = (Optional) the random vector.
  mobility = the mobility matrix. You can pass it like a list of lists or
             a list of numpy arrays. It is not used if mobility_mult
             is passed (see below).
  mobility_mult = function that computes a matrix vector product 
                  with the mobility matrix. It allows to use matrix
                  free methods.
  L_mult = function that computes a matrix vector product 
           between the preconditioner matrix L and the noise generated by
           the Lanczos algorithm. L should obey the relation
           M \approx L*L^T.
  
  Output:
  The code returns 
  (stochastic_forcing, iterations) 
  with
  stochastic_forcing = (factor * M^{1/2} * z)
  iterations = total number of iterations.
  '''

  # Define array dimension 
  if dim is None:
    dim = len(z)

  if factor == 0.0:
    return (np.zeros(dim), 0)

  # Create matrix v (initial column is random)
  # Note: v will have shape (iteration, dim);
  # in the standard notation used in the Lanczos
  # scheme v will be the matrix V^T
  if z is None:
    v = np.random.randn(1, dim)
  else:
    v = np.copy(np.reshape(z, (1, dim)))

  # Normalize v
  v_norm = np.linalg.norm(v[0])
  v[0] /= v_norm 

  # Create list for the data of the symmetric tridiagonal matrix h 
  h_sup = []
  h_diag = []

  # Create vectors noise
  noise = np.zeros(dim) 
  noise_old = np.zeros(dim) 

  # Iterate until convergence or max_iter
  for i in range(max_iter+1):
    # w = mobility * v[i]
    if mobility is None:
      w = np.reshape(mobility_mult(v[i]), dim)
    else:
      w = np.dot(mobility, v[i])

    # w = w - h[i-1, i] * v[i-1]   
    if i > 0:
      w = w - h_sup[i-1] * v[i-1] 

    # h(i, i) = <w, v[i]> 
    h_diag.append( np.dot(w, v[i]) ) 

    # w = w - h(i, i)*v(i)
    w = w - h_diag[i] * v[i]

    # h(i+1, i) = h(i, i+1) = <w, w>
    h_sup.append( np.linalg.norm(w) )

    # w = w/normw;
    if h_sup[i] > 0:
      w /= h_sup[i]
    else:
      w[0] = 1.0;
    
    # Build tridiagonal matrix h
    h = h_diag * np.eye(len(h_diag)) + h_sup * np.eye(len(h_sup), k=-1) + (h_sup * np.eye(len(h_sup), k=-1)).T

    # Compute eigenvalues and eigenvectors of h
    # IMPORTANT: this is NOT optimized for tridiagonal matrices
    eig_values, eig_vectors = np.linalg.eigh(h)

    # Compute the square root of positive eigenvalues set to zero otherwise
    eig_values_sqrt = np.array([np.sqrt(x) if x > 0 else 0 for x in eig_values])
    
    # Create vector e_1
    e_1 = np.zeros(len(eig_values))
    e_1[0] = 1.0

    # Compute noise approximation as in Eq. 16 of Ando et al. 2012
    noise = np.dot(v.T, np.dot(eig_vectors, v_norm * factor * eig_values_sqrt * np.dot(eig_vectors.T, e_1)))

    # Orthogonalize base with modified Gram-Schmidt;
    # we use that norm(v[i])=norm(w)=1
    for row in v:
      w = w - (np.dot(row, w)) * row 

    # v(i+1) = w
    v = np.concatenate([v, [w]])

    if i > 0:
      # Compute difference with noise of previous iteration
      noise_old_norm = np.linalg.norm(noise_old)
      diff_norm = np.linalg.norm(noise - noise_old)

      # (Optional) Print residual
      if i > 0 and print_residual is True:
        if i == 1:
          print('lanczos =  0 1')
        print('lanczos = ', i, diff_norm / noise_old_norm)

      # Check convergence and return if difference < tolerance
      if diff_norm / np.maximum(noise_old_norm, np.finfo(float).eps) < tolerance:
        if L_mult is None:
          return (noise, i)
        else:
          return (np.reshape(L_mult(noise), dim), i)
          
    # Save noise to check convergence in the next iteration
    noise_old = np.copy(noise)

  # Return UNCONVERGED noise
  if L_mult is None:
    return (noise, max_iter)
  else:
    return (np.reshape(L_mult(noise), dim), max_iter)





