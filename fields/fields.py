import numpy as np
from numba import njit, prange
import math
# Try to import the visit_writer (boost implementation)
try:
  import visit.visit_writer as visit_writer
except ImportError:
  pass
  

class fields(object):
  '''

  '''
  def __init__(self, grid_options, save_number_density = False, save_velocity = False, save_stress = False, stress_inf_correction = 0, blob_radius = None):
    '''
    
    '''
    # Save general options
    self.save_number_density = save_number_density
    self.save_velocity = save_velocity
    self.save_stress = save_stress
    self.stress_inf_correction = stress_inf_correction
    self.blob_radius = blob_radius
    
    # Save grid options
    grid = np.reshape(grid_options, (3,3)).T
    self.length_mesh = grid[1] - grid[0]
    self.lower_corner = grid[0]
    self.upper_corner = grid[1]
    self.mesh_points = np.array(grid[2], dtype=np.int32)
    self.num_points = int(grid[2,0] * grid[2,1] * grid[2,2])

    # Create grid
    self.dx_mesh = self.length_mesh / self.mesh_points
    self.mesh_x = np.array([grid[0,0] + self.dx_mesh[0] * (x+0.5) for x in range(self.mesh_points[0])])
    self.mesh_y = np.array([grid[0,1] + self.dx_mesh[1] * (x+0.5) for x in range(self.mesh_points[1])])
    self.mesh_z = np.array([grid[0,2] + self.dx_mesh[2] * (x+0.5) for x in range(self.mesh_points[2])])
    if self.mesh_points[0] > 1:
      dx = self.dx_mesh[0]
    else:
      dx = 1.0
    if self.mesh_points[1] > 1:
      dy = self.dx_mesh[1]
    else:
      dy = 1.0
    if self.mesh_points[2] > 1:
      dz = self.dx_mesh[2]
    else:
      dz = 1.0
    self.volume_cell = dx * dy * dz

    # Be aware, x is the fast axis.
    zz, yy, xx = np.meshgrid(self.mesh_z, self.mesh_y, self.mesh_x, indexing = 'ij')
    self.mesh_coor = np.zeros((self.num_points, 3))
    self.mesh_coor[:,0] = np.reshape(xx, xx.size)
    self.mesh_coor[:,1] = np.reshape(yy, yy.size)
    self.mesh_coor[:,2] = np.reshape(zz, zz.size)

    # Create variables
    self.counter = 0
    if self.save_number_density:
      self.number_density_avg = np.zeros(self.num_points)
      self.number_density_var = np.zeros(self.num_points)
    if self.save_velocity:
      self.velocity_avg = np.zeros((self.num_points,6))
      self.velocity_var = np.zeros((self.num_points, 6))
    if self.save_stress:
      self.stress_avg = np.zeros((self.num_points, 9))
      self.stress_var = np.zeros((self.num_points, 9))
    return



  def save(self, bodies, vw = None, r_vectors_blobs = None, force_blobs = None):
    '''

    '''
    # Get bodies variables
    q = np.zeros((len(bodies), 3))
    b_length = np.zeros(len(bodies))
    if vw is None:
      vw = np.zeros((len(bodies), 6))

    for i, b in enumerate(bodies):
      q[i] = np.copy(b.location)
      b_length[i] = b.body_length
      
    if self.save_number_density or save_velocity:
      number_density, velocity = self.compute_number_density_velocity(q, 
                                                                      vw, 
                                                                      b_length, 
                                                                      self.mesh_x, 
                                                                      self.mesh_y, 
                                                                      self.mesh_z, 
                                                                      self.lower_corner, 
                                                                      self.length_mesh, 
                                                                      self.mesh_points,
                                                                      self.volume_cell)

      if self.save_number_density:
        self.number_density_avg += (number_density - self.number_density_avg) / (self.counter + 1)
        self.number_density_var += (number_density - self.number_density_avg)**2 * (self.counter / (self.counter+1)) 
      if self.save_velocity:
        self.velocity_avg += (velocity - self.velocity_avg) / (self.counter + 1)
        self.velocity_var += (velocity - self.velocity_avg)**2 * (self.counter / (self.counter+1)) 
    if self.save_stress:
      if force_blobs is None:
        force_blobs = np.zeros_like(r_vectors_blobs)
      stress = self.compute_stress_tensor(r_vectors_blobs.reshape(r_vectors_blobs.size // 3, 3),
                                          self.mesh_coor,
                                          force_blobs.reshape(force_blobs.size // 3, 3),
                                          self.blob_radius,
                                          self.stress_inf_correction)
      self.stress_avg += (stress - self.stress_avg) / (self.counter + 1)
      self.stress_var += (stress - self.stress_avg)**2 * (self.counter / (self.counter+1)) 
        
    self.counter += 1
    return

  
  def restart(self):
    '''

    '''
    self.counter = 0
    if save_number_density:
      self.number_density_avg[:] = 0
      self.number_density_var[:] = 0
    if save_velocity:
      self.velocity_avg[:] = 0
      self.velocity_var[:] = 0
    if save_stress:
      self.stress_avg[:] = 0
      self.stress_var[:] = 0
    return


  def print_files(self, name_output):
    '''

    '''
    # Prepare mesh for VTK
    mesh_x = self.mesh_x - self.dx_mesh[0] * 0.5
    mesh_y = self.mesh_y - self.dx_mesh[1] * 0.5
    mesh_z = self.mesh_z - self.dx_mesh[2] * 0.5
    mesh_x = np.concatenate([mesh_x, [self.upper_corner[0]]])
    mesh_y = np.concatenate([mesh_y, [self.upper_corner[1]]])
    mesh_z = np.concatenate([mesh_z, [self.upper_corner[2]]])
    
    if self.save_number_density:
      number_density_variance = self.number_density_var / max(1.0, self.counter - 1)
      variables = [self.number_density_avg, number_density_variance]
      dims = np.array([self.mesh_points[0]+1, self.mesh_points[1]+1, self.mesh_points[2]+1], dtype=np.int32)
      nvars = 2
      vardims =   np.array([1,1], dtype=np.int32)
      centering = np.array([0,0], dtype=np.int32)
      varnames = ['number_density\0', 'number_density_variance\0']
      name = name_output + '.number_density_field.vtk'

      # Write field
      visit_writer.boost_write_rectilinear_mesh(name,      # File's name
                                                0,         # 0=ASCII,  1=Binary
                                                dims,      # {mx, my, mz}
                                                mesh_x,    # xmesh
                                                mesh_y,    # ymesh
                                                mesh_z,    # zmesh
                                                nvars,     # Number of variables
                                                vardims,   # Size of each variable, 1=scalar, velocity=3*scalars
                                                centering, # Write to cell centers of corners
                                                varnames,  # Variables' names
                                                variables) # Variables

    if self.save_velocity:
      velocity_variance = self.velocity_var / max(1.0, self.counter - 1)
      variables = [np.copy(self.velocity_avg[:,0]), 
                   np.copy(self.velocity_avg[:,1]), 
                   np.copy(self.velocity_avg[:,2]), 
                   np.copy(self.velocity_avg[:,3]), 
                   np.copy(self.velocity_avg[:,4]), 
                   np.copy(self.velocity_avg[:,5]), 
                   np.copy(velocity_variance[:,0]), 
                   np.copy(velocity_variance[:,1]), 
                   np.copy(velocity_variance[:,2]), 
                   np.copy(velocity_variance[:,3]), 
                   np.copy(velocity_variance[:,4]), 
                   np.copy(velocity_variance[:,5])]
      dims = np.array([self.mesh_points[0]+1, self.mesh_points[1]+1, self.mesh_points[2]+1], dtype=np.int32)
      nvars = 12
      vardims =   np.array([1,1,1,1,1,1,1,1,1,1,1,1], dtype=np.int32)
      centering = np.array([0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.int32)
      varnames = ['velocity_X\0', 'velocity_Y\0', 'velocity_Z\0', 'omega_X\0', 'omega_Y\0', 'omega_Z\0', 'velocity_variance_X\0', 'velocity_variance_Y\0', 'velocity_variance_Z\0', 'omega_variance_X\0', 'omega_variance_Y\0', 'omega_variance_Z\0']
      name = name_output + '.bodies_velocity_field.vtk'

      # Write field
      visit_writer.boost_write_rectilinear_mesh(name,      # File's name
                                                0,         # 0=ASCII,  1=Binary
                                                dims,      # {mx, my, mz}
                                                mesh_x,    # xmesh
                                                mesh_y,    # ymesh
                                                mesh_z,    # zmesh
                                                nvars,     # Number of variables
                                                vardims,   # Size of each variable, 1=scalar, velocity=3*scalars
                                                centering, # Write to cell centers of corners
                                                varnames,  # Variables' names
                                                variables) # Variables

    if self.save_stress:
      stress_variance = self.stress_var / max(1.0, self.counter - 1)
      variables = [np.copy(self.stress_avg[:,0]), np.copy(self.stress_avg[:,1]), np.copy(self.stress_avg[:,2]), np.copy(self.stress_avg[:,3]), np.copy(self.stress_avg[:,4]), np.copy(self.stress_avg[:,5]), np.copy(self.stress_avg[:,6]), np.copy(self.stress_avg[:,7]), np.copy(self.stress_avg[:,8]), np.copy(stress_variance[:,0]), np.copy(stress_variance[:,1]), np.copy(stress_variance[:,2]), np.copy(stress_variance[:,3]), np.copy(stress_variance[:,4]), np.copy(stress_variance[:,5]), np.copy(stress_variance[:,6]), np.copy(stress_variance[:,7]), np.copy(stress_variance[:,8])]
      dims = np.array([self.mesh_points[0]+1, self.mesh_points[1]+1, self.mesh_points[2]+1], dtype=np.int32)
      nvars = 18
      vardims =   np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dtype=np.int32)
      centering = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.int32)
      varnames = ['stress_XX\0', 'stress_XY\0', 'stress_XZ\0', 'stress_YX\0', 'stress_YY\0', 'stress_YZ\0', 'stress_ZX\0', 'stress_ZY\0', 'stress_ZZ\0', 'stress_variance_XX\0', 'stress_variance_XY\0', 'stress_variance_XZ\0', 'stress_variance_YX\0', 'stress_variance_YY\0', 'stress_variance_YZ\0', 'stress_variance_ZX\0', 'stress_variance_ZY\0', 'stress_variance_ZZ\0']
      name = name_output + '.stress_field.vtk'
    
      # Write velocity field
      visit_writer.boost_write_rectilinear_mesh(name,      # File's name
                                                0,         # 0=ASCII,  1=Binary
                                                dims,      # {mx, my, mz}
                                                mesh_x,    # xmesh
                                                mesh_y,    # ymesh
                                                mesh_z,    # zmesh
                                                nvars,     # Number of variables
                                                vardims,   # Size of each variable, 1=scalar, velocity=3*scalars
                                                centering, # Write to cell centers of corners
                                                varnames,  # Variables' names
                                                variables) # Variables
    return


  @staticmethod
  @njit(parallel=True, fastmath=True)
  def compute_number_density_velocity(q, vw, b_length, mesh_x, mesh_y, mesh_z, lower_corner, length_mesh, mesh_points, volume_cell):
    '''

    '''
    # Prepare variables
    N = b_length.size
    M = mesh_points[0] * mesh_points[1] * mesh_points[2]
    dx = length_mesh[0] / mesh_points[0]
    dy = length_mesh[1] / mesh_points[1]
    dz = length_mesh[2] / mesh_points[2]
    number_density = np.zeros(M)
    velocity = np.zeros((M, 6))

    # Loop over bodies
    for i in range(N):
      
      # Find cell closer holding body
      if mesh_points[0] > 1:
        kx = int((q[i,0] - lower_corner[0]) / dx)
        x_body = b_length[i]
      else:
        kx = 0
        x_body = 1.0
      if mesh_points[1] > 1:
        ky = int((q[i,1] - lower_corner[1]) / dy)
        y_body = b_length[i]
      else:
        ky = 0
        y_body = 1.0
      if mesh_points[2] > 1:
        kz = int((q[i,2] - lower_corner[2]) / dz)
        z_body = b_length[i]
      else:
        kz = 0
        z_body = 1.0
      volume_body = x_body * y_body * z_body

      # Spread number_density field in first neighbors' cells
      for ix in range(kx-1, kx+2):
        for iy in range(ky-1, ky+2):
          for iz in range(kz-1, kz+2):
            # Check if cell exist in the grid
            if (ix > -1) and (ix < mesh_points[0]) and (iy > -1) and (iy < mesh_points[1]) and (iz > -1) and (iz < mesh_points[2]):
              # Compute overlap assuming cubic (square) particle in 3D (2D)
              if mesh_points[0] > 1:
                x_min = max(lower_corner[0] +  ix    * dx, q[i,0] - b_length[i] * 0.5)
                x_max = min(lower_corner[0] + (ix+1) * dx, q[i,0] + b_length[i] * 0.5)
                x_length = max(x_max - x_min, 0)
              else:
                x_length = 1.0
              if mesh_points[1] > 1:                
                y_min = max(lower_corner[1] +  iy    * dy, q[i,1] - b_length[i] * 0.5)
                y_max = min(lower_corner[1] + (iy+1) * dy, q[i,1] + b_length[i] * 0.5)
                y_length = max(y_max - y_min, 0)
              else:
                y_length = 1.0
              if mesh_points[2] > 1:
                z_min = max(lower_corner[2] +  iz    * dz, q[i,2] - b_length[i] * 0.5)
                z_max = min(lower_corner[2] + (iz+1) * dz, q[i,2] + b_length[i] * 0.5)
                z_length = max(z_max - z_min, 0)
              else:
                z_length = 1.0
              volume_overlap = x_length * y_length * z_length

              # Save number_density
              if volume_overlap > 0:
                k = ix + iy * mesh_points[0] + iz * mesh_points[0] * mesh_points[1]
                number_density[k] += volume_overlap / (volume_body * volume_cell)
                velocity[k] += (volume_overlap / (volume_body * volume_cell)) * vw[i]

    sel = number_density > 0
    velocity[sel, 0] = velocity[sel, 0] / number_density[sel]
    velocity[sel, 1] = velocity[sel, 1] / number_density[sel]
    velocity[sel, 2] = velocity[sel, 2] / number_density[sel]
    velocity[sel, 3] = velocity[sel, 3] / number_density[sel]
    velocity[sel, 4] = velocity[sel, 4] / number_density[sel]
    velocity[sel, 5] = velocity[sel, 5] / number_density[sel]
    return number_density, velocity

  
  @staticmethod
  @njit(parallel=True, fastmath=True)
  def compute_stress_tensor(r_vectors, r_grid, force_blobs, blob_radius, beta):
    '''
    Compute stress like 
    
    stress = (I(r) - beta * I(infinity)) / r**3 * (f \tensor_product r_vec)
    
    with
    r_vec = displacement vector from blob to node
    I(r) = integral_0^{r} y**2 S(y) dy
    
    where S(y) is the kernel. We assume that it is a Gaussian
    S(y) = exp(-y**2 / (2*sigma**2)) / (2*pi*sigma**2)**1.5
    
    sigma = blob_radius / sqrt(pi)
    
    beta = 1 or 0 to make the stress calculation local or not.  
    '''
    # Variables
    sigma = blob_radius / np.sqrt(np.pi)
    Nblobs = r_vectors.size // 3
    Nnodes = r_grid.size // 3
    force_blobs = force_blobs.reshape((Nblobs, 3))
    stress = np.zeros((Nnodes, 9))
    factor_1 = 0.25 / np.pi
    factor_2 = 1.0 / (np.sqrt(2.0) * sigma)
    factor_3 = 1.0 / (np.power(2*np.pi, 1.5) * sigma)
    factor_4 = 1.0 / (2.0 * sigma**2)
    r = sigma * 100
    I_inf = factor_1 * math.erf(factor_2 * r) - factor_3 * r * np.exp(-factor_4 * r**2)
  
    rx_blobs = np.copy(r_vectors[:,0])
    ry_blobs = np.copy(r_vectors[:,1])
    rz_blobs = np.copy(r_vectors[:,2])
    rx_grid = np.copy(r_grid[:,0])
    ry_grid = np.copy(r_grid[:,1])
    rz_grid = np.copy(r_grid[:,2])

    for i in prange(Nnodes):
      rxi = rx_grid[i]
      ryi = ry_grid[i]
      rzi = rz_grid[i]
      for j in range(Nblobs):
        # Compute displacement vector and distance
        rx = rxi - rx_blobs[j]
        ry = ryi - ry_blobs[j] 
        rz = rzi - rz_blobs[j]

        # Compute distance
        r2 = rx*rx + ry*ry + rz*rz
        r = np.sqrt(r2)
        if r == 0:
          continue
      
        # Compute kernel integral
        I = factor_1 * math.erf(factor_2 * r) - factor_3 * r * np.exp(-factor_4 * r**2)
      
        # Compute stress
        factor_5 = (I - beta * I_inf) / r**3
        stress[i,0] += factor_5 * force_blobs[j,0] * rx
        stress[i,1] += factor_5 * force_blobs[j,0] * ry
        stress[i,2] += factor_5 * force_blobs[j,0] * rz
        stress[i,3] += factor_5 * force_blobs[j,1] * rx
        stress[i,4] += factor_5 * force_blobs[j,1] * ry
        stress[i,5] += factor_5 * force_blobs[j,1] * rz
        stress[i,6] += factor_5 * force_blobs[j,2] * rx
        stress[i,7] += factor_5 * force_blobs[j,2] * ry
        stress[i,8] += factor_5 * force_blobs[j,2] * rz
    return stress
