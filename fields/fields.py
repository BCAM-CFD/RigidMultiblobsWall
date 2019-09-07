import numpy as np
from numba import njit, prange
# Try to import the visit_writer (boost implementation)
try:
  import visit.visit_writer as visit_writer
except ImportError:
  pass
  

class fields(object):
  '''

  '''
  def __init__(self, grid_options, save_density = False, save_velocity = False):
    '''
    
    '''
    # Save general options
    self.save_density = save_density
    self.save_velocity = save_velocity
    
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

    # Be aware, x is the fast axis.
    zz, yy, xx = np.meshgrid(self.mesh_z, self.mesh_y, self.mesh_x, indexing = 'ij')
    self.mesh_coor = np.zeros((self.num_points, 3))
    self.mesh_coor[:,0] = np.reshape(xx, xx.size)
    self.mesh_coor[:,1] = np.reshape(yy, yy.size)
    self.mesh_coor[:,2] = np.reshape(zz, zz.size)

    # Create variables
    self.counter = 0
    if self.save_density:
      self.density_avg = np.zeros(self.num_points)
      self.density_var = np.zeros(self.num_points)
    if self.save_velocity:
      self.velocity_avg = np.zeros((self.num_points,3))
      self.velocity_var = np.zeros((self.num_points, 3))

    
    return



  def save(self, bodies, v = None):
    '''

    '''
    # Get bodies coordinates
    q = np.zeros((len(bodies), 3))
    b_length = np.zeros(len(bodies))
    if v is None:
      v = np.zeros_like(q)

    for i, b in enumerate(bodies):
      q[i] = np.copy(b.location)
      b_length[i] = b.body_length
      
    if self.save_density or save_velocity:
      density, velocity = self.compute_density_velocity(q, v, b_length, self.mesh_x, self.mesh_y, self.mesh_z, self.lower_corner, self.length_mesh, self.mesh_points)
      if self.save_density:
        self.density_avg += (density - self.density_avg) / (self.counter + 1)
        self.density_var += (density - self.density_avg)**2 * (self.counter / (self.counter+1)) 
      if self.save_velocity:
        self.velocity_avg += (velocity - self.velocity_avg) / (self.counter + 1)
        self.velocity_var += (velocity - self.velocity_avg)**2 * (self.counter / (self.counter+1)) 
      
    self.counter += 1
    return

  
  def restart(self):
    '''

    '''
    self.counter = 0
    if save_density:
      self.density_avg[:] = 0
      self.density_var[:] = 0
    if save_velocity:
      self.velocity_avg[:] = 0
      self.velocity_var[:] = 0
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
    
    if self.save_density:
      variables = [self.density_avg, self.density_var]
      dims = np.array([self.mesh_points[0]+1, self.mesh_points[1]+1, self.mesh_points[2]+1], dtype=np.int32)
      nvars = 2
      vardims =   np.array([1,1], dtype=np.int32)
      centering = np.array([0,0], dtype=np.int32)
      varnames = ['density\0', 'density_variance\0']
      name = name_output + '.density_field.vtk'

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
      variables = [np.copy(self.velocity_avg[:,0]), np.copy(self.velocity_avg[:,1]), np.copy(self.velocity_avg[:,2]), np.copy(self.velocity_var[:,0]), np.copy(self.velocity_var[:,1]), np.copy(self.velocity_var[:,2])]
      dims = np.array([self.mesh_points[0]+1, self.mesh_points[1]+1, self.mesh_points[2]+1], dtype=np.int32)
      nvars = 6
      vardims =   np.array([1,1,1,1,1,1], dtype=np.int32)
      centering = np.array([0,0,0,0,0,0], dtype=np.int32)
      varnames = ['velocity_X\0', 'velocity_Y\0', 'velocity_Z\0', 'velocity_variance_X\0', 'velocity_variance_Y\0', 'velocity_variance_Z\0']
      name = name_output + '.velocity_field.vtk'

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

      
    return


  @staticmethod
  @njit(parallel=True, fastmath=True)
  def compute_density_velocity(q, v, b_length, mesh_x, mesh_y, mesh_z, lower_corner, length_mesh, mesh_points):
    '''

    '''
    # Prepare variables
    N = b_length.size
    M = mesh_points[0] * mesh_points[1] * mesh_points[2]
    dx = length_mesh[0] / mesh_points[0]
    dy = length_mesh[1] / mesh_points[1]
    dz = length_mesh[2] / mesh_points[2]
    density = np.zeros(M)
    velocity = np.zeros((M, 3))

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
        y_body = 0
      if mesh_points[2] > 1:
        kz = int((q[i,2] - lower_corner[2]) / dz)
        z_body = b_length[i]
      else:
        kz = 0
        z_body = b_length[i]
      volume_body = x_body * y_body * z_body

      # Spread density field in first neighbors' cells
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

              # Save density
              if volume_overlap > 0:
                k = ix + iy * mesh_points[0] + iz * mesh_points[0] * mesh_points[1]
                density[k] += volume_overlap / volume_body
                velocity[k] += (volume_overlap / volume_body) * v[i]
    return density, velocity
