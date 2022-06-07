
## To install meshio use this command "pip install meshio"

import numpy as np
import meshio
import argparse

import scipy.linalg
import subprocess
from shutil import copyfile
from functools import partial
import sys
import time
import linecache
import re


# Find project functions
found_functions = False
path_to_append = ''
while found_functions is False:
  try:
    from read_input import read_input
    try:
      found_HydroGrid = True
    except ImportError:
      found_HydroGrid = False
    found_functions = True
  except ImportError as exc:
    sys.stderr.write('Error: failed to import settings module ({})\n'.format(exc))
    path_to_append += '../'
    print('searching functions in path ', path_to_append)
    sys.path.append(path_to_append)
    if len(path_to_append) > 21:
      print('\nProjected functions not found. Edit path in multi_bodies.py')
      sys.exit()

# Try to import the visit_writer (boost implementation)
try:
  # import visit.visit_writer as visit_writer
  from visit import visit_writer as visit_writer
except ImportError as e:
  print(e)
  pass


num_steps = 25
string = 'plot_velocity_field'
file_name = 'inputfile_bacteria_constant_torque.dat'
file = open(file_name, "r")

flag = 0
index = 0  
for line in file:  
  index  = index + 1 
  if string in line:
    flag = 1
    break 
if flag == 0: 
  print('String', string , 'Not Found') 
else: 
  print('String', string, 'Found In Line', index)
file.close() 

string_line = linecache.getline(file_name, index)
numbers = [int(s) for s in re.findall(r'[-+]?\d+[,.]?\d*', string_line)]
grid = numbers[0:9]
print(grid)  
   


mesh = meshio.read('run_constant_torque.step.00000000.velocity_field.vtk')
points = mesh.points
cells = mesh.cells
vel = mesh.cell_data
velocity = vel['velocity'][0]
vel_sum = np.zeros(velocity.shape)

count = 0
for i in range(1,num_steps):
  name = 'run_constant_torque.step.'  + str(i).zfill(8)  + '.velocity_field.vtk'
  print(name)
  mesh = meshio.read(name)
  points = mesh.points
  cells = mesh.cells
  vel = mesh.cell_data
  velocity = vel['velocity'][0]
  vel_sum = vel_sum + velocity
  count = count+1
  
vel_avg = vel_sum /count 



if True:
  
# Prepare grid values
  grid = np.reshape(grid[0:9], (3,3)).T
  grid_length = grid[1] - grid[0]
  grid_points = np.array(grid[2], dtype=np.int32)
  num_points = grid_points[0] * grid_points[1] * grid_points[2]

# Set grid coordinates
  dx_grid = grid_length / grid_points
  grid_x = np.array([grid[0,0] + dx_grid[0] * (x+0.5) for x in range(grid_points[0])])
  grid_y = np.array([grid[0,1] + dx_grid[1] * (x+0.5) for x in range(grid_points[1])])
  grid_z = np.array([grid[0,2] + dx_grid[2] * (x+0.5) for x in range(grid_points[2])])
# Be aware, x is the fast axis.
  zz, yy, xx = np.meshgrid(grid_z, grid_y, grid_x, indexing = 'ij')
  grid_coor = np.zeros((num_points, 3))
  grid_coor[:,0] = np.reshape(xx, xx.size)
  grid_coor[:,1] = np.reshape(yy, yy.size)
  grid_coor[:,2] = np.reshape(zz, zz.size)

  grid_velocity = vel_avg

 # Prepara data for VTK writer 
  variables = [np.reshape(grid_velocity, grid_velocity.size)] 
  dims = np.array([grid_points[0]+1, grid_points[1]+1, grid_points[2]+1], dtype=np.int32) 
  nvars = 1
  vardims = np.array([3])
  centering = np.array([0])
  varnames = ['velocity\0']
  name = 'average' + '.velocity_field.vtk'
  grid_x = grid_x - dx_grid[0] * 0.5
  grid_y = grid_y - dx_grid[1] * 0.5
  grid_z = grid_z - dx_grid[2] * 0.5
  grid_x = np.concatenate([grid_x, [grid[1,0]]])
  grid_y = np.concatenate([grid_y, [grid[1,1]]])
  grid_z = np.concatenate([grid_z, [grid[1,2]]])

  

# Write velocity field
visit_writer.boost_write_rectilinear_mesh(name,      # File's name
                                            0,         # 0=ASCII,  1=Binary
                                            dims,      # {mx, my, mz}
                                            grid_x,     # xmesh
                                            grid_y,     # ymesh
                                            grid_z,     # zmesh
                                            nvars,     # Number of variables
                                            vardims,   # Size of each variable, 1=scalar, velocity=3*scalars
                                            centering, # Write to cell centers of corners
                                            varnames,  # Variables' names
                                            variables) # Variables
  


