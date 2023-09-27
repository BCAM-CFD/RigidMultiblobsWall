# Example of spikes diffusion above a shell

## 1. Order of structures
The input file should give at least two structures.
The first one will be the shell, its clones files should have a single body.
The other structures are the spikes, they will be attracted to the shell by an harmonic potential and their orientation respect
the shell will be controlled by another harmonic potential.

See an example in the input file.

## 2. Variable blob radius
To use blobs with different radius you should do two things.
First, in the input file use the option

```
mobility_vector_prod_implementation      radii_numba_no_wall
```

or if you want simulations above an infinite wall

```
mobility_vector_prod_implementation      radii_numba
```

Second, if the vertex files have four columns the fourth sets the blob radius.
If not the blobs use the default value given in the input file.


## 3. Harmonic potential for the spikes centers
The center of mass of the spikes are attracted to the shell with an harmonic potential.
The formula

```
U = 0.5 * k * (r_norm - d0)**2
```

where `k` is the harmonic constant, `r_norm` the distances between the spikes and shell centers
and `d0` is the equilibrium distance.


## 4. Harmonic potential for the spikes orientation
An harmonic potential orients the spikes respect the shell.
The torque acting on the spikes is

```
torque = -k_angle * (r \times axis) / r_norm
```

where `r` is the vector from the spike to the shell center, `r_norm` is its norm, `k_angle` is the harmonic constant
and the most important `axis` is the axis (1,0,0) of the spike rotated to the laboratory frame of reference.


## 5. How to set the parameters
In the input file you can with the equilibrium distances and the harmonic constants with the option

```
omega_one_roller       d0  k  k_angle
```

## 6. How to modify the potential
Maybe you want to modify the potential.
For example, for the orientation you may want to use the axis (0,0,1).
You only have to edit the file `user_defined_functions.py`.

