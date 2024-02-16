# Colloids with magnetic interactions
Example to simulate colloids in an unbounded slip.

## External magnetic field
The external magnetic field is computed in two steps, first

```
B1 = (B_x0 * cos(omega_x * t + phi_x), B_y0 * cos(omega_y * t + phi_y), B_z0 * cos(omega_z * t + phi_z))
```

then this magnetic field is rotated by the rotation matrix defined by the quaternion `quaternion_B`

```
B = R(quaternion_B) * B1
```


## Characteristic time scales
The utility code `characteristic_times.py` can be used to compute the characteristic times of the problem.


