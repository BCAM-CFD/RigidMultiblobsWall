# Example of dipole interactions
This example shows how to include dipole interactions between particles and an external magnetic field.

It is necessary to give some additional parameters in the input file.
`mu` is the dipole of the bodies in their body frame of reference.
All bodies have the same dipole.
`vacuum_permeability` by default it has the right value in units of micrometer, second and miligram.
`B0` the magnitude of the external magnetic field.
`omega` the frequency of the rotation of the external magnetic field.
In principle the magnetic field rotates in the xy plane.
`dipole_dipole` if set to True the bodies interact with dipole dipole interactions.

