#!/bin/sh
mesh=unit_square
gmsh $mesh.geo -format msh2 -clscale 0.025
