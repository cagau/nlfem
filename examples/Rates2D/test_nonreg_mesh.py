# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:15:55 2019

@author: vollmann
"""


import sys
sys.path.append("/home/vollmann/Dropbox/JOB/PYTHON/BIB")
sys.path.append("/media/vollmann/DATA/Dropbox/JOB/PYTHON/BIB")
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
import numpy as np
import bib3 as bib
from time import time
from time import strftime
import os
import matplotlib.pyplot as plt
import conf2, conf
import MESH_nonreg
from scipy.spatial import Delaunay

# k = 0
# output_mesh_data = conf2.folder    + 'mesh_data_' + str(k) + '.npy'
#
#
def plot_tri(liste, closed, fill, color):
    for i in liste:
        plt.gca().add_patch(plt.Polygon(verts[i], closed=closed, fill = fill, color = color, alpha  = 1))
# a=2
# b=1
# def transform(x):
#     y1 = (1. - np.exp(-a * x[0])) / (1.- np.exp(-a))
#     y2 = np.sin(np.pi * x[1] / 2)**b
#     return np.array([y1, y2])
#     # return x
#
#
# mesh = MESH_nonreg.Mesh(np.load(output_mesh_data, allow_pickle=True))
#
#

#
# plt.figure()
# from scipy.spatial import Delaunay
# tri = Delaunay(mesh.verts, incremental = True)
# tris = tri.simplices
#
# """ ACHTUNG: LABELING interaction domain und domain passen nicht mehr !!!"""
# from scipy.spatial import Delaunay
#
# new_triangles = Delaunay(mesh.verts).simplices
# mesh.triangles[:, 1:] = new_triangles
#
# mesh.bary = (mesh.verts[mesh.triangles[:, 1]] + mesh.verts[mesh.triangles[:, 2]] + mesh.verts[
#     mesh.triangles[:, 3]]) / 3.
# new_omega = list(np.where(np.linalg.norm(mesh.bary - np.ones(1) * 0.5, axis=1, ord=np.inf) < 0.5)[0])
# new_omega_i = list(set(range(len(mesh.triangles))) - set(new_omega))
# mesh.triangles[new_omega, 0] = 1
# mesh.triangles[new_omega_i, 0] = 2
# mesh.omega = mesh.triangles[new_omega]
#
# labels_domain = np.sort(np.unique(mesh.triangles[:, 0])).tolist()
# for label in labels_domain:
#     omega = mesh.triangles[np.where(mesh.triangles[:, 0] == label)[0]]
#     plot_tri(omega[:, 1:], closed=True, fill=False, color=color[label - 1])
#
# plt.show()


# params = h1, h2, delta, transform_switch, transform

k = 0
h1, h2 = conf2.H1[k], conf2.H2[k]
delta = conf.delta

n1, n2 = int(np.ceil((1. + 2.*delta) / h1)) + 1, int(np.ceil((1. + 2.*delta)/h2)) + 1 # int(1./h)+1

x, ret = np.linspace(-delta, 1.+delta, n1, endpoint=True, retstep=1)
y = np.linspace(-delta, 1.+delta, n2, endpoint=True)
xv, yv = np.meshgrid(x, y, indexing='ij')


verts = np.around(np.array(np.meshgrid(x, y, indexing='ij')).T.reshape(-1, 2), decimals=12)

for j in range(len(verts)):
    plt.plot(verts[j][0], verts[j][1], 'ro')
    plt.annotate(str(j), ((verts[j][0]), (verts[j][1])) )
plt.show()
omega = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) < 0.5)
boundary = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) == 0.5)
omega_i = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) > 0.5)


verts = np.concatenate((verts[omega], verts[boundary], verts[omega_i]))

omega = list(range(len(verts[omega]))) #np.where(np.linalg.norm(verts - np.ones(1) * 0.5, axis=1, ord=np.inf) < 0.5)
boundary = list(range(len(verts[omega]), len(verts[omega])+len(verts[boundary]))) #np.where(np.linalg.norm(verts - np.ones(1) * 0.5, axis=1, ord=np.inf) == 0.5)
omega_i = list(range(len(verts[omega])+len(verts[boundary]), len(verts)) )#np.where(np.linalg.norm(verts - np.ones(1) * 0.5, axis=1, ord=np.inf) > 0.5)
nodes = omega

if conf2.transform_switch:

    for i in nodes:
        verts[i] = conf2.transform(verts[i])


triangles = Delaunay(verts).simplices
bary = (verts[triangles[:, 0]] + verts[triangles[:, 1]] + verts[
    triangles[:, 2]]) / 3.

triangles = np.concatenate( (np.zeros((len(triangles),1), dtype=np.int), triangles ), axis=1)



new_omega = list(np.where(np.linalg.norm(bary - np.ones(1) * 0.5, axis=1, ord=np.inf) < 0.5)[0])

new_omega_i = list(set(range(len(triangles))) - set(new_omega))

triangles[new_omega, 0] = 1
triangles[new_omega_i, 0] = 2

omega = triangles[new_omega]


def diam(T):
    length_of_edges = np.array(
        [np.linalg.norm(T[0] - T[1]), np.linalg.norm(T[0] - T[2]), np.linalg.norm(T[1] - T[2])])
    return np.max(length_of_edges)
diameter = [diam(np.array([verts[triangles[i,][1]], verts[triangles[i,][2]], verts[triangles[i,][3]]])) for i in range(len(triangles))]
diam = np.max(diameter)
print('grid size = max. diameter = ', diam)



###
# PLOT FOR TESTING
###
plt.figure()
color = ['black', 'red']
labels_domain = np.sort(np.unique(triangles[:, 0])).tolist()
for label in labels_domain:
    omega = triangles[np.where(triangles[:, 0] == label)[0]]
    plot_tri(omega[:, 1:], closed=True, fill=False, color=color[label - 1])

plt.plot(verts[nodes][:, 0], verts[nodes][:, 1], 'yo')
plt.plot(verts[boundary][:, 0], verts[boundary][:, 1], 'ro')
plt.plot(verts[omega_i][:, 0], verts[omega_i][:, 1], 'go')
plt.plot(bary[:, 0], bary[:, 1], 'bx')
plt.show()
plt.axis('equal')