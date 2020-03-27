import numpy as np
import os
import bib3 as bib
import conf2
import matplotlib.pyplot as plt

geofile = "unit_square"

k = 1

# ##### GMSH FILE adaption with correct interaction horizon for outer boundary
# textfile = open('mesh/'+geofile, 'r')
# data = textfile.readlines()
#
# tmpfile = open('mesh/test.txt', 'w+')
# tmpfile.write('delta = ' + str(conf.delta) + ';\n')
#
# for line in data[1:]:
#     tmpfile.write(line)
#
# tmpfile.close()
#
# os.system('rm mesh/'+geofile+'.geo')
# current_path = os.path.dirname(os.path.abspath(__file__))
# os.system('mv '+current_path+'/mesh/test.txt ' + current_path +'/mesh/'+fil_init+'.geo')
# ##### GMSH FILE adation END #####


os.system('gmsh -v 0 mesh/'+geofile+'.geo -2 -clscale '+str(conf2.H1[k])+' -o mesh/'+geofile+'.msh' )
verts, lines, triangles = bib.read_mesh('mesh/'+geofile+'.msh')
verts = verts[:,0:2]
aux = np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf)
sort_indices = np.argsort(aux)
verts = verts[sort_indices]
sort_indices_inv = np.arange(len(sort_indices))[np.argsort(sort_indices)]
def f(n):
    return sort_indices_inv[n]
triangles[:,1:] = f(triangles[:,1:])

### plot
# for list in triangles[:, 1:]:
#     plt.gca().add_patch(plt.Polygon(verts[list], closed=1, fill=0, alpha=1))
boundary = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) == 0.5)
nodes = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) <= 0.5)
bary = (verts[triangles[:, 1]] + verts[triangles[:, 2]] + verts[triangles[:, 3]]) / 3.
new_omega = list(np.where(np.linalg.norm(bary - np.ones(1) * 0.5, axis=1, ord=np.inf) < 0.5)[0])
new_omega_i = list(set(range(len(triangles))) - set(new_omega))
triangles[new_omega, 0] = 1
triangles[new_omega_i, 0] = 2
omega = triangles[new_omega]
def diam(T):
    length_of_edges = np.array(
        [np.linalg.norm(T[0] - T[1]), np.linalg.norm(T[0] - T[2]), np.linalg.norm(T[1] - T[2])])
    return np.max(length_of_edges)

diameter = [diam(np.array([verts[triangles[i,][1]], verts[triangles[i,][2]], verts[triangles[i,][3]]])) for i in
            range(len(triangles))]
diam = np.max(diameter)

proc_mesh_data = [triangles, omega, verts, [], boundary, nodes, [], diam, [], [], bary,
                  boundary, [], [], [], len(verts), len(triangles), len(omega), len(nodes)]
# mesh = Mesh(proc_mesh_data)



###
# PLOT FOR TESTING
###
# def plot_tri(liste, closed, fill, color):
#     for i in liste:
#         plt.gca().add_patch(plt.Polygon(verts[i], closed=closed, fill = fill, color = color, alpha  = 1))
# plt.figure()
# color = ['black', 'red']
# labels_domain = np.sort(np.unique(triangles[:, 0])).tolist()
# for label in labels_domain:
#     omega = triangles[np.where(triangles[:, 0] == label)[0]]
#     plot_tri(omega[:, 1:], closed=True, fill=False, color=color[label - 1])
#
# nodes = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) < 0.5)
# omega_i = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) > 0.5)
# plt.plot(verts[nodes][:, 0], verts[nodes][:, 1], 'yo')
# plt.plot(verts[boundary][:, 0], verts[boundary][:, 1], 'ro')
# plt.plot(verts[omega_i][:, 0], verts[omega_i][:, 1], 'go')
# plt.plot(bary[:, 0], bary[:, 1], 'bx')
# plt.show()
# plt.axis('equal')