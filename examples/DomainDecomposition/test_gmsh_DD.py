import numpy as np
import os
import bib3 as bib
import matplotlib.pyplot as plt
# import MESH_nonreg

geofile = "DD_simple"

element_size = 0.05
delta = 0.1

# ##### GMSH FILE adaption with correct interaction horizon for outer boundary
textfile = open('mesh/'+geofile+'.geo', 'r')
data = textfile.readlines()

tmpfile = open('mesh/test.txt', 'w+')
tmpfile.write('delta = ' + str(delta) + ';\n')

for line in data[1:]:
    tmpfile.write(line)

tmpfile.close()

os.system('rm mesh/'+geofile+'.geo')
current_path = os.path.dirname(os.path.abspath(__file__))
os.system('mv '+current_path+'/mesh/test.txt ' + current_path +'/mesh/'+geofile+'.geo')
# ##### GMSH FILE adation END #####


os.system('gmsh -v 0 mesh/'+geofile+'.geo -2 -clscale '+str(element_size)+' -o mesh/'+geofile+'.msh' )
verts, lines, triangles = bib.read_mesh('mesh/'+geofile+'.msh')
triangles = triangles[triangles[:,0].argsort()]

labels_domains = np.sort(np.unique(triangles[:,0]))

verts = verts[:,0:2]




# sort verts so that vertices in Omega_I are at the end
aux = np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf)
sort_indices = np.argsort(aux)
verts = verts[sort_indices]
sort_indices_inv = np.arange(len(sort_indices))[np.argsort(sort_indices)]
def f(n):
    return sort_indices_inv[n]
triangles[:,1:] = f(triangles[:,1:])
lines[:,1:] = f(lines[:,1:])
#------------------------------------------------------------------------



# extract subdomains and interfaces
# omega1 = triangles[np.where(triangles[:, 0] == labels_domains[0])]
# nodes1 = list(np.unique(omega1[:, 1:4]))
# omega2 = triangles[np.where(triangles[:, 0] == labels_domains[1])]
# nodes2 = list(np.unique(omega2[:, 1:4]))
# omega_I = triangles[np.where(triangles[:, 0] == labels_domains[2])]
# nodes_I = list(np.unique(omega_I[:, 1:4]))
# interface = np.unique(lines[np.where(lines[:,0]==12)][:,1:]).tolist()
# boundary1 = np.unique(lines[np.where(lines[:,0]==11)][:,1:]).tolist()
# boundary2 = np.unique(lines[np.where(lines[:,0]==22)][:,1:]).tolist()
# for i in omega1:
#     plt.gca().add_patch(plt.Polygon(verts[i[1:]], closed=True, fill=False, color='r', alpha=1))
# for i in omega2:
#     plt.gca().add_patch(plt.Polygon(verts[i[1:]], closed=True, fill=False, color='b', alpha=1))
# for i in omega_I:
#     plt.gca().add_patch(plt.Polygon(verts[i[1:]], closed=True, fill=False, color='black', alpha=1))
# plt.plot(verts[interface][:,0], verts[interface][:,1], 'bo')
# plt.plot(verts[boundary1][:,0], verts[boundary1][:,1], 'rx')
# plt.plot(verts[boundary2][:,0], verts[boundary2][:,1], 'gx')
# plt.show()
# plt.axis('equal')



# BARYCENTERS
bary = (verts[triangles[:, 1]] + verts[triangles[:, 2]] + verts[triangles[:, 3]]) / 3.
#------------------------------------------------------------------------


# DIAMETER
def diam(T):
    length_of_edges = np.array(
        [np.linalg.norm(T[0] - T[1]), np.linalg.norm(T[0] - T[2]), np.linalg.norm(T[1] - T[2])])
    return np.max(length_of_edges)
diameter = [diam(np.array([verts[triangles[i,][1]], verts[triangles[i,][2]], verts[triangles[i,][3]]])) for i in
            range(len(triangles))]
diam = np.max(diameter)
#------------------------------------------------------------------------


# Nonlocal Interface
Gamma_hat = []
interface = np.unique(lines[np.where(lines[:,0]==12)][:,1:]).tolist()
for i in interface:
    Gamma_hat += np.where(np.linalg.norm(bary - np.tile(verts[i], (len(bary),1)) , axis =1) <= delta/2.+diam)[0].tolist()
for i in Gamma_hat:
    plt.gca().add_patch(plt.Polygon(verts[triangles[i,1:]], closed=True, fill=False, color='olive', alpha=1))
plt.show()
plt.axis('equal')
#------------------------------------------------------------------------


# GENERATE MESH 1
def submesh_gen(k):
    boundary1 = np.unique(lines[np.where(lines[:, 0] == 11*(k+1))][:, 1:]).tolist()
    Gamma_1 = []
    for i in interface:
        Gamma_1 += np.where(np.linalg.norm(bary - np.tile(verts[i], (len(bary),1)) , axis =1) <= delta/2. + diam)[0].tolist()
    for i in boundary1:
        Gamma_1 += np.where(np.linalg.norm(bary - np.tile(verts[i], (len(bary),1)) , axis =1) <= delta + diam)[0].tolist()
    Gamma_1 = list(np.unique(np.array(Gamma_1)))
    triangles_1_bool = np.zeros(len(triangles), dtype = bool)
    triangles_1_bool[np.where(triangles[:, 0] == labels_domains[k])] = True
    triangles_1_bool[Gamma_1] = True
    nE_1 = len(triangles[triangles_1_bool])
    triangles_1 = triangles[triangles_1_bool]

    element_labels_1 = np.zeros(nE_1, dtype = bool)
    element_labels_1[np.where(triangles_1[:, 0] != labels_domains[2])] = True
    triangles_1 = triangles_1[:,1:]

    vertices_1_bool = np.zeros(len(verts), dtype = bool)
    aux = np.unique(triangles_1.reshape(3*len(triangles_1)))
    vertices_1_bool[aux] = True
    vertices_1 = verts[vertices_1_bool]
    nV_1 = len(vertices_1)

    # embedding elements
    aux_elements = -np.ones(len(triangles), dtype=int)
    aux_elements[np.where(triangles_1_bool)[0]] = np.arange(nE_1)
    embedding_elements_1 = np.array([list(aux_elements).index(i) for i in range(len(triangles_1))])

    # nummern in triangles_1 anpassen, damit sie zu vertices_1 passen
    aux_vertices = -np.ones(len(verts), dtype=int)
    aux_vertices[np.where(vertices_1_bool)[0]] = np.arange(nV_1)
    embedding_vertices_1 = np.array([list(aux_vertices).index(i) for i in range(len(vertices_1))])

    def large_to_small_1(i):
        return aux_vertices[i]
    triangles_1 = large_to_small_1(triangles_1)

    return vertices_1, triangles_1, embedding_vertices_1, embedding_elements_1, element_labels_1
#------------------------------------------------------------------------


# PLOT FOR TESTING
vertices_1, triangles_1, embedding_vertices_1, embedding_elements_1, element_labels_1 = submesh_gen(1)

plt.figure()
for i in range(len(triangles_1)):
    if element_labels_1[i]:
        if np.any(triangles[embedding_elements_1[i]][1:] != embedding_vertices_1[triangles_1[i]]):
            print(i, triangles[embedding_elements_1[i]][1:], embedding_vertices_1[triangles_1[i]])
        plt.gca().add_patch(plt.Polygon(verts[triangles[embedding_elements_1[i]][1:]], closed=True, fill=True, color='magenta',alpha=1))
    else:
        plt.gca().add_patch(plt.Polygon(verts[triangles[embedding_elements_1[i]][1:]], closed=True, fill=False, color='black', alpha=1))
    bary_i = verts[triangles[embedding_elements_1[i]][1:]].sum(axis=0)/3.
    plt.annotate(str(i), (bary_i[0], bary_i[1]))
plt.show()
plt.axis('equal')

plt.figure()
for i in range(len(triangles_1)):
    if element_labels_1[i]:
        plt.gca().add_patch(plt.Polygon(verts[embedding_vertices_1[triangles_1[i]]], closed=True, fill=True, color='blue',alpha=1))
    else:
        plt.gca().add_patch(plt.Polygon(verts[embedding_vertices_1[triangles_1[i]]], closed=True, fill=False, color='black', alpha=1))
plt.show()
plt.axis('equal')

plt.figure()
for i in range(len(triangles_1)):
    if element_labels_1[i]:
        plt.gca().add_patch(plt.Polygon(vertices_1[triangles_1[i]], closed=True, fill=True, color='red', alpha=1))
    else:
        plt.gca().add_patch(plt.Polygon(vertices_1[triangles_1[i]], closed=True, fill=False, color='black', alpha=1))
    bary_i = vertices_1[triangles_1[i]].sum(axis=0) / 3.
    plt.annotate(str(i), (bary_i[0], bary_i[1]))
plt.show()
plt.axis('equal')


