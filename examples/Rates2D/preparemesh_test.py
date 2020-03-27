"""
Created on Mon Feb 25 13:31:10 2019

@author: vollmann
"""

import sys
import matplotlib.pyplot as plt
sys.path.append("/home/vollmann/Dropbox/JOB/PYTHON/BIB")
sys.path.append("/media/vollmann/1470789c-8ccb-4f32-9a95-12d258c8d70c/Dropbox/JOB/PYTHON/BIB")
import warnings
import assemble
import conf, conf2
import nlocal
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
import numpy as np
import examples.Rates2D.bib3 as bib
from time import time
from time import strftime
import scipy.sparse as sparse
import os
import MESH_nonreg



os.system('mkdir ' + conf2.folder)


# ------------------------------------------------------------------------------
print('\nconf2.Norm  = ', conf2.Norm, '\n\n')
ass_time = []
for k in range(conf2.num_grids)[1:2]:


    mesh1, mesh_data1 = MESH_nonreg.prepare_mesh_nonreg(conf2.H1[k], conf2.H2[k], conf.delta, conf2.transform_switch, conf2.transform)

    for j in range(len(mesh1.verts)):
        plt.plot(mesh1.verts[j][0], mesh1.verts[j][1], 'ro')
        plt.annotate(str(j), ((mesh1.verts[j][0]), (mesh1.verts[j][1])) )

    # proc_mesh_data = [triangles, omega, verts, [], boundary, nodes, [], diam, [], [], bary, boundary, [], [], [], len(verts), len(triangles), len(omega), len(nodes)]

    nodes_inner = range(len(mesh1.nodes) - len(mesh1.boundary))
    nodes_rest = range(len(nodes_inner), len(mesh1.verts))
    def plot_tri(liste, closed, fill, color):
        for i in liste:
            plt.gca().add_patch(plt.Polygon(mesh1.verts[i], closed=closed, fill=fill, color=color, alpha=1))
    # color = ['black', 'red']
    # labels_domain = np.sort(np.unique(mesh.triangles[:, 0])).tolist()
    # for label in labels_domain:
    #     omega = mesh.triangles[np.where(mesh.triangles[:, 0] == label)[0]]
    #     plot_tri(omega[:, 1:], closed=True, fill=False, color=color[label - 1])
    # plt.plot(mesh1.verts[nodes_inner][:, 0], mesh1.verts[nodes_inner][:, 1], 'bo')
    plt.plot(mesh1.verts[nodes_rest][:, 0], mesh1.verts[nodes_rest][:, 1], 'bo')
    # plt.plot(mesh1.verts[mesh1.boundary][:, 0], mesh1.verts[mesh1.boundary][:, 1], 'go')
    # plt.plot(mesh1.bary[:, 0], mesh1.bary[:, 1], 'rx')
    # plt.axis('equal')
    # plt.show()

    nodes_inner = range(len(mesh1.nodes) - len(mesh1.boundary))


    mesh2, mesh_data2 = MESH_nonreg.prepare_mesh_nonreg_depricated([conf2.H1[k], conf2.H2[k]], conf.delta, bib.norm_dict[conf2.Norm], conf2.num_cores, conf2.transform_switch, conf2.transform)

    plt.figure("depricated")
    for j in range(len(mesh2.verts)):
        plt.plot(mesh2.verts[j][0], mesh2.verts[j][1], 'ro')
        plt.annotate(str(j), ((mesh2.verts[j][0]), (mesh2.verts[j][1])) )



    nodes_inner = range(len(mesh2.nodes) - len(mesh2.boundary))
    nodes_rest = range(len(nodes_inner), len(mesh2.verts))

    def plot_tri(liste, closed, fill, color):
        for i in liste:
            plt.gca().add_patch(plt.Polygon(mesh2.verts[i], closed=closed, fill=fill, color=color, alpha=1))
    color = ['black', 'red']
    labels_domain = np.sort(np.unique(mesh2.triangles[:, 0])).tolist()
    # for label in labels_domain:
    #     omega = mesh2.triangles[np.where(mesh2.triangles[:, 0] == label)[0]]
    #     plot_tri(omega[:, 1:], closed=True, fill=False, color=color[label - 1])
    # plt.plot(mesh2.verts[nodes_inner][:, 0], mesh2.verts[nodes_inner][:, 1], 'rx')
    # plt.plot(mesh2.verts[nodes_rest][:, 0], mesh2.verts[nodes_rest][:, 1], 'rx')
    # plt.plot(mesh2.verts[mesh2.boundary][:, 0], mesh2.verts[mesh2.boundary][:, 1], 'yx')
    # plt.plot(mesh2.bary[:, 0], mesh2.bary[:, 1], 'bx')
    # plt.axis('equal')
    plt.show()



    # for i in range(len(mesh_data1)):
    #     print(i, np.allclose(mesh_data2[i], mesh_data1[i] ))
    #     print(i, mesh_data2[i].shape, mesh_data1[i].shape )

    # print(mesh_data1[10])
    # print(mesh_data2[10])