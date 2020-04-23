
"""
Created on Mon Feb 25 14:59:05 2019

@author: vollmann
"""

import bib3 as bib
import numpy as np
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
import conf, conf2
import MESH_nonreg
import scipy.sparse as ss

#==============================================================================
#                                   INPUT
#==============================================================================




def main():
    #==============================================================================

    def plot_tri(liste, closed, fill, color):
        for i in liste:
            plt.gca().add_patch(plt.Polygon(mesh.verts[i], closed=closed, fill = fill, color = color, alpha  = 0.1))
    #==============================================================================
    #                 GO
    #==============================================================================
    for k in range(conf2.num_grids):
        print("Run: "+str(k+1)+"\n------\n")
        output_mesh_data = conf2.folder    + 'mesh_data_' + str(k) + '.npy'
        output_A = conf2.folder            + 'A_' + str(k)+ '.npz'
        output_A_omom = conf2.folder       + 'A_omom_' + str(k)+ '.npz'
        output_A_omomi = conf2.folder      + 'A_omomi_' + str(k) + '.npz'
        output_M = conf2.folder            + 'M_' + str(k) + '.npz'
        output_M_full = conf2.folder       + 'M_full_' + str(k) + '.npz'
        output_M_omom = conf2.folder       + 'M_omom_' + str(k) + '.npz'
        output_u      = conf2.folder       + 'u_' + str(k)

        M = bib.load_sparse_csr(output_M)
        M_omom = bib.load_sparse_csr(output_M_omom)
        A_omom = bib.load_sparse_csr(output_A_omom)
        A_omomi = bib.load_sparse_csr(output_A_omomi)


        mesh = MESH_nonreg.Mesh(np.load(output_mesh_data, allow_pickle=True))



        print('start rhs')
        p = Pool(conf2.num_cores)
        aux = np.array(p.map(conf2.source, mesh.verts[mesh.nodes]))
        b = M.dot(aux)
        print('rhs computed')
        nodes_inner = range(len(mesh.nodes)-len(mesh.boundary))
        nodes_rest = range(len(nodes_inner), len(mesh.verts)) # Nodes in Omega_I (including the boundary)
        b_aux = b[nodes_inner] - A_omomi.dot(np.array(list(map(conf2.g_d, mesh.verts[nodes_rest]))))

        print( 'start solving \n')
        # u = ssl.spsolve(A_omom, b_aux)
        u = ssl.cg(A_omom, b_aux)[0]

        np.save(output_u, u)#np.concatenate((u, map(conf2.g_d, mesh.verts[nodes_rest]))))

        if conf2.plot_solve:
            if k == 0:
                # plt.figure('u_reshaped'+str(k)+ '   APPROX')
                # plt.imshow(u.reshape(int(np.sqrt(u.size)), int(np.sqrt(u.size))))
                # plt.show()



                color = ['black', 'red']
                #approximation
                u_all = np.concatenate((u, list(map(conf2.g_d, mesh.verts[nodes_rest]))))

                bib.plot_all(mesh, u_all, title = 'u'+str(k)+ '   APPROX')

                # exact
        #            bib.plot_all(mesh, np.array(map(conf2.g_d, mesh.verts)), title = 'u'+params+ '   EXACT')
                # error
        #            bib.plot_all(mesh, np.abs( np.concatenate((u, map(conf2.g_d, mesh.verts[nodes_rest])))-np.array(map(conf2.g_d, mesh.verts)) ), title = 'u'+params + '   ERROR')
                labels_domain = np.sort(np.unique(mesh.triangles[:,0])).tolist()
                for label in labels_domain:
                   omega = mesh.triangles[np.where(mesh.triangles[:,0] == label)[0]]
                   plot_tri(omega[:,1:], closed = True, fill = False, color = color[label-1])
                nodes_inner = range(len(mesh.nodes) - len(mesh.boundary))
                # plt.plot(mesh.verts[nodes_inner][:,0], mesh.verts[nodes_inner][:,1], 'ro')
                # plt.plot(mesh.verts[mesh.boundary][:, 0], mesh.verts[mesh.boundary][:, 1], 'yo')
                # plt.plot(mesh.bary[:, 0], mesh.bary[:, 1], 'rx')
                plt.axis('equal')
                plt.show()

if __name__ == "__main__":
    main()