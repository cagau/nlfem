"""
Created on Mon Feb 25 13:31:10 2019

@author: vollmann
"""

import sys
import matplotlib.pyplot as plt
# sys.path.append("/home/vollmann/Dropbox/JOB/PYTHON/BIB")
# sys.path.append("/media/vollmann/1470789c-8ccb-4f32-9a95-12d258c8d70c/Dropbox/JOB/PYTHON/BIB")
import warnings
import nlcfem as assemble
import nlocal


import conf, conf2

warnings.filterwarnings('ignore', 'The iteration is not making good progress')
import numpy as np
import examples.Rates2D.bib3 as bib
from time import time
from time import strftime
import scipy.sparse as sparse
import os
import MESH_nonreg


def main():
    os.system('mkdir ' + conf2.folder)


    # ------------------------------------------------------------------------------
    print('\nconf2.Norm  = ', conf2.Norm, '\n\n')
    ass_time, gridsizes = [], []
    for k in range(conf2.num_grids):
        output_mesh_data = conf2.folder    + 'mesh_data_' + str(k)
        output_A = conf2.folder            + 'A_' + str(k)
        output_A_omom = conf2.folder       + 'A_omom_' + str(k)
        output_A_omomi = conf2.folder      + 'A_omomi_' + str(k)
        output_M = conf2.folder            + 'M_' + str(k)
        output_M_full = conf2.folder       + 'M_full_' + str(k)
        output_M_omom = conf2.folder       + 'M_omom_' + str(k)
        output_ass_time = conf2.folder     + 'ass_time'
        output_gridsizes = conf2.folder    + 'gridsizes'

        print('-------------------------------------------------')
        print('h1 =', conf2.H1[k], 'h2 =', conf2.H2[k],'\n')

        #  MESH


        if conf2.depricated_mesh:
            mesh, mesh_data = MESH_nonreg.prepare_mesh_nonreg_depricated([conf2.H1[k], conf2.H2[k]], conf.delta, bib.norm_dict[conf2.Norm], conf2.num_cores, conf2.transform_switch, conf2.transform)
        elif conf2.gmsh:
            mesh, mesh_data = MESH_nonreg.prepare_mesh_gmsh(conf2.H1[k], geofile = conf2.geofile)
        else:
            mesh, mesh_data = MESH_nonreg.prepare_mesh_nonreg(conf2.H1[k], conf2.H2[k], conf.delta, conf2.transform_switch, conf2.transform)


        if conf2.plot_mesh:

            def plot_tri(liste, closed, fill, color):
                for i in liste:
                    plt.gca().add_patch(plt.Polygon(mesh.verts[i], closed=closed, fill=fill, color=color, alpha=1))
            color = ['black', 'red']
            labels_domain = np.sort(np.unique(mesh.triangles[:, 0])).tolist()
            for label in labels_domain:
                omega = mesh.triangles[np.where(mesh.triangles[:, 0] == label)[0]]
                plot_tri(omega[:, 1:], closed=True, fill=False, color=color[label - 1])
            nodes_inner = range(len(mesh.nodes) - len(mesh.boundary))
            # plt.plot(mesh.verts[nodes_inner][:, 0], mesh.verts[nodes_inner][:, 1], 'ro')
            # plt.plot(mesh.verts[nodes_rest][:, 0], mesh.verts[nodes_rest][:, 1], 'go')
            # plt.plot(mesh.verts[mesh.boundary][:, 0], mesh.verts[mesh.boundary][:, 1], 'yo')
            # plt.plot(mesh.bary[:, 0], mesh.bary[:, 1], 'rx')
            plt.axis('equal')
            plt.show()



        print('\n mesh prepared and carefully saved \n')
        print('\n Len(nodes): ', len(mesh.nodes))
        print(' Len(verts): ', len(mesh.verts))
        print(' Maximal diameter:', np.max(mesh.diam) )
        print(' Minimal diameter:', np.min(mesh.diam), '\n')



        # ==============================================================================
        #              ASSEMBLY AND SAVE
        # ==============================================================================
        t1 = time()
        # if conf2.approx:
        #     A = bib.assembly_coupling_approx_full(mesh, gam, conf2.Norm,
        #                                           num_cores)  # bib.assembly_coupling_full_approx(mesh, gam, bib.retriangulate_dict[conf2.Ball], conf2.Norm, num_cores, hash_onthefly = hash_onthefly)
        # else:

        if conf2.Ball == 'exactcaps':
            A = bib.assembly_coupling_full_exactcaps(mesh, conf2.gam, bib.retriangulate_dict[conf2.ball], bib.norm_dict[conf2.Norm], conf2.num_cores, hash_onthefly = 1)
        else:
            A, f = assemble.assemble(nlocal.Mesh(mesh, conf.ansatz, conf.boundaryConditionType), conf.py_Px,
                                     conf.py_Py,
                                     conf.dx, conf.dy, conf.delta,
                                     model_f=conf.model_f,
                                     model_kernel=conf.model_kernel,
                                     integration_method=conf2.integration_method,
                                     is_PlacePointOnCap = conf2.is_PlacePointOnCap)

            # A = bib.assembly_coupling_full_standard(mesh, conf2.gam, bib.retriangulate_dict[conf2.ball], bib.norm_dict[conf2.Norm], conf2.num_cores, hash_onthefly = 1)
            # A = bib.assembly_coupling_full_bary(mesh, conf2.gam, bib.retriangulate_dict[conf2.ball], bib.norm_dict[conf2.Norm], conf2.num_cores, hash_onthefly = 1)
            # A = bib.assembly_coupling_full_shifted(mesh, gam, bib.retriangulate_dict[conf2.Ball], conf2.Norm, num_cores, hash_onthefly = 1)

        ass_time += [time() - t1 ]
        gridsizes += [np.min(mesh.diam)]
        print('\n time for A:', time() - t1 ,'\n')
        print('-------------------------------------------------')

        A = sparse.lil_matrix(A)
        nodes_inner = range(len(mesh.nodes) - len(mesh.boundary))
        A_omom = A[nodes_inner[0]:nodes_inner[-1] + 1, nodes_inner[0]:nodes_inner[-1] + 1]
        A_omomi = A[nodes_inner[0]:nodes_inner[-1] + 1, nodes_inner[-1] + 1:]

        # assembly mass matrix
        M = bib.mass_matrix2(mesh)# omega
        M_full = bib.mass_matrix_full(mesh)# full
        M.tolil()
        M_omom = M[nodes_inner[0]:nodes_inner[-1] + 1, nodes_inner[0]:nodes_inner[-1] + 1]# omom

        # SAVE
        np.save(output_mesh_data, mesh_data)
        bib.save_sparse_csr(output_A,       A.tocsr()) # A
        bib.save_sparse_csr(output_A_omom,  A_omom.tocsr())
        bib.save_sparse_csr(output_A_omomi, A_omomi.tocsr())
        bib.save_sparse_csr(output_M,       M.tocsr()) # M
        bib.save_sparse_csr(output_M_full,  M_full.tocsr())
        bib.save_sparse_csr(output_M_omom,  M_omom.tocsr())
        np.save(output_ass_time, ass_time)
        np.save(output_gridsizes, gridsizes)



    # ==============================================================================
    #              PRINT INPUT - PARAMETERS INTO TEXT-FILE
    # ==============================================================================
    timestr = strftime('%d-%m-%Y')
    config1 = open('conf.py', 'r')
    config2 = open('conf2.py', 'r')
    textfile = open(conf2.folder + 'parameters.txt', 'w')
    textfile.write("\n----------------------\n  DATE: "+timestr+"\n----------------------\n")
    textfile.write("\n\nCONF 2: \n----------------------\n")
    for line in config2:
        textfile.write(line)
    textfile.write("\n\nCONF John: \n----------------------\n")
    for line in config1:
        textfile.write(line)

    config1.close()
    config2.close()
    textfile.close()

if __name__ == "__main__":
    main()