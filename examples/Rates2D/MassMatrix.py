"""
Created on Mon Feb 25 13:31:10 2019

@author: vollmann
"""

import sys
import matplotlib.pyplot as plt
sys.path.append("/home/vollmann/Dropbox/JOB/PYTHON/BIB")
sys.path.append("/media/vollmann/1470789c-8ccb-4f32-9a95-12d258c8d70c/Dropbox/JOB/PYTHON/BIB")
import warnings
# import assemble
import conf, conf2
# import nlocal
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
import numpy as np
import examples.Rates2D.bib3 as bib
from time import time
from time import strftime
import scipy.sparse as sparse
import os
import MESH_nonreg

# ------------------------------------------------------------------------------
def main():
    for k in range(conf2.num_grids, conf2.num_grids_mat):
        output_mesh_data = conf2.folder    + 'mesh_data_' + str(k) + '.npy'

        output_M = conf2.folder            + 'M_' + str(k)
        output_M_full = conf2.folder       + 'M_full_' + str(k)
        output_M_omom = conf2.folder       + 'M_omom_' + str(k)

        print('-------------------------------------------------')
        print('h1 =', conf2.H1[k], ',       h2 =', conf2.H2[k])

        #  MESH
        if conf2.depricated_mesh:
            mesh, mesh_data = MESH_nonreg.prepare_mesh_nonreg_depricated([conf2.H1[k], conf2.H2[k]], conf.delta,
                                                                         bib.norm_dict[conf2.Norm], conf2.num_cores,
                                                                         conf2.transform_switch, conf2.transform)
        elif conf2.gmsh:
            mesh, mesh_data = MESH_nonreg.prepare_mesh_gmsh(conf2.H1[k], geofile=conf2.geofile)
        else:
            mesh, mesh_data = MESH_nonreg.prepare_mesh_nonreg(conf2.H1[k], conf2.H2[k], conf.delta,
                                                              conf2.transform_switch, conf2.transform)

        print('mesh done, start assembling...')
        print('\n Len(nodes): ', len(mesh.nodes))
        print(' Len(verts): ', len(mesh.verts))
        print(' Maximal diameter:', np.max(mesh.diam), '\n')
        # assembly mass matrix
        nodes_inner = range(len(mesh.nodes) - len(mesh.boundary))
        M = bib.mass_matrix2(mesh)# omega
        M_full = bib.mass_matrix_full(mesh)# full
        M.tolil()
        M_omom = M[nodes_inner[0]:nodes_inner[-1] + 1, nodes_inner[0]:nodes_inner[-1] + 1]# omom
        # SAVE
        bib.save_sparse_csr(output_M,       M.tocsr()) # M
        bib.save_sparse_csr(output_M_full,  M_full.tocsr())
        bib.save_sparse_csr(output_M_omom,  M_omom.tocsr())
        np.save(output_mesh_data, mesh_data)


if __name__ == "__main__":
    main()







