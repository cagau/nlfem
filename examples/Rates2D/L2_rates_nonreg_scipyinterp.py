# -*- coding: utf-8 -*-
"""
Created on Mar 25 2020
@author: vollmann
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import bib3 as bib
import scipy.optimize as optimization
import conf, conf2
import MESH_nonreg
import xlwt
import scipy.interpolate as interp
from assemble import evaluateMass
import nlocal

def main():
    numgrids = len(conf2.H1)
    num_fem_sols = conf2.num_grids
    k_finest = conf2.num_grids_mat_l2rates - 1 #num_fem_sols #- 1 # wir haben num_fem_sols solutions, aber nummeriert sind sie 0, 1, 2, num_fem_sols-1

    massmat = bib.load_sparse_csr(conf2.folder + 'M_omom_' + str(k_finest) + '.npz')

    H1 = [conf2.h1 * 2 ** -k for k in range(0, max([num_fem_sols, k_finest]))]
    H2 = [conf2.h2 * 2 ** -k for k in range(0, max([num_fem_sols, k_finest]))]
    h1_finest =  conf2.h1 * 2**-(k_finest)
    h2_finest =  conf2.h2 * 2**-(k_finest)

    ##=================================#
    #  LOAD AND EXTEND FEM SOLUTIONS   #
    ##=================================#
    # Load the solutions + extend them by the (sharp local) boundary data
    U = []
    gridsizes = np.zeros((num_fem_sols, 2))
    for k in range(num_fem_sols):
        output_u = conf2.folder + 'u_' + str(k)
        u = np.load(output_u + '.npy')
        output_mesh_data = conf2.folder + 'mesh_data_' + str(k) + '.npy'
        mesh = MESH_nonreg.Mesh(np.load(output_mesh_data, allow_pickle=True))
        nodes_inner = list(range(len(mesh.nodes) - len(mesh.boundary)))
        nodes_rest = list(range(len(nodes_inner), len(mesh.verts)))
        u_all = np.concatenate((u, list(map(conf2.g_d, mesh.verts[nodes_rest]))))
        U += [interp.LinearNDInterpolator(mesh.verts[nodes_inner+nodes_rest], u_all)]
        gridsizes[k,:] = np.array([np.min(mesh.diam), np.max(mesh.diam)])

    ##==========================#
    #  COMPUTE EXACT SOLUTION   #
    ##==========================#
    if conf2.gmsh:
        output_mesh_data = conf2.folder + 'mesh_data_' + str(k_finest) + '.npy'
        mesh = MESH_nonreg.Mesh(np.load(output_mesh_data, allow_pickle=True))
        nodes_inner = list(range(len(mesh.nodes) - len(mesh.boundary)))
        verts_finest = mesh.verts[nodes_inner]

    else:
        n1, n2 = int(1./h1_finest)+1, int(1./h2_finest)+1 # int(1./h)+1
        n1, n2 = int(np.ceil(1. / h1_finest)) + 1, int(np.ceil(1. / h2_finest)) + 1  # int(1./h)+1
        x = np.linspace(0., 1., n1, endpoint=True)
        y = np.linspace(0., 1., n2, endpoint=True)
        verts = np.around(np.array(np.meshgrid(x, y, indexing='ij')).T.reshape(-1, 2), decimals=12)
        omega = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) < 0.5)
        verts_finest = verts[omega]

    solution = verts_finest[:,0]**2 * verts_finest[:,1] + verts_finest[:,1]**2

    ##=================#
    #  COMPUTE RATES   #
    ##=================#
    def dist_L2(u,v):
        return np.sqrt( np.dot(u-v, massmat.dot(u-v) ) )
        # aux = evaluateMass(nlocal.Mesh(mesh, conf.ansatz, conf.boundaryConditionType), u-v, conf.py_Px, conf.dx)
        # return np.sqrt( np.dot(u-v, aux ) )

    error_l2 = np.zeros(len(U))
    for i in range(len(U)):
        error_l2[i] = dist_L2(U[i].__call__(verts_finest), solution)

    print()
    print()
    print('L2 ERROR')
    for a in error_l2:
        print(np.around(a, decimals = 4))

    print('\nRATES')
    rates_l2 = [0]
    for i in range(len(error_l2) - 1):
        rates_l2 += [np.log(error_l2[i] / error_l2[i + 1]) / np.log(2)]
        print(np.around(np.log(error_l2[i] / error_l2[i + 1]) / np.log(2), decimals = 2))
    rates_l2 = np.array(rates_l2)

    leftout = 0
    xdata = np.log(np.array([2 ** i for i in range(len(
        U) - leftout)]))  # np.array([0.6 * 2**-(i+1) for i in range(9)])# np.log(np.array([1./(0.2 * 2**-(i)) for i in range(9)]))
    ydata = np.log(error_l2)  # L2#
    x0 = np.zeros(2)
    sigma = np.ones(len(U) - leftout)
    def func(x, a, b):
        return a + b * x
    a, b = tuple(optimization.curve_fit(func, xdata, ydata, x0, sigma)[0])
    print()
    print('GRADIENT OF LINEAR FIT: ', str(b), "\n\n")

    #===================#
    # WRITE SPREADSHEET #
    #===================#
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet 1")
    header = ["min(diam)", "max(diam)","h1", "h2", "L2-Error", "Rate", "time [s]"]  # , "time-Rate"]
    output_ass_time = conf2.folder + 'ass_time' + '.npy'
    ass_time = np.array(np.load(output_ass_time))

    data = np.concatenate((np.around(gridsizes, decimals=7), np.array(conf2.H1[0:conf2.num_grids])[:, np.newaxis], np.array(conf2.H2[0:conf2.num_grids])[:, np.newaxis],
                           np.around(error_l2[:, np.newaxis], decimals=7), np.around(rates_l2[:, np.newaxis], decimals=2),
                            np.around(ass_time[:, np.newaxis])), axis=1)

    print("min(diam)", "max(diam)", " h1 ", " h2  "," L2-Error  ", "Rate  ", "   Assembly time [s]")
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    print(data)
    for i in range(len(header)):  # columns
        sheet1.write(1, 1 + i, header[i])
        for j in range(len(error_l2)):  # rows
            sheet1.write(1 + j + 1, 1 + i, data[j, i])
    book.save(conf2.folder+ conf2.folder[7:-1]  + "_rates.xls")

    #===================#
    # WRITE TEX TABLE   #
    #===================#
    textable = open(conf2.folder+"table.tex", "w")
    textable.write(
    """
    \documentclass{scrartcl}
    \\usepackage[ngerman]{babel}
    \\begin{document}
    
    \\begin{tabular}{ c   | c c  } \n
    \\multicolumn{1}{c }{}& \multicolumn{2}{c }{(4) \em method} \\\\	
    \\noalign{\smallskip}\hline\\noalign{\smallskip}
    $h$ 	 	& $\|u-u_h\|_{L^2}$   	&rate 		\\\\
    \\noalign{\smallskip}\hline\\noalign{\smallskip}
    """
    )

    for j in range(len(error_l2)):
        textable.write("$h_"  +  str(j+1)  +  "$ & "  +  str(data[j, 2])  +  "& "  +  str(data[j, 3])  +  "\\\\")

    textable.write(
    """
    \\noalign{\smallskip}\hline
    \\end{tabular}
         
    \end{document}
    """
    )

    textable.close()

if __name__ == "__main__":
    main()