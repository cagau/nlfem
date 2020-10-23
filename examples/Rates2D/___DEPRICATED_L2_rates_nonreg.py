# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:37:19 2019

@author: vollmann
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/vollmann/Dropbox/JOB/PYTHON/BIB")
sys.path.append("/media/vollmann/1470789c-8ccb-4f32-9a95-12d258c8d70c/Dropbox/JOB/PYTHON/BIB")

import bib3 as bib
import scipy.optimize as optimization
import conf, conf2
import MESH_nonreg
import xlwt
np.set_printoptions(linewidth = 400)
def main():


    numgrids = len(conf2.H1)

    num_fem_sols = conf2.num_grids
    k_finest = conf2.num_grids_mat_l2rates - 1 #num_fem_sols #- 1 # wir haben num_fem_sols solutions, aber nummeriert sind sie 0, 1, 2, num_fem_sols-1

    massmat = bib.load_sparse_csr(conf2.folder + 'M_omom_' + str(k_finest) + '.npz')

    H1 = [conf2.h1 * 2 ** -k for k in range(0, max([num_fem_sols, k_finest]))]
    H2 = [conf2.h2 * 2 ** -k for k in range(0, max([num_fem_sols, k_finest]))]

    h1_finest =  conf2.h1 * 2**-(k_finest)
    h2_finest =  conf2.h2 * 2**-(k_finest)

    def interpolate(u, L):
        """
        interpolates on regular grid on the square [0,1]^2
        arrays must contain values on the boundary

        L contains grid points in each dimension
        shape(u) = L[0] * L[1]
        """
        # L = int(np.sqrt(len(u)))
        u = u.reshape((L[0], L[1]))#, order = 'F')[::-1]#np.reshape(u, (L,L)#
        L1_new = 2*L[0] - 1
        L2_new = 2 * L[1] - 1

        U = np.zeros((L1_new, L2_new))

        for i in range(L[0]-1):
            for j in range(L[1]-1):

                U[2*i, 2*j]    = u[i,j]
                U[2*i, 2*j +1] = 0.5  * (u[i,j] + u[i,j+1])
                U[2*i+1,2*j]   = 0.5  * (u[i,j] + u[i+1,j])
                U[2*i+1,2*j+1] = 0.25 * (u[i,j] + u[i,j+1] + u[i+1,j]+ u[i+1,j+1])

        for j in range(L[1]-1):
            U[-1, 2*j]    = u[-1,j]
            U[-1, 2*j +1] = 0.5  * (u[-1,j] + u[-1,j+1])

        for i in range(L[0]-1):
            U[2*i, -1]    = u[i,-1]
            U[2*i+1,-1]   = 0.5  * (u[i,-1] + u[i+1,-1])

        U[-1,-1] = u[-1,-1]

        return U.reshape(L1_new * L2_new)



    #-----------------------------------------------------------------------------#

    # Load the solutions + extend them by the (sharp local) boundary data
    U = []
    for k in range(num_fem_sols):
        output_u = conf2.folder + 'u_' + str(k)
        u = np.load(output_u + '.npy')


        if conf2.depricated_mesh and conf2.H1[k] == 0.1:
            u = np.concatenate((u[9:], u[0:9]))

        L =  [int(np.ceil(1./H1[k]))-1, int(np.ceil(1./H2[k]))-1 ]


        print(k, len(u), L)

        #
        # plt.figure("loaded u's"+str(k))
        # plt.imshow(u.reshape(L[0], L[1]))
        # plt.show()
        #

        n1, n2 = L[0]+2, L[1]+2#int(1./h)+1
        x = np.linspace(0, 1, n1)
        y = np.linspace(0, 1, n2)
        xv, yv = np.meshgrid(x,y, indexing='ij')
        XY = np.meshgrid(x, y, indexing='ij')


        XY = conf2.transform(XY)
        # U_aux hat so exakt dieselbe nummerierung wie unser input u
        # dh: U_aux[i] und u[i] beinhaltne die Funktionswerte am selben knoten x_i
        U_aux = conf2.u_exact(XY).reshape(n1, n2)
        # U_aux = np.array([u_exact(np.array([xv[i,j], yv[i,j]])) for i in range(n1) for j in range(n2)]).reshape(n1,n2)

        ###
        # plt.figure(str(k)+'  Uaux   ')
        # plt.imshow(U_aux)
        ###

        #
        """
        ATTENTION: if numbering of verts is "from left to right, bottom to top" (opposed to my own mesh routine)
        then the order has to be 'F' !!
        """
        if conf2.depricated_mesh:
            # FOR DEPRICATED MESH!
            U_aux[1:-1, 1:-1] = np.reshape(u, (L[0], L[1]))
        else:
            U_aux[1:-1, 1:-1] = np.reshape(u, (L[0],L[1]) , order = 'F')#[::-1]

        if k == 1:
            print(np.around(U_aux, decimals = 2)[1:-1, 1:-1])

            print(np.around(np.reshape(u, (L[0], L[1])), decimals = 2))
            #
            # plt.figure(str(k) + ' new Uaux  ')
            # plt.imshow(U_aux.reshape(n1,n2))
            verts = np.around(np.array(np.meshgrid(x,y, indexing='ij')).T.reshape(-1, 2), decimals=12)
            # plt.tricontourf(verts[:, 0], verts[:, 1], U_aux.reshape(n1*n2), 100, interpolation='nearest',
            #                 cmap=plt.cm.get_cmap('rainbow'), vmin=np.min(U_aux), vmax=np.max(U_aux))
            plt.plot(verts[:, 0], verts[:, 1], 'rx')
            plt.show()
            #

        U_aux = U_aux.reshape(n1*n2)

        U += [U_aux]




    #-----------------------------------------------------------------------------#
    """ compute solution """
    n1, n2 = int(1./h1_finest)+1, int(1./h2_finest)+1 # int(1./h)+1
    x = np.linspace(0, 1, n1)[1:-1]
    y = np.linspace(0, 1, n2)[1:-1]
    xv, yv = np.meshgrid(x, y, indexing='ij')
    XY =  np.meshgrid(x, y, indexing='ij')


    XY = conf2.transform(XY)

    solution = conf2.u_exact(XY).reshape((n2-2)*(n1-2))#, order ='F')#np.array([conf2.u_exact(np.array([xv[i, j], yv[i, j]])) for i in range(n1) for j in range(n2)])  # np.reshape(##, (n2,n1), order = 'F')[::-1]
    ##
    # plt.figure('  solution   ')
    # plt.imshow(solution.reshape(n1-2, n2-2))
    # plt.show()
    ##



    #-----------------------------------------------------------------------------#
    # Interpolate all to finest mesh
    for i in range(0, len(U)):

        u_aux = U[i]



        for j in range(i, k_finest):
            L = [int(1./H1[j])+1, int(1./H2[j])+1]
            u_aux = interpolate(u_aux, L)

        L1, L2 = int(1. / h1_finest) + 1, int(1. / h2_finest) + 1
        u_aux = u_aux.reshape(L1, L2)[1:-1,1:-1]#(L2,L1), order = 'F')[::-1]

        #
        # plt.figure(str(i)+' interpolated u_aux   ')
        # plt.imshow(u_aux)
        # plt.show()
        #
        u_aux = u_aux.reshape((L1-2)*(L2-2))

        U[i] = u_aux




    #-----------------------------------------------------------------------------#
    def dist_L2(u,v):
        return np.sqrt( np.dot(u-v, massmat.dot(u-v) ) )

    error_l2 = np.zeros(len(U))

    for i in range(len(U)):
        error_l2[i] = dist_L2(U[i], solution)

    ##==============================================================================
    # """ L2"""
    ##==============================================================================

    print()
    print()
    print('L2 ERROR')
    for a in error_l2:
        print(a)

    print('RATES')
    rates_l2 = [0]
    for i in range(len(error_l2) - 1):
        rates_l2 += [np.log(error_l2[i] / error_l2[i + 1]) / np.log(2)]
        print(np.log(error_l2[i] / error_l2[i + 1]) / np.log(2))
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
    print('GRADIENT OF LINEAR FIT: ', b)


    ## PLOT ###
    def fit2(x):
        return a + b * x
    # plt.figure()
    ### plot log of error
    # plt.plot(xdata, ydata, 'rx' )
    ### plot linear fit
    # plt.plot(xdata, fit2(xdata))
    # plt.title('solution: l2 error')
    # plt.xlabel('log(h)')
    # plt.ylabel('log(error)')



    #===================#
    # WRITE SPREADSHEET #
    #===================#
    book = xlwt.Workbook(encoding="utf-8")
    sheet1 = book.add_sheet("Sheet 1")
    header = ["h1", "h2", "L2-Error", "Rate", "Assembly time [s]"]  # , "time-Rate"]
    output_ass_time = conf2.folder + 'ass_time' + '.npy'
    ass_time = np.array(np.load(output_ass_time))
    data = np.concatenate((np.array(conf2.H1[0:conf2.num_grids])[:, np.newaxis], np.array(conf2.H2[0:conf2.num_grids])[:, np.newaxis],
                           np.around(error_l2[:, np.newaxis], decimals=7), np.around(rates_l2[:, np.newaxis], decimals=2),
                           np.around(ass_time[:, np.newaxis])), axis=1)

    print("      h1   ", "      h2   ","        L2-Error  ", "       Rate  ", "   Assembly time [s]")
    print(str(data))
    for i in range(len(header)):  # columns
        sheet1.write(1, 1 + i, header[i])
        for j in range(len(error_l2)):  # rows
            sheet1.write(1 + j + 1, 1 + i, data[j, i])
    book.save(conf2.folder+ conf2.folder[7:-1]  + "_rates.xls")

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