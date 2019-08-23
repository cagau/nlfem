#-*- coding:utf-8 -*-
#cython: language_level=3
# #cython: boundscheck=False, cdivision=True
# Setting this compiler directive will given a minor increase of speed.

cimport cython
import numpy as np
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nbhd import xnotinNbhd
from libc.math cimport sqrt, pow, pi, floor

class clsFEM:
    """ **Mesh Class**

    Let :math:`K` be the number of basis functions and :math:`J` the number of finite elements. The ordering of the vertices
    V is such that the first :math:`K_{\Omega}` vertices lie in the interior of :math:`\Omega`.

    :ivar V: nd.array, real, shape (K, 2) List of vertices in the 2D plane
    :ivar T: nd.array, int, shape (J, 3) List of indices mapping an index Tdx to 3 corresponding vertices of V.
    :ivar K: Number of basis functions.
    :ivar K_Omega: Number if basis functions in the interior of :math:`\Omega`.
    :ivar J: Number of finite elements :math:`\Omega`.
    :ivar J_Omega: Number of finite elements in
    """
    def __init__(self, mshfile, ansatz):
        """Constructor

        Executes read_mesh and prepare.
        """
        args = self.mesh(*self.read_mesh(mshfile))
        # args = Verts, Triangles, J, J_Omega, L, L_Omega
        self.V = args[0]
        self.T = args[1][:, 1:]
        self.J = args[2]
        self.J_Omega = args[3]
        self.L = args[4]
        self.L_Omega = args[5]

        args = self.basis(ansatz=ansatz)
        self.K = args[0]
        self.K_Omega = args[1]
        self.Adx = args[2]
        self.Adx_inOmega = args[3]

        self.Neighbours = []
        self.Triangles = []

        for Tdx in range(self.J):
            Vdx = self.T[Tdx]
            self.Triangles.append(clsTriangle(self.V[Vdx]))
            self.Neighbours.append(self.get_neighbor(Tdx))

    def read_mesh(self, mshfile):
        """meshfile = .msh - file genrated by gmsh


        :param mshfile:
        :return: Verts, Lines, Triangles
        """

        fid = open(mshfile, "r")

        for line in fid:

            if line.find('$Nodes') == 0:
                # falls in der Zeile 'Nodes' steht, dann steht in der...
                line = fid.readline()  # ...naechsten Zeile...
                npts = int(line.split()[0])  # ..die anzahl an nodes

                Verts = np.zeros((npts, 3), dtype=float)  # lege array for nodes an anzahl x dim

                for i in range(0, npts):
                    # run through all nodes
                    line = fid.readline()  # put current line to be the one next
                    data = line.split()  # split line into its atomic characters
                    Verts[i, :] = list(map(float, data[
                                                  1:]))  # read out the coordinates of the node by applying the function float() to the characters in data

            if line.find('$Elements') == 0:
                line = fid.readline()
                nelmts = int(line.split()[0])  # number of elements

                Lines = []
                Triangles = []
                # Squares = np.array([])

                for i in range(0, nelmts):
                    line = fid.readline()
                    data = line.split()
                    if int(data[1]) == 1:
                        """ 
                        we store [physical group, node1, node2, node3], 
                        -1 comes from python starting to count from 0
                        """
                        # see ordering:

                        #                   0----------1 --> x

                        Lines += [int(data[3]), int(data[-2]) - 1, int(data[-1]) - 1]

                    if int(data[1]) == 2:
                        """
                        we store [physical group, node1, node2, node3]
                        """
                        # see ordering:

                        #                    y
                        #                    ^
                        #                    |
                        #                    2
                        #                    |`\
                        #                    |  `\
                        #                    |    `\
                        #                    |      `\
                        #                    |        `\
                        #                    0----------1 --> x

                        Triangles += [int(data[3]), int(int(data[-3]) - 1), int(int(data[-2]) - 1),
                                      int(int(data[-1]) - 1)]

        Triangles = np.array(Triangles).reshape(-1, 4)
        Lines = np.array(Lines).reshape(-1, 3)

        return Verts, Lines, Triangles

    def mesh(self, Verts, Lines, Triangles):
        """Prepare mesh from Verts, Lines and Triangles.

        :param Verts: List of Vertices
        :param Lines: List of Lines
        :param Triangles: List of Triangles
        :return: Verts, Triangles, K, K_Omega, J, J_Omega
        """
        # Sortiere Triangles so, das die Omega-Dreieck am Anfang des Array liegen --------------------------------------
        Triangles = Triangles[Triangles[:, 0].argsort()]

        # Sortiere die Verts, sodass die Indizes der Nodes in Omega am Anfang des Arrays Verts liegen ------------------
        Verts = Verts[:, :2]  # Wir machen 2D, deshalb ist eine Spalte hier unnütz.
        # T heißt Triangle, dx index
        Tdx_Omega = np.where(Triangles[:, 0] == 1)
        # V heißt Vertex, is bedeutet er nimmt die Kategorialen Werte 0,1,2 an.
        Vis_inOmega = np.array([2] * len(Verts), dtype=np.int)

        # Wähle die Indizes heraus, die an Dreiecken in Omega.
        Vdx_inOmega = np.unique(Triangles[Tdx_Omega][1:].flatten())
        Vis_inOmega[Vdx_inOmega] = 0  # Sie werden auf 2 gesetzt.
        Vdx_Boundary = np.unique(Lines[np.where(Lines[:, 0] == 9)][:, 1:])
        Vis_inOmega[Vdx_Boundary] = 1  # Die Punkte auf dem Rand allerdings werden auf 1 gesetzt.

        piVdx_argsort = np.argsort(Vis_inOmega, kind="mergesort")  # Permutation der der Vertex indizes

        # Auf Triangles und Lines müssen wir die inverse Permutation anwenden.
        # Der Code wäre mit np.argsort kurz und für Node-Zahl unter 1000 auch schnell, allerdings ist
        # sortieren nicht in der richtigen Effizienzklasse. (Eigentlich muss ja nur eine Matrix transponiert werden)
        # siehe https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy
        def invert_permutation(p):
            """
            The function inverts a given permutation.
            :param p: nd.array, shape (m,) The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
            :return: nd.array, shape (m,) Returns an array s, where s[i] gives the index of i in p.
            """
            s = np.empty(p.size, p.dtype)
            s[p] = np.arange(p.size)
            return s

        piVdx_invargsort = invert_permutation(piVdx_argsort)
        piVdx = lambda dx: piVdx_invargsort[dx]  # Permutation definieren

        # Wende die Permutation auf Verts, Lines und Triangles an
        Verts = Verts[piVdx_argsort]
        Triangles[:, 1:] = piVdx(Triangles[:, 1:])
        Lines[:, 1:] = piVdx(Lines[:, 1:])

        ## TEST PLOT ###
        # plt.scatter(Verts.T[0], Verts.T[1])
        # plt.scatter(Verts.T[0, :K_Omega], Verts.T[1, :K_Omega])
        # plt.show()

        # Setze J_Omega und J
        # Das ist die Anzahl der Dreiecke. Diese Zahlen sind für die Schleifendurchläufe wichtig.
        J_Omega = np.sum(Triangles[:, 0] == 1)
        J = len(Triangles)

        ## Setze L_Omega und L
        ## Das ist die Anzahl der finiten Elemente (in Omega und insgesamt).
        ## Diese Zahlen dienen als Dimensionen für die diskreten Matrizen und Vektoren.
        L_Omega = np.sum(Vis_inOmega == 0)
        # L_dOmega = np.sum(Vis_inOmega == 1)
        L = len(Verts)
        # Im Falle von "CG" gilt K=L, K_Omega==L_Omega

        ## TEST PLOT ###
        # V = Verts[Triangles[:J_Omega, 1:]]
        # plt.scatter(Verts.T[0], Verts.T[1])
        # for v in V:
        #    plt.scatter(v.T[0], v.T[1], c="r")
        # plt.show()
        return Verts, Triangles, J, J_Omega, L, L_Omega

    def basis(self, ansatz):

        if ansatz == "CG":
            ## Setze K_Omega und K
            ## Das ist die Anzahl der finiten Elemente (in Omega und insgesamt).
            ## Diese Zahlen dienen als Dimensionen für die diskreten Matrizen und Vektoren.
            K_Omega = self.L_Omega
            #K_dOmega = np.sum(Vis_inOmega == 1)
            K = self.L

        elif ansatz == "DG":
            # Im Falle der DG-Methode is die Anzahl der Basisfunktionen 3-Mal die Anzahl der Dreiecke.
            K_Omega = self.J_Omega*3
            #K_dOmega = np.sum(Vis_inOmega == 1)
            K = self.J*3

        else:
            raise ValueError("in clsFEM.basis(). No valid method (str) chosen.")

        Adx = getattr(self, "Adx" + ansatz)
        Adx_inOmega = getattr(self, "Adx" + ansatz + "_inOmega")

        return K, K_Omega, Adx, Adx_inOmega

    def __getitem__(self, Tdx):
        #Adx = self.T[Tdx]
        #E = self.V[Adx]
        #return clsTriangle(E)
        return self.Triangles[Tdx]

    def AdxCG_inOmega(self, Tdx):
        """
        Returns the indices of the nodes of Triangle with index Tdx
        as index w.r.t the triangle (dx_inOmega) and as index w.r.t to
        the array Verts (Adx) for the CG-Basis.

        :param Tdx:
        :return: dx_inOmega, nd.array, int, shape (3,) The indices of the nodes w.r.t T.E which lie in Omega.
        :return: Adx, nd.array, int, shape (3,) The indices of the nodes w.r.t Verts which lie in Omega.
        """
        Adx = self.T[Tdx]
        # The following replaces np.where (see https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
        dx_inOmega = np.flatnonzero(Adx < self.L_Omega)
        Adx = Adx[dx_inOmega]
        return dx_inOmega, Adx

    def AdxCG(self, Tdx):
        """
        Returns the indices of the nodes of Triangle with index Tdx as index w.r.t to
        the array Verts (Adx) for the CG-Basis.

        :param Tdx:
        :return: Adx, nd.array, int, shape (3,) The indices of the nodes w.r.t Verts.
        """
        Adx = self.T[Tdx]
        return Adx

    def AdxDG_inOmega(self, Tdx):
        """
        Returns the indices of the nodes of Triangle with index Tdx
        as index w.r.t the triangle (dx_inOmega) and as index w.r.t to
        the array Verts (Adx) for the DG-Basis.

        :param Tdx:
        :return: dx_inOmega, nd.array, int, shape (3,) The indices of the nodes w.r.t T.E which lie in Omega.
        :return: Adx, nd.array, int, shape (3,) The indices of the nodes w.r.t Verts which lie in Omega.
        """
        # The following replaces np.where (see https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
        if Tdx < self.J_Omega:
            dx_inOmega = np.ones((3,), dtype=int)
            Adx = Tdx * 3 + np.arange(0, 3, dtype=int)
        else:
            dx_inOmega = np.zeros((3,), dtype=int)
            Adx = []
        return dx_inOmega, Adx

    def AdxDG(self, Tdx):
        """
        Returns the indices of the nodes of Triangle with index Tdx as index w.r.t to
        the array Verts (Adx) for the DG-Basis.

        :param Tdx:
        :return: Adx, nd.array, int, shape (3,) The indices of the nodes w.r.t Verts.
        """
        Adx = Tdx * 3 + np.arange(0, 3, dtype=int)
        return Adx

    def get_neighbor(self, Tdx):
        """
        Find neighbour for index Tdx.

        :param Tdx:
        :return:
        """
        T = self.T[Tdx]
        sameVert1 = np.sum(np.array(T[0] == self.T, dtype=int), axis=1)
        sameVert2 = np.sum(np.array(T[1] == self.T, dtype=int), axis=1)
        sameVert3 = np.sum(np.array(T[2] == self.T, dtype=int), axis=1)

        idx = np.where(sameVert1 + sameVert2 + sameVert3 >= 2)[0]

        #idx = np.unique(np.concatenate((w1, w2, w3)))
        #idx = idx[np.where(Tdx != idx)]
        # verts = Verts[Triangles[idx, 1:]]
        return idx

    def neighbor(self, Tdx):
        """
        Return list of indices of neighbours of Tdx.

        :param Tdx:
        :return:
        """
        return self.Neighbours[Tdx]

    def plot(self, Tdx, is_plotmsh=False, pdfname="meshplot", delta=None, title="", refPoints=None):
        """
        Plot triangle with index Tdx.

          *  Link to matplotlib markers: https://matplotlib.org/3.1.0/api/markers_api.html
          *  Link to plt.scatter snippet: https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size

        :param Tdx: int or list of int, Index of Triangle or list of indices of Triangle.
        :param is_plotmsh: bool, default=False Switch for surrounding FEM-Mesh.
        :param pdfname: str Name of output pdf.
        :param delta: optional, Interaction radius. A :math:`\ell_2`-circle will be drawn to show its size.
        :param title: optional, Title of the plot.
        :param refPoints: optional, Reference points for integration. If set to P the physical Points will be plotted
            into the Triangles given by Tdx.
        :return: None
        """

        if type(Tdx) is int:
            Tdx = [Tdx]
        pp = PdfPages(pdfname+".pdf")

        fig, ax = plt.subplots()
        plt.gca().set_aspect('equal')
        plt.title(title)
        if is_plotmsh:
            plt.triplot(self.V[:, 0], self.V[:, 1], self.T, lw=0.5, color='blue', alpha=.7)

        aTdx = Tdx[0]
        aT = self[Tdx[0]]
        # Some extras for the central Triangle

        if delta is not None:
            circle = plt.Circle(aT.baryCenter(), delta, color='b', fill=False, lw=0.5)
            ax.add_artist(circle)
        plt.scatter(aT.baryCenter()[0], aT.baryCenter()[1], s=1, c="black")

        for tdx in Tdx:
            T = self[tdx]
            # Here we need to choose AdxCG_inOmega irrespective of the chosen ansatz
            # as we are interested in the index of the Verticies of some Triangle, which is
            # exactly what AdxCG provides.
            dx_inOmega, Vdx = self.AdxCG_inOmega(tdx)
            E_O = self.V[Vdx]
            if refPoints is not None and delta is not None:
                P = aT.toPhys(refPoints)
                Pdx_inNbhd = xnotinNbhd(refPoints, aT, T, delta)
                plt.scatter(P[0, Pdx_inNbhd], P[1, Pdx_inNbhd], s=.1, c="black")

            #plt.scatter(T.E[:, 0], T.E[:, 1], s=marker_size, c="black", marker="o", label="E")
            plt.fill(T.E[:, 0], T.E[:, 1], "r", alpha=.3)#, s=marker_size, c="black", marker="o", label="E")
            #plt.scatter(E_O[:, 0], E_O[:, 1], s=marker_size, c="r", marker="X", label="E in Omega (Adx)")
            #plt.scatter(T.E[dx_inOmega, 0], T.E[dx_inOmega, 1], s=marker_size, c="w", marker="+",
            #            label="E in Omega (dx_inOmega)")
            #plt.legend()


        plt.savefig(pp, format='pdf')
        plt.close()

        pp.close()
        return

class clsTriangle:
    """**Triangle Class**

    :ivar E: nd.array, real, shape (3,2). Physical nodes of Triangle.
    """
    def __init__(self, E):
        self.E = E
        a, b, c = self.E
        self.M_ = np.array([b - a, c - a]).T
        self.Minv_ = None
        self.a_ = a[:, np.newaxis]
        self.baryCenter_ = None
        self.absDet_ = None
        self.toPhys_ = None

    def baryCenter(self):
        if self.baryCenter_ is None:
            self.baryCenter_ = np.sum(self.E, axis=0)/3
            return self.baryCenter_
        else:
            return self.baryCenter_

    def absDet(self):
        if self.absDet_ is None:
            M = self.M_
            self.absDet_ = np.abs(M[0, 0] * M[1, 1] - M[1, 0] * M[0, 1])
            return self.absDet_
        else:
            return self.absDet_

    def toPhys(self, P):
        """Push reference points P to physical domain.

        :param P: nd.array, real, shape (2, n). Reference points, e.g. quadrature points of the reference element.
        :return: nd.array, real, shape (2, n). Physical points.
        """
        self.toPhys_ = self.M_ @ P + self.a_
        return self.toPhys_

    def toRef(self, X):
        """Pull physical points to reference domain.

        :param X: nd.array, real, shape (2, n). Physical points.
        :return: nd.array, real, shape (2, n). Reference points, e.g. quadrature points of the reference element.
        """
        self.Minv_ = np.linalg.inv(self.M_)
        return  self.Minv_ @ (X - self.a_)

    def __eq__(self, other):
        return (self.E == other.E).all()

class clsInt:
    """**Integrator Class**

    Contains the formulas, quadrature rule and functions fPhys and kerPhys.
    The rule P, weights and delta are handed over when the object is constructed.

    :param P: nd.array, real, shape (2,n). Quadtrature points for the outer, and the inner integral.
    :param weights: nd.array, real, shape (n,). Weights corresponding to the quadrature rule.
    :param delta: real. Interaction horizon.
    :param outerIntMethod: str. Name of integration method for the outer integral. Options are *outerInt_full* (default), *outerInt_retriangulate*.
    :param innerIntMethod: str. Name of integration method for the inner integral  Options are *innerInt_bary*, *innerInt_retriangulate* (default).
    """

    def __init__(self, P, weights, delta, outerIntMethod="outerInt_full", innerIntMethod="innerInt_retriangulate"):

        self.delta = delta
        # Changing the order of psi, does not really have an effect in this very simple case!
        psi0 = 1 - P[0, :] - P[1, :]
        psi1 = P[0, :]
        psi2 = P[1, :]
        psi = np.array([psi0, psi1, psi2])
        self.psi = psi
        self.P = P
        self.weights = weights
        #self.outerInt = getattr(self, outerIntMethod)
        #self.innerInt = getattr(self, innerIntMethod)

    def f(self, aBdx_O, aT):
        """
        Assembles the right side f.

        :param aBdx_O: tupel of int, i=0,1,2. Index of reference basis functions which lie in Omega.
        :param aT: Triangle. Triangle to integrate over.
        :return: Integral.
        """
        cdef:
            int i,k,a
            int nB=aBdx_O.shape[0]
            double [:,:] P = self.P
            int nP=P.shape[1]
            double [:,:] psi = self.psi
            double [:] dx = self.weights
            double x[2]
            py_f = np.zeros((3,))
            double [:] cy_f = py_f

        for k in range(nB):
            a = aBdx_O[k]
            for i in range(nP):
                cy_toPhys(aT.E, P[:, i], x)
                cy_f[k] += psi[a, i] * c_fPhys(&x[0] ) * aT.absDet() * dx[i]
        return py_f[:nB]#(self.psi[aBdx_O] * self.fPhys(aT.toPhys(self.P))) @ dx * aT.absDet()

    def fPhys(self, x):
        return 1

    def A(self, double [:,:] aTE, double [:,:] bTE, is_allInteract=True):
        """Compute the local and nonlocal terms of the integral.

        :param py_a: int. Index of vertex to find the correct reference basis function.
        :param py_b: int. Index of vertex to find the correct reference basis function.
        :param aT: Triangle, Triangle a.
        :param bT: Triangle, Triangle b.
        :param is_allInteract: bool. True if all points in aT interact with all points in bT.
        :return  termLocal, termNonloc:
        """
        cdef:
            int i=0, j=0, a, b
            double kerd, innerIntLocal, innerIntNonloc
            double sqdelta = self.delta**2
            double [:] dy = self.weights
            double [:] dx = dy
            double outerInt[2]
            double [:,:] psi = self.psi
            double [:,:] P = self.P
            double aTdet = cy_absDet(aTE), bTdet = cy_absDet(bTE)
            int nP=P.shape[1]
            termLocal = np.zeros((3,3))
            termNonloc = np.zeros((3,3))
            double [:,:] cy_termLocal = termLocal
            double [:,:] cy_termNonloc = termNonloc

        if is_allInteract:
            for a in range(3):
                for b in range(3):
                    for i in range(nP):
                        innerIntLocal=0
                        innerIntNonloc=0
                        for j in range(nP):
                            innerIntLocal += c_kernelPhys(&(P[:, i])[0], &(P[:, j])[0], sqdelta) * dy[j]
                            innerIntNonloc += c_kernelPhys(&(P[:, i])[0], &(P[:, j])[0], sqdelta) * dy[j] * psi[b, j]
                        cy_termLocal[a][b] += psi[a, i] * psi[b, i] * innerIntLocal * dx[i] * aTdet*bTdet
                        cy_termNonloc[a][b] += psi[a,i] * innerIntNonloc * dx[i] * aTdet*bTdet
            return termLocal, termNonloc
        else:
            cy_outerInt_full(aTdet, bTdet, aTE, bTE, P, nP, dx, dy, psi, sqdelta, cy_termLocal, cy_termNonloc)

            return termLocal, termNonloc

cdef void cy_outerInt_full(double aTdet, double bTdet, double[:,:] aTE, double [:,:] bTE,
                           double [:,:] P, int nP, double [:] dx, double [:] dy, double [:,:] psi,
                           double sqdelta,
                           double [:, :] cy_termLocal, double [:,:] cy_termNonloc):
    cdef:
        int i=0, k=0, Rdx=0
        double x[2], innerLocal=0
        double innerNonloc[3]
        double [:,:] RT = np.zeros((9*3, 2))

    for k in range(nP):
        cy_toPhys(aTE, P[:,k], x)
        Rdx = cy_retriangulate(x, bTE, sqdelta, RT)
        cy_innerInt_retriangulate(x, bTdet, bTE, P, nP, dy, sqdelta, Rdx, RT, &innerLocal, innerNonloc)
        for b in range(3):
            for a in range(3):
                cy_termLocal[a][b] += aTdet * psi[a][k] * psi[b][k] * dx[k] * innerLocal #innerLocal
                cy_termNonloc[a][b] += aTdet * psi[a][k] * dx[k] * innerNonloc[b] #innerNonloc

cdef cy_innerInt_retriangulate(double [:] x,
                                    double Tdet, double [:,:] TE,
                                    double [:,:] P, int nP, double [:] dy,
                                    double sqdelta,
                                    int Rdx, double[:,:] RT,
                                    double * innerLocal,
                                    double [:] innerNonloc):
    cdef:
        int nRT = 0, i=0, k=0, rTdx=0, b=0
        double psi_rp=0, ker=0, rTdet=0
        double [:] p
        double ry[2]
        double rp[2]

    innerLocal[0] = 0
    for b in range(3):
        innerNonloc[b] = 0
    if Rdx == 0:
        return

    for rTdx in range(Rdx):
        for i in range(nP):
            cy_toPhys(RT[rTdx*3:rTdx*3+3, :], P[:, i], ry)
            cy_toRef(TE, ry, rp)
            ker = c_kernelPhys(&x[0], &ry[0], sqdelta)
            rTdet = cy_absDet(RT[rTdx*3:rTdx*3+3, :])
            innerLocal[0] += (ker * dy[i]) * rTdet # Local Term
            for b in range(3):
                psi_rp = cy_evalPsi(rp, b)
                innerNonloc[b] += (psi_rp * ker * dy[i]) * rTdet # Nonlocal Term

cdef double cy_evalPsi(double * p, int psidx):
    if psidx == 0:
        return 1 - p[0] - p[1]
    elif psidx == 1:
        return p[0]
    elif psidx == 2:
        return p[1]
    else:
        raise ValueError("in cy_evalPsi. Invalid psi index.")

cdef int cy_retriangulate(double [:] x_center, double [:,:] TE, double sqdelta, double [:,:] out_RE):
        """ Retriangulates a given triangle.

        :param x_center: nd.array, real, shape (2,). Center of normball, e.g. pyhsical quadtrature point.
        :param T: clsTriangle. Triangle to be retriangulated.
        :return: list of clsTriangle.
        """
        #py_R = np.zeros((9,2))

        # Python Objects to allocate space which is accessible from Python.
        #py_E = np.zeros((3,2))

        # Please hardcode via some modulo operation in future!!
        #py_edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32)
        # see eddx0, edgdx1

        # C Variables and Arrays.
        cdef:
            int i=0, k=0, edgdx0=0, edgdx1=0
            double v=0, lam1=0, lam2=0, t=0, term1=0, term2=0
            double c_p[2]
            double c_q[2]
            double c_a[2]
            double c_b[2]
            double c_y1[2]
            double c_y2[2]
            # An upper bound for the number of intersections between a circle and a triangle is 9
            # Hence we can hardcode how much space needs to bee allocated
            double c_R[9][2]
            # Hence 9*3 is an upper bound to encode all resulting triangles
            #double c_RE[9*3][2]
        # Memory Views.
        #cdef:
            #int [:,:] edges = py_edges
            #double [:] x_center=py_x_center
            #double [:, :] RE=c_RE
        Rdx=0
        for i in range(9):
            for k in range(2):
                c_R[i][k] = 0.0

        for k in range(3):
            edgdx0 = k
            edgdx1 = (k+1) % 3

            for i in range(2):
                c_p[i] = TE[edgdx0, i]
                c_q[i] = TE[edgdx1, i]
                c_a[i] = c_q[i] - x_center[i]
                c_b[i] = c_p[i] - c_q[i]
            v = pow(c_vecdot(&c_a[0], &c_b[0], 2),2) - (c_vecdot(&c_a[0], &c_a[0], 2) - sqdelta)*c_vecdot(&c_b[0],&c_b[0], 2)

            if v >= 0:
                term1 = -c_vecdot(&c_a[0], &c_b[0], 2)/c_vecdot(&c_b[0], &c_b[0], 2)
                term2 = sqrt(v)/c_vecdot(&c_b[0], &c_b[0], 2)
                lam1 = term1 + term2
                lam2 = term1 - term2
                for i in range(2):
                    c_y1[i] = lam1*(c_p[i]-c_q[i]) + c_q[i]
                    c_y2[i] = lam2*(c_p[i]-c_q[i]) + c_q[i]

                if c_vecdist_sql2(&c_p[0], &x_center[0], 2) <= sqdelta:
                    for i in range(2):
                        c_R[Rdx][i] = c_p[i]
                    Rdx += 1
                if 0 <= lam1 <= 1:
                    for i in range(2):
                        c_R[Rdx][i] = c_y1[i]
                    Rdx += 1
                if (0 <= lam2 <= 1) and (scaldist_sql2(lam1, lam2) >= 1e-9):
                    for i in range(2):
                        c_R[Rdx][i] = c_y2[i]
                    Rdx += 1
            else:
                if c_vecdist_sql2(c_p, &x_center[0], 2)  <= sqdelta:
                    for i in range(2):
                        c_R[Rdx][i] = c_p[i]
                    Rdx += 1
        # Construct List of Triangles from intersection points
        if Rdx < 3:
            # In this case the content of the array out_RE will not be touched.
            return 0
        else:
            for k in range(Rdx - 2):
                for i in range(2):
                    out_RE[3*k + 0, i] = c_R[0][i]
                    out_RE[3*k + 1, i] = c_R[k+1][i]
                    out_RE[3*k + 2, i] = c_R[k+2][i]
            # Excessing the bound out_Rdx will not lead to an error but simply to corrupted data!

            return Rdx - 2 # So that, it acutally contains the number of triangles in the retriangulation
# Define Right side f
cdef double c_fPhys(double * x):
        """ Right side of the equation.

        :param x: nd.array, real, shape (2,). Physical point in the 2D plane
        :return: real
        """
        # f = 1
        return 1.0

cdef void cy_toPhys(double [:,:] E, double [:] p, double [:] out_x):
    cdef:
        int i=0
    for i in range(2):
        out_x[i] = (E[1][i] - E[0][i])*p[0] + (E[2][i] - E[0][i])*p[1] + E[0][i]

cdef double c_kernelPhys(double * x, double * y, double sqdelta):
    return 4 / (pi * pow(sqdelta, 2))

cdef double c_vecdist_sql2(double * x, double * y, int length):
    """
    Computes squared l2 distance
    
    :param x: * double array
    :param y: * dobule array
    :param length: int length of vectors
    :return: dobule
    """
    cdef:
        double r=0
        int i=0
    for i in range(length):
        r += pow((x[i] - y[i]), 2)
    return r

cdef double scaldist_sql2(double x, double y):
    """
    Computes squared l2 distance
    
    :param x: double array
    :param y: double array
    :return: double
    """
    return pow((x-y), 2)

cdef double c_vecdot(double * x, double * y, int length):
    """
    Computes scalar product of two vectors.
    
    :param x: * double array
    :param y: * double array
    :param lengt: int length of vectors
    :return: double
    """
    cdef:
        double r=0
        int i=0
    for i in range(length):
        r += x[i]*y[i]
    return r

cdef void cy_solve2x2(double [:,:] A, double [:] b, double [:] x):
    cdef:
        int i=0, dx0 = 0, dx1 = 1
        double l=0, u=0

    # Column Pivot Strategy
    if abs(A[0,0]) < abs(A[1,0]):
        dx0 = 1
        dx1 = 0

    # Check invertibility
    if A[dx0,0] == 0:
        raise LinAlgError("in cy_solve2x2. Matrix not invertible.")

    # LU Decomposition
    l = A[dx1,0]/A[dx0,0]
    u = A[dx1,1] - l*A[dx0,1]

    # Check invertibility
    if u == 0:
        raise LinAlgError("in cy_solve2x2. Matrix not invertible.")

    # LU Solve
    x[1] = (b[dx1] - l*b[dx0])/u
    x[0] = (b[dx0] - A[dx0,1]*x[1])/A[dx0,0]
    return

cdef void cy_toRef(double [:,:] E, double [:] phys_x, double [:] ref_p):
    cdef:
        double M[2][2]
        double b[2]
        int i=0, j=0

    for i in range(2):
        M[i][0] = E[1][i] - E[0][i]
        M[i][1] = E[2][i] - E[0][i]
        b[i] = phys_x[i] - E[0][i]

    cy_solve2x2(M, b, ref_p)
    return

cdef double cy_absDet(double [:,:] E):
    cdef:
        double M[2][2]
    for i in range(2):
        M[i][0] = E[1][i] - E[0][i]
        M[i][1] = E[2][i] - E[0][i]
    return abs(M[0][0]*M[1][1] - M[0][1]*M[1][0])

cdef double baryCenter(double [:,:] E, double [:] bary):
    cdef:
        int i
    bary[0] = 0
    bary[1] = 0
    for i in range(3):
        bary[0] += E[i, 0]
        bary[1] += E[i, 1]
    bary[0] = bary[0]/3
    bary[1] = bary[1]/3

def inNbhd(double [:,:] aTE, double [:,:] bTE, double sqdelta, double [:] M):
    """
    Check whether two triangles interact w.r.t :math:`L_{2}`-ball of size delta.
    Returns an array of boolen values. Compares the barycenter of bT with the vertices of Ea.

    :param aT: clsTriangle, Triangle a
    :param bT: clsTriangle, Triangle b
    :param delta: Size of L-2 ball
    :param v: Verbose mode.
    :return: ndarray, bool, shape (3,) Entry i is True if Ea[i] and the barycenter of Eb interact.
    """

    cdef:
        int i=0
        double bary[2]
    baryCenter(bTE, bary)

    for i in range(3):
        M[i] = (c_vecdist_sql2(&(aTE[i,:])[0], &bary[0], 2) <= sqdelta)