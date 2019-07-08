#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class clsMesh:
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
    def __init__(self, mshfile):
        """Constructor

        Executes read_mesh and prepare.
        """
        args = self.prepare(*self.read_mesh(mshfile))
        # args = Verts, Triangles, K, K_Omega, J, J_Omega
        self.V = args[0]
        self.T = args[1][:, 1:]
        self.K = args[2]
        self.K_Omega = args[3]
        self.J = args[4]
        self.J_Omega = args[5]

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

    def prepare(self, Verts, Lines, Triangles):
        """Prepare mesh from Verts, Lines and Triangles.

        :param Verts: List of Vertices
        :param Lines: List of Lines
        :param Triangles: List of Triangles
        :return: Verts, Triangles, K, K_Omega, J, J_Omega
        """
        # Sortiere Triangles so, das die Omega-Dreieck am Anfang des Array liegen --------------------------------------
        Triangles = Triangles[Triangles[:, 0].argsort()]

        # Sortiere die Verts, sodass die Indizes der Nodes in Omega am Anfang des Arrays Verts liegen ---------------------------------
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

        # Setze K_Omega und K
        # Das ist die Anzahl der finiten Elemente (in Omega und insgesamt).
        # Diese Zahlen dienen als Dimensionen für die diskreten Matrizen und Vektoren.
        K_Omega = np.sum(Vis_inOmega == 0)
        K_dOmega = np.sum(Vis_inOmega == 1)
        K = len(Verts)

        ## TEST PLOT ###
        # plt.scatter(Verts.T[0], Verts.T[1])
        # plt.scatter(Verts.T[0, :K_Omega], Verts.T[1, :K_Omega])
        # plt.show()

        # Setze J_Omega und J
        # Das ist die Anzahl der Dreiecke. Diese Zahlen sind für die Schleifendurchläufe wichtig.
        J_Omega = np.sum(Triangles[:, 0] == 1)
        J = len(Triangles)

        ## TEST PLOT ###
        # V = Verts[Triangles[:J_Omega, 1:]]
        # plt.scatter(Verts.T[0], Verts.T[1])
        # for v in V:
        #    plt.scatter(v.T[0], v.T[1], c="r")
        # plt.show()
        return Verts, Triangles, K, K_Omega, J, J_Omega

    def __getitem__(self, Tdx):
        #Vdx = self.T[Tdx]
        #E = self.V[Vdx]
        #return clsTriangle(E)
        return self.Triangles[Tdx]

    def Vdx_inOmega(self, Tdx):
        """
        Returns the indices of the nodes of Triangle with index Tdx
        as index w.r.t the triangle (dx_inOmega) and as index w.r.t to
        the array Verts (Vdx)

        :param Tdx:
        :return: dx_inOmega, nd.array, int, shape (3,) The indices of the nodes w.r.t T.E which lie in Omega.
        :return: Vdx, nd.array, int, shape (3,) The indices of the nodes w.r.t Verts which lie in Omega.
        """
        Vdx = self.T[Tdx]
        # The following replaces np.where (see https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
        dx_inOmega = np.flatnonzero(Vdx < self.K_Omega)
        Vdx = Vdx[dx_inOmega]
        return dx_inOmega, Vdx

    def Vdx(self, Tdx):
        """
        Returns the indices of the nodes of Triangle with index Tdx as index w.r.t to
        the array Verts (Vdx)

        :param Tdx:
        :return: Vdx, nd.array, int, shape (3,) The indices of the nodes w.r.t Verts.
        """
        Vdx = self.T[Tdx]
        return Vdx

    def get_neighbor(self, Tdx):
        """
        Find neighbour for index Tdx.

        :param Tdx:
        :return:
        """
        T = self.T[Tdx]
        w1, _ = np.where(T[0] == self.T)
        w2, _ = np.where(T[1] == self.T)
        w3, _ = np.where(T[2] == self.T)

        idx = np.unique(np.concatenate((w1, w2, w3)))
        idx = idx[np.where(Tdx != idx)]
        # verts = Verts[Triangles[idx, 1:]]
        return idx

    def neighbor(self, Tdx):
        """
        Return list of indices of neighbours of Tdx.

        :param Tdx:
        :return:
        """
        return self.Neighbours[Tdx]

    def plot(self, Tdx, is_plotmsh=False, pdfname="meshplot", delta=None):
        """
        Plot triangle with index Tdx.

          *  Link to matplotlib markers: https://matplotlib.org/3.1.0/api/markers_api.html
          *  Link to plt.scatter snippet: https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size

        :param Tdx: int or list of int, Index of Triangle or list of indices of Triangle.
        :param is_plotmsh: bool, default=False Switch for surrounding FEM-Mesh.
        :param pdfname: str Name of output pdf.
        :param delta: optional, Interaction radius. A :math:`\ell_2`-circle will be drawn to show its size.
        :return: None
        """

        pp = PdfPages(pdfname+".pdf")

        fig, ax = plt.subplots()
        plt.gca().set_aspect('equal')
        if is_plotmsh:
            plt.triplot(self.V[:, 0], self.V[:, 1], self.T, lw=0.5, color='blue', alpha=.7)

        if len(Tdx) == 1:
            Tdx = [Tdx]

        for tdx in Tdx:
            T = self[tdx]
            dx_inOmega, Vdx = self.Vdx_inOmega(tdx)
            E_O = self.V[Vdx]

            if delta is not None:
                circle = plt.Circle(T.baryCenter(), delta, color='b', fill=False, lw=0.5)
                ax.add_artist(circle)
            plt.scatter(T.E[:, 0], T.E[:, 1], s=50, c="b", marker="o", label="E")
            plt.scatter(E_O[:, 0], E_O[:, 1], s=50, c="r", marker="X", label="E in Omega (Vdx)")
            plt.scatter(T.E[dx_inOmega, 0], T.E[dx_inOmega, 1], s=50, c="w", marker="+",
                        label="E in Omega (dx_inOmega)")
            #plt.legend()
        plt.savefig(pp, format='pdf')
        plt.close()

        pp.close()
        return

class clsTriangle:
    """
    Triangle Classe

    :ivar E: nd.array, real, shape (3,2). Physical nodes of Triangle.
    """
    def __init__(self, E):
        self.E = E
        a, b, c = self.E
        self.M_ = np.array([b - a, c - a])
        self.a_ = a.reshape((2, 1))
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
    def __eq__(self, other):
        return (self.E == other.E).all()


class clsInt:
    """**Integrator Class**

    Contains the formulas, quadrature rule and functions fPhys and kerPhys.
    The rule P, weights and delta are handed over when the object is constructed.

    :ivar psi: Values of basis functions for given rule P.
    """
    def __init__(self, P, weights, delta):
        self.delta = delta
        psi0 = 1 - P[0, :] - P[1, :]
        psi1 = P[0, :]
        psi2 = P[1, :]
        psi = np.array([psi0, psi1, psi2])
        self.psi = psi
        self.P = P
        self.weights = weights
    def A(self, a, b, aT, bT, is_allInteract=True):
        """Compute the local and nonlocal terms of the integral.

        :param a: int. Index of vertex to find the correct reference basis function.
        :param b: int. Index of vertex to find the correct reference basis function.
        :param aT: Triangle, Triangle a.
        :param bT: Triangle, Triangle b.
        :param is_allInteract: bool. True if all points in aT interact with all points in bT.
        :return:
        """
        if is_allInteract:
            # P, weights and psi are just views. The data is not copied. See
            # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
            P = self.P
            weights = self.weights
            psi = self.psi

        else:
            dx_sInteract = []
            # The following line increases the computation time by x2.
            dx_sInteract = xinNbhd(self.P, aT, bT, self.delta)
            # Advanced indexing!
            # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
            # This will copy the data to P and weights every time!
            P = self.P[:, dx_sInteract]
            weights = self.weights[dx_sInteract]
            psi = self.psi[:, dx_sInteract]

        kerd = self.kernelPhys(P, aT, bT)
        termLocal = aT.absDet() * bT.absDet() * (psi[a] * psi[b] * (kerd @ weights)) @ weights
        termNonloc = aT.absDet() * bT.absDet() * psi[a] * ((psi[b] * kerd) @ weights) @ weights
        return termLocal, termNonloc

    def f(self, aBdx_O, aT):
        """
        Assembles the right side f.

        :param aBdx_O: tupel of int, i=0,1,2. Index of reference basis functions which lie in Omega.
        :param aT: Triangle. Triangle to intgrate over.
        :return: Integral.
        """
        return (self.psi[aBdx_O] * self.fPhys(aT.toPhys(self.P))) @ self.weights * aT.absDet()

    # Define Right side f
    def fPhys(self, x):
        """ Right side of the equation.

        :param x: nd.array, real, shape (2,). Physical point in the 2D plane
        :return: real
        """
        # f = 1
        return 1

    def kernelPhys(self, P, Tx, Ty):
        """ Constant integration kernel.

        :param P: ndarray, real, shape (2, n). Reference points for integration.
        :param Tx: Triangle. Triangle of x-Component.
        :param Ty: Triangle. Triangle of y-Component.
        :return: real. Evaluates the kernel on the full grid.
        """

        # $\gamma(x,y) = 4 / (pi * \delta**4)$
        # Wir erwarten $u(x) = 1/4 (1 - ||x||^2)$

        n_P = P.shape[1]
        return 4 / (np.pi * self.delta ** 4) * np.ones((n_P, n_P))