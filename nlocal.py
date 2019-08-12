#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nbhd import xnotinNbhd


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
            dx_inOmega, Vdx = self.Vdx_inOmega(tdx)
            E_O = self.V[Vdx]
            if refPoints is not None and delta is not None:
                P = aT.toPhys(refPoints)
                Pdx_inNbhd = xnotinNbhd(refPoints, aT, T, delta)
                plt.scatter(P[0, Pdx_inNbhd], P[1, Pdx_inNbhd], s=.1, c="black")

            #plt.scatter(T.E[:, 0], T.E[:, 1], s=marker_size, c="black", marker="o", label="E")
            plt.fill(T.E[:, 0], T.E[:, 1], "r", alpha=.3)#, s=marker_size, c="black", marker="o", label="E")
            #plt.scatter(E_O[:, 0], E_O[:, 1], s=marker_size, c="r", marker="X", label="E in Omega (Vdx)")
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

    :ivar psi: Values of basis functions for given rule P.
    """
    def __init__(self, P, weights, delta, outerIntMethod="outerInt_full", innerIntMethod="innerInt_retriangulate"):
        """
        Constructor.

        :param P: nd.array, real, shape (2,n). Quadtrature points for the outer, and the inner integral.
        :param weights: nd.array, real, shape (n,). Weights corresponding to the quadrature rule.
        :param delta: real. Interaction horizon.
        :param outerIntMethod: str. Name of integration method for the outer integral. Options are *outerInt_full* (default), *outerInt_retriangulate*.
        :param innerIntMethod: str. Name of integration method for the inner integral  Options are *innerInt_bary*, *innerInt_retriangulate* (default).
        """
        self.delta = delta
        # Changing the order of psi, does not really have an effect in this very simple case!
        psi0 = 1 - P[0, :] - P[1, :]
        psi1 = P[0, :]
        psi2 = P[1, :]
        psi = np.array([psi0, psi1, psi2])
        self.psi = psi
        self.P = P
        self.weights = weights
        self.outerInt = getattr(self, outerIntMethod)
        self.innerInt = getattr(self, innerIntMethod)


    def A(self, a, b, aT, bT, is_allInteract=True):
        """Compute the local and nonlocal terms of the integral.

        :param a: int. Index of vertex to find the correct reference basis function.
        :param b: int. Index of vertex to find the correct reference basis function.
        :param aT: Triangle, Triangle a.
        :param bT: Triangle, Triangle b.
        :param is_allInteract: bool. True if all points in aT interact with all points in bT.
        :return  termLocal, termNonloc:
        """
        dy = self.weights
        dx = dy
        psi = self.psi
        P = self.P

        if is_allInteract:
            kerd = self.kernelPhys(P)
            termLocal = aT.absDet() * bT.absDet() * (psi[a] * psi[b] * (kerd @ dy)) @ dx
            termNonloc = aT.absDet() * bT.absDet() * psi[a] * (kerd @ (psi[b] * dy)) @ dx
            return termLocal, termNonloc
        else:
            termLocal, termNonloc = self.outerInt(a, b, aT, bT)
            return termLocal, termNonloc

    def innerInt_retriangulate(self, x, T, b):
        """
        Computes the inner integral using retriangulation. As the resulting term will be smooth in x this
        function should work well with outerInt_full()

        :param x: nd.array, real, shape (2,). Point of evaluation.
        :param T: clsTriangle. Triangle over which we want to integrate.
        :param b: int. Index of reference basis function which we want to integrate.
        :return: real. The integral is computed by retriangulating the domain w.r.t. the
        interaction between x and T.
        """
        P = self.P
        dy = self.weights
        innerLocal = 0
        innerNonloc = 0
        RT = self.retriangulate(x, T)
        for rT in RT:
            rX = rT.toPhys(P)
            psi_r = self.evalPsi(T.toRef(rX), b)
            ker = self.xkernelPhys(x[:, np.newaxis], rX)
            innerLocal += (ker @ dy) * rT.absDet()
            innerNonloc += (psi_r * ker @ dy)*rT.absDet()
        return innerLocal, innerNonloc

    def innerInt_bary(self, x, T, b):
        """
        Computes the inner integral using the bary center method. As the resulting term will not be smooth in x this
        function should work well with outerInt_retriangluate()

        :param x: nd.array, real, shape (2,). Point of evaluation.
        :param T: clsTriangle. Triangle over which we want to integrate.
        :param b: int. Index of reference basis function which we want to integrate.
        :return: real. The integrals are 0 if x and T do not interact and :math:`\int_T \gamma' \phi' dy`, :math:`\int_T \gamma  dy` otherwise.
        """
        if np.linalg.norm(x - T.baryCenter()) > self.delta:
            return 0, 0
        else:
            dy = self.weights
            X = T.toPhys(self.P)
            psi = self.psi[b]
            ker = self.xkernelPhys(x[:, np.newaxis], X)
            innerLocal = (ker @ dy) * T.absDet()
            innerNonloc = (psi * ker @ dy)*T.absDet()
            return innerLocal, innerNonloc

    def outerInt_retriangulate(self, a, b, aT, bT):
        """
        Computes the outer integral using retriangulation.

        :param a: int. Index of the outer reference basis function.
        :param b: int. Index of the 'inner' reference basis function.
        :param aT: clsTriangle. Outer Triangle.
        :param bT: clsTriangle. Inner Triangle.
        :return: real. The integral is computed by retriangulating the domain aT w.r.t. the
        interaction between x in aT and bT.
        """
        P = self.P
        dx = self.weights
        psi = self.psi
        termLocal = 0
        termNonloc = 0
        RT = self.retriangulate(bT.baryCenter(), aT)

        for rT in RT:
            rX = rT.toPhys(P)
            psia_r = self.evalPsi(aT.toRef(rX), a)
            psib_r = self.evalPsi(aT.toRef(rX), b)
            for k, x in enumerate(rX.T):
                innerLocal, innerNonloc = self.innerInt(x, bT, b)
                termLocal += rT.absDet() * psia_r[k] * psib_r[k] * dx[k] * innerLocal
                termNonloc += rT.absDet() * psia_r[k] * dx[k] * innerNonloc
        return termLocal, termNonloc

    def outerInt_full(self, a, b, aT, bT):
        """
        Computes the outer integral.

        :param a: int. Index of the outer reference basis function.
        :param b: int. Index of the 'inner' reference basis function.
        :param aT: clsTriangle. Outer Triangle.
        :param bT: clsTriangle. Inner Triangle.
        :return: real. Integral.
        """
        P = self.P
        dx = self.weights
        psi = self.psi
        termLocal = 0
        termNonloc = 0

        for k, p in enumerate(P.T):
            x = aT.toPhys(p[:, np.newaxis])[:, 0]
            innerLocal, innerNonloc = self.innerInt(x, bT, b)
            termLocal += aT.absDet() * psi[a][k] * psi[b][k] * dx[k] * innerLocal
            termNonloc += aT.absDet() * psi[a][k] * dx[k] * innerNonloc
        return termLocal, termNonloc

    def retriangulate(self, x_center, T):
        """ Retriangulates a given triangle.

        :param x_center: nd.array, real, shape (2,). Center of normball, e.g. pyhsical quadtrature point.
        :param T: clsTriangle. Triangle to be retriangulated.
        :return: list of clsTriangle.
        """
        R = []
        edges = np.array([[0, 1], [1, 2], [2, 0]])
        for k in range(3):
            p = T.E[edges[k][0]]
            q = T.E[edges[k][1]]
            a = q - x_center
            b = p - q
            v = (a@b)**2 - (a@a - self.delta**2)*(b@b)

            if v >= 0:
                lam1 = -(a@b)/(b@b) + np.sqrt(v)/(b@b)
                lam2 = -(a@b)/(b@b) - np.sqrt(v)/(b@b)
                #print("lam1 \t", lam1)
                #print("lam2 \t", lam2)
                y1 = lam1*(p-q) + q
                y2 = lam2*(p-q) + q

                if np.sum((T.E[edges[k][0]] - x_center)**2) <= self.delta**2:
                    R.append(T.E[edges[k][0]])
                if 0 <= lam1 <= 1:
                    R.append(y1)
                if 0 <= lam2 <= 1 and np.abs(lam1 - lam2) >= 1e-9:
                    R.append(y2)
            else:
                if np.sum((T.E[edges[k][0]] - x_center)**2) <= self.delta**2:
                    R.append(T.E[edges[k][0]])
        RE = []
        while len(R) >= 3:
            rE = R[0:3]
            RE.append(np.array(rE))
            del R[1]


        return [clsTriangle(E) for E in RE]

    def evalPsi(self, P, psidx):
        """Evaluate basis function for given reference points.

        :param P: nd.array, real, shape (2, n).
        Reference points, e.g. quadrature points of the reference element.
        :param psidx: Index of Basis function.
        :return: nd.array, real, shape (3, n). Values of basis function.
        """
        if psidx == 0:
            return 1 - P[0, :] - P[1, :]
        if psidx == 1:
            return P[0, :]
        if psidx == 2:
            return P[1, :]

    def f(self, aBdx_O, aT):
        """
        Assembles the right side f.

        :param aBdx_O: tupel of int, i=0,1,2. Index of reference basis functions which lie in Omega.
        :param aT: Triangle. Triangle to integrate over.
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

    def kernelPhys(self, P):
        """ Constant integration kernel.

        :param P: ndarray, real, shape (2, n). Reference points for integration.
        :return: real. Evaluates the kernel on the full grid.
        """

        # $\gamma(x,y) = 4 / (pi * \delta**4)$
        # Wir erwarten $u(x) = 1/4 (1 - ||x||^2)$

        n_P = P.shape[1]
        return 4 / (np.pi * self.delta ** 4) * np.ones((n_P, n_P))

    def xkernelPhys(self, x, y):
        """ Constant integration kernel.

        :param x: ndarray, real, shape (2, n). Physical points for evaluation.
        :param y: ndarray, real, shape (2, n). Physical points for evaluation.
        :return: real. Evaluates the kernel.
        """

        # $\gamma(x,y) = 4 / (pi * \delta**4)$
        # Wir erwarten $u(x) = 1/4 (1 - ||x||^2)$
        n_x = x.shape[1]
        n_y = y.shape[1]
        return 4 / (np.pi * self.delta ** 4) * np.ones((n_x, n_y))
