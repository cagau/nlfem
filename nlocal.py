#-*- coding:utf-8 -*-

import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
solvers.options['show_progress'] = False


def S_infty(E, delta):
    """
    Returns the corners of the :math:`\delta - L_{\infty}`-Box around a Triangle.

    :param E: nd.array, real, shape (2,3) Vertices of the triangle
    :param delta: real, Size of the L-infinity box

    :return: nd.array, real, shape (2,12) Corners of $B$
    """
    D = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])*delta
    return np.repeat(E, 4, axis =1) + np.tile(D.transpose(), 3)

def inNbhd(aT, bT, delta, method="Ml2", v=False):
    """
    Checks whether two triangles interact. If they do it returns True. That means this function
    only allows a very coarse information hand hence a very coarse approximation of the integrals.

    :param aT: clsTriangle, Triangle a
    :param bT: clsTriangle, Triangle b
    :param delta: real, Interaction Radius
    :param method: str, default="l2". Norm w.r.t which the Ball is constructed. Options l2, inf
    :param v: bool, Verbose switch
    :return:
    """
    if method == "inf":
        return inNbhd_inf(aT, bT, delta, v)
    elif method == "l2Bary":
        return inNbhd_l2_bary(aT, bT, delta, v)
    elif method == "Ml2":
        return MinNbhd_l2(aT, bT, delta, v)
    elif method == "Mfulll2":
        return MinNbhdfull_l2(aT, bT, delta, v)
    return None

def xinNbhd(P, aT, bT, delta):
    """ Tests whether a reference point p in Triangle a interacts
    with triangle b w.r.t. :math:`L_2`-ball of radius delta.

    :param P: ndarray, real, shape (2,m) Point in the reference triangle.
    :param aT: clsTriangle Triangle a
    :param bT: clsTriangle Triangle b
    :param delta: real Radius of L2-Ball
    :return: bool True if the triangles interact.
    """
    a_elPoints = aT.toPhys(P)
    b_baryC = bT.baryCenter()[:, np.newaxis]
    # In order to make this work for a refPoints of shape (2,m) do
    # is_inNbhd = np.linalg.norm(a_elPoint - b_baryC, axis=0) <= delta
    is_inNbhd = np.sum((a_elPoints - b_baryC)**2, axis=0) <= delta**2
    Pdx_inNbhd = np.flatnonzero(is_inNbhd)
    return Pdx_inNbhd

def MinNbhd_l2(aT, bT, delta, v=False):
    """
    Check whether two triangles interact w.r.t :math:`L_{2}`-ball of size delta.
    Returns an array of boolen values. If all are True Eb lies in a subset of the
    interaction set S(Ea)

    :param aT: clsTriangle, Triangle a
    :param bT: clsTriangle, Triangle b
    :param delta: Size of L-2 ball
    :param v: Verbose mode.
    :return: ndarray, bool, shape (3,) Entry i is True if Ea[i] and the barycenter of Eb interact.
    """
    M =  np.sum((aT.E - bT.baryCenter()[np.newaxis])**2, axis=1)
    M = M <= delta**2
    return M

def MinNbhdfull_l2(aT, bT, delta, v=False):
    """
    Check whether two triangles interact w.r.t :math:`L_{2}`-ball of size delta.
    Returns an array of boolen values. If all are True Eb lies in a subset of the
    interaction set S(Ea)

    :param aT: clsTriangle, Triangle a
    :param bT: clsTriangle, Triangle b
    :param delta: Size of L-2 ball
    :param v: Verbose mode.
    :return: ndarray, bool, shape (3,3) Entry i,j is True if Ea[i] and Eb[j] interact.
    """
    M = np.sqrt(np.sum((aT.E[np.newaxis] - bT.E[:, np.newaxis])**2, axis=2))
    M = M <= delta
    return M

def inNbhd_l2_bary(aT, bT, delta, v=False):
    """
    Check whether two triangles interact w.r.t :math:`L_{2}`-ball of size delta.

    :param aT: clsTriangle, Triangle a
    :param bT: clsTriangle, Triangle b
    :param delta: Size of L-2 ball
    :param v: Verbose mode.
    :return: bool True if Ea and Eb interact.
    """
    difference = aT.baryCenter() - bT.baryCenter()
    norm = np.linalg.norm(difference)
    return np.all(norm <= delta)

def inNbhd_inf(Ea, Eb, delta, v=True):
    """
    Check whether two triangles interact w.r.t :math:`L_{\infty}`-ball of size delta.

    :param aT: clsTriangle, Triangle a
    :param bT: clsTriangle, Triangle b
    :param delta: Size of L-infinity ball
    :param v: Verbose mode.
    :return: bool True if Ea and Eb interact.
    """
    # Check whether the triangles share a point
    Ea = aT.E.T
    Eb = bT.E.T
    intersection = np.intersect1d(Ea, Eb)
    if intersection.size > 0:
        return True

    # Determine extreme points of infty-norm ball around Ea
    SEa = S_infty(Ea, delta=delta)
    
    if v:
        plt.scatter(SEa[0], SEa[1], c="r", alpha=.2)
        plt.scatter(Eb[0], Eb[1], c="b", alpha=.2)
        plt.show()
        
    low_a = np.amin(SEa, axis=1)
    upp_a = np.amax(SEa, axis=1)
    low_b = np.amin(Eb, axis=1)
    upp_b = np.amax(Eb, axis=1)

    # Check whether the triangles are very far away from each other
    if any(low_a >= upp_b) or any(upp_a <= low_b):
        if v:
            print("no lp solve required because: ")
            print("Low_a", low_a, "Upp_b", upp_b)
            print("Low_b", low_b, "Upp_a", upp_a)
        return False

    # If not solve an LP
    data = np.concatenate((-SEa, Eb), axis=1)
    bias = np.array([[-1] * 12 + [1] * 3])
    data = np.concatenate((data, bias), axis=0).T

    # Solve full problem
    G = matrix(data, size=(15, 3))
    c = matrix(0., size=(3, 1))
    h = matrix(-1., size=(15, 1))

    x = solvers.lp(c, G, h)
    return x["status"] != "optimal"


class clsMesh:
    """
    Mesh Class


    """
    def __init__(self, Verts, Lines, Triangles):
        """Constructor Method

        Takes Verts, Lines and Triangles from readmesh and executes prepare.
        """
        args = self.prepare(Verts, Lines, Triangles)
        # args = Verts, Triangles, K, K_Omega, J, J_Omega
        self.V = args[0]
        self.T = args[1][:, 1:]
        self.K = args[2]
        self.K_Omega = args[3]
        self.J = args[4]
        self.J_Omega = args[5]

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
        Vdx = self.T[Tdx]
        E = self.V[Vdx]
        return clsTriangle(E)

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

    def neighbor(self, Tdx):
        """
        Return list of indices of neighbours of Tdx.

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
    def __init__(self, P, weights, delta):
        self.delta = delta
        psi0 = 1 - P[0, :] - P[1, :]
        psi1 = P[0, :]
        psi2 = P[1, :]
        psi = np.array([psi0, psi1, psi2])
        self.psi = psi
        self.P = P
        self.weights = weights
        self.log_xinNbhd = 0
        self.log_xnotinNbhd = 0
        self.counter = 0

    def A(self, a, b, aT, bT, is_allInteract=True):
        if is_allInteract:
            # P, weights and psi are just views. The data is not copied. See
            # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
            P = self.P
            weights = self.weights
            psi = self.psi

        else:
            dx_sInteract = []

            dx_sInteract = xinNbhd(self.P, aT, bT, self.delta)
            self.log_xinNbhd += int(len(dx_sInteract) == 7)
            self.log_xnotinNbhd += int(len(dx_sInteract) == 0)
            self.counter += 1
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