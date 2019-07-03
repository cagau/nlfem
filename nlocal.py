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

def p_in_nbhd(refPoint, aT, bT, delta):
    """ Tests whether a reference point p in Triangle a interacts
    with triangle b w.r.t. :math:`L_2`-ball of radius delta.

    :param refPoint: ndarray, real, shape (2,) Point in the reference triangle.
    :param aT: clsTriangle Triangle a
    :param bT: clsTriangle Triangle b
    :param delta: real Radius of L2-Ball
    :return: bool True if the triangles interact.
    """
    if refPoint.shape != (2,):
        raise ValueError("p_in_nbhd() only accepts a single refPoint.")
    a_elPoint = aT.toPhys(refPoint)
    b_baryC = bT.baryCenter().reshape((-1, 1))
    # In order to make this work for a refPoints of shape (2,m) do
    # is_inNbhd = np.linalg.norm(a_elPoint - b_baryC, axis=0) <= delta
    is_inNbhd = np.linalg.norm(a_elPoint - b_baryC) <= delta
    #Pdx_inNbhd = np.flatnonzero(is_inNbhd)
    return is_inNbhd

def inNbhd(aT, bT, delta, norm="inf", v=False):
    """
    Checks whether two triangles interact. If they do it returns True. That means this function
    only allows a very coarse information hand hence a very coarse approximation of the integrals.

    :param aT: clsTriangle, Triangle a
    :param bT: clsTriangle, Triangle b
    :param delta: real, Interaction Radius
    :param norm: str, default="l2". Norm w.r.t which the Ball is constructed. Options l2, inf
    :param v: bool, Verbose switch
    :return:
    """
    if norm == "inf":
        return inNbhd_inf(aT, bT, delta, v)
    elif norm == "l2":
        return inNbhd_l2(aT, bT, delta, v)
    return None

def inNbhd_l2(aT, bT, delta, v=False):
    """
    Check whether two triangles interact w.r.t :math:`L_{2}`-ball of size delta.

    :param aT: clsTriangle, Triangle a
    :param bT: clsTriangle, Triangle b
    :param delta: Size of L-2 ball
    :param v: Verbose mode.
    :return: bool True if Ea and Eb interact.
    """
    return np.linalg.norm(aT.baryCenter() - bT.baryCenter) <= delta

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
            plt.legend()
        plt.savefig(pp, format='pdf')
        plt.close()

        pp.close()
        return

class clsTriangle:
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