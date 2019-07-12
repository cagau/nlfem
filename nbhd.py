#-*- coding:utf-8 -*-

import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def S_infty(E, delta):
    """
    Returns the corners of the :math:`\delta - L_{\infty}`-Box around a Triangle.

    :param E: nd.array, real, shape (2,3) Vertices of the triangle
    :param delta: real, Size of the L-infinity box

    :return: nd.array, real, shape (2,12) Corners of $B$
    """
    D = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])*delta
    return np.repeat(E, 4, axis =1) + np.tile(D.transpose(), 3)

def inNbhd(aT, bT, delta, method="Ml2Bary", v=False):
    """
    Wrapper for the other inNbhd-functions.

     - The options *inf, lsBary* return a boolean value.
     - The options *Ml2Bary, Mfulll2* return a matrix of boolean values.

    :param aT: clsTriangle, Triangle a
    :param bT: clsTriangle, Triangle b
    :param delta: real, Interaction Radius
    :param method: str, default="l2". Norm w.r.t which the Ball is constructed. Options inf, l2Bary, Ml2Bary, Mfulll2
    :param v: bool, Verbose switch
    :return:
    """
    if method == "inf":
        return inNbhd_inf(aT, bT, delta, v)
    elif method == "l2Bary":
        return inNbhd_l2Bary(aT, bT, delta, v)
    elif method == "Ml2Bary":
        return MinNbhd_l2Bary(aT, bT, delta, v)
    elif method == "Mfulll2":
        return MinNbhdfull_l2(aT, bT, delta, v)
    return None


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

def inNbhd_l2Bary(aT, bT, delta, v=False):
    """
    Check whether two triangles interact w.r.t :math:`L_{2}`-ball of size delta.
    To that end, the function compares the barycenters of the triangles.

    :param aT: clsTriangle, Triangle a
    :param bT: clsTriangle, Triangle b
    :param delta: Size of L-2 ball
    :param v: Verbose mode.
    :return: bool True if Ea and Eb interact.
    """
    difference = aT.baryCenter() - bT.baryCenter()
    norm = np.linalg.norm(difference)
    return np.all(norm <= delta)

def MinNbhd_l2Bary(aT, bT, delta, v=False):
    """
    Check whether two triangles interact w.r.t :math:`L_{2}`-ball of size delta.
    Returns an array of boolen values. Compares the barycenter of bT with the vertices of Ea.

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
    Returns an array of boolen values. Compares all vertices of aT with all vertices of bT.

    :param aT: clsTriangle, Triangle a
    :param bT: clsTriangle, Triangle b
    :param delta: Size of L-2 ball
    :param v: Verbose mode.
    :return: ndarray, bool, shape (3,3) Entry i,j is True if Ea[i] and Eb[j] interact.
    """
    M = np.sqrt(np.sum((aT.E[np.newaxis] - bT.E[:, np.newaxis])**2, axis=2))
    M = M <= delta
    return M

def xnotinNbhd(P, aT, bT, delta):
    """ Tests whether a reference point p in Triangle a interacts
    with triangle b w.r.t. :math:`L_2`-ball of radius delta.

    :param P: ndarray, real, shape (2,m) Point in the reference triangle.
    :param aT: clsTriangle Triangle a
    :param bT: clsTriangle Triangle b
    :param delta: real Radius of L2-Ball
    :return: bool False if the physical point xi in Triangle a does interact with the barycenter of Triangle b.
    """
    a_elPoints = aT.toPhys(P)
    b_baryC = bT.baryCenter()[:, np.newaxis]
    # In order to make this work for a refPoints of shape (2,m) do
    # is_inNbhd = np.linalg.norm(a_elPoint - b_baryC, axis=0) <= delta
    is_notinNbhd = np.logical_not(np.sum((a_elPoints - b_baryC)**2, axis=0) <= delta**2)
    Pdx_notinNbhd = np.flatnonzero(is_notinNbhd)
    return Pdx_notinNbhd

if __name__=="__main__":
    from conf import P, weights, mesh_name, delta
    from nlocal import clsMesh
    # Mesh construction --------------------
    mesh = clsMesh("circle_" + mesh_name + ".msh")

    for i in range(mesh.J):
        for j in range(i):
            aT = mesh.Triangles[i]
            bT = mesh.Triangles[j]
            Mis_interact = inNbhd(aT, bT, delta, method="Ml2Bary")
            if Mis_interact.any() and not Mis_interact.all():
                Pdx_xnotinNbhd = xnotinNbhd(P, aT, bT, delta)
                print(Pdx_xnotinNbhd)
                title = "aTdx "+str(i)+", bTdx " + str(j) + "\n" + str(Pdx_xnotinNbhd)
                mesh.plot([i, j], is_plotmsh=True, pdfname="output/xinNbhd/" + str(i)+"_"+str(j), title=title, delta=delta, refPoints=P[:, Pdx_xnotinNbhd])

                #title = "aTdx "+str(j)+", bTdx " + str(i) + "\n" + str(xnotinNbhd(P, bT, aT, delta))
                #mesh.plot([j, i], is_plotmsh=True, pdfname="output/xinNbhd/" + str(i)+"_"+str(j)+"_T", title=title, delta=delta, refPoints=P)
