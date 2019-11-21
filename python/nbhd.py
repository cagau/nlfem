#-*- coding:utf-8 -*-

import numpy as np

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
    is_notinNbhd = (np.sum((a_elPoints - b_baryC)**2, axis=0) > delta**2)
    Pdx_notinNbhd = np.flatnonzero(is_notinNbhd)
    return Pdx_notinNbhd
