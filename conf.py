#-*- coding:utf-8 -*-

import numpy as np
#mesh_name = "../compare_data/target_shape"
mesh_name = "circle_huge"
delta = .1
SOLVE = True
ansatz = "CG"
# quad_order_outer == 8
py_Px = np.array([[0.33333333, 0.33333333],
              [0.45929259, 0.45929259],
              [0.45929259, 0.08141482],
              [0.08141482, 0.45929259],
              [0.17056931, 0.17056931],
              [0.17056931, 0.65886138],
              [0.65886138, 0.17056931],
              [0.05054723, 0.05054723],
              [0.05054723, 0.89890554],
              [0.89890554, 0.05054723],
              [0.26311283, 0.72849239],
              [0.72849239, 0.00839478],
              [0.00839478, 0.26311283],
              [0.72849239, 0.26311283],
              [0.26311283, 0.00839478],
              [0.00839478, 0.72849239]])

dx = 0.5 * np.array([0.14431560767779
                       , 0.09509163426728
                       , 0.09509163426728
                       , 0.09509163426728
                       , 0.10321737053472
                       , 0.10321737053472
                       , 0.10321737053472
                       , 0.03245849762320
                       , 0.03245849762320
                       , 0.03245849762320
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443])
# quad_order_inner == 5:
py_Py = np.array([[0.33333333333333, 0.33333333333333],
               [0.47014206410511, 0.47014206410511],
               [0.47014206410511, 0.05971587178977],
               [0.05971587178977, 0.47014206410511],
               [0.10128650732346, 0.10128650732346],
               [0.10128650732346, 0.79742698535309],
               [0.79742698535309, 0.10128650732346]])

dy = 0.5 * np.array([0.22500000000000,
                     0.13239415278851,
                     0.13239415278851,
                     0.13239415278851,
                     0.12593918054483,
                     0.12593918054483,
                     0.12593918054483])