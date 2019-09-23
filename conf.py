#-*- coding:utf-8 -*-

import numpy as np
mesh_name = "circle_large"
delta = .1
SOLVE = False
ansatz = "CG"

py_P = np.array([[0.33333333333333,    0.33333333333333],
          [0.47014206410511,    0.47014206410511],
          [0.47014206410511,    0.05971587178977],
          [0.05971587178977,    0.47014206410511],
          [0.10128650732346,    0.10128650732346],
          [0.10128650732346,    0.79742698535309],
          [0.79742698535309,    0.10128650732346]])

weights = np.array([0.22500000000000,
                0.13239415278851,
                0.13239415278851,
                0.13239415278851,
                0.12593918054483,
                0.12593918054483,
                0.12593918054483])/2
