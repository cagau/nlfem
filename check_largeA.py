import numpy as np

cy_Ad_O = np.load("Ad_O.npy")
py_Ad_O = np.load("0820_12-14-38Ad_O.npy")

print("Check\t", np.linalg.norm(cy_Ad_O - py_Ad_O))