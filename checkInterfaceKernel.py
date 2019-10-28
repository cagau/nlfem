import numpy as np
import matplotlib.pyplot as plt
my_Ad_O = np.load("Ad_O.npy")
chris_Ad_O = np.load("../compare_data/A_Chris.npy")
diff_norm = np.linalg.norm(chris_Ad_O - my_Ad_O)
print("L2 Norm Difference:\t", diff_norm)
Ad_diff = my_Ad_O - chris_Ad_O
minim = np.min(Ad_diff)
maxim = np.max(Ad_diff)
plt.imshow(Ad_diff)
plt.show()