import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import readmesh as rm
from nlocal import clsMesh
import numpy as np

mesh = clsMesh(*rm.read_mesh("circle_medium.msh"))
# arg = Ad_O, fd, ud_Ext, fd_Ext
file_name = "0704_12-17-56_medium2"
arg = np.load(file_name + ".npz")
Ad_O = arg["arr_0"]
ud_Ext = arg["arr_2"]
fd_Ext = arg["arr_3"]

pp = PdfPages(file_name + ".pdf")

plt.gca().set_aspect('equal')
plt.triplot(mesh.V[:, 0], mesh.V[:, 1], mesh.T, lw=0.5, color='white', alpha=.5)
plt.tricontourf(mesh.V[:, 0], mesh.V[:, 1], mesh.T, fd_Ext)
plt.colorbar(orientation='horizontal', shrink=.7)
plt.title("Right side f")
plt.savefig(pp, format='pdf')
plt.close()

fig = plt.figure()
ax = fig.gca(projection='3d', title="Right side f")
ax.plot_trisurf(mesh.V[:, 0], mesh.V[:, 1], fd_Ext)
plt.savefig(pp, format='pdf')
plt.close()

plt.gca().set_aspect('equal')
plt.triplot(mesh.V[:, 0], mesh.V[:, 1], mesh.T, lw=0.5, color='white', alpha=.5)
plt.tricontourf(mesh.V[:, 0], mesh.V[:, 1], mesh.T, ud_Ext)
plt.colorbar(orientation='horizontal', shrink=.7)
plt.title("Solution u")
plt.savefig(pp, format='pdf')
plt.close()

fig = plt.figure()
ax = fig.gca(projection='3d', title="Solution u")
ax.plot_trisurf(mesh.V[:, 0], mesh.V[:, 1], ud_Ext)
plt.savefig(pp, format='pdf')
plt.close()

plt.imshow(Ad_O)
plt.colorbar(orientation='horizontal', shrink=.7)
plt.title(r"$A_{\Omega\Omega}$")
plt.savefig(pp, format='pdf')
plt.close()

pp.close()
