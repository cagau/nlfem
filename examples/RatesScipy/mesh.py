import scipy.spatial as sp
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean as l2dist
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class RegMesh2D:
    def __init__(self, delta, n, ufunc=None, coarseMesh=None):
        self.n = n
        self.delta = delta

        # Construct Meshgrid
        line1d = np.linspace(-delta,1+delta,self.n+1)
        X,Y = np.meshgrid(line1d,line1d)
        x = X.flatten()
        y = Y.flatten()

        # Construct and Sort Vertices
        self.vertices = np.array([x,y]).transpose()
        self.vert_labels = np.array([self.vertexLabel(v) for v in self.vertices])
        self.argsort_labels = np.argsort(self.vert_labels)
        self.vertices = self.vertices[self.argsort_labels]
        self.vert_labels = self.vert_labels[self.argsort_labels]

        # Separate Omega from OmegaI
        self.omega = np.where(self.vert_labels == 1)

        # Set up Delaunay Triangulation
        self.triangulation = Delaunay(self.vertices)
        self.elements = self.triangulation.simplices
        self.neighbours = self.triangulation.neighbors

        self.u = None
        if ufunc is not None:
            if hasattr(ufunc, '__call__'):
                self.set_u(ufunc)
        elif coarseMesh is not None:
            self.interpolator = LinearNDInterpolator(coarseMesh.triangulation, coarseMesh.u)
            self.u = self.interpolator(self.vertices)

    def set_u(self,ufunc):
        self.u = np.zeros(self.vertices.shape[0])
        for i, x in enumerate(self.vertices):
            self.u[i] = ufunc(x)

    def vertexLabel(self, v):
        if np.max(np.abs(v - 0.5)) < 0.5:
            return 1
        else:
            return 2

    def plot(self, pp=None):

        plt.tricontourf(self.vertices[:, 0], self.vertices[:, 1], self.elements, self.u)
        plt.triplot(self.vertices[:, 0], self.vertices[:, 1], self.elements,lw=1, color='white', alpha=.3)
        plt.scatter(self.vertices[self.omega, 0], self.vertices[self.omega, 1], c = "w",  alpha=.3)
        if pp is None:
            plt.show()
        else:
            plt.savefig(pp, format='pdf')
            plt.close()

if __name__ == "__main__":
    delta = .1
    n_start = 12
    layers = list(range(3))
    N  = [n_start*2**(l) for l in layers]
    N_fine = N[-1]*2

    print("Interpolating Linear Function")
    ufunc = lambda x: 3*x[0]+4*x[1]
    mesh_exact = RegMesh2D(delta,N_fine, ufunc)
    pp = PdfPages("testplot.pdf")
    for n in N:
        mesh=RegMesh2D(delta, n, ufunc)
        mesh.plot(pp)
        mesh = RegMesh2D(delta,N_fine, coarseMesh=mesh)
        print("L2 Error: ", l2dist(mesh_exact.u, mesh.u))
    pp.close()