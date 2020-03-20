import scipy.spatial as sp
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean as l2dist
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class RegMesh2D:
    def __init__(self, delta, n, ufunc=None, coarseMesh=None, dim=2, ansatz="CG", boundaryConditionType="Dirichlet"):
        self.n = n
        self.dim = dim
        self.delta = delta

        # Construct Meshgrid -------------------------------------------------------------
        if self.dim == 2:
            line1d = np.linspace(-delta,1+delta,self.n+1)
            self.h = np.abs(line1d[0]- line1d[1])
            X,Y = np.meshgrid(line1d,line1d)
            x = X.flatten()
            y = Y.flatten()
            self.vertices = np.array([x,y]).transpose()
        if self.dim == 3:
            line1d = np.linspace(-delta,1+delta,self.n+1)
            self.h = np.abs(line1d[0]- line1d[1])
            X, Y, Z = np.meshgrid(line1d,line1d, line1d)
            x = X.flatten()
            y = Y.flatten()
            z = Z.flatten()
            self.vertices = np.array([x,y,z]).transpose()

        # Construct and Sort Vertices ----------------------------------------------------

        self.vertexLabels = np.array([self.get_vertexLabel(v) for v in self.vertices])
        self.argsort_labels = np.argsort(self.vertexLabels)
        self.vertices = self.vertices[self.argsort_labels]
        self.vertexLabels = self.vertexLabels[self.argsort_labels]

        # Get Number of Vertices in Omega ------------------------------------------------
        self.omega = np.where(self.vertexLabels == 1)[0]
        self.nV = self.vertices.shape[0]
        self.nV_Omega = self.omega.shape[0]

        # Set up Delaunay Triangulation --------------------------------------------------
        self.triangulation = Delaunay(self.vertices)
        self.elements = np.array(self.triangulation.simplices, dtype=np.int)

        # Get Triangle Labels ------------------------------------------------------------
        self.nE = self.elements.shape[0]
        self.elementLabels = np.zeros(self.nE, dtype=np.int)
        for k, E in enumerate(self.elements):
            self.elementLabels[k] = self.get_elementLabel(E)
        self.nE_Omega = np.sum(self.elementLabels == 1)

        # Read adjaciency list
        self.neighbours = np.array(self.triangulation.neighbors, dtype=np.int)
        self.neighbours = np.where(self.neighbours != -1, self.neighbours, self.nE)

        # Set Matrix Dimensions ----------------------------------------------------------
        # In case of Neumann conditions we assemble a Maitrx over Omega + OmegaI.
        # In order to achieve that we "redefine" Omega := Omega + OmegaI
        # This is rather a shortcut to make things work quickly.
        self.is_NeumannBoundary = False
        if boundaryConditionType == "Neumann":
            self.nE_Omega = self.nE
            self.nV_Omega = self.nV
            self.is_NeumannBoundary = True

        if ansatz == "DG":
            self.K = self.nE*3
            self.K_Omega = self.nE_Omega*3
            self.is_DiscontinuousGalerkin = True
        else:
            self.K = self.nV
            self.K_Omega = self.nV_Omega
            self.is_DiscontinuousGalerkin = False

        # Set Mesh Data if provided ------------------------------------------------------
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

    def write_u(self, udata, ufunc):
        self.set_u(ufunc)
        self.u[:self.K_Omega] = udata

    def get_vertexLabel(self, v):
        if np.max(np.abs(v - 0.5)) < 0.5:
            return 1
        else:
            return 2

    def get_elementLabel(self, E):
        # If any vertex of an element lies in Omega then the element does.
        # This is due to the fact that only vertices in the interior of Omega
        # have label 1.
        for vdx in E:
            if self.vertexLabels[vdx] == 1:
                return 1
        return 2

    def plot(self, pp=None):
        if self.dim == 2:
            plt.tricontourf(self.vertices[:, 0], self.vertices[:, 1], self.elements, self.u)
            plt.triplot(self.vertices[:, 0], self.vertices[:, 1], self.elements,lw=1, color='white', alpha=.3)
            plt.scatter(self.vertices[self.omega, 0], self.vertices[self.omega, 1], c = "w", s=1.4, alpha=1)
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