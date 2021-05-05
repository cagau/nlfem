import scipy.spatial as sp
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean as l2dist
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from nlfem import constructAdjaciencyGraph
import meshzoo

class RegMesh2D:
    def __init__(self, delta, n, ufunc=None, coarseMesh=None,
                 dim=2, ansatz="CG", boundaryConditionType="Dirichlet",
                 is_constructAdjaciencyGraph=True):
        ### TEST 27.07.2020
        self.Zeta = np.arange(12, dtype=np.int).reshape(4, 3)

        #####


        self.n = n
        self.dim = dim
        self.delta = delta
        self.sqdelta = delta**2
        # Construct Meshgrid -------------------------------------------------------------
        if self.dim == 2:
            self.h = 12/n/10
            points, cells = meshzoo.rectangle(
                xmin=-self.delta, xmax=1.0+self.delta,
                ymin=-self.delta, ymax=1.0+self.delta,
                nx=n+1, ny=n+1,
                variant="zigzag"
            )
            self.vertices = np.array(points[:,:2])


        # Set up Delaunay Triangulation --------------------------------------------------
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
        self.elements = np.array(cells, dtype=np.int)
        self.remapElements()

        # Get Triangle Labels ------------------------------------------------------------
        self.nE = self.elements.shape[0]
        self.elementLabels = np.zeros(self.nE, dtype=np.int)
        for k, E in enumerate(self.elements):
            self.elementLabels[k] = self.get_elementLabel(E)
        self.nE_Omega = np.sum(self.elementLabels == 1)

        # Read adjaciency list
        if is_constructAdjaciencyGraph:
            self.neighbours = constructAdjaciencyGraph(self.elements)
            self.nNeighbours = self.neighbours.shape[1]
        else:
            self.neighbours = None

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
        self.u_exact = None
        self.ud = None

        if ufunc is not None:
            if hasattr(ufunc, '__call__'):
                self.set_u_exact(ufunc)

        if coarseMesh is not None:
            self.interpolator = LinearNDInterpolator(coarseMesh.vertices, coarseMesh.ud)
            self.ud = self.interpolator(self.vertices)

    def save(self, path):
        def writeattr(file, attr_name):
            file.write(attr_name+"\n")
            file.write(str(self.__dict__[attr_name])+"\n")

        f = open(path + "/mesh.conf", "w+")
        confList = [
            "K_Omega",
            "K",
            "nE",
            "nE_Omega",
            "nV",
            "nV_Omega",
            "sqdelta",
            "is_DiscontinuousGalerkin",
            "is_NeumannBoundary",
            "nNeighbours",
            "dim"]
        [writeattr(f, attr_name) for attr_name in confList]

        self.vertices.tofile(path+"/mesh.verts")
        self.elements.tofile(path+"/mesh.elemt")
        self.neighbours.tofile(path+"/mesh.neigh")
        self.elementLabels.tofile(path+"/mesh.elelb")

    def remapElements(self):
        def invert_permutation(p):
            """
            The function inverts a given permutation.
            :param p: nd.array, shape (m,) The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
            :return: nd.array, shape (m,) Returns an array s, where s[i] gives the index of i in p.
            """
            s = np.empty(p.size, p.dtype)
            s[p] = np.arange(p.size)
            return s
        piVdx_invargsort = invert_permutation(self.argsort_labels)
        piVdx = lambda dx: piVdx_invargsort[dx]  # Permutation definieren
        self.elements = piVdx(self.elements)

    def set_u_exact(self, ufunc):
        self.u_exact = np.zeros(self.vertices.shape[0])
        for i, x in enumerate(self.vertices):
            self.u_exact[i] = ufunc(x)

    def write_ud(self, udata, ufunc):
        self.ud = np.zeros(self.vertices.shape[0])
        for i, x in enumerate(self.vertices):
            self.ud[i] = ufunc(x)
        self.ud[:self.K_Omega] = udata

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

    def plot_ud(self, pp=None):
        if self.dim == 2:
            plt.tricontourf(self.vertices[:, 0], self.vertices[:, 1], self.elements, self.ud)
            plt.triplot(self.vertices[:, 0], self.vertices[:, 1], self.elements,lw=.1, color='white', alpha=.3)
            #plt.scatter(self.vertices[self.omega, 0], self.vertices[self.omega, 1], c = "black", s=.2, alpha=.7)
            if pp is None:
                plt.show()
            else:
                plt.savefig(pp, format='pdf')
                plt.close()
    def plot_u_exact(self, pp=None):
        if self.dim == 2:
            plt.tricontourf(self.vertices[:, 0], self.vertices[:, 1], self.elements, self.u_exact)
            plt.triplot(self.vertices[:, 0], self.vertices[:, 1], self.elements,lw=.1, color='white', alpha=.3)
            #plt.scatter(self.vertices[self.omega, 0], self.vertices[self.omega, 1], c = "black", s=.2, alpha=.7)
            if pp is None:
                plt.show()
            else:
                plt.savefig(pp, format='pdf')
                plt.close()


def testInterpolation2D():
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
        mesh.plot_ud(pp)
        mesh = RegMesh2D(delta,N_fine, coarseMesh=mesh)
        print("L2 Error: ", l2dist(mesh_exact.ud, mesh.ud))
    pp.close()

if __name__ == "__main__":
    n = 12
    M = meshzoo.cube(-0.1, 1.1, -.1, 1.1, nx=n+1, ny=n+1, nz=n+1)
    print(M)