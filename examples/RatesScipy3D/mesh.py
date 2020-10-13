import scipy.spatial as sp
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean as l2dist
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import meshzoo
from nlfem import constructAdjaciencyGraph

class RegMesh2D:
    def __init__(self, delta, n, ufunc=None, coarseMesh=None,
                 dim=2, ansatz="CG", boundaryConditionType="Dirichlet",
                 is_constructAdjaciencyGraph=True):
        self.n = n
        self.dim = dim
        self.delta = delta
        self.sqdelta = delta**2
        self.h = np.round((1+2*delta)/n, decimals=4)

        # Construct Meshgrid -------------------------------------------------------------
        if self.dim == 2:
            self.vertices, self.elements = meshzoo.rectangle(
                xmin=-delta, xmax=1+delta,
                ymin=-delta, ymax=1+delta,
                nx=n+1, ny=n+1,
                zigzag=False)
            self.elements = np.array(self.elements, dtype=np.int)
            self.vertices = self.vertices[:, :2]

        if self.dim == 3:
            self.vertices, self.elements = meshzoo.cube(-delta, 1+delta, -delta, 1+delta,
                                                        -delta, 1+delta,
                                                        nx=n+1, ny=n+1, nz=n+1)

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
        #self.neighbours = np.array(self.triangulation.neighbors, dtype=np.int)
        #self.neighbours = np.where(self.neighbours != -1, self.neighbours, self.nE)


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
        self.u_exact = np.zeros(self.vertices.shape[0])
        self.ud = np.zeros(self.vertices.shape[0])

        if ufunc is not None:
            if hasattr(ufunc, '__call__'):
                self.set_u_exact(ufunc)

        if coarseMesh is not None:
            self.interpolator = LinearNDInterpolator(coarseMesh.vertices,
                                                     coarseMesh.ud)
            self.ud = self.interpolator(self.vertices)

    def set_u_exact(self, ufunc):
        for i, x in enumerate(self.vertices):
            self.u_exact[i] = ufunc(x)

    def write_ud(self, udata, ufunc):
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

    def plot_ud(self, pp=None):
        if self.dim == 2:
            if self.u_exact is None:
                plt.tricontourf(self.vertices[:, 0], self.vertices[:, 1], self.elements, self.ud)
            else:
                plt.tricontourf(self.vertices[:, 0], self.vertices[:, 1], self.elements, self.ud-self.u_exact)
            plt.triplot(self.vertices[:, 0], self.vertices[:, 1], self.elements,lw=.1, color='white', alpha=.3)
            #plt.triplot(self.vertices[:, 0], self.vertices[:, 1], self.elements[np.where(self.elementLabels==1)[0]], lw=1, color='white', alpha=.4)
            #plt.scatter(self.vertices[self.omega, 0], self.vertices[self.omega, 1], c = "white", alpha=1)
            if pp is None:
                plt.show()
            else:
                plt.savefig(pp, format='pdf')
                plt.close()
    def plot_u_exact(self, pp=None):
        if self.dim == 2:
            plt.tricontourf(self.vertices[:, 0], self.vertices[:, 1], self.elements, self.u_exact)
            plt.triplot(self.vertices[:, 0], self.vertices[:, 1], self.elements,lw=.1, color='white', alpha=.3)
            plt.triplot(self.vertices[:, 0], self.vertices[:, 1], self.elements[np.where(self.elementLabels==1)[0]], lw=1, color='white', alpha=1)
            plt.scatter(self.vertices[self.omega, 0], self.vertices[self.omega, 1], c = "white", alpha=1)
            if pp is None:
                plt.show()
            else:
                plt.savefig(pp, format='pdf')
                plt.close()
    def plot3D(self, filename="foo.vtk", dataDict={}):
        import meshio
        if self.dim == 3:

            point_data = {"u_exact": self.u_exact,
                          "ud": self.ud,
                          "u_diff": self.u_exact-self.ud,
                          "labels": self.vertexLabels}
            point_data.update(dataDict)
            m = meshio.Mesh(points  = self.vertices, cells = [("tetra", self.elements)],
                            point_data= point_data)
            meshio.write(filename, m, file_format="vtk")

    def save(self, path):
        import os
        os.makedirs(path, exist_ok=True)

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

class RegMesh:
    def __init__(self, delta, n, ufunc=None, coarseMesh=None, dim=2, ansatz="CG", boundaryConditionType="Dirichlet", **kwargs):
        self.n = n
        self.dim = dim
        self.delta = delta
        self.sqdelta = delta**2

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
        # http://qhull.org/html/qh-optq.htm#Qg
        # https://github.com/scipy/scipy/issues/2626
        self.triangulation = Delaunay(self.vertices, qhull_options="Qt")
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
        self.u_exact = None# np.zeros(self.vertices.shape[0])
        self.ud = np.zeros(self.vertices.shape[0])

        if ufunc is not None:
            if hasattr(ufunc, '__call__'):
                self.set_u_exact(ufunc)

        if coarseMesh is not None:
            self.interpolator = LinearNDInterpolator(coarseMesh.vertices,
                                                     coarseMesh.ud)
            self.ud = self.interpolator(self.vertices)

    def set_u_exact(self, ufunc):
        self.u_exact = np.zeros(self.vertices.shape[0])
        for i, x in enumerate(self.vertices):
            self.u_exact[i] = ufunc(x)

    def write_ud(self, udata, ufunc):
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

    def plot2D(self, pp=None):
        if self.dim == 2:
            plt.tricontourf(self.vertices[:, 0], self.vertices[:, 1], self.elements, self.u)
            plt.triplot(self.vertices[:, 0], self.vertices[:, 1], self.elements,lw=1, color='white', alpha=.3)
            plt.scatter(self.vertices[self.omega, 0], self.vertices[self.omega, 1], c = "w", s=1.4, alpha=1)
            if pp is None:
                plt.show()
            else:
                plt.savefig(pp, format='pdf')
                plt.close()
    def plot3D(self, filename="foo.vtk"):
        import meshio
        if self.dim == 3:
            #class Mesh:
            #def __init__(
            #        self,
            #        points, (ndarray)
            #        cells, (list of tuple (celltype, data))
            #        point_data=None, (dict)
            #        cell_data=None, (dict)
            #        field_data=None,
            #        point_sets=None,
            #        cell_sets=None,
            #        gmsh_periodic=None,
            #        info=None,
            #):
            m = meshio.Mesh(points  = self.vertices, cells = [("tetra", self.elements)],
                            point_data= {"u_exact": self.u_exact,
                                         "ud": self.ud,
                                         "u_diff": self.u_exact-self.ud,
                                         "labels": self.vertexLabels},
                            cell_data={"labels": self.elementLabels}   )
            meshio.write(filename, m, file_format="vtk")

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
            "dim"]
        [writeattr(f, attr_name) for attr_name in confList]

        self.vertices.tofile(path+"/mesh.verts")
        self.elements.tofile(path+"/mesh.elemt")
        self.neighbours.tofile(path+"/mesh.neigh")
        self.elementLabels.tofile(path+"/mesh.elelb")



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
        mesh.plot2D(pp)
        mesh = RegMesh2D(delta,N_fine, coarseMesh=mesh)
        print("L2 Error: ", l2dist(mesh_exact.u, mesh.u))
    pp.close()

if __name__ == "__main__":
    import examples.RatesScipy3D.conf3D as conf
    u_exact = lambda x: x[0]*2 + x[1] *3 - x[2]*10 + 10
    delta = .1
    n = 12
    mesh = RegMesh(delta, n, ufunc=u_exact, dim=3)
    mesh.plot3D(conf.fnames["tetPlot.vtk"])
