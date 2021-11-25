from scipy.spatial.distance import euclidean as l2dist
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

#from nlfem import constructAdjaciencyGraph
import meshzoo

class RegMesh2D:
    def __init__(self, delta, n,
                 ufunc=None,
                 coarseMesh=None,
                 dim=2, outdim=1,
                 n_start=12,
                 ansatz="CG",
                 #is_constructAdjaciencyGraph=True,
                 variant="up",
                 deltaK=None):
        ### TEST 27.07.2020
        #self.Zeta = np.arange(12, dtype=np.int).reshape(4, 3)
        #####
        deltaK = int(np.round(delta * 10))
        if not deltaK:
            raise ValueError("Delta has to be of the form delta = deltaK/10. for deltaK in N.")

        n_start = 5 + 2*deltaK
        self.lowerLeft = 0.0
        self.upperRight = 0.5

        self.n = n
        self.dim = dim
        self.delta = delta
        self.sqdelta = delta**2
        # Construct Meshgrid -------------------------------------------------------------
        if self.dim == 2:
            self.h = n_start/n/10.
            points, cells = meshzoo.rectangle(
                xmin=self.lowerLeft-self.delta, xmax=self.upperRight+self.delta,
                ymin=self.lowerLeft-self.delta, ymax=self.upperRight+self.delta,
                nx=self.n+1, ny=self.n+1,
                variant=variant
            )
            self.vertices = np.array(points[:, :2])


        # Set up Delaunay Triangulation --------------------------------------------------
        # Construct Vertices ----------------------------------------------------
        self.vertexLabels = np.array([self.get_vertexLabel(v) for v in self.vertices])

        # Get Number of Vertices in Omega ------------------------------------------------
        self.nV = self.vertices.shape[0]
        self.nV_Omega = np.sum(self.vertexLabels > 0)

        # Set up Delaunay Triangulation --------------------------------------------------
        self.elements = np.array(cells, dtype=np.int)

        # Get Triangle Labels ------------------------------------------------------------
        self.nE = self.elements.shape[0]
        self.elementLabels = np.zeros(self.nE, dtype=np.int)
        for k, E in enumerate(self.elements):
            self.elementLabels[k] = self.get_elementLabel(E)
        self.nE_Omega = np.sum(self.elementLabels > 0)

        # Read adjaciency list -----------------------------------------------------------
        #if is_constructAdjaciencyGraph:
        #    self.neighbours = constructAdjaciencyGraph(self.elements)
        #    self.nNeighbours = self.neighbours.shape[1]
        #else:
        #    self.neighbours = None

        # Set Matrix Dimensions ----------------------------------------------------------
        self.is_NeumannBoundary = False
        # This option is deprecated. It is sufficient to simply not
        self.outdim = outdim
        if ansatz == "DG":
            self.K = self.nE*(self.dim+1)*self.outdim
            self.K_Omega = self.nE_Omega*(self.dim+1)*self.outdim
            self.nodeLabels = np.repeat(self.elementLabels, (self.dim+1)*self.outdim)
            self.is_DiscontinuousGalerkin = True
        else:
            self.K = self.nV*self.outdim
            self.K_Omega = self.nV_Omega*self.outdim
            self.nodeLabels = np.repeat(self.vertexLabels, self.outdim)
            self.is_DiscontinuousGalerkin = False

        # Set Mesh Data if provided ------------------------------------------------------
        self.u_exact = None
        self.ud = None

        self.diam = self.h*np.sqrt(2)
        if ufunc is not None:
            if hasattr(ufunc, '__call__'):
                self.set_u_exact(ufunc)

        if coarseMesh is not None:
            if coarseMesh.is_DiscontinuousGalerkin:
                # TODO Check again, whether this is the right way to go for interpolation in DG space
                coarseDGverts = np.zeros(((coarseMesh.K // coarseMesh.outdim), coarseMesh.dim))
                for i, E in enumerate(coarseMesh.elements):
                    for ii, vdx in enumerate(E):
                        vert = coarseMesh.vertices[vdx]
                        coarseDGverts[(self.dim+1)*i + ii] = vert
                self.interpolator = LinearNDInterpolator(coarseDGverts, coarseMesh.ud)

                ud_aux = self.interpolator(self.vertices)

                self.ud = np.zeros(((self.K // self.outdim), self.outdim))
                for i, E in enumerate(self.elements):
                    for ii, vdx in enumerate(E):
                        self.ud[(self.dim + 1)*i + ii] = ud_aux[vdx]
                pass
            else:
                self.interpolator = LinearNDInterpolator(coarseMesh.vertices, coarseMesh.ud)
                self.ud = self.interpolator(self.vertices)

    def save(self, path):
        import os

        def writeattr(file, attr_name):
            file.write(attr_name+"\n")
            file.write(str(self.__dict__[attr_name])+"\n")

        os.makedirs(path, exist_ok=True)
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

        np.save(path + "/vertices.npy", self.vertices)
        np.save(path + "/elements.npy", self.elements)
        np.save(path + "/neighbours.npy", self.neighbours)
        np.save(path + "/elementsLabels.npy", self.elementLabels)
        np.save(path + "/verticesLabels.npy", self.vertexLabels)

    def set_u_exact(self, ufunc):
        if self.is_DiscontinuousGalerkin:
            self.u_exact = np.zeros(((self.K // self.outdim), self.outdim))
            for i, E in enumerate(self.elements):
                for ii, Vdx in enumerate(E):
                    vert = self.vertices[Vdx]
                    self.u_exact[(self.dim+1)*i + ii] = ufunc(vert)
        else:
            self.u_exact = np.zeros((self.vertices.shape[0], self.outdim))
            for i, x in enumerate(self.vertices):
                self.u_exact[i] = ufunc(x)

    def write_ud(self, udata, ufunc):
        nNodes = self.K // self.outdim
        #nNodesOmega = self.K_Omega // self.outdim
        self.ud = np.zeros((nNodes, self.outdim))
        if self.is_DiscontinuousGalerkin:
            for i, E in enumerate(self.elements):
                if self.elementLabels[i] < 0:
                    for ii, Vdx in enumerate(E):
                        vert = self.vertices[Vdx]
                        self.ud[(self.dim+1)*i + ii] = ufunc(vert)
        else:
            for i, x in enumerate(self.vertices):
                self.ud[i] = ufunc(x)
        self.ud[(self.nodeLabels > 0)[::self.outdim]] = udata

    def get_vertexLabel(self, v):
        label = 0
        if np.max(np.abs(v - self.upperRight/2.)) < self.upperRight/2. - 1e-9:
            label = 1
            if np.max(np.abs(v - self.upperRight/2.)) <= self.upperRight/4.:
                label = 2
        else:
            label = -1
        return label

    def get_elementLabel(self, E):
        # If any vertex of an element lies in Omega then the element does.
        # This is due to the fact that only vertices in the interior of Omega
        # have label 1.
        for vdx in E:
            if self.vertexLabels[vdx] > 0:
                return self.vertexLabels[vdx]
        return -1

    def plot_ud(self, pp=None, is_quickDG=True):
        if self.dim == 2:
            if self.is_DiscontinuousGalerkin:
                plt.triplot(self.vertices[:, 0], self.vertices[:, 1], self.elements, lw=.1, color='white', alpha=.3)

                if is_quickDG:
                    ud_aux = np.zeros(self.nV)
                    for i, E in enumerate(self.elements):
                        for ii, Vdx in enumerate(E):
                            ud_aux[Vdx] = self.ud[(self.dim+1)*i + ii]

                    plt.tricontourf(self.vertices[:, 0], self.vertices[:, 1], self.elements, ud_aux)

                #plt.scatter(self.vertices[self.omega, 0], self.vertices[self.omega, 1], c = "black", s=.2, alpha=.7)
                else:
                    minval = np.min(self.ud)
                    maxval = np.max(self.ud)

                    for k in range(self.nE):
                        if self.elementLabels[k] != 0:
                            Vdx = self.elements[k]
                            plt.tricontourf(self.vertices[Vdx, 0],
                                            self.vertices[Vdx, 1],
                                            self.ud[3*k:(3*k+3)].ravel(),  5,
                                            cmap=plt.cm.get_cmap('rainbow'), vmin=minval, vmax=maxval)
                    ax, _ = matplotlib.colorbar.make_axes(plt.gca(), shrink=.7)
                    matplotlib.colorbar.ColorbarBase(ax, cmap=plt.cm.get_cmap('rainbow'),
                                                     norm=matplotlib.colors.Normalize(vmin=minval, vmax=maxval))


                if pp is None:
                    plt.show()
                else:
                    plt.savefig(pp, format='pdf')
                    plt.close()
            else:
                plt.tricontourf(self.vertices[:, 0], self.vertices[:, 1], self.elements, self.ud.ravel())
                plt.triplot(self.vertices[:, 0], self.vertices[:, 1], self.elements,lw=.1, color='white', alpha=.3)
                #plt.scatter(self.vertices[self.omega, 0], self.vertices[self.omega, 1], c = "black", s=.2, alpha=.7)
                if pp is None:
                    plt.show()
                else:
                    plt.savefig(pp, format='pdf')
                    plt.close()

    def plot_vertexLabels(self, pp=None):
        if self.dim == 2:
            if self.is_DiscontinuousGalerkin:
                ud_aux = np.zeros(self.nV)
                for i, E in enumerate(self.elements):
                    for ii, Vdx in enumerate(E):
                        ud_aux[Vdx] = self.ud[(self.dim+1)*i + ii]

                plt.tricontourf(self.vertices[:, 0], self.vertices[:, 1], self.elements, self.vertexLabels)
                plt.triplot(self.vertices[:, 0], self.vertices[:, 1], self.elements, lw=.1, color='white', alpha=.3)
                #plt.scatter(self.vertices[self.omega, 0], self.vertices[self.omega, 1], c = "black", s=.2, alpha=.7)
                if pp is None:
                    plt.show()
                else:
                    plt.savefig(pp, format='pdf')
                    plt.close()
            else:
                plt.tricontourf(self.vertices[:, 0], self.vertices[:, 1], self.elements, self.vertexLabels)
                plt.triplot(self.vertices[:, 0], self.vertices[:, 1], self.elements, lw=.1, color='white', alpha=.3)
                #plt.scatter(self.vertices[self.omega, 0], self.vertices[self.omega, 1], c = "black", s=.2, alpha=.7)
                if pp is None:
                    plt.show()
                else:
                    plt.savefig(pp, format='pdf')
                    plt.close()
    def plot_u_exact(self, pp=None):
        if self.dim == 2:
            if self.is_DiscontinuousGalerkin:
                ud_aux = np.zeros(self.nV)
                for i, E in enumerate(self.elements):
                    for ii, Vdx in enumerate(E):
                        ud_aux[Vdx] = self.u_exact[3*i + ii]

                plt.tricontourf(self.vertices[:, 0], self.vertices[:, 1], self.elements, ud_aux)
                plt.triplot(self.vertices[:, 0], self.vertices[:, 1], self.elements, lw=.1, color='white', alpha=.3)
                #plt.scatter(self.vertices[self.omega, 0], self.vertices[self.omega, 1], c = "black", s=.2, alpha=.7)
                if pp is None:
                    plt.show()
                else:
                    plt.savefig(pp, format='pdf')
                    plt.close()
            else:
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

def plotRegMesh():
    n = 12

    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages("results/RegMeshes.pdf")

    def baryCenter(E):
        return np.sum(E, axis=0)/3
    mesh1 = RegMesh2D(0.1, n, variant="up")
    mesh2 = RegMesh2D(0.1, n, variant="zigzag")

    bC1 = np.array([baryCenter(mesh1.vertices[Vdx]) for Vdx in mesh1.elements])
    bC2 = np.array([baryCenter(mesh2.vertices[Vdx]) for Vdx in mesh2.elements])

    bCdx = 30
    circle1 = plt.Circle(bC1[bCdx], 0.1, fill=None)
    circle2 = plt.Circle(bC1[bCdx], 0.1, fill=None)

    fig, ax = plt.subplots()

    ax1 = plt.subplot("221")
    plt.ylim((.25, .75))
    plt.xlim((.0, .5))
    plt.scatter(bC1[bCdx, 0], bC1[bCdx, 1], c="black", alpha=.4)

    plt.scatter(bC1[:, 0], bC1[:, 1], alpha = .7, marker = "x", c = "red")
    plt.triplot(mesh1.vertices[:,0], mesh1.vertices[:,1], mesh1.elements, c = "red", alpha = .5)
    ax1.add_artist(circle1)

    ax2 = plt.subplot("223")
    plt.ylim((.25, .75))
    plt.xlim((.0, .5))
    plt.scatter(bC1[bCdx, 0], bC1[bCdx, 1], c="black", alpha=.4)

    plt.scatter(bC1[bCdx, 0], bC1[bCdx, 1], c="black", alpha=.4)
    plt.scatter(bC2[:, 0], bC2[:, 1], alpha = .7, marker = "o", c = "blue")
    plt.triplot(mesh2.vertices[:,0], mesh2.vertices[:,1], mesh2.elements, c = "blue", alpha = .5)

    ax2.add_artist(circle2)

    pp.savefig()
    plt.close()
    pp.close()