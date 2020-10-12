from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from nlfem import constructAdjaciencyGraph

class RegMesh1D:
    def __init__(self, a,b, delta, h, interface, ufunc = None, is_constructAdjaciencyGraph=True, coarseMesh=None):

        # VERTICES
        # vertices as concatenation of np.linspace
        self.h = h
        self.delta = delta

        N_left, N_Omega_1, N_Omega_2, N_right = int(delta/h)+1, int((interface-a)/h)+1, int((b-interface)/h)+1, int(delta/h)+1
        OmegaI_left, h_left = np.linspace(a-delta,a, N_left, retstep = True)
        Omega_1, h_omega_1 = np.linspace(a, interface, N_Omega_1, retstep = True)
        Omega_2, h_omega_2 = np.linspace(interface,b, N_Omega_2, retstep = True)
        OmegaI_right, h_right = np.linspace(b,b+delta, N_right, retstep = True)
        self.dim = 1

        self.vertices = np.concatenate((OmegaI_left[:-1],Omega_1[:-1], Omega_2[:-1],OmegaI_right ))
        # ELEMENTS
        # elements are the corresponding intervals
        self.elements = []
        # Omega_1
        for i in range(N_left-1, N_left+N_Omega_1-2):
            self.elements += [[1, i, i+1]]
        # Omega_2
        for i in range(N_left+N_Omega_1-2, N_left+N_Omega_1+N_Omega_2-3):
            self.elements += [[1, i, i+1]]
        # Omega_I_left
        for i in range(0, N_left-1):
            self.elements += [[2, i, i+1]]
        # Omega_I_right
        for i in range(N_left+N_Omega_1+N_Omega_2-3, N_left+N_Omega_1+N_Omega_2+N_right-4):
            self.elements += [[2, i, i+1]]
        self.elementLabels = (np.array(self.elements, dtype = int)[:, 0]).copy()
        self.elements = (np.array(self.elements, dtype = int)[:, 1:]).copy()

        # SORT VERTICES
        # (Omega_1, interface, Omega_2, boundary, Omega)
        verts_sort = list(range(N_left, N_left+N_Omega_1-2)) \
                     + [N_left + N_Omega_1-2] \
                     +list(range(N_left+N_Omega_1-1, N_left+N_Omega_1+N_Omega_2-3)) \
                     + [N_left-1, N_left+N_Omega_1+N_Omega_2-3] \
                     +list(range(0, N_left-1)) \
                     +list(range(N_left+N_Omega_1+N_Omega_2-2, N_left+N_Omega_1+N_Omega_2+N_right-3))
        self.vertices = self.vertices[verts_sort]
        verts_sort = np.array(verts_sort)
        verts_sort_inv = np.arange(len(verts_sort))[np.argsort(verts_sort)]
        def vert_sort_func(i):
            return verts_sort_inv[i]
        self.elements = vert_sort_func(self.elements)

        # DEFINE NUMBERS
        self.nV, self.nV_Omega, nV_Omega_1, nV_interface, nV_Omega_2, nV_Omega_boundary  = len(self.vertices), N_Omega_1+N_Omega_2-3 , N_Omega_1-2,1, N_Omega_1-2, 2
        self.nE, self.nE_Omega, nE_Omega_1, nE_Omega_2 = self.nV - 1, N_Omega_1+N_Omega_2-2 , N_Omega_1-1, N_Omega_1-1

        self.K = self.nV
        self.nV_Omega = nV_Omega_1 + nV_Omega_2
        self.K_Omega = self.nV_Omega
        self.is_DiscontinuousGalerkin = False
        self.is_NeumannBoundary = False
        self.diam = np.max([h_left,h_omega_1, h_omega_2, h_right])
        self.neighbours = None
        if is_constructAdjaciencyGraph:
            self.neighbours = constructAdjaciencyGraph(self.elements)
        if coarseMesh is not None:
            #self.interpolator = LinearNDInterpolator(coarseMesh.vertices, coarseMesh.ud)
            self.interpolator = interp1d(coarseMesh.vertices, coarseMesh.ud, fill_value="extrapolate")
            self.ud = self.interpolator(self.vertices)
        if ufunc is not None:
            if hasattr(ufunc, '__call__'):
                self.set_u_exact(ufunc)

    def plot(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.vertices, np.zeros(self.nV))

        plt.show()

    def set_u_exact(self, ufunc):
        self.u_exact = np.zeros(self.vertices.shape[0])
        for i, x in enumerate(self.vertices):
            self.u_exact[i] = ufunc(x)

    def write_ud(self, udata, ufunc):
        self.ud = np.zeros(self.vertices.shape[0])
        for i, x in enumerate(self.vertices):
            self.ud[i] = ufunc(x)
        self.ud[:self.K_Omega] = udata

    def plot_ud(self, pp=None):
        args = np.argsort(self.vertices)
        plt.plot(self.vertices[args], self.ud[args])
        plt.scatter(self.vertices[args], self.ud[args])

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


if __name__ == "__main__":
    a, b = -1, 1
    delta = 0.1
    h = 0.1
    interface = 0
    mesh = RegMesh1D(a,b, delta, h, interface)
    mesh.plot()