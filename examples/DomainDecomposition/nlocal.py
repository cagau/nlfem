#-*- coding:utf-8 -*-
"""@package docstring
Documentation for this module.
"""

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix, csr_matrix, csc_matrix, diags
import meshio
import assemble
import datetime

# Auf Triangles und Lines müssen wir die inverse Permutation anwenden.
# Der Code wäre mit np.argsort kurz und für Node-Zahl unter 1000 auch schnell, allerdings ist
# sortieren nicht in der richtigen Effizienzklasse. (Eigentlich muss ja nur eine Matrix transponiert werden)
# siehe https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy

def timestamp():
    """
    Returns current timestamp as string.

    :return: string, format %m%d_%H-%M-%S
    """
    # Link to strftime Doc
    # http://strftime.org/
    return datetime.datetime.now().strftime("%m%d_%H-%M-%S")

def invert_permutation(p):
    """
    The function inverts a given permutation.
    :param p: nd.array, shape (m,) The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    :return: nd.array, shape (m,) Returns an array s, where s[i] gives the index of i in p.
    """
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s

class MeshfromDict(meshio._mesh.Mesh):
    def __init__(self, iterable=(), **kwargs):
        self.K = None
        self.K_Omega = None
        self.nE = None
        self.nE_Omega = None
        self.nV = None
        self.nV_Omega = None
        self.dim = None
        self.is_DiscontinuousGalerkin = None
        self.is_NeumannBoundary = None
        self.vertices = None
        self.elements = None
        self.elementLabels = None
        self.subdomainLabels = None
        self.embedding_vertices = None
        self.embedding_elements = None
        self.Gamma_hat = None

        self.__dict__.update(iterable, **kwargs)
        self.neighbours = assemble.constructAdjaciencyGraph(self.elements)
        self.vertexLabels = np.ones(self.nV)
        self.vertexLabels[:self.nV_Omega] = 0
        self.nZeta = 0
        self.Zeta = None

        super(MeshfromDict, self).__init__(self.vertices, [["triangle", self.elements]],
                                           point_data={"vertexLabels": self.vertexLabels},
                                           cell_data={"elementLabels": self.elementLabels,
                                                      "subdomainLabels": self.subdomainLabels}
                                           )
    def add_u(self, u_input, name):
        u = np.zeros(self.K)
        u[:self.K_Omega] = u_input
        self.point_data[name] = u
        return u

def setZeta(mesh_list):
    print("Setup Zeta")
    # pi is the list of projections from Omega to Omegai
    pi = []
    n_submeshes = len(mesh_list) - 1
    # indicator is the step function of Omegai in Omega
    indicator = np.zeros((mesh_list[0].nE, n_submeshes), dtype=np.int)
    main = mesh_list[0]
    for k in range(n_submeshes):
        subm = mesh_list[k+1]
        pi.append(np.zeros((subm.nE, main.nE), dtype=np.int))
        for aT in range(subm.nE):
            aT_parent = subm.embedding_elements[aT]
            pi[k][aT, aT_parent] = 1
            indicator[aT_parent, k] += 1
        pi[k] = csr_matrix(pi[k])
    subsets = indicator @ indicator.T - 1
    subsets *= subsets > 0
    subsets = csc_matrix(subsets)

    for k in range(n_submeshes):
        subm = mesh_list[k+1]
        subZeta = (pi[k] @ subsets) @ pi[k].T

        # For the plots in paraview
        subm.cell_data["Zeta"] = np.array([subZeta[k, k] for k in range(subm.nE)], dtype=np.int)

        nZeta = subZeta.getnnz()
        subZeta = coo_matrix(subZeta)
        subm.Zeta = np.zeros((nZeta, 3), dtype=np.int)
        # For the assembly routine
        subm.Zeta[:, 0] = subZeta.row[:]
        subm.Zeta[:, 1] = subZeta.col[:]
        subm.Zeta[:, 2] = subZeta.data[:]
    return mesh_list


class MeshIO(meshio._mesh.Mesh):
    def __init__(self, mesh_data, **kwargs):
        print("Constructing Mesh\n")
        self.dim = 2
        # Read mesh into meshio.__mesh.Mesh Class
        mbNeighbours = None
        parentMesh = meshio.read(mesh_data + ".msh")

        # Run Constructor of Parent Class
        super(MeshIO, self).__init__(parentMesh.points, parentMesh.cells, parentMesh.point_data,
                      parentMesh.cell_data)

        # DIMENSION + Number of Elements and Verts ---------------------------------------------------------------------
        # Identify in which dimension we work in and set Elements
        elements = None
        for block in self.cells:
            if block.type == 'triangle':
                elements = np.array(block.data, dtype=np.int)
        if elements is None:
            raise ValueError("No triangles found")

        vertices = self.points[:, :self.dim]
        self.nE = elements.shape[0]
        self.nV = vertices.shape[0]
        print("In nlocal.MeshIO, Dimension assumed to be {}D.".format(self.dim))

        # Labels are hard coded in C++ Code
        interactionLabel = 2
        domainLabel = 1

        # ELEMENT LABELS + Number of Elements in Omega -----------------------------------------------------------------
        self.elementLabels = np.array(self.cell_data["gmsh:physical"][1], dtype=np.int)
        # As Omega is considered an open set
        # we label each node in the complement by the interactionLabel
        VertexLabels = np.full(self.nV, domainLabel)
        for i, label in enumerate(self.elementLabels):
            if label == interactionLabel: # label of Interaction domain
                Vdx = elements[i]
                VertexLabels[Vdx] = interactionLabel
        self.point_data["vertexLabels"] = VertexLabels.copy()

        self.nE_Omega = np.sum(self.elementLabels == domainLabel) # mesh.nE_Omega
        self.nV_Omega = np.sum(VertexLabels == domainLabel) # mesh.nV_Omega

        # VERTEX AND ELEMENT ORDER -------------------------------------------------------------------------------------
        piVdx_argsort = np.argsort(VertexLabels, kind="mergesort")  # Permutation der der Vertex indizes
        self.piVdx_argsort = piVdx_argsort
        piVdx_invargsort = invert_permutation(piVdx_argsort)
        self.piVdx_invargsort = piVdx_invargsort
        piVdx = lambda dx: piVdx_invargsort[dx]  # Permutation definieren
        # Wende die Permutation auf Verts, Lines und Triangles an
        self.vertices = vertices[piVdx_argsort] # Vorwärts Permutieren
        self.elements = piVdx(elements) # Inverse Permutation
        # reorder elements
        piTdx_argsort = np.argsort(self.elementLabels, kind="mergesort")
        self.elements = self.elements[piTdx_argsort]

        # INCLUDE CONFS boundaryCondition, ansatz + Matrix Dimension ---------------------------------------------------
        self.boundaryConditionType = kwargs["boundaryConditionType"]
        if kwargs["boundaryConditionType"] == "Dirichlet":
            self.is_NeumannBoundary = 0
        else:
            self.is_NeumannBoundary = 1
            # In case of Neumann conditions we assemble a Maitrx over Omega + OmegaI.
            # In order to achieve that we "redefine" Omega := Omega + OmegaI
            # This is rather a shortcut to make things work quickly.
            self.nE_Omega = self.nE
            self.nV_Omega = self.nV

        self.ansatz = kwargs["ansatz"]
        if kwargs["ansatz"] =="DG":
            self.K = self.nE*3
            self.K_Omega = self.nE_Omega*3
            self.is_DiscontinuousGalerkin=1

        elif kwargs["ansatz"] =="CG":
            self.K = self.nV
            self.K_Omega = self.nV_Omega
            self.is_DiscontinuousGalerkin=0
        else:
            print("Ansatz ", kwargs["ansatz"], " not provided.")
            raise ValueError

    # Setup adjaciency graph of the mesh --------------------------
        if kwargs.get("isNeighbours", True):
            self.neighbours = assemble.constructAdjaciencyGraph(self.elements)
        # built in Neighbour Routine of MeshBulder yields mbNeighbours.

        self.baryCenter = np.zeros((self.nE, self.dim))
        for i in range(self.nE):
            corners = self.vertices[self.elements[i]]
            bC = np.sum(corners, 0)/(self.dim+1)
            self.baryCenter[i] = bC

        # Zeta Test
        self.nZeta = 4
        G = np.eye(self.nZeta)
        G = coo_matrix(G)

        self.Zeta = np.zeros((self.nZeta, 3), dtype=np.int)
        self.Zeta[:, 0] = G.row[:]
        self.Zeta[:, 1] = G.col[:]
        self.Zeta[:, 2] = G.data[:]

        print("Done [Constructing Mesh]\n")