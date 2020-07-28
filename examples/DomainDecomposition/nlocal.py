#-*- coding:utf-8 -*-
"""@package docstring
Documentation for this module.
"""

import numpy as np
from scipy.sparse import coo_matrix
import meshio
import assemble

# Auf Triangles und Lines müssen wir die inverse Permutation anwenden.
# Der Code wäre mit np.argsort kurz und für Node-Zahl unter 1000 auch schnell, allerdings ist
# sortieren nicht in der richtigen Effizienzklasse. (Eigentlich muss ja nur eine Matrix transponiert werden)
# siehe https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy

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
        self.__dict__.update(iterable, **kwargs)
        self.neighbours = assemble.constructAdjaciencyGraph(self.elements)
        self.vertexLabels = np.ones(self.nV)
        self.vertexLabels[:self.nV_Omega] = 0
        self.nCeta = 0
        self.Ceta = None

        super(MeshfromDict, self).__init__(self.vertices, [["triangle", self.elements]],
                                           point_data={"vertexLabels": self.vertexLabels},
                                           cell_data={"elementLabels": self.elementLabels,
                                                      "subdomainLabels": self.subdomainLabels}
                                           )

def setCeta(mesh_list):
    projection_list = []
    submesh_map = np.zeros((mesh_list[0].nE, 2), dtype=np.int)
    n_submeshes = len(mesh_list) -1
    for k in range(n_submeshes):
        subm = mesh_list[k+1]
        projection_list.append(-np.ones(mesh_list[0].nE, dtype=np.int))
        for aT in range(subm.nE):
            parent_aT = subm.embedding_elements[aT]
            projection_list[k][parent_aT] = aT
            submesh_map[parent_aT, k] += 1
    subsets = submesh_map @ submesh_map.T-1
    subsets *= subsets > 0
    G = coo_matrix(subsets)

    nCeta = G.nnz
    Ceta =  np.zeros((nCeta, 3), dtype=np.int)
    Ceta[:, 0] = G.row[:]
    Ceta[:, 1] = G.col[:]
    Ceta[:, 2] = G.data[:]
    Ceta_list = [Ceta]

    for k in range(n_submeshes):
        isOmegak = submesh_map[:, k]
        subm = mesh_list[k+1]
        subm.nCeta = Ceta.shape[0]
        subm.Ceta = Ceta.copy()
        subm.Ceta[:, 0] = projection_list[k][Ceta[:, 0]]
        subm.Ceta[:, 1] = projection_list[k][Ceta[:, 1]]
        subm.cell_data["Ceta"] = np.zeros(subm.nE, dtype=np.int)
        subm.cell_data["Ceta"][subm.Ceta[:, 0]] = 1

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

        # Ceta Test
        self.nCeta = 4
        G = np.eye(self.nCeta)
        G = coo_matrix(G)

        self.Ceta = np.zeros((self.nCeta, 3), dtype=np.int)
        self.Ceta[:, 0] = G.row[:]
        self.Ceta[:, 1] = G.col[:]
        self.Ceta[:, 2] = G.data[:]

        print("Done [Constructing Mesh]\n")