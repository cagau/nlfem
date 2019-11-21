#-*- coding:utf-8 -*-
import numpy as np


class dummyMesh:
    def __init__(self, nE, nE_Omega, nV, nV_Omega, vertices, triangles, ansatz="CG", boundaryConditionType="Dirichlet"):

        self.vertices = vertices
        self.triangles = triangles
        self.nE = nE
        self.nE_Omega = nE_Omega
        self.nV = nV
        self.nV_Omega = nV_Omega

        if ansatz=="CG":
            self.K = self.nV
            self.K_Omega = self.nV_Omega
            self.ansatz = "CG"
        else:
            self.K = self.nE*3
            self.K_Omega = self.nE_Omega*3
            self.ansatz = "DG"
        self.boundaryConditionType = boundaryConditionType
class Mesh:
    """ **Mesh Class**

    Let :math:`K` be the number of basis functions and :math:`J` the number of finite elements. The ordering of the vertices
    V is such that the first :math:`K_{\Omega}` vertices lie in the interior of :math:`\Omega`.

    :ivar vertices: nd.array, real, shape (K, 2) List of vertices in the 2D plane
    :ivar triangles: nd.array, int, shape (J, 3) List of indices mapping an index Tdx to 3 corresponding vertices of V.
    :ivar K: Number of basis functions.
    :ivar K_Omega: Number if basis functions in the interior of :math:`\Omega`.
    :ivar nE: Number of finite elements :math:`\Omega`.
    :ivar nE_Omega: Number of finite elements in
    """
    def __init__(self, mesh_data, ansatz, boundaryConditionType="Dirichlet"):
        """Constructor

        Executes read_mesh and prepare.
        """
        if type(mesh_data) == str:
            args = self.mesh(*self.read_mesh(mesh_data))
        else:
            args = []
            args.append(mesh_data.vertices)
            args.append(mesh_data.triangles)
            args.append(mesh_data.nE)
            args.append(mesh_data.nE_Omega)
            args.append(mesh_data.nV)
            args.append(mesh_data.nV_Omega)

        # args = Verts, Triangles, J, J_Omega, L, L_Omega
        self.vertices = args[0]
        self.triangles = args[1]#[:, 1:]
        self.nE = args[2]
        self.nE_Omega = args[3]
        self.nV = args[4]
        self.nV_Omega = args[5]

        args = self.basis(ansatz=ansatz)
        self.K = args[0]
        self.K_Omega = args[1]

        self.ansatz = ansatz
        self.boundaryConditionType = boundaryConditionType
    def get_state_dict(self):
        return {"Verts": self.vertices, "Triangles": self.triangles, "J":self.nE, "J_Omega":self.nE_Omega,
                "L":self.nV, "L_Omega":self.nV_Omega, "K":self.K, "K_Omega":self.K_Omega}
    def read_mesh(self, mshfile):
        """meshfile = .msh - file genrated by gmsh


        :param mshfile:
        :return: Verts, Lines, Triangles
        """

        fid = open(mshfile, "r")

        for line in fid:

            if line.find('$Nodes') == 0:
                # falls in der Zeile 'Nodes' steht, dann steht in der...
                line = fid.readline()  # ...naechsten Zeile...
                npts = int(line.split()[0])  # ..die anzahl an nodes

                Verts = np.zeros((npts, 3), dtype=float)  # lege array for nodes an anzahl x dim

                for i in range(0, npts):
                    # run through all nodes
                    line = fid.readline()  # put current line to be the one next
                    data = line.split()  # split line into its atomic characters
                    Verts[i, :] = list(map(float, data[
                                                  1:]))  # read out the coordinates of the node by applying the function float() to the characters in data

            if line.find('$Elements') == 0:
                line = fid.readline()
                nelmts = int(line.split()[0])  # number of elements

                Lines = []
                Triangles = []
                # Squares = np.array([])

                for i in range(0, nelmts):
                    line = fid.readline()
                    data = line.split()
                    if int(data[1]) == 1:
                        """ 
                        we store [physical group, node1, node2, node3], 
                        -1 comes from python starting to count from 0
                        """
                        # see ordering:

                        #                   0----------1 --> x

                        Lines += [int(data[3]), int(data[-2]) - 1, int(data[-1]) - 1]

                    if int(data[1]) == 2:
                        """
                        we store [physical group, node1, node2, node3]
                        """
                        # see ordering:

                        #                    y
                        #                    ^
                        #                    |
                        #                    2
                        #                    |`\
                        #                    |  `\
                        #                    |    `\
                        #                    |      `\
                        #                    |        `\
                        #                    0----------1 --> x

                        Triangles += [int(data[3]), int(int(data[-3]) - 1), int(int(data[-2]) - 1),
                                      int(int(data[-1]) - 1)]

        Triangles = np.array(Triangles).reshape(-1, 4)
        Lines = np.array(Lines).reshape(-1, 3)

        return Verts, Lines, Triangles

    def mesh(self, Verts, Lines, Triangles):
        """Prepare mesh from Verts, Lines and Triangles.

        :param Verts: List of Vertices
        :param Lines: List of Lines
        :param Triangles: List of Triangles
        :return: Verts, Triangles, K, K_Omega, J, J_Omega
        """
        # Sortiere Triangles so, das die Omega-Dreieck am Anfang des Array liegen --------------------------------------
        Triangles = Triangles[Triangles[:, 0].argsort()]

        # Sortiere die Verts, sodass die Indizes der Nodes in Omega am Anfang des Arrays Verts liegen ------------------
        Verts = Verts[:, :2]  # Wir machen 2D, deshalb ist eine Spalte hier unnütz.
        # T heißt Triangle, dx index
        # Das größte label bezeichnet den nichtlokalen Rand
        elementlabelOmegaI = max(Triangles[:, 0])
        Tdx_Omega = np.where(Triangles[:, 0] < elementlabelOmegaI)
        # V heißt Vertex, is bedeutet er nimmt die Kategorialen Werte 0,1,2 an.
        Vord_Omega = np.array([2] * len(Verts), dtype=np.int)

        # Wähle die Indizes heraus, die an Dreiecken in Omega.
        Vdx_inOmega = np.unique(Triangles[Tdx_Omega][1:].flatten())
        Vord_Omega[Vdx_inOmega] = 0  # Sie werden auf 0 gesetzt.
        linelabelOmegaI = max(Lines[:,0])
        Vdx_Boundary = np.unique(Lines[np.where(Lines[:, 0] == linelabelOmegaI)][:, 1:])
        Vord_Omega[Vdx_Boundary] = 1  # Die Punkte auf dem Rand allerdings werden auf 1 gesetzt.

        piVdx_argsort = np.argsort(Vord_Omega, kind="mergesort")  # Permutation der der Vertex indizes

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

        piVdx_invargsort = invert_permutation(piVdx_argsort)
        piVdx = lambda dx: piVdx_invargsort[dx]  # Permutation definieren

        # Wende die Permutation auf Verts, Lines und Triangles an
        Verts = Verts[piVdx_argsort]

        Triangles[:, 1:] = piVdx(Triangles[:, 1:])
        Lines[:, 1:] = piVdx(Lines[:, 1:])

        ## TEST PLOT ###
        # plt.scatter(Verts.T[0], Verts.T[1])
        # plt.scatter(Verts.T[0, :K_Omega], Verts.T[1, :K_Omega])
        # plt.show()

        # Setze J_Omega und J
        # Das ist die Anzahl der Dreiecke. Diese Zahlen sind für die Schleifendurchläufe wichtig.
        J_Omega = np.sum(Triangles[:, 0] < elementlabelOmegaI)
        J = len(Triangles)

        ## Setze L_Omega und L
        ## Das ist die Anzahl der Knotenpunkte.
        L_Omega = np.sum(Vord_Omega == 0)
        # L_dOmega = np.sum(Vis_inOmega == 1)
        L = len(Verts)
        # Im Falle von "CG" ist die Anzahl der Knotenpunkte gleich der Anzahl der Basisfunktionen.

        ## TEST PLOT ###
        # V = Verts[Triangles[:J_Omega, 1:]]
        # plt.scatter(Verts.T[0], Verts.T[1])
        # for v in V:
        #    plt.scatter(v.T[0], v.T[1], c="r")
        # plt.show()
        return Verts, Triangles, J, J_Omega, L, L_Omega

    def basis(self, ansatz):

        if ansatz == "CG":
            ## Setze K_Omega und K
            ## Das ist die Anzahl der finiten Elemente (in Omega und insgesamt).
            ## Diese Zahlen dienen als Dimensionen für die diskreten Matrizen und Vektoren.
            K_Omega = self.nV_Omega
            #K_dOmega = np.sum(Vis_inOmega == 1)
            K = self.nV

        elif ansatz == "DG":
            # Im Falle der DG-Methode is die Anzahl der Basisfunktionen 3-Mal die Anzahl der Dreiecke.
            K_Omega = self.nE_Omega * 3
            #K_dOmega = np.sum(Vis_inOmega == 1)
            K = self.nE * 3

        else:
            raise ValueError("in Mesh.basis(). No valid method (str) chosen.")

        return K, K_Omega