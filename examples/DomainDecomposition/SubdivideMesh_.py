import numpy as np
import os
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------------------------------------


###### INPUT
geofile = "DD_simple" # .geo file
element_size = 0.05 # to control grid size via gmsh (element size factor)
delta = 0.1 # interaction horizon (attention: here only l2-norm)
#-----------------------------------------------------------------------------------------------------------------------


""" HELPER FUNCTIONS """
##### READ .msh file
def read_mesh(mshfile):
    """meshfile = .msh - file genrated by gmsh """
    fid = open(mshfile, "r")
    for line in fid:
        if line.find('$Nodes') == 0:
            # falls in der Zeile 'Nodes' steht, dann steht in der...
            line = fid.readline()  #...naechsten Zeile...
            npts = int(line.split()[0]) #..die anzahl an nodes
            Verts = np.zeros((npts, 3), dtype=float) #lege array for nodes an anzahl x dim
            for i in range(0, npts):
                # run through all nodes
                line = fid.readline() # put current line to be the one next
                data = line.split() # split line into its atomic characters
                Verts[i, :] = list(map(float, data[1:])) # read out the coordinates of the node by applying the function float() to the characters in data
        if line.find('$Elements') == 0:
            line = fid.readline()
            nelmts = int(line.split()[0]) # number of elements
            Lines = []
            Triangles = []
            for i in range(0, nelmts):
                line = fid.readline()
                data = line.split()
                if int(data[1]) == 1:
                    Lines += [int(data[3]), int(data[-2])-1, int(data[-1])-1]
                if int(data[1]) == 2:
                    Triangles += [int(data[3]), int(int(data[-3])-1), int(int(data[-2])-1), int(int(data[-1])-1)]
    return Verts, np.array(Lines).reshape(int(len(Lines)/3), 3), np.array(Triangles).reshape(int(len(Triangles)/4), 4)
#-----------------------------------------------------------------------------------------------------------------------
##### GMSH FILE adaption with correct interaction horizon for outer boundary
# CONVENTION: first line of geo file is "delta = x.xx;"
def geofile_correction(geofile, delta):
    textfile = open('mesh/'+geofile+'.geo', 'r') # load current .geo file
    data = textfile.readlines() # read all lines

    tmpfile = open('mesh/test.txt', 'w+') # build a temporary file
    tmpfile.write('delta = ' + str(delta) + ';\n') # write delta in first line

    for line in data[1:]: # copy lines 2. - n. from original geo file in temp file
        tmpfile.write(line)
    tmpfile.close() # done

    os.system('rm mesh/'+geofile+'.geo') # remove old geo file
    current_path = os.path.dirname(os.path.abspath(__file__)) # get current path
    os.system('mv '+current_path+'/mesh/test.txt ' + current_path +'/mesh/'+geofile+'.geo') # substitute old geo with temp file
#-----------------------------------------------------------------------------------------------------------------------
#### PLOT ROUTINE TO TEST a MESH
def PlotMesh(elements, vertices, subdomainLabels, title = "Mesh"):
    import matplotlib.pyplot as plt
    colors = ['b', 'k','r','g',  'c', 'm', 'y',  'w']
    plt.figure(title)
    labels = list(np.unique(subdomainLabels))
    for i in range(len(labels)):
        elements_i = elements[np.where(subdomainLabels==labels[i])]
        for k in range(len(elements_i)):
            plt.gca().add_patch(plt.Polygon(vertices[elements_i[k]], closed=True, fill=False, color=colors[i],alpha=1))
    plt.show()
    plt.axis('equal')
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

### GENERATE MOTHER MESH

def mesh_data(geofile, element_size, delta):


    ### Correct the geo file with correct delta
    geofile_correction(geofile, delta)

    ##### RUN GMSH and READ .msh file
    # to control grid size we use <element_size>
    # we sort from smallest to largest label
    # CONVENTION: largest label (if numbers) = interaction domain Omega_I
    # TO DO: labels should better be dictionary?!
    os.system('gmsh -v 0 mesh/'+geofile+'.geo -2 -clscale '+str(element_size)+' -o mesh/'+geofile+'.msh') # run gmsh
    vertices, lines, elements = read_mesh('mesh/' + geofile + '.msh') # read .msh file
    elements = elements[elements[:, 0].argsort()] # smallest to largest label (Convention?!)
    labels = np.sort(np.unique(elements[:, 0])) # smallest to largest label (Convention?!)
    vertices = vertices[:, 0:2] # we only need two dimensions
    #-----------------------------------------------------------------------------------------------------------------------

    ##### SORT VERTICES and ADAPT ELEMENTS (so that vertices in Omega_I are at the end)
    # CONVENTION: largest label (if numbers) = interaction domain Omega_I
    IntDomLabel = labels[-1]
    aux = elements[np.where(elements[:, 0] == labels[-1])]
    aux = np.unique(aux[:,1:].reshape(3 * len(aux))) # = indices of vertices which lie in interaction domain
    # Plot to test
    # plt.plot(vertices[aux][:,0], vertices[aux][:,1], 'ro' )
    # plt.plot(vertices[list(set(np.arange(len(vertices))).difference(set(aux)))][:,0], vertices[list(set(np.arange(len(vertices))).difference(set(aux)))][:,1], 'bo')
    # sort indices = disjoint splitting of range(len(vertices)) with indices of interac. domain at the end
    # =  {  range(len(vertices))) \setminus (indices of interaction domain) } \cup  { (indices of interaction domain) }
    K_Omega = len(vertices) - len(aux)
    sort_indices = np.array(list(set(np.arange(len(vertices))).difference(set(aux))) + list(aux))
    vertices = vertices[sort_indices] # sort vertices from interaction domain to the end
    sort_indices_inv = np.arange(len(sort_indices))[np.argsort(sort_indices)] # that is the inverse function
    def f(n): # since ordering in vertices changed, we need to adapt <elements> and <lines> accordingly
        return sort_indices_inv[n]
    elements[:, 1:] = f(elements[:, 1:]) # adapt elements
    lines[:,1:] = f(lines[:,1:]) # adapt lines
    #-------------------------------------------------------------------------------------------------------------------


    # COMPUTE DIAMETER
    def diam(T):
        length_of_edges = np.array(
            [np.linalg.norm(T[0] - T[1]), np.linalg.norm(T[0] - T[2]), np.linalg.norm(T[1] - T[2])])
        return np.max(length_of_edges)

    diameter = [diam(np.array([vertices[elements[i,1]], vertices[elements[i,2]], vertices[elements[i,3]]])) for i in range(len(elements))]
    diam = np.max(diameter)



    # for i in Gamma_hat:
    #     plt.gca().add_patch(plt.Polygon(vertices[elements[i, 1:]], closed=True, fill=False, color='olive', alpha=1))
    # plt.show()


    #### SINGEL DOMAIN mesh (= mother mesh)
    # Gamma_hat = set above
    # neighbours = set by John
    elementLabels = elements[:,0] # first column in <elements> -> to be changed
    subdomainLabels = elements[:,0].copy() # first column in <elements>
    elementLabels[np.where(elementLabels != labels[-1])] = 1 # <set all labels to 1 except for interaction domain>
    elements = elements[:,1:] # delete first column in <elements>
    # vertices = set above
    # lines = set above
    # Gamma_hat
    Gamma_hat = []
    bary = (vertices[elements[:, 0]] + vertices[elements[:, 1]] + vertices[elements[:, 2]]) / 3.
    interface = np.unique(lines[np.where(lines[:, 0] == 12)][:, 1:]).tolist()
    for i in interface:
        Gamma_hat += np.where(np.linalg.norm(bary - np.tile(vertices[i], (len(bary), 1)), axis=1) <= delta / 2. + diam)[0].tolist()
    Gamma_hat = list(set(Gamma_hat).difference(set(np.where(subdomainLabels == labels[-1])[0].tolist())))

    mesh_dict = {
        "K": len(vertices),
        "K_Omega": K_Omega,
        "nE": len(elements),
        "nE_Omega": len(np.where(elementLabels != labels[-1])[0]),# number of elements with label 1
        "nV": len(vertices),
        "nV_Omega": K_Omega, # since CG
        "dim": 2,
        "is_DiscontinuousGalerkin": False,
        "is_NeumannBoundary": False,
        "vertices": vertices,
        "elements": elements,
        "elementLabels": elementLabels,
        "Gamma_hat": np.array(Gamma_hat),
        "subdomainLabels": subdomainLabels
    }

    return elements, vertices, lines, elementLabels, subdomainLabels, K_Omega, diam, Gamma_hat, mesh_dict
#-----------------------------------------------------------------------------------------------------------------------



# GENERATE CHILD MESHES

def submesh_data(elements, vertices, lines, subdomainLabels, diam):

    # COMPUTE BARYCENTERS
    bary = (vertices[elements[:, 0]] + vertices[elements[:, 1]] + vertices[elements[:, 2]]) / 3.
    # ------------------------------------------------------------------------------------------------------------------

    labels = list(np.unique(subdomainLabels)) # unique list of all labels used except for the interaction domain
    submeshes_data = [] # we collect all submesh_data as a list in the list submeshes_data
    submesh_dicts = []
    for k in range(len(labels)-1): # run through all labels
        ## compute the interaction domain of the subdomain
        boundary_i = np.unique(lines[np.where(lines[:, 0] == 11 * (k + 1))][:, 1:]).tolist() # all vertices on the boundary of subdomain label[k]
        interface = np.unique(lines[np.where(lines[:, 0] == 12)][:, 1:]).tolist()

        IntDomain_i = [] # to be filled with all triangles in the approximated interaction domain of the subdomain
        for l in boundary_i: # run trough all vertices on the boundary
            IntDomain_i += np.where(np.linalg.norm(bary - np.tile(vertices[l], (len(bary), 1)), axis =1) <= delta + diam)[0].tolist()
        for i in interface:
            IntDomain_i += np.where(np.linalg.norm(bary - np.tile(vertices[i], (len(bary), 1)), axis=1) <= delta / 2. + diam)[0].tolist()
        IntDomain_i = list(np.unique(np.array(IntDomain_i))) # clearly this list is not unique, since neighboring vertices hit the same elements multiple times

        # mark all mother elements True which are in the subdomain \cup its interaction domain
        elements_i_bool = np.zeros(len(elements), dtype = bool) # array of bools as long as mother elements
        elements_i_bool[np.where(subdomainLabels == labels[k])] = True # mark all elements True which are in subdomain
        elements_i_bool[IntDomain_i] = True # mark all elements True which are in the interaction domain of the subdomain
        elements_i = elements[elements_i_bool] # child elements (note: still contains mother indices)

        subdomainLabels_i = subdomainLabels[elements_i_bool]

        # embedding elements
        aux_elements = - np.ones(len(elements), dtype=int) # array of -1 as long as mother elements (note: objects in this list must not be used for indexing, i.e., -1!)
        aux_elements[np.where(elements_i_bool)[0]] = np.arange(len(elements_i)) # number the Trues in <elements_i_bool> from 0 to ...
        embedding_elements_i = np.array([list(aux_elements).index(i) for i in range(len(elements_i))]) # array as long as child elements
        # note: embedding_elements_i[k] gives you the mother element index for child element index k

        # compute child element_labels (i.e., set all child elements to 1 which are not in child interaction domain
        elements_labels_i = np.zeros(len(elements_i), dtype = bool) # array of bools as long as child elements
        elements_labels_i[np.where(subdomainLabels_i != labels[-1])] = True # mark all elements True which are in subdomain

        # compute child vertices
        vertices_i_bool = np.zeros(len(vertices), dtype = bool) # array of bools as long as mother vertices
        aux = np.unique(elements_i.reshape(3*len(elements_i))) # all mother indices of child vertices
        vertices_i_bool[aux] = True # mark all mother vertices True which are in child vertices
        vertices_i = vertices[vertices_i_bool] # child bertices

        # embedding elements vertices
        aux_vertices = - np.ones(len(vertices), dtype=int) # array of -1 as long as mother elements (note: objects in this list must not be used for indexing, i.e., -1!)
        aux_vertices[np.where(vertices_i_bool)[0]] = np.arange(len(vertices_i)) # number the Trues in <vertices_i_bool> from 0 to ...
        embedding_vertices_i = np.array([list(aux_vertices).index(i) for i in range(len(vertices_i))]) # array as long as child vertices
        # note: embedding_vertices_i[k] gives you the mother vertices index for child vertices index k

        # adapt indices in elements_i (still mother indices), so that they fit to child vertices vertices_i
        def large_to_small_1(i):
            return aux_vertices[i]
        elements_i = large_to_small_1(elements_i)

        # SUBMESH ------------------------------------------------------------------------------------------------------
        # neighbours_i = ...
        #elements_labels_i = set above
        #subdomainLabels_i = set above
        #elements_labels_i = set above
        # vertices = set above
        #K_i = len(vertices_i)
        aux = elements_i[np.where(subdomainLabels_i == labels[-1])]
        aux = np.unique(aux.reshape(3 * len(aux)))  # = indices of vertices which lie in child interaction domain
        K_Omega_i = len(vertices_i) - len(aux)
        #nE_i = len(elements[elements_i]) # number of child elements
        #nE_Omega_i = len(np.where(elements_labels_i != labels[-1])[0])  # number of elements with label 1
        #nV_i = len(vertices_i)
        #nV_Omega_i = K_Omega_i  # since CG
        #dim = 2
        #is_DiscontinuosGalerkin = False
        #is_NeumannBoundary = False
        # -----------------------------------------------------------------------------------------------------------

        ##### SORT VERTICES and ADAPT ELEMENTS (so that vertices in child Omega_I are at the end)
        # this is necessary, since former boundary points may become inner points for floating parts
        # CONVENTION: largest label (if numbers) = interaction domain Omega_I
        sort_indices = np.array(list(set(np.arange(len(vertices_i))).difference(set(aux))) + list(aux))
        vertices_i = vertices_i[sort_indices]  # sort vertices from interaction domain to the end
        sort_indices_inv = np.arange(len(sort_indices))[np.argsort(sort_indices)]  # that is the inverse function
        def f(n):  # since ordering in vertices changed, we need to adapt <elements> and <lines> accordingly
            return sort_indices_inv[n]
        elements_i = f(elements_i)  # adapt elements
        embedding_vertices_i = embedding_vertices_i[sort_indices]

        # -------------------------------------------------------------------------------------------------------------------
        submesh_k = [elements_i, vertices_i, elements_labels_i, subdomainLabels_i, embedding_vertices_i, embedding_elements_i, K_Omega_i]
        submeshes_data += [submesh_k]

        submesh_dicts.append({
            "K": len(vertices_i),
            "K_Omega": K_Omega_i,
            "nE": len(elements_i),
            "nE_Omega": np.sum(elements_labels_i),# number of elements with label 1
            "nV": len(vertices_i),
            "nV_Omega": K_Omega_i, # since CG
            "dim": 2,
            "is_DiscontinuousGalerkin": False,
            "is_NeumannBoundary": False,
            "vertices": vertices_i.copy(),
            "elements": elements_i.copy(),
            "elementLabels": np.array(elements_labels_i, dtype=np.int).copy(),
            "subdomainLabels": subdomainLabels_i.copy(),
            "embedding_vertices": embedding_vertices_i.copy(),
            "embedding_elements": embedding_elements_i.copy()
        })

    return submeshes_data, submesh_dicts
#-----------------------------------------------------------------------------------------------------------------------




if __name__=="__main__":
    ### PLOT the meshes
    # mother mesh
    elements, vertices, lines, elementLabels, subdomainLabels, K_Omega, diam, Gamma_hat, mesh_dict = mesh_data(geofile, element_size, delta)
    PlotMesh(elements, vertices, subdomainLabels, title = "Mother Mesh")

    # plot Gamma_hat
    for i in Gamma_hat:
        plt.gca().add_patch(plt.Polygon(vertices[elements[i]], closed=True, fill=True, color='yellow', alpha=0.25))


    # plot vertices
    # plt.plot(vertices[0:K_Omega, 0], vertices[0:K_Omega, 1], 'ro')
    # plt.plot(vertices[K_Omega:, 0], vertices[K_Omega:, 1], 'bo')

    # test elements embedding
    # for i in range(len(elements)):
    #     bary_i = vertices[elements[i]].sum(axis=0) / 3.
    #     plt.annotate(str(i), (bary_i[0], bary_i[1]), size=10)

    # test vertices embedding
    # for i in range(len(vertices)):
    #     plt.annotate(str(i), (vertices[i,0], vertices[i,1]), size=10)




    # submeshes
    labels = list(np.unique(subdomainLabels))
    submeshes_data, submeshes_dict = submesh_data(elements, vertices, lines, subdomainLabels, diam)
    for k in range(len(labels)-1):
        submesh = submeshes_data[k]
        elements_i = submesh[0]
        vertices_i = submesh[1]
        element_Labels_i = submesh[2]
        subdomainLabels_i = submesh[3]
        embedding_vertices_i = submesh[4]
        embedding_elements_i = submesh[5]
        K_Omega_i = submesh[6]

        PlotMesh(elements_i, vertices_i, element_Labels_i, title="Child Mesh"+str(labels[k]))

        # plot vertices
        # plt.plot(vertices_i[0:K_Omega_i, 0], vertices_i[0:K_Omega_i, 1], 'ro')
        # plt.plot(vertices_i[K_Omega_i:, 0], vertices_i[K_Omega_i:, 1], 'bo')

        # test elements embedding
        # for i in range(len(elements_i)):
        #     bary_i = vertices_i[elements_i[i]].sum(axis=0) / 3.
        #     plt.annotate(str(embedding_elements_i[i]), (bary_i[0], bary_i[1]), size = 10)

        # test vertices embedding
        # for i in range(len(vertices_i)):
        #     plt.annotate(str(embedding_vertices_i[i]), (vertices_i[i,0], vertices_i[i,1]), size = 10)

    #-----------------------------------------------------------------------------------------------------------------------














