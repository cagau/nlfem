import numpy as np

def read_mesh(mshfile):
    """meshfile = .msh - file genrated by gmsh


    :param mshfile:
    :return: Verts, Lines, Triangles
    """

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
            #Squares = np.array([])
            
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
                    
                    Lines += [int(data[3]), int(data[-2])-1, int(data[-1])-1]
                    
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
                                
                    Triangles += [int(data[3]), int(int(data[-3])-1), int(int(data[-2])-1), int(int(data[-1])-1)]

    Triangles = np.array(Triangles).reshape(-1, 4)
    Lines = np.array(Lines).reshape(-1, 3)

    return Verts, Lines, Triangles


