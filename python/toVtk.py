# In order to find the scripts in the python/ directory
# we add the current project path == working directory to sys-path
import sys
sys.path.append(".")

import meshio
from python.nlocal import MeshIO
from python.conf import *
import glob

confDict = {"domainPhysicalName": 1, "boundaryPhysicalName": 9, "interactiondomainPhysicalName": 2,
            "boundaryConditionType": boundaryConditionType, "ansatz": ansatz, "isNeighbours": False}

def readRHS():
    import matplotlib.pyplot as plt
    import numpy as np


    f = open("fd_Assembly11", "r")
    f.readline()
    size = int(f.readline().split()[0])
    fd = []

    for i in range(size):
        fd.append(f.readline().split())
    fd = [float(s[0]) for s in fd]
    plt.plot(fd, c="r", alpha=.3)
    f.close()

    f = open("fd_Mass11", "r")
    f.readline()
    size = int(f.readline().split()[0])
    Mfd = []
    for i in range(size):
        Mfd.append(f.readline().split())
    f.close()
    Mfd = [float(s[0]) for s in Mfd]
    plt.plot(Mfd, c="b", alpha=.3)

    plt.show()

def showAd():
    f = open("output/Ad_O.txt", "r")
    print(f.readline())
    size = f.readline().split()
    size = [int(s) for s in np.array(size)]
    print("Size ", size)
    Ad_O = np.zeros(size)
    for k,line in enumerate(f):
        Ad_O[k] = line.split()
    f.close()
    print(Ad_O)

    f = open("output/fd.txt", "r")
    print(f.readline())
    size = f.readline().split()
    size = [int(s) for s in np.array(size)]
    print("Size ", size)
    fd = np.zeros(size)
    for k,line in enumerate(f):
        fd[k] = line.split()
    f.close()
    print(fd)

def convert():
    for mesh_name_ in glob.glob("data/*vrt"):
        mesh_name_ = mesh_name_[5:]
        print(mesh_name_)
        mesh = MeshIO(DATA_PATH + mesh_name_, **confDict)
        mesh.point_data["ud"] = mesh.point_data["u"]
        meshio.write(OUTPUT_PATH + mesh_name_ + ".vtk", mesh)

        # Check Neighbour Routine -------------------------------
        import matplotlib.pyplot as plt
        for aTdx in range(min(0, mesh.nE)):
            fig = plt.figure()  # create a figure object
            ax = fig.add_subplot(1, 1, 1)

            plt.gca().set_aspect('equal')
            grid = np.arange(-1.5, 1.5, .5)
            plt.yticks(grid)
            plt.xticks(grid)
            plt.grid(True, color='white', lw=0.1, alpha=.6)
            plt.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.elements, alpha=.5)
            TE = mesh.vertices[mesh.elements[aTdx]]
            ax.fill(TE[:, 0], TE[:, 1], color="red", alpha=.3)
            for k in range(3):
                if mesh.neighbours[aTdx, k] < mesh.nE:
                    TE = mesh.vertices[mesh.elements[mesh.neighbours[aTdx, k]]]
                    ax.fill(TE[:, 0], TE[:, 1], color="blue", alpha=.3)
            plt.show()

def plot():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from python.plot import plot_mesh_CG

    for mesh_name_ in glob.glob("data/*vrt"):
        mesh_name_ = mesh_name_[5:]
        print(mesh_name_)
        mesh = MeshIO(DATA_PATH + mesh_name_, **confDict)
        plot_mesh_CG(mesh, delta, OUTPUT_PATH + mesh_name_, ud = mesh.point_data["u"])

if __name__=="__main__":
    #convert()
    #readRHS()
    plot()