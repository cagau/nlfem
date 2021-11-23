import matplotlib.pyplot as plt
import meshio
from matplotlib.backends.backend_pdf import PdfPages


def plot_2d_peridynamics(pp, mesh, u, f):
    vertices = mesh["vertices"]
    elements = mesh["elements"]

    ax = plt.gca()
    ax.set_aspect(1)
    shifted_verts = vertices + u.reshape(-1,2)
    plt.triplot(shifted_verts[:, 0], shifted_verts[:, 1], elements, lw=0.1, color='black', alpha=.9)
    plt.savefig(pp, format='pdf')
    plt.close()


def plot_2d_diffusion(pp, mesh, u, f):
    vertices = mesh["vertices"]
    elements = mesh["elements"]
    elementlabels = mesh["elementLabels"]

    # Plot element labels
    ax = plt.gca()
    ax.set_aspect(1)
    plt.triplot(vertices[:, 0], vertices[:, 1], elements, lw=0.3, color='white', alpha=.9)
    plt.tripcolor(vertices[:, 0], vertices[:, 1], elements, elementlabels, alpha=.6)
    plt.colorbar()
    plt.savefig(pp, format='pdf')
    plt.close()

    # Plot forcing term
    ax = plt.gca()
    ax.set_aspect(1)
    plt.triplot(vertices[:, 0], vertices[:, 1], elements, lw=0.1, color='white', alpha=.9)
    plt.tricontourf(vertices[:, 0], vertices[:, 1], elements, f)
    plt.colorbar()
    plt.savefig(pp, format='pdf')
    plt.close()

    # Plot solution
    ax = plt.gca()
    ax.set_aspect(1)
    plt.triplot(vertices[:, 0], vertices[:, 1], elements, lw=0.1, color='white', alpha=.9)
    plt.tricontourf(vertices[:, 0], vertices[:, 1], elements, u)
    plt.colorbar()
    plt.savefig(pp, format='pdf')
    plt.close()


def plot_3d_peridynamics(pp, mesh, u, f):
    vertices = mesh["vertices"]
    elements = mesh["elements"]
    elementlabels = mesh["elementLabels"]

    m = meshio.Mesh(points=vertices + u.reshape((-1, 3)), cells=[("tetra", elements)],
                    point_data={"u": np.linalg.norm(u.reshape(-1, 3), axis=1),
                                "labels": vertexlabels},
                    cell_data={"labels": elementlabels})
    meshio.write("plots/3Dsolve.vtk", m, file_format="vtk")


plot_function_dict = {
    # dim
    2:
        {
            # outdim
            2: plot_2d_peridynamics,
            1: plot_2d_diffusion
        },
    3:
        {
            3: plot_3d_peridynamics
        }
}


def saveplot(filename, mesh, u, f):
    dim = mesh["dim"]
    outdim = mesh["outdim"]
    pp = PdfPages(filename)
    plot_function_dict[dim][outdim](pp, mesh, u, f)
    pp.close()