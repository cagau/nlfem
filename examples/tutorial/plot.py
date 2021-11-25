import matplotlib.pyplot as plt
import meshio
from matplotlib.backends.backend_pdf import PdfPages

def saveshow(pp):
    if pp is not None:
        plt.savefig(pp, format='pdf')
        plt.close()
    else:
        plt.show()

def plot_2d_peridynamics(mesh, u, f, pp=None, **kwargs):
    vertices = mesh["vertices"]
    elements = mesh["elements"]

    ax = plt.gca()
    ax.set_aspect(1)
    shifted_verts = vertices + u.reshape(-1,2)
    plt.triplot(shifted_verts[:, 0], shifted_verts[:, 1], elements, lw=0.1, color='black', alpha=.9)
    saveshow(pp)


# Plot element labels
def plot_2d_diffusion_elementlabels(mesh, u=None, f=None, pp=None, **kwargs):
    vertices = mesh["vertices"]
    elements = mesh["elements"]
    elementlabels = mesh["elementLabels"]

    plt.figure(figsize=(kwargs.get("xsize", 7), kwargs.get("ysize", 7)))
    plt.triplot(vertices[:, 0], vertices[:, 1], elements, lw=0.4, color='black', alpha=.3)
    plt.tripcolor(vertices[:, 0], vertices[:, 1], elements, elementlabels, cmap=plt.cm.magma, alpha=.6)
    #plt.colorbar()
    saveshow(pp)


# Plot vertex labels
def plot_2d_diffusion_vertexlabels(mesh, u=None, f=None, pp=None, **kwargs):
    vertices = mesh["vertices"]
    elements = mesh["elements"]
    vertexlabels = mesh["vertexLabels"]

    plt.figure(figsize=(kwargs.get("xsize", 7), kwargs.get("ysize", 7)))
    plt.triplot(vertices[:, 0], vertices[:, 1], elements, lw=0.4, color='black', alpha=.3)
    plt.tricontourf(vertices[:, 0], vertices[:, 1], elements, vertexlabels, cmap=plt.cm.magma, alpha=.6)
    #plt.colorbar()
    saveshow(pp)


# Plot forcing term
def plot_2d_diffusion_forcing_term(mesh, u, f, pp=None, **kwargs):
    vertices = mesh["vertices"]
    elements = mesh["elements"]

    plt.figure(figsize=(kwargs.get("xsize", 7), kwargs.get("ysize", 7)))
    plt.triplot(vertices[:, 0], vertices[:, 1], elements, lw=0.1, color='black', alpha=.3)
    plt.tricontourf(vertices[:, 0], vertices[:, 1], elements, f, cmap=plt.cm.magma, alpha=.6)
    #plt.colorbar()
    saveshow(pp)


# Plot solution
def plot_2d_diffusion_solution(mesh, u, f=None, pp=None, **kwargs):
    vertices = mesh["vertices"]
    elements = mesh["elements"]
    elementlabels = mesh["elementLabels"]

    plt.figure(figsize=(kwargs.get("xsize", 7), kwargs.get("ysize", 7)))
    plt.triplot(vertices[:, 0], vertices[:, 1], elements, lw=0.1, color='black', alpha=.3)
    plt.tricontourf(vertices[:, 0], vertices[:, 1], elements, u, cmap=plt.cm.magma, alpha=.6)
    #plt.colorbar()
    saveshow(pp)


def plot_2d_diffusion(mesh, u, f, pp=None, **kwargs):
    plot_2d_diffusion_elementlabels(mesh, u, f, pp)
    plot_2d_diffusion_forcing_term(mesh, u, f, pp)
    plot_2d_diffusion_solution(mesh, u, f, pp)


def plot_3d_peridynamics(mesh, u, f, pp=None, **kwargs):
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

def showplot(mesh, u, f):
    dim = mesh["dim"]
    outdim = mesh["outdim"]
    plot_function_dict[dim][outdim](mesh, u, f, None)

def saveplot(filename, mesh, u, f):
    dim = mesh["dim"]
    outdim = mesh["outdim"]
    pp = PdfPages(filename)
    plot_function_dict[dim][outdim](mesh, u, f, pp)
    pp.close()
