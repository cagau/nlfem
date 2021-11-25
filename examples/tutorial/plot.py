import matplotlib.pyplot as plt
import meshio
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def saveshow(pp):
    if pp is not None:
        plt.savefig(pp, format='pdf')
        plt.close()
    else:
        plt.show()

def plot_2d_peridynamics_cg(mesh, u, f, pp=None, **kwargs):
    vertices = mesh["vertices"]
    elements = mesh["elements"]

    ax = plt.gca()
    ax.set_aspect(1)
    shifted_verts = vertices + u.reshape(-1,2)
    plt.triplot(vertices[:, 0], vertices[:, 1], elements, lw=0.1, color='black', alpha=.9)
    saveshow(pp)

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

    plt.figure(figsize=(kwargs.get("xsize", 8), kwargs.get("ysize", 7)))
    plt.triplot(vertices[:, 0], vertices[:, 1], elements, lw=0.4, color='black', alpha=.3)
    plt.tripcolor(vertices[:, 0], vertices[:, 1], elements, elementlabels, cmap=plt.cm.magma)
    plt.colorbar()
    saveshow(pp)


# Plot vertex labels
def plot_2d_diffusion_vertexlabels(mesh, u=None, f=None, pp=None, **kwargs):
    vertices = mesh["vertices"]
    elements = mesh["elements"]
    vertexlabels = mesh["vertexLabels"]

    plt.figure(figsize=(kwargs.get("xsize", 8), kwargs.get("ysize", 7)))
    plt.triplot(vertices[:, 0], vertices[:, 1], elements, lw=0.4, color='black', alpha=.3)
    plt.tricontourf(vertices[:, 0], vertices[:, 1], elements, vertexlabels, cmap=plt.cm.magma)
    plt.colorbar()
    saveshow(pp)


# Plot forcing term
def plot_2d_diffusion_forcing_term_cg(mesh, u, f, pp=None, **kwargs):
    vertices = mesh["vertices"]
    elements = mesh["elements"]

    plt.figure(figsize=(kwargs.get("xsize", 8), kwargs.get("ysize", 7)))
    plt.triplot(vertices[:, 0], vertices[:, 1], elements, lw=0.1, color='black', alpha=.3)
    plt.tricontourf(vertices[:, 0], vertices[:, 1], elements, f, cmap=plt.cm.magma)
    plt.colorbar()
    saveshow(pp)


# Plot forcing term
def plot_2d_diffusion_forcing_term_dg(mesh, u, f, pp=None, **kwargs):
    vertices = mesh["vertices"]
    elements = mesh["elements"]

    plt.figure(figsize=(kwargs.get("xsize", 8), kwargs.get("ysize", 7)))
    plt.triplot(vertices[:, 0], vertices[:, 1], elements, lw=0.1, color='black', alpha=.3)
    plt.tripcolor(vertices[:, 0], vertices[:, 1], elements, f[::3], cmap=plt.cm.magma)
    plt.colorbar()
    saveshow(pp)


# Plot solution CG
def plot_2d_diffusion_solution_cg(mesh, u, f=None, pp=None, **kwargs):
    vertices = mesh["vertices"]
    elements = mesh["elements"]
    elementlabels = mesh["elementLabels"]

    plt.figure(figsize=(kwargs.get("xsize", 8), kwargs.get("ysize", 7)))
    plt.triplot(vertices[:, 0], vertices[:, 1], elements, lw=0.1, color='black', alpha=.3)
    plt.tricontourf(vertices[:, 0], vertices[:, 1], elements, u, cmap=plt.cm.magma)
    plt.colorbar()

    saveshow(pp)


# Plot solution DG
def plot_2d_diffusion_solution_dg(mesh, u, f=None, pp=None, **kwargs):
    vertices = mesh["vertices"]
    elements = mesh["elements"]
    elementlabels = mesh["elementLabels"]
    is_detailed = kwargs.get("detailed", True)
    plt.figure(figsize=(kwargs.get("xsize", 8), kwargs.get("ysize", 7)))
    plt.triplot(vertices[:, 0], vertices[:, 1], elements, lw=0.1, color='black', alpha=.3)

    if not is_detailed:
        plt.tripcolor(vertices[:, 0], vertices[:, 1], elements, u[::3], cmap=plt.cm.magma)
        plt.colorbar()
    else:
        minval = min(u)
        maxval = max(u)
        for k in range(mesh["nE"]):
            if mesh["elementLabels"][k] > 0:
                Vdx = mesh["elements"][k]
                plt.tricontourf(mesh["vertices"][Vdx, 0], mesh["vertices"][Vdx, 1], u[3*k:(3*k+3)],  5,
                                cmap=plt.cm.get_cmap('magma'), vmin=minval, vmax=maxval)
        ax, _ = matplotlib.colorbar.make_axes(plt.gca(), shrink=1.)
        matplotlib.colorbar.ColorbarBase(ax, cmap=plt.cm.get_cmap('magma'),
                norm=matplotlib.colors.Normalize(vmin=minval, vmax=maxval))
    saveshow(pp)


def plot_2d_diffusion_cg(mesh, u, f, pp=None, **kwargs):
    plot_2d_diffusion_elementlabels(mesh, u, f, pp)
    plot_2d_diffusion_forcing_term_cg(mesh, u, f, pp)
    plot_2d_diffusion_solution_cg(mesh, u, f, pp)


def plot_2d_diffusion_dg(mesh, u, f, pp=None, **kwargs):
    plot_2d_diffusion_elementlabels(mesh, u, f, pp)
    plot_2d_diffusion_forcing_term_dg(mesh, u, f, pp)
    plot_2d_diffusion_solution_dg(mesh, u, f, pp)


def plot_3d_peridynamics(mesh, u, f, pp=None, **kwargs):
    vertices = mesh["vertices"]
    elements = mesh["elements"]
    elementlabels = mesh["elementLabels"]
    vertexlabels = mesh["vertexLabels"]

    m = meshio.Mesh(points=vertices + u.reshape((-1, 3)), cells=[("tetra", elements)],
                    point_data={"u": np.linalg.norm(u.reshape(-1, 3), axis=1),
                                "labels": vertexlabels},
                    cell_data={"labels": elementlabels})
    meshio.write("plots/3Dsolve.vtk", m, file_format="vtk")


plot_function_dict = {
    # dim of domain
    2:
        {
            # outdim of kernel
            2: {
                # ansatz space
                "CG": plot_2d_peridynamics_cg,
            },
            1: {
                "CG": plot_2d_diffusion_cg,
                "DG": plot_2d_diffusion_dg,
            }
        },
    3:
        {
            3: {
                "CG": plot_3d_peridynamics
            }
        }
}

def showplot(mesh, u, f):
    dim = mesh["dim"]
    outdim = mesh["outdim"]
    ansatz = mesh["ansatz"]

    plot_function_dict[dim][outdim][ansatz](mesh, u, f, None)

def saveplot(filename, mesh, u, f):
    dim = mesh["dim"]
    outdim = mesh["outdim"]
    ansatz = mesh["ansatz"]

    pp = PdfPages(filename)
    plot_function_dict[dim][outdim][ansatz](mesh, u, f, pp)
    pp.close()
