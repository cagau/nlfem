import numpy as np
import meshio
import meshzoo


def set_element_dirichlet_sign(elements, vertices, delta, elementlabels=None):
    #upperRight = 0.5
    n_verts = elements.shape[1]
    if elementlabels is None:
        elementlabels = np.ones(elements.shape[0], dtype=np.int)

    for k, T in enumerate(elements):
        #shiftedBC = np.sum(vertices[T], axis=0)/nVerts - upperRight/2.
        #is_Omega = np.max(np.abs(shiftedBC)) < upperRight/2.
        is_omega = np.sum(vertices[T], axis=0)[1]/n_verts > 0.0
        if not is_omega:
            elementlabels[k] *= -1
    return elementlabels


def add_artificial_vertex(meshio_mesh, vertices, elements, elementlabels):
    n_vertex = vertices.shape[0]

    vertices = np.concatenate([vertices, [[0.0, 0.0]]])
    lines = meshio_mesh.cells_dict["line"]
    new_elements = np.array([[*line, n_vertex] for line in lines])
    n_new_elements = new_elements.shape[0]

    elements = np.concatenate([elements, new_elements])
    elementlabels = np.concatenate([elementlabels, np.zeros(n_new_elements, dtype=np.int)])

    return vertices, elements, elementlabels


def trim_artificial_vertex(nlfem_mesh, u=None, f_base=None):
    nlfem_mesh["elements"] = nlfem_mesh["elements"][nlfem_mesh["elementLabels"] != 0]
    nlfem_mesh["vertices"] = nlfem_mesh["vertices"][nlfem_mesh["vertexLabels"] != 0]
    nlfem_mesh["vertexLabels"] = nlfem_mesh["vertexLabels"][nlfem_mesh["vertexLabels"] != 0]
    nlfem_mesh["elementLabels"] = nlfem_mesh["elementLabels"][nlfem_mesh["elementLabels"] != 0]
    if u is not None:
        u = u[nlfem_mesh["nodeLabels"] != 0]
    if f_base is not None:
        f_base = f_base[nlfem_mesh["nodeLabels"] != 0]
    nlfem_mesh["nodeLabels"] = nlfem_mesh["nodeLabels"][nlfem_mesh["nodeLabels"] != 0]

    return nlfem_mesh, u, f_base


def read_gmsh(filename, set_art_vertex=True):
    meshio_mesh = meshio.read(filename)

    vertices = np.array(meshio_mesh.points[:, :2])
    elements = meshio_mesh.cells_dict["triangle"]
    elementlabels = np.array(meshio_mesh.cell_data_dict["gmsh:physical"]["triangle"], dtype=np.int)
    elementlabels[elementlabels == np.max(elementlabels)] *= -1

    if set_art_vertex:
        vertices, elements, elementlabels = add_artificial_vertex(meshio_mesh, vertices, elements, elementlabels)

    return vertices, elements, elementlabels


def regular_square(n = 24, delta = 0.1, variant = "zigzag"):
    points, elements = meshzoo.rectangle(
        xmin=- delta, xmax=1 + delta,
        ymin=- delta, ymax=1 + delta,
        nx= n + 1, ny=n + 1,
        variant=variant
    )
    vertices = points[:, :2].copy()
    elementlabels = set_element_dirichlet_sign(elements, vertices, delta)
    return vertices, elements, elementlabels


def regular_cube(n = 24, delta = 0.1):
    vertices, elements = meshzoo.cube(-delta, 0.5 + delta, -delta, 0.5 + delta,
                                      -delta, 0.5 + delta,
                                      nx=n + 1, ny=n + 1, nz=n + 1)
    elementlabels = set_element_dirichlet_sign(elements, vertices, delta)
    return vertices, elements, elementlabels
