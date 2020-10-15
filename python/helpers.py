def read_arma_spMat(path, is_verbose=False):
    """
    Read sparse Matrix in armadillo spMat format from file.

    :param path: string, Path to file.
    :param is_verbose: bool, Verbose mode.
    :return: scipy.csr_matrix, Matrix of double.
    """

    import scipy.sparse as sp
    import numpy as np

    sizeof_double = 8

    f = open(path, "rb")
    # Read Armadillo header
    arma_header = f.readline()
    if arma_header != b'ARMA_SPM_BIN_FN008\n':
        raise ValueError("in read_arma_spMat(), input file is of wrong format.")
    # Get shape of sparse matrix
    arma_shape = f.readline()
    n_rows, n_cols, n_nonzero = tuple([int(x) for x in arma_shape.decode("utf-8").split()])
    if is_verbose: print("Shape (", n_rows, ", ", n_cols, ")", sep="")
    # Raw binary of sparse Matrix in csc-format
    b_data = f.read()
    b_values = b_data[:sizeof_double * n_nonzero]
    b_pointers = b_data[sizeof_double * n_nonzero:]

    values = np.frombuffer(b_values)
    if is_verbose: print("Values ", values)

    pointers = np.frombuffer(b_pointers, dtype=np.uint)
    row_index = pointers[:n_nonzero]
    if is_verbose: print("Row index", row_index)
    col_pointer = pointers[n_nonzero:]
    if is_verbose: print("Column pointer", col_pointer)

    A = sp.csc_matrix((values, row_index, col_pointer), shape=(n_rows, n_cols)).transpose()
    A = A.tocsr() # This is efficient, linearly in n_nonzeros.
    if is_verbose: print(A.todense())
    return A

def read_arma_mat(path, is_verbose=False):
    """
    Read armadillo vector format from file.

    :param path: string, Path to file.
    :param is_verbose: bool, Verbose mode.
    :return: np.ndarray, Vector of double.
    """
    import numpy as np

    sizeof_double = 8

    f = open(path, "rb")
    # Read Armadillo header
    arma_header = f.readline()
    if arma_header != b'ARMA_MAT_BIN_FN008\n':
        raise ValueError("in read_arma_mat(), input file is of wrong format.")
    # Get shape of sparse matrix
    arma_shape = f.readline()
    n_rows, n_cols = tuple([int(x) for x in arma_shape.decode("utf-8").split()])
    if is_verbose: print("Shape (", n_rows, ", ", n_cols, ")", sep="")
    # Raw binary of sparse Matrix in csc-format
    b_data = f.read()
    f.close()

    b_values = b_data[:sizeof_double * n_rows * n_cols]
    values = np.array(np.frombuffer(b_values)).reshape((n_rows, n_cols), order="F")
    if is_verbose: print("Values ", values)
    if is_verbose: print(values)

    return values

def read_arma_Mat_uword(path, is_verbose=False):
    """
    Read armadillo vector format from file.

    :param path: string, Path to file.
    :param is_verbose: bool, Verbose mode.
    :return: np.ndarray, Vector of double.
    """
    import numpy as np

    sizeof_long = 8

    f = open(path, "rb")
    # Read Armadillo header
    arma_header = f.readline()
    if arma_header != b'ARMA_MAT_BIN_IU008\n':
        raise ValueError("in read_arma_Mat_uword(), input file is of wrong format.")
    # Get shape of sparse matrix
    arma_shape = f.readline()
    n_rows, n_cols = tuple([int(x) for x in arma_shape.decode("utf-8").split()])
    if is_verbose: print("Shape (", n_rows, ", ", n_cols, ")", sep="")
    # Raw binary of sparse Matrix in csc-format
    b_data = f.read()
    f.close()

    b_values = b_data[:sizeof_long * n_rows * n_cols]
    values = np.array(np.frombuffer(b_values, dtype=np.uint)).reshape((n_rows, n_cols), order="F")
    if is_verbose: print("Values ", values)
    if is_verbose: print(values)

    return values

if __name__=="__main__":
    import numpy as np

    print("\nIndices")
    indices = read_arma_Mat_uword("../examples/Rates2D/indices_all", True)
    print(indices.shape)
    print(np.max(indices))
    print("\nValues")
    values = read_arma_mat("../examples/Rates2D/values_all", True)