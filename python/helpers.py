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


if __name__=="__main__":
    read_arma_spMat("../sp_Ad", True)