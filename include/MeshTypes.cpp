//
// Created by klar on 14.09.20.
//
#include "MeshTypes.h"

int sparseMatrix::append(entryStruct &entry) {
    if (n_buffer >= reserved_buffer - 1) {
        cout << "Buffer filled Up!" << endl;
        mergeBuffer();
    }
    buffer_A[n_buffer].dx = entry.dx;
    buffer_A[n_buffer].value = entry.value;
    n_buffer++;
    //cout << "Appended! " << endl;
    return 0;
}

int sparseMatrix::mergeBuffer(){
    // Copy Buffer
    // Sort Buffer into indexValuePairs
    // inplace merge indexValuePairs

    sort(buffer_A, buffer_A+n_buffer);
    n_buffer = reduce(buffer_A, n_buffer);

    inplace_merge(A, buffer_A, buffer_A+n_buffer);
    n_entries += n_buffer;
    n_entries = reduce(A, n_entries);

    buffer_A = A + n_entries;
    n_buffer = 0;

    cout << "MergeBuffer! New Size is " << n_buffer << endl;

    if (n_entries + reserved_buffer > reserved_total){
        cout << "Chunk larger than expected. Resize Sparse Matrix!" << endl;
        size_guess += size_guess;
        unsigned long estimatedNNZ = size_guess *
                                     static_cast<unsigned long>(pow(2*ceil(mesh.delta / mesh.maxDiameter + 1)*
                                                                    mesh.outdim, mesh.dim));
        //estimatedNNZ = size_guess;
        reserved_total = reserved_buffer + estimatedNNZ;
        cout << "Now newly reserved " << reserved_total << endl;
        auto auxPtr = static_cast<entryStruct *>(realloc(indexValuePairs, sizeof(entryType) * reserved_total));

        if ( auxPtr != nullptr){
            indexValuePairs = auxPtr;
            A = indexValuePairs;
            buffer_A = A+n_entries;
            return 0;
        }
        printf("Error in sparseMatrix::mergeBuffer. Could not allocate enough memory.");
        abort();
    }
    return 0;
}
unsigned long sparseMatrix::reduce(entryStruct * mat, const unsigned long length) {
    unsigned long duplicate_counter=0;
    for(ulong k=1; k<length; k++){
        bool is_duplicate = (mat[k - 1] == mat[k]);
        //printf("is duplicate %i", is_duplicate);
        if (is_duplicate) {
            duplicate_counter += 1;
            double val = mat[k].value;
            mat[k - duplicate_counter].value += val;
        } else {
            mat[k - duplicate_counter].value = mat[k].value;
            mat[k - duplicate_counter].dx = mat[k].dx;
        }
    }
    cout << "Reduced Buffer or Matrix!" << endl;
    return length - duplicate_counter;
}

