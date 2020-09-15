//
// Created by klar on 14.09.20.
//
#include "MeshTypes.h"

int sparseMatrix::append(entryStruct &entry) {
    if (n_buffer >= reserved_buffer - 1) {
        //cout << "Buffer filled Up!" << endl;
        mergeBuffer();
    }
    buffer[n_buffer].dx = entry.dx;
    buffer[n_buffer].value = entry.value;
    n_buffer++;
    //cout << "Appended! " << endl;
    return 0;
}

int sparseMatrix::mergeBuffer(){
    // Copy Buffer
    // Sort Buffer into indexValuePairs
    // inplace merge indexValuePairs

    sort(buffer, buffer + n_buffer);
    n_buffer = reduce(buffer, n_buffer);

    // Too expensive. If ommited, we cannot reduce A.
    //inplace_merge(data, buffer, buffer + n_buffer);
    n_entries += n_buffer;
    // If omitted, then merge is even more expensive.
    // We also need more memory.
    //n_entries = reduce(data, n_entries);

    buffer = data + n_entries;
    n_buffer = 0;

    //cout << "MergeBuffer! New Size is " << n_buffer << endl;

    if (n_entries + reserved_buffer > reserved_total){

        size_guess *= 2;
        unsigned long estimatedNNZ = size_guess *
                                     static_cast<unsigned long>(pow(2*ceil(mesh.delta / mesh.maxDiameter + 1)*
                                                                    mesh.outdim, mesh.dim));
        reserved_total = reserved_buffer + estimatedNNZ;
        printf("Thread %i resizes buffer to %li entries.\n", omp_get_thread_num(), reserved_total);
        //printf("Now newly reserved %li\n", reserved_total);
        //printf("%f GB\n", static_cast<double>(reserved_total)*16.0 * 1e-9);
        auto auxPtr = static_cast<entryStruct *>(realloc(indexValuePairs, sizeof(entryType) * reserved_total));

        if ( auxPtr != nullptr){
            indexValuePairs = auxPtr;
            data = indexValuePairs;
            buffer = data + n_entries;
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
    //cout << "Reduced Buffer or Matrix!" << endl;
    return length - duplicate_counter;
}

