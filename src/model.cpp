//
// Created by klar on 13.03.20.
//
#ifndef NONLOCAL_ASSEMBLY_MODEL_CPP
#define NONLOCAL_ASSEMBLY_MODEL_CPP
#include <mathhelpers.cpp>

// ### KERNEL ##########################################################################################################
// Pointer -------------------------------------------------------------------------------------------------------------
double (*model_kernel)(const double * x, long labelx, const double * y, long labely, double sqdelta);

// Implementations -----------------------------------------------------------------------------------------------------
double kernel_constant(const double * x, const long labelx, const double * y, const long labely, const double sqdelta){
    return 4 / (M_PI * pow(sqdelta, 2));
}

double kernel_labeled(const double * x, const long labelx, const double * y, const long labely, const double sqdelta){
    double dist;
    long label;

    label = 10*labelx + labely;
    dist = vec_sqL2dist(x, y, 2);
    if (dist >= sqdelta) {
        cout << "Error in model_kernel. Distance smaller delta not expected." << endl;
        cout << dist << endl;
        abort();
    }
    if (label <= 12) {
        return 0.01 * 3. / (4*pow(sqdelta, 2));
    } else if (label>=21){
        return (100 *  3. / (4*pow(sqdelta, 2))) * (1 - (dist/sqdelta) );
    } else if (label == 13){
        return 0.0;
    } else {
        cout << "No such case " << endl;
        abort();
    }
}

// ### RIGHT HAND SIDE #################################################################################################

// Pointer -------------------------------------------------------------------------------------------------------------
double (*model_f)(const double * x);

// Implementations -----------------------------------------------------------------------------------------------------
double f_constant(const double * x){
    return 1.0;
}
double f_linear(const double * x){
    return -2. * (x[1] + 1.);
}

// ### BASIS FUNCTION ##################################################################################################

void model_basisFunction(const double * p, double *psi_vals){
    psi_vals[0] = 1 - p[0] - p[1];
    psi_vals[1] = p[0];
    psi_vals[2] = p[1];
}

void model_basisFunction(const double * p, const MeshType & mesh, double *psi_vals){
    int i=0;

    psi_vals[0] = 1;
    for (i=0; i<mesh.dim; i++){
        psi_vals[0] -= p[i];
        psi_vals[i+1] = p[i];
    }
}

#endif //NONLOCAL_ASSEMBLY_MODEL_CPP