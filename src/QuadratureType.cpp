//
// Created by klar on 29.10.20.
//

#include "QuadratureType.h"
// ### BASIS FUNCTION ##################################################################################################

QuadratureType::QuadratureType(const long dim_, const double *Px_, const double *dx_, const long nPx_,
                               const double *Py_, const double *dy_, const long nPy_, const double *Pg_,
                               const double *dg_, const long tensorGaussDegree_):
        dim(dim_),
        Px(Px_),
        dx(dx_),
        nPx(nPx_),
        Py(Py_),
        dy(dy_),
        nPy(nPy_),
        Pg(Pg_),
        dg(dg_),
        tensorGaussDegree(tensorGaussDegree_) {
        long dVertex = dim+1;
        for(int h=0; h<nPx; h++){
            // This works due to Column Major ordering of Armadillo Matricies!
            model_basisFunction(& Px[dim*h], dim, & psix[dVertex * h]);
        }
        for(int h=0; h<nPy; h++) {
            // This works due to Column Major ordering of Armadillo Matricies!
            model_basisFunction(&Py[dim * h], dim, &psiy[dVertex * h]);
        }
};

void model_basisFunction(const double * p, const long dim, double *psi_vals){
    psi_vals[0] = 1;
    for (int i=0; i<dim; i++){
        psi_vals[0] -= p[i];
        psi_vals[i+1] = p[i];
    }
};

void model_basisFunction(const double * p, double *psi_vals){
    psi_vals[0] = 1 - p[0] - p[1];
    psi_vals[1] = p[0];
    psi_vals[2] = p[1];
};