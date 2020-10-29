//
// Created by klar on 29.10.20.
//

#include "Quadrature.h"
// ### BASIS FUNCTION ##################################################################################################

//void model_basisFunction(const double * p, const int dim, double *psi_vals){
//    int i=0;
//
//    psi_vals[0] = 1;
//    for (i=0; i<dim; i++){
//        psi_vals[0] -= p[i];
//        psi_vals[i+1] = p[i];
//    }
//}