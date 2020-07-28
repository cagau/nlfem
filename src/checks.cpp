//
// Created by klar on 16.03.20.
//
#ifndef NONLOCAL_ASSEMBLY_CHECKS_CPP
#define NONLOCAL_ASSEMBLY_CHECKS_CPP
#include <cassert>
#include <armadillo>

#include "mathhelpers.cpp"
#include "MeshTypes.h"

void chk_QuadratureRule(QuadratureType & quadRule){
    // Volume of elements
    double elementVolume = 1./ static_cast<double>(faculty(quadRule.dim));

    double dx_sum = vec_sum(quadRule.dx, quadRule.nPx);
    assert((double_eq(dx_sum, elementVolume) && "Weights dx do not sum up to element volume."));

    double dy_sum = vec_sum(quadRule.dy, quadRule.nPy);
    assert((double_eq(dy_sum, elementVolume) && "Weights dy do not sum up to element volume."));
}

void chk_BasisFunction(QuadratureType & quadRule){
    // Integral of Basis Functions
    double elementIntegral = 1./ static_cast<double>(faculty(quadRule.dim + 1));

    arma::vec dx(quadRule.dx, quadRule.nPx);
    double psix_integral = arma::dot(dx, quadRule.psix.row(0));
    assert((double_eq(psix_integral, elementIntegral)  && "Wrong integral of basis function w.r.t. weights dx."));

    arma::vec dy(quadRule.dy, quadRule.nPy);
    double psiy_integral = arma::dot(dy, quadRule.psiy.row(0));
    assert((double_eq(psiy_integral, elementIntegral) && "Wrong integral of basis function w.r.t. weights dy."));
}

void chk_Mesh(MeshType & mesh){
    long nV_Omega = mesh.L_Omega;
    const long nE = mesh.J;
    long chk_nE_Omega=0;
    const unsigned long d = mesh.dim;
    const long nCeta = mesh.nCeta;

    for(long k=0; k<nE; k++){
        if (mesh.LabelTriangles(k)==2) {
            for (unsigned long i = 0; i < d; i++) {
                // For description of element labels see MeshTypes.h
                assert((mesh.Triangles(i, k) >= nV_Omega && "Incorrect vertex order or incorrect element label."));
            }
        } else {
            chk_nE_Omega++;
        }
    }
    assert((mesh.J_Omega == chk_nE_Omega && "Number of elements with label!=2 does not coincide with nE_Omega."));

    for(long k=0; k < nCeta; k++){
        assert((mesh.ptrCeta[3*k] >= 0 && mesh.ptrCeta[3*k+1] >= 0 && mesh.ptrCeta[3*k+2] >= 0 && "Some entries in Ceta are negative."));
        assert((mesh.ptrCeta[3*k] < nE && mesh.ptrCeta[3*k+1] < nE && mesh.ptrCeta[3*k+2] < nE && "Some entries in Ceta exceed the number of triangles."));
    }
}
#endif //NONLOCAL_ASSEMBLY_CHECKS_CPP