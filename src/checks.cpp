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
    assert(("Weights dx do not sum up to element volume.", double_eq(dx_sum, elementVolume)));

    double dy_sum = vec_sum(quadRule.dy, quadRule.nPy);
    assert(("Weights dy do not sum up to element volume.", double_eq(dy_sum, elementVolume)));
}

void chk_BasisFunction(QuadratureStruct & quadRule){
    // Integral of Basis Functions
    double elementIntegral = 1./ static_cast<double>(faculty(quadRule.dim + 1));

    arma::vec dx(quadRule.dx, quadRule.nPx);
    double psix_integral = arma::dot(dx, quadRule.psix.row(0));
    assert(("Wrong integral of basis function w.r.t. weights dx.", double_eq(psix_integral, elementIntegral)));

    arma::vec dy(quadRule.dy, quadRule.nPy);
    double psiy_integral = arma::dot(dy, quadRule.psiy.row(0));
    assert(("Wrong integral of basis function w.r.t. weights dy..", double_eq(psiy_integral, elementIntegral)));
}
#endif //NONLOCAL_ASSEMBLY_CHECKS_CPP