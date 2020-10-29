//
// Created by klar on 29.10.20.
//

#ifndef NONLOCAL_ASSEMBLY_QUADRATURE_H
#define NONLOCAL_ASSEMBLY_QUADRATURE_H

#include "cmath"
#include "armadillo"

class Quadrature {
    /**
     * @brief Contains the quadrature rules.
     *
     *
     */
public:
    const long dim;

    const double * Px;
    const double * dx;
    const long nPx;

    const double * Py;
    const double * dy;
    const long nPy;

    const double * Pg;
    const double * dg;
    const long tensorGaussDegree;
    const long nPg = pow(tensorGaussDegree, dim * 2);

    //const interactionMethodType interactionMethod;
    arma::Mat<double> psix{arma::Mat<double>(this->dim +1, this->nPx)};
    arma::Mat<double> psiy{arma::Mat<double>(this->dim +1, this->nPy)};
    /**
     * @brief Contains the quadrature rules.
     *
     * @param dim_ Dimension of the domain.
     * @param Px_ Integration points for outer integral on standard simplex.
     * @param dx_ Integration weights for outer integral on standard simplex.
     * @param nPx_ Number of integration points (outer).
     * @param Py_ Integration points for inner integral on standard simplex.
     * @param dy_ Integration weights for inner integral on standard simplex.
     * @param nPy_ Number of integration points (inner).
     * @param Pg_ Integration points of tensor Gauss quadrature (for singular kernels).
     * @param dg_ Integration weights of tensor Gauss quadrature (for singular kernels).
     * @param tensorGaussDegree_ Degree of tensor Gauss quadrature rule.
     */
    Quadrature(const long dim_,
               const double * Px_,
               const double * dx_,
               const long nPx_,
               const double * Py_,
               const double * dy_,
               const long nPy_,
               const double * Pg_,
               const double * dg_,
               const long tensorGaussDegree_=0):
               dim(dim_),
               Px(Px_),
               dx(dx_),
               nPx(nPx_),
               Py(Py_),
               dy(dy_),
               nPy(nPy_),
               Pg(Pg_),
               dg(dg_),
               tensorGaussDegree(tensorGaussDegree_)
               {};
};

/**
 * @brief  Definition of basis function.
 * @param p Quadrature point
 * @param dim Dimension of domain (2,3).
 * @param psi_vals  Value of 3 or 4 basis functions, depending on the dimension.
 */
void model_basisFunction(const double * p, int dim, double *psi_vals);

#endif //NONLOCAL_ASSEMBLY_QUADRATURE_H
