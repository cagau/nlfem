/**
    Some input checks.
    @file checks.cpp
    @author Manuel Klar
    @version 0.1 25/08/20
*/

#ifndef NONLOCAL_ASSEMBLY_CHECKS_CPP
#define NONLOCAL_ASSEMBLY_CHECKS_CPP

#include <armadillo>
#include "mathhelpers.h"
#include "MeshTypes.h"

void abortIfFalse(const bool assertion, const char * message){
    if (!assertion){
        cout << "ERROR in checks.cpp: " << message << endl;
        abort();
    }
}
void chk_QuadratureRule(QuadratureType & quadRule){
    // Volume of elements
    double elementVolume = 1./ static_cast<double>(faculty(quadRule.dim));

    double dx_sum = vec_sum(quadRule.dx, quadRule.nPx);
    abortIfFalse(double_eq(dx_sum, elementVolume, EPSILON_CHKS) , "Weights dx do not sum up to element volume.");

    double dy_sum = vec_sum(quadRule.dy, quadRule.nPy);
    abortIfFalse(double_eq(dy_sum, elementVolume, EPSILON_CHKS) , "Weights dy do not sum up to element volume.");
}

void chk_BasisFunction(QuadratureType & quadRule){
    // Integral of Basis Functions
    double elementIntegral = 1./ static_cast<double>(faculty(quadRule.dim + 1));

    arma::vec dx(quadRule.dx, quadRule.nPx);
    double psix_integral = arma::dot(dx, quadRule.psix.row(0));
    abortIfFalse(double_eq(psix_integral, elementIntegral, EPSILON_CHKS)  , "Wrong integral of basis function w.r.t. weights dx.");

    arma::vec dy(quadRule.dy, quadRule.nPy);
    double psiy_integral = arma::dot(dy, quadRule.psiy.row(0));
    abortIfFalse(double_eq(psiy_integral, elementIntegral, EPSILON_CHKS) , "Wrong integral of basis function w.r.t. weights dy.");
}

void chk_Mesh(MeshType & mesh){
    long nV_Omega = mesh.nV_Omega;
    const long nE = mesh.nE;
    long chk_nE_Omega=0;
    const unsigned long d = mesh.dim;
    const long nZeta = mesh.nZeta;

    for(long k=0; k<nE; k++){
        if (mesh.LabelTriangles(k)<=0) {
            for (unsigned long i = 0; i < d; i++) {
                // For description of element labels see MeshTypes.h
                //abortIfFalse(mesh.Triangles(i, k) >= nV_Omega, "Incorrect vertex order or incorrect element label.");
            }
        } else {
            chk_nE_Omega++;
        }
    }
    abortIfFalse(mesh.nE_Omega == chk_nE_Omega , "Number of elements with label>0 does not coincide with nE_Omega.");
    mesh.nE == chk_nE_Omega ? printf("WARNING: No Dirichlet boundary found! Check your element labels if that is not what you want."):0;

    if (mesh.is_DiscontinuousGalerkin) {
        abortIfFalse(mesh.outdim * mesh.nE * mesh.dVertex == mesh.K , "Matrix dimension does not match #basis functions and output dimension.");
    } else {
        abortIfFalse(mesh.outdim * mesh.nV == mesh.K , "Matrix dimension does not match #basis functions and output dimension.");
    }

    for(long k=0; k < nZeta; k++){
        abortIfFalse(mesh.ptrZeta[3*k] >= 0 && mesh.ptrZeta[3*k+1] >= 0 && mesh.ptrZeta[3*k+2] >= 0 , "Some entries in Zeta are negative.");
        abortIfFalse(mesh.ptrZeta[3*k] < nE && mesh.ptrZeta[3*k+1] < nE && mesh.ptrZeta[3*k+2] < nE , "Some entries in Zeta exceed the number of triangles.");
    }
}

void chk_Conf(MeshType & mesh, ConfigurationType & conf, QuadratureType & quadRule){
    if (mesh.dim == 3 || mesh.dim == 1){
        abortIfFalse(conf.integration_method == "baryCenter" ,
        "Only Bary-Center is currently implemented as integration method for 3D and 1D.");
    } else if (mesh.dim != 2 )
    {
        cout << "Dimension is not equal to 1, 2 or 3." << endl;
        abort();
    }
    if(conf.integration_method == "linearPrototypeMicroelastic" ){
        abortIfFalse(quadRule.tensorGaussDegree, "You chose a singular kernel, but no quadrature rule for it.");
    }
}
#endif //NONLOCAL_ASSEMBLY_CHECKS_CPP