/**
    Integration and retriangulation routines.
  All integration routines share a common signature.
The central difference between different routines is the way they handle the truncation of the domain
 of integration. In almost all cases, the truncation of the inner triangle bT
is performed for each quadrature point in the outer
triangle aT.
In addition for singular kernels special care is needed. Due to the necessity of
transformations of the domain of integration those integration routines are tied to a kernel.

The integration routine changes the data in *termLocal and *termNonloc to. **The arrays have to be zero-initialized.**

    * termLocal = int_aT phiA(x) phiB(x) int_bT ker(x,y) dy dx\n,
    * termNonloc = int_aT phiA(x) int_bT phiB(y) ker(y,x) dy dx.

Please note that the nonlocal term has to be subtracted, while the local term has to be added to the stiffness
matrix.

@param aT    Triangle of the outer integral.
@param bT    Triangle of the inner integral.
@param quadRule Quadrature rule.
@param mesh  Mesh.
@param conf  Confuration.
@param is_firstbfslayer Switch to tell whether the integration is happening in the first layer of the breadth first
search. This variable is true only if the kernel is singular. In that case the integrals between aT and its immediate
neighbours have to be handled with special care.
@param termLocal This term contains the local part of the integral
@param termNonloc This term contains the nonlocal part of the integral

@file integration.cpp
@author Manuel Klar
@version 0.1 25/08/20
**/

#ifndef NONLOCAL_ASSEMBLY_INTEGRATION_CPP
#define NONLOCAL_ASSEMBLY_INTEGRATION_CPP

#include <list>
#include "MeshTypes.h"
#include "mathhelpers.h"
#include "model.h"
#include "integration.h"

const double SCALEDET = 0.0625;
void (*integrate)(const ElementType &aT, const ElementType &bT, const QuadratureType &quadRule, const MeshType &mesh,
                         const ConfigurationType &conf, bool is_firstbfslayer, double *termLocal, double *termNonloc);
// ___ INTEGRATION IMPLEMENTATION ______________________________________________________________________________________

// Integration Methods #################################################################################################

void integrate_linearPrototypeMicroelastic_retriangulate(const ElementType &aT, const ElementType &bT,
                                                         const QuadratureType &quadRule,
                                                         const MeshType &mesh, const ConfigurationType &conf,
                                                         bool is_firstbfslayer, double *termLocal,
                                                         double *termNonloc){
    if (is_firstbfslayer) {
        integrate_linearPrototypeMicroelastic_tensorgauss(aT, bT, quadRule, mesh, conf, is_firstbfslayer, termLocal,
                                                          termNonloc);
    } else {
        integrate_retriangulate(aT, bT, quadRule, mesh, conf, is_firstbfslayer, termLocal, termNonloc);
    }
}
void integrate_linearPrototypeMicroelastic_baryCenter(const ElementType &aT, const ElementType &bT,
                                                      const QuadratureType &quadRule,
                                                      const MeshType &mesh, const ConfigurationType &conf,
                                                      bool is_firstbfslayer, double *termLocal,
                                                      double *termNonloc){
    if (is_firstbfslayer) {
        integrate_linearPrototypeMicroelastic_tensorgauss(aT, bT, quadRule, mesh, conf, is_firstbfslayer, termLocal,
                                                          termNonloc);
    } else {
        integrate_baryCenter(aT, bT, quadRule, mesh, conf, is_firstbfslayer, termLocal, termNonloc);
    }
}
void integrate_linearPrototypeMicroelastic_baryCenterRT(const ElementType &aT, const ElementType &bT,
                                                        const QuadratureType &quadRule,
                                                        const MeshType &mesh, const ConfigurationType &conf,
                                                        bool is_firstbfslayer, double *termLocal,
                                                        double *termNonloc){
    if (is_firstbfslayer) {
        integrate_linearPrototypeMicroelastic_tensorgauss(aT, bT, quadRule, mesh, conf, is_firstbfslayer, termLocal,
                                                          termNonloc);
    } else {
        integrate_baryCenterRT(aT, bT, quadRule, mesh, conf, is_firstbfslayer, termLocal, termNonloc);
    }
}

void integrate_linearPrototypeMicroelastic_tensorgauss(const ElementType &aT, const ElementType &bT,
                                                       const QuadratureType &quadRule,
                                                       const MeshType &mesh, const ConfigurationType &conf,
                                                       bool is_firstbfslayer,
                                                       double * termLocal, double * termNonloc){
    const std::list<double(*)(double *)> traffoCommonVertex = {traffoCommonVertex0,
                                                               traffoCommonVertex1};
    const std::list<double(*)(double *)> traffoCommonEdge = {traffoCommonEdge0,
                                                             traffoCommonEdge1,
                                                             traffoCommonEdge2,
                                                             traffoCommonEdge3,
                                                             traffoCommonEdge4};
    const std::list<double(*)(double *)> traffoIdentical = {traffoIdentical0,
                                                            traffoIdentical1,
                                                            traffoIdentical2,
                                                            traffoIdentical3,
                                                            traffoIdentical4,
                                                            traffoIdentical5};
    ElementStruct aTsorted, bTsorted;
    int argSortA[3], argSortB[3];

    int nEqual = join(aT, bT, mesh, aTsorted, bTsorted, argSortA, argSortB);
    std::list<double (*)(double *)> traffoList;

    double factors;
    //const int dim = mesh.dim;
    double alpha[4], alphaCanceled[4], traffodet, traffodetCanceled, x[2], y[2],
    x_canceled[2], y_canceled[2];//, kernel_val=0.;

    double kernel_val[mesh.outdim*mesh.outdim];
    double psix[3], psiy[3];
    //cout << "Tensor Gauss" << endl;
    if (nEqual == 1){
        traffoList = traffoCommonVertex;
    } else if (nEqual == 2){
        traffoList = traffoCommonEdge;
    } else if (nEqual == 3) {
        traffoList = traffoIdentical;
    } else {
        cout << "Error in integrate_linearPrototypeMicroelastic_tensorgauss: This should not have happened." << endl;
        abort();
    }

    int traffoCounter = 0;
    for(auto & traffo : traffoList) {
        for (int k = 0; k < quadRule.nPg; k++) {
            for (int j = 0; j < 4; j++) {
                alpha[j] = quadRule.Pg[4 * k + j];
                alphaCanceled[j] = quadRule.Pg[4 * k + j];
            }
            scale(alpha);
            scale(alphaCanceled);

            traffodetCanceled = pow(alphaCanceled[0], 2);
            alphaCanceled[0] = 1.0;

            traffodet = traffo(alpha);
            traffodetCanceled *= traffo(alphaCanceled);

            mirror(alpha);
            mirror(alphaCanceled);

            toPhys(aTsorted.E, &alpha[0], 2, x);
            toPhys(bTsorted.E, &alpha[2], 2, y);
            toPhys(aTsorted.E, &alphaCanceled[0], 2, x_canceled);
            toPhys(bTsorted.E, &alphaCanceled[2], 2, y_canceled);

            // Eval Kernel(x-y)
            model_kernel(x_canceled, aTsorted.label, y_canceled, bTsorted.label, mesh.sqdelta, kernel_val);

            //cout << "x " << endl;
            //cout << x_canceled[0] << ", " << x_canceled[1] << endl;
            //cout << "y " << endl;
            //cout << y_canceled[0] << ", " << y_canceled[1] << endl;

            // Eval C(x-y)
            model_basisFunction(&alpha[0], 2, psix);
            model_basisFunction(&alpha[2], 2, psiy);
            //double outTest[mesh.outdim*mesh.outdim*mesh.dVertex*mesh.dVertex];
            // double psitest[3] = {20., 30., 50.};
            // [7] for (int a = 0; a < mesh.dVertex*mesh.outdim; a++) {
            for (int a = 0; a < mesh.dVertex*mesh.outdim; a++) {
                for (int b = 0; b < mesh.dVertex*mesh.outdim; b++) {
                    //outTest[mesh.outdim*mesh.dVertex * a + b] = kernel_val[mesh.outdim * (a%mesh.outdim) + b%mesh.outdim]
                    //        + psitest[a/mesh.outdim]*psitest[b/mesh.outdim];
                    //cout << outTest[mesh.outdim*mesh.dVertex * a + b] << ",   ";

                    factors = kernel_val[mesh.outdim * (a%mesh.outdim) + b%mesh.outdim] * traffodetCanceled * SCALEDET *
                            quadRule.dg[k] * aTsorted.absDet * bTsorted.absDet;
                    //factors = kernel_val * traffodet * scaledet * quadRule.dg[k] * aTsorted.absDet * bTsorted.absDet;
                    termLocal[mesh.outdim*mesh.dVertex * argSortA[a] + argSortA[b]] += 2*factors * psix[a/mesh.outdim] * psix[b/mesh.outdim];
                    //termLocal[mesh.dVertex * a + b] += 2*factors* psix[a] * psix[b];
                    // [10] siehe [9]
                    termNonloc[mesh.outdim*mesh.dVertex * argSortA[a] + argSortB[b]] += 2*factors * psix[a/mesh.outdim] * psiy[b/mesh.outdim];
                    //termNonloc[mesh.dVertex * a + b] += 2*factors * psix[a] * psiy[b];
                    //cout << termLocal[mesh.outdim*mesh.dVertex * a + b] << endl;
                }
                //cout << endl;
            }
            //cout << "Thank you!" << endl;
            //abort();
        }
        traffoCounter++;
    }

}

void integrate_tensorgauss(const ElementType &aT, const ElementType &bT, const QuadratureType &quadRule,
                           const MeshType &mesh, const ConfigurationType &conf, bool is_firstbfslayer,
                           double * termLocal, double * termNonloc){
    if (is_firstbfslayer) {
        //printf("Ok, fine!");
        //abort();
        const std::list<double(*)(double *)> traffoCommonVertex = {traffoCommonVertex0,
                                                                   traffoCommonVertex1};
        const std::list<double(*)(double *)> traffoCommonEdge = {traffoCommonEdge0,
                                                                 traffoCommonEdge1,
                                                                 traffoCommonEdge2,
                                                                 traffoCommonEdge3,
                                                                 traffoCommonEdge4};
        const std::list<double(*)(double *)> traffoIdentical = {traffoIdentical0,
                                                                traffoIdentical1,
                                                                traffoIdentical2,
                                                                traffoIdentical3,
                                                                traffoIdentical4,
                                                                traffoIdentical5};
        ElementStruct aTsorted, bTsorted;
        int argSortA[3], argSortB[3];
        int nEqual = join(aT, bT, mesh, aTsorted, bTsorted, argSortA, argSortB);
        std::list<double (*)(double *)> traffoList;

        double factors;
        //const int dim = mesh.dim;
        double alpha[4], traffodet, x[2], y[2];//, kernel_val=0.;

        double kernel_val[mesh.outdim*mesh.outdim];
        double psix[3], psiy[3];
        //cout << "Tensor Gauss" << endl;
        if (nEqual == 1){
            traffoList = traffoCommonVertex;
        } else if (nEqual == 2){
            traffoList = traffoCommonEdge;
        } else if (nEqual == 3) {
            traffoList = traffoIdentical;
        } else {
            cout << "Error in integrate_tensorgauss: This should not have happened." << endl;
            abort();
        }

        int traffoCounter = 0;
        for(auto & traffo : traffoList) {
            for (int k = 0; k < quadRule.nPg; k++) {
                for (int j = 0; j < 4; j++) {
                    alpha[j] = quadRule.Pg[4 * k + j];
                    //alphaCanceled[j] = quadRule.Pg[4 * k + j];
                }
                //traffodetCanceled = (nEqual==1 && traffoCounter==0) ? alphaCanceled[0] : pow(alphaCanceled[0], 2);
                //alphaCanceled[0] = 1.0;

                scale(alpha);
                //scale(alphaCanceled);
                traffodet = traffo(alpha);
                //traffodetCanceled *= traffo(alphaCanceled);

                mirror(alpha);
                //mirror(alphaCanceled);

                toPhys(aTsorted.E, &alpha[0], 2, x);
                toPhys(bTsorted.E, &alpha[2], 2, y);
                //toPhys(aTsorted.E, &alphaCanceled[0], 2, x_canceled);
                //toPhys(bTsorted.E, &alphaCanceled[2], 2, y_canceled);

                // Eval Kernel(x-y)
                model_kernel(x, aTsorted.label, y, bTsorted.label, mesh.sqdelta, kernel_val);

                //cout << "x " << endl;
                //cout << x_canceled[0] << ", " << x_canceled[1] << endl;
                //cout << "y " << endl;
                //cout << y_canceled[0] << ", " << y_canceled[1] << endl;

                // Eval C(x-y)
                model_basisFunction(&alpha[0], 2, psix);
                model_basisFunction(&alpha[2], 2, psiy);
                //double outTest[mesh.outdim*mesh.outdim*mesh.dVertex*mesh.dVertex];
                // double psitest[3] = {20., 30., 50.};
                // [7] for (int a = 0; a < mesh.dVertex*mesh.outdim; a++) {
                for (int a = 0; a < mesh.dVertex*mesh.outdim; a++) {
                    for (int b = 0; b < mesh.dVertex*mesh.outdim; b++) {
                        //outTest[mesh.outdim*mesh.dVertex * a + b] = kernel_val[mesh.outdim * (a%mesh.outdim) + b%mesh.outdim]
                        //        + psitest[a/mesh.outdim]*psitest[b/mesh.outdim];
                        //cout << outTest[mesh.outdim*mesh.dVertex * a + b] << ",   ";

                        factors = kernel_val[mesh.outdim * (a%mesh.outdim) + b%mesh.outdim] * traffodet * SCALEDET *
                                  quadRule.dg[k] * aTsorted.absDet * bTsorted.absDet;
                        // In case of the local term both entries belong to the outer integral. Hence...argSortA[a] + argSortA[b]
                        termLocal[mesh.outdim*mesh.dVertex * argSortA[a] + argSortA[b]] += 2*factors* psix[a/mesh.outdim] * psix[b/mesh.outdim];
                        // In this case we write ...argSortA[a] + argSortB[b].
                        termNonloc[mesh.outdim*mesh.dVertex * argSortA[a] + argSortB[b]] += 2*factors* psix[a/mesh.outdim] * psiy[b/mesh.outdim];
                    }
                    //cout << endl;
                }
                //cout << "Thank you!" << endl;
                //abort();
            }
            traffoCounter++;
        }
    } else {
        integrate_retriangulate(aT, bT, quadRule, mesh, conf, is_firstbfslayer, termLocal, termNonloc);
    }
}
void integrate_fullyContained(const ElementType &aT, const ElementType &bT, const QuadratureType &quadRule,
                              const MeshType &mesh, const ConfigurationType &conf, bool is_firstbfslayer,
                              double *termLocal, double *termNonloc){

    const int dim = mesh.dim;
    double x[2], y[2];//, kernel_val=0.;
    double kernel_val[mesh.outdim*mesh.outdim];
    double kernelT_val[mesh.outdim*mesh.outdim];

    for (int k = 0; k < quadRule.nPx; k++) {
        toPhys(aT.E, &(quadRule.Px[dim * k]), dim, x);
        for (int i = 0; i < quadRule.nPy; i++) {
            toPhys(bT.E, &(quadRule.Py[dim * i]), dim, y);

            // Eval Kernel(x-y)
            model_kernel(x, aT.label, y, bT.label, mesh.sqdelta, kernel_val);
            model_kernel(y, bT.label, x, aT.label, mesh.sqdelta, kernelT_val);

            for (int a = 0; a < mesh.dVertex * mesh.outdim; a++) {
                for (int b = 0; b < mesh.dVertex * mesh.outdim; b++) {

                    termLocal[mesh.outdim * mesh.dVertex * a + b] += quadRule.dx[k] * quadRule.dy[i] *
                            kernel_val[mesh.outdim * (a % mesh.outdim) + b % mesh.outdim] * aT.absDet * bT.absDet *
                            2 * quadRule.psix(a/mesh.outdim, k) * quadRule.psix(b/mesh.outdim, k);

                    termNonloc[mesh.outdim * mesh.dVertex * a + b] += quadRule.dx[k] * quadRule.dy[i] *
                            kernel_val[mesh.outdim * (a % mesh.outdim) + b % mesh.outdim] * aT.absDet * bT.absDet *
                            2 * quadRule.psix(a/mesh.outdim, k) * quadRule.psiy(b/mesh.outdim, i);

                }
                //cout << endl;
            }
            //cout << "Thank you!" << endl;
            //abort()
        }
    }
}

void integrate_retriangulate(const ElementType &aT, const ElementType &bT, const QuadratureType &quadRule,
                             const MeshType &mesh, const ConfigurationType &conf, bool is_firstbfslayer, double *termLocal,
                             double *termNonloc) {


    if ((mesh.maxDiameter > EPSILON) && (mesh.delta - 2*mesh.maxDiameter > 0) && isFullyContained(aT, bT, mesh)){
        integrate_fullyContained(aT, bT, quadRule, mesh, conf, is_firstbfslayer, termLocal, termNonloc);
        return;
    }

    const int dim = mesh.dim;
    int k = 0, a = 0, b = 0;
    double x[dim];
    // [x 11] [mesh.outdim*mesh.outdim]
    //double innerLocal = 0;
    double innerLocal[mesh.outdim*mesh.outdim];

    // [x 12] [mesh.outdim*mesh.outdim*mesh.dVertex]
    // double innerNonloc[mesh.dVertex];
    double innerNonloc[mesh.outdim*mesh.outdim*mesh.dVertex];

    int i = 0, rTdx = 0, Rdx = 0;
    // [x 13] kernel_val [mesh.outdim*mesh.outdim]
    double kernel_val[mesh.outdim*mesh.outdim];

    double rTdet = 0;
    double physical_quad[dim];
    double reference_quad[dim];
    double psi_value[mesh.dVertex];
    //double psi_value_test[3] = {20., 30., 50.};
    double reTriangle_list[36 * mesh.dVertex * dim];
    doubleVec_tozero(reTriangle_list, 36 * mesh.dVertex * dim);

    //[DEBUG]
    //printf("\nouterInt_full----------------------------------------\n");
    for (k = 0; k < quadRule.nPx; k++) {
        //printf("k %i, quadRule.nPx %i\n", k, quadRule.nPx);
        toPhys(aT.E, &(quadRule.Px[dim * k]), mesh.dim, x);
        //printf("\nInner Integral, Iterate %i\n", k);
        //printf("\Physical x [%17.16e, %17.16e]\n",  x[0], x[1]);
        //innerInt_retriangulate(x, aT, bT, quadRule, sqdelta, &innerLocal, innerNonloc);

        //is_placePointOnCap = true;
        Rdx = method_retriangulate(x, bT, mesh, reTriangle_list, conf.is_placePointOnCap); // innerInt_retriangulate
        //Rdx = baryCenterMethod(x, bT, mesh, reTriangle_list, is_placePointOnCap);
        //Rdx = quadRule.interactionMethod(x, bT, mesh, reTriangle_list);

        //[DEBUG]
        //printf("Retriangulation Rdx %i\n", Rdx);
        //for (i=0;i<Rdx;i++){
        //printf("[%17.16e, %17.16e]\n", reTriangle_list[2 * 3 * i], reTriangle_list[2 * 3 * i+1]);
        //printf("[%17.16e, %17.16e]\n", reTriangle_list[2 * 3 * i+2], reTriangle_list[2 * 3 * i+3]);
        //printf("[%17.16e, %17.16e]\n", reTriangle_list[2 * 3 * i+4], reTriangle_list[2 * 3 * i+5]);
        //printf("absDet %17.16e\n", absDet(&reTriangle_list[2 * 3 * i]));
        //}
        // [x 14] doubleVec_tozero(innerLocal, mesh.outdim*mesh.outdim);
        //innerLocal = 0.0;
        doubleVec_tozero(innerLocal, mesh.outdim*mesh.outdim);

        // [x 15] doubleVec_tozero(innerLocal, mesh.outdim*mesh.outdim*mesh.dVertex);
        // doubleVec_tozero(innerNonloc, mesh.dVertex);
        doubleVec_tozero(innerNonloc, mesh.outdim*mesh.outdim*mesh.dVertex);

        if (Rdx == 0) {
        } else {
            //printf("\nInner Integral\n");
            for (rTdx = 0; rTdx < Rdx; rTdx++) {
                //printf("rTdx %i \n",rTdx);
                for (i = 0; i < quadRule.nPy; i++) {
                    // Push quadrature point P[i] to physical triangle reTriangle_list[rTdx] (of the retriangulation!)
                    toPhys(&reTriangle_list[dim * mesh.dVertex * rTdx], &(quadRule.Py[dim * i]), physical_quad);
                    // Determinant of Triangle of retriangulation
                    rTdet = absDet(&reTriangle_list[dim * mesh.dVertex * rTdx]);
                    // inner Local integral with ker
                    model_kernel(x, aT.label, physical_quad, bT.label, mesh.sqdelta, kernel_val);
                    // [x 16]
                    // INNER LOCAL ORDER [(0,0), (0,1), (1,0), (1,1)] = KERNEL ORDER
                    for (int o=0; o<mesh.outdim*mesh.outdim; o++){
                              innerLocal[o] += kernel_val[o] * quadRule.dy[i] * rTdet; // Local Term
                    }
                    //innerLocal += kernel_val * quadRule.dy[i] * rTdet; // Local Term

                    // Pull resulting physical point ry to the (underlying!) reference Triangle aT.
                    toRef(bT.E, physical_quad, reference_quad);
                    // Evaluate ker on physical quad (note this is ker')
                    model_kernel(physical_quad, bT.label, x, aT.label, mesh.sqdelta, kernel_val);
                    // Evaluate basis function on resulting reference quadrature point
                    model_basisFunction(reference_quad, mesh.dim, psi_value);

                    // [17]
                    // INNER NON-LOCAL ORDER
                    // [(b 0, ker (0,0)), (b 0, ker (0,1)), (b 0, ker (1,0)), (b 0, ker (1,1)),
                    //  (b 1, ker (0,0)), (b 1, ker (0,1)), (b 1, ker (1,0)), (b 1, ker (1,1)),
                    //  (b 2, ker (0,0)), (b 2, ker (0,1)), (b 2, ker (1,0)), (b 2, ker (1,1))]
                    //  = (PSI ORDER) * (KERNEL ORDER)

                    for (b = 0; b < mesh.dVertex*mesh.outdim*mesh.outdim; b++) {
                    // for (b = 0; b < mesh.dVertex; b++) {
                        // [x 18]
                        innerNonloc[b] +=
                                psi_value[b/(mesh.outdim*mesh.outdim)] *
                                kernel_val[b%(mesh.outdim*mesh.outdim)] *
                                quadRule.dy[i] * rTdet; // Nonlocal Term
                        //innerNonloc[b] += psi_value[b] * kernel_val * quadRule.dy[i] * rTdet; // Nonlocal Term
                    }
                    //[DEBUG]
                    //printf("i %i \n",i);
                    //printf("GAM %17.16e\n", ker * dy[i] * rTdet);
                    //printf("Basis0 %17.16e\n", psi_value[0]);
                    //printf("Basis1 %17.16e\n", psi_value[1]);
                    //printf("Basis2 %17.16e\n", psi_value[2]);
                }
                //printf("Chris: v0 %17.16e\nv1 %17.16e\nv2 %17.16e\n", innerNonloc[0], innerNonloc[1], innerNonloc[2]);
                //printf("Chris: v %17.16e\n", innerLocal);
            }
        }

        // TERM LOCAL & TERM NON-LOCAL ORDER
        // Note: This order is not trivially obtained from innerNonloc, as b switches in between.
        // However it mimics the matrix which results from the multiplication.
        //                                      Kernel switches back here. v
        // [(a 0, b 0, ker (0,0)), (a 0, b 0, ker (0,1)), (a 0, b 1, ker (0,0)), (a 0, b 1, ker (0,1)), (a 0, b 2, ker (0,0)), (a 0, b 2, ker (0,1)),
        //  (a 0, b 0, ker (1,0)), (a 0, b 0, ker (1,1)), (a 0, b 0, ker (1,0)), (a 0, b 1, ker (1,1)), (a 0, b 2, ker (1,0)), (a 0, b 2, ker (1,1)),
        //  (a 1, b 0, ker (0,0)), (a 1, b 0, ker (0,1)), (a 1, b 1, ker (0,0)), (a 1, b 1, ker (0,1)), (a 1, b 2, ker (0,0)), (a 1, b 2, ker (0,1)),
        //  (a 1, b 0, ker (1,0)), (a 1, b 0, ker (1,1)), (a 1, b 0, ker (1,0)), (a 1, b 1, ker (1,1)), (a 1, b 2, ker (1,0)), (a 0, b 2, ker (1,1)),
        //  (a 2, b 0, ker (0,0)), (a 2, b 0, ker (0,1)), (a 2, b 1, ker (0,0)), (a 2, b 1, ker (0,1)), (a 2, b 2, ker (0,0)), (a 2, b 2, ker (0,1)),
        //  (a 2, b 0, ker (1,0)), (a 2, b 0, ker (1,1)), (a 2, b 0, ker (1,0)), (a 2, b 1, ker (1,1)), (a 2, b 2, ker (1,0)), (a 2, b 2, ker (1,1))]

        //  = (PSI ORDER) * (PSI ORDER) * (INNER LOCAL ORDER)
        //  = (PSI ORDER) *' (INNER NON-LOCAL ORDER)

        //printf("Local %17.16e\n", innerLocal);
        //printf("Nonloc [%17.16e, %17.16e, %17.16e, %17.16e] \n", innerNonloc[0], innerNonloc[1], innerNonloc[2], innerNonloc[3]);

        // [x 19] for (a = 0; a < mesh.dVertex*mesh.outdim; a++) {
        for (a = 0; a < mesh.dVertex * mesh.outdim; a++) {
            // [x 20] for (b = 0; b < mesh.dVertex*mesh.outdim; b++) {
            for (b = 0; b < mesh.dVertex * mesh.outdim; b++) {
                // [x 21] termLocal[mesh.dVertex * mesh.outputdim * a + b] +=
                termLocal[mesh.dVertex * mesh.outdim * a + b] +=
                        2 * aT.absDet * quadRule.psix(a/mesh.outdim, k) * quadRule.psix(b/mesh.outdim, k) * quadRule.dx[k] *
                        innerLocal[mesh.outdim*(a%mesh.outdim) + (b%mesh.outdim)]; //innerLocal
                // psi_value_test[a/mesh.outdim]*psi_value_test[b/mesh.outdim]+innerLocal[mesh.outdim*(a%mesh.outdim) + (b%mesh.outdim)];
                //printf("a %6.4e, b %6.4e, innerLocal %6.4e \n", psi_value_test[a/mesh.outdim], psi_value_test[b/mesh.outdim], innerLocal[mesh.outdim*(a%mesh.outdim) + (b%mesh.outdim)]);
                // [x 22] 2 * aT.absDet * quadRule.psix(a/mesh.outdim, k) * quadRule.psix(b/mesh.outdim, k) * quadRule.dx[k] * ...

                //printf("quadRule.psix(%i,%i) %17.16e\nquadRule.psix(%i,%i) %17.16e \n", a,k, quadRule.psix(a,k), b,k,quadRule.psix(b,k));
                // [x 24] termNonloc[mesh.dVertex * mesh.outputdim * a + b] +=
                termNonloc[mesh.dVertex * mesh.outdim * a + b] +=
                        2 * aT.absDet * quadRule.psix(a/mesh.outdim, k) * quadRule.dx[k] *
                        innerNonloc[(a%mesh.outdim)*mesh.outdim +
                                    mesh.outdim*mesh.outdim*(b/mesh.outdim) +
                                    (b%mesh.outdim)];
                //printf("a %6.4e, innerNonloc %6.4e \n", psi_value_test[a/mesh.outdim],
                // innerNonloc[(a%mesh.outdim)*mesh.outdim + mesh.outdim*mesh.outdim*(b/mesh.outdim) + (b%mesh.outdim)]);
                // [x 25] 2 * aT.absDet * quadRule.psix(a/mesh.outdim, k) * quadRule.dx[k] *
                //2 * aT.absDet * quadRule.psix(a, k) * quadRule.dx[k] * innerNonloc[b]; //innerNonloc
            }
        }
    }
}

void
integrate_baryCenter(const ElementType &aT, const ElementType &bT, const QuadratureType &quadRule, const MeshType &mesh,
                     const ConfigurationType &conf, bool is_firstbfslayer, double *termLocal, double *termNonloc) {
    const int dim = mesh.dim;
    int k = 0, a = 0, b = 0;
    double x[dim];
    // [x 26]
    double innerLocal[mesh.outdim*mesh.outdim];

    // [x 27]
    double innerNonloc[mesh.outdim*mesh.outdim*mesh.dVertex];

    int i = 0;
    double kernel_val[mesh.outdim*mesh.outdim];
    double rTdet = 0.0;
    double physical_quad[dim];
    double reTriangle_list[36 * mesh.dVertex * dim];
    doubleVec_tozero(reTriangle_list, 36 * mesh.dVertex * dim);

    //[DEBUG]
    //printf("\nouterInt_full----------------------------------------\n");
    for (k = 0; k < quadRule.nPx; k++) {
        toPhys(aT.E, &(quadRule.Px[dim * k]), mesh.dim, x);

        doubleVec_tozero(innerLocal, mesh.outdim*mesh.outdim);
        // [x 28]
        doubleVec_tozero(innerNonloc, mesh.outdim*mesh.outdim*mesh.dVertex);
        if (method_baryCenter(x, bT, mesh, reTriangle_list, false)) {
            for (i = 0; i < quadRule.nPy; i++) {
                // Push quadrature point P[i] to physical triangle reTriangle_list[rTdx] (of the retriangulation!)
                toPhys(bT.E, &(quadRule.Py[dim * i]), mesh.dim, physical_quad);
                // Determinant of Triangle of retriangulation
                rTdet = absDet(bT.E, mesh.dim);
                // inner Local integral with ker
                model_kernel(x, aT.label, physical_quad, bT.label, mesh.sqdelta, kernel_val);
                // [x 29]
                for (int o=0; o<mesh.outdim*mesh.outdim; o++) {
                    innerLocal[o] += rTdet * kernel_val[o] * quadRule.dy[i]; // Local Term
                }
                // Evaluate ker on physical quad (note this is ker')
                model_kernel(physical_quad, bT.label, x, aT.label, mesh.sqdelta, kernel_val);
                // Evaluate basis function on resulting reference quadrature point

                // INNER NON-LOCAL ORDER
                // [(b 0, ker (0,0)), (b 0, ker (0,1)), (b 0, ker (1,0)), (b 0, ker (1,1)),
                //  (b 1, ker (0,0)), (b 1, ker (0,1)), (b 1, ker (1,0)), (b 1, ker (1,1)),
                //  (b 2, ker (0,0)), (b 2, ker (0,1)), (b 2, ker (1,0)), (b 2, ker (1,1))]
                //  = (PSI ORDER) * (KERNEL ORDER)

                for (b = 0; b < mesh.dVertex*mesh.outdim*mesh.outdim; b++) {
                    // [x 31]
                    //innerNonloc[b] += quadRule.psiy(b, i) * kernel_val * quadRule.dy[i] * rTdet; // Nonlocal Term
                    innerNonloc[b] +=
                            quadRule.psiy(b/(mesh.outdim*mesh.outdim), i) *
                            kernel_val[b%(mesh.outdim*mesh.outdim)] *
                            quadRule.dy[i] * rTdet; // Nonlocal Term
                }
            }
        }
        //printf("Local %17.16e\n", innerLocal);
        //printf("Nonloc [%17.16e, %17.16e, %17.16e, %17.16e] \n", innerNonloc[0], innerNonloc[1], innerNonloc[2], innerNonloc[3]);

        // TERM LOCAL & TERM NON-LOCAL ORDER
        // Note: This order is not trivially obtained from innerNonloc, as b switches in between.
        // However it mimics the matrix which results from the multiplication.
        //                                      Kernel switches back here. v
        // [(a 0, b 0, ker (0,0)), (a 0, b 0, ker (0,1)), (a 0, b 1, ker (0,0)), (a 0, b 1, ker (0,1)), (a 0, b 2, ker (0,0)), (a 0, b 2, ker (0,1)),
        //  (a 0, b 0, ker (1,0)), (a 0, b 0, ker (1,1)), (a 0, b 0, ker (1,0)), (a 0, b 1, ker (1,1)), (a 0, b 2, ker (1,0)), (a 0, b 2, ker (1,1)),
        //  (a 1, b 0, ker (0,0)), (a 1, b 0, ker (0,1)), (a 1, b 1, ker (0,0)), (a 1, b 1, ker (0,1)), (a 1, b 2, ker (0,0)), (a 1, b 2, ker (0,1)),
        //  (a 1, b 0, ker (1,0)), (a 1, b 0, ker (1,1)), (a 1, b 0, ker (1,0)), (a 1, b 1, ker (1,1)), (a 1, b 2, ker (1,0)), (a 0, b 2, ker (1,1)),
        //  (a 2, b 0, ker (0,0)), (a 2, b 0, ker (0,1)), (a 2, b 1, ker (0,0)), (a 2, b 1, ker (0,1)), (a 2, b 2, ker (0,0)), (a 2, b 2, ker (0,1)),
        //  (a 2, b 0, ker (1,0)), (a 2, b 0, ker (1,1)), (a 2, b 0, ker (1,0)), (a 2, b 1, ker (1,1)), (a 2, b 2, ker (1,0)), (a 2, b 2, ker (1,1))]

        //  = (PSI ORDER) * (PSI ORDER) * (INNER LOCAL ORDER)
        //  = (PSI ORDER) *' (INNER NON-LOCAL ORDER)

        // [x 32]
        for (a = 0; a < mesh.dVertex*mesh.outdim; a++) {
            // [x 33]
            for (b = 0; b < mesh.dVertex*mesh.outdim; b++) {
                // [x 33]
                termLocal[mesh.dVertex * mesh.outdim * a + b] +=
                    2 * aT.absDet *
                    quadRule.psix(a/mesh.outdim, k) * quadRule.psix(b/mesh.outdim, k) *
                    quadRule.dx[k] *
                    innerLocal[mesh.outdim*(a%mesh.outdim) + (b%mesh.outdim)]; //innerLocal
                    // [x 34]
                    // [x 35]
                //2 * aT.absDet * quadRule.psix(a, k) * quadRule.psix(b, k) * quadRule.dx[k] * innerLocal;

                // [x 36]
                termNonloc[mesh.dVertex * mesh.outdim * a + b] +=
                // [x 37]
                //2 * aT.absDet * quadRule.psix(a, k) * quadRule.dx[k] * innerNonloc[b];
                    2 * aT.absDet *
                    quadRule.psix(a/mesh.outdim, k) *
                    quadRule.dx[k] *
                    innerNonloc[(a%mesh.outdim)*mesh.outdim +
                                mesh.outdim*mesh.outdim*(b/mesh.outdim) +
                                (b%mesh.outdim)];//innerNonloc
            }
        }
    }
}


void
integrate_subSuperSetBalls(const ElementType &aT, const ElementType &bT, const QuadratureType &quadRule, const MeshType &mesh,
                     const ConfigurationType &conf, bool is_firstbfslayer, double *termLocal, double *termNonloc) {

    double averageWeights[mesh.dVertex];
    doubleVec_tozero(averageWeights, mesh.dVertex);
    //int doesInteract;

    if (conf.integration_method == "superSetBall"){
        averageWeights[0]=1.0;
        averageWeights[1]=1.0;
        averageWeights[2]=1.0;
    } else if (conf.integration_method == "subSetBall"){
        //doesInteract = 3;
        averageWeights[2]=1.0;
    } else if (conf.integration_method == "averageBall") {
        averageWeights[0]= 0.0;
        averageWeights[1]= 1.0;
        averageWeights[2]= 1.0;
    } else {
            cout << "Error in integrate_subSuperSetBalls: No such integration_method: " <<
            conf.integration_method << endl;
            abort();
    }

    const int dim = mesh.dim;
    int k = 0, a = 0, b = 0;
    double x[dim];
    // [x 26]
    double innerLocal[mesh.outdim*mesh.outdim];

    // [x 27]
    double innerNonloc[mesh.outdim*mesh.outdim*mesh.dVertex];

    int i = 0;
    double kernel_val[mesh.outdim*mesh.outdim];
    double rTdet = 0.0;
    double physical_quad[dim];


    //[DEBUG]
    //printf("\nouterInt_full----------------------------------------\n");
    for (k = 0; k < quadRule.nPx; k++) {
        toPhys(aT.E, &(quadRule.Px[dim * k]), mesh.dim, x);

        doubleVec_tozero(innerLocal, mesh.outdim*mesh.outdim);
        // [x 28]
        doubleVec_tozero(innerNonloc, mesh.outdim*mesh.outdim*mesh.dVertex);
        int nContained = method_subSuperSetBalls(x, bT, mesh);
        if (nContained >= 1) {
            //averageWeight = isAverage ? 0.5 * static_cast<double>(interaction) : 1.0;

            for (i = 0; i < quadRule.nPy; i++) {
                // Push quadrature point P[i] to physical triangle reTriangle_list[rTdx] (of the retriangulation!)
                toPhys(bT.E, &(quadRule.Py[dim * i]), mesh.dim, physical_quad);
                // Determinant of Triangle of retriangulation
                rTdet = absDet(bT.E, mesh.dim);
                // inner Local integral with ker
                model_kernel(x, aT.label, physical_quad, bT.label, mesh.sqdelta, kernel_val);
                // [x 29]
                for (int o=0; o<mesh.outdim*mesh.outdim; o++) {
                    innerLocal[o] += averageWeights[nContained-1] * rTdet * kernel_val[o] * quadRule.dy[i]; // Local Term
                }
                // Evaluate ker on physical quad (note this is ker')
                model_kernel(physical_quad, bT.label, x, aT.label, mesh.sqdelta, kernel_val);
                // Evaluate basis function on resulting reference quadrature point

                // INNER NON-LOCAL ORDER
                // [(b 0, ker (0,0)), (b 0, ker (0,1)), (b 0, ker (1,0)), (b 0, ker (1,1)),
                //  (b 1, ker (0,0)), (b 1, ker (0,1)), (b 1, ker (1,0)), (b 1, ker (1,1)),
                //  (b 2, ker (0,0)), (b 2, ker (0,1)), (b 2, ker (1,0)), (b 2, ker (1,1))]
                //  = (PSI ORDER) * (KERNEL ORDER)

                for (b = 0; b < mesh.dVertex*mesh.outdim*mesh.outdim; b++) {
                    // [x 31]
                    //innerNonloc[b] += quadRule.psiy(b, i) * kernel_val * quadRule.dy[i] * rTdet; // Nonlocal Term
                    innerNonloc[b] += averageWeights[nContained-1] *
                            quadRule.psiy(b/(mesh.outdim*mesh.outdim), i) *
                            kernel_val[b%(mesh.outdim*mesh.outdim)] *
                            quadRule.dy[i] * rTdet; // Nonlocal Term
                }
            }
        }
        //printf("Local %17.16e\n", innerLocal);
        //printf("Nonloc [%17.16e, %17.16e, %17.16e, %17.16e] \n", innerNonloc[0], innerNonloc[1], innerNonloc[2], innerNonloc[3]);

        // TERM LOCAL & TERM NON-LOCAL ORDER
        // Note: This order is not trivially obtained from innerNonloc, as b switches in between.
        // However it mimics the matrix which results from the multiplication.
        //                                      Kernel switches back here. v
        // [(a 0, b 0, ker (0,0)), (a 0, b 0, ker (0,1)), (a 0, b 1, ker (0,0)), (a 0, b 1, ker (0,1)), (a 0, b 2, ker (0,0)), (a 0, b 2, ker (0,1)),
        //  (a 0, b 0, ker (1,0)), (a 0, b 0, ker (1,1)), (a 0, b 0, ker (1,0)), (a 0, b 1, ker (1,1)), (a 0, b 2, ker (1,0)), (a 0, b 2, ker (1,1)),
        //  (a 1, b 0, ker (0,0)), (a 1, b 0, ker (0,1)), (a 1, b 1, ker (0,0)), (a 1, b 1, ker (0,1)), (a 1, b 2, ker (0,0)), (a 1, b 2, ker (0,1)),
        //  (a 1, b 0, ker (1,0)), (a 1, b 0, ker (1,1)), (a 1, b 0, ker (1,0)), (a 1, b 1, ker (1,1)), (a 1, b 2, ker (1,0)), (a 0, b 2, ker (1,1)),
        //  (a 2, b 0, ker (0,0)), (a 2, b 0, ker (0,1)), (a 2, b 1, ker (0,0)), (a 2, b 1, ker (0,1)), (a 2, b 2, ker (0,0)), (a 2, b 2, ker (0,1)),
        //  (a 2, b 0, ker (1,0)), (a 2, b 0, ker (1,1)), (a 2, b 0, ker (1,0)), (a 2, b 1, ker (1,1)), (a 2, b 2, ker (1,0)), (a 2, b 2, ker (1,1))]

        //  = (PSI ORDER) * (PSI ORDER) * (INNER LOCAL ORDER)
        //  = (PSI ORDER) *' (INNER NON-LOCAL ORDER)

        // [x 32]
        for (a = 0; a < mesh.dVertex*mesh.outdim; a++) {
            // [x 33]
            for (b = 0; b < mesh.dVertex*mesh.outdim; b++) {
                // [x 33]
                termLocal[mesh.dVertex * mesh.outdim * a + b] +=
                        2 * aT.absDet *
                        quadRule.psix(a/mesh.outdim, k) * quadRule.psix(b/mesh.outdim, k) *
                        quadRule.dx[k] *
                        innerLocal[mesh.outdim*(a%mesh.outdim) + (b%mesh.outdim)]; //innerLocal
                // [x 34]
                // [x 35]
                //2 * aT.absDet * quadRule.psix(a, k) * quadRule.psix(b, k) * quadRule.dx[k] * innerLocal;

                // [x 36]
                termNonloc[mesh.dVertex * mesh.outdim * a + b] +=
                        // [x 37]
                        //2 * aT.absDet * quadRule.psix(a, k) * quadRule.dx[k] * innerNonloc[b];
                        2 * aT.absDet *
                        quadRule.psix(a/mesh.outdim, k) *
                        quadRule.dx[k] *
                        innerNonloc[(a%mesh.outdim)*mesh.outdim +
                                    mesh.outdim*mesh.outdim*(b/mesh.outdim) +
                                    (b%mesh.outdim)];//innerNonloc
            }
        }
    }
}

void integrate_baryCenterRT(const ElementType &aT, const ElementType &bT, const QuadratureType &quadRule,
                            const MeshType &mesh, const ConfigurationType &conf, bool is_firstbfslayer, double *termLocal,
                            double *termNonloc) {
    const int dim = mesh.dim;
    int k = 0, a = 0, b = 0;
    double physical_quad[dim], reference_quad[dim], psix[mesh.dVertex];
    double bTbaryC[dim];
    // [x 37]
    double innerLocal[mesh.outdim*mesh.outdim];// = 0;
    // [x 38]
    double innerNonloc[mesh.outdim*mesh.outdim*mesh.dVertex];

    int i = 0, Rdx, rTdx;
    // [x 39]
    double kernel_val[mesh.outdim*mesh.outdim];
    double rTdet = 0, bTdet = 0;
    double y[dim];
    double reTriangle_list[36 * mesh.dVertex * dim];
    doubleVec_tozero(reTriangle_list, 36 * mesh.dVertex * dim);

    //[DEBUG]
    //printf("\nouterInt_full----------------------------------------\n");
    baryCenter(bT.E, &bTbaryC[0]);
    Rdx = method_retriangulate(bTbaryC, aT, mesh, reTriangle_list, conf.is_placePointOnCap);

    if (!Rdx) {
        return;
    } else {
        // Determinant of Triangle of retriangulation
        bTdet = absDet(bT.E);

        for (rTdx = 0; rTdx < Rdx; rTdx++) {
            rTdet = absDet(&reTriangle_list[dim * mesh.dVertex * rTdx]);

            for (k = 0; k < quadRule.nPx; k++) {
                toPhys(&reTriangle_list[dim * mesh.dVertex * rTdx], &(quadRule.Px[dim * k]), mesh.dim,
                       physical_quad);

                // Compute Integral over Triangle bT
                doubleVec_tozero(innerLocal, mesh.outdim * mesh.outdim);
                doubleVec_tozero(innerNonloc, mesh.dVertex);

                for (i = 0; i < quadRule.nPy; i++) {
                    // Push quadrature point P[i] to physical triangle reTriangle_list[rTdx] (of the retriangulation!)
                    toPhys(bT.E, &(quadRule.Py[dim * i]), mesh.dim, y);
                    // inner Local integral with ker
                    // Local Term
                    model_kernel(physical_quad, aT.label, y, bT.label, mesh.sqdelta, kernel_val);
                    // [x 40]
                    for (int o = 0; o < mesh.outdim * mesh.outdim; o++) {
                        // innerLocal[o] += kernel_val[o] * quadRule.dy[i] * rTdet; // Local Term
                        innerLocal[o] += kernel_val[o] * quadRule.dy[i] * bTdet;
                    }
                    // Evaluate kernel on physical quad (note this is kernel')
                    model_kernel(y, bT.label, physical_quad, aT.label, mesh.sqdelta, kernel_val);
                    // Evaluate basis function on resulting reference quadrature point
                    // [x 41]
                    for (b = 0; b < mesh.dVertex * mesh.outdim * mesh.outdim; b++) {
                        // [42]
                        innerNonloc[b] += quadRule.psiy(b / (mesh.outdim * mesh.outdim), i) *
                                          // kernel_val[b%(mesh.outputdim*mesh.outputdim)] * quadRule.dy[i] * rTdet; // Nonlocal Term
                                          kernel_val[b % (mesh.outdim * mesh.outdim)] *
                                          quadRule.dy[i] * bTdet; // Nonlocal Term

                        //innerNonloc[b] +=
                        //        quadRule.psiy(b, i) * kernel_val * quadRule.dy[i] * bTdet; // Nonlocal Term
                    }
                }

                toRef(aT.E, physical_quad, reference_quad);
                model_basisFunction(reference_quad, psix);

                // [x 43]
                for (a = 0; a < mesh.dVertex * mesh.outdim; a++) {
                    // [x 44]
                    for (b = 0; b < mesh.dVertex * mesh.outdim; b++) {
                        // [x 45]
                        termLocal[mesh.dVertex * mesh.outdim * a + b] +=
                                //2 * rTdet * psix[a] * psix[b] * quadRule.dx[k] * innerLocal; //innerLocal
                                2 * rTdet * psix[a / mesh.outdim] * psix[b / mesh.outdim] *
                                quadRule.dx[k] * innerLocal[mesh.outdim * (a % mesh.outdim) + (b % mesh.outdim)];

                        // [x 47]
                        termNonloc[mesh.dVertex * mesh.outdim * a + b] +=
                                // [48]
                                2 * rTdet * psix[a / mesh.outdim] * quadRule.dx[k] *
                                innerNonloc[mesh.outdim * (a % mesh.outdim) + b % mesh.outdim]; //innerNonloc
                        //2 * rTdet * psix[a] * quadRule.dx[k] * innerNonloc[b]; //innerNonloc
                    }
                }
            }
        }
    }
}
// Helpers -------------------------------------------------------------------------------------------------------------
// Sub Set Method ------------------------------------------------------------------------------------------------------
// Super, Sub and Average Set Method -----------------------------------------------------------------------------------
int method_subSuperSetBalls(const double * x, const ElementType & T, const MeshType & mesh){
    int nContained = 0;
    for (int i=0; i<mesh.dVertex; i++){
        nContained += (vec_sqL2dist(x, &T.E[mesh.dim * i], mesh.dim) < mesh.sqdelta);
    }
    return nContained;
}

// Bary Center Method --------------------------------------------------------------------------------------------------
int method_baryCenter(const double * x_center, const ElementType & T, const MeshType & mesh, double * reTriangle_list, int is_placePointOnCap){
    int i,k;
    double distance;
    arma::vec baryC(mesh.dim);
    //void baryCenter(const int dim, const double * E, double * bary);
    baryCenter(mesh.dim, T.E, &baryC[0]);
    distance = vec_sqL2dist(x_center, &baryC[0], mesh.dim);

    if (distance > mesh.sqdelta){
        return 0;
    } else {
        for (i=0; i<mesh.dim; i++){
            for(k=0; k<mesh.dVertex; k++) {
                reTriangle_list[2 * k + i] = T.E[2 * k + i];
            }
        }
        return -1;
    }
}

// Retriangulation Method ----------------------------------------------------------------------------------------------
bool inTriangle(const double * y_new, const double * p, const double * q, const double * r,
                const double *  nu_a, const double * nu_b, const double * nu_c){
    bool a, b, c;
    double vec[2];

    doubleVec_subtract(y_new, p, vec, 2);
    a = vec_dot(nu_a, vec , 2) >= 0;

    doubleVec_subtract(y_new, q, vec, 2);
    b = vec_dot(nu_b, vec , 2) >= 0;

    doubleVec_subtract(y_new, r, vec, 2);
    c = vec_dot(nu_c, vec , 2) >= 0;

    return a && b && c;
}

int placePointOnCap(const double * y_predecessor, const double * y_current,
                    const double * x_center, const double sqdelta, const double * TE,
                    const double * nu_a, const double * nu_b, const double * nu_c,
                    const double orientation, const int Rdx, double * R){
    // Place a point on the cap.
    //y_predecessor = &R[2*(Rdx-1)];
    double y_new[2], s_midpoint[2], s_projectionDirection[2];
    double scalingFactor;

    doubleVec_midpoint(y_predecessor, y_current, s_midpoint, 2);
    // Note, this yields the left normal from y_predecessor to y0
    rightNormal(y_current, y_predecessor, orientation, s_projectionDirection);
    // Simple way
    scalingFactor = sqrt( sqdelta / vec_dot(s_projectionDirection, s_projectionDirection, 2));
    doubleVec_scale(scalingFactor, s_projectionDirection, s_projectionDirection, 2);
    doubleVec_add(x_center, s_projectionDirection, y_new, 2);

    if ( inTriangle(y_new, &TE[0], &TE[2], &TE[4], nu_a, nu_b, nu_c)){
        // Append y_new (Point on the cap)
        doubleVec_copyTo(y_new, &R[2*Rdx], 2);
        return 1;
    } else {
        return 0;
    }
}

bool isFullyContained(const ElementType & aT, const ElementType & bT, const MeshType & mesh){
    //cout << "Max Diameter: " << mesh.maxDiameter << endl;
    //abort();

    double abary[mesh.dim], bbary[mesh.dim];

    baryCenter(mesh.dim, aT.E, abary);
    baryCenter(mesh.dim, bT.E, bbary);

    double l2dist = vec_sqL2dist(abary, bbary, mesh.dim);

    return ((mesh.delta - 2*mesh.maxDiameter) > 0) && ( l2dist < pow(mesh.delta - 2*mesh.maxDiameter,2) );
}

int method_retriangulate(const double * xCenter, const ElementType & T,
                         const MeshType & mesh, double * reTriangleList,
                         int isPlacePointOnCap){
    // C Variables and Arrays.
    int i=0, k=0, edgdx0=0, edgdx1=0, Rdx=0;
    double v=0, lam1=0, lam2=0, term1=0, term2=0;
    double nu_a[2], nu_b[2], nu_c[2]; // Normals
    arma::vec p(2);
    arma::vec q(2);
    arma::vec a(2);
    arma::vec b(2);
    arma::vec y1(2);
    arma::vec y2(2);
    arma::vec vec_x_center(xCenter, 2);
    double orientation;

    bool is_onEdge=false, is_firstPointLiesOnVertex=true;
    // The upper bound for the number of required points is 9
    // Hence 9*2 is an upper bound to encode all resulting triangles
    // Hence we can hardcode how much space needs to bee allocated
    // (This upper bound is thight! Check Christian Vollmann's thesis for more information.)

    double R[9*2]; // Vector containing all intersection points.
    doubleVec_tozero(R, 9*2);

    // Compute Normals of the Triangle
    orientation = -signDet(T.E);
    rightNormal(&T.E[0], &T.E[2], orientation, nu_a);
    rightNormal(&T.E[2], &T.E[4], orientation, nu_b);
    rightNormal(&T.E[4], &T.E[0], orientation, nu_c);

    for (k=0; k<3; k++){
        edgdx0 = k;
        edgdx1 = (k+1) % 3;

        doubleVec_copyTo(&T.E[2*edgdx0], &p[0], 2);
        doubleVec_copyTo(&T.E[2*edgdx1], &q[0], 2);

        a = q - vec_x_center;
        b = p - q;

        if (vec_sqL2dist(&p[0], xCenter, 2) <= mesh.sqdelta){
            doubleVec_copyTo(&p[0], &R[2*Rdx], 2);
            is_onEdge = false; // This point does not lie on the edge.
            Rdx += 1;
        }
        // PQ-Formula to solve quadratic problem
        v = pow( dot(a, b), 2) - (dot(a, a) - mesh.sqdelta) * dot(b, b);
        // If there is no sol to the quadratic problem, there is nothing to do.
        if (v >= 0){
            term1 = - dot(a, b) / dot(b, b);
            term2 = sqrt(v) / dot(b, b);

            // Vieta's Formula for computing the roots
            if (term1 > 0){
                lam1 = term1 + term2;
                lam2 = 1/lam1*(dot(a, a) - mesh.sqdelta) / dot(b, b);
            } else {
                lam2 = term1 - term2;
                lam1 = 1/lam2*(dot(a, a) - mesh.sqdelta) / dot(b, b);
            }
            y1 = lam1*b + q;
            y2 = lam2*b + q;

            // Check whether the first lambda "lies on the Triangle".
            if ((0 <= lam1) && (lam1 <= 1)){
                is_firstPointLiesOnVertex = is_firstPointLiesOnVertex && (bool)Rdx;
                // Check whether the predecessor lied on the edge
                if (is_onEdge && isPlacePointOnCap){
                    Rdx += placePointOnCap(&R[2*(Rdx-1)], &y1[0], xCenter, mesh.sqdelta, T.E, nu_a, nu_b, nu_c, orientation, Rdx, R);
                }
                // Append y1
                doubleVec_copyTo(&y1[0], &R[2*Rdx], 2);
                is_onEdge = true; // This point lies on the edge.
                Rdx += 1;
            }
            // Check whether the second lambda "lies on the Triangle".
            if ((0 <= lam2) && (lam2 <= 1) && (scal_sqL2dist(lam1, lam2) > 0)){
                is_firstPointLiesOnVertex = is_firstPointLiesOnVertex && (bool)Rdx;

                // Check whether the predecessor lied on the edge
                if (is_onEdge && isPlacePointOnCap){
                    Rdx += placePointOnCap(&R[2*(Rdx-1)], &y2[0], xCenter, mesh.sqdelta, T.E, nu_a, nu_b, nu_c, orientation, Rdx, R);
                }
                // Append y2
                doubleVec_copyTo(&y2[0], &R[2*Rdx], 2);
                is_onEdge = true; // This point lies on the edge.
                Rdx += 1;
            }
        }
    }
    //[DEBUG]
    //(len(RD)>1) cares for the case that either the first and the last point lie on an endge
    // and there is no other point at all.
    //shift=1;
    if (is_onEdge && (!is_firstPointLiesOnVertex && Rdx > 1) && isPlacePointOnCap){
        Rdx += placePointOnCap(&R[2*(Rdx-1)], &R[0], xCenter, mesh.sqdelta, T.E, nu_a, nu_b, nu_c, orientation, Rdx, R);
    }

    // Construct List of Triangles from intersection points
    if (Rdx < 3){
        // In this case the content of the array out_RE will not be touched.
        return 0;
    } else {

        for (k=0; k < (Rdx - 2); k++){
            for (i=0; i<2; i++){
                // i is the index which runs first, then h (which does not exist here), then k
                // hence if we increase i, the *-index (of the pointer) inreases in the same way.
                // if we increase k, there is quite a 'jump'
                reTriangleList[2 * (3 * k + 0) + i] = R[i];
                reTriangleList[2 * (3 * k + 1) + i] = R[2 * (k + 1) + i];
                reTriangleList[2 * (3 * k + 2) + i] = R[2 * (k + 2) + i];
            }
        }
        // Excessing the bound out_Rdx will not lead to an error but simply to corrupted data!

        return Rdx - 2; // So that, it acutally contains the number of triangles in the retriangulation
    }
}


int method_retriangulate(const double * xCenter, const double * TE,
                         double sqdelta, double * reTriangleList,
                         int isPlacePointOnCap){
    // C Variables and Arrays.
    int i=0, k=0, edgdx0=0, edgdx1=0, Rdx=0;
    double v=0, lam1=0, lam2=0, term1=0, term2=0;
    double nu_a[2], nu_b[2], nu_c[2]; // Normals
    arma::vec p(2);
    arma::vec q(2);
    arma::vec a(2);
    arma::vec b(2);
    arma::vec y1(2);
    arma::vec y2(2);
    arma::vec vec_x_center(xCenter, 2);
    double orientation;

    bool is_onEdge=false, is_firstPointLiesOnVertex=true;
    // The upper bound for the number of required points is 9
    // Hence 9*3 is an upper bound to encode all resulting triangles
    // Hence we can hardcode how much space needs to bee allocated
    // (This upper bound is thight! Check Christian Vollmann's thesis for more information.)

    double R[9*2]; // Vector containing all intersection points.
    doubleVec_tozero(R, 9*2);

    // Compute Normals of the Triangle
    orientation = -signDet(TE);
    rightNormal(&TE[0], &TE[2], orientation, nu_a);
    rightNormal(&TE[2], &TE[4], orientation, nu_b);
    rightNormal(&TE[4], &TE[0], orientation, nu_c);

    for (k=0; k<3; k++){
        edgdx0 = k;
        edgdx1 = (k+1) % 3;

        doubleVec_copyTo(&TE[2*edgdx0], &p[0], 2);
        doubleVec_copyTo(&TE[2*edgdx1], &q[0], 2);

        a = q - vec_x_center;
        b = p - q;

        if (vec_sqL2dist(&p[0], xCenter, 2) <= sqdelta){
            doubleVec_copyTo(&p[0], &R[2*Rdx], 2);
            is_onEdge = false; // This point does not lie on the edge.
            Rdx += 1;
        }
        // PQ-Formula to solve quadratic problem
        v = pow( dot(a, b), 2) - (dot(a, a) - sqdelta) * dot(b, b);
        // If there is no sol to the quadratic problem, there is nothing to do.
        if (v >= 0){
            term1 = - dot(a, b) / dot(b, b);
            term2 = sqrt(v) / dot(b, b);

            // Vieta's Formula for computing the roots
            if (term1 > 0){
                lam1 = term1 + term2;
                lam2 = 1/lam1*(dot(a, a) - sqdelta) / dot(b, b);
            } else {
                lam2 = term1 - term2;
                lam1 = 1/lam2*(dot(a, a) - sqdelta) / dot(b, b);
            }
            y1 = lam1*b + q;
            y2 = lam2*b + q;

            // Check whether the first lambda "lies on the Triangle".
            if ((0 <= lam1) && (lam1 <= 1)){
                is_firstPointLiesOnVertex = is_firstPointLiesOnVertex && (bool)Rdx;
                // Check whether the predecessor lied on the edge
                if (is_onEdge && isPlacePointOnCap){
                    Rdx += placePointOnCap(&R[2*(Rdx-1)], &y1[0], xCenter, sqdelta, TE, nu_a, nu_b, nu_c, orientation, Rdx, R);
                }
                // Append y1
                doubleVec_copyTo(&y1[0], &R[2*Rdx], 2);
                is_onEdge = true; // This point lies on the edge.
                Rdx += 1;
            }
            // Check whether the second lambda "lies on the Triangle".
            if ((0 <= lam2) && (lam2 <= 1) && (scal_sqL2dist(lam1, lam2) > 0)){
                is_firstPointLiesOnVertex = is_firstPointLiesOnVertex && (bool)Rdx;

                // Check whether the predecessor lied on the edge
                if (is_onEdge && isPlacePointOnCap){
                    Rdx += placePointOnCap(&R[2*(Rdx-1)], &y2[0], xCenter, sqdelta, TE, nu_a, nu_b, nu_c, orientation, Rdx, R);
                }
                // Append y2
                doubleVec_copyTo(&y2[0], &R[2*Rdx], 2);
                is_onEdge = true; // This point lies on the edge.
                Rdx += 1;
            }
        }
    }
    //[DEBUG]
    //(len(RD)>1) cares for the case that either the first and the last point lie on an endge
    // and there is no other point at all.
    //shift=1;
    if (is_onEdge && (!is_firstPointLiesOnVertex && Rdx > 1) && isPlacePointOnCap){
        Rdx += placePointOnCap(&R[2*(Rdx-1)], &R[0], xCenter, sqdelta, TE, nu_a, nu_b, nu_c, orientation, Rdx, R);
    }

    // Construct List of Triangles from intersection points
    if (Rdx < 3){
        // In this case the content of the array out_RE will not be touched.
        return 0;
    } else {

        for (k=0; k < (Rdx - 2); k++){
            for (i=0; i<2; i++){
                // i is the index which runs first, then h (which does not exist here), then k
                // hence if we increase i, the *-index (of the pointer) inreases in the same way.
                // if we increase k, there is quite a 'jump'
                reTriangleList[2 * (3 * k + 0) + i] = R[i];
                reTriangleList[2 * (3 * k + 1) + i] = R[2 * (k + 1) + i];
                reTriangleList[2 * (3 * k + 2) + i] = R[2 * (k + 2) + i];
            }
        }
        // Excessing the bound out_Rdx will not lead to an error but simply to corrupted data!

        return Rdx - 2; // So that, it acutally contains the number of triangles in the retriangulation
    }
}

// Helpers Peridynamics ------------------------------------------------------------------------------------------------
void setupElement(const MeshType &mesh, const long * Vdx_new, ElementType &T){
    T.matE = arma::vec(mesh.dim*(mesh.dim+1));
    for (int k=0; k<mesh.dVertex; k++) {
        //Vdx = mesh.Triangles(k, Tdx);
        for (int j = 0; j < mesh.dim; j++) {
            T.matE[mesh.dim * k + j] = mesh.Verts(j, Vdx_new[k]);
            //printf ("aT %3.2f ", T.matE[mesh.dim * k + j]);
        }
    }

    // Initialize Structs
    T.E = T.matE.memptr();
    T.absDet = absDet(T.E, mesh.dim);
    T.signDet = static_cast<int>(signDet(T.E, mesh));
    T.dim = mesh.dim;
}
int join(const ElementType &aT, const ElementType &bT, const MeshType &mesh,
         ElementType &aTsorted, ElementType &bTsorted, int * argSortA, int * argSortB){
    //cout << "Welcome to join()" << endl;
    int nEqual = 0;
    int AinB[3], BinA[3];
    const long * aVdx = &(mesh.Triangles(0, aT.Tdx));
    const long * bVdx = &(mesh.Triangles(0, bT.Tdx));
    long aVdxsorted[3], bVdxsorted[3];

    intVec_tozero(AinB, 3);
    intVec_tozero(BinA, 3);

    for (int a=0; a<mesh.dVertex; a++){
        for (int b=0; b<mesh.dVertex; b++) {
            if (aVdx[a] == bVdx[b]) {
                AinB[a] += 1;
                BinA[b] += 1;
                aVdxsorted[nEqual] = aVdx[a];
                argSortA[nEqual] = a;
                bVdxsorted[nEqual] = bVdx[b];
                argSortB[nEqual] = b;
                nEqual += 1;
            }
        }
    }
    int ia = 0, ib = 0;
    for (int i=0; i<3; i++){
        if (!AinB[i]){
            aVdxsorted[nEqual + ia] = aVdx[i];
            argSortA[nEqual + ia] = i;
            ia++;
        }
        if (!BinA[i]){
            bVdxsorted[nEqual + ib] = bVdx[i];
            argSortB[nEqual + ib] = i;
            ib++;
        }
        //printf("%li, %li | %li, %li \n", aVdx[i], bVdx[i], aVdxsorted[i],  bVdxsorted[i] );
        //printf("%i, %i \n", argSortA[i], argSortB[i]);
    }
    setupElement(mesh, aVdxsorted, aTsorted);
    setupElement(mesh, bVdxsorted, bTsorted);
    //abort();
    return nEqual;
}

double traffoCommonVertex0(double * alpha){
    //xi, eta1, eta2, eta3 = alpha;
    double  xi = alpha[0],
            eta1 = alpha[1],
            eta2 = alpha[2],
            eta3 = alpha[3];

    alpha[0] = xi;
    alpha[1] = eta1 * xi;
    alpha[2] = eta2 * xi;
    alpha[3] = eta2 * eta3 * xi;
    return pow(xi,3)*eta2;
}
double traffoCommonVertex1(double * alpha){
    //xi, eta1, eta2, eta3 = alpha;
    double  xi = alpha[0],
            eta1 = alpha[1],
            eta2 = alpha[2],
            eta3 = alpha[3];
    alpha[0] = xi*eta2;
    alpha[1] = xi*eta2*eta3;
    alpha[2] = xi;
    alpha[3] = xi*eta1;

    return pow(xi,3)*eta2;
}

double traffoCommonEdge0( double * alpha){
    double  xi = alpha[0],
            eta1 = alpha[1],
            eta2 = alpha[2],
            eta3 = alpha[3];

    alpha[0] = xi;
    alpha[1] = xi*eta1*eta3;
    alpha[2] = xi*(1. - eta1*eta2);
    alpha[3] = xi*eta1*(1.-eta2);
    return pow(xi,3)*pow(eta1,2);
}

double traffoCommonEdge1( double * alpha){
    double  xi = alpha[0],
            eta1 = alpha[1],
            eta2 = alpha[2],
            eta3 = alpha[3];

    alpha[0] = xi;
    alpha[1] = xi*eta1;
    alpha[2] = xi*(1. - eta1*eta2*eta3);
    alpha[3] = xi*eta1*eta2*(1.-eta3);
    return pow(xi,3)*pow(eta1,2)*eta2;
}

double traffoCommonEdge2( double * alpha){
    double  xi = alpha[0],
            eta1 = alpha[1],
            eta2 = alpha[2],
            eta3 = alpha[3];

    alpha[0] = xi*(1. - eta1*eta2);
    alpha[1] = xi*eta1*(1. - eta2);
    alpha[2] = xi;
    alpha[3] = xi*eta1*eta2*eta3;

    return pow(xi,3)*pow(eta1,2)*eta2;
}

double traffoCommonEdge3( double * alpha){
    double  xi = alpha[0],
            eta1 = alpha[1],
            eta2 = alpha[2],
            eta3 = alpha[3];

    alpha[0] = xi*(1. - eta1*eta2*eta3);
    alpha[1] = xi*eta1*eta2*(1. - eta3);
    alpha[2] = xi;
    alpha[3] = xi*eta1;

    return pow(xi,3)*pow(eta1,2)*eta2;
}

double traffoCommonEdge4( double * alpha){
    double  xi = alpha[0],
            eta1 = alpha[1],
            eta2 = alpha[2],
            eta3 = alpha[3];

    alpha[0] = xi*(1. - eta1*eta2*eta3);
    alpha[1] = xi*eta1*(1. - eta2*eta3);
    alpha[2] = xi;
    alpha[3] = xi*eta1*eta2;

    return pow(xi,3)*pow(eta1,2)*eta2;
}

double traffoIdentical0( double * alpha){
    double  xi = alpha[0],
            eta1 = alpha[1],
            eta2 = alpha[2],
            eta3 = alpha[3];

    alpha[0] = xi;
    alpha[1] = xi*(1. - eta1 + eta1*eta2);
    alpha[2] = xi*(1. - eta1*eta2*eta3);
    alpha[3] = xi*(1. - eta1);

    return pow(xi,3)*pow(eta1,2)*eta2;
}

double traffoIdentical1( double * alpha){
    double  xi = alpha[0],
            eta1 = alpha[1],
            eta2 = alpha[2],
            eta3 = alpha[3];

    alpha[0] = xi*(1. - eta1*eta2*eta3);
    alpha[1] = xi*(1. - eta1);
    alpha[2] = xi;
    alpha[3] = xi*(1. - eta1 + eta1* eta2);

    return pow(xi,3)*pow(eta1,2)*eta2;
}

double traffoIdentical2( double * alpha){
    double  xi = alpha[0],
            eta1 = alpha[1],
            eta2 = alpha[2],
            eta3 = alpha[3];

    alpha[0] = xi;
    alpha[1] = xi*eta1*(1. - eta2 + eta2*eta3);
    alpha[2] = xi*(1. - eta1*eta2);
    alpha[3] = xi*eta1*(1. - eta2);

    return pow(xi,3)*pow(eta1,2)*eta2;
}

double traffoIdentical3( double * alpha){
    double  xi = alpha[0],
            eta1 = alpha[1],
            eta2 = alpha[2],
            eta3 = alpha[3];

    alpha[0] = xi*(1. - eta1*eta2);
    alpha[1] = xi*eta1*(1. - eta2);
    alpha[2] = xi;
    alpha[3] = xi*eta1*(1. - eta2 + eta2*eta3);

    return pow(xi,3)*pow(eta1,2)*eta2;
}

double traffoIdentical4( double * alpha){
    double  xi = alpha[0],
            eta1 = alpha[1],
            eta2 = alpha[2],
            eta3 = alpha[3];

    alpha[0] = xi*(1. - eta1*eta2*eta3);
    alpha[1] = xi*eta1*(1. - eta2*eta3);
    alpha[2] = xi;
    alpha[3] = xi*eta1*(1. - eta2);

    return pow(xi,3)*pow(eta1,2)*eta2;
}

double traffoIdentical5( double * alpha){
    double  xi = alpha[0],
            eta1 = alpha[1],
            eta2 = alpha[2],
            eta3 = alpha[3];

    alpha[0] = xi;
    alpha[1] = xi*eta1*(1. - eta2);
    alpha[2] = xi*(1. - eta1*eta2*eta3);
    alpha[3] = xi*eta1*(1. - eta2*eta3);

    return pow(xi,3)*pow(eta1,2)*eta2;
}
/**
 * @brief Rescales a cube \f$ [-1, 1]^4 \f$ to a cube
 * \f$ [0, 1]^4 \f$. The determinant of this
 * transformation is given by the constant SCALEDET.
 *
 * @param alpha, a single input point of dimension 4.
 */
void scale(double * alpha){
    for(int k=0; k<4; k++){
        alpha[k] = alpha[k]*0.5 + 0.5;
    }
}
/**
 * @brief Maps a tuple of two triangles \f$\left\lbrace z | z_2 \in
 * [0,1], z_1 \ in [0,z_2] \right\rbrace\f$
 * to a tuple of two standard simplices
 * \f$\left\lbrace z | z_1 \in [0,1], z_2 \ in [0,1-z_1] \right\rbrace\f$.
 * The determinant of this transformation is 1.
 *
 * @param alpha, a single input point of dimension 4.
 */
void mirror(double * alpha){
    alpha[0] = alpha[0] - alpha[1];
    alpha[2] = alpha[2] - alpha[3];
}
// [End] Helpers Peridynamics ---------------------------------------------------------------------------------------
#endif //NONLOCAL_ASSEMBLY_INTEGRATION_CPP