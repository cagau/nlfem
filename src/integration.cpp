//
// Created by klar on 16.03.20.
//
#ifndef NONLOCAL_ASSEMBLY_INTEGRATION_CPP
#define NONLOCAL_ASSEMBLY_INTEGRATION_CPP

#include <MeshTypes.h>
#include <mathhelpers.cpp>
#include <model.cpp>

// ___ INTEGRATION DECLARATION _________________________________________________________________________________________

// Integration Routine #################################################################################################
void (*integrate)(     const ElementType & aT,
                       const ElementType & bT,
                    const QuadratureType & quadRule,
                    const MeshType & mesh,
                    double * termLocal, double * termNonloc);

// Integration Methods #################################################################################################
// Methods -------------------------------------------------------------------------------------------------------------
void integrate_retriangulate(    const  ElementType & aT,
                                 const ElementType & bT,
                              const QuadratureType & quadRule,
                              const MeshType & mesh,
                              double * termLocal, double * termNonloc);
void integrate_baryCenter(   const   ElementType & aT,
                            const ElementType & bT,
                           const QuadratureType & quadRule,
                           const MeshType & mesh,
                           double * termLocal, double * termNonloc);
void integrate_baryCenterRT(   const   ElementType & aT,
                             const ElementType & bT,
                             const QuadratureType & quadRule,
                             const MeshType & mesh,
                             double * termLocal, double * termNonloc);
// Helpers -------------------------------------------------------------------------------------------------------------
int method_baryCenter(const double * x_center, const ElementType & T, const MeshType & mesh, double * reTriangle_list, int is_placePointOnCap);
int method_retriangulate(const double * x_center, const ElementType & T, const MeshType & mesh, double * re_Triangle_list, int is_placePointOnCap);
int placePointOnCap(const double * y_predecessor, const double * y_current,
                   const double * x_center, double sqdelta, const double * TE,
                   const double * nu_a, const double * nu_b, const double * nu_c,
                   double orientation, int Rdx, double * R);
bool inTriangle(const double * y_new, const double * p, const double * q, const double * r,
                const double *  nu_a, const double * nu_b, const double * nu_c);

// ___ INTEGRATION IMPLEMENTATION ______________________________________________________________________________________

// Integration Methods #################################################################################################

void integrate_retriangulate(     const ElementType & aT,
                    const ElementType & bT,
                    const QuadratureType & quadRule,
                    const MeshType & mesh,
                    double * termLocal, double * termNonloc){

    const int dim = mesh.dim;
    int k=0, a=0, b=0;
    double x[dim];
    double innerLocal=0;
    double innerNonloc[mesh.dVertex];

    int i=0, rTdx=0, Rdx=0;
    double ker=0, rTdet=0;
    double physical_quad[dim];
    double reference_quad[dim];
    double psi_value[mesh.dVertex];
    double reTriangle_list[36*mesh.dVertex*dim];
    doubleVec_tozero(reTriangle_list, 36*mesh.dVertex*dim);
    bool is_placePointOnCap;

    //[DEBUG]
    //printf("\nouterInt_full----------------------------------------\n");
    for (k=0; k<quadRule.nPx; k++){
        //printf("k %i, quadRule.nPx %i\n", k, quadRule.nPx);
        toPhys(aT.E, &(quadRule.Px[dim*k]), mesh, x);
        //printf("\nInner Integral, Iterate %i\n", k);
        //printf("\Physical x [%17.16e, %17.16e]\n",  x[0], x[1]);
        //innerInt_retriangulate(x, aT, bT, quadRule, sqdelta, &innerLocal, innerNonloc);

        is_placePointOnCap = true;
        Rdx = method_retriangulate(x, bT, mesh, reTriangle_list, is_placePointOnCap); // innerInt_retriangulate
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

        innerLocal = 0.0;
        doubleVec_tozero(innerNonloc, mesh.dVertex);
        if (Rdx == 0){
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
                    innerLocal += model_kernel(x, aT.label, physical_quad, bT.label, mesh.sqdelta) * quadRule.dy[i] *
                                  rTdet; // Local Term
                    // Pull resulting physical point ry to the (underlying!) reference Triangle aT.
                    toRef(bT.E, physical_quad, reference_quad);
                    // Evaluate ker on physical quad (note this is ker')
                    ker = model_kernel(physical_quad, bT.label, x, aT.label, mesh.sqdelta);
                    // Evaluate basis function on resulting reference quadrature point
                    model_basisFunction(reference_quad, mesh, psi_value);
                    for (b = 0; b < mesh.dVertex; b++) {
                        innerNonloc[b] += psi_value[b] * ker * quadRule.dy[i] * rTdet; // Nonlocal Term
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

        //printf("Local %17.16e\n", innerLocal);
        //printf("Nonloc [%17.16e, %17.16e, %17.16e, %17.16e] \n", innerNonloc[0], innerNonloc[1], innerNonloc[2], innerNonloc[3]);
        for (a=0; a<mesh.dVertex; a++){
            for (b=0; b<mesh.dVertex; b++){
                termLocal[mesh.dVertex*a+b] += 2 * aT.absDet * quadRule.psix(a,k) * quadRule.psix(b,k) * quadRule.dx[k] * innerLocal; //innerLocal
                //printf("quadRule.psix(%i,%i) %17.16e\nquadRule.psix(%i,%i) %17.16e \n", a,k, quadRule.psix(a,k), b,k,quadRule.psix(b,k));
                termNonloc[mesh.dVertex*a+b] += 2 * aT.absDet * quadRule.psix(a,k) * quadRule.dx[k] * innerNonloc[b]; //innerNonloc
            }
        }
    }
}


void integrate_baryCenter(     const ElementType & aT,
                                const ElementType & bT,
                                const QuadratureType & quadRule,
                                const MeshType & mesh,
                                double * termLocal, double * termNonloc){

    const int dim = mesh.dim;
    int k=0, a=0, b=0;
    double x[dim];
    double innerLocal=0;
    double innerNonloc[mesh.dVertex];

    int i=0;
    double ker=0, rTdet=0;
    double physical_quad[dim];
    double reTriangle_list[36*mesh.dVertex*dim];
    doubleVec_tozero(reTriangle_list, 36*mesh.dVertex*dim);

    //[DEBUG]
    //printf("\nouterInt_full----------------------------------------\n");
    for (k=0; k<quadRule.nPx; k++){
        toPhys(aT.E, &(quadRule.Px[dim*k]), mesh, x);

        innerLocal = 0.0;
        doubleVec_tozero(innerNonloc, mesh.dVertex);
        if(method_baryCenter(x, bT, mesh, reTriangle_list, false)){
            for (i = 0; i < quadRule.nPy; i++) {
                // Push quadrature point P[i] to physical triangle reTriangle_list[rTdx] (of the retriangulation!)
                toPhys(bT.E, &(quadRule.Py[dim * i]), mesh, physical_quad);
                // Determinant of Triangle of retriangulation
                rTdet = absDet(bT.E);
                // inner Local integral with ker
                innerLocal += model_kernel(x, aT.label, physical_quad, bT.label, mesh.sqdelta) * quadRule.dy[i] * rTdet; // Local Term
                // Evaluate ker on physical quad (note this is ker')
                ker = model_kernel(physical_quad, bT.label, x, aT.label, mesh.sqdelta);
                // Evaluate basis function on resulting reference quadrature point
                for (b = 0; b < mesh.dVertex; b++) {
                    innerNonloc[b] += quadRule.psiy(b,i) * ker * quadRule.dy[i] * rTdet; // Nonlocal Term
                }
            }
        }
        //printf("Local %17.16e\n", innerLocal);
        //printf("Nonloc [%17.16e, %17.16e, %17.16e, %17.16e] \n", innerNonloc[0], innerNonloc[1], innerNonloc[2], innerNonloc[3]);
        for (a=0; a<mesh.dVertex; a++){
            for (b=0; b<mesh.dVertex; b++){
                termLocal[mesh.dVertex*a+b] += 2 * aT.absDet * quadRule.psix(a,k) * quadRule.psix(b,k) * quadRule.dx[k] * innerLocal; //innerLocal
                termNonloc[mesh.dVertex*a+b] += 2 * aT.absDet * quadRule.psix(a,k) * quadRule.dx[k] * innerNonloc[b]; //innerNonloc
            }
        }
    }
}

void integrate_baryCenterRT(     const ElementType & aT,
                               const ElementType & bT,
                               const QuadratureType & quadRule,
                               const MeshType & mesh,
                               double * termLocal, double * termNonloc){

    const int dim = mesh.dim;
    int k=0, a=0, b=0;
    double x[dim];
    double bTbaryC[dim];
    double innerLocal=0;
    double innerNonloc[mesh.dVertex];

    int i=0, Rdx, rTdx;
    double ker=0, rTdet=0, bTdet=0;
    double physical_quad[dim];
    double reTriangle_list[36*mesh.dVertex*dim];
    doubleVec_tozero(reTriangle_list, 36*mesh.dVertex*dim);

    //[DEBUG]
    //printf("\nouterInt_full----------------------------------------\n");
    baryCenter(bT.E, &bTbaryC[0]);
    Rdx = method_retriangulate(bTbaryC, aT, mesh, reTriangle_list, true);

    if (!Rdx){
        return;
    } else {
        // Determinant of Triangle of retriangulation
        bTdet = absDet(bT.E);

        for (rTdx = 0; rTdx < Rdx; rTdx++) {
            rTdet = absDet(&reTriangle_list[dim * mesh.dVertex * rTdx]);

            for (k = 0; k < quadRule.nPx; k++) {
                toPhys(&reTriangle_list[dim * mesh.dVertex * rTdx], &(quadRule.Px[dim * k]), mesh, x);

                // Compute Integral over Triangle bT
                innerLocal = 0.0;
                doubleVec_tozero(innerNonloc, mesh.dVertex);

                for (i = 0; i < quadRule.nPy; i++) {
                    // Push quadrature point P[i] to physical triangle reTriangle_list[rTdx] (of the retriangulation!)
                    toPhys(bT.E, &(quadRule.Py[dim * i]), mesh, physical_quad);
                    // inner Local integral with ker
                    innerLocal += model_kernel(x, aT.label, physical_quad, bT.label, mesh.sqdelta) * quadRule.dy[i] *
                                  bTdet; // Local Term
                    // Evaluate ker on physical quad (note this is ker')
                    ker = model_kernel(physical_quad, bT.label, x, aT.label, mesh.sqdelta);
                    // Evaluate basis function on resulting reference quadrature point
                    for (b = 0; b < mesh.dVertex; b++) {
                        innerNonloc[b] += quadRule.psiy(b, i) * ker * quadRule.dy[i] * bTdet; // Nonlocal Term
                    }
                }

                for (a = 0; a < mesh.dVertex; a++) {
                    for (b = 0; b < mesh.dVertex; b++) {
                        termLocal[mesh.dVertex * a + b] += 2 * rTdet * quadRule.psix(a, k) *
                                quadRule.psix(b, k) * quadRule.dx[k] * innerLocal; //innerLocal
                        termNonloc[mesh.dVertex * a + b] +=
                                2 * rTdet * quadRule.psix(a, k) * quadRule.dx[k] * innerNonloc[b]; //innerNonloc
                    }
                }
            }
        }
    }
}
// Helpers -------------------------------------------------------------------------------------------------------------

// Bary Center Method --------------------------------------------------------------------------------------------------
int method_baryCenter(const double * x_center, const ElementType & T, const MeshType & mesh, double * reTriangle_list, int is_placePointOnCap){
    int i,k;
    double distance;
    arma::vec baryC(mesh.dim);

    baryCenter(T.E, &baryC[0]);
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
    };
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

int method_retriangulate(const double * x_center, const ElementType & T,
                         const MeshType & mesh, double * re_Triangle_list,
                         int is_placePointOnCap){
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
    arma::vec vec_x_center(x_center, 2);
    double orientation;

    bool is_onEdge=false, is_firstPointLiesOnVertex=true;
    // The upper bound for the number of required points is 9
    // Hence 9*3 is an upper bound to encode all resulting triangles
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

        if (vec_sqL2dist(&p[0], x_center, 2) <= mesh.sqdelta){
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
                if (is_onEdge && is_placePointOnCap){
                    Rdx += placePointOnCap(&R[2*(Rdx-1)], &y1[0], x_center, mesh.sqdelta, T.E, nu_a, nu_b, nu_c, orientation, Rdx, R);
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
                if (is_onEdge && is_placePointOnCap){
                    Rdx += placePointOnCap(&R[2*(Rdx-1)], &y2[0], x_center, mesh.sqdelta, T.E, nu_a, nu_b, nu_c, orientation, Rdx, R);
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
    if (is_onEdge && (!is_firstPointLiesOnVertex && Rdx > 1) && is_placePointOnCap){
        Rdx += placePointOnCap(&R[2*(Rdx-1)], &R[0], x_center, mesh.sqdelta, T.E, nu_a, nu_b, nu_c, orientation, Rdx, R);
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
                re_Triangle_list[2 * (3 * k + 0) + i] = R[i];
                re_Triangle_list[2 * (3 * k + 1) + i] = R[2 * (k + 1) + i];
                re_Triangle_list[2 * (3 * k + 2) + i] = R[2 * (k + 2) + i];
            }
        }
        // Excessing the bound out_Rdx will not lead to an error but simply to corrupted data!

        return Rdx - 2; // So that, it acutally contains the number of triangles in the retriangulation
    }
}
#endif //NONLOCAL_ASSEMBLY_INTEGRATION_CPP