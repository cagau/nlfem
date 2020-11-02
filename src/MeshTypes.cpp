//
// Created by klar on 14.09.20.
//
#include "MeshTypes.h"

ElementType::ElementType(const MeshType & mesh_) : dim(mesh_.dim), mesh(mesh_){
    matE = arma::vec(this->dim*(dim+1), arma::fill::zeros);
    E = matE.memptr();
}

void ElementType::setData(const int Tdx_) {
        /*
        Copy coordinates of Triange b to bTE.

         Tempalte of Triangle Point data.
         2D Case, a, b, c are the vertices of a triangle
         T.E -> | a1 | a2 | b1 | b2 | c1 | c2 |
         Hence, if one wants to put T.E into col major order matrix it would be of shape\
                                     | a1 | b1 | c1 |
         M(mesh.dim, mesh.dVerts) =  | a2 | b2 | c2 |
        */

        int j, k, Vdx;
        //T.matE = arma::vec(dim*(dim+1));
        for (k=0; k<mesh.dVertex; k++) {
            //Vdx = mesh.ptrTriangles[(mesh.dVertex+1)*Tdx + k+1];
            Vdx = mesh.Elements(k, Tdx_);
            for (j=0; j<mesh.dim; j++){
                matE[mesh.dim * k + j] = mesh.Verts(j, Vdx);
                //printf ("%3.2f ", T.matE[mesh.dim * k + j]);
                //T.matE[mesh.dim * k + j] = mesh.ptrVerts[ mesh.dim*Vdx + j];
            }
            //printf("\n");
        }
        // Initialize Struct
        E = matE.memptr();
        absDetValue = absDet(E, mesh.dim);
        this->signDetValue = static_cast<int>(signDet(E, mesh));
        label = mesh.LabelElements(Tdx_);
        Tdx = Tdx_;
}

void ElementType::setData(const long * Vdx_new) {
    for (int k = 0; k < mesh.dVertex; k++) {
        //Vdx = mesh.Elements(k, Tdx);
        for (int j = 0; j < mesh.dim; j++) {
            matE[mesh.dim * k + j] = mesh.Verts(j, Vdx_new[k]);
            //printf ("aT %3.2f ", T.matE[mesh.dim * k + j]);
        }
    }
}