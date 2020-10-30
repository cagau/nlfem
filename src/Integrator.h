//
// Created by klar on 30.10.20.
//

#ifndef NONLOCAL_ASSEMBLY_INTEGRATOR_H
#define NONLOCAL_ASSEMBLY_INTEGRATOR_H


#include <utility>

#include "MeshTypes.h"
#include "Quadrature.h"
#include "Mesh.h"
#include "Kernel.h"
#include "Configuration.h"

class Integrator {
public:
    const Mesh mesh;
    const Quadrature quadRule;
    const Kernel kernel;
    const Configuration conf;

    Integrator(Mesh &mesh_, Quadrature &quadrature_, Kernel &kernel_, Configuration &conf_) :
            mesh(mesh_),
            quadRule(quadrature_),
            kernel(kernel_),
            conf(conf_) {
        cout << "Constructor of Base Integrator knows kernel delta: " << kernel.delta << endl;
    };
    virtual void approxBall(const ElementType &aT, const ElementType &bT,
                            double *termLocal, double *termNonloc) {
            cout << "Approx Ball of Integrator evoked." <<
            endl;
        };
    virtual void tensorGauss(const ElementType &aT, const ElementType &bT,
                            double *termLocal, double *termNonloc) {
        cout << "tensorGauss Ball of Integrator evoked." <<
             endl;
    };
};

class BaryCenter : public Integrator {
public:
    BaryCenter(Mesh & mesh_, Quadrature & quadrature_, Kernel & kernel_, Configuration & conf_):
    Integrator(mesh_, quadrature_, kernel_, conf_){
        cout << "Constructor of BaryCenter Integrator knows kernel delta: " << kernel.delta << endl;
    };
    void approxBall(const ElementType &aT, const ElementType &bT,
                            double *termLocal, double *termNonloc) override {};
};

class BaryCenterRT : public Integrator {
public:
    BaryCenterRT(Mesh & mesh_, Quadrature & quadrature_, Kernel & kernel_, Configuration & conf_):
    Integrator(mesh_, quadrature_, kernel_, conf_){
        cout << "Constructor of BaryCenterRT Integrator knows kernel delta: " << kernel.delta << endl;
    };
    void approxBall(const ElementType &aT, const ElementType &bT,
                    double *termLocal, double *termNonloc) final {};
};


class Retriangulate : public Integrator {
public:
    Retriangulate(Mesh & mesh_, Quadrature & quadrature_, Kernel & kernel_, Configuration & conf_):
    Integrator(mesh_, quadrature_, kernel_, conf_){
        cout << "Constructor of Retriangulate Integrator knows kernel delta: " << kernel.delta << endl;
    };
    void approxBall(const ElementType &aT, const ElementType &bT,
                    double *termLocal, double *termNonloc) final {};
};

class AverageBall : public Integrator {
public:
    AverageBall(Mesh & mesh_, Quadrature & quadrature_, Kernel & kernel_, Configuration & conf_):
    Integrator(mesh_, quadrature_, kernel_, conf_){
        cout << "Constructor of AverageBall Integrator knows kernel delta: " << kernel.delta << endl;
    };
    void approxBall(const ElementType &aT, const ElementType &bT,
                    double *termLocal, double *termNonloc) final {};
};
#endif //NONLOCAL_ASSEMBLY_INTEGRATOR_H
