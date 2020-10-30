//
// Created by klar on 30.10.20.
//

#ifndef NONLOCAL_ASSEMBLY_INTEGRATOR_H
#define NONLOCAL_ASSEMBLY_INTEGRATOR_H


#include <utility>

#include "MeshTypes.h"
#include "QuadratureType.h"
#include "MeshType.h"
#include "Kernel.h"
#include "ConfigurationType.h"

class Integrator {
public:
    const MeshType mesh;
    const QuadratureType quadRule;
    const Kernel kernel;
    const ConfigurationType conf;

    Integrator(MeshType &mesh_, QuadratureType &quadrature_, Kernel &kernel_, ConfigurationType &conf_) :
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
    BaryCenter(MeshType & mesh_, QuadratureType & quadrature_, Kernel & kernel_, ConfigurationType & conf_):
    Integrator(mesh_, quadrature_, kernel_, conf_){
        cout << "Constructor of BaryCenter Integrator knows kernel delta: " << kernel.delta << endl;
    };
    void approxBall(const ElementType &aT, const ElementType &bT,
                            double *termLocal, double *termNonloc) override {};
};

class BaryCenterRT : public Integrator {
public:
    BaryCenterRT(MeshType & mesh_, QuadratureType & quadrature_, Kernel & kernel_, ConfigurationType & conf_):
    Integrator(mesh_, quadrature_, kernel_, conf_){
        cout << "Constructor of BaryCenterRT Integrator knows kernel delta: " << kernel.delta << endl;
    };
    void approxBall(const ElementType &aT, const ElementType &bT,
                    double *termLocal, double *termNonloc) final {};
};


class Retriangulate : public Integrator {
public:
    Retriangulate(MeshType & mesh_, QuadratureType & quadrature_, Kernel & kernel_, ConfigurationType & conf_):
    Integrator(mesh_, quadrature_, kernel_, conf_){
        cout << "Constructor of Retriangulate Integrator knows kernel delta: " << kernel.delta << endl;
    };
    void approxBall(const ElementType &aT, const ElementType &bT,
                    double *termLocal, double *termNonloc) final {};
};

class AverageBall : public Integrator {
public:
    AverageBall(MeshType & mesh_, QuadratureType & quadrature_, Kernel & kernel_, ConfigurationType & conf_):
    Integrator(mesh_, quadrature_, kernel_, conf_){
        cout << "Constructor of AverageBall Integrator knows kernel delta: " << kernel.delta << endl;
    };
    void approxBall(const ElementType &aT, const ElementType &bT,
                    double *termLocal, double *termNonloc) final {};
};
#endif //NONLOCAL_ASSEMBLY_INTEGRATOR_H
