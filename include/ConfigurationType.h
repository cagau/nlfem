//
// Created by klar on 29.10.20.
//

#ifndef NONLOCAL_ASSEMBLY_CONFIGURATIONTYPE_H
#define NONLOCAL_ASSEMBLY_CONFIGURATIONTYPE_H

#include <utility>

#include "string"
using namespace std;

class ConfigurationType {
public:
    const string path_spAd;
    const string path_fd;
    const string model_kernel;
    const string model_f;
    const string integration_method;
    const bool is_placePointOnCap;
    const double kernelHorizon;

    // Deprecated variables
    bool is_singularKernel;

    ConfigurationType(string path_spAd_, string path_fd_,
                      string model_kernel_, string model_f_,
                      string integration_method_,
                      bool is_placePointOnCap_,
                      double kernelHorizon_):
            path_spAd(std::move(path_spAd_)),
            path_fd(std::move(path_fd_)),
            model_kernel(std::move(model_kernel_)),
            model_f(std::move(model_f_)),
            integration_method(std::move(integration_method_)),
            is_placePointOnCap(is_placePointOnCap_),
            kernelHorizon(kernelHorizon_){
            };
};


#endif //NONLOCAL_ASSEMBLY_CONFIGURATIONTYPE_H
