//
// Created by klar on 29.10.20.
//

#ifndef NONLOCAL_ASSEMBLY_CONFIGURATION_H
#define NONLOCAL_ASSEMBLY_CONFIGURATION_H

#include "string"
using namespace std;

class Configuration {
public:
    const string path_spAd;
    const string path_fd;
    const string model_kernel;
    const string model_f;
    const string integration_method;
    const bool is_placePointOnCap;
    const double kernelHorizon;

    Configuration(string &path_spAd_, string &path_fd_,
                  string &model_kernel_, string &model_f_,
                  string &integration_method_,
                  bool is_placePointOnCap_,
                  double kernelHorizon_):
            path_spAd(path_spAd_),
            path_fd(path_fd_),
            model_kernel(model_kernel_),
            model_f(model_f_),
            integration_method(integration_method_),
            is_placePointOnCap(is_placePointOnCap_),
            kernelHorizon(kernelHorizon_){
            };
};


#endif //NONLOCAL_ASSEMBLY_CONFIGURATION_H
