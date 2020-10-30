from libcpp.string cimport string

cdef extern from "Configuration.h":
    cdef cppclass Configuration:
        Configuration(string &path_spAd_, string &path_fd_,
                      string &model_kernel_, string &model_f_,
                      string &integration_method_,
                      int is_placePointOnCap_,
                      double kernelHorizon_) except +

