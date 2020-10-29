import nlfem
import configuration as conf

g = nlfem.stiffnessMatrix(conf.mesh, conf.kernel, conf.configuration)