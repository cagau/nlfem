import numpy as np
from examples.RatesScipy3D.mesh import RegMesh
import examples.RatesScipy3D.conf3D as conf

A = np.random.rand(6,6)
# Write Binary
f = open("results/A.bin", "wb")
A.tofile(f, sep="")
f.close()
# Write Text
f = open("results/A.txt", "w+")
A.tofile(f, sep=" ")
f.close()

B = np.fromfile("results/A.bin")
print(B)

if __name__ == "__main__":
    mesh = RegMesh(.1, 3)
    mesh.save(conf.outputdir)
    conf.save(conf.outputdir)
