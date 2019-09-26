from assemble import py_retriangulate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot(x_center, delta, TE, RD, pp):
    ax = plt.gca()
    plt.gca().set_aspect('equal')
    plt.fill(TE[:, 0], TE[:, 1], edgecolor='r', fill=False, alpha=.5)
    plt.scatter(x_center[0], x_center[1], s=1, c="red", alpha=.5)
    circ = plt.Circle(x_center, delta, fill=False, color="b", lw=.1, alpha=.7)
    ax.add_artist(circ)

    if RD.size > 0:
        plt.scatter(RD[:, 0], RD[:, 1], s=5, color="b", alpha=1)
    plt.savefig(pp, format='pdf')
    plt.close()



deltaList = [.1, .1, .85, .3, .3, .3, .9, .6, .4, .3]
x_centerList = [np.array([-.2,.6]),
           np.array([.2,.6]),
           np.array([.3,.6]),
           np.array([.6,.4]),
           np.array([.1,.1]),
           np.array([.6,.8]),
           np.array([.8,.3]),
           np.array([.3,.7]),
           np.array([.3,.7]),
           np.array([.25,.6])]

TE = np.array([[0,0],[0,1],[1,1]], dtype=float)
c_TE = TE.flatten("C")
print("Triangle: ", c_TE)

pp = PdfPages("Retriangulate" + ".pdf")
for i in range(len(deltaList)):
    print("Page ", i + 1)
    delta  = deltaList[i]
    x_center = x_centerList[i]

    Rdx, RD = py_retriangulate(x_center, c_TE, delta)
    print("Rdx ", Rdx,"\n", RD)
    RD = RD.reshape((-1,2))
    plot(x_center, delta, TE, np.array(RD[:3*Rdx]), pp)
pp.close()