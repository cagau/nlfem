#-*- coding:utf-8 -*-
import numpy as np
import os

delta = .1
ansatz = "CG"
boundaryConditionType = "Dirichlet" # "Neumann" #
model_f = "linear" # "constant" #
model_kernel = "constant"#"parabola" "linearPrototypeMicroelastic"# "constant" # "labeled" #
integration_method = "retriangulate" # "tensorgauss" "retriangulate" # "baryCenter" # subSetBall # superSetBall averageBall
is_PlacePointOnCap = True
quadrule_outer = "7"
quadrule_inner = "1"
tensorGaussDegree = 6

n_start = 12
n_layers = 4
N  = [n_start*2**(l) for l in list(range(n_layers))]
N_fine = N[-1]*4
def u_exact(x):
    return x[0] ** 2 * x[1] + x[1] ** 2

outputdir  = "results/"
os.makedirs(outputdir, exist_ok=True)
fnames = {"triPlot.pdf": outputdir+"auto_plot.pdf",
          "rates.md": outputdir+"auto_rates.md",
          "rates.pdf": outputdir+"auto_rates.pdf",
          "timePlot.pdf": outputdir+"timePlot.pdf",
          "report.pdf": outputdir+"auto_report.pdf"}
data = {"h": [], "L2 Error": [], "Rates": [], "Assembly Time": [], "nV_Omega": []}

# Quadrature rules -----------------------------------------------------------------------------------------------------
quadrules = {
    "7":    [
        np.array([[0.33333333333333,    0.33333333333333],
                  [0.47014206410511,    0.47014206410511],
                  [0.47014206410511,    0.05971587178977],
                  [0.05971587178977,    0.47014206410511],
                  [0.10128650732346,    0.10128650732346],
                  [0.10128650732346,    0.79742698535309],
                  [0.79742698535309,    0.10128650732346]]),

        0.5 * np.array([0.22500000000000,
                        0.13239415278851,
                        0.13239415278851,
                        0.13239415278851,
                        0.12593918054483,
                        0.12593918054483,
                        0.12593918054483])
    ],
    "16":    [
        np.array([[0.33333333, 0.33333333],
                  [0.45929259, 0.45929259],
                  [0.45929259, 0.08141482],
                  [0.08141482, 0.45929259],
                  [0.17056931, 0.17056931],
                  [0.17056931, 0.65886138],
                  [0.65886138, 0.17056931],
                  [0.05054723, 0.05054723],
                  [0.05054723, 0.89890554],
                  [0.89890554, 0.05054723],
                  [0.26311283, 0.72849239],
                  [0.72849239, 0.00839478],
                  [0.00839478, 0.26311283],
                  [0.72849239, 0.26311283],
                  [0.26311283, 0.00839478],
                  [0.00839478, 0.72849239]]),

        0.5 * np.array([0.14431560767779
                           , 0.09509163426728
                           , 0.09509163426728
                           , 0.09509163426728
                           , 0.10321737053472
                           , 0.10321737053472
                           , 0.10321737053472
                           , 0.03245849762320
                           , 0.03245849762320
                           , 0.03245849762320
                           , 0.02723031417443
                           , 0.02723031417443
                           , 0.02723031417443
                           , 0.02723031417443
                           , 0.02723031417443
                           , 0.02723031417443])
    ],
    "1":    [
        np.array([[0.33333333, 0.33333333]]),
        0.5 * np.array([1.0])
    ],
    "laursenGellert15a":
        [      np.array([[0.33333333, 0.33333333],
                         [0.42508621, 0.42508621],
                         [0.02330887, 0.02330887],
                         [0.42508621, 0.14982758],
                         [0.02330887, 0.95338226],
                         [0.14982758, 0.42508621],
                         [0.95338226, 0.02330887],
                         [0.14792563, 0.22376697],
                         [0.02994603, 0.35874014],
                         [0.03563256, 0.14329537],
                         [0.6283074,  0.14792563],
                         [0.61131383, 0.02994603],
                         [0.82107207, 0.03563256],
                         [0.22376697, 0.6283074 ],
                         [0.35874014, 0.61131383],
                         [0.14329537, 0.82107207],
                         [0.22376697, 0.14792563],
                         [0.35874014, 0.02994603],
                         [0.14329537, 0.03563256],
                         [0.6283074,  0.22376697],
                         [0.61131383, 0.35874014],
                         [0.82107207, 0.14329537],
                         [0.14792563, 0.6283074 ],
                         [0.02994603, 0.61131383],
                         [0.03563256, 0.82107207]]) ,
               np.array([0.03994725237062 ,
                         0.0355619011161885 ,
                         0.004111909345232 ,
                         0.0355619011161885 ,
                         0.004111909345232 ,
                         0.0355619011161885 ,
                         0.004111909345232 ,
                         0.022715296148085 ,
                         0.0186799281171525 ,
                         0.015443328442282 ,
                         0.022715296148085 ,
                         0.0186799281171525 ,
                         0.015443328442282 ,
                         0.022715296148085 ,
                         0.0186799281171525 ,
                         0.015443328442282 ,
                         0.022715296148085 ,
                         0.0186799281171525 ,
                         0.015443328442282 ,
                         0.022715296148085 ,
                         0.0186799281171525 ,
                         0.015443328442282 ,
                         0.022715296148085 ,
                         0.0186799281171525 ,
                         0.015443328442282 ])
               ],
    "vioreanu_rokhlin":
        [np.array([[0.00319145, 0.00319145],
                   [0.01646557, 0.01646557],
                   [0.04984951, 0.04984951],
                   [0.48821932, 0.48821932],
                   [0.09411486, 0.09411486],
                   [0.15683697, 0.15683697],
                   [0.44791797, 0.44791797],
                   [0.22735936, 0.22735936],
                   [0.38431662, 0.38431662],
                   [0.30654709, 0.30654709],
                   [0.00319145, 0.99361709],
                   [0.01646557, 0.96706886],
                   [0.04984951, 0.90030098],
                   [0.48821932, 0.02356135],
                   [0.09411486, 0.81177028],
                   [0.15683697, 0.68632607],
                   [0.44791797, 0.10416405],
                   [0.22735936, 0.54528128],
                   [0.38431662, 0.23136675],
                   [0.30654709, 0.38690581],
                   [0.99361709, 0.00319145],
                   [0.96706886, 0.01646557],
                   [0.90030098, 0.04984951],
                   [0.02356135, 0.48821932],
                   [0.81177028, 0.09411486],
                   [0.68632607, 0.15683697],
                   [0.10416405, 0.44791797],
                   [0.54528128, 0.22735936],
                   [0.23136675, 0.38431662],
                   [0.38690581, 0.30654709],
                   [0.9783946 , 0.01787156],
                   [0.9573766 , 0.03947158],
                   [0.92976302, 0.06627351],
                   [0.88561454, 0.1102078 ],
                   [0.82760493, 0.16807057],
                   [0.94070253, 0.04040669],
                   [0.76145695, 0.23413045],
                   [0.69109825, 0.30446773],
                   [0.61696057, 0.37857652],
                   [0.53821615, 0.45728517],
                   [0.89940139, 0.07945793],
                   [0.84545816, 0.1323574 ],
                   [0.78209575, 0.19501815],
                   [0.7137263 , 0.26311914],
                   [0.85364825, 0.09441798],
                   [0.64266306, 0.33408249],
                   [0.56742143, 0.40911971],
                   [0.79588364, 0.14964191],
                   [0.73168885, 0.21246609],
                   [0.66332112, 0.28044306],
                   [0.75210551, 0.14843151],
                   [0.58984241, 0.35350754],
                   [0.51147604, 0.43142904],
                   [0.68513574, 0.21304613],
                   [0.61118872, 0.28614579],
                   [0.53104705, 0.36531163],
                   [0.61675541, 0.2236119 ],
                   [0.54093912, 0.29777012],
                   [0.45983724, 0.37745983],
                   [0.46664954, 0.30321004],
                   [0.00373384, 0.9783946 ],
                   [0.00315182, 0.9573766 ],
                   [0.00396347, 0.92976302],
                   [0.00417766, 0.88561454],
                   [0.0043245 , 0.82760493],
                   [0.01889079, 0.94070253],
                   [0.0044126 , 0.76145695],
                   [0.00443403, 0.69109825],
                   [0.00446292, 0.61696057],
                   [0.00449868, 0.53821615],
                   [0.02114068, 0.89940139],
                   [0.02218444, 0.84545816],
                   [0.0228861 , 0.78209575],
                   [0.02315456, 0.7137263 ],
                   [0.05193377, 0.85364825],
                   [0.02325445, 0.64266306],
                   [0.02345887, 0.56742143],
                   [0.05447444, 0.79588364],
                   [0.05584507, 0.73168885],
                   [0.05623582, 0.66332112],
                   [0.09946297, 0.75210551],
                   [0.05665005, 0.58984241],
                   [0.05709492, 0.51147604],
                   [0.10181813, 0.68513574],
                   [0.10266548, 0.61118872],
                   [0.10364132, 0.53104705],
                   [0.1596327 , 0.61675541],
                   [0.16129077, 0.54093912],
                   [0.16270293, 0.45983724],
                   [0.23014042, 0.46664954],
                   [0.01787156, 0.00373384],
                   [0.03947158, 0.00315182],
                   [0.06627351, 0.00396347],
                   [0.1102078 , 0.00417766],
                   [0.16807057, 0.0043245 ],
                   [0.04040669, 0.01889079],
                   [0.23413045, 0.0044126 ],
                   [0.30446773, 0.00443403],
                   [0.37857652, 0.00446292],
                   [0.45728517, 0.00449868],
                   [0.07945793, 0.02114068],
                   [0.1323574 , 0.02218444],
                   [0.19501815, 0.0228861 ],
                   [0.26311914, 0.02315456],
                   [0.09441798, 0.05193377],
                   [0.33408249, 0.02325445],
                   [0.40911971, 0.02345887],
                   [0.14964191, 0.05447444],
                   [0.21246609, 0.05584507],
                   [0.28044306, 0.05623582],
                   [0.14843151, 0.09946297],
                   [0.35350754, 0.05665005],
                   [0.43142904, 0.05709492],
                   [0.21304613, 0.10181813],
                   [0.28614579, 0.10266548],
                   [0.36531163, 0.10364132],
                   [0.2236119 , 0.1596327 ],
                   [0.29777012, 0.16129077],
                   [0.37745983, 0.16270293],
                   [0.30321004, 0.23014042],
                   [0.01787156, 0.9783946 ],
                   [0.03947158, 0.9573766 ],
                   [0.06627351, 0.92976302],
                   [0.1102078 , 0.88561454],
                   [0.16807057, 0.82760493],
                   [0.04040669, 0.94070253],
                   [0.23413045, 0.76145695],
                   [0.30446773, 0.69109825],
                   [0.37857652, 0.61696057],
                   [0.45728517, 0.53821615],
                   [0.07945793, 0.89940139],
                   [0.1323574 , 0.84545816],
                   [0.19501815, 0.78209575],
                   [0.26311914, 0.7137263 ],
                   [0.09441798, 0.85364825],
                   [0.33408249, 0.64266306],
                   [0.40911971, 0.56742143],
                   [0.14964191, 0.79588364],
                   [0.21246609, 0.73168885],
                   [0.28044306, 0.66332112],
                   [0.14843151, 0.75210551],
                   [0.35350754, 0.58984241],
                   [0.43142904, 0.51147604],
                   [0.21304613, 0.68513574],
                   [0.28614579, 0.61118872],
                   [0.36531163, 0.53104705],
                   [0.2236119 , 0.61675541],
                   [0.29777012, 0.54093912],
                   [0.37745983, 0.45983724],
                   [0.30321004, 0.46664954],
                   [0.00373384, 0.01787156],
                   [0.00315182, 0.03947158],
                   [0.00396347, 0.06627351],
                   [0.00417766, 0.1102078 ],
                   [0.0043245 , 0.16807057],
                   [0.01889079, 0.04040669],
                   [0.0044126 , 0.23413045],
                   [0.00443403, 0.30446773],
                   [0.00446292, 0.37857652],
                   [0.00449868, 0.45728517],
                   [0.02114068, 0.07945793],
                   [0.02218444, 0.1323574 ],
                   [0.0228861 , 0.19501815],
                   [0.02315456, 0.26311914],
                   [0.05193377, 0.09441798],
                   [0.02325445, 0.33408249],
                   [0.02345887, 0.40911971],
                   [0.05447444, 0.14964191],
                   [0.05584507, 0.21246609],
                   [0.05623582, 0.28044306],
                   [0.09946297, 0.14843151],
                   [0.05665005, 0.35350754],
                   [0.05709492, 0.43142904],
                   [0.10181813, 0.21304613],
                   [0.10266548, 0.28614579],
                   [0.10364132, 0.36531163],
                   [0.1596327 , 0.2236119 ],
                   [0.16129077, 0.29777012],
                   [0.16270293, 0.37745983],
                   [0.23014042, 0.30321004],
                   [0.9783946 , 0.00373384],
                   [0.9573766 , 0.00315182],
                   [0.92976302, 0.00396347],
                   [0.88561454, 0.00417766],
                   [0.82760493, 0.0043245 ],
                   [0.94070253, 0.01889079],
                   [0.76145695, 0.0044126 ],
                   [0.69109825, 0.00443403],
                   [0.61696057, 0.00446292],
                   [0.53821615, 0.00449868],
                   [0.89940139, 0.02114068],
                   [0.84545816, 0.02218444],
                   [0.78209575, 0.0228861 ],
                   [0.7137263 , 0.02315456],
                   [0.85364825, 0.05193377],
                   [0.64266306, 0.02325445],
                   [0.56742143, 0.02345887],
                   [0.79588364, 0.05447444],
                   [0.73168885, 0.05584507],
                   [0.66332112, 0.05623582],
                   [0.75210551, 0.09946297],
                   [0.58984241, 0.05665005],
                   [0.51147604, 0.05709492],
                   [0.68513574, 0.10181813],
                   [0.61118872, 0.10266548],
                   [0.53104705, 0.10364132],
                   [0.61675541, 0.1596327 ],
                   [0.54093912, 0.16129077],
                   [0.45983724, 0.16270293],
                   [0.46664954, 0.23014042]]),

         np.array([7.15429620e-05, 2.77498318e-04, 1.42410434e-03, 2.11768291e-03,
                   2.32182957e-03, 3.87743907e-03, 4.43599992e-03, 5.19737967e-03,
                   5.97962751e-03, 6.24665604e-03, 7.15429620e-05, 2.77498318e-04,
                   1.42410434e-03, 2.11768291e-03, 2.32182957e-03, 3.87743907e-03,
                   4.43599992e-03, 5.19737967e-03, 5.97962751e-03, 6.24665604e-03,
                   7.15429620e-05, 2.77498318e-04, 1.42410434e-03, 2.11768291e-03,
                   2.32182957e-03, 3.87743907e-03, 4.43599992e-03, 5.19737967e-03,
                   5.97962751e-03, 6.24665604e-03, 1.82652260e-04, 1.86216417e-04,
                   3.53016454e-04, 5.56181908e-04, 6.95916250e-04, 7.07183222e-04,
                   7.75808288e-04, 8.18011073e-04, 8.74557018e-04, 9.28120841e-04,
                   1.10489501e-03, 1.45756221e-03, 1.70149579e-03, 1.81118911e-03,
                   1.84131161e-03, 1.90055367e-03, 2.04489977e-03, 2.30692764e-03,
                   2.60134378e-03, 2.80893694e-03, 3.05292773e-03, 3.05334851e-03,
                   3.22728101e-03, 3.60219558e-03, 4.03034387e-03, 4.32824058e-03,
                   4.44987478e-03, 4.94219666e-03, 5.24175492e-03, 5.77351027e-03,
                   1.82652260e-04, 1.86216417e-04, 3.53016454e-04, 5.56181908e-04,
                   6.95916250e-04, 7.07183222e-04, 7.75808288e-04, 8.18011073e-04,
                   8.74557018e-04, 9.28120841e-04, 1.10489501e-03, 1.45756221e-03,
                   1.70149579e-03, 1.81118911e-03, 1.84131161e-03, 1.90055367e-03,
                   2.04489977e-03, 2.30692764e-03, 2.60134378e-03, 2.80893694e-03,
                   3.05292773e-03, 3.05334851e-03, 3.22728101e-03, 3.60219558e-03,
                   4.03034387e-03, 4.32824058e-03, 4.44987478e-03, 4.94219666e-03,
                   5.24175492e-03, 5.77351027e-03, 1.82652260e-04, 1.86216417e-04,
                   3.53016454e-04, 5.56181908e-04, 6.95916250e-04, 7.07183222e-04,
                   7.75808288e-04, 8.18011073e-04, 8.74557018e-04, 9.28120841e-04,
                   1.10489501e-03, 1.45756221e-03, 1.70149579e-03, 1.81118911e-03,
                   1.84131161e-03, 1.90055367e-03, 2.04489977e-03, 2.30692764e-03,
                   2.60134378e-03, 2.80893694e-03, 3.05292773e-03, 3.05334851e-03,
                   3.22728101e-03, 3.60219558e-03, 4.03034387e-03, 4.32824058e-03,
                   4.44987478e-03, 4.94219666e-03, 5.24175492e-03, 5.77351027e-03,
                   1.82652260e-04, 1.86216417e-04, 3.53016454e-04, 5.56181908e-04,
                   6.95916250e-04, 7.07183222e-04, 7.75808288e-04, 8.18011073e-04,
                   8.74557018e-04, 9.28120841e-04, 1.10489501e-03, 1.45756221e-03,
                   1.70149579e-03, 1.81118911e-03, 1.84131161e-03, 1.90055367e-03,
                   2.04489977e-03, 2.30692764e-03, 2.60134378e-03, 2.80893694e-03,
                   3.05292773e-03, 3.05334851e-03, 3.22728101e-03, 3.60219558e-03,
                   4.03034387e-03, 4.32824058e-03, 4.44987478e-03, 4.94219666e-03,
                   5.24175492e-03, 5.77351027e-03, 1.82652260e-04, 1.86216417e-04,
                   3.53016454e-04, 5.56181908e-04, 6.95916250e-04, 7.07183222e-04,
                   7.75808288e-04, 8.18011073e-04, 8.74557018e-04, 9.28120841e-04,
                   1.10489501e-03, 1.45756221e-03, 1.70149579e-03, 1.81118911e-03,
                   1.84131161e-03, 1.90055367e-03, 2.04489977e-03, 2.30692764e-03,
                   2.60134378e-03, 2.80893694e-03, 3.05292773e-03, 3.05334851e-03,
                   3.22728101e-03, 3.60219558e-03, 4.03034387e-03, 4.32824058e-03,
                   4.44987478e-03, 4.94219666e-03, 5.24175492e-03, 5.77351027e-03,
                   1.82652260e-04, 1.86216417e-04, 3.53016454e-04, 5.56181908e-04,
                   6.95916250e-04, 7.07183222e-04, 7.75808288e-04, 8.18011073e-04,
                   8.74557018e-04, 9.28120841e-04, 1.10489501e-03, 1.45756221e-03,
                   1.70149579e-03, 1.81118911e-03, 1.84131161e-03, 1.90055367e-03,
                   2.04489977e-03, 2.30692764e-03, 2.60134378e-03, 2.80893694e-03,
                   3.05292773e-03, 3.05334851e-03, 3.22728101e-03, 3.60219558e-03,
                   4.03034387e-03, 4.32824058e-03, 4.44987478e-03, 4.94219666e-03,
                   5.24175492e-03, 5.77351027e-03])]
}

py_Px = quadrules[quadrule_outer][0]
dx = quadrules[quadrule_outer][1]
py_Py = quadrules[quadrule_inner][0]
dy = quadrules[quadrule_inner][1]

def writeattr(file, attr_name):
    file.write(attr_name+"\n")
    file.write(str(eval(attr_name))+"\n")

def save(path):
    # Save Configuration
    confList = [
        "model_kernel",
        "model_f",
        "integration_method",
        "is_PlacePointOnCap"]

    f = open(path + "/conf", "w+")
    [writeattr(f, attr_name) for attr_name in confList]
    f.close()

    # Provide Quadrature Rules
    py_Px.tofile(path+"/quad.Px")
    py_Py.tofile(path+"/quad.Py")
    dx.tofile((path+"/quad.dx"))
    dy.tofile((path+"/quad.dy"))
