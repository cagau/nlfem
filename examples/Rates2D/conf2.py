import numpy as np
import conf



# folder = 'withcaps_a2_b2/'
folder = 'TEST/'
folder = 'results/' + folder


transform_switch = 0
gmsh = 0
geofile = "unit_square_3"

h1, h2 = 0.1, 0.1
num_grids = 4
num_grids_mat = 6
num_grids_mat_l2rates = 6
H1 = [h1 * 2 ** -k for k in range(0, max(num_grids, num_grids_mat))]
H2 = [h2 * 2 ** -k for k in range(0, max(num_grids, num_grids_mat))]

plot_solve = 0
plot_mesh = 0


def transform(x):
    a = 2
    b = 2
    y1 = (1. - np.exp(-a * x[0])) / (1. - np.exp(-a))
    y2 = np.sin(np.pi * x[1] / 2) ** b
    return np.array([y1, y2])
    # return np.array([x[0], x[1]**2])


def source(x):
    return -2. * (x[1] + 1)
def g_d(x):
    return x[0] ** 2 * x[1] + x[1] ** 2
def u_exact(x):
    return x[0] ** 2 * x[1] + x[1] ** 2


depricated_mesh = 0
num_cores = 7
# new_mesh = 1
# reg_mesh = 1  # create regular grid on [0,1]^2


Norm = 'L2'  # choose from ['L1', 'L2', 'Linf']
ball = 'exact_L2'
Ball = [ball]
approx = 0

# Kernel
def phi(r):
    return 1  # (1 - ((r/delta )**2) )


# g11 = 0.001 * 3. / (4 * delta ** 4)
# g22 = 100 * 3. / (4 * delta ** 4)
loclim = 4. / (np.pi * conf.delta ** 4)


def gam11(x, y):
    # s = norm(x - y)
    return loclim  # np.where(s >= delta, 0, loclim)# np.where(s > delta, 0, )#g11 #* (1 - ((s / delta) ** 2)) # np.where(s > eps2, 0, g11)


def gam22(x, y):
    # s = norm(x - y)
    # print(s)
    return loclim  # np.where(s >= delta, 0, loclim)#loclim  # np.where(s > delta, 0,   ) #* (1 - ((s/delta )**2) )


def gam13(x, y):
    return loclim  #gam22(x, y)


#        def gam1(x,y):
#            s = norm(x-y)
#            return 4. / (np.pi * delta **4) * phi(s) #np.where(s > delta, 0 , phi(s)) #

gam = {'11': gam11, '22': gam22, '12': gam11, '21': gam22, '13': gam13, '32': gam22, '23': gam22, '33': gam22,
       'eps1': conf.delta, 'eps2': conf.delta, 'eps3': conf.delta}