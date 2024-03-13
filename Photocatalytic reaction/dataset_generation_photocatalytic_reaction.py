import numpy as np
import matplotlib.pyplot as plt
import math


def second_kinetic(c0, k, time):
    curves = []
    for t in time:
        ct = c0/(1+c0*k*t)
        curves.append(ct/c0)
    return curves


def run_simulation():
    curves = []
    np.random.seed(0)
    t_start = 0
    t_end = 210
    time = np.linspace(t_start, t_end, 210).tolist()
    mask = [0, 9, 29, 59, 99, 149, 209]

    x = []
    k_list = [1.112, 1.274, 1.640, 2.337, 7.220, 0.399, 1.140, 3.577, 0.981, 4.748]
    c0_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for k in k_list:
        pos = k_list.index(k)
        k = k*0.001
        type_encoded = [0 for j in range(len(k_list))]
        type_encoded[pos] = 1
        for c0 in c0_list:
            x.append(type_encoded+[c0])
            curve = second_kinetic(c0, k, time)
            curves.append(np.array(curve)[mask])
            plt.plot(time, curve)
            plt.scatter(np.array(time)[mask], np.array(curve)[mask])
    plt.show()

    np.save('x_second.npy', x)
    np.save('y_second.npy', curves)