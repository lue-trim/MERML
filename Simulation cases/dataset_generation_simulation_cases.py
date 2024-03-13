import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time


def kinetic_equation_m1(t, abcd, k1, k2, k3, k4):
    a, b, c, d = abcd
    dadt = k2*d - k1*a*c
    dbdt = k3*d - k4*c*b
    dcdt = (k2+k3)*d - (k1*a + k4*b)*c
    dddt = (k1*a + k4*b)*c - (k2+k3)*d
    return [dadt, dbdt, dcdt, dddt]


def run_simulation_m1():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()


    # k_list = []
    # for index in range(5):
    #     if index == 0:
    #         k1, k2, k3, k4 = [np.random.choice(k1_available), np.random.choice(k2_available),
    #                               np.random.choice(k3_available), np.random.choice(k4_available),
    #                               ]
    #     else:
    #         while (k1, k2, k3, k4) in k_list:
    #             k1, k2, k3, k4 = [np.random.choice(k1_available), np.random.choice(k2_available),
    #                                   np.random.choice(k3_available), np.random.choice(k4_available),
    #                                   ]
    #     k_list.append((k1, k2, k3, k4))
    #     print(k1, k2, k3, k4)
    #     conc_a = (np.array([10, 50, 100, 200, 500, 1000]) * 0.25).tolist()
    #     conc_c = (np.array([1, 2, 4, 6, 8, 10, ]) * 0.01).tolist()
    #     for a in conc_a:
    #         fig, ax = plt.subplots(2, 3)
    #         for c in conc_c:
    #             abcdef = [a, 0, a * c, 0]
    #             sol = solve_ivp(kinetic_equation_m1, [t_start, t_end], abcdef, t_eval=t_eval,
    #                             args=(k1, k2, k3, k4))
    #
    #             pos = conc_c.index(c)
    #             ax_temp = ax[pos // 3][pos % 3]
    #             # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
    #             ax_temp.plot(sol.t, sol.y[0], c='r')
    #             ax_temp.plot(sol.t, sol.y[1], c='g')
    #             ax_temp.plot(sol.t, sol.y[2], c='b')
    #             ax_temp.plot(sol.t, sol.y[3], 'grey')
    #         plt.show()
    #     time.sleep(0.5)

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4 = [np.random.choice(k1_available), np.random.choice(k2_available),
                              np.random.choice(k3_available), np.random.choice(k4_available),]
        else:
            while (k1, k2, k3, k4) in k_list:
                k1, k2, k3, k4 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                  np.random.choice(k3_available), np.random.choice(k4_available), ]
        k_list.append((k1, k2, k3, k4))
        print(k1, k2, k3, k4)
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000]) * 0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10, ]) * 0.01).tolist()
        plt.figure()
        for a in conc_a:
            for c in conc_c:
                abcd = [a, 0, a * c, 0]
                sol = solve_ivp(kinetic_equation_m1, [t_start, t_end], abcd, t_eval=t_eval,
                                args=(k1, k2, k3, k4,))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask] / a)
                plt.plot(sol.t, sol.y[0] / a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m1.npy', x)
    np.save('y_m1.npy', curves)


def kinetic_equation_m2(t, abcde, k1, k2, k3, k4, k5, k6):
    a, b, c, d, e = abcde
    dadt = k2*d - k1*a*c
    dbdt = k3*d - k4*c*b
    dcdt = (k2+k3)*d + 2*k6*e - (k1*a+k4*b+2*k5*c)*c
    dddt = (k1*a + k4*b)*c - (k2+k3)*d
    dedt = k5*c*c - k6*e
    return [dadt, dbdt, dcdt, dddt, dedt]


def run_simulation_m2():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k6_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available), np.random.choice(k6_available)]
        else:
            while (k1, k2, k3, k4, k5, k6) in k_list:
                k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                          np.random.choice(k3_available), np.random.choice(k4_available),
                                          np.random.choice(k5_available), np.random.choice(k6_available)]
        k_list.append((k1, k2, k3, k4, k5, k6))
        print((k1, k2, k3, k4, k5, k6))
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcd = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m2, [t_start, t_end], abcd, t_eval=t_eval, args=(k1, k2, k3, k4, k5, k6))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                ax_temp.plot(sol.t, sol.y[0], c='r')
                ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available), np.random.choice(k6_available)]
        else:
            while (k1, k2, k3, k4, k5, k6) in k_list:
                k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                          np.random.choice(k3_available), np.random.choice(k4_available),
                                          np.random.choice(k5_available), np.random.choice(k6_available)]
        k_list.append((k1, k2, k3, k4, k5, k6))
        print((k1, k2, k3, k4, k5, k6))
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        for a in conc_a:
            for c in conc_c:
                abcd = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m2, [t_start, t_end], abcd, t_eval=t_eval, args=(k1, k2, k3, k4, k5, k6))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m2.npy', x)
    np.save('y_m2.npy', curves)


def kinetic_equation_m3(t, abcde, k1, k2, k3, k4, k5, k6):
    a, b, c, d, e = abcde
    dadt = k2*e - k1*a*d
    dbdt = k3*e - k4*d*b
    dcdt = 2*k6*d - 2*k5*c*c
    dddt = (k2+k3)*e + k5*c*c - (k1*a+k4*b+k6)*d
    dedt = (k1*a + k4*b)*d - (k2+k3)*e
    return [dadt, dbdt, dcdt, dddt, dedt]


def run_simulation_m3():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k6_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available), np.random.choice(k6_available)]
        else:
            while (k1, k2, k3, k4, k5, k6) in k_list:
                k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                          np.random.choice(k3_available), np.random.choice(k4_available),
                                          np.random.choice(k5_available), np.random.choice(k6_available)]
        k_list.append((k1, k2, k3, k4, k5, k6))
        print((k1, k2, k3, k4, k5, k6))
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcd = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m3, [t_start, t_end], abcd, t_eval=t_eval, args=(k1, k2, k3, k4, k5, k6))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                ax_temp.plot(sol.t, sol.y[0], c='r')
                ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available), np.random.choice(k6_available)]
        else:
            while (k1, k2, k3, k4, k5, k6) in k_list:
                k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                          np.random.choice(k3_available), np.random.choice(k4_available),
                                          np.random.choice(k5_available), np.random.choice(k6_available)]
        k_list.append((k1, k2, k3, k4, k5, k6))
        print((k1, k2, k3, k4, k5, k6))
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        for a in conc_a:
            for c in conc_c:
                abcd = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m3, [t_start, t_end], abcd, t_eval=t_eval, args=(k1, k2, k3, k4, k5, k6))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m3.npy', x)
    np.save('y_m3.npy', curves)


def kinetic_equation_m4(t, abcde, k1, k2, k3, k4):
    a, b, c, d, e = abcde
    dadt = k2*e*d - k1*a*c
    dbdt = k3*e*d - k4*c*b
    dcdt = (k2+k3)*e*d - (k1*a+k4*b)*c
    dddt = (k1*a + k4*b)*c - (k2+k3)*e*d
    dedt = (k1*a + k4*b)*c - (k2+k3)*e*d
    return [dadt, dbdt, dcdt, dddt, dedt]


def run_simulation_m4():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      ]
        else:
            while (k1, k2, k3, k4) in k_list:
                k1, k2, k3, k4 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                          np.random.choice(k3_available), np.random.choice(k4_available),
                                         ]
        k_list.append((k1, k2, k3, k4))
        print((k1, k2, k3, k4))
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcd = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m4, [t_start, t_end], abcd, t_eval=t_eval, args=(k1, k2, k3, k4))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                ax_temp.plot(sol.t, sol.y[0], c='r')
                ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        k1, k2, k3, k4 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                  np.random.choice(k3_available), np.random.choice(k4_available),]
        if index == 0:
            k_list.append((k1, k2, k3, k4))
        else:
            while (k1, k2, k3, k4) in k_list:
                k1, k2, k3, k4 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                  np.random.choice(k3_available), np.random.choice(k4_available), ]
            k_list.append((k1, k2, k3, k4))
        print(k1, k2, k3, k4)
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        for a in conc_a:
            for c in conc_c:
                abcd = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m4, [t_start, t_end], abcd, t_eval=t_eval, args=(k1, k2, k3, k4))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m4.npy', x)
    np.save('y_m4.npy', curves)


def kinetic_equation_m5(t, abcde, k1, k2, k3, k4, k5, k6):
    a, b, c, d, e = abcde
    dadt = k2*d - k1*a*c
    dbdt = k5*e - k6*c*b
    dcdt = k2*d + k5*e - (k1*a+k6*b)*c
    dddt = (k1*a + k4*e)*c - (k2+k3*c)*d
    dedt = k3*d*c + k6*b*c - (k5+k4*c)*e
    return [dadt, dbdt, dcdt, dddt, dedt]


def run_simulation_m5():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 0.1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k6_available = (np.array([0.1, 0.5, 1]) * 0.1).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available), np.random.choice(k6_available)]
        else:
            while (k1, k2, k3, k4, k5, k6) in k_list:
                k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                          np.random.choice(k3_available), np.random.choice(k4_available),
                                          np.random.choice(k5_available), np.random.choice(k6_available)]
        k_list.append((k1, k2, k3, k4, k5, k6))
        print((k1, k2, k3, k4, k5, k6))
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcd = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m5, [t_start, t_end], abcd, t_eval=t_eval, args=(k1, k2, k3, k4, k5, k6))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                ax_temp.plot(sol.t, sol.y[0], c='r')
                ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available), np.random.choice(k6_available)]
        else:
            while (k1, k2, k3, k4, k5, k6) in k_list:
                k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                          np.random.choice(k3_available), np.random.choice(k4_available),
                                          np.random.choice(k5_available), np.random.choice(k6_available)]
        k_list.append((k1, k2, k3, k4, k5, k6))
        print((k1, k2, k3, k4, k5, k6))
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000]) * 0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10, ]) * 0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcd = [a, 0, a * c, 0, 0]
                sol = solve_ivp(kinetic_equation_m5, [t_start, t_end], abcd, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5, k6))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask] / a)
                plt.plot(sol.t, sol.y[0] / a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m5.npy', x)
    np.save('y_m5.npy', curves)


def kinetic_equation_m6(t, abcde, k1, k2, k3, k4, k5):
    a, b, c, d, e = abcde
    dadt = k2*e - k1*a*d
    dbdt = k3*e - k4*d*b
    dcdt = -1*k5*c
    dddt = k5*c + (k2+k3)*e - (k1*a+k4*b)*d
    dedt = (k1*a+k4*b)*d - (k2+k3)*e
    return [dadt, dbdt, dcdt, dddt, dedt]


def run_simulation_m6():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.1).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.005).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print((k1, k2, k3, k4, k5))
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcd = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m6, [t_start, t_end], abcd, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                ax_temp.plot(sol.t, sol.y[0], c='r')
                ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print((k1, k2, k3, k4, k5))

        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcd = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m6, [t_start, t_end], abcd, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m6.npy', x)
    np.save('y_m6.npy', curves)


def kinetic_equation_m7(t, abcde, k1, k2, k3, k4, k5, k6):
    a, b, c, d, e = abcde
    dadt = k2*e - k1*a*d + k6*d - k5*a*c
    dbdt = k3*e - k4*d*b
    dcdt = k6*d - k5*a*c
    dddt = k5*a*c - k6*d + (k2+k3)*e - (k1*a+k4*b)*d
    dedt = (k1*a+k4*b)*d - (k2+k3)*e
    return [dadt, dbdt, dcdt, dddt, dedt]


def run_simulation_m7():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.1).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k6_available = (np.array([0.1, 0.5, 1]) * 0.0001).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available), np.random.choice(k6_available)]
        else:
            while (k1, k2, k3, k4, k5, k6) in k_list:
                k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                          np.random.choice(k3_available), np.random.choice(k4_available),
                                          np.random.choice(k5_available), np.random.choice(k6_available)]
        k_list.append((k1, k2, k3, k4, k5, k6))
        print((k1, k2, k3, k4, k5, k6))
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcd = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m7, [t_start, t_end], abcd, t_eval=t_eval, args=(k1, k2, k3, k4, k5, k6))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                # ax_temp.plot(sol.t, sol.y[0], c='r')
                # ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available), np.random.choice(k6_available)]
        else:
            while (k1, k2, k3, k4, k5, k6) in k_list:
                k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                          np.random.choice(k3_available), np.random.choice(k4_available),
                                          np.random.choice(k5_available), np.random.choice(k6_available)]
        k_list.append((k1, k2, k3, k4, k5, k6))
        print((k1, k2, k3, k4, k5, k6))
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000]) * 0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10, ]) * 0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcd = [a, 0, a * c, 0, 0]
                sol = solve_ivp(kinetic_equation_m7, [t_start, t_end], abcd, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5, k6))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask] / a)
                plt.plot(sol.t, sol.y[0] / a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m7.npy', x)
    np.save('y_m7.npy', curves)


def kinetic_equation_m8(t, abcdef, k1, k2, k3, k4, k5, k6):
    a, b, c, d, e, f = abcdef
    dadt = k2*e - k1*a*d
    dbdt = k3*e - k4*d*b
    dcdt = k6*d*f - k5*c
    dddt = k5*c - k6*d*f + (k2+k3)*e - (k1*a+k4*b)*d
    dedt = (k1*a+k4*b)*d - (k2+k3)*e
    dfdt = k5*c - k6*f*d
    return [dadt, dbdt, dcdt, dddt, dedt, dfdt]


def run_simulation_m8():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.1).tolist()
    k6_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available), np.random.choice(k6_available)]
        else:
            while (k1, k2, k3, k4, k5, k6) in k_list:
                k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                          np.random.choice(k3_available), np.random.choice(k4_available),
                                          np.random.choice(k5_available), np.random.choice(k6_available)]
        k_list.append((k1, k2, k3, k4, k5, k6))
        print((k1, k2, k3, k4, k5, k6))
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcdef = [a, 0, a*c, 0, 0, 0]
                sol = solve_ivp(kinetic_equation_m8, [t_start, t_end], abcdef, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5, k6))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                # ax_temp.plot(sol.t, sol.y[0], c='r')
                # ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
                ax_temp.plot(sol.t, sol.y[5], 'orange')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available), np.random.choice(k6_available)]
        else:
            while (k1, k2, k3, k4, k5, k6) in k_list:
                k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                          np.random.choice(k3_available), np.random.choice(k4_available),
                                          np.random.choice(k5_available), np.random.choice(k6_available)]
        k_list.append((k1, k2, k3, k4, k5, k6))
        print((k1, k2, k3, k4, k5, k6))
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000]) * 0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10, ]) * 0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcdef = [a, 0, a * c, 0, 0, 0]
                sol = solve_ivp(kinetic_equation_m8, [t_start, t_end], abcdef, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5, k6))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask] / a)
                plt.plot(sol.t, sol.y[0] / a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m8.npy', x)
    np.save('y_m8.npy', curves)


def kinetic_equation_m9(t, abcde, k1, k2, k3, k4, k5):
    a, b, c, d, e = abcde
    dadt = k2*d - k1*a*c
    dbdt = k3*d - k4*c*b
    dcdt = (k2+k3)*d - (k1*a+k4*b)*c
    dddt = (k1*a+k4*b)*c - (k2+k3)*d
    dedt = k5*c
    return [dadt, dbdt, dcdt, dddt, dedt]


def run_simulation_m9():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.005).tolist()

    # k_list = []
    # for index in range(5):
    #     if index == 0:
    #         k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
    #                                   np.random.choice(k3_available), np.random.choice(k4_available),
    #                                   np.random.choice(k5_available)]
    #     else:
    #         while (k1, k2, k3, k4, k5) in k_list:
    #             k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
    #                                   np.random.choice(k3_available), np.random.choice(k4_available),
    #                                   np.random.choice(k5_available)]
    #     k_list.append((k1, k2, k3, k4, k5))
    #     print(k1, k2, k3, k4, k5)
    #     conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
    #     conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
    #     for a in conc_a:
    #         fig, ax = plt.subplots(2, 3)
    #         for c in conc_c:
    #             abcd = [a, 0, a*c, 0, 0]
    #             sol = solve_ivp(kinetic_equation_m9, [t_start, t_end], abcd, t_eval=t_eval,
    #                             args=(k1, k2, k3, k4, k5))
    #
    #             pos = conc_c.index(c)
    #             ax_temp = ax[pos//3][pos%3]
    #             # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
    #             ax_temp.plot(sol.t, sol.y[0], c='r')
    #             ax_temp.plot(sol.t, sol.y[1], c='g')
    #             ax_temp.plot(sol.t, sol.y[2], c='b')
    #             ax_temp.plot(sol.t, sol.y[3], 'grey')
    #             ax_temp.plot(sol.t, sol.y[4], 'pink')
    #         plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                  np.random.choice(k3_available), np.random.choice(k4_available),
                                  np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcd = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m9, [t_start, t_end], abcd, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m9.npy', x)
    np.save('y_m9.npy', curves)


def kinetic_equation_m10(t, abcdef, k1, k2, k3, k4, k5):
    a, b, c, d, e, f = abcdef
    dadt = k2*d - k1*a*c
    dbdt = k3*d - k4*c*b
    dcdt = (k2+k3)*d - (k1*a+k4*b+k5*e)*c
    dddt = (k1*a+k4*b)*c - (k2+k3)*d
    dedt = -1*k5*c*e
    dfdt = k5*e*c
    return [dadt, dbdt, dcdt, dddt, dedt, dfdt]


def run_simulation_m10():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.005).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcdef = [a, 0, a*c, 0, a*c*0.5, 0]
                sol = solve_ivp(kinetic_equation_m10, [t_start, t_end], abcdef, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                # ax_temp.plot(sol.t, sol.y[0], c='r')
                # ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
                ax_temp.plot(sol.t, sol.y[5], 'orange')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                  np.random.choice(k3_available), np.random.choice(k4_available),
                                  np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcd = [a, 0, a*c, 0, a*c*0.5, 0]
                sol = solve_ivp(kinetic_equation_m10, [t_start, t_end], abcd, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m10.npy', x)
    np.save('y_m10.npy', curves)


def kinetic_equation_m11(t, abcde, k1, k2, k3, k4, k5):
    a, b, c, d, e = abcde
    dadt = k2*d - (k1+k5)*a*c
    dbdt = k3*d - k4*c*b
    dcdt = (k2+k3)*d - (k1*a+k4*b+k5*a)*c
    dddt = (k1*a+k4*b)*c - (k2+k3)*d
    dedt = k5*c*a
    return [dadt, dbdt, dcdt, dddt, dedt]


def run_simulation_m11():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.0005).tolist()

    # k_list = []
    # for index in range(5):
    #     if index == 0:
    #         k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
    #                                   np.random.choice(k3_available), np.random.choice(k4_available),
    #                                   np.random.choice(k5_available)]
    #     else:
    #         while (k1, k2, k3, k4, k5) in k_list:
    #             k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
    #                                   np.random.choice(k3_available), np.random.choice(k4_available),
    #                                   np.random.choice(k5_available)]
    #     k_list.append((k1, k2, k3, k4, k5))
    #     print(k1, k2, k3, k4, k5)
    #     conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
    #     conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
    #     for a in conc_a:
    #         fig, ax = plt.subplots(2, 3)
    #         for c in conc_c:
    #             abcdef = [a, 0, a*c, 0, 0]
    #             sol = solve_ivp(kinetic_equation_m11, [t_start, t_end], abcdef, t_eval=t_eval,
    #                             args=(k1, k2, k3, k4, k5))
    #
    #             pos = conc_c.index(c)
    #             ax_temp = ax[pos//3][pos%3]
    #             # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
    #             ax_temp.plot(sol.t, sol.y[0], c='r')
    #             ax_temp.plot(sol.t, sol.y[1], c='g')
    #             ax_temp.plot(sol.t, sol.y[2], c='b')
    #             ax_temp.plot(sol.t, sol.y[3], 'grey')
    #             ax_temp.plot(sol.t, sol.y[4], 'pink')
    #         plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                  np.random.choice(k3_available), np.random.choice(k4_available),
                                  np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcd = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m11, [t_start, t_end], abcd, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m11.npy', x)
    np.save('y_m11.npy', curves)


def kinetic_equation_m12(t, abcde, k1, k2, k3, k4, k5):
    a, b, c, d, e = abcde
    dadt = k2*d - k1*a*c
    dbdt = k3*d - (k4+k5)*c*b
    dcdt = (k2+k3)*d - (k1*a+k4*b+k5*b)*c
    dddt = (k1*a+k4*b)*c - (k2+k3)*d
    dedt = k5*c*b
    return [dadt, dbdt, dcdt, dddt, dedt]


def run_simulation_m12():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.0005).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcdef = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m12, [t_start, t_end], abcdef, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                ax_temp.plot(sol.t, sol.y[0], c='r')
                ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                  np.random.choice(k3_available), np.random.choice(k4_available),
                                  np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcd = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m12, [t_start, t_end], abcd, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()
        time.sleep(0.5)

    np.save('x_m12.npy', x)
    np.save('y_m12.npy', curves)


def kinetic_equation_m13(t, abcde, k1, k2, k3, k4, k5):
    a, b, c, d, e = abcde
    dadt = k2*d - k1*a*c
    dbdt = k3*d - k4*c*b
    dcdt = (k2+k3)*d - (k1*a+k4*b+2*k5*c)*c
    dddt = (k1*a+k4*b)*c - (k2+k3)*d
    dedt = k5*c*c
    return [dadt, dbdt, dcdt, dddt, dedt]


def run_simulation_m13():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.005).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcdef = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m13, [t_start, t_end], abcdef, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                ax_temp.plot(sol.t, sol.y[0], c='r')
                ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                  np.random.choice(k3_available), np.random.choice(k4_available),
                                  np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcd = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m13, [t_start, t_end], abcd, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m13.npy', x)
    np.save('y_m13.npy', curves)


def kinetic_equation_m14(t, abcde, k1, k2, k3, k4, k5):
    a, b, c, d, e = abcde
    dadt = k2*d - k1*a*c
    dbdt = k3*d - k4*c*b
    dcdt = (k2+k3)*d - (k1*a+k4*b)*c
    dddt = (k1*a+k4*b)*c - (k2+k3+k5)*d
    dedt = k5*d
    return [dadt, dbdt, dcdt, dddt, dedt]


def run_simulation_m14():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.005).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcdef = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m14, [t_start, t_end], abcdef, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                # ax_temp.plot(sol.t, sol.y[0], c='r')
                # ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                  np.random.choice(k3_available), np.random.choice(k4_available),
                                  np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcd = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m14, [t_start, t_end], abcd, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m14.npy', x)
    np.save('y_m14.npy', curves)


def kinetic_equation_m15(t, abcdef, k1, k2, k3, k4, k5):
    a, b, c, d, e, f = abcdef
    dadt = k2*d - k1*a*c
    dbdt = k3*d - k4*c*b
    dcdt = (k2+k3)*d - (k1*a+k4*b)*c
    dddt = (k1*a+k4*b)*c - (k2+k3+k5*e)*d
    dedt = -k5*d*e
    dfdt = k5*e*d
    return [dadt, dbdt, dcdt, dddt, dedt, dfdt]


def run_simulation_m15():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.005).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcdef = [a, 0, a*c, 0, a*c*0.5, 0]
                sol = solve_ivp(kinetic_equation_m15, [t_start, t_end], abcdef, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                # ax_temp.plot(sol.t, sol.y[0], c='r')
                # ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
                ax_temp.plot(sol.t, sol.y[5], 'orange')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                  np.random.choice(k3_available), np.random.choice(k4_available),
                                  np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcd = [a, 0, a*c, 0, a*c*0.5, 0]
                sol = solve_ivp(kinetic_equation_m15, [t_start, t_end], abcd, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m15.npy', x)
    np.save('y_m15.npy', curves)


def kinetic_equation_m16(t, abcde, k1, k2, k3, k4, k5):
    a, b, c, d, e = abcde
    dadt = k2*d - k1*a*c - k5*a*d
    dbdt = k3*d - k4*c*b
    dcdt = (k2+k3)*d - (k1*a+k4*b)*c
    dddt = (k1*a+k4*b)*c - (k2+k3+k5*a)*d
    dedt = k5*d*a
    return [dadt, dbdt, dcdt, dddt, dedt]


def run_simulation_m16():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcde = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m16, [t_start, t_end], abcde, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                # ax_temp.plot(sol.t, sol.y[0], c='r')
                # ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                  np.random.choice(k3_available), np.random.choice(k4_available),
                                  np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcde = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m16, [t_start, t_end], abcde, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m16.npy', x)
    np.save('y_m16.npy', curves)


def kinetic_equation_m17(t, abcde, k1, k2, k3, k4, k5):
    a, b, c, d, e = abcde
    dadt = k2*d - k1*a*c
    dbdt = k3*d - k4*c*b - k5*b*d
    dcdt = (k2+k3)*d - (k1*a+k4*b)*c
    dddt = (k1*a+k4*b)*c - (k2+k3+k5*b)*d
    dedt = k5*d*b
    return [dadt, dbdt, dcdt, dddt, dedt]


def run_simulation_m17():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcde = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m17, [t_start, t_end], abcde, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                # ax_temp.plot(sol.t, sol.y[0], c='r')
                # ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                  np.random.choice(k3_available), np.random.choice(k4_available),
                                  np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcde = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m17, [t_start, t_end], abcde, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m17.npy', x)
    np.save('y_m17.npy', curves)


def kinetic_equation_m18(t, abcde, k1, k2, k3, k4, k5):
    a, b, c, d, e = abcde
    dadt = k2*d - k1*a*c
    dbdt = k3*d - k4*c*b
    dcdt = (k2+k3)*d - (k1*a+k4*b)*c
    dddt = (k1*a+k4*b)*c - (k2+k3+k5*d)*d
    dedt = k5*d*d
    return [dadt, dbdt, dcdt, dddt, dedt]


def run_simulation_m18():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcde = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m18, [t_start, t_end], abcde, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                # ax_temp.plot(sol.t, sol.y[0], c='r')
                # ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                  np.random.choice(k3_available), np.random.choice(k4_available),
                                  np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcde = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m18, [t_start, t_end], abcde, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m18.npy', x)
    np.save('y_m18.npy', curves)


def kinetic_equation_m19(t, abcde, k1, k2, k3, k4, k5):
    a, b, c, d, e = abcde
    dadt = k2*d - k1*a*c
    dbdt = k3*d - k4*c*b
    dcdt = (k2+k3)*d - (k1*a+k4*b+k5*d)*c
    dddt = (k1*a+k4*b)*c - (k2+k3+k5*c)*d
    dedt = k5*c*d
    return [dadt, dbdt, dcdt, dddt, dedt]


def run_simulation_m19():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcde = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m19, [t_start, t_end], abcde, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                # ax_temp.plot(sol.t, sol.y[0], c='r')
                # ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                  np.random.choice(k3_available), np.random.choice(k4_available),
                                  np.random.choice(k5_available)]
        else:
            while (k1, k2, k3, k4, k5) in k_list:
                k1, k2, k3, k4, k5 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available)]
        k_list.append((k1, k2, k3, k4, k5))
        print(k1, k2, k3, k4, k5)
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcde = [a, 0, a*c, 0, 0]
                sol = solve_ivp(kinetic_equation_m19, [t_start, t_end], abcde, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m19.npy', x)
    np.save('y_m19.npy', curves)


def kinetic_equation_m20(t, abcdef, k1, k2, k3, k4, k5, k6):
    a, b, c, d, e, f = abcdef
    dadt = k2*d - k1*a*c
    dbdt = k3*d - k4*c*b
    dcdt = (k2+k3)*d - (k1*a+k4*b+k5)*c
    dddt = (k1*a+k4*b)*c - (k2+k3+k6)*d
    dedt = k5*c
    dfdt = k6*d
    return [dadt, dbdt, dcdt, dddt, dedt, dfdt]


def run_simulation_m20():
    np.random.seed(0)
    t_start = 0
    t_end = 1000
    # t_eval = [0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12, 14, 16, 18, 20]
    t_eval = np.linspace(t_start, t_end, 2000).tolist()
    mask = [0, 99, 199, 399, 799, 1199, 1599, 1999]

    k1_available = (np.array([0.1, 0.5, 1]) * 0.01).tolist()
    k2_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k3_available = (np.array([0.1, 0.5, 1]) * 10).tolist()
    k4_available = (np.array([0.1, 0.5, 1]) * 1).tolist()
    k5_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()
    k6_available = (np.array([0.1, 0.5, 1]) * 0.001).tolist()

    k_list = []
    for index in range(5):
        if index == 0:
            k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available), np.random.choice(k6_available)]
        else:
            while (k1, k2, k3, k4, k5, k6) in k_list:
                k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available), np.random.choice(k6_available)]
        k_list.append((k1, k2, k3, k4, k5, k6))
        print(k1, k2, k3, k4, k5, k6)
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        for a in conc_a:
            fig, ax = plt.subplots(2, 3)
            for c in conc_c:
                abcdef = [a, 0, a*c, 0, 0, 0]
                sol = solve_ivp(kinetic_equation_m20, [t_start, t_end], abcdef, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5, k6))

                pos = conc_c.index(c)
                ax_temp = ax[pos//3][pos%3]
                # ax_temp.scatter(sol.t[mask], sol.y[0][mask])
                # ax_temp.plot(sol.t, sol.y[0], c='r')
                # ax_temp.plot(sol.t, sol.y[1], c='g')
                ax_temp.plot(sol.t, sol.y[2], c='b')
                ax_temp.plot(sol.t, sol.y[3], 'grey')
                ax_temp.plot(sol.t, sol.y[4], 'pink')
                ax_temp.plot(sol.t, sol.y[5], 'orange')
            plt.show()

    curves = []
    x = []
    k_list = []
    num_type = 10
    for index in range(num_type):
        if index == 0:
            k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                  np.random.choice(k3_available), np.random.choice(k4_available),
                                  np.random.choice(k5_available), np.random.choice(k6_available)]
        else:
            while (k1, k2, k3, k4, k5, k6) in k_list:
                k1, k2, k3, k4, k5, k6 = [np.random.choice(k1_available), np.random.choice(k2_available),
                                      np.random.choice(k3_available), np.random.choice(k4_available),
                                      np.random.choice(k5_available), np.random.choice(k6_available)]
        k_list.append((k1, k2, k3, k4, k5, k6))
        print(k1, k2, k3, k4, k5, k6)
        type_encode = [0 for j in range(num_type)]
        type_encode[index] = 1
        conc_a = (np.array([10, 50, 100, 200, 500, 1000])*0.25).tolist()
        conc_c = (np.array([1, 2, 4, 6, 8, 10,])*0.01).tolist()
        plt.figure()
        plt.ylim(-0.1, 1.1)
        for a in conc_a:
            for c in conc_c:
                abcdef = [a, 0, a*c, 0, 0, 0]
                sol = solve_ivp(kinetic_equation_m20, [t_start, t_end], abcdef, t_eval=t_eval,
                                args=(k1, k2, k3, k4, k5, k6))
                pos = conc_c.index(c)
                plt.scatter(sol.t[mask], sol.y[0][mask]/a)
                plt.plot(sol.t, sol.y[0]/a)

                x_temp = type_encode + [a, a * c]
                x.append(x_temp)
                curves.append(sol.y[0][mask] / a)
        plt.show()

    np.save('x_m20.npy', x)
    np.save('y_m20.npy', curves)


if __name__ == '__main__':
    run_simulation_m10()

