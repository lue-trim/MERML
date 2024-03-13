import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import sympy as sy
from scipy.optimize import minimize
import copy
import matplotlib as mpl
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def h_t(x, h_0, io3_0, kw=-14):
    '''calculated the concentration of H+ according to IO3-'''
    oh = kw - h_0
    delta_io3 = x - math.pow(10, io3_0)
    b_h = math.pow(10, oh) - math.pow(10, h_0) - 6 * delta_io3
    a_h = 1
    c_h = -1 * math.pow(10, kw)
    delta_h = b_h ** 2 - 4 * a_h * c_h
    return (-1 * b_h + np.sqrt(delta_h)) / 2 * a_h


def result_t(h, io3, i, f=0, k=0):
    '''kinetic model'''
    ht = lambda x: h_t(x, h, io3)
    it = lambda x: i_t(x, i, io3)

    if f == 0:
        k_1 = 1.1 * math.pow(10, 8)
        f = lambda t, x: -1 * k_1 * math.pow(x, 0.94) * ht(x) * ht(x) * math.pow(it(x), 1.55)
    elif f == 1:
        k_1 = 2.5*1e6
        k_2 = 4.2*1e8
        f = lambda t, x : -1 * ht(x) * ht(x) * x * (k_1*it(x)+k_2*it(x)*it(x))
    elif f == 2:
        f = lambda t, x: -1 * k_pred[index] * math.pow(x, 1) * math.pow(ht(x), 1.3) * math.pow(it(x), 2)
    else:
        k_1 = 9.4 * math.pow(10, 7)
        k_2 = 4.6 * math.pow(10, -11)
        k_3 = 9.5
        f = lambda t, x: -1 * k_1 * x * ht(x) * ht(x) * it(x) * (k_2 + k_3 * it(x)) / (
                1 + k_2 + k_3 * it(x))

    result = [math.pow(10, io3)]
    ode15s = scipy.integrate.ode(f).set_integrator('vode', method='bdf', order=15)
    ode15s.set_initial_value(math.pow(10, io3), 0).set_f_params()
    dt = 0.5
    Flag = False
    while ode15s.successful() and ode15s.t < 300:
        result.append(ode15s.integrate(ode15s.t + dt)[0])
    if Flag:
        return None, None, None, None
    else:
        io3_result = result
        i3_func = lambda u: math.pow(10, i) - 5.0 * math.pow(10, io3) + 5 * u - it(u)
        i3_result = list(map(i3_func, result))
        i_result = list(map(it, result))
        h_result = list(map(ht, result))
        return io3_result, i3_result, i_result, h_result


def i_t(x, i0, io3_0):
    '''claculate the concentration of I- according to IO3-'''
    a_i1 = kT
    delta_io3 = x - math.pow(10, io3_0)
    aplha_1 = 5 * delta_io3 + math.pow(10, i0)
    beta_1 = -3 * delta_io3
    b_i1 = -1 * (kT * (aplha_1 + beta_1) + 1)
    c_i1 = kT * aplha_1 * beta_1
    delta_i1 = b_i1 ** 2 - 4 * a_i1 * c_i1
    return -1 * (-1 * b_i1 - np.sqrt(delta_i1)) / 2 / a_i1 + 5 * delta_io3 + math.pow(10, i0)


def func_v(k, h_list, io3_list, i_list, v_list):
    result = []
    for item in zip(h_list, io3_list, i_list, v_list):
        h, io3, i, v = item
        result.append((-1*k*math.pow(h, 1.3)*math.pow(io3, 1)*math.pow(i, 2)+v)**2)
    # print(result)
    return sum(result)[0]


def func_v_list(k, h_list, io3_list, i_list, v_list):
    result = []
    for item in zip(h_list, io3_list, i_list, v_list):
        h, io3, i, v = item
        result.append(k*math.pow(h, 1.3)*math.pow(io3, 1)*math.pow(i, 2))
    return result


def func_k(x_list, h, io3, i):
    ht = lambda x: h_t(x, h, io3)
    it = lambda x: i_t(x, i, io3)

    h_list = np.array(list(map(ht, x_list)))*math.pow(10, 4)
    i_list = np.array(list(map(it, x_list)))*math.pow(10, 3)
    x_list = np.array(x_list)
    v_list = (np.array(x_list[0:-1]) - np.array(x_list[1:]))/0.5

    v_func = lambda k: func_v(k, (np.array(h_list[0:-1])+np.array(h_list[1:]))/2,
                              (np.array(x_list[0:-1])+np.array(x_list[1:]))/2,
                              (np.array(i_list[0:-1])+np.array(i_list[1:]))/2,
                              np.array(v_list))
    # 最小化目标函数
    ori = 1
    step = 0
    r2_list = []
    ori_list = []
    while(True):
        k = minimize(v_func, ori, method='BFGS')
        v_pred = func_v_list(k['x'][0], (np.array(h_list[0:-1])+np.array(h_list[1:]))/2,
                                  (np.array(x_list[0:-1])+np.array(x_list[1:]))/2,
                                  (np.array(i_list[0:-1])+np.array(i_list[1:]))/2,
                                  np.array(v_list))
        r2 = r2_score(v_list, v_pred)
        r2_list.append(r2)
        ori_list.append(ori)
        if (r2) < 0.7:
            ori = ori*0.1
            step += 1
            if step > 10:
                pos = r2_list.index(max(r2_list))
                ori_best = ori_list[pos]
                k = minimize(v_func, ori_best, method='BFGS')
                v_pred = func_v_list(k['x'][0], (np.array(h_list[0:-1]) + np.array(h_list[1:])) / 2,
                                     (np.array(x_list[0:-1]) + np.array(x_list[1:])) / 2,
                                     (np.array(i_list[0:-1]) + np.array(i_list[1:])) / 2,
                                     np.array(v_list))
                r2_temp_list = []
                for s in range(6, 11, 1):
                    v_pred = np.ones(len(v_pred))*math.pow(10, -s)
                    r2_temp_list.append(r2_score(v_list, v_pred))
                pos_temp = r2_temp_list.index(max(r2_temp_list))
                s_best = 6+pos_temp

                r2_temp2_list = []
                for j in range(1, 11, 1):
                    v_pred = np.ones(len(v_pred)) * math.pow(10, -s_best)*(1+(j-5)*0.1)
                    r2_temp2_list.append(r2_score(v_list, v_pred))
                pos_temp2 = r2_temp2_list.index(max(r2_temp2_list))
                v_pred = np.ones(len(v_pred)) * math.pow(10, -s_best) * (1 + (pos_temp2 - 5) * 0.1)
                break
        else:
            break
    # print(r2_score(v_list, v_pred), x_list/x_list[0], v_list)
    return np.array(v_pred), k['x'][0]


def fit_profile(k, x_list, h, io3, i):
    ht = lambda x: h_t(x, h, io3)
    it = lambda x: i_t(x, i, io3)

    h_list = np.array(list(map(ht, x_list)))*math.pow(10, 4)
    i_list = np.array(list(map(it, x_list)))*math.pow(10, 3)
    x_list = np.array(x_list)
    v_list = (np.array(x_list[0:-1]) - np.array(x_list[1:]))/0.5

    v_pred = func_v_list(k, (np.array(h_list[0:-1])+np.array(h_list[1:]))/2,
                              (np.array(x_list[0:-1])+np.array(x_list[1:]))/2,
                              (np.array(i_list[0:-1])+np.array(i_list[1:]))/2,
                              np.array(v_list))
    return np.array(v_pred)


def EP_plot(true, pred):
    mpl.rcParams["mathtext.fontset"] = 'custom'
    mpl.rcParams["mathtext.bf"] = "Arial:bold"
    mpl.rcParams["mathtext.default"] = 'regular'
    plt.figure()
    # mpl.rcParams["markers.fillstyle"] = 'none'
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 110)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels([0, 20, 40, 60, 80, 100],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([0, 20, 40, 60, 80, 100],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('Predicted Conc.' + ' (%)', labelpad=5, fontsize=22, fontweight='500',
                       fontfamily='Arial')
    ax.set_xlabel('Measured Conc.' + ' (%)', labelpad=5, fontsize=22,
                       fontfamily='Arial')
    ax.tick_params(length=4, width=1.5, which='major')
    plt.scatter(true, pred, marker='o', s=125, lw=0.2, alpha=0.6, c='#B7E1FC', edgecolor='black', zorder=1)  #  #D2D2FF
    plt.plot([-10, 110], [-10, 110], c='#C26275', linestyle='dashed', zorder=0)
    ax.text(x=65, y=15, s=r'$R^2$'+' = ' + str(np.round(r2_score(true, pred), 3)), fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    ax.text(x=65, y=5, s='MAE = '+ str(np.round(mean_absolute_error(true, pred), 3)), fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    ax.text(x=65, y=-5, s='RMSE = '+ str(np.round(np.sqrt(mean_squared_error(true, pred)), 3)), fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    plt.show()


def get_k(data, data_time):
    # data_time = np.arange(0, 30, 3)[0:]
    all_time = np.arange(0, 300, 0.5).tolist()
    mask = [all_time.index(t) for t in data_time]
    k_list = []
    true = []
    preds = []
    for index in range(data.shape[0]):
        io3_0 = data[index, 1]
        h_0 = data[index, 0]
        i_0 = data[index, 2]
        pred = data[index, 6:] * math.pow(10, io3_0)
        pred = pred[mask]
        v, k = func_k(pred, h_0, io3_0, i_0)
        k_list.append(k)

        io3_pred_model_3 = [pred[0]]
        for ele in v:
            io3_pred_model_3.append(io3_pred_model_3[-1] - 0.5 * ele)

        pred = (pred / math.pow(10, io3_0))
        pred_model_3 = (np.array(io3_pred_model_3) / math.pow(10, io3_0))

        true.append(pred.tolist()[1:])
        preds.append(pred_model_3.tolist()[1:])

    true = np.array(true)
    pred = np.array(preds)

    true = np.array(true).reshape(1, -1)[0]
    pred = np.array(pred).reshape(1, -1)[0]
    # EP_plot((true) * 100, (pred) * 100)

    return data[:, 0:5], np.array(k_list)


def EP_plot_k(true, pred):
    mpl.rcParams["mathtext.fontset"] = 'custom'
    mpl.rcParams["mathtext.bf"] = "Arial:bold"
    mpl.rcParams["mathtext.default"] = 'regular'
    plt.figure()
    # mpl.rcParams["markers.fillstyle"] = 'none'
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 13)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xticklabels(np.array([0, 3, 6, 9, 12])*2,
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_yticks([0, 3, 6, 9, 12])
    ax.set_yticklabels(np.array([0, 3, 6, 9, 12])*2,
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel(r'Predicted k ' + r'$(\times 10^{-4})$', labelpad=5, fontsize=22, fontweight='500',
                       fontfamily='Arial')
    ax.set_xlabel(r'Measured k ' + r'$(\times 10^{-4})$', labelpad=5, fontsize=22,
                       fontfamily='Arial')
    ax.tick_params(length=4, width=1.5, which='major')
    plt.scatter(true, pred, marker='o', s=125, lw=0.2, alpha=0.6, c='#B7E1FC', edgecolor='black', zorder=1)  #  #D2D2FF
    plt.plot([-1, 13], [-1, 13], c='#C26275', linestyle='dashed', zorder=0)
    ax.text(x=8, y=2.5, s=r'$R^2$'+' = ' + str(np.round(r2_score(true, pred), 3)), fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    ax.text(x=8, y=1, s='MAE = '+ str(np.round(mean_absolute_error(true, pred), 3)), fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    ax.text(x=8, y=-0.5, s='RMSE = '+ str(np.round(np.sqrt(mean_squared_error(true, pred)), 3)), fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    plt.show()


if __name__ == '__main__':
    '''reference model'''
    kw = -14
    T = 25 + 273.15
    kT = math.pow(10, 555 / T + 7.355 - 2.575 * math.log10(T))
    data_time = np.arange(0, 50.5, 5)[0:]

    train = np.loadtxt('Dushman test-io3-train.txt')
    x_train, y_train = get_k(train, data_time)
    test = np.loadtxt('Dushman test-io3.txt')
    x_test, y_test = get_k(test, data_time)
    # print(x_test.shape)

    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(random_state=42)
    rf.fit(x_train, y_train)
    k_pred = rf.predict(x_test)
    # EP_plot_k(y_test*5000, k_pred*5000)

    all_time = np.arange(0, 300, 0.5).tolist()
    mask = [all_time.index(t) for t in data_time]
    true = []
    preds = []
    for index in range(test.shape[0]):
        io3_0 = test[index, 1]
        h_0 = test[index, 0]
        i_0 = test[index, 2]
        profile = test[index, 6:]* math.pow(10, io3_0)
        profile = profile[mask]

        pred = fit_profile(k_pred[index], profile, h_0, io3_0, i_0)
        result = [math.pow(10, io3_0)]
        Flag = False
        for ele in pred:
            if result[-1] - 0.5 * ele >= 0 and not Flag:
                result.append(result[-1] - 0.5 * ele)
            else:
                result.append(1)
                Flag = True
        true.append((profile[1:])/math.pow(10, io3_0))
        preds.append((np.array(result)[1:])/math.pow(10, io3_0))
    true = np.array(true)
    pred = np.array(preds)
    io3_pred = copy.copy(pred)
    true = np.array(true).reshape(1, -1)[0]
    pred = np.array(pred).reshape(1, -1)[0]
    # EP_plot((true) * 100, (pred) * 100)

    test = np.loadtxt('Dushman-i-test.txt')
    i_trues = []
    i_preds = []
    for index in range(test.shape[0]):
        io3_0 = test[index, 1]
        h_0 = test[index, 0]
        i_0 = test[index, 2]
        i_true = test[index, 6:][mask][1:]
        i_trues.append(i_true)
        i_func = lambda x: i_t(x, i_0, io3_0)
        i_pred= np.array(list(map(i_func, io3_pred[index, :]*math.pow(10, io3_0))))
        i_preds.append(np.array(i_pred)/math.pow(10, i_0))

    true = np.array(i_trues)
    pred = np.array(i_preds)
    print(true.shape)
    print(pred.shape)
    true = np.array(true).reshape(1, -1)[0]
    pred = np.array(pred).reshape(1, -1)[0]
    EP_plot((true) * 100, (pred) * 100)
    '''reference model'''








