import pandas as pd
import numpy as np
import xgboost as xgb
from joblib import dump, load
import copy
import math
from sklearn.metrics import r2_score
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib as mpl


def preprocess_data_ori():
    for index in range(1, 13, 1):
        data_temp = np.array(pd.read_excel('PMAR-origin.xlsx', sheet_name='Sheet' + str(index)))
        time = data_temp[:, 13]

        for j in range(int(data_temp.shape[0]/5)):
            time_temp = time[j*5:j*5+5] - time[j*5]
            for l in range(time_temp.shape[0]):
                time_temp[l] = np.round(time_temp[l], 1)
            time[j * 5:j * 5 + 5] = time_temp

        for m in range(int(data_temp.shape[0]/5)):
            data_temp_0 = data_temp[m*4:m*4+4, 1:13]
            temp = copy.copy(data_temp_0[0, :])
            temp[6:8] = temp[3:5]
            data_ele = np.r_[temp.reshape(1, -1), data_temp_0]
            if m == 0:
                data_new = data_ele
            else:
                data_new = np.r_[data_new, data_ele]
        data_new = np.c_[np.c_[data_new[:, 0:6], time.reshape(-1, 1)], data_new[:, 6:]]
        data_all = data_new if index == 1 else np.r_[data_all, data_new]

    switch_a = {'4-Methoxybenzaldehyde': 'A3',
                '2-Chlorobenzaldehyde': 'A1',
                '4-Chlorobenzaldehyde': 'A4',
                '4-Nitrobenzaldhyde': 'A2'
                }
    switch_b = {'Acetone': 'B1',
                'Cyclohexanone': 'B2',
                }
    switch_c = {'Proline': 'C1',
                'Proline Tetrazole': 'C2',
                'Proline-OTBS': 'C3',
                }
    for index in range(data_all.shape[0]):
        data_all[index, 0] = switch_a.get(data_all[index, 0])
        data_all[index, 1] = switch_b.get(data_all[index, 1])
        data_all[index, 2] = switch_c.get(data_all[index, 2])
    np.savetxt('PMAR.txt', data_all, fmt='%s')


def huber(gap, sigma=0.3):
    grad = np.sign(sigma) * sigma * gap / np.sqrt(sigma ** 2 + gap ** 2)
    hess = np.sign(sigma) * (sigma ** 3) / np.power(sigma ** 2 + gap ** 2, 1.5)
    return grad, hess


def huber_object_matrix(pred, data_matrix):
    gap = pred - data_matrix.get_label()
    grad, hess = huber(gap)
    return grad, hess


def load(path='data.xlsx'):
    for index in range(1, 13, 1):
        data_temp = np.array(pd.read_excel(path, sheet_name='Sheet'+str(index)))
        if index == 1:
            data = data_temp
        else:
            data = np.r_[data, data_temp]
    return data


def switch_encode(input):
    a = input[0]
    s = input[1]
    if a == 'A':
        temp = [0, 0, 0, 0]
        temp[int(s)-1] = 1
    elif a == 'B':
        temp = [0, 0]
        temp[int(s)-1] = 1
    elif a == 'C':
        temp = [0, 0, 0]
        temp[int(s)-1] = 1
    else:
        raise ValueError
    return temp


def preprocess(data):
    samples = []
    # for i in range(int(data.shape[0]/5)):
    #     data_temp = data[i*5:(i+1)*5, :]
    #     time = data_temp[:, -2] - data_temp[0, -2]
    #     rent_ratio = data_temp[:, -1]/data_temp[0, -1]
    #     conc = data_temp[0, 3:6].tolist()

    for i in range(int(data.shape[0]/5)):
        data_temp = data[i*5:(i+1)*5, :]
        time = data_temp[:, 6]
        rent_ratio = data_temp[:, 7].astype(float)/float(data_temp[0, 3])  # 7 and 3 for A and 8 and 4 for B
        conc = data_temp[0, 3:6].tolist()
        # print(conc)

        for j in range(3):
            if j == 0:
                species = switch_encode(data_temp[0, j])
            else:
                species += switch_encode(data_temp[0, j])
        sample = species+conc + time.tolist() + rent_ratio.tolist()
        samples.append(sample)
    return np.array(samples)


def model_train(train_data, params, num, lag=2, shifts=[1],):
    train = []
    for index in range(train_data.shape[0]):
        x = train_data[index, 0:12]
        for j in range(4):
            t = train_data[index, 13+j]
            r = train_data[index, 18+j]
            p_t = train_data[index, 12+j]
            r_t = train_data[index, 17+j]
            x_plus = [p_t, t, t-p_t, r_t]
            x_temp = x.tolist() + x_plus
            train.append(x_temp+[r])
        if lag == 2:
            for k in range(3):
                t = train_data[index, 14 + k]
                r = train_data[index, 19 + k]
                p_t = train_data[index, 12 + k]
                r_t = train_data[index, 17 + k]
                x_plus = [p_t, t, t - p_t, r_t]
                x_temp = x.tolist() + x_plus
                train.append(x_temp+[r])
    train = np.array(train)
    train_matrix = xgb.DMatrix(train[:, :-1], label=train[:, -1])

    model_train = xgb.train(params, dtrain=train_matrix,
                            num_boost_round=num) # obj=huber_object_matrix,
    y_pred = model_train.predict(train_matrix)
    # EP_plot(train[:, -1]*100, y_pred*100)
    dump(model_train, 'MCB')
    return model_train


def model_test(model_trained, test_data, lag=2, shifts=[1]):
    result = [[], []]
    for num_p in range(4):
        x = test_data[:, 0:12]
        if num_p == 0:
            p_t = test_data[:, 12+num_p]
            p_r = test_data[:, 17+num_p]
            for k in range(len(result)):
                t = test_data[:, 12+num_p+k+1]
                x_plus = np.c_[np.c_[p_t.reshape(-1, 1), t.reshape(-1, 1)], np.c_[(t-p_t).reshape(-1, 1), p_r.reshape(-1, 1)]]
                x_test = np.c_[x, x_plus]
                pred_temp = model_trained.predict(xgb.DMatrix(x_test, label=test_data[:, 16+num_p+k+1].reshape(-1, 1)))
                result[k].append(pred_temp)
        elif num_p == 1:
            p_t = test_data[:, 12+num_p]
            p_r = result[0][num_p - 1]
            for k in range(len(result)):
                t = test_data[:, 12+num_p+k+1]
                x_plus = np.c_[
                    np.c_[p_t.reshape(-1, 1), t.reshape(-1, 1)], np.c_[(t - p_t).reshape(-1, 1), p_r.reshape(-1, 1)]]
                x_test = np.c_[x, x_plus]
                pred_temp = model_trained.predict(xgb.DMatrix(x_test, label=test_data[:, 16 + num_p + k + 1].reshape(-1, 1)))
                result[k].append(pred_temp)
        else:
            p_t = test_data[:, 12+num_p]
            p_r = (result[0][num_p-1] + result[1][num_p-2])/2
            for k in range(len(result)):
                t = test_data[:, 12+num_p+k+1]
                x_plus = np.c_[
                    np.c_[p_t.reshape(-1, 1), t.reshape(-1, 1)], np.c_[(t - p_t).reshape(-1, 1), p_r.reshape(-1, 1)]]
                x_test = np.c_[x, x_plus]
                pred_temp = model_trained.predict(xgb.DMatrix(x_test))
                result[k].append(pred_temp)
    for index in range(4):
        if index == 0:
            pred_result = result[0][0].reshape(-1, 1)
        else:
            pred_result = np.c_[pred_result, ((result[0][index] + result[1][index - 1]) / 2).reshape(-1, 1)]
    true = test_data[:, -4:].reshape(-1, 1)
    pred = pred_result.reshape(-1, 1)
    EP_plot(true * 100, pred * 100)

    # true = np.c_[np.ones((test_data.shape[0], 1)), test_data[:, -4:]]
    # pred = np.c_[np.ones((test_data.shape[0], 1)), pred_result]
    # import matplotlib.pyplot as plt
    # color = ["#C8557C", "#A050A0", "#4292C6", "#5F9EA0", "#E9967A", "#F5DEB3", "#663366",]
    # plt.figure()
    # left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    # rect = [left, bottom, width, height]
    # ax = plt.axes(rect)
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['right'].set_linewidth(1.5)
    # ax.spines['top'].set_linewidth(1.5)
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.set_xlim(-2.5, 25)
    # ax.set_xticks([0, 5, 10, 15, 20, 25])
    # ax.set_xticklabels([0, 5, 10, 15, 20, 25],
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_ylim(-0.1, 1.1)
    # ax.set_yticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    # ax.set_yticklabels(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_ylabel('Conversion', labelpad=5, fontsize=22, fontweight='500')
    # ax.set_xlabel('Time (min)', labelpad=5, fontsize=22)
    # ax.tick_params(length=4, width=1.5, which='major')
    # for index in range(true.shape[0]):
    #     # if index != 6:
    #     plt.scatter([0, 5.5, 11, 16.5, 22], (1-true[index, :]).tolist(), c=color[index%6], marker='o')
    #     plt.plot([0, 5.5, 11, 16.5, 22], (1-pred[index, :]).tolist(), c=color[index%6])
    # plt.savefig('E-P.jpg')


def func_line(x, k):
    return k*x


def EP_plot_rate(true, pred):
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
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_xticks(np.array([0, 20, 40, 60, 80, 100])/20)
    ax.set_xticklabels(['0E-3', '1E-3', '2E-3', '3E-3', '4E-3', '5E-3'],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_yticks(np.array([0, 20, 40, 60, 80, 100])/20)
    ax.set_yticklabels(['0E-3', '1E-3', '2E-3', '3E-3', '4E-3', '5E-3'],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('Predicted Rate (M/s)', labelpad=5, fontsize=22, fontweight='500',
                  fontfamily='Arial')
    ax.set_xlabel('Measured Rate (M/s)', labelpad=5, fontsize=22,
                  fontfamily='Arial')
    ax.tick_params(length=4, width=1.5, which='major')
    plt.scatter(true, pred, marker='o', s=125, lw=0.2, alpha=0.6, c='#B7E1FC', edgecolor='black',
                zorder=1)  # #D2D2FF
    plt.plot([-0.5, 5.5], [-0.5, 5.5], c='#C26275', linestyle='dashed', zorder=0)
    ax.text(x=65/20, y=15/20, s=r'$R^2$' + ' = ' + str(np.round(r2_score(true, pred), 3)),
            fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    ax.text(x=65/20, y=5/20, s='MAE = ' + str(np.round(mean_absolute_error(true, pred), 3)),
            fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    ax.text(x=65/20, y=-5/20, s='RMSE = ' + str(np.round(np.sqrt(mean_squared_error(true, pred)), 3)),
            fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    plt.show()


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
    plt.scatter(true, pred, marker='o', s=125, lw=0.2, alpha=0.6, c='#B7E1FC', edgecolor='black',
                zorder=1)  # #D2D2FF
    plt.plot([-10, 110], [-10, 110], c='#C26275', linestyle='dashed', zorder=0)
    ax.text(x=65, y=15, s=r'$R^2$' + ' = ' + str(np.round(r2_score(true, pred), 3)),
            fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    ax.text(x=65, y=5, s='MAE = ' + str(np.round(mean_absolute_error(true, pred), 3)),
            fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    ax.text(x=65, y=-5, s='RMSE = ' + str(np.round(np.sqrt(mean_squared_error(true, pred)), 3)),
            fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    plt.show()


if __name__ == '__main__':
    '''data preprocess'''
    # data = load()

    data_ori = np.loadtxt('PMAR.txt', dtype=str)
    data_ori_rate = np.array(pd.read_excel('PMAR-merged.xlsx', sheet_name='Sheet1', header=None))

    '''encode and dataset split'''
    data = preprocess(data_ori).astype(float)
    mask = [i for i in range(data.shape[0])]
    np.random.seed(0)
    ratio = 0.2
    np.random.shuffle(mask)
    train = data[mask[0:-int(data.shape[0]*ratio)], :]
    test = data[mask[-int(data.shape[0]*ratio):], :]

    '''reference model'''
    test_ori = []
    k_b = []
    rate_list = []
    rate_pred_list = []
    for index in mask[-int(data.shape[0]*ratio):]:
        test_profile = data_ori_rate[[index*5+j for j in range(1, 5, 1)], :]
        rate = test_profile[:, -1].tolist()
        values = []
        for l in range(4):
            a, b, c = test_profile[l, 7], test_profile[l, 8], test_profile[l, 9]
            oa, ob, oc = test_profile[l, 10], test_profile[l, 11], test_profile[l, 12]
            value = math.pow(a, oa)*math.pow(b, ob)*math.pow(c, oc)
            values.append(value)
        k, b = np.polyfit(values, rate, 1)
        # k = curve_fit(func_line, values, rate)[0][0]
        rate_pred = [k * v + b for v in values]

        rate_list.append(rate)
        rate_pred_list.append(rate_pred)
        # print(r2_score(rate, rate_pred))
        k_b.append([k, b] + [oa, ob, oc])
    # EP_plot_rate(np.array(rate_list).reshape(1, -1)[0]*1000, np.array(rate_pred_list).reshape(1, -1)[0]*1000)

    true = []
    pred = []
    for s in range(test.shape[0]):
        a0, b0, c0 = test[s, 9], test[s, 10], test[s, 11]
        conc = test[s, -4:]*a0
        time = test[s, 13:17]
        k, b, oa, ob, oc = k_b[s]
        print(oa, ob, oc)
        performance = []
        kk_list = np.arange(0.1, 1.5, 0.01)
        for kk in kk_list:
            func = lambda t, x: -(kk*math.pow(x, oa)*math.pow(x-a0+b0, ob)*math.pow(c0, oc)+b)

            conc_pred = []
            result = []
            ode15s = scipy.integrate.ode(func).set_integrator('vode', method='bdf', order=15)
            ode15s.set_initial_value(a0, 0).set_f_params()
            dt = 0.1
            Flag = False
            while ode15s.successful() and ode15s.t < 30:
                if np.round(ode15s.t + dt, 2) in time:
                    conc_pred.append(ode15s.integrate(ode15s.t + dt)[0])
                result.append(ode15s.integrate(ode15s.t + dt)[0])
            mae = mean_absolute_error(conc_pred, conc)
            performance.append(mae)
        pos_mask = np.argsort(performance)
        # print(pos_mask)
        # print(np.array(performance)[pos_mask])

        kk_best = kk_list[pos_mask[0]]
        func = lambda t, x: -kk * (k * math.pow(x, oa) * math.pow(x - a0 + b0, ob) * math.pow(c0, oc) + b)
        conc_pred = []
        result = []
        ode15s = scipy.integrate.ode(func).set_integrator('vode', method='bdf', order=15)
        ode15s.set_initial_value(a0, 0).set_f_params()
        dt = 0.1
        Flag = False
        while ode15s.successful() and ode15s.t < 30:
            if np.round(ode15s.t + dt, 2) in time:
                conc_pred.append(ode15s.integrate(ode15s.t + dt)[0])
            result.append(ode15s.integrate(ode15s.t + dt)[0])
        true.append(conc / a0)
        pred.append(np.array(conc_pred) / a0)

    EP_plot((np.array(true).reshape(1, -1)[0])*100, (np.array(pred).reshape(1, -1)[0])*100)

    '''ERML'''
    params = {'eta': 0.1, 'max_depth': 6, 'min_child_weight': 4}
    # params = {'eta': 0.1, 'max_depth': 4, 'min_child_weight': 4}
    model_trained = model_train(train, params, num=500)
    # model_trained = load('MCB')
    model_test(model_trained, test)
