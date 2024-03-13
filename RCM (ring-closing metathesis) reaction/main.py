import xgboost as xgb
from joblib import dump, load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
import matplotlib as mpl
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


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
    # plt.savefig('picture.svg', dpi=600)
    plt.show()


def EP_plot_para(true, pred, type=0):
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
    if type == 0:
        true = true*1000
        pred = pred*1000
        ax.set_xlim(-0.5, 14.5)
        ax.set_ylim(-0.5, 14.5)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax.set_xticklabels([0, 2, 4, 6, 8, 10, 12, 14],
                           fontfamily='Arial', fontsize=18, fontweight='500')
        ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax.set_yticklabels([0, 2, 4, 6, 8, 10, 12, 14],
                           fontfamily='Arial', fontsize=18, fontweight='500')
        ax.set_xlabel(r'$K_{act}$'+ ' ' + r'$(\times 10^{-3})$', labelpad=5, fontsize=22, fontweight='500',
                      fontfamily='Arial')
        ax.set_ylabel(r'$K_{act}$'+' (with noise, ' + r'$\times 10^{-3})$', labelpad=5, fontsize=22,
                      fontfamily='Arial')
        ax.tick_params(length=4, width=1.5, which='major')
        plt.scatter(true, pred, marker='o', s=125, lw=0.2, alpha=0.6, c='#B7E1FC', edgecolor='black',
                    zorder=1)  # #D2D2FF
        plt.plot([-0.5, 14.5], [-0.5, 14.5], c='#C26275', linestyle='dashed', zorder=0)
        ax.text(x=9, y=3.5, s=r'$R^2$' + ' = ' + str(np.round(r2_score(true, pred), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
        ax.text(x=9, y=2, s='MAE = ' + str(np.round(mean_absolute_error(true, pred), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
        ax.text(x=9, y=0.5, s='RMSE = ' + str(np.round(np.sqrt(mean_squared_error(true, pred)), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
        # plt.savefig('picture.svg', dpi=600)
        plt.show()
    elif type == 1:
        true = true*1000
        pred = pred*1000
        ax.set_xlim(-1.5, 45.5)
        ax.set_ylim(-1.5, 45.5)
        ax.set_xticks([0, 15, 30, 45])
        ax.set_xticklabels([0, 15, 30, 45],
                           fontfamily='Arial', fontsize=18, fontweight='500')
        ax.set_yticks([0, 15, 30, 45])
        ax.set_yticklabels([0, 15, 30, 45],
                           fontfamily='Arial', fontsize=18, fontweight='500')
        ax.set_xlabel(r'$K_{dec}$'+ ' ' + r'$(\times 10^{-3})$', labelpad=5, fontsize=22, fontweight='500',
                      fontfamily='Arial')
        ax.set_ylabel(r'$K_{dec}$'+' (with noise, ' + r'$\times 10^{-3})$', labelpad=5, fontsize=22,
                      fontfamily='Arial')
        ax.tick_params(length=4, width=1.5, which='major')
        plt.scatter(true, pred, marker='o', s=125, lw=0.2, alpha=0.6, c='#B7E1FC', edgecolor='black',
                    zorder=1)  # #D2D2FF
        plt.plot([-1.5, 45.5], [-1.5, 45.5], c='#C26275', linestyle='dashed', zorder=0)
        ax.text(x=27, y=12, s=r'$R^2$' + ' = ' + str(np.round(r2_score(true, pred), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
        ax.text(x=27, y=7, s='MAE = ' + str(np.round(mean_absolute_error(true, pred), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
        ax.text(x=27, y=2, s='RMSE = ' + str(np.round(np.sqrt(mean_squared_error(true, pred)), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
        plt.show()
    elif type == 2:
        true = true * 1000
        pred = pred * 1000
        ax.set_xlim(-1.5/2, 45.5/2)
        ax.set_ylim(-1.5/2, 45.5/2)
        ax.set_xticks(np.array([0, 15, 30, 45])/2)
        ax.set_xticklabels(np.array([0, 15, 30, 45])/2,
                           fontfamily='Arial', fontsize=18, fontweight='500')
        ax.set_yticks(np.array([0, 15, 30, 45])/2)
        ax.set_yticklabels(np.array([0, 15, 30, 45])/2,
                           fontfamily='Arial', fontsize=18, fontweight='500')
        ax.set_xlabel(r'$K_{s}$' + ' ' + r'$(\times 10^{-3})$', labelpad=5, fontsize=22, fontweight='500',
                      fontfamily='Arial')
        ax.set_ylabel(r'$K_{s}$' + ' (with noise, ' + r'$\times 10^{-3})$', labelpad=5, fontsize=22,
                      fontfamily='Arial')
        ax.tick_params(length=4, width=1.5, which='major')
        plt.scatter(true, pred, marker='o', s=125, lw=0.2, alpha=0.6, c='#B7E1FC', edgecolor='black',
                    zorder=1)  # #D2D2FF
        plt.plot([-1.5/2, 45.5/2], [-1.5/2, 45.5/2], c='#C26275', linestyle='dashed', zorder=0)
        ax.text(x=27/2, y=12/2, s=r'$R^2$' + ' = ' + str(np.round(r2_score(true, pred), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
        ax.text(x=27/2, y=7/2, s='MAE = ' + str(np.round(mean_absolute_error(true, pred), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
        ax.text(x=27/2, y=2/2, s='RMSE = ' + str(np.round(np.sqrt(mean_squared_error(true, pred)), 3)),
                fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
        plt.show()
    plt.show()


def model_train(train_data, params, round, lag=2, shifts=[1],):
    train = []
    length = 12
    num = train_data.shape[1]-length-1
    time = [0.5, 1, 2, 4, 7, 11, 16, 23, 30]
    time = np.array([1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30])
    # time = np.array([4, 8, 12, 16, 20, 24, 28])

    for index in range(train_data.shape[0]):
        x = train_data[index, 0:length]
        for j in range(num):
            t = time[j]
            r = train_data[index, length+1+j]
            p_t = time[j-1] if j >= 1 else 0
            r_t = train_data[index, length+j]
            x_plus = [p_t, t, t-p_t, r_t]
            x_temp = x.tolist() + x_plus
            train.append(x_temp+[r])
        if lag == 2:
            for k in range(num-1):
                t = time[k+1]
                r = train_data[index, length+2+k]
                p_t = time[k-1] if k >= 1 else 0
                r_t = train_data[index, length+k]
                x_plus = [p_t, t, t - p_t, r_t]
                x_temp = x.tolist() + x_plus
                train.append(x_temp+[r])
    train = np.array(train)
    try:
        raise ValueError
        model = RandomForestRegressor(random_state=42, min_samples_leaf=1, n_estimators=50)
        model_train = model.fit(train[:, :-1], train[:, -1])
        y_pred = model_train.predict(train[:, :-1])
    except:
        train_matrix = xgb.DMatrix(train[:, :-1], label=train[:, -1])
        model_train = xgb.train(params, dtrain=train_matrix,
                                num_boost_round=round) # obj=huber_object_matrix,
        y_pred = model_train.predict(train_matrix)

    # EP_plot(train[:, -1]*100, y_pred*100)
    dump(model_train, 'MCB')
    return model_train


def model_test(model_trained, test_data, lag=2, shifts=[1]):
    result = [[], []]
    length = 12
    num = train_data.shape[1] - length - 1
    time = [0.5, 1, 2, 4, 7, 11, 16, 23, 30]
    time = np.array([1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30])
    # time = np.array([4, 8, 12, 16, 20, 24, 28])

    for num_p in range(num):
        x = test_data[:, 0:length]
        if num_p == 0:
            p_t = time[num_p-1] if num_p >= 1 else 0
            p_r = test_data[:, length+num_p]
            for k in range(len(result)):
                t = time[num_p+k]
                x_plus = np.c_[np.c_[p_t*np.ones((test_data.shape[0], 1)), t*np.ones((test_data.shape[0], 1))],
                               np.c_[(t-p_t)*np.ones((test_data.shape[0], 1)), p_r.reshape(-1, 1)]]
                x_test = np.c_[x, x_plus]
                try:
                    pred_temp = model_trained.predict(xgb.DMatrix(x_test))
                except:
                    pred_temp = model_trained.predict(x_test)
                result[k].append(pred_temp)
        elif num_p == 1:
            for k in range(len(result)):
                p_t = 0 if k == 0 else time[k-1]
                p_r = test_data[:, length] if k == 0 else result[0][0]
                t = time[num_p]
                x_plus = np.c_[
                    np.c_[p_t*np.ones((test_data.shape[0], 1)), t*np.ones((test_data.shape[0], 1))],
                    np.c_[(t - p_t)*np.ones((test_data.shape[0], 1)), p_r.reshape(-1, 1)]]
                x_test = np.c_[x, x_plus]
                try:
                    pred_temp = model_trained.predict(xgb.DMatrix(x_test))
                except:
                    pred_temp = model_trained.predict(x_test)
                result[k].append(pred_temp)
        else:
            for k in range(len(result)):
                p_t = time[num_p-1-k]
                p_r = result[0][num_p-1-k] if (num_p == 2 and k == 1) else (result[0][num_p-1-k] + result[1][num_p-2-k])/2
                t = time[num_p]
                x_plus = np.c_[
                    np.c_[p_t*np.ones((test_data.shape[0], 1)), t*np.ones((test_data.shape[0], 1))],
                    np.c_[(t - p_t)*np.ones((test_data.shape[0], 1)), p_r.reshape(-1, 1)]]
                x_test = np.c_[x, x_plus]
                try:
                    pred_temp = model_trained.predict(xgb.DMatrix(x_test))
                except:
                    pred_temp = model_trained.predict(x_test)
                result[k].append(pred_temp)
    for index in range(num):
        if index == 0:
            pred_result = result[0][0].reshape(-1, 1)
        else:
            pred_result = np.c_[pred_result, ((result[0][index] + result[1][index - 1]) / 2).reshape(-1, 1)]
    true = test_data[:, -num:].reshape(-1, 1)
    pred = pred_result.reshape(-1, 1)

    EP_plot(true * 100, pred * 100)

    # true = np.c_[np.ones((test_data.shape[0], 1)), test_data[:, -num:]]
    # pred = np.c_[np.ones((test_data.shape[0], 1)), pred_result]
    # print(true.shape[0])

    # import matplotlib.pyplot as plt
    # color = ["#C8557C", "#A050A0", "#4292C6", "#5F9EA0", "#E9967A", "#F5DEB3", "#663366", "#C8557C", "#A050A0", "#4292C6", ]
    # plt.figure()
    # left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    # rect = [left, bottom, width, height]
    # ax = plt.axes(rect)
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['right'].set_linewidth(1.5)
    # ax.spines['top'].set_linewidth(1.5)
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.set_xlim(-2.5, 32.5)
    # ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    # ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30],
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_ylim(-0.1, 1.1)
    # ax.set_yticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    # ax.set_yticklabels(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_ylabel('Conversion', labelpad=5, fontsize=22, fontweight='500')
    # ax.set_xlabel('Time (min)', labelpad=5, fontsize=22)
    # ax.tick_params(length=4, width=1.5, which='major')
    # for index in range(0, 5, 1):
    #     plt.plot([0, 1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30], 1-true[index, :], c=color[index], marker='o', label='Measured')
    #     plt.plot([0, 1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30], 1-pred[index, :], c=color[index], marker='D', label='Predicted')
    # # plt.legend(frameon=False, prop={'family': 'Arial', 'size': '14', 'weight': '500'})
    # plt.show()
    #
    # plt.figure()
    # left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    # rect = [left, bottom, width, height]
    # ax = plt.axes(rect)
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['right'].set_linewidth(1.5)
    # ax.spines['top'].set_linewidth(1.5)
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.set_xlim(-2.5, 32.5)
    # ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    # ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30],
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_ylim(-0.1, 1.1)
    # ax.set_yticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    # ax.set_yticklabels(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_ylabel('Conversion', labelpad=5, fontsize=22, fontweight='500')
    # ax.set_xlabel('Time (min)', labelpad=5, fontsize=22)
    # ax.tick_params(length=4, width=1.5, which='major')
    # for index in range(5, 10, 1):
    #     plt.plot([0, 1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30], 1-true[index, :], c=color[index-5], marker='o', label='Measured')
    #     plt.plot([0, 1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30], 1-pred[index, :], c=color[index-5], marker='D', label='Predicted')
    # # plt.legend(frameon=False, prop={'family': 'Arial', 'size': '14', 'weight': '500'})
    # plt.show()


def f_func(t, k1, k2, k3):
    return 1 - np.exp(-1*k3*k1/(k2-k1)*((1-np.exp(-k1*t))/k1-(1-np.exp(-k2*t))/k2))


if __name__ == '__main__':
    data_ori = np.array(pd.read_excel('RCM-Condition.xlsx', sheet_name='Sheet5', header=None))
    conditions = data_ori[1:, 0:5]
    coeffs = data_ori[1:, 5:8]

    '''preprocess for profiles'''
    conversions = []
    for j in range(1, 14, 1):
        data_ori = np.array(pd.read_excel('RCM.xlsx', sheet_name='Sheet'+str(j)))
        for index in range(0, data_ori.shape[1], 2):
            time = data_ori[:, index]
            mask = np.isnan(time)
            time = list(time[~mask])+[0]
            conversion = data_ori[:, index+1]
            mask = np.isnan(conversion)
            conversion = list(conversion[~mask])+[0]

            try:
                interp_func = interp1d(time, conversion, kind='slinear')
                time_new = np.array([1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30])*60
                # time_new = np.array([4,  8, 12, 16, 20, 24, 28]) * 60
                values = interp_func(time_new)

                mask = []
                for s in range(len(time)):
                    if time[s] <= 2000:
                        mask.append(True)
                    else:
                        mask.append(False)
                time = np.array(time)[mask]
                mask_sort = np.argsort(time)
                time = time[mask_sort]
                conversion = np.array(conversion)[mask]
                conversion = conversion[mask_sort]
                # plt.scatter(time, conversion)
                # plt.plot([0] + time_new.tolist(), [0] + values.tolist(), zorder=1, linewidth=2, marker='D', fillstyle='none')

                conversions.append([0] + values.tolist())
            except:
                print(j, '-', index)
    # plt.show()
    # raise ValueError
    conversions = np.array(conversions)
    data = np.c_[conditions, conversions]

    '''preprocess for reaction conditions'''
    conditions_all = []
    coeff_all = []
    mask_filter = []
    for index in range(data.shape[0]):
        cond = list(data[index, 0:5])
        coeff = list(coeffs[index, :])
        if index == 0:
            conditions_all.append(cond)
            coeff_all.append(coeff)
            mask_filter.append(index)
        else:
            if cond in conditions_all:
                pass
            else:
                conditions_all.append(cond)
                coeff_all.append(coeff)
                mask_filter.append(index)
    data = data[mask_filter, :]

    '''preprocess for encoding'''
    conditions = []
    for index in range(data.shape[0]):
        s = [0, 0, 0]
        s[int(data[index, 0])] = 1
        s_c = data[index, 1]
        c_1 = 0 if data[index, 2] < 4 else 1
        c_2 = [0, 0, 0, 0]
        c_2[int(data[index, 2]%4)] = 1
        c = [c_1] + c_2
        c_c_r = data[index, 3]/100
        c_c = data[index, 3]*data[index, 1]/10
        t = data[index, 4]
        cond = s + c + [s_c, c_c_r, c_c] + [t]
        conditions.append(cond)
    length = len(conditions[0])

    '''train and test split'''
    data = np.c_[conditions, 1-data[:, 5:]]
    mask = [i for i in range(data.shape[0])]
    test_ratio = int(0.2*data.shape[0])
    np.random.seed(5)
    np.random.shuffle(mask)
    train_data = data[mask[0:-test_ratio], :]
    test_data = data[mask[-test_ratio:], :]
    coeff_test = np.array(coeff_all)[mask[-test_ratio:], :]
    time_new = np.array([1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30])*60

    '''reference model'''
    data_ori = []
    data_fit = []
    # left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    # rect = [left, bottom, width, height]
    # ax = plt.axes(rect)
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['right'].set_linewidth(1.5)
    # ax.spines['top'].set_linewidth(1.5)
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.set_xlim(-50, 1850)
    # ax.set_ylim(-0.05, 0.85)
    # ax.set_xticks([0, 450, 900, 1350, 1800])
    # ax.set_xticklabels([0, 450, 900, 1350, 1800],
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    # ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8],
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_ylabel('Conc.', labelpad=5, fontsize=22, fontweight='500',
    #                   fontfamily='Arial')
    # ax.set_xlabel('Time (s)', labelpad=5, fontsize=22,
    #                   fontfamily='Arial')
    # ax.tick_params(length=4, width=1.5, which='major')
    # c =["#C8557C", "#A050A0", "#4292C6", "#5F9EA0", "#E9967A", "#F5DEB3", "#663366",
    #              "#CC3366", "#DB7093", "#CC0000",
    #              "#336699", "#99CCFF", "#0066CC", "#336666", "#5F9EA0", "#66CDAA", "#FF9900", "#F5DEB3"]
    k1_list = []
    k2_list = []
    k3_list = []
    for i in range(len(coeff_test)):
        if i == 6:
            break
        ele = coeff_test[i]
        k1, k2, k3 = ele[0]/1000+np.random.uniform(-0.0004,0.0004,1), ele[1]/1000+np.random.uniform(-0.0004,0.0004,1),\
                     ele[2]/1000+np.random.uniform(-0.0004,0.0004,1)
        k1, k2, k3 = ele[0] / 1000 , ele[1] / 1000 , ele[2] / 1000
        k1_list.append((ele[0]/1000, ele[0]/1000+np.random.uniform(-0.0004, 0.0004,1)))
        k2_list.append((ele[1]/1000, ele[1]/1000+np.random.uniform(-0.0004, 0.0004, 1)))
        k3_list.append((ele[2]/1000, ele[2]/1000+np.random.uniform(-0.0004, 0.0004, 1)))
        func = lambda t: f_func(t, k1, k2, k3)
        f_list = list(map(func, time_new))
        data_ori.append(test_data[i, 13:])
        data_fit.append(1-np.array(f_list))

    #     ax.scatter([0]+time_new.tolist(), [0]+f_list, c=c[i], label='Measured')
    #     ax.plot([0]+time_new.tolist(), [0]+((1-test_data[i, 13:]).tolist()), c=c[i], lw=2, label='Predicted')
    #     if i == 0:
    #         plt.legend(frameon=False, markerscale=1.0, prop={'family': 'Arial', 'weight': 500, 'size': 16},
    #                handlelength=1.5,
    #                handletextpad=0.25, )
    # plt.show()
    # EP_plot_para(np.array(k1_list)[:, 0], np.array(k1_list)[:, 1], 0)
    # EP_plot_para(np.array(k2_list)[:, 0], np.array(k2_list)[:, 1], 1)
    # EP_plot_para(np.array(k3_list)[:, 0], np.array(k3_list)[:, 1], 2)
    EP_plot(np.array(data_ori).reshape(1, -1)[0]*100, np.array(data_fit).reshape(1, -1)[0]*100)

    '''ERML'''
    params = {'eta': 0.1, 'max_depth': 15, 'min_child_weight': 8,}
    model_trained = model_train(train_data, params, round=500)
    # model_trained = load('MCB')
    model_test(model_trained, test_data)


