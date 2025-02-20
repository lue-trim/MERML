import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import xgboost as xgb
import globalvaribale
from joblib import dump
import copy
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math
import scipy
import matplotlib as mpl
from preprocess import list_reduce
import globalvaribale
mpl.rcParams["mathtext.fontset"] = 'custom'
mpl.rcParams["mathtext.bf"] = "Arial:bold"
mpl.rcParams["mathtext.default"] = 'regular'


def huber(gap):
    sigma = globalvaribale.get_value("SIGMA")
    grad = np.sign(sigma) * sigma * gap / np.sqrt(sigma ** 2 + gap ** 2)
    hess = np.sign(sigma) * (sigma ** 3) / np.power(sigma ** 2 + gap ** 2, 1.5)
    return grad, hess


def huber_object_matrix(pred, data_matrix):
    gap = pred - data_matrix.get_label()
    grad, hess = huber(gap)
    return grad, hess


class MCB:
    def __init__(self, base_model, train, test, lag, shifts=[1, 2, 3],
                 train_time=[2, 5, 9, 15, 22, 30], test_time=[2, 5, 9, 15, 22, 30], flag=False, mask=[]):
        self.base_model = copy.copy(base_model)
        self.train_matrix = None
        self.model_trained = None
        self.train = train
        self.test = test
        self.lag = lag
        self.shifts = shifts
        self.train_time = train_time  # the sampling point time fo the kinetic profile in training set
        self.test_time = test_time  # the sampling point time fo the kinetic profile in training set
        self.flag = flag  # save the model or not
        self.mask = mask  # filter the feature whose index is not in mask

    @staticmethod
    def train_struct(train, time, lag, shifts):
        number_sample = train.shape[0]
        condition = train[:, 0:6]
        available_time = np.arange(0.5, 300, 0.5).tolist()
        index = [available_time.index(t) for t in time]
        curve = (train[:, 7:])[:, index]
        mask_plus = [0 for index in range(number_sample)]
        n_col = curve.shape[1]

        for shift in shifts:
            time_temp = time[shift - 1:]
            time_pre = np.c_[np.zeros((number_sample, shift + lag - 1)),
                             np.array([time[0:len(time) - shift]])[mask_plus]]
            curve_pre = np.c_[np.ones((number_sample, shift + lag - 1)), curve[:, 0:n_col - shift]]
            for index in range(len(time_temp)):
                time_value = time_temp[index]
                curve_temp = curve[:, index + shift - 1]
                time_pre_temp = time_pre[:, index + shift - 1:index + lag + shift - 1]
                time_gap_temp = time_value - time_pre_temp
                curve_pre_temp = curve_pre[:, index + shift - 1:index + lag + shift - 1]

                x_temp = np.c_[np.c_[np.c_[condition, time_pre_temp], time_gap_temp], curve_pre_temp]
                y_temp = curve_temp.reshape(-1, 1)
                x = x_temp if index == 0 else np.r_[x, x_temp]
                y = y_temp if index == 0 else np.r_[y, y_temp]
            x_data = x if shifts.index(shift) == 0 else np.r_[x_data, x]
            y_data = y if shifts.index(shift) == 0 else np.r_[y_data, y]

        shift_min = min(shifts)
        if shift_min == 1:
            pass
        else:
            time_temp = time[0:shift_min - 1]
            time_pre = np.zeros((number_sample, lag))
            curve_temp = curve[:, 0:shift_min - 1].reshape(number_sample, -1)
            curve_pre = np.ones((number_sample, lag))
            for index in range(len(time_temp)):
                time_gap_temp = time_temp[index] - time_pre
                x_temp = np.c_[np.c_[np.c_[condition, time_pre], time_gap_temp], curve_pre]
                y_temp = curve_temp[:, index].reshape(-1, 1)
                x = x_temp if index == 0 else np.r_[x, x_temp]
                y = y_temp if index == 0 else np.r_[y, y_temp]
            x_data = np.r_[x_data, x]
            y_data = np.r_[y_data, y]
        # np.savetxt('train.txt', np.c_[np.c_[x_data[:, 0], x_data[:, -3:]], y_data])
        return x_data, y_data

    def train_loop(self):
        x, y = self.train_struct(self.train, self.train_time, self.lag, self.shifts)
        if len(self.mask) != 0:
            x = x[:, self.mask]
        train_matrix = xgb.DMatrix(x, label=y)
        if isinstance(self.base_model, str):
            if self.base_model == 'XGBoost':
                params = copy.copy(globalvaribale.get_value('params'))
                num_boost_round = params['num_boost_round']
                del params['num_boost_round']
                model_train = xgb.train(params, dtrain=train_matrix,
                                        num_boost_round=num_boost_round) #obj=huber_object_matrix,
                y_pred = model_train.predict(train_matrix)
                # EP_plot(y*100, y_pred*100)
            else:
                raise ValueError
        else:
            model_train = self.base_model.fit(x, y.ravel())
            y_pred = model_train.predict(x)
        self.model_trained = model_train
        if self.flag:
            dump(model_train, 'Result/MCB')
        return y, y_pred

    def inference(self, data):
        y_train_true, y_train_pred = self.train_loop()
        model_trained = self.model_trained
        y_test_true, y_test_pred, x_data, preds = self.inference_method(model_trained, data, self.lag, self.test_time,
                                                                        self.shifts, self.mask)
        return y_test_true, y_test_pred, x_data, preds, y_train_true, y_train_pred

    @staticmethod
    def inference_method(model, data, lag, test_time, shifts, mask):
        number_sample = data.shape[0]
        condition = data[:, 0:6]

        available_time = np.arange(0.5, 300, 0.5).tolist()
        index = [available_time.index(t) for t in test_time]
        curve = copy.copy(data[:, 7:])[:, index]
        curve_ori = copy.copy(data[:, 7:])[:, index]
        mask_plus = [0 for index in range(number_sample)]

        pred = [[] for index in range(len(shifts) + 1)]
        x_data = [[] for index in range(len(shifts) + 1)]
        shift_min = min(shifts)
        for index in range(len(test_time)):
            if index <= shift_min - 1:
                time_value = (test_time[0:shift_min])[index]
                time_pre = np.zeros((number_sample, lag))
                curve_pre = np.ones((number_sample, lag))
                time_gap = time_value - time_pre
                x_temp = np.c_[np.c_[np.c_[condition, time_pre], time_gap], curve_pre]
                if len(mask) != 0:
                    x_temp = x_temp[:, mask]
                y_temp = curve[:, index].reshape(number_sample, -1)
                try:
                    pred_temp = model.predict(x_temp)
                except:
                    x_matrix = xgb.DMatrix(x_temp, y_temp)
                    pred_temp = model.predict(x_matrix)
                x_data[-1].append(x_temp)
                pred[-1].append(pred_temp)
                curve[:, index] = pred_temp
            else:
                time_value = test_time[index]
                y_temp = curve_ori[:, index].reshape(-1, 1)
                flag = False
                count = 0
                for shift in shifts:
                    time_pre_ele = [
                        test_time[index - shift - lag + index_t + 1] if index - shift - lag + index_t + 1 >= 0
                        else 0 for index_t in range(lag)]
                    if sum(time_pre_ele) == 0:
                        flag = True
                    time_gap_ele = time_value - np.array(time_pre_ele)
                    time_pre = np.array([time_pre_ele])[mask_plus]
                    time_gap = np.array([time_gap_ele])[mask_plus]
                    curve_pre = (np.c_[np.ones((number_sample, shift + lag - 1)),
                                       curve[:, 0:curve_ori.shape[1] - shift]])[:, index:index + lag]
                    x_temp = np.c_[np.c_[np.c_[condition, time_pre], time_gap], curve_pre]
                    if len(mask) != 0:
                        x_temp = x_temp[:, mask]
                    try:
                        y_pred = model.predict(x_temp)
                    except:
                        x_matrix = xgb.DMatrix(x_temp, label=y_temp)
                        y_pred = model.predict(x_matrix)
                    if shifts.index(shift) == 0:
                        pred_temp = copy.copy(y_pred)
                    else:
                        pred_temp += y_pred
                    count += 1
                    x_data[shifts.index(shift)].append(x_temp)
                    pred[shifts.index(shift)].append(y_pred)
                    if flag:
                        break
                curve[:, index] = pred_temp / count
        return curve_ori.reshape(-1, 1), curve.reshape(-1, 1), x_data, pred

    def evaluate(self):
        test_result = self.inference(self.test)
        return r2_score(test_result[0], test_result[1]), mean_absolute_error(test_result[0], test_result[1]), \
               np.sqrt(mean_squared_error(test_result[0], test_result[1]))


class MT:
    def __init__(self, base_model, train, test, train_time=[2, 5, 9, 15, 22, 30], test_time=[2, 5, 9, 15, 22, 30]):
        self.base_model = copy.copy(base_model)
        # print(base_model.get_params())
        self.train_matrix = None
        self.train = train
        self.test = test
        self.train_time = train_time
        self.test_time = test_time

    @staticmethod
    def data_struct(data, data_time):
        all_time = np.arange(0.5, 300, 0.5).tolist()
        index = [all_time.index(t) for t in data_time]
        time = np.array(data_time).reshape(-1, 1)
        curve = data[:, 7:]
        conversions = curve[:, index].reshape(-1, 1)

        number_sample = data.shape[0]
        index_time = [index % len(time) for index in range(number_sample * len(time))]
        index_sample = [index // len(time) for index in range(number_sample * len(time))]
        data_input = data[index_sample, 0:6]
        data_time = time[index_time, :]
        x = np.c_[data_input, data_time]
        y = conversions
        return x, y

    def train_loop(self):
        x, y = MT.data_struct(self.train, self.train_time)
        if isinstance(self.base_model, str):
            if self.base_model == 'XGBoost':
                train_matrix = xgb.DMatrix(x, label=y)
                params = copy.copy(globalvaribale.get_value('params'))
                num_boost_round = params['num_boost_round']
                del params['num_boost_round']
                model_train = xgb.train(params, dtrain=train_matrix,
                                        num_boost_round=num_boost_round) #obj=huber_object_matrix,
                y_pred = model_train.predict(train_matrix)
            else:
                raise ValueError
        else:
            model_train = self.base_model.fit(x, y.ravel())
            y_pred = model_train.predict(x)
        self.model_trained = model_train
        return y, y_pred

    def inference(self, data):
        y_train_true, y_train_pred = self.train_loop()
        y_test_true, y_test_pred, x = self.inference_method(self.model_trained, data, self.test_time)
        return y_test_true, y_test_pred, x, y_train_true, y_train_pred

    @staticmethod
    def inference_method(model, data, test_time):
        x, y = MT.data_struct(data, test_time)
        try:
            pred = model.predict(x)
        except:
            data_matrix = xgb.DMatrix(x, label=y)
            pred = model.predict(data_matrix)
        return y.reshape(-1, 1), pred.reshape(-1, 1), x

    def evaluate(self):
        test_result = self.inference(self.test)
        return r2_score(test_result[0], test_result[1]), mean_absolute_error(test_result[0], test_result[1]), \
               np.sqrt(mean_squared_error(test_result[0], test_result[1]))


def train_test_split(data, ratio, seed):
    number_sample = data.shape[0]
    number_test = int(number_sample*ratio)
    index_sample = [index for index in range(number_sample)]
    np.random.seed(seed)
    np.random.shuffle(index_sample)
    index_test = index_sample[0:number_test]
    index_train = index_sample[number_test:]
    return index_train, index_test


def h_t(x, h0, kw=-14):
    '''calculated the concentration of H+ according to IO3-'''
    oh = kw - h0
    delta_io3 = x - math.pow(10, io3)
    b_h = math.pow(10, oh) - math.pow(10, h) - 6 * delta_io3
    a_h = 1
    c_h = -1 * math.pow(10, kw)
    delta_h = b_h ** 2 - 4 * a_h * c_h
    return (-1 * b_h + np.sqrt(delta_h)) / 2 * a_h


def result_t(h, io3, i):
    '''kinetic model'''
    ht = lambda x: h_t(x, h)
    it = lambda x: i_t(x, i)
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


def i_t(x, i0):
    '''claculate the concentration of I- according to IO3-'''
    a_i1 = kT
    delta_io3 = x - math.pow(10, io3)
    aplha_1 = 5 * delta_io3 + math.pow(10, i0)
    beta_1 = -3 * delta_io3
    b_i1 = -1 * (kT * (aplha_1 + beta_1) + 1)
    c_i1 = kT * aplha_1 * beta_1
    delta_i1 = b_i1 ** 2 - 4 * a_i1 * c_i1
    return -1 * (-1 * b_i1 - np.sqrt(delta_i1)) / 2 / a_i1 + 5 * delta_io3 + math.pow(10, i0)


def reverse_profile(profiles_attr, io3_0):
    '''transform profiles from M to mM'''
    profiles = copy.copy(profiles_attr)
    for i in range(profiles.shape[0]):
        profiles[i, :] = profiles[i, :]*math.pow(10, io3_0[i]+3)
    return profiles


def k_fold_mask(number_sample, k_fold=5, seed=0):
    mask = [index for index in range(number_sample)]
    np.random.seed(seed)
    np.random.shuffle(mask)
    mask_list = []
    gap = number_sample // k_fold
    for index in range(k_fold):
        mask_temp = mask[index*gap:(index+1)*gap]
        mask_list.append(mask_temp)
    return mask_list


if __name__ == '__main__':
    '''set the value for some variable'''
    kw = -14
    T = 25 + 273.15
    kT = math.pow(10, 555 / T + 7.355 - 2.575 * math.log10(T))
    
    k_1 = 9.4 * math.pow(10, 7)
    k_2 = 4.6 * math.pow(10, -11)
    k_3 = 9.5
    
    
    profiles_exp = pd.read_excel('Dushman reaction.xlsx', sheet_name='exp')
    profiles_exp = np.array(profiles_exp)[:, 1:].T
    condition_exp = pd.read_excel('Dushman reaction.xlsx', sheet_name='cond')
    condition_exp = np.array(condition_exp)[:, 1:]

    '''generate simulated data'''
    condition_simulate = []
    h_list = np.arange(-5, -1.9, 0.2).tolist()
    i_list = np.arange(-4, -0.9, 0.2).tolist()
    io3_list = np.arange(-5, -0.9, 0.2).tolist()
    
    count = 0
    io3_simulate = []
    i_simulate = []
    i3_simulate = []
    for h in h_list:
        for i in i_list:
            for io3 in io3_list:
                count += 1
                print(count, end=': ')
                io3_result, i3_result, i_result, h_result = result_t(h, io3, i)
                if io3_result is None:
                    print('Pass-start')
                    continue
                else:
                    condition_simulate.append([h, io3, i, h - i, h - io3, io3 - i])
                    i3_simulate.append(i3_result)
                    io3_simulate.append(io3_result)
                    i_simulate.append(i_result)
                    print([h, i, io3, i - h, io3 - h, io3 - i], ': ', i3_result)
    
    np.save('condition_simulate-used.npy', condition_simulate)
    np.save('i3_simulate-used.npy', i3_simulate)
    np.save('io3_simulate-used.npy', io3_simulate)
    np.save('i_simulate-used.npy', i_simulate)
    print(len(condition_simulate))
    #raise ValueError

    '''compare the profiles from experiment and simulation'''
    color = ["#C8557C", "#A050A0", "#4292C6", "#5F9EA0", "#E9967A", "#F5DEB3", "#663366",]
    for index in range(0, condition_exp.shape[0], 1):
        condition = condition_exp[index, :]
        h, io3, i = np.log10(condition[0])-3, np.log10(condition[1])-3, np.log10(condition[2])-3
        # print(h ,io3, i)
        io3_result, i3_result, i_result, h_result = result_t(h, io3, i)
        # print(io3_result)
        if index == 0:
            plt.figure()
            left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
            rect = [left, bottom, width, height]
            ax = plt.axes(rect)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            ax.spines['top'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
        ax.plot(np.arange(0, 300.1, 0.5), np.array(i3_result)*1e6, c=color[index%6], lw=2, label='Run-'+str(index%6+1))
        ax.plot(np.arange(0, 300.1, 0.5), [0] + profiles_exp[index, :].tolist(), linestyle='--', c=color[index%6], lw=2)
        if (index+1)%6 == 0:
            # plt.title(str(condition[0])+'-'+str(condition[1])+'-'+str(condition[2]))
            ax.set_xlim(-5, 305)
            ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
            ax.set_xticklabels([0, 50, 100, 150, 200, 250, 300],
                               fontfamily='Arial', fontsize=18, fontweight='500')
            ax.set_ylim(-1, 26)
            ax.set_yticks(np.array([0, 5, 10, 15, 20, 25]))
            ax.set_yticklabels(np.array([0, 5, 10, 15, 20, 25]),
                               fontfamily='Arial', fontsize=18, fontweight='500')
            ax.set_ylabel('Conc. (' + r'$\mu$' + 'M)', labelpad=5, fontsize=22, fontweight='500')
            ax.set_xlabel('Time (s)', labelpad=5, fontsize=22)
            ax.tick_params(length=4, width=1.5, which='major')
            ax.plot([7.5, 22.5], [19.5, 19.5], c='black')
            ax.plot([125, 140], [19.5, 19.5], c='black', linestyle='--')
            ax.text(x=32.5, y=19, s='Experiment', fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'})
            ax.text(x=150, y=19, s='Simulation',
                    fontdict={'fontfamily': 'Arial', 'fontsize': '18', 'fontweight': '500'})
            ax.legend(ncol=3, loc='upper left', markerscale=0.5, frameon=False, numpoints=0.25, handlelength=1,
                       prop={'family': 'Arial', 'size': '16', 'weight': '500'})
    
            plt.savefig('Figures/S9-'+str((index+1)//6)+'.svg', dpi=600)
            plt.figure()
            left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
            rect = [left, bottom, width, height]
            ax = plt.axes(rect)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)
            ax.spines['top'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
    # raise ValueError


    '''Figure s15'''
    plt.figure()
    left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    rect = [left, bottom, width, height]
    ax = plt.axes(rect)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.hist(condition_simulate[:, 0], width=0.225, color="#87CEFA", align="mid",
            linewidth=1, edgecolor='black', alpha=0.6)
    ax.set_xlim(-5.05, -2)
    ax.set_xticks([-5, -4, -3, -2])
    ax.set_xticklabels([-5, -4, -3, -2],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylim(0, 200)
    ax.set_yticks(np.array([0, 50, 100, 150, 200]))
    ax.set_yticklabels(np.array([0, 50, 100, 150, 200]),
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('Frequency', labelpad=5, fontsize=22, fontweight='500')
    ax.set_xlabel(r'$Log_{10}$' + r'$([H^+])$', labelpad=5, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')
    plt.show()
    plt.savefig('Figures/S15-1.svg', dpi=600)

    ax.hist(condition_simulate[:, 1], width=0.225, color="#87CEFA", align="mid",
            linewidth=1, edgecolor='black', alpha=0.6)
    ax.set_xlim(-5.05, -1)
    ax.set_xticks([-5, -4, -3, -2, -1])
    ax.set_xticklabels([-5, -4, -3, -2, -1],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylim(0, 250)
    ax.set_yticks(np.array([0, 50, 100, 150, 200, 250]))
    ax.set_yticklabels(np.array([0, 50, 100, 150, 200, 250]),
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('Frequency', labelpad=5, fontsize=22, fontweight='500')
    ax.set_xlabel(r'$Log_{10}$' + r'$([IO_3^-])$', labelpad=5, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')
    plt.savefig('Figures/S10-2.svg', dpi=600)
    
    ax.hist(condition_simulate[:, 2], width=0.225, color="#87CEFA",  align="mid",
              linewidth=1, edgecolor='black', alpha=0.6)
    ax.set_xlim(-4, -0.85)
    ax.set_xticks([-4, -3, -2, -1])
    ax.set_xticklabels([-4, -3, -2, -1],
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylim(0, 200)
    ax.set_yticks(np.array([0, 50, 100, 150, 200]))
    ax.set_yticklabels(np.array([0, 50, 100, 150, 200]),
                       fontfamily='Arial', fontsize=18, fontweight='500')
    ax.set_ylabel('Frequency', labelpad=5, fontsize=22, fontweight='500')
    ax.set_xlabel(r'$Log_{10}$'+r'$([I^-])$', labelpad=5, fontsize=22)
    ax.tick_params(length=4, width=1.5, which='major')
    # plt.show()
    plt.savefig('Figures/S15-2.svg', dpi=600)


    '''preprocess the data'''
    # condition = []
    # for index in range(condition_exp.shape[0]):
    #     h = np.log(condition_exp[index, 0]) -3
    #     io3 = np.log(condition_exp[index, 1]) -3
    #     i = np.log(condition_exp[index, 2]) -3
    #     condition.append([h, io3, i, h-i, h-io3, io3-i])
    # condition_exp = np.array(condition)
    #
    # condition_simulate = np.load('condition_simulate-used.npy', allow_pickle=True)
    # profiles_simulate = np.load('i_simulate-used.npy', allow_pickle=True)
    # i3_profiles = np.load('i3_simulate-used.npy', allow_pickle=True)
    # io3_profiles = np.load('io3_simulate-used.npy', allow_pickle=True)
    #
    # mask_filter = []
    # for index in range(condition_simulate.shape[0]):
    #     profile = io3_profiles[index, :]
    #     if 1e-6 < max(i3_profiles[index, :]) < 40*1e-6:
    #         if min(profile)/math.pow(10, condition_simulate[index, 1]) < 0.99 \
    #                 and profile[1]/math.pow(10, condition_simulate[index, 1]) > 0.01:
    #             mask_filter.append(True)
    #         else:
    #             mask_filter.append(False)
    #     else:
    #         mask_filter.append(False)
    #
    # condition_simulate = condition_simulate[mask_filter, :]
    # profiles_simulate = profiles_simulate[mask_filter, :]
    # print(condition_simulate.shape)
    #
    # for i in range(profiles_simulate.shape[0]):
    #     for j in range(profiles_simulate.shape[1]):
    #         if profiles_simulate[i, j] < 0:
    #             profiles_simulate[i, j] = 0
    #
    # for index in range(condition_simulate.shape[0]):
    #     profiles_simulate[index, :] = profiles_simulate[index, :]/math.pow(10, condition_simulate[index, 2])

    # for index in range(871):
    #     profile = profiles_simulate[index, 0:10]
    #     plt.plot(np.arange(0, profile.shape[0], 1), profile.tolist())
    # plt.show()


    '''model compare'''
    # globalvaribale.init()
    # data_simulate = np.c_[condition_simulate[:, 0:6], profiles_simulate]
    #
    # mask_cv = k_fold_mask(data_simulate.shape[0], 5, seed=0)
    # mask_all = [index for index in range(data_simulate.shape[0])]
    # metrics = []
    # infos = []
    # for num in range(2, 6, 1):
    #     for depth in range(6, 15, 2):
    #         for child in range(1, 10, 1):
    #             params = {'eta': 0.1, 'max_depth': depth, 'min_child_weight': child,
    #                               'num_boost_round': 100 * num}
    #             globalvaribale.set_value('params', params)
    #             metric_a = []
    #             metric_b = []
    #             metric_c = []
    #             for mask_test in mask_cv:
    #                 train = data_simulate[list(list_reduce(mask_all, mask_test)), :]
    #                 test = data_simulate[mask_test, :]
    #                 time = np.arange(0, 300, 30)[1:]
    #                 mcb = MCB('XGBoost', train, test, 1, np.arange(1, 2+1, 1).tolist(), time, time, True)
    #                 y_test_true, y_test_pred, x_data, preds, y_train_true, y_train_pred = mcb.inference(test)
    #                 a = mean_absolute_error(y_test_true, y_test_pred)
    #
    #                 # mcb = MCB('XGBoost', train, test, 1, np.arange(1, 1 + 1, 1).tolist(), time, time, True)
    #                 # y_test_true, y_test_pred, x_data, preds, y_train_true, y_train_pred = mcb.inference(test)
    #                 # b = mean_absolute_error(y_test_true, y_test_pred)
    #                 #
    #                 # mt = MT('XGBoost', train, test, time, time)
    #                 # y_test_true, y_test_pred, x, y_train_true, y_train_pred = mt.inference(test)
    #                 # c = mean_absolute_error(y_test_true, y_test_pred)
    #
    #                 metric_a.append(a)
    #                 # metric_b.append(b)
    #             #     metric_c.append(c)
    #             #
    #             # print(str(num), '-', str(depth), '-', str(child), ':',
    #             #               np.mean(metric_a), '-', np.mean(metric_b), '-', np.mean(metric_c))
    #             # metrics.append(metric_a + metric_b + metric_c)
    #             print(str(num), '-', str(depth), '-', str(child), ':',
    #                           np.mean(metric_a))
    #             metrics.append(metric_a)
    #             infos.append([num, depth, child])
    #
    # np.save('metrics_dushman_opt_hp-i-1-2-30.npy', metrics)
    # np.save('infos_dushman_opt_hp-i-1-2-30.npy', infos)
    # raise ValueError

    # data = np.load('metrics_dushman_opt_hp-30-i.npy', allow_pickle=True)
    # infos = np.load('infos_dushman_opt_hp-30-i.npy', allow_pickle=True)
    # metric_erml = data[:, 0:5]
    # metric_rml = data[:, 5:10]
    # metric_ml = data[:, 10:15]
    #
    # average_erml = np.mean(metric_erml, axis=1)
    # mask_erml = np.argsort(average_erml)
    # print(average_erml[mask_erml])
    # infos_erml = infos[mask_erml]
    # print(infos_erml)
    #
    # average_rml = np.mean(metric_rml, axis=1)
    # mask_rml = np.argsort(average_rml)
    # print(average_rml[mask_rml])
    # infos_rml = infos[mask_rml]
    # print(infos_rml)
    #
    # average_ml = np.mean(metric_ml, axis=1)
    # mask_ml = np.argsort(average_ml)
    # print(average_ml[mask_ml])
    # infos_ml = infos[mask_ml]
    # print(infos_ml)
    # raise ValueError


    '''model training and evaluation'''
    # from plot import EP_plot
    # globalvaribale.init()
    # data_simulate = np.c_[condition_simulate[:, 0:6], profiles_simulate]
    # mask_train, mask_test = train_test_split(data_simulate, 0.2, 1)
    # train, test = data_simulate[mask_train, :], data_simulate[mask_test, :]
    # print(test.shape)
    # print(test)
    # np.savetxt('Dushman-i-train.txt', train)
    # np.savetxt('Dushman-i-test.txt', test)

    # time = np.arange(0, 50.5, 5)[1:]
    # # time = np.arange(0, 90, 0.5)[1:]
    #
    # params = {'eta': 0.1, 'max_depth': 8, 'min_child_weight': 11,
    #               'num_boost_round': 100 * 5}
    # globalvaribale.set_value('params', params)
    # model = MCB('XGBoost', train, test, 1, [1, 2], time, time, True)
    # y_test_true, y_test_pred, x_data, preds, y_train_true, y_train_pred = model.inference(test)
    #
    # profiles_true = y_test_true.T.reshape(test.shape[0], -1)
    # profiles_pred = y_test_pred.T.reshape(test.shape[0], -1)
    # #
    # profiles_pred_r = reverse_profile(profiles_pred, test[:, 1])
    # profiles_true_r = reverse_profile(profiles_true, test[:, 1])
    #
    # true = np.array(profiles_true_r).reshape(1, -1)[0]
    # pred = np.array(profiles_pred_r).reshape(1, -1)[0]
    # EP_plot(profiles_true[:, -10:].reshape(1, -1)[0]*100, profiles_pred[:, -10:].reshape(1, -1)[0]*100)
    # print(mean_absolute_error(profiles_true[:, 0:10].reshape(1, -1)[0], profiles_pred[:, 0:10].reshape(1, -1)[0]))

    # params = {'eta': 0.1, 'max_depth': 10, 'min_child_weight': 2,
    #           'num_boost_round': 100 * 2}
    # globalvaribale.set_value('params', params)
    # mt = MT('XGBoost', train, test, time, time)
    # y_test_true, y_test_pred, x, y_train_true, y_train_pred = mt.inference(test)
    # profiles_true = y_test_true.T.reshape(test.shape[0], -1)
    # profiles_pred = y_test_pred.T.reshape(test.shape[0], -1)
    #
    # EP_plot(profiles_true[:, -10:].reshape(1, -1)[0]*100, profiles_pred[:, -10:].reshape(1, -1)[0]*100)
    #
    # profiles_pred_r = reverse_profile(profiles_pred, test[:, 1])
    # profiles_true_r = reverse_profile(profiles_true, test[:, 1])
    #
    # true = np.array(profiles_true_r).reshape(1, -1)[0]
    # pred = np.array(profiles_pred_r).reshape(1, -1)[0]
    # print(mean_absolute_error(y_test_true, y_test_pred))

    # maes_profile = []
    # maes_profile_r = []
    # for index in range(profiles_true.shape[0]):
    #     mae_temp = mean_absolute_error(profiles_true[index, :], profiles_pred[index, :])
    #     maes_profile.append(mae_temp)
    #     mae_temp_r = mean_absolute_error(profiles_true_r[index, :], profiles_pred_r[index, :])
    #     maes_profile_r.append(mae_temp_r)
    # print(len(maes_profile_r))
    # print(len([mae for mae in maes_profile_r if mae > 10*1e-3]))
    # raise ValueError


    '''Fig S17-a and 17-d'''
    # ax.set_xlim(-1, 11)
    # ax.set_ylim(-1, 11)
    # ax.set_xticks([0, 2, 4, 6, 8, 10])
    # ax.set_xticklabels([0, 2, 4, 6, 8, 10],
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_yticks([0, 2, 4, 6, 8, 10])
    # ax.set_yticklabels([0, 2, 4, 6, 8, 10],
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # plt.plot([-1, 11], [-1, 11], c='#C26275', linestyle='dashed', zorder=0)
    # ax.text(x=6.5, y=1.5, s=r'$R^2$' + ' = ' + str(np.round(r2_score(true, pred), 3)),
    #         fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    # ax.text(x=6.5, y=0.5, s='MAE = ' + str(np.round(mean_absolute_error(true, pred), 3)),
    #         fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    # ax.text(x=6.5, y=-0.5, s='RMSE = ' + str(np.round(np.sqrt(mean_squared_error(true, pred)), 3)),
    #         fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    # plt.show()

    '''figure S18-a'''
    # ax.set_xlim(-0.5, 5.5)
    # ax.set_ylim(-0.5, 5.5)
    # ax.set_xticks([0, 1, 2, 3, 4, 5])
    # ax.set_xticklabels([0, 1, 2, 3, 4, 5],
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_yticks([0, 1, 2, 3, 4, 5])
    # ax.set_yticklabels([0, 1, 2, 3, 4, 5],
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # plt.plot([-0.5, 5.5], [-0.5, 5.5], c='#C26275', linestyle='dashed', zorder=0)
    # ax.text(x=6.5/2, y=1.5/2, s=r'$R^2$' + ' = ' + str(np.round(r2_score(true, pred), 3)),
    #         fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    # ax.text(x=6.5/2, y=0.5/2, s='MAE = ' + str(np.round(mean_absolute_error(true, pred), 3)),
    #         fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    # ax.text(x=6.5/2, y=-0.5/2, s='RMSE = ' + str(np.round(np.sqrt(mean_squared_error(true, pred)), 3)),
    #         fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})

    '''figure s18-d'''
    # ax.set_xlim(-0.25, 2.75)
    # ax.set_ylim(-0.25, 2.75)
    # ax.set_xticks([0, 0.5, 1.0, 1.5, 2, 2.5])
    # ax.set_xticklabels([0, 0.5, 1.0, 1.5, 2, 2.5],
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_yticks([0, 0.5, 1.0, 1.5, 2, 2.5])
    # ax.set_yticklabels([0, 0.5, 1.0, 1.5, 2, 2.5],
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # plt.plot([-0.25, 2.75], [-0.25, 2.75], c='#C26275', linestyle='dashed', zorder=0)
    # ax.text(x=6.5 / 4, y=1.5 / 4, s=r'$R^2$' + ' = ' + str(np.round(r2_score(true, pred), 3)),
    #         fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    # ax.text(x=6.5 / 4, y=0.5 / 4, s='MAE = ' + str(np.round(mean_absolute_error(true, pred), 3)),
    #         fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    # ax.text(x=6.5 / 4, y=-0.5 / 4, s='RMSE = ' + str(np.round(np.sqrt(mean_squared_error(true, pred)), 3)),
    #         fontdict={'family': 'Arial', 'size': '18', 'weight': '500'})
    # plt.show()

    '''Fig S17-b, S17-e, S18-b and S18-e'''
    # import seaborn as sns
    # left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    # rect = [left, bottom, width, height]
    # ax = plt.axes(rect)
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['right'].set_linewidth(1.5)
    # ax.spines['top'].set_linewidth(1.5)
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.set_ylabel('Density', labelpad=10, fontsize=22, fontweight='500')
    # ax.set_xlabel('Error of Conc.' + r'$(\mu$'+'M)', labelpad=10, fontsize=22)
    # ax.tick_params(length=4, width=1.5, which='major')
    # from scipy.stats import norm
    # sns.distplot(np.array(maes_profile_r)*1e3, bins=50, hist=True,
    #                  kde=True, ax=ax,
    #                  kde_kws={'linestyle': '-', 'linewidth': '0', 'color': "#4292C6"},  # '#A050A0' '#D2D2FF'
    #                  hist_kws={'width': 0.8, 'align': 'mid', 'color': '#BBE6FF', "edgecolor": '#000000',
    #                            'linewidth': '1.0'},
    #                  fit=None)

    # ax.set_xlim(-5, 300)
    # ax.set_ylim(0, 0.2)
    # ax.set_xticks([0, 100, 200, 300])
    # ax.set_xticklabels([0, 100, 200, 300],
    #                        fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
    # ax.set_yticklabels(['0.00', '0.05', '0.10', '0.15', '0.20'],
    #                        fontfamily='Arial', fontsize=18, fontweight='500')
    # plt.show()

    # ax.set_xlim(-2.5, 62.5)
    # ax.set_ylim(0, 0.7)
    # ax.set_xticks([0, 15, 30, 45, 60])
    # ax.set_xticklabels([0, 15, 30, 45, 60],
    #                        fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_yticks([0, 0.2, 0.4, 0.6])
    # ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6'],
    #                        fontfamily='Arial', fontsize=18, fontweight='500')

    # ax.set_xlim(-50, 1550)
    # ax.set_ylim(0, 0.035)
    # ax.set_xticks([0, 375, 750, 1125, 1500])
    # ax.set_xticklabels([0, 375, 750, 1125, 1500],
    #                        fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_yticks([0, 0.010, 0.020, 0.030])
    # ax.set_yticklabels(['0.000', '0.010', '0.020', '0.030'],
    #                        fontfamily='Arial', fontsize=18, fontweight='500')
    # plt.show()

    # ax.set_xlim(-5, 205)
    # ax.set_ylim(0, 0.2)
    # ax.set_xticks([0, 50, 100, 150, 200])
    # ax.set_xticklabels([0, 50, 100, 150, 200],
    #                        fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20])
    # ax.set_yticklabels(['0.00', '0.05', '0.10', '0.15', '0.20'],
    #                        fontfamily='Arial', fontsize=18, fontweight='500')
    # plt.show()


    '''Fig S17c, S17f, S18c, and S18f'''
    # from plot import range_brace
    # plt.figure()
    # left, bottom, width, height = 0.18, 0.18, 0.78, 0.78
    # rect = [left, bottom, width, height]
    # ax = plt.axes(rect)
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['right'].set_linewidth(1.5)
    # ax.spines['top'].set_linewidth(1.5)
    # ax.spines['bottom'].set_linewidth(1.5)
    #
    # ax.set_ylim(-0.1, 1.1)
    # ax.set_yticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    # ax.set_yticklabels(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    #                        fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.set_ylabel(r'$C/C_0$', labelpad=5, fontsize=22, fontweight='500')
    # ax.set_xlabel('Time (s)', labelpad=5, fontsize=22)
    # ax.tick_params(length=4, width=1.5, which='major')
    # mask = np.argsort(maes_profile)
    # c = ['#9DC3E6', '#A9D18E', '#F4B183']

    # mask_selected = [9, 81, 168] # S12-1
    # mask_selected = [8, 77, 156] # S 14-1
    # for index in range(len(mask_selected)):
    #     print(math.pow(10, test[mask_selected[index], 0]+3), math.pow(10, test[mask_selected[index], 1]+3),
    #           math.pow(10, test[mask_selected[index], 2]+3))
    #     print(maes_profile_r[mask[mask_selected[index]]]*1e3)
    #     ax.plot(np.arange(0, 5.5, 0.5),
    #                 [1] + profiles_true[mask_selected[index], :].tolist(),
    #                  marker='o', c=c[index], label='Simulated', markersize=8,)
    #     ax.plot(np.arange(0, 5.5, 0.5),
    #                     [1] + profiles_pred[mask_selected[index], :].tolist(),
    #                  marker='D', c=c[index], label='Predicted', markersize=8)
    # plt.legend(loc=(0.025, 0.01), frameon=False, prop={'family': 'Arial', 'size': '14', 'weight': '500'})
    # ax.set_xlim(-0.25, 5.25)
    # ax.set_xticks([0, 1, 2, 3, 4, 5])
    # ax.set_xticklabels([0, 1, 2, 3, 4, 5],
    #                    fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.plot(range_brace(0, 3.5, height=0.05)[1] * 1 + 1.6, range_brace(0, 4)[0] / 30 + 0.425-0.115, color='black', lw=1.0,
    #             clip_on=False)
    # ax.plot(range_brace(0, 3.5, height=0.05)[1] * 1 + 1.6, range_brace(0, 4)[0] / 30 + 0.25-0.115, color='black', lw=1.0,
    #             clip_on=False)
    # ax.plot(range_brace(0, 3.5, height=0.05)[1] * 1 + 1.6, range_brace(0, 4)[0] / 30 + 0.07-0.115, color='black', lw=1.0,
    #             clip_on=False)
    # ax.text(1.7, 0.35+0.025, s='low error', fontdict={'fontfamily': 'Arial', 'fontsize': '14', 'fontweight': '500'},
    #             ha='left', va='center')
    # ax.text(1.7, 0.175+0.025, s='medium error', fontdict={'fontfamily': 'Arial', 'fontsize': '14', 'fontweight': '500'},
    #             ha='left', va='center')
    # ax.text(1.7, 0.025, s='high error', fontdict={'fontfamily': 'Arial', 'fontsize': '14', 'fontweight': '500'},
    #             ha='left', va='center')

    # mask_selected = [13, 64, 172] # S12-2
    # mask_selected = [8, 73, 171] # S14-2
    # for index in range(len(mask_selected)):
    #     print(math.pow(10, test[mask_selected[index], 0]+3), math.pow(10, test[mask_selected[index], 1]+3),
    #           math.pow(10, test[mask_selected[index], 2]+3))
    #     print(maes_profile_r[mask[mask_selected[index]]]*1e3)
    #     ax.plot(np.arange(0, 300, 30),
    #                 [1] + profiles_true[mask_selected[index], :].tolist(),
    #                  marker='o', c=c[index], label='Simulated', markersize=8,)
    #     ax.plot(np.arange(0, 300, 30),
    #                     [1] + profiles_pred[mask_selected[index], :].tolist(),
    #                  marker='D', c=c[index], label='Predicted', markersize=8)
    # plt.legend(loc=(0.025, 0.01), frameon=False, prop={'family': 'Arial', 'size': '14', 'weight': '500'})
    # ax.set_xlim(-15, 300)
    # ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    # ax.set_xticklabels([0, 50, 100, 150, 200, 250, 300],
    #                        fontfamily='Arial', fontsize=18, fontweight='500')
    # ax.plot(range_brace(0, 3.5, height=0.05)[1] * 30 + 90, range_brace(0, 4)[0] / 30 + 0.425-0.115, color='black', lw=1.0,
    #             clip_on=False)
    # ax.plot(range_brace(0, 3.5, height=0.05)[1] * 30 + 90, range_brace(0, 4)[0] / 30 + 0.25-0.115, color='black', lw=1.0,
    #             clip_on=False)
    # ax.plot(range_brace(0, 3.5, height=0.05)[1] * 30 + 90, range_brace(0, 4)[0] / 30 + 0.07-0.115, color='black', lw=1.0,
    #             clip_on=False)
    # ax.text(100, 0.35 + 0.025, s='low error', fontdict={'fontfamily': 'Arial', 'fontsize': '14', 'fontweight': '500'},
    #         ha='left', va='center')
    # ax.text(100, 0.175 + 0.025, s='medium error',
    #         fontdict={'fontfamily': 'Arial', 'fontsize': '14', 'fontweight': '500'},
    #         ha='left', va='center')
    # ax.text(100, 0.025, s='high error', fontdict={'fontfamily': 'Arial', 'fontsize': '14', 'fontweight': '500'},
    #         ha='left', va='center')

    # plt.show()

