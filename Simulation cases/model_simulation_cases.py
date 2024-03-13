import numpy as np
import xgboost as xgb
from joblib import dump, load
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def EP_plot(true, pred, name):
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
    plt.savefig(name+'.svg', dpi=600)
    # plt.show()
    return (np.round(r2_score(true, pred), 3), np.round(mean_absolute_error(true, pred), 3),
            np.round(np.sqrt(mean_squared_error(true, pred)), 3))


def model_train(train_data, params, num, lag=2, shifts=[1],):
    train = []
    time_single = [0, 50, 100, 200, 400, 600, 800, 1000]
    time = np.array(time_single*train_data.shape[0]).reshape(-1, len(time_single))
    length_x = 12

    for index in range(train_data.shape[0]):
        x = train_data[index, 0:length_x]
        for j in range(len(time_single)-1):
            t = time[index, 1+j]
            r = train_data[index, length_x+1+j]
            p_t = time[index, j]
            r_t = train_data[index, length_x+j]
            x_plus = [p_t, t, t-p_t, r_t]
            x_temp = x.tolist() + x_plus
            train.append(x_temp+[r])
        if lag == 2:
            for k in range(len(time_single)-2):
                t = time[index, 2 + k]
                r = train_data[index, length_x+2+k]
                p_t = time[index, k]
                r_t = train_data[index, length_x+k]
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


def model_test(model_trained, test_data, name,lag=2, shifts=[1]):
    time_single = [0, 50, 100, 200, 400, 600, 800, 1000]
    time = np.array(time_single * test_data.shape[0]).reshape(-1, len(time_single))
    length_x = 12

    result = [[], []]
    for num_p in range(len(time_single)-1):
        x = test_data[:, 0:length_x]
        if num_p == 0:
            p_t = time[:, num_p]
            p_r = test_data[:, length_x+num_p]
            for k in range(len(result)):
                t = time[:, num_p+k+1]
                x_plus = np.c_[np.c_[p_t.reshape(-1, 1), t.reshape(-1, 1)], np.c_[(t-p_t).reshape(-1, 1), p_r.reshape(-1, 1)]]
                x_test = np.c_[x, x_plus]
                pred_temp = model_trained.predict(xgb.DMatrix(x_test, label=test_data[:, 7+num_p+k+1].reshape(-1, 1)))
                result[k].append(pred_temp)
        elif num_p == 1:
            p_t = test_data[:, num_p]
            p_r = result[0][num_p - 1]
            for k in range(len(result)):
                t = test_data[:, num_p+k+1]
                x_plus = np.c_[
                    np.c_[p_t.reshape(-1, 1), t.reshape(-1, 1)], np.c_[(t - p_t).reshape(-1, 1), p_r.reshape(-1, 1)]]
                x_test = np.c_[x, x_plus]
                pred_temp = model_trained.predict(xgb.DMatrix(x_test, label=test_data[:, 7 + num_p + k + 1].reshape(-1, 1)))
                result[k].append(pred_temp)
        else:
            p_t = test_data[:, num_p]
            p_r = (result[0][num_p-1] + result[1][num_p-2])/2
            for k in range(len(result)):
                t = test_data[:, num_p+k+1]
                x_plus = np.c_[
                    np.c_[p_t.reshape(-1, 1), t.reshape(-1, 1)], np.c_[(t - p_t).reshape(-1, 1), p_r.reshape(-1, 1)]]
                x_test = np.c_[x, x_plus]
                pred_temp = model_trained.predict(xgb.DMatrix(x_test))
                result[k].append(pred_temp)
    for index in range(len(time_single)-1):
        if index == 0:
            pred_result = result[0][0].reshape(-1, 1)
        else:
            pred_result = np.c_[pred_result, ((result[0][index] + result[1][index - 1]) / 2).reshape(-1, 1)]
    true = test_data[:, -(len(time_single)-1):].reshape(-1, 1)
    pred = pred_result.reshape(-1, 1)
    metric = EP_plot(true * 100, pred * 100, name)
    return metric


if __name__ == '__main__':
    metrics = []
    for index in range(1, 21, 1):
        x = np.load('x_m'+str(index)+'.npy')
        y = np.load('y_m'+str(index)+'.npy')
        print(x.shape)

        mask_all = [index for index in range(x.shape[0])]
        np.random.seed(0)
        np.random.shuffle(mask_all)
        mask_test = mask_all[0:int(x.shape[0]*0.2)]
        mask_train = mask_all[int(x.shape[0]*0.2):]

        data = np.c_[x, y]
        data_train = data[mask_train, :]
        data_test = data[mask_test, :]

        params = {'eta': 0.02, 'max_depth': 13, 'min_child_weight': 5} 
        model_trained = model_train(data_train, params, num=400)
        metric = model_test(model_trained, data_test, 'M'+str(index))
        metrics.append(metric)
    np.savetxt("metrics.txt", metrics)
