# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sample_functions
from sklearn import metrics
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV

method_name = 'pls'  # 'pls' or 'svr'
add_nonlinear_terms_flag = False  # True (二乗項・交差項を追加) or False (追加しない)

fold_number = 5  # N-fold CV の N
max_number_of_principal_components = 20  # 使用する主成分の最大数
svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # C の候補
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # ε の候補
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補

dataset_train = pd.read_csv('tep_13_train_with_y.csv', index_col=0)
y_measured_dataset_train = dataset_train.iloc[0:1, :]
for sample_number in range(1, dataset_train.shape[0]):
    if y_measured_dataset_train.iloc[-1, 0] != dataset_train.iloc[sample_number, 0]:
        y_measured_dataset_train = pd.concat(
            [y_measured_dataset_train, dataset_train.iloc[sample_number:sample_number + 1, :]], axis=0)
y_train = y_measured_dataset_train.iloc[:, 0]
x_train = y_measured_dataset_train.iloc[:, 1:]

dataset_test = pd.read_csv('tep_13_test_with_y.csv', index_col=0)
x_test_prediction = dataset_test.iloc[:, 1:]
y_measured_dataset_test = dataset_test.iloc[0:1, :]
measured_index = [0]
for sample_number in range(1, dataset_test.shape[0]):
    if y_measured_dataset_test.iloc[-1, 0] != dataset_test.iloc[sample_number, 0]:
        y_measured_dataset_test = pd.concat(
            [y_measured_dataset_test, dataset_test.iloc[sample_number:sample_number + 1, :]], axis=0)
        measured_index.append(sample_number)
y_test = y_measured_dataset_test.iloc[:, 0]
x_test = y_measured_dataset_test.iloc[:, 1:]

if add_nonlinear_terms_flag:
    x_train = sample_functions.add_nonlinear_terms(x_train)  # 説明変数の二乗項や交差項を追加
    x_test = sample_functions.add_nonlinear_terms(x_test)
    x_test_prediction = sample_functions.add_nonlinear_terms(x_test_prediction)
    # 標準偏差が 0 の説明変数を削除
    std_0_variable_flags = x_train.std() == 0
    x_train = x_train.drop(x_train.columns[std_0_variable_flags], axis=1)
    x_test = x_test.drop(x_test.columns[std_0_variable_flags], axis=1)
    x_test_prediction = x_test_prediction.drop(x_test_prediction.columns[std_0_variable_flags], axis=1)

# オートスケーリング
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()
autoscaled_x_test_prediction = (x_test_prediction - x_train.mean()) / x_train.std()

if method_name == 'pls':
    # CV による成分数の最適化
    components = []  # 空の list の変数を作成して、成分数をこの変数に追加していきます同じく成分数をこの変数に追加
    r2_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の r2 をこの変数に追加
    for component in range(1, min(np.linalg.matrix_rank(autoscaled_x_train), max_number_of_principal_components) + 1):
        # PLS
        model = PLSRegression(n_components=component)  # PLS モデルの宣言
        estimated_y_in_cv = pd.DataFrame(cross_val_predict(model, autoscaled_x_train, autoscaled_y_train,
                                                           cv=fold_number))  # クロスバリデーション推定値の計算し、DataFrame型に変換
        estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()  # スケールをもとに戻す
        r2_in_cv = metrics.r2_score(y_train, estimated_y_in_cv)  # r2 を計算
        print(component, r2_in_cv)  # 成分数と r2 を表示
        r2_in_cv_all.append(r2_in_cv)  # r2 を追加
        components.append(component)  # 成分数を追加

    # 成分数ごとの CV 後の r2 をプロットし、CV 後のr2が最大のときを最適成分数に
    optimal_component_number = sample_functions.plot_and_selection_of_hyperparameter(components, r2_in_cv_all,
                                                                                     'number of components',
                                                                                     'cross-validated r2')
    print('\nCV で最適化された成分数 :', optimal_component_number)
    # PLS
    model = PLSRegression(n_components=optimal_component_number)  # モデルの宣言
elif method_name == 'svr':
    # グラム行列の分散を最大化することによる γ の最適化
    optimal_svr_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x_train, svr_gammas)
    # CV による ε の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
                               cv=fold_number, iid=False)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_svr_epsilon = model_in_cv.best_params_['epsilon']
    # CV による C の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                               {'C': svr_cs}, cv=fold_number, iid=False)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_svr_c = model_in_cv.best_params_['C']
    # CV による γ の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                               {'gamma': svr_gammas}, cv=fold_number, iid=False)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_svr_gamma = model_in_cv.best_params_['gamma']
    # 最適化された C, ε, γ
    print('C : {0}\nε : {1}\nGamma : {2}'.format(optimal_svr_c, optimal_svr_epsilon, optimal_svr_gamma))
    # SVR
    model = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)  # モデルの宣言

model.fit(autoscaled_x_train, autoscaled_y_train)  # モデルの構築
if method_name == 'pls':
    # 標準回帰係数
    standard_regression_coefficients = pd.DataFrame(model.coef_, index=x_train.columns,
                                                    columns=['standard_regression_coefficients'])
    standard_regression_coefficients.to_csv(
        'pls_standard_regression_coefficients.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
sample_functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test)

# テストデータにおけるすべてのサンプルの推定
estimated_y_test_prediction = pd.DataFrame(model.predict(autoscaled_x_test_prediction) * y_train.std() + y_train.mean())
estimated_y_test_prediction.columns = ['estimated_y']  # 列の名前を変更
estimated_y_test_prediction.index = x_test_prediction.index  # 行の名前をサンプル名に変更
estimated_y_test_prediction.to_csv('estimated_y_test_all.csv')  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# 実測値と推定値の時間プロット
plt.rcParams['font.size'] = 18
plt.plot(range(estimated_y_test_prediction.shape[0]), estimated_y_test_prediction.iloc[:, 0], 'b.-',
         label='estimated y')
plt.plot(measured_index, y_measured_dataset_test.iloc[:, 0], 'r.', markersize=15, label='actual y')
plt.xlabel('time')
plt.ylabel(dataset_train.columns[0])
plt.xlim([0, estimated_y_test_prediction.shape[0] + 1])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
plt.show()
# 実測値と推定値の時間プロット (拡大)
plt.plot(range(estimated_y_test_prediction.shape[0]), estimated_y_test_prediction.iloc[:, 0], 'b.-',
         label='estimated y')
plt.plot(measured_index, y_measured_dataset_test.iloc[:, 0], 'r.', markersize=15, label='actual y')
plt.xlabel('time')
plt.ylabel(dataset_train.columns[0])
plt.xlim([894, 960])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
plt.show()
