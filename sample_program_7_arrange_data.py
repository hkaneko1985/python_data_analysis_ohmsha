# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import numpy as np
import pandas as pd
import sample_functions
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV

method_name = 'pls'  # pls or svr
add_nonlinear_terms_flag = False  # True (二乗項・交差項を追加) or False (追加しない)

number_of_test_samples = 800
fold_number = 2  # N-fold CV の N
max_number_of_principal_components = 20  # 使用する主成分の最大数
svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # C の候補
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # ε の候補
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補

#dataset_train = pd.read_csv('te_0_train.csv', encoding='SHIFT-JIS', index_col=0)
#y_train = dataset_train.iloc[:, -1]
#x_train_tmp = dataset_train.iloc[:, 0:22]
#dataset_test = pd.read_csv('te_13_test.csv', encoding='SHIFT-JIS', index_col=0)
#y_test = dataset_test.iloc[:, -1]
#x_test_tmp = dataset_test.iloc[:, 0:22]

dataset_train = pd.read_csv('te_13_train_pre.csv', index_col=0)
tmp = dataset_train.iloc[0, 0]
for i in range(1, dataset_train.shape[0]):
    if  i % 5 == 0:
        tmp = dataset_train.iloc[i, 0]
    else:
        dataset_train.iloc[i, 0] = tmp
dataset_train.to_csv('te_13_train.csv')

dataset_test = pd.read_csv('te_13_test_pre.csv', encoding='SHIFT-JIS', index_col=0)
tmp = dataset_test.iloc[0, 0]
for i in range(1, dataset_test.shape[0]):
    if  i % 5 == 0:
        tmp = dataset_test.iloc[i, 0]
    else:
        dataset_test.iloc[i, 0] = tmp
dataset_test.to_csv('te_13_test.csv')

        
#y_train = dataset_train.iloc[:, 0]
#x_train_tmp = dataset_train.iloc[:, 1:]
#dataset_test = pd.read_csv('te_13_test.csv', encoding='SHIFT-JIS', index_col=0)
#y_test = dataset_test.iloc[:, 0]
#x_test_tmp = dataset_test.iloc[:, 1:]

#if add_nonlinear_terms_flag:
#    x_train_tmp = sample_functions.add_nonlinear_terms(x_train_tmp)  # 説明変数の二乗項や交差項を追加
#    x_test_tmp = sample_functions.add_nonlinear_terms(x_test_tmp)  # 説明変数の二乗項や交差項を追加
#    # 標準偏差が 0 の説明変数を削除
#    x_train = x_train_tmp.drop(x_train_tmp.columns[x_train_tmp.std() == 0], axis=1)
#    x_test = x_test_tmp.drop(x_train_tmp.columns[x_train_tmp.std() == 0], axis=1)
#else:
#    x_train = x_train_tmp.copy()
#    x_test = x_test_tmp.copy()
#
## オートスケーリング
#autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
#autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
#autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()
#
#if method_name == 'pls':
#    # CV による成分数の最適化
#    components = []  # 空の list の変数を作成して、成分数をこの変数に追加していきます同じく成分数をこの変数に追加
#    r2_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の r2 をこの変数に追加
#    for component in range(1, min(np.linalg.matrix_rank(autoscaled_x_train), max_number_of_principal_components) + 1):
#        # PLS
#        model = PLSRegression(n_components=component)  # PLS モデルの宣言
#        estimated_y_in_cv = pd.DataFrame(cross_val_predict(model, autoscaled_x_train, autoscaled_y_train,
#                                                           cv=fold_number))  # クロスバリデーション推定値の計算し、DataFrame型に変換
#        estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()  # スケールをもとに戻す
#        r2_in_cv = metrics.r2_score(y_train, estimated_y_in_cv)  # r2 を計算
#        print(component, r2_in_cv)  # 成分数と r2 を表示
#        r2_in_cv_all.append(r2_in_cv)  # r2 を追加
#        components.append(component)  # 成分数を追加
#
#    # 成分数ごとの CV 後の r2 をプロットし、CV 後のr2が最大のときを最適成分数に
#    optimal_component_number = sample_functions.plot_and_selection_of_hyperparameter(components, r2_in_cv_all,
#                                                                                     'number of components',
#                                                                                     'cross-validated r2')
#    print('\nCV で最適化された成分数 :', optimal_component_number)
#    # PLS
#    model = PLSRegression(n_components=optimal_component_number)  # モデルの宣言
#elif method_name == 'svr':
#    # グラム行列の分散を最大化することによる γ の最適化
#    print('グラム行列の分散を最大化することによる γ の最適化')
#    variance_of_gram_matrix = list()
#    for index, svr_gamma in enumerate(svr_gammas):
#        print(index + 1, '/', len(svr_gammas))
#        gram_matrix = np.exp(
#            -svr_gamma * cdist(autoscaled_x_train, autoscaled_x_train, metric='seuclidean'))
#        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
#    optimal_svr_gamma = svr_gammas[np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
#    # CV による ε の最適化
#    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
#                               cv=fold_number, iid=False)
#    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
#    optimal_svr_epsilon = model_in_cv.best_params_['epsilon']
#    # CV による C の最適化
#    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
#                               {'C': svr_cs}, cv=fold_number, iid=False)
#    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
#    optimal_svr_c = model_in_cv.best_params_['C']
#    # CV による γ の最適化
#    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
#                               {'gamma': svr_gammas}, cv=fold_number, iid=False)
#    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
#    optimal_svr_gamma = model_in_cv.best_params_['gamma']
#    # 最適化された C, ε, γ
#    print('C : {0}\nε : {1}\nGamma : {2}'.format(optimal_svr_c, optimal_svr_epsilon, optimal_svr_gamma))
#    # SVR
#    model = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)  # モデルの宣言
#
#model.fit(autoscaled_x_train, autoscaled_y_train)  # モデルの構築
#if method_name == 'pls':
#    # 標準回帰係数
#    standard_regression_coefficients = pd.DataFrame(model.coef_)  # Pandas の DataFrame 型に変換
#    standard_regression_coefficients.index = x_train.columns  # 説明変数に対応する名前を、元のデータセットにおける説明変数の名前に
#    standard_regression_coefficients.columns = ['standard_regression_coefficients']  # 列名を変更
#    standard_regression_coefficients.to_csv(
#        'pls_standard_regression_coefficients.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
## トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
#sample_functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
#                                                                               autoscaled_x_test, y_test)
