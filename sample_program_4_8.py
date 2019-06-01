# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import numpy as np
import pandas as pd
import sample_functions
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV

svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # C の候補
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # ε の候補
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補
fold_number = 5  # N-fold CV の N

number_of_test_samples = 150  # テストデータのサンプル数
dataset = pd.read_csv('boston.csv', index_col=0)

# データ分割
y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数

# ランダムにトレーニングデータとテストデータとに分割
# random_state に数字を与えることで、別のときに同じ数字を使えば、ランダムとはいえ同じ結果にすることができます
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)

# オートスケーリング
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

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

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
sample_functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test)
