# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import numpy as np
import pandas as pd
import sample_functions
from sklearn import svm
import matplotlib.pyplot as plt

ocsvm_nu = 0.003  # OCSVM における ν。トレーニングデータにおけるサンプル数に対する、サポートベクターの数の下限の割合
ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補
x_train = pd.read_csv('tep_0.csv', index_col=0)
x_test = pd.read_csv('tep_7.csv', index_col=0)

# オートスケーリング
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_x_test = (x_test - x_test.mean()) / x_test.std()

# グラム行列の分散を最大化することによる γ の最適化
optimal_ocsvm_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x_train, ocsvm_gammas)
# 最適化された γ
print('最適化された gamma :', optimal_ocsvm_gamma)

# OCSVM による異常検出モデル
model = svm.OneClassSVM(kernel='rbf', gamma=optimal_ocsvm_gamma, nu=ocsvm_nu)
model.fit(autoscaled_x_train)  # モデル構築

# トレーニングデータのデータ密度 (f(x) の値)
data_density_train = model.decision_function(autoscaled_x_train)
number_of_support_vectors = len(model.support_)
number_of_outliers_in_training_data = sum(data_density_train < 0)
print('\nトレーニングデータにおけるサポートベクター数 :', number_of_support_vectors)
print('トレーニングデータにおけるサポートベクターの割合 :', number_of_support_vectors / x_train.shape[0])
print('\nトレーニングデータにおける外れサンプル数 :', number_of_outliers_in_training_data)
print('トレーニングデータにおける外れサンプルの割合 :', number_of_outliers_in_training_data / x_train.shape[0])
data_density_train = pd.DataFrame(data_density_train, index=x_train.index, columns=['ocsvm_data_density'])
data_density_train.to_csv('ocsvm_data_density_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# テストデータのデータ密度 (f(x) の値)
data_density_test = model.decision_function(autoscaled_x_test)
data_density_test = pd.DataFrame(data_density_test, index=x_test.index, columns=['ocsvm_data_density'])
data_density_test.to_csv('ocsvm_data_density_test.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# トレーニングデータのデータ密度の時間プロット
y_max = max(data_density_train.iloc[:, 0].max(), data_density_test.iloc[:, 0].max())
y_min = min(data_density_train.iloc[:, 0].min(), data_density_test.iloc[:, 0].min())
plt.rcParams['font.size'] = 18
plt.plot(range(data_density_train.shape[0]), data_density_train.iloc[:, 0], 'b.-')
plt.plot([-1, data_density_train.shape[0] + 1], [0, 0], 'r-')
plt.xlabel('time')
plt.ylabel('output of OCSVM')
plt.ylim(y_min - 0.03 * (y_max - y_min), y_max + 0.03 * (y_max - y_min))
plt.xlim([0, data_density_train.shape[0]])
plt.show()

# テストデータのデータ密度の時間プロット
plt.rcParams['font.size'] = 18
plt.plot(range(data_density_test.shape[0]), data_density_test.iloc[:, 0], 'b.-')
plt.plot([-1, data_density_test.shape[0] + 1], [0, 0], 'r-')
plt.xlabel('time')
plt.ylabel('output of OCSVM')
plt.ylim(y_min - 0.03 * (y_max - y_min), y_max + 0.03 * (y_max - y_min))
plt.xlim([0, data_density_test.shape[0]])
plt.show()
