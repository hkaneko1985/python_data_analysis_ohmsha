# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import numpy as np
import pandas as pd
import sample_functions
from sklearn import svm

ocsvm_nu = 0.003  # OCSVM における ν。トレーニングデータにおけるサンプル数に対する、サポートベクターの数の下限の割合
ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補

dataset_train = pd.read_csv('tep_13_train_with_y.csv', index_col=0)
dataset_test = pd.read_csv('tep_13_test_with_y.csv', index_col=0)
dataset = pd.concat([dataset_train, dataset_test], axis=0)

## y が測定された時刻のサンプルのみを使用する場合はこちらをご利用ください
#dataset = dataset_train.iloc[0:1, :]
#for sample_number in range(1, dataset_train.shape[0]):
#    if dataset.iloc[-1, 0] != dataset_train.iloc[sample_number, 0]:
#        dataset = pd.concat(
#            [dataset, dataset_train.iloc[sample_number:sample_number + 1, :]], axis=0)
#dataset = pd.concat([dataset, dataset_test.iloc[0:1, :]], axis=0)
#for sample_number in range(1, dataset_test.shape[0]):
#    if dataset.iloc[-1, 0] != dataset_test.iloc[sample_number, 0]:
#        dataset = pd.concat(
#            [dataset, dataset_test.iloc[sample_number:sample_number + 1, :]], axis=0)

x = dataset.iloc[:, 1:]
autoscaled_x = (x - x.mean()) / x.std()  # オートスケーリング

# グラム行列の分散を最大化することによる γ の最適化
optimal_ocsvm_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x, ocsvm_gammas)
# 最適化された γ
print('最適化された gamma (OCSVM) :', optimal_ocsvm_gamma)

# OCSVM による AD
ad_model = svm.OneClassSVM(kernel='rbf', gamma=optimal_ocsvm_gamma, nu=ocsvm_nu)  # AD モデルの宣言
ad_model.fit(autoscaled_x)  # モデル構築

# トレーニングデータのデータ密度 (f(x) の値)
data_density_train = ad_model.decision_function(autoscaled_x)
number_of_support_vectors = len(ad_model.support_)
number_of_outliers_in_training_data = sum(data_density_train < 0)
print('\nトレーニングデータにおけるサポートベクター数 :', number_of_support_vectors)
print('トレーニングデータにおけるサポートベクターの割合 :', number_of_support_vectors / x.shape[0])
print('\nトレーニングデータにおける外れサンプル数 :', number_of_outliers_in_training_data)
print('トレーニングデータにおける外れサンプルの割合 :', number_of_outliers_in_training_data / x.shape[0])
data_density_train = pd.DataFrame(data_density_train, index=x.index, columns=['ocsvm_data_density'])
data_density_train.to_csv('ocsvm_data_density_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# トレーニングデータに対して、AD の中か外かを判定
inside_ad_flag_train = data_density_train >= 0  # AD 内のサンプルのみ TRUE
inside_ad_flag_train.columns = ['inside_ad_flag']  # 列名を変更
inside_ad_flag_train.to_csv('inside_ad_flag_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
