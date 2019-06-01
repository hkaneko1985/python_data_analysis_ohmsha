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

dataset = pd.read_csv('unique_m.csv', index_col=-1)
dataset = dataset.sort_values('critical_temp', ascending=False).iloc[:4000, :]
original_x = dataset.iloc[:, :86]
original_x = (original_x.T / original_x.T.sum()).T
# 標準偏差が 0 の説明変数を削除
x = original_x.drop(original_x.columns[original_x.std() == 0], axis=1)
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
inside_ad_flag_train.columns = ['inside_ad_flag']
inside_ad_flag_train.to_csv('inside_ad_flag_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
