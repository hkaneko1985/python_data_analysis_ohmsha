# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn import svm
from sklearn.model_selection import train_test_split

ocsvm_nu = 0.045  # OCSVM における ν。トレーニングデータにおけるサンプル数に対する、サポートベクターの数の下限の割合
ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補
number_of_test_samples = 150  # テストデータのサンプル数
dataset = pd.read_csv('boston.csv', index_col=0)

# データ分割
y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数
# ランダムにトレーニングデータとテストデータとに分割
# random_state に数字を与えることで、別のときに同じ数字を使えば、ランダムとはいえ同じ結果にすることができます
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True,
                                                    random_state=99)

# オートスケーリング
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_x_test = (x_test - x_test.mean()) / x_test.std()

# グラム行列の分散を最大化することによる γ の最適化
variance_of_gram_matrix = list()
for ocsvm_gamma in ocsvm_gammas:
    gram_matrix = np.exp(
        -ocsvm_gamma * cdist(autoscaled_x_train, autoscaled_x_train, metric='sqeuclidean'))
    variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
optimal_ocsvm_gamma = ocsvm_gammas[np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
# 最適化された γ
print('最適化された gamma :', optimal_ocsvm_gamma)

# OCSVM による AD
ad_model = svm.OneClassSVM(kernel='rbf', gamma=optimal_ocsvm_gamma, nu=ocsvm_nu)  # AD モデルの宣言
ad_model.fit(autoscaled_x_train)  # モデル構築

# トレーニングデータのデータ密度 (f(x) の値)
data_density_train = ad_model.decision_function(autoscaled_x_train)
number_of_support_vectors = len(ad_model.support_)
number_of_outliers_in_training_data = sum(data_density_train < 0)
print('\nトレーニングデータにおけるサポートベクター数 :', number_of_support_vectors)
print('トレーニングデータにおけるサポートベクターの割合 :', number_of_support_vectors / x_train.shape[0])
print('\nトレーニングデータにおける外れサンプル数 :', number_of_outliers_in_training_data)
print('トレーニングデータにおける外れサンプルの割合 :', number_of_outliers_in_training_data / x_train.shape[0])
data_density_train = pd.DataFrame(data_density_train, index=x_train.index, columns=['ocsvm_data_density'])
data_density_train.to_csv('ocsvm_data_density_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# トレーニングデータに対して、AD の中か外かを判定
inside_ad_flag_train = data_density_train >= 0
inside_ad_flag_train.columns = ['inside_ad_flag']
inside_ad_flag_train.to_csv('inside_ad_flag_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# テストデータのデータ密度 (f(x) の値)
data_density_test = ad_model.decision_function(autoscaled_x_test)
number_of_outliers_in_test_data = sum(data_density_test < 0)
print('\nテストデータにおける外れサンプル数 :', number_of_outliers_in_test_data)
print('テストデータにおける外れサンプルの割合 :', number_of_outliers_in_test_data / x_test.shape[0])
data_density_test = pd.DataFrame(data_density_test, index=x_test.index, columns=['ocsvm_data_density'])
data_density_test.to_csv('ocsvm_data_density_test.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# テストデータに対して、AD の中か外かを判定
inside_ad_flag_test = data_density_test >= 0
inside_ad_flag_test.columns = ['inside_ad_flag']
inside_ad_flag_test.to_csv('inside_ad_flag_test.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
