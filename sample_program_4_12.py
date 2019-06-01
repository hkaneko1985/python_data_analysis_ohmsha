# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import numpy as np
import pandas as pd
import sample_functions
from scipy.spatial.distance import cdist
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV

svm_cs = 2 ** np.arange(-5, 11, dtype=float)
svm_gammas = 2 ** np.arange(-20, 11, dtype=float)
fold_number = 5  # N-fold CV の N
number_of_test_samples = 50  # テストデータのサンプル数
dataset = pd.read_csv('iris.csv', index_col=0)  # あやめのデータの読み込み
# 2 クラス 1 (positive), -1 (negative)  にします
dataset.iloc[0:100, 0] = 'positive'  # setosa と versicolor を 1 (positive) のクラスに
dataset.iloc[100:, 0] = 'negative'  # virginica を -1 (negative) のクラスに

# データ分割
y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数

# ランダムにトレーニングデータとテストデータとに分割
# random_state に数字を与えることで、別のときに同じ数字を使えば、ランダムとはいえ同じ結果にすることができます
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True,
                                                    random_state=21)

# オートスケーリング
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()

# グラム行列の分散を最大化することによる γ の最適化
optimal_svm_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x_train, svm_gammas)

# CV による C の最適化
model_in_cv = GridSearchCV(svm.SVC(kernel='rbf', gamma=optimal_svm_gamma),
                           {'C': svm_cs}, cv=fold_number, iid=False)
model_in_cv.fit(autoscaled_x_train, y_train)
optimal_svm_c = model_in_cv.best_params_['C']

# CV による γ の最適化
model_in_cv = GridSearchCV(svm.SVC(kernel='rbf', C=optimal_svm_c),
                           {'gamma': svm_gammas}, cv=fold_number, iid=False)
model_in_cv.fit(autoscaled_x_train, y_train)
optimal_svm_gamma = model_in_cv.best_params_['gamma']
print('CV で最適化された C :', optimal_svm_c)
print('CV で最適化された γ:', optimal_svm_gamma)

# SVM
model = svm.SVC(kernel='rbf', C=optimal_svm_c, gamma=optimal_svm_gamma)  # モデルの宣言
model.fit(autoscaled_x_train, y_train)  # モデルの構築

# トレーニングデータ・テストデータの推定、混同行列の作成、正解率の値の表示、推定値の保存
sample_functions.estimation_and_performance_check_in_classification_train_and_test(model, autoscaled_x_train, y_train,
                                                                                   autoscaled_x_test, y_test)
