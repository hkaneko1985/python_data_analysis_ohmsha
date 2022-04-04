# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd
import sample_functions
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

number_of_principal_components = 8  # 使用する主成分の数
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
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()

# PCA
pca = PCA(n_components=number_of_principal_components)
pca.fit(autoscaled_x_train)
t_train = pd.DataFrame(pca.transform(autoscaled_x_train))  # トレーニングデータの主成分スコアの計算
t_train.index = x_train.index
t_test = pd.DataFrame(pca.transform(autoscaled_x_test))  # テストデータの主成分スコアの計算
t_test.index = x_test.index

# OLS
model = LinearRegression()  # モデルの宣言
model.fit(t_train, autoscaled_y_train)  # モデルの構築

# 標準回帰係数
standard_regression_coefficients = pd.DataFrame(pca.components_.T.dot(model.coef_), index=x_train.columns,
                                                columns=['standard_regression_coefficients'])
standard_regression_coefficients.to_csv(
    'pcr_standard_regression_coefficients.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
sample_functions.estimation_and_performance_check_in_regression_train_and_test(model, t_train, y_train, t_test, y_test)
