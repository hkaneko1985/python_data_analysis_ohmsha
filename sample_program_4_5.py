# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd
import sample_functions
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_predict  # クロスバリデーションにより推定値を計算するために使用

max_number_of_principal_components = 13  # 使用する主成分の最大数。説明変数の数より小さい必要があります
fold_number = 5  # N-fold CV の N
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

# CV による成分数の最適化
pca = PCA()  # PCA ですべての主成分を計算しておきます
pca.fit(autoscaled_x_train)
t_train = pd.DataFrame(pca.transform(autoscaled_x_train))  # トレーニングデータの主成分スコアの計算
t_train.index = x_train.index
t_test = pd.DataFrame(pca.transform(autoscaled_x_test))  # テストデータの主成分スコアの計算
t_test.index = x_test.index
components = []  # 空の list の変数を作成して、成分数をこの変数に追加していきます同じく成分数をこの変数に追加
r2_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の r2 をこの変数に追加
for component in range(1, max_number_of_principal_components + 1):
    # OLS
    model = LinearRegression()  # モデルの宣言
    estimated_y_in_cv = pd.DataFrame(cross_val_predict(model, t_train.iloc[:, 0:component], autoscaled_y_train,
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

# PCA
pca = PCA(n_components=optimal_component_number)
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
