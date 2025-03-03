# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd
import sample_functions
from sklearn import metrics
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, cross_val_predict

max_number_of_principal_components = 20  # 使用する主成分の最大数。説明変数の数より小さい必要があります
fold_number = 5  # N-fold CV の N
number_of_test_samples = 150  # テストデータのサンプル数
dataset = pd.read_csv('boston.csv', index_col=0)

# データ分割
y = dataset.iloc[:, 0]  # 目的変数
original_x = dataset.iloc[:, 1:]  # 説明変数

# 説明変数の二乗項や交差項を追加
x = original_x.copy()  # 元の説明変数のデータセット
x_square = original_x ** 2  # 二乗項
# 追加
for i in range(original_x.shape[1]):
    for j in range(original_x.shape[1]):
        if i == j:  # 二乗項
            x = pd.concat(
                [x, x_square.rename(columns={x_square.columns[i]: '{0}^2'.format(x_square.columns[i])}).iloc[:, i]],
                axis=1)
        elif i < j:  # 交差項
            x = pd.concat([x, original_x.iloc[:, i] * original_x.iloc[:, j]], axis=1)
            x = x.rename(columns={0: '{0}*{1}'.format(x_square.columns[i], x_square.columns[j])})

# ランダムにトレーニングデータとテストデータとに分割
# random_state に数字を与えることで、別のときに同じ数字を使えば、ランダムとはいえ同じ結果にすることができます
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True,
                                                    random_state=99)

# オートスケーリング
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()

# CV による成分数の最適化
components = []  # 空の list の変数を作成して、成分数をこの変数に追加していきます同じく成分数をこの変数に追加
r2_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の r2 をこの変数に追加
for component in range(1, max_number_of_principal_components + 1):
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
model.fit(autoscaled_x_train, autoscaled_y_train)  # モデルの構築

# 標準回帰係数
standard_regression_coefficients = pd.DataFrame(model.coef_, index=x_train.columns,
                                                columns=['standard_regression_coefficients'])  # Pandas の DataFrame 型に変換
standard_regression_coefficients.to_csv(
    'pls_standard_regression_coefficients.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# トレーニングデータ・テストデータの推定、実測値 vs. 推定値のプロット、r2, RMSE, MAE の値の表示、推定値の保存
sample_functions.estimation_and_performance_check_in_regression_train_and_test(model, autoscaled_x_train, y_train,
                                                                               autoscaled_x_test, y_test)
