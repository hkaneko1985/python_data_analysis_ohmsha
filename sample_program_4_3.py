# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

number_of_test_samples = 150
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

# 最小二乗法による線形単回帰分析による標準回帰係数の計算
model = LinearRegression()  # モデルの宣言
model.fit(autoscaled_x_train, autoscaled_y_train)  # モデルの構築

# 標準回帰係数
standard_regression_coefficients = pd.DataFrame(model.coef_, index=x_train.columns,
                                                columns=['standard_regression_coefficients'])
standard_regression_coefficients.to_csv(
    'standard_regression_coefficients.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# トレーニングデータの推定
autoscaled_estimated_y_train = model.predict(autoscaled_x_train)  # y の推定
estimated_y_train = autoscaled_estimated_y_train * y_train.std() + y_train.mean()  # スケールをもとに戻す
estimated_y_train = pd.DataFrame(estimated_y_train, index=x_train.index, columns=['estimated_y'])

# トレーニングデータの実測値 vs. 推定値のプロット
plt.rcParams['font.size'] = 18
plt.figure(figsize=figure.figaspect(1))  # 図の形を正方形に
plt.scatter(y_train, estimated_y_train.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
y_max = max(y_train.max(), estimated_y_train.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
y_min = min(y_train.min(), estimated_y_train.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
plt.xlabel('actual y')  # x 軸の名前
plt.ylabel('estimated y')  # y 軸の名前
plt.show()  # 以上の設定で描画

# トレーニングデータのr2, RMSE, MAE
print('r^2 for training data :', metrics.r2_score(y_train, estimated_y_train))
print('RMSE for training data :', metrics.mean_squared_error(y_train, estimated_y_train) ** 0.5)
print('MAE for training data :', metrics.mean_absolute_error(y_train, estimated_y_train))

# トレーニングデータの結果の保存
y_train_for_save = pd.DataFrame(y_train)  # Series のため列名は別途変更
y_train_for_save.columns = ['actual_y']
y_error_train = y_train_for_save.iloc[:, 0] - estimated_y_train.iloc[:, 0]
y_error_train = pd.DataFrame(y_error_train)  # Series のため列名は別途変更
y_error_train.columns = ['error_of_y(actual_y-estimated_y)']
results_train = pd.concat([estimated_y_train, y_train_for_save, y_error_train], axis=1)
results_train.to_csv('estimated_y_train.csv')  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください


# テストデータの、トレーニングデータを用いたオートスケーリング
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()

# テストデータの推定
autoscaled_estimated_y_test = model.predict(autoscaled_x_test)  # y の推定
estimated_y_test = autoscaled_estimated_y_test * y_train.std() + y_train.mean()  # スケールをもとに戻す
estimated_y_test = pd.DataFrame(estimated_y_test, index=x_test.index, columns=['estimated_y'])

# テストデータの実測値 vs. 推定値のプロット
plt.rcParams['font.size'] = 18
plt.figure(figsize=figure.figaspect(1))  # 図の形を正方形に
plt.scatter(y_test, estimated_y_test.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
y_max = max(y_test.max(), estimated_y_test.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
y_min = min(y_test.min(), estimated_y_test.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
plt.xlabel('actual y')  # x 軸の名前
plt.ylabel('estimated y')  # y 軸の名前
plt.show()  # 以上の設定で描画

# テストデータのr2, RMSE, MAE
print('r^2 for test data :', metrics.r2_score(y_test, estimated_y_test))
print('RMSE for test data :', metrics.mean_squared_error(y_test, estimated_y_test) ** 0.5)
print('MAE for test data :', metrics.mean_absolute_error(y_test, estimated_y_test))

# テストデータの結果の保存
y_test_for_save = pd.DataFrame(y_test)  # Series のため列名は別途変更
y_test_for_save.columns = ['actual_y']
y_error_test = y_test_for_save.iloc[:, 0] - estimated_y_test.iloc[:, 0]
y_error_test = pd.DataFrame(y_error_test)  # Series のため列名は別途変更
y_error_test.columns = ['error_of_y(actual_y-estimated_y)']
results_test = pd.concat([estimated_y_test, y_test_for_save, y_error_test], axis=1)
results_test.to_csv('estimated_y_test.csv')  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
