# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.figure as figure  # 図の調整に使用
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('beer_consumption.csv', index_col=0)

# モデル構築 1. データ分割
y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数

# モデル構築 2. オートスケーリング
autoscaled_y = (y - y.mean()) / y.std()
autoscaled_x = (x - x.mean()) / x.std()

# モデル構築 3. 最小二乗法による線形単回帰分析による標準回帰係数の計算
model = LinearRegression()  # モデルの宣言
model.fit(autoscaled_x, autoscaled_y)  # モデルの構築

x_new = dataset.iloc[:, 1:]  # 今回はモデル構築に用いたデータセットと同じデータセットの y を推定します

# 新しいデータの推定 1. モデル構築用のデータを用いたオートスケーリング
autoscaled_x_new = (x_new - x.mean()) / x.std()

# 新しいデータの推定 2. 推定
autoscaled_estimated_y_new = model.predict(autoscaled_x_new)  # y の推定

# 新しいデータの推定 3. 推定値のスケールをもとに戻す  
estimated_y_new = autoscaled_estimated_y_new * y.std() + y.mean()
estimated_y_new = pd.DataFrame(estimated_y_new, index=dataset.index,
                               columns=['estimated_y'])  # Pandas の DataFrame 型に変換。行の名前・列の名前も設定
estimated_y_new.to_csv('estimated_y.csv')  # 推定値を csv ファイルに保存

# 実測値 vs. 推定値のプロット
plt.rcParams['font.size'] = 18
plt.figure(figsize=figure.figaspect(1))  # 図の形を正方形に
plt.scatter(y, estimated_y_new.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
y_max = max(y.max(), estimated_y_new.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
y_min = min(y.min(), estimated_y_new.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
plt.xlabel('actual y')  # x 軸の名前
plt.ylabel('estimated y')  # y 軸の名前
plt.show()  # 以上の設定で描画

# 標準回帰係数、傾き、y 切片の表示
print('標準回帰係数 :', model.coef_[0])
print('傾き :', model.coef_[0] * y.std() / x.std()[0])
print('y 切片 :', y.mean() - model.coef_[0] * x.mean()[0] * y.std() / x.std()[0])
