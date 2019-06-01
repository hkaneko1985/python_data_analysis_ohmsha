# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA  # scikit-learn の中の PCA を実行するためのライブラリのインポート

dataset = pd.read_csv('iris_without_species.csv', index_col=0)
autoscaled_dataset = (dataset - dataset.mean()) / dataset.std()  # オートスケーリング

# PCA
pca = PCA()  # PCA を行ったり PCA の結果を格納したりするための変数を、pca として宣言
pca.fit(autoscaled_dataset)  # PCA を実行

# ローディング
loadings = pd.DataFrame(pca.components_.T)  # ローディングを pandas の DataFrame 型に変換
loadings.index = dataset.columns  # P の行の名前を、元の多変量データの特徴量の名前に
loadings.to_csv('pca_loadings.csv') 

# スコア
score = pd.DataFrame(pca.transform(autoscaled_dataset))  # 主成分スコアの計算した後、pandas の DataFrame 型に変換
score.index = dataset.index  # 主成分スコアのサンプル名を、元のデータセットのサンプル名に
score.to_csv('pca_score.csv')

# 第 1 主成分と第 2 主成分の散布図
plt.rcParams['font.size'] = 18
plt.scatter(score.iloc[:, 0], score.iloc[:, 1], c='blue')
plt.xlabel('t_1')
plt.ylabel('t_2')
plt.show()

# 寄与率、累積寄与率
contribution_ratios = pd.DataFrame(pca.explained_variance_ratio_)  # 寄与率を DataFrame 型に変換
cumulative_contribution_ratios = contribution_ratios.cumsum()  # cumsum() で寄与率の累積和を計算
cont_cumcont_ratios = pd.concat(
    [contribution_ratios, cumulative_contribution_ratios],
    axis=1).T
cont_cumcont_ratios.index = ['contribution_ratio', 'cumulative_contribution_ratio']  # 行の名前を変更
cont_cumcont_ratios.to_csv('pca_cont_cumcont_ratios.csv')

# 寄与率を棒グラフで、累積寄与率を線で入れたプロット図を重ねて描画
x_axis = range(1, contribution_ratios.shape[0] + 1)  # 1 から成分数までの整数が x 軸の値
plt.rcParams['font.size'] = 18
plt.bar(x_axis, contribution_ratios.iloc[:, 0], align='center')  # 寄与率の棒グラフ
plt.plot(x_axis, cumulative_contribution_ratios.iloc[:, 0], 'r.-')  # 累積寄与率の線を入れたプロット図
plt.xlabel('Number of principal components')  # 横軸の名前
plt.ylabel('Contribution ratio(blue),\nCumulative contribution ratio(red)')  # 縦軸の名前。\n で改行しています
plt.show()
