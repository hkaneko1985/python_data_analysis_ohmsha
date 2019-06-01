# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

tsne_perplexity_optimization = False  # True にすると t-SNE の perplexity を candidates_of_perplexity の中から k3n-error が最小になるように決めます(時間がかかります)。False にすると 下の perplexity が用いられます
perplexity = 20  # t-SNE の perplexity
candidates_of_perplexity = np.arange(5, 105, 5, dtype=int)
k_in_k3n_error = 10

dataset = pd.read_csv('unique_m.csv', index_col=-1)
# dataset = dataset.sort_values('critical_temp', ascending=False).iloc[:1000, :]
original_x = dataset.iloc[:, :86]
x = original_x.drop(original_x.columns[original_x.std() == 0], axis=1)
x = (x.T / x.T.sum()).T
autoscaled_x = (x - x.mean()) / x.std()  # オートスケーリング

# PCA
pca = PCA()  # PCA を行ったり PCA の結果を格納したりするための変数を、pca として宣言
pca.fit(autoscaled_x)  # PCA を実行
# ローディング
loadings = pd.DataFrame(pca.components_.T, index=x.columns)
loadings.to_csv('pca_loadings.csv')
# スコア
score = pd.DataFrame(pca.transform(autoscaled_x), index=x.index)
score.to_csv('pca_score.csv')
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
# 第 1 主成分と第 2 主成分の散布図 (転移温度の値でサンプルに色付け)
plt.scatter(score.iloc[:, 0], score.iloc[:, 1], c=dataset.iloc[:, 86], cmap=plt.get_cmap('jet'))
plt.colorbar()
plt.xlabel('t_1 (PCA)')
plt.ylabel('t_2 (PCA)')
plt.show()

# t-SNE
selected_indexes = np.arange(0, autoscaled_x.shape[0], 5)
autoscaled_x = autoscaled_x.iloc[selected_indexes, :]
t = TSNE(perplexity=perplexity, n_components=2, init='pca', random_state=10).fit_transform(autoscaled_x)
t = pd.DataFrame(t, index=autoscaled_x.index, columns=['t_1 (t-SNE)', 't_2 (t-SNE)'])
t.columns = ['t_1 (t-SNE)', 't_2 (t-SNE)']
t.to_csv('tsne_t.csv')
# t1 と t2 の散布図 (転移温度の値でサンプルに色付け)
plt.rcParams['font.size'] = 18
plt.scatter(t.iloc[:, 0], t.iloc[:, 1], c=dataset.iloc[selected_indexes, 86], cmap=plt.get_cmap('jet'))
plt.colorbar()
plt.xlabel('t_1 (t-SNE)')
plt.ylabel('t_2 (t-SNE)')
plt.show()
