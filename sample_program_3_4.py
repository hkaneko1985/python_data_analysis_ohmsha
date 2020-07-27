# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster  # SciPy の中の階層的クラスタリングを実行したり樹形図を作成したりするためのライブラリをインポート
from sklearn.decomposition import PCA

number_of_clusters = 3  # クラスターの数

selected_sample_numbers = [0, 1, 2, 3, 4, 59, 60, 64, 79, 81, 102, 105, 107, 109, 117]  # デンドログラムの結果を見やすくするため、事前にサンプル選択

dataset = pd.read_csv('iris_without_species.csv', index_col=0)
x = dataset.iloc[selected_sample_numbers, :]
autoscaled_x = (x - x.mean()) / x.std()  # オートスケーリング

# 階層的クラスタリング
clustering_results = linkage(autoscaled_x, metric='euclidean', method='ward')
# 　metric, method を下のように変えることで、それぞれ色々な距離、手法で階層的クラスタリングを実行可能
#
# metric の種類
# - euclidean : ユークリッド距離
# - cityblock : マンハッタン距離(シティブロック距離)
# など
#
# method の種類
# - single : 最近隣法
# - complete : 最遠隣法
# - weighted : 重心法
# - average : 平均距離法
# - ward : ウォード法
# など

# デンドログラムの作成
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
dendrogram(clustering_results, labels=list(x.index), color_threshold=0,
           orientation='right')  # デンドログラムの作成。labels=x.index でサンプル名を入れています
plt.xlabel('distance')  # 横軸の名前
plt.show()

# クラスター番号の保存
cluster_numbers = fcluster(clustering_results, number_of_clusters, criterion='maxclust')  # クラスターの数で分割し、クラスター番号を出力
cluster_numbers = pd.DataFrame(cluster_numbers, index=x.index,
                               columns=['cluster_numbers'])  # DataFrame 型に変換。行の名前・列の名前も設定
cluster_numbers.to_csv('cluster_numbers.csv')

# 主成分分析 (Principal Component Analysis, PCA) によるクラスタリング結果の可視化
pca = PCA()
pca.fit(autoscaled_x)
# スコア
score = pd.DataFrame(pca.transform(autoscaled_x), index=x.index)
# 第 1 主成分と第 2 主成分の散布図
plt.rcParams['font.size'] = 18
plt.scatter(score.iloc[:, 0], score.iloc[:, 1], c=cluster_numbers.iloc[:, 0],
            cmap=plt.get_cmap('jet'))  # 散布図の作成。クラスター番号ごとにプロットの色を変えています
plt.xlabel('t_1')
plt.ylabel('t_2')
plt.show()
