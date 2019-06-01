# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE  # scikit-learn の中の t-SNE を実行するためのライブラリのインポート

perplexity = 30  # perplexity (基本的には 5 から 50 の間)

dataset = pd.read_csv('iris_without_species.csv', index_col=0)
autoscaled_dataset = (dataset - dataset.mean()) / dataset.std()  # オートスケーリング

# t-SNE
t = TSNE(perplexity=perplexity, n_components=2, init='pca', random_state=0).fit_transform(autoscaled_dataset)
t = pd.DataFrame(t, index=dataset.index, columns=['t_1', 't_2'])  # pandas の DataFrame 型に変換。行の名前・列の名前も設定
t.to_csv('tsne_t.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# t1 と t2 の散布図
plt.rcParams['font.size'] = 18
plt.scatter(t.iloc[:, 0], t.iloc[:, 1], c='blue')
plt.xlabel('t_1')
plt.ylabel('t_2')
plt.show()
