# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import sample_functions

k_in_k3n_error = 10
candidates_of_perplexity = np.arange(5, 105, 5, dtype=int)

dataset = pd.read_csv('iris_without_species.csv', index_col=0)
autoscaled_dataset = (dataset - dataset.mean()) / dataset.std()  # オートスケーリング

# k3n-error を用いた perplexity の最適化 
k3n_errors = []
for index, perplexity in enumerate(candidates_of_perplexity):
    print(index + 1, '/', len(candidates_of_perplexity))
    t = TSNE(perplexity=perplexity, n_components=2, init='pca', random_state=10).fit_transform(autoscaled_dataset)
    scaled_t = (t - t.mean(axis=0)) / t.std(axis=0, ddof=1)

    k3n_errors.append(
        sample_functions.k3n_error(autoscaled_dataset, scaled_t, k_in_k3n_error) + sample_functions.k3n_error(
            scaled_t, autoscaled_dataset, k_in_k3n_error))
plt.scatter(candidates_of_perplexity, k3n_errors, c='blue')
plt.xlabel("perplexity")
plt.ylabel("k3n-errors")
plt.show()
optimal_perplexity = candidates_of_perplexity[np.where(k3n_errors == np.min(k3n_errors))[0][0]]
print('k3n-error による perplexity の最適値 :', optimal_perplexity)

# t-SNE
t = TSNE(perplexity=optimal_perplexity, n_components=2, init='pca', random_state=10).fit_transform(autoscaled_dataset)
t = pd.DataFrame(t, index=dataset.index, columns=['t_1', 't_2'])  # pandas の DataFrame 型に変換。行の名前・列の名前も設定
t.to_csv('tsne_t.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# t1 と t2 の散布図
plt.rcParams['font.size'] = 18
plt.scatter(t.iloc[:, 0], t.iloc[:, 1], c='blue')
plt.xlabel('t_1')
plt.ylabel('t_2')
plt.show()
