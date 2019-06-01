# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('virtual_resin.csv', index_col=0)

# 物性間の散布図
plt.rcParams['font.size'] = 18
plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c='blue')
plt.xlabel(dataset.columns[0])
plt.ylabel(dataset.columns[1])
plt.show()

# 相関行列
correlation_coefficients = dataset.corr()  # 相関行列の計算
correlation_coefficients.to_csv('correlation_coefficients.csv')  # 相関行列を csv ファイルとして保存
# 相関行列のヒートマップ (相関係数の値あり) 
plt.rcParams['font.size'] = 12
sns.heatmap(correlation_coefficients, vmax=1, vmin=-1, cmap='seismic', square=True, annot=True)
plt.show()

# すべての散布図
plt.rcParams['font.size'] = 10  # 横軸や縦軸の名前の文字などのフォントのサイズ
pd.plotting.scatter_matrix(dataset, c='blue')
plt.show()
