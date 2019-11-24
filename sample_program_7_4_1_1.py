# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('tep_0.csv', index_col=0)

# 相関行列
correlation_coefficients = dataset.corr()  # 相関行列の計算
correlation_coefficients.to_csv('correlation_coefficients.csv')  # 相関行列を csv ファイルとして保存
# 相関行列のヒートマップ (相関係数の値なし) 
plt.rcParams['font.size'] = 12
sns.heatmap(correlation_coefficients, vmax=1, vmin=-1, cmap='seismic', square=True, annot=False, xticklabels=1, yticklabels=1)
plt.xlim([0, correlation_coefficients.shape[0]])
plt.ylim([0, correlation_coefficients.shape[0]])
plt.show()

# 最も相関係数の絶対値の高い x7 と x8 の散布図
variable_number_1 = 6
variable_number_2 = 7
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.scatter(dataset.iloc[:, variable_number_1], dataset.iloc[:, variable_number_2], c='blue')
plt.xlabel(dataset.columns[variable_number_1])
plt.ylabel(dataset.columns[variable_number_2])
plt.show()
