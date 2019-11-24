# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

number_of_bins = 50  # ヒストグラムのビンの数

dataset = pd.read_csv('unique_m.csv', index_col=-1)
original_x = dataset.iloc[:, :86]
x = original_x.drop(original_x.columns[original_x.std() == 0], axis=1)
x = (x.T / x.T.sum()).T
new_dataset = pd.concat([dataset.iloc[:, 86], x], axis=1)

# 転移温度のヒストグラム
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.hist(new_dataset.iloc[:, 0], bins=number_of_bins)  # ヒストグラムの作成
plt.xlabel(new_dataset.columns[0])  # 横軸の名前
plt.ylabel('frequency')  # 縦軸の名前
plt.show()  # 以上の設定において、グラフを描画

# サンプルごとに用いられている原子の種類数のヒストグラム
numbers_of_atom_types = []
for sample_number in range(x.shape[0]):
    numbers_of_atom_types.append(len(x.iloc[sample_number, :].value_counts()))
plt.hist(numbers_of_atom_types, bins=number_of_bins)  # ヒストグラムの作成
plt.xlabel('number of atoms')  # 横軸の名前
plt.ylabel('frequency')  # 縦軸の名前
plt.show()  # 以上の設定において、グラフを描画

# 相関行列
correlation_coefficients = new_dataset.corr()  # 相関行列の計算
correlation_coefficients.to_csv('correlation_coefficients.csv')  # 相関行列を csv ファイルとして保存
# 相関行列のヒートマップ (相関係数の値なし) 
plt.rcParams['font.size'] = 12
sns.heatmap(correlation_coefficients, vmax=1, vmin=-1, cmap='seismic', square=True, annot=False)
plt.xlim([0, correlation_coefficients.shape[0]])
plt.ylim([0, correlation_coefficients.shape[0]])
plt.show()

# 転移温度との相関係数のヒストグラム
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.hist(correlation_coefficients.iloc[1:, 0], bins=number_of_bins)  # ヒストグラムの作成
plt.xlabel('correlation coef. with {0}'.format(new_dataset.columns[0]))  # 横軸の名前
plt.ylabel('frequency')  # 縦軸の名前
plt.show()  # 以上の設定において、グラフを描画

# 転移温度と最も相関係数の絶対値の高い Cu と転移温度の散布図
variable_number_1 = 26
variable_number_2 = 0
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.scatter(new_dataset.iloc[:, variable_number_1], new_dataset.iloc[:, variable_number_2], c='blue')
plt.xlabel(new_dataset.columns[variable_number_1])
plt.ylabel(new_dataset.columns[variable_number_2])
plt.show()
