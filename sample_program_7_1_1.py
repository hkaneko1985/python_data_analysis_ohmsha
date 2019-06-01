# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

number_of_bins = 20  # ヒストグラムのビンの数

dataset = pd.read_csv('tep_13_train_with_y.csv', index_col=0)
y_measured_dataset = dataset.iloc[0:1, :]
measured_index = [0]
for sample_number in range(1, dataset.shape[0]):
    if y_measured_dataset.iloc[-1, 0] != dataset.iloc[sample_number, 0]:
        y_measured_dataset = pd.concat([y_measured_dataset, dataset.iloc[sample_number:sample_number + 1, :]], axis=0)
        measured_index.append(sample_number)

print('\nすべてのサンプル数: ', dataset.shape[0])
print('y が測定されているサンプル数: ', y_measured_dataset.shape[0])

# y の時間プロット
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.scatter(measured_index, y_measured_dataset.iloc[:, 0], color='blue')  # ヒストグラムの作成
plt.xlabel('time')  # 横軸の名前
plt.ylabel(y_measured_dataset.columns[0])  # 縦軸の名前
plt.xlim([0, dataset.shape[0]])
plt.show()  # 以上の設定において、グラフを描画

# y のヒストグラム
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.hist(y_measured_dataset.iloc[:, 0], bins=number_of_bins)  # ヒストグラムの作成
plt.xlabel(y_measured_dataset.columns[0])  # 横軸の名前
plt.ylabel('frequency')  # 縦軸の名前
plt.show()  # 以上の設定において、グラフを描画

# 相関行列
correlation_coefficients = y_measured_dataset.corr()  # 相関行列の計算
correlation_coefficients.to_csv('correlation_coefficients.csv')  # 相関行列を csv ファイルとして保存
# 相関行列のヒートマップ (相関係数の値なし) 
plt.rcParams['font.size'] = 12
sns.heatmap(correlation_coefficients, vmax=1, vmin=-1, cmap='seismic', square=True, annot=False)
plt.show()

# y との相関係数のヒストグラム
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.hist(correlation_coefficients.iloc[1:, 0], bins=number_of_bins)  # ヒストグラムの作成
plt.xlabel('correlation coef. with {0}'.format(y_measured_dataset.columns[0]))  # 横軸の名前
plt.ylabel('frequency')  # 縦軸の名前
plt.show()  # 以上の設定において、グラフを描画

# y と最も相関係数の絶対値の高い x6 と y の散布図
variable_number_1 = 6
variable_number_2 = 0
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.scatter(y_measured_dataset.iloc[:, variable_number_1], y_measured_dataset.iloc[:, variable_number_2], c='blue')
plt.xlabel(y_measured_dataset.columns[variable_number_1])
plt.ylabel(y_measured_dataset.columns[variable_number_2])
plt.show()
