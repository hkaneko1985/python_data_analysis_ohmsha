# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import pandas as pd

variable_number_1 = 0  # 散布図における横軸の特徴量の番号 (0 から始まるため注意)
variable_number_2 = 1  # 散布図における縦軸の特徴量の番号

color_list = ['k', 'r', 'b', 'g', 'y', 'c', 'm']

dataset = pd.read_csv('iris.csv', index_col=0)
iris_types = dataset.iloc[:, 0] # あやめの種類
x = dataset.iloc[:, 1:]  # 数値の特徴量のみのデータセット

# 以下で散布図を描画します
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
for index, sample_class in enumerate(set(iris_types)):
    x_class = x[iris_types == sample_class]
    plt.scatter(x_class.iloc[:, variable_number_1], x_class.iloc[:, variable_number_2], c=color_list[index], label=sample_class)  # 散布図の作成

plt.xlabel(x.columns[variable_number_1])  # 横軸の名前。ここでは、variable_number_1 番目の列の名前
plt.ylabel(x.columns[variable_number_2])  # 縦軸の名前。ここでは、variable_number_2 番目の列の名前
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
plt.show()  # 以上の設定において、グラフを描画
