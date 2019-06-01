# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt  # matplotlib.pyplot の取り込み。一般的に plt と名前を省略して取り込みます
import pandas as pd

variable_number = 0  # ヒストグラムを描画する特徴量の番号。0 から始まるため注意
number_of_bins = 20  # ヒストグラムのビンの数

dataset = pd.read_csv('iris_without_species.csv', index_col=0)

# 以下でヒストグラムを描画します
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.hist(dataset.iloc[:, variable_number], bins=number_of_bins)  # ヒストグラムの作成
plt.xlabel(dataset.columns[variable_number])  # 横軸の名前。ここでは、variable_number 番目の列の名前
plt.ylabel('frequency')  # 縦軸の名前
plt.show()  # 以上の設定において、グラフを描画
