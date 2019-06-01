# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('iris_without_species.csv', index_col=0)

# 以下で箱ひげ図を描画します
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
figures, axises = plt.subplots(figsize=(10, dataset.shape[1]))  # 図の調整
axises.boxplot(dataset.values, vert=False)  # 箱ひげ図の作成
axises.set_yticklabels(dataset.columns)  # 縦軸の各箱ひげ図の名前
plt.xlabel('values')  # 横軸の名前
plt.show()  # 以上の設定において、グラフを描画
