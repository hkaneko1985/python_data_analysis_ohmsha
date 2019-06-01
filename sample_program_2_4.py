# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('iris_without_species.csv', index_col=0)

# 以下で散布図を描画します
plt.rcParams['font.size'] = 10  # 横軸や縦軸の名前の文字などのフォントのサイズ
pd.plotting.scatter_matrix(dataset)
plt.show()
