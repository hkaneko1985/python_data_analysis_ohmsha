# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd
import sample_functions

dataset = pd.read_csv('unique_m.csv', index_col=-1)
dataset = dataset.sort_values('critical_temp', ascending=False).iloc[:4000, :]
x = dataset.iloc[:, :86]
x = (x.T / x.T.sum()).T
# 標準偏差が 0 の説明変数を削除
x = x.drop(x.columns[x.std() == 0], axis=1)
# 説明変数の二乗項や交差項を追加
x = sample_functions.add_nonlinear_terms(x)
# 保存
x.to_csv('x_superconductor.csv')
