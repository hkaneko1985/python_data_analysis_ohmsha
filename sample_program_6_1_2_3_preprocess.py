# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd
import sample_functions
from sklearn.model_selection import train_test_split

number_of_test_samples = 800

dataset = pd.read_csv('unique_m.csv', index_col=-1)
dataset = dataset.sort_values('critical_temp', ascending=False).iloc[:4000, :]
y = dataset.iloc[:, 86].copy()
x = dataset.iloc[:, :86]
x = (x.T / x.T.sum()).T
# ランダムにトレーニングデータとテストデータとに分割
# random_state に数字を与えることで、別のときに同じ数字を使えば、ランダムとはいえ同じ結果にすることができます
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True,
                                                    random_state=21)
# 標準偏差が 0 の説明変数を削除
std_0_variable_flags = x_train.std() == 0
x_train = x_train.drop(x_train.columns[std_0_variable_flags], axis=1)
x_test = x_test.drop(x_test.columns[std_0_variable_flags], axis=1)
# 説明変数の二乗項や交差項を追加
x_train = sample_functions.add_nonlinear_terms(x_train)
x_test = sample_functions.add_nonlinear_terms(x_test)
# 保存
x_train.to_csv('x_train_superconductor.csv')
x_test.to_csv('x_test_superconductor.csv')
