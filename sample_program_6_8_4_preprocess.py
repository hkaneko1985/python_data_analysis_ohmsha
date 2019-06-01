# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import numpy as np
import pandas as pd
import sample_functions
from sklearn.model_selection import train_test_split

# 下の y_name を、'boiling_point', 'logS', 'melting_point', 'pIC50', 'pIC50_class', 'pIGC50', 'pIGC50_class' のいずれかにしてください。
# descriptors_with_[y_name].csv というファイルを dataset として読み込み計算します。
# さらに、y_name を別の名前に変えて、ご自身で別途 sample_program_6_8_0_csv.py もしくは
# sample_program_6_8_0_sdf.py で descriptors_with_[y_name].csv というファイルを、
# 他のファイルと同様の形式で準備すれば、同じように計算することができます。

y_name = 'boiling_point'
# 'boiling_point' : 沸点のデータセットの場合
# 'logS' : 水溶解度のデータセットの場合
# 'melting_point' : 融点のデータセットの場合
# 'pIC50' : 薬理活性のデータセットの場合
# 'pIC50_class' : クラス分類用の薬理活性のデータセットの場合
# 'pIGC50' : 環境毒性のデータセットの場合
# 'pIGC50_class' : クラス分類用の環境毒性のデータセットの場合

rate_of_test_samples = 0.25 # テストデータのサンプル数の割合

dataset = pd.read_csv('descriptors_with_{0}.csv'.format(y_name), index_col=0)  # 物性・活性と記述子のデータセットの読み込み
dataset = dataset.replace(np.inf, np.nan).fillna(np.nan)  # inf を NaN に置き換え
nan_variable_flags = dataset.isnull().any()  # NaN を含む変数
dataset = dataset.drop(dataset.columns[nan_variable_flags], axis=1)  # NaN を含む変数を削除
number_of_test_samples = round(dataset.shape[0] * rate_of_test_samples)
y = dataset.iloc[:, 0].copy()
x = dataset.iloc[:, 1:]
# ランダムにトレーニングデータとテストデータとに分割
# random_state に数字を与えることで、別のときに同じ数字を使えば、ランダムとはいえ同じ結果にすることができます
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True, random_state=0)
# 標準偏差が 0 の説明変数を削除
std_0_variable_flags = x_train.std() == 0
x_train = x_train.drop(x_train.columns[std_0_variable_flags], axis=1)
x_test = x_test.drop(x_test.columns[std_0_variable_flags], axis=1)
# 説明変数の二乗項や交差項を追加
x_train = sample_functions.add_nonlinear_terms(x_train)
x_test = sample_functions.add_nonlinear_terms(x_test)
# 保存
x_train.to_csv('x_train_{0}.csv'.format(y_name))
x_test.to_csv('x_test_{0}.csv'.format(y_name))
