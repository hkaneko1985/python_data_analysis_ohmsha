# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import math

import numpy as np
import pandas as pd
import sample_functions
from sklearn.ensemble import RandomForestClassifier  # RF モデルの構築に使用
from sklearn.model_selection import train_test_split

rf_number_of_trees = 300  # RF における決定木の数
rf_x_variables_rates = np.arange(1, 11, dtype=float) / 10  # 1 つの決定木における説明変数の数の割合の候補

fold_number = 5  # N-fold CV の N
number_of_test_samples = 50  # テストデータのサンプル数
dataset = pd.read_csv('iris.csv', index_col=0)

# データ分割
y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数

# ランダムにトレーニングデータとテストデータとに分割
# random_state に数字を与えることで、別のときに同じ数字を使えば、ランダムとはいえ同じ結果にすることができます
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True,
                                                    random_state=21)

# OOB (Out-Of-Bugs) による説明変数の数の割合の最適化
accuracy_oob = []
for index, x_variables_rate in enumerate(rf_x_variables_rates):
    print(index + 1, '/', len(rf_x_variables_rates))
    model_in_validation = RandomForestClassifier(n_estimators=rf_number_of_trees, max_features=int(
        max(math.ceil(x_train.shape[1] * x_variables_rate), 1)), oob_score=True)
    model_in_validation.fit(x_train, y_train)
    accuracy_oob.append(model_in_validation.oob_score_)
optimal_x_variables_rate = sample_functions.plot_and_selection_of_hyperparameter(rf_x_variables_rates,
                                                                                 accuracy_oob,
                                                                                 'rate of x-variables',
                                                                                 'accuracy for OOB')
print('\nOOB で最適化された説明変数の数の割合 :', optimal_x_variables_rate)

# RF
model = RandomForestClassifier(n_estimators=rf_number_of_trees,
                               max_features=int(max(math.ceil(x_train.shape[1] * optimal_x_variables_rate), 1)),
                               oob_score=True)  # RF モデルの宣言
model.fit(x_train, y_train)  # モデルの構築

# 説明変数の重要度
x_importances = pd.DataFrame(model.feature_importances_, index=x_train.columns, columns=['importance'])
x_importances.to_csv('rf_x_importances.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# トレーニングデータ・テストデータの推定、混同行列の作成、正解率の値の表示、推定値の保存
sample_functions.estimation_and_performance_check_in_classification_train_and_test(model, x_train, y_train, x_test,
                                                                                   y_test)
