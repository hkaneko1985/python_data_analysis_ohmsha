# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd
import numpy as np
import sample_functions
from sklearn import tree, metrics
from sklearn.model_selection import cross_val_predict

max_max_depth = 10  # 木の深さの上限、の最大値
min_samples_leaf = 3  # 葉ごとのサンプル数の最小値

fold_number = 2  # N-fold CV の N
number_of_test_samples = 50  # テストデータのサンプル数
x = pd.read_csv('tep_7.csv', index_col=0)

# OCSVM モデルで検出された異常サンプルを positive として目的変数を設定
x = x.iloc[:177, :]
y = np.full(x.shape[0], 'negative')
y[167:] = 'positive'

# クロスバリデーションによる木の深さの最適化
accuracy_cv = []
max_depthes = []
for max_depth in range(1, max_max_depth):
    model_in_cv = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    estimated_y_in_cv = cross_val_predict(model_in_cv, x, y, cv=fold_number)
    accuracy_cv.append(metrics.accuracy_score(y, estimated_y_in_cv))
    max_depthes.append(max_depth)
optimal_max_depth = sample_functions.plot_and_selection_of_hyperparameter(max_depthes, accuracy_cv,
                                                                          'maximum depth of tree', 'accuracy in CV')
print('\nCV で最適化された木の深さ :', optimal_max_depth)

# DT 
model = tree.DecisionTreeClassifier(max_depth=optimal_max_depth, min_samples_leaf=min_samples_leaf)  # モデルの宣言
model.fit(x, y)  # モデルの構築

# トレーニングデータのクラスの推定
estimated_y_train = pd.DataFrame(model.predict(x), index=x.index, columns=[
    'estimated_class'])  # トレーニングデータのクラスを推定し、Pandas の DataFrame 型に変換。行の名前・列の名前も設定
estimated_y_train.to_csv('estimated_y_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# トレーニングデータの混同行列
class_types = list(set(y))  # クラスの種類。これで混同行列における縦と横のクラスの順番を定めます
class_types.sort(reverse=True)  # 並び替え
confusion_matrix_train = pd.DataFrame(
    metrics.confusion_matrix(y, estimated_y_train, labels=class_types), index=class_types,
    columns=class_types)  # 混同行列を作成し、Pandas の DataFrame 型に変換。行の名前・列の名前を定めたクラスの名前として設定
confusion_matrix_train.to_csv('confusion_matrix_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
print(confusion_matrix_train)  # 混同行列の表示
print('Accuracy for training data :', metrics.accuracy_score(y, estimated_y_train), '\n')  # 正解率の表示


# 決定木のモデルを確認するための dot ファイルの作成
with open('tree.dot', 'w') as f:
    if model.classes_.dtype == 'object':
        class_names = model.classes_
    else:
        # クラス名が数値のときの対応
        class_names = []
        for class_name_number in range(0, model.classes_.shape[0]):
            class_names.append(str(model.classes_[class_name_number]))
    tree.export_graphviz(model, out_file=f, feature_names=x.columns, class_names=class_names)
