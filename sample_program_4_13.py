# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd
import sample_functions
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, cross_val_predict

max_max_depth = 10  # 木の深さの上限、の最大値
min_samples_leaf = 3  # 葉ごとのサンプル数の最小値

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

# クロスバリデーションによる木の深さの最適化
accuracy_cv = []
max_depthes = []
for max_depth in range(1, max_max_depth):
    model_in_cv = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    estimated_y_in_cv = cross_val_predict(model_in_cv, x_train, y_train, cv=fold_number)
    accuracy_cv.append(metrics.accuracy_score(y_train, estimated_y_in_cv))
    max_depthes.append(max_depth)
optimal_max_depth = sample_functions.plot_and_selection_of_hyperparameter(max_depthes, accuracy_cv,
                                                                          'maximum depth of tree', 'accuracy in CV')
print('\nCV で最適化された木の深さ :', optimal_max_depth)

# DT 
model = tree.DecisionTreeClassifier(max_depth=optimal_max_depth, min_samples_leaf=min_samples_leaf)  # モデルの宣言
model.fit(x_train, y_train)  # モデルの構築

# トレーニングデータ・テストデータの推定、混同行列の作成、正解率の値の表示、推定値の保存
sample_functions.estimation_and_performance_check_in_classification_train_and_test(model, x_train, y_train, x_test,
                                                                                   y_test)

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
