# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd
import sample_functions
from sklearn import metrics, model_selection
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

max_number_of_k = 20  # 使用する k の最大値
fold_number = 5  # N-fold CV の N
number_of_test_samples = 50  # テストデータのサンプル数
dataset = pd.read_csv('iris.csv', index_col=0)  # あやめのデータの読み込み

# データ分割
y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数

# ランダムにトレーニングデータとテストデータとに分割
# random_state に数字を与えることで、別のときに同じ数字を使えば、ランダムとはいえ同じ結果にすることができます
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True,
                                                    random_state=21, stratify=y)

# オートスケーリング
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()

# CV による k の最適化
accuracy_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の 正解率 をこの変数に追加していきます
ks = []  # 同じく k の値をこの変数に追加していきます
for k in range(1, max_number_of_k + 1):
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')  # k-NN モデルの宣言
    # クロスバリデーション推定値の計算し、DataFrame型に変換
    estimated_y_in_cv = pd.DataFrame(model_selection.cross_val_predict(model, autoscaled_x_train, y_train,
                                                                       cv=fold_number))
    accuracy_in_cv = metrics.accuracy_score(y_train, estimated_y_in_cv)  # 正解率を計算
    print(k, accuracy_in_cv)  # k の値と r2 を表示
    accuracy_in_cv_all.append(accuracy_in_cv)  # r2 を追加
    ks.append(k)  # k の値を追加

# k の値ごとの CV 後の正解率をプロットし、CV 後の正解率が最大のときを k の最適値に
optimal_k = sample_functions.plot_and_selection_of_hyperparameter(ks, accuracy_in_cv_all, 'k',
                                                                  'cross-validated accuracy')
print('\nCV で最適化された k :', optimal_k, '\n')

# k-NN
model = KNeighborsClassifier(n_neighbors=optimal_k, metric='euclidean')  # モデルの宣言
model.fit(autoscaled_x_train, y_train)  # モデルの構築

# トレーニングデータ・テストデータの推定、混同行列の作成、正解率の値の表示、推定値の保存
sample_functions.estimation_and_performance_check_in_classification_train_and_test(model, autoscaled_x_train, y_train,
                                                                                   autoscaled_x_test, y_test)
