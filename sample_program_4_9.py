# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

k_in_knn = 5  # k-NN における k の値
fold_number = 5  # N-fold CV の N
number_of_test_samples = 50  # テストデータのサンプル数
dataset = pd.read_csv('iris.csv', index_col=0)  # あやめのデータの読み込み

# データ分割
y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数

# ランダムにトレーニングデータとテストデータとに分割
# random_state に数字を与えることで、別のときに同じ数字を使えば、ランダムとはいえ同じ結果にすることができます
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True,
                                                    random_state=9)

# オートスケーリング
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()

# k-NN
model = KNeighborsClassifier(n_neighbors=k_in_knn, metric='euclidean')  # モデルの宣言
model.fit(autoscaled_x_train, y_train)  # モデルの構築

# テストデータのクラスの推定
estimated_y_test = pd.DataFrame(model.predict(autoscaled_x_test), index=x_test.index,
                                columns=['estimated_class'])  # テストデータのクラスを推定し、Pandas の DataFrame 型に変換。行の名前・列の名前も設定
y_test_for_save = pd.DataFrame(y_test)  # Series のため列名は別途変更
y_test_for_save.columns = ['actual_class']
y_error_test = y_test_for_save.iloc[:, 0] == estimated_y_test.iloc[:, 0]
y_error_test = pd.DataFrame(y_error_test)  # Series のため列名は別途変更
y_error_test.columns = ['TRUE_if_estimated_class_is_correct']
results_test = pd.concat([estimated_y_test, y_test_for_save, y_error_test], axis=1)
results_test.to_csv('estimated_y_test.csv')  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
