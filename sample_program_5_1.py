# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors  # k-NN

rate_of_training_samples_inside_ad = 0.8  # AD 内となるトレーニングデータの割合。AD　のしきい値を決めるときに使用
k_in_knn = 5  # k-NN における k
number_of_test_samples = 150  # テストデータのサンプル数
dataset = pd.read_csv('boston.csv', index_col=0)

# データ分割
y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数
# ランダムにトレーニングデータとテストデータとに分割
# random_state に数字を与えることで、別のときに同じ数字を使えば、ランダムとはいえ同じ結果にすることができます
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True,
                                                    random_state=99)

# オートスケーリング
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()

# k-NN による AD
ad_model = NearestNeighbors(n_neighbors=k_in_knn, metric='euclidean')  # AD モデルの宣言
ad_model.fit(autoscaled_x_train)  # k-NN による AD では、トレーニングデータの x を model_ad に格納することに対応

# サンプルごとの k 最近傍サンプルとの距離に加えて、k 最近傍サンプルのインデックス番号も一緒に出力されるため、出力用の変数を 2 つに
# トレーニングデータでは k 最近傍サンプルの中に自分も含まれ、自分との距離の 0 を除いた距離を考える必要があるため、k_in_knn + 1 個と設定
knn_distance_train, knn_index_train = ad_model.kneighbors(autoscaled_x_train, n_neighbors=k_in_knn + 1)
knn_distance_train = pd.DataFrame(knn_distance_train, index=x_train.index)  # DataFrame型に変換
mean_of_knn_distance_train = pd.DataFrame(knn_distance_train.iloc[:, 1:].mean(axis=1),
                                          columns=['mean_of_knn_distance'])  # 自分以外の k_in_knn 個の距離の平均
mean_of_knn_distance_train.to_csv('mean_of_knn_distance_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# トレーニングデータのサンプルの rate_of_training_samples_inside_ad * 100 % が含まれるようにしきい値を設定
sorted_mean_of_knn_distance_train = mean_of_knn_distance_train.iloc[:, 0].sort_values(ascending=True)  # 距離の平均の小さい順に並び替え
ad_threshold = sorted_mean_of_knn_distance_train.iloc[
    round(autoscaled_x_train.shape[0] * rate_of_training_samples_inside_ad) - 1]

# トレーニングデータに対して、AD の中か外かを判定
inside_ad_flag_train = mean_of_knn_distance_train <= ad_threshold  # AD 内のサンプルのみ TRUE
inside_ad_flag_train.columns = ['inside_ad_flag']
inside_ad_flag_train.to_csv('inside_ad_flag_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# テストデータに対する k-NN 距離の計算
knn_distance_test, knn_index_test = ad_model.kneighbors(autoscaled_x_test)
knn_distance_test = pd.DataFrame(knn_distance_test, index=x_test.index)  # DataFrame型に変換
mean_of_knn_distance_test = pd.DataFrame(knn_distance_test.mean(axis=1),
                                         columns=['mean_of_knn_distance'])  # k_in_knn 個の距離の平均
mean_of_knn_distance_test.to_csv('mean_of_knn_distance_test.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# テストデータに対して、AD の中か外かを判定
inside_ad_flag_test = mean_of_knn_distance_test <= ad_threshold  # AD 内のサンプルのみ TRUE
inside_ad_flag_test.columns = ['inside_ad_flag']
inside_ad_flag_test.to_csv('inside_ad_flag_test.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
