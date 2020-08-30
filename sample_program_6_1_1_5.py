# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import sys

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sample_functions
from sklearn import metrics
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.neighbors import NearestNeighbors

method_name = 'pls'  # 'pls' or 'svr'
add_nonlinear_terms_flag = True  # True (二乗項・交差項を追加) or False (追加しない)
rate_of_training_samples_inside_ad = 0.68  # AD 内となるトレーニングデータの割合。AD　のしきい値を決めるときに使用
number_of_y = 2  # 目的変数の数

k_in_knn = 1  # k-NN における k
fold_number = 10  # N-fold CV の N
max_number_of_principal_components = 20  # 使用する主成分の最大数
svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # C の候補
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # ε の候補
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補

if method_name != 'pls' and method_name != 'svr':
    sys.exit('\'{0}\' という回帰分析手法はありません。method_name を見直してください。'.format(method_name))
    
dataset = pd.read_csv('virtual_resin.csv', index_col=0)
ys = dataset.iloc[:, 0:number_of_y]  # 目的変数
if add_nonlinear_terms_flag:
    x_tmp = sample_functions.add_nonlinear_terms(dataset.iloc[:, number_of_y:])  # 説明変数の二乗項や交差項を追加
    x = x_tmp.drop(x_tmp.columns[x_tmp.std() == 0], axis=1)  # 標準偏差が 0 の説明変数を削除
else:
    x = dataset.iloc[:, number_of_y:]  # 説明変数
autoscaled_x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)  # オートスケーリング

models = []  # ここに y ごとの回帰モデルを格納
for y_number in range(number_of_y):
    y = ys.iloc[:, y_number]
    autoscaled_y = (y - y.mean()) / y.std(ddof=1)  # オートスケーリング
    if method_name == 'pls':
        # CV による成分数の最適化
        components = []  # 空の list の変数を作成して、成分数をこの変数に追加していきます同じく成分数をこの変数に追加
        r2_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の r2 をこの変数に追加
        for component in range(1, min(np.linalg.matrix_rank(autoscaled_x), max_number_of_principal_components) + 1):
            # PLS
            model = PLSRegression(n_components=component)  # PLS モデルの宣言
            estimated_y_in_cv = pd.DataFrame(cross_val_predict(model, autoscaled_x, autoscaled_y,
                                                               cv=fold_number))  # クロスバリデーション推定値の計算し、DataFrame型に変換
            estimated_y_in_cv = estimated_y_in_cv * y.std() + y.mean()  # スケールをもとに戻す
            r2_in_cv = metrics.r2_score(y, estimated_y_in_cv)  # r2 を計算
            print(component, r2_in_cv)  # 成分数と r2 を表示
            r2_in_cv_all.append(r2_in_cv)  # r2 を追加
            components.append(component)  # 成分数を追加

        # 成分数ごとの CV 後の r2 をプロットし、CV 後のr2が最大のときを最適成分数に
        optimal_component_number = sample_functions.plot_and_selection_of_hyperparameter(components, r2_in_cv_all,
                                                                                         'number of components',
                                                                                         'cross-validated r2')
        print('\nCV で最適化された成分数 :', optimal_component_number)
        # PLS
        model = PLSRegression(n_components=optimal_component_number)  # モデルの宣言
    elif method_name == 'svr':
        # グラム行列の分散を最大化することによる γ の最適化
        optimal_svr_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x, svr_gammas)
        # CV による ε の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
                                   cv=fold_number, iid=False)
        model_in_cv.fit(autoscaled_x, autoscaled_y)
        optimal_svr_epsilon = model_in_cv.best_params_['epsilon']
        # CV による C の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                                   {'C': svr_cs}, cv=fold_number, iid=False)
        model_in_cv.fit(autoscaled_x, autoscaled_y)
        optimal_svr_c = model_in_cv.best_params_['C']
        # CV による γ の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                                   {'gamma': svr_gammas}, cv=fold_number, iid=False)
        model_in_cv.fit(autoscaled_x, autoscaled_y)
        optimal_svr_gamma = model_in_cv.best_params_['gamma']
        # 最適化された C, ε, γ
        print('C : {0}\nε : {1}\nGamma : {2}'.format(optimal_svr_c, optimal_svr_epsilon, optimal_svr_gamma))
        # SVR
        model = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)  # モデルの宣言

    model.fit(autoscaled_x, autoscaled_y)  # モデルの構築
    models.append(model)
    if method_name == 'pls':
        # 標準回帰係数
        standard_regression_coefficients = pd.DataFrame(model.coef_)  # Pandas の DataFrame 型に変換
        standard_regression_coefficients.index = x.columns  # 説明変数に対応する名前を、元のデータセットにおける説明変数の名前に
        standard_regression_coefficients.columns = ['standard_regression_coefficients']  # 列名を変更
        standard_regression_coefficients.to_csv(
            'pls_standard_regression_coefficients_y{0}.csv'.format(
                y_number))  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# k-NN による AD
ad_model = NearestNeighbors(n_neighbors=k_in_knn, metric='euclidean')  # AD モデルの宣言
ad_model.fit(autoscaled_x)  # k-NN による AD では、トレーニングデータの x を model_ad に格納することに対応
# サンプルごとの k 最近傍サンプルとの距離に加えて、k 最近傍サンプルのインデックス番号も一緒に出力されるため、出力用の変数を 2 つに
# トレーニングデータでは k 最近傍サンプルの中に自分も含まれ、自分との距離の 0 を除いた距離を考える必要があるため、k_in_knn + 1 個と設定
knn_distance_train, knn_index_train = ad_model.kneighbors(autoscaled_x, n_neighbors=k_in_knn + 1)
knn_distance_train = pd.DataFrame(knn_distance_train)  # DataFrame型に変換
knn_distance_train.index = x.index  # サンプル名をトレーニングデータのサンプル名に
mean_of_knn_distance_train = pd.DataFrame(knn_distance_train.iloc[:, 1:].mean(axis=1))  # 自分以外の k_in_knn 個の距離の平均。Series 型のため、列名の設定は別に
mean_of_knn_distance_train.columns = ['mean_of_knn_distance']  # 列名を変更
# トレーニングデータのサンプルの rate_of_training_samples_inside_ad * 100 % が含まれるようにしきい値を設定
sorted_mean_of_knn_distance_train = mean_of_knn_distance_train.iloc[:, 0].sort_values(ascending=True)  # 距離の平均の小さい順に並び替え
ad_threshold = sorted_mean_of_knn_distance_train.iloc[
    round(autoscaled_x.shape[0] * rate_of_training_samples_inside_ad) - 1]

# 新しいサンプルにおける予測
dataset_prediction = pd.read_csv('virtual_resin_prediction.csv', encoding='SHIFT-JIS', index_col=0)
if add_nonlinear_terms_flag:
    x_prediction_tmp = sample_functions.add_nonlinear_terms(dataset_prediction) # 説明変数の二乗項や交差項を追加
    x_prediction = x_prediction_tmp.drop(x_tmp.columns[x_tmp.std() == 0], axis=1)  # 標準偏差が 0 の説明変数を削除
else:
    x_prediction = dataset_prediction.copy()  # 説明変数
# オートスケーリング
autoscaled_x_prediction = (x_prediction - x.mean(axis=0)) / x.std(axis=0, ddof=1)
# y ごとに予測
predicted_ys = pd.DataFrame()
for y_number in range(number_of_y):
    predicted_y = pd.DataFrame(models[y_number].predict(autoscaled_x_prediction))
    predicted_ys = pd.concat([predicted_ys, predicted_y], axis=1)
predicted_ys.index = x_prediction.index
predicted_ys.columns = ys.columns
predicted_ys.to_csv('predicted_ys.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# 予測用データに対して、AD の中か外かを判定
knn_distance_prediction, knn_index_prediction = ad_model.kneighbors(autoscaled_x_prediction, n_neighbors=k_in_knn)
knn_distance_prediction = pd.DataFrame(knn_distance_prediction, index=x_prediction.index)
knn_distance_prediction.index = x_prediction.index  # サンプル名をトレーニングデータのサンプル名に
mean_of_knn_distance_prediction = pd.DataFrame(knn_distance_prediction.mean(axis=1))  # k_in_knn 個の距離の平均
inside_ad_flag_prediction = mean_of_knn_distance_prediction <= ad_threshold  # AD 内のサンプルのみ TRUE
inside_ad_flag_prediction.columns = ['inside_ad_flag']  # 列名を変更
inside_ad_flag_prediction.to_csv('inside_ad_flag_prediction.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# パレート最適解の探索
print('\nパレート最適解の探索')
predicted_ys_inside_ad = predicted_ys.iloc[inside_ad_flag_prediction.values[:, 0], :]
dataset_prediction_inside_ad = dataset_prediction.iloc[inside_ad_flag_prediction.values[:, 0], :]
pareto_optimal_index = []
for sample_number in range(predicted_ys_inside_ad.shape[0]):
    if (sample_number + 1) % 500 == 0:
        print(sample_number + 1, '/', predicted_ys_inside_ad.shape[0])
    flag = predicted_ys_inside_ad <= predicted_ys_inside_ad.iloc[sample_number, :]
    if flag.any(axis=1).all():
        pareto_optimal_index.append(sample_number)
samples_inside_ad = pd.concat([predicted_ys_inside_ad, dataset_prediction_inside_ad], axis=1)
pareto_optimal_samples = samples_inside_ad.iloc[pareto_optimal_index, :]
pareto_optimal_samples.to_csv('pareto_optimal_samples_prediction.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# 予測された y の散布図
plt.rcParams['font.size'] = 18
plt.figure(figsize=figure.figaspect(1))
plt.scatter(predicted_ys.iloc[:, 0], predicted_ys.iloc[:, 1], color='0.75', label='samples outside AD in prediction')
plt.scatter(predicted_ys.iloc[inside_ad_flag_prediction.values[:, 0], 0],
            predicted_ys.iloc[inside_ad_flag_prediction.values[:, 0], 1], color='black',
            label='samples inside AD in prediction')
plt.scatter(predicted_ys_inside_ad.iloc[pareto_optimal_index, 0],
            predicted_ys_inside_ad.iloc[pareto_optimal_index, 1], color='blue',
            label='Pareto optimum samples in prediction')
plt.scatter(ys.iloc[:, 0], ys.iloc[:, 1], color='red', label='training samples')
plt.xlabel(ys.columns[0])
plt.ylabel(ys.columns[1])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
plt.show()
