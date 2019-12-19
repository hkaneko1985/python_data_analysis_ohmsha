# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import math

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sample_functions
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.neighbors import NearestNeighbors

method_name = 'pls'  # ソフトセンサーの種類は以下の通りです
# 'pls' : 最初のトレーニングデータで構築した PLS モデル (適応的ソフトセンサーではありません)
# 'svr' : 最初のトレーニングデータで構築した SVR モデル (適応的ソフトセンサーではありません)
# 'mwpls' : Moving Window PLS
# 'mvsvr' : Moving Window SVR
# 'jitpls' : Just-In-Time PLS
# 'jitsvr' : Just-In-Time SVR
# 'lwpls' : Locally-Weighted PLS

number_of_samples_in_modeling = 100  # MW や JIT モデルにおけるモデル構築用サンプルの数 
max_sample_size = 10000  # データベースにおける最大のサンプル数
y_measurement_delay = 5  # y の測定時間の遅れ
dynamics_max = 0  # いくつまで時間遅れ変数を追加するか。0 なら時間遅れ変数は追加されません
dynamics_span = 2  # いくつずつ時間を遅らせた変数を追加するか
add_nonlinear_terms_flag = False  # True (二乗項・交差項を追加) or False (追加しない)

fold_number = 5  # N-fold CV の N
max_number_of_principal_components = 20  # 使用する主成分の最大数
svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # C の候補
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # ε の候補
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補
lwpls_lambdas = 2 ** np.arange(-9, 6, dtype=float)

# データセットの読み込み
dataset_train = pd.read_csv('tep_13_train_with_y.csv', index_col=0)
dataset_test = pd.read_csv('tep_13_test_with_y.csv', index_col=0)
y_train = dataset_train.iloc[:, 0:1]
x_train_tmp = dataset_train.iloc[:, 1:]
y_test = dataset_test.iloc[:, 0:1]
x_test_tmp = dataset_test.iloc[:, 1:]

# 説明変数の非線形変換
if add_nonlinear_terms_flag:
    x_train_tmp = sample_functions.add_nonlinear_terms(x_train_tmp)  # 説明変数の二乗項や交差項を追加
    x_test_tmp = sample_functions.add_nonlinear_terms(x_test_tmp)
    x_train = x_train_tmp.drop(x_train_tmp.columns[x_train_tmp.std() == 0], axis=1)  # 標準偏差が 0 の説明変数を削除
    x_test = x_test_tmp.drop(x_train_tmp.columns[x_train_tmp.std() == 0], axis=1)
else:
    x_train = x_train_tmp.copy()
    x_test = x_test_tmp.copy()

# 時間遅れ変数の追加
dataset_train = pd.concat([y_train, x_train], axis=1)
dataset_train_with_dynamics = sample_functions.add_time_delayed_variable(dataset_train, dynamics_max, dynamics_span)
dataset_test = pd.concat([y_test, x_test], axis=1)
dataset_test_with_dynamics = sample_functions.add_time_delayed_variable(dataset_test, dynamics_max, dynamics_span)

# トレーニングデータでは y が測定されたサンプルのみ収集
y_measured_dataset_train = dataset_train_with_dynamics[0:1, :]
for sample_number in range(1, dataset_train_with_dynamics.shape[0]):
    if y_measured_dataset_train[-1, 0] != dataset_train_with_dynamics[sample_number, 0]:
        y_measured_dataset_train = np.r_[
            y_measured_dataset_train, dataset_train_with_dynamics[sample_number:sample_number + 1, :]]
y_measured_dataset_train = np.delete(y_measured_dataset_train, 0, 0)
y_measured_dataset_train = pd.DataFrame(y_measured_dataset_train)
y_train = y_measured_dataset_train.iloc[:, 0]
x_train = y_measured_dataset_train.iloc[:, 1:]
# テストデータ
dataset_test_with_dynamics = pd.DataFrame(dataset_test_with_dynamics)
y_test_all = dataset_test_with_dynamics.iloc[:, 0]
x_test = dataset_test_with_dynamics.iloc[:, 1:]

# サンプルの調整
if method_name[0:2] == 'mw':
    y_train = y_train.iloc[-number_of_samples_in_modeling:]
    x_train = x_train.iloc[-number_of_samples_in_modeling:, :]
else:
    y_train = y_train.iloc[-max_sample_size:]
    x_train = x_train.iloc[-max_sample_size:, :]
    if method_name[0:3] == 'jit':
        nn_model = NearestNeighbors(metric='euclidean')  # サンプル選択用の k-NN モデルの宣言

# オートスケーリング    
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()

# ハイパーパラメータの最適化やモデリング
if method_name == 'pls' or method_name == 'mwpls':
    # CV による成分数の最適化
    components = []  # 空の list の変数を作成して、成分数をこの変数に追加していきます同じく成分数をこの変数に追加
    r2_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の r2 をこの変数に追加
    for component in range(1, min(np.linalg.matrix_rank(autoscaled_x_train), max_number_of_principal_components) + 1):
        # PLS
        model = PLSRegression(n_components=component)  # PLS モデルの宣言
        estimated_y_in_cv = pd.DataFrame(cross_val_predict(model, autoscaled_x_train, autoscaled_y_train,
                                                           cv=fold_number))  # クロスバリデーション推定値の計算し、DataFrame型に変換
        estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()  # スケールをもとに戻す
        r2_in_cv = metrics.r2_score(y_train, estimated_y_in_cv)  # r2 を計算
        print(component, r2_in_cv)  # 成分数と r2 を表示
        r2_in_cv_all.append(r2_in_cv)  # r2 を追加
        components.append(component)  # 成分数を追加
    optimal_component_number = components[r2_in_cv_all.index(max(r2_in_cv_all))]  # 最適成分数
    # PLS
    model = PLSRegression(n_components=optimal_component_number)  # モデルの宣言
    model.fit(autoscaled_x_train, autoscaled_y_train)  # モデルの構築
elif method_name == 'svr' or method_name == 'mwsvr' or method_name == 'jitsvr':
    # グラム行列の分散を最大化することによる γ の最適化
    variance_of_gram_matrix = list()
    for svr_gamma in svr_gammas:
        gram_matrix = np.exp(
            -svr_gamma * cdist(autoscaled_x_train, autoscaled_x_train, metric='seuclidean'))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    optimal_svr_gamma = svr_gammas[np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
    # CV による ε の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
                               cv=fold_number, iid=False)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_svr_epsilon = model_in_cv.best_params_['epsilon']
    # CV による C の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                               {'C': svr_cs}, cv=fold_number, iid=False)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_svr_c = model_in_cv.best_params_['C']
    # CV による γ の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                               {'gamma': svr_gammas}, cv=fold_number, iid=False)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_svr_gamma = model_in_cv.best_params_['gamma']
    # 最適化された C, ε, γ
    print('C : {0}\nε : {1}\nGamma : {2}'.format(optimal_svr_c, optimal_svr_epsilon, optimal_svr_gamma))
    # SVR
    model = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)  # モデルの宣言
    model.fit(autoscaled_x_train, autoscaled_y_train)  # モデルの構築
elif method_name == 'lwpls':
    # CV によるハイパーパラメータの最適化
    autoscaled_x_train = np.array(autoscaled_x_train)
    autoscaled_y_train = np.array(autoscaled_y_train)
    y_train_array = np.array(y_train)
    r2cvs = np.empty(
        (min(np.linalg.matrix_rank(autoscaled_x_train), max_number_of_principal_components), len(lwpls_lambdas)))
    min_number = math.floor(x_train.shape[0] / fold_number)
    mod_numbers = x_train.shape[0] - min_number * fold_number
    index = np.matlib.repmat(np.arange(1, fold_number + 1, 1), 1, min_number).ravel()
    if mod_numbers != 0:
        index = np.r_[index, np.arange(1, mod_numbers + 1, 1)]
    indexes_for_division_in_cv = np.random.permutation(index)
    np.random.seed()
    for parameter_number, lambda_in_similarity in enumerate(lwpls_lambdas):
        estimated_y_in_cv = np.empty((len(y_train_array), r2cvs.shape[0]))
        for fold in np.arange(1, fold_number + 1, 1):
            autoscaled_x_train_in_cv = autoscaled_x_train[indexes_for_division_in_cv != fold, :]
            autoscaled_y_train_in_cv = autoscaled_y_train[indexes_for_division_in_cv != fold]
            autoscaled_x_validation_in_cv = autoscaled_x_train[indexes_for_division_in_cv == fold, :]

            estimated_y_validation_in_cv = sample_functions.lwpls(autoscaled_x_train_in_cv, autoscaled_y_train_in_cv,
                                                                  autoscaled_x_validation_in_cv, r2cvs.shape[0],
                                                                  lambda_in_similarity)
            estimated_y_in_cv[indexes_for_division_in_cv == fold, :] = estimated_y_validation_in_cv * y_train_array.std(
                ddof=1) + y_train_array.mean()

        estimated_y_in_cv[np.isnan(estimated_y_in_cv)] = 99999
        ss = (y_train_array - y_train_array.mean()).T.dot(y_train_array - y_train_array.mean())
        press = np.diag(
            (np.matlib.repmat(y_train_array.reshape(len(y_train_array), 1), 1,
                              estimated_y_in_cv.shape[1]) - estimated_y_in_cv).T.dot(
                np.matlib.repmat(y_train_array.reshape(len(y_train_array), 1), 1,
                                 estimated_y_in_cv.shape[1]) - estimated_y_in_cv))
        r2cvs[:, parameter_number] = 1 - press / ss

    best_candidate_number = np.where(r2cvs == r2cvs.max())

    optimal_component_number = best_candidate_number[0][0] + 1
    optimal_lambda_in_similarity = lwpls_lambdas[best_candidate_number[1][0]]

# y の推定やモデリング
if method_name == 'pls' or method_name == 'svr':
    estimated_y_test_all = model.predict(autoscaled_x_test) * y_train.std() + y_train.mean()
else:
    estimated_y_test_all = np.zeros((len(y_test_all)))
    for test_sample_number in range(len(y_test_all)):
        print(test_sample_number + 1, '/', len(y_test_all))
        autoscaled_x_test = (x_test.iloc[
                             test_sample_number:test_sample_number + 1, ] - x_train.mean()) / x_train.std()  # オートスケーリング
        # y の推定
        if method_name[0:2] == 'mw':
            autoscaled_estimated_y_test_tmp = model.predict(autoscaled_x_test)
        elif method_name == 'lwpls':
            autoscaled_estimated_y_test_tmp = sample_functions.lwpls(autoscaled_x_train, autoscaled_y_train,
                                                                     autoscaled_x_test, optimal_component_number,
                                                                     optimal_lambda_in_similarity)
            autoscaled_estimated_y_test_tmp = autoscaled_estimated_y_test_tmp[:, optimal_component_number - 1]
        elif method_name[0:3] == 'jit':
            # サンプル選択
            nn_model.fit(autoscaled_x_train)
            tmp, nn_index_test = nn_model.kneighbors(autoscaled_x_test,
                                                     n_neighbors=min(number_of_samples_in_modeling, x_train.shape[0]))
            x_train_jit = x_train.iloc[nn_index_test[0, :], :]
            y_train_jit = y_train.iloc[nn_index_test[0, :]]
            # オートスケーリング    
            autoscaled_x_train_jit = (x_train_jit - x_train_jit.mean()) / x_train_jit.std()
            autoscaled_y_train_jit = (y_train_jit - y_train_jit.mean()) / y_train_jit.std()
            autoscaled_x_test_jit = (x_test.iloc[
                                     test_sample_number:test_sample_number + 1, ] - x_train_jit.mean()) / x_train_jit.std()
            # ハイパーパラメータの最適化
            if method_name == 'jitpls':
                # CV による成分数の最適化
                components = []  # 空の list の変数を作成して、成分数をこの変数に追加していきます同じく成分数をこの変数に追加
                r2_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の r2 をこの変数に追加
                for component in range(1, min(np.linalg.matrix_rank(autoscaled_x_train_jit),
                                              max_number_of_principal_components) + 1):
                    model = PLSRegression(n_components=component)  # PLS モデルの宣言
                    estimated_y_in_cv = pd.DataFrame(
                        cross_val_predict(model, autoscaled_x_train_jit, autoscaled_y_train_jit,
                                          cv=fold_number))  # クロスバリデーション推定値の計算し、DataFrame型に変換
                    estimated_y_in_cv = estimated_y_in_cv * y_train_jit.std() + y_train_jit.mean()  # スケールをもとに戻す
                    r2_in_cv = metrics.r2_score(y_train_jit, estimated_y_in_cv)  # r2 を計算
                    r2_in_cv_all.append(r2_in_cv)  # r2 を追加
                    components.append(component)  # 成分数を追加
                optimal_component_number = components[r2_in_cv_all.index(max(r2_in_cv_all))]  # 最適成分数
                # PLS
                model = PLSRegression(n_components=optimal_component_number)  # モデルの宣言
            # モデリング
            model.fit(autoscaled_x_train_jit, autoscaled_y_train_jit)  # モデルの構築
            autoscaled_estimated_y_test_tmp = model.predict(autoscaled_x_test_jit)  # 推定
        if np.isnan(autoscaled_estimated_y_test_tmp):
            if test_sample_number == 0:
                estimated_y_test_all[test_sample_number] = 0
            else:
                estimated_y_test_all[test_sample_number] = estimated_y_test_all[test_sample_number - 1]
        else:
            estimated_y_test_all[test_sample_number] = autoscaled_estimated_y_test_tmp * y_train.std() + y_train.mean()

        if test_sample_number - y_measurement_delay >= 1:
            if y_test_all[test_sample_number - y_measurement_delay] - y_test_all[
                test_sample_number - y_measurement_delay - 1] != 0:
                x_train = pd.concat([x_train, x_test.iloc[
                                              test_sample_number - y_measurement_delay:test_sample_number - y_measurement_delay + 1, ]],
                                    axis=0)
                y_train = pd.concat([y_train, y_test_all.iloc[test_sample_number - y_measurement_delay:test_sample_number - y_measurement_delay + 1]])
                # サンプルの調整
                if method_name[0:2] == 'mw':
                    y_train = y_train.iloc[0:number_of_samples_in_modeling]
                    x_train = x_train.iloc[0:number_of_samples_in_modeling, :]
                else:
                    y_train = y_train.iloc[0:max_sample_size]
                    x_train = x_train.iloc[0:max_sample_size, :]
                # オートスケーリング    
                autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
                autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
                # ハイパーパラメータの最適化
                if method_name == 'mwpls':
                    # CV による成分数の最適化
                    components = []  # 空の list の変数を作成して、成分数をこの変数に追加していきます同じく成分数をこの変数に追加
                    r2_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の r2 をこの変数に追加
                    for component in range(1, min(np.linalg.matrix_rank(autoscaled_x_train),
                                                  max_number_of_principal_components) + 1):
                        model = PLSRegression(n_components=component)  # PLS モデルの宣言
                        estimated_y_in_cv = pd.DataFrame(
                            cross_val_predict(model, autoscaled_x_train, autoscaled_y_train,
                                              cv=fold_number))  # クロスバリデーション推定値の計算し、DataFrame型に変換
                        estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()  # スケールをもとに戻す
                        r2_in_cv = metrics.r2_score(y_train, estimated_y_in_cv)  # r2 を計算
                        r2_in_cv_all.append(r2_in_cv)  # r2 を追加
                        components.append(component)  # 成分数を追加
                    optimal_component_number = components[r2_in_cv_all.index(max(r2_in_cv_all))]  # 最適成分数
                    # PLS
                    model = PLSRegression(n_components=optimal_component_number)  # モデルの宣言
                # モデリング
                if method_name[0:2] == 'mw':
                    model.fit(autoscaled_x_train, autoscaled_y_train)  # モデルの構築

estimated_y_test_all = pd.DataFrame(estimated_y_test_all, index=x_test.index, columns=['estimated_y'])
estimated_y_test_all.to_csv('estimated_y_test.csv')  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# テストデータの y の実測値と推定値において y が測定されたサンプルのみ収集
ys_test = pd.concat([y_test_all, estimated_y_test_all], axis=1)
y_measured_ys_test = ys_test.iloc[0:1, :]
measured_index_test = []
for sample_number in range(1, ys_test.shape[0]):
    if y_measured_ys_test.iloc[-1, 0] != ys_test.iloc[sample_number, 0]:
        y_measured_ys_test = pd.concat([y_measured_ys_test, ys_test.iloc[sample_number:sample_number + 1, :]], axis=0)
        measured_index_test.append(sample_number)
y_measured_ys_test = y_measured_ys_test.drop(0, axis=0)

# テストデータの実測値 vs. 推定値のプロット
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.figure(figsize=figure.figaspect(1))  # 図の形を正方形に
plt.scatter(y_measured_ys_test.iloc[:, 0], y_measured_ys_test.iloc[:, 1], c='blue')  # 実測値 vs. 推定値プロット
y_max = max(y_measured_ys_test.iloc[:, 0].max(), y_measured_ys_test.iloc[:, 1].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
y_min = min(y_measured_ys_test.iloc[:, 0].min(), y_measured_ys_test.iloc[:, 1].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
plt.xlabel('actual y')  # x 軸の名前
plt.ylabel('estimated y')  # y 軸の名前
plt.show()  # 以上の設定で描画

# テストデータのr2, RMSE, MAE
print('r^2 for test data :', metrics.r2_score(y_measured_ys_test.iloc[:, 0], y_measured_ys_test.iloc[:, 1]))
print('RMSE for test data :',
      metrics.mean_squared_error(y_measured_ys_test.iloc[:, 0], y_measured_ys_test.iloc[:, 1]) ** 0.5)
print('MAE for test data :', metrics.mean_absolute_error(y_measured_ys_test.iloc[:, 0], y_measured_ys_test.iloc[:, 1]))

# 実測値と推定値の時間プロット
plt.rcParams['font.size'] = 18
plt.plot(range(estimated_y_test_all.shape[0]), estimated_y_test_all.iloc[:, 0], 'b.-',
         label='estimated y')
plt.plot(measured_index_test, y_measured_ys_test.iloc[:, 0], 'r.', markersize=15, label='actual y')
plt.xlabel('time')
plt.ylabel(dataset_train.columns[0])
plt.xlim([0, estimated_y_test_all.shape[0] + 1])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
plt.show()
# 実測値と推定値の時間プロット (拡大)
plt.plot(range(estimated_y_test_all.shape[0]), estimated_y_test_all.iloc[:, 0], 'b.-',
         label='estimated y')
plt.plot(measured_index_test, y_measured_ys_test.iloc[:, 0], 'r.', markersize=15, label='actual y')
plt.xlabel('time')
plt.ylabel(dataset_train.columns[0])
plt.xlim([894, 960])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
plt.show()
