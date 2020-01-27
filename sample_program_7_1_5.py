# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sample_functions
from sklearn import metrics
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV

number_of_submodels = 50  # サブモデルの数
rate_of_selected_x_variables = 0.8  # 各サブデータセットで選択される説明変数の数の割合。0 より大きく 1 未満
method_name = 'pls'  # 'pls' or 'svr'
add_nonlinear_terms_flag = False  # True (二乗項・交差項を追加) or False (追加しない)

fold_number = 5  # N-fold CV の N
max_number_of_principal_components = 20  # 使用する主成分の最大数
svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # C の候補
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # ε の候補
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補

if method_name != 'pls' and method_name != 'svr':
    sys.exit('\'{0}\' という回帰分析手法はありません。method_name を見直してください。'.format(method_name))
    
dataset_train = pd.read_csv('tep_13_train_with_y.csv', index_col=0)
dataset_test = pd.read_csv('tep_13_test_with_y.csv', index_col=0)
x_prediction = pd.read_csv('tep_prediction.csv', index_col=0)  # 予測用データ

# y が測定された時刻のサンプルのみ
dataset = dataset_train.iloc[0:1, :]
for sample_number in range(1, dataset_train.shape[0]):
    if dataset.iloc[-1, 0] != dataset_train.iloc[sample_number, 0]:
        dataset = pd.concat(
            [dataset, dataset_train.iloc[sample_number:sample_number + 1, :]], axis=0)
dataset = pd.concat([dataset, dataset_test.iloc[0:1, :]], axis=0)
for sample_number in range(1, dataset_test.shape[0]):
    if dataset.iloc[-1, 0] != dataset_test.iloc[sample_number, 0]:
        dataset = pd.concat(
            [dataset, dataset_test.iloc[sample_number:sample_number + 1, :]], axis=0)
y = dataset.iloc[:, 0]
x = dataset.iloc[:, 1:]

if add_nonlinear_terms_flag:
    x = sample_functions.add_nonlinear_terms(x)  # 説明変数の二乗項や交差項を追加
    x_prediction = sample_functions.add_nonlinear_terms(x_prediction)
    # 標準偏差が 0 の説明変数を削除
    std_0_variable_flags = x.std() == 0
    x = x.drop(x.columns[std_0_variable_flags], axis=1)
    x_prediction = x_prediction.drop(x_prediction.columns[std_0_variable_flags], axis=1)

# オートスケーリング
autoscaled_x = (x - x.mean()) / x.std()
autoscaled_y = (y - y.mean()) / y.std()
autoscaled_x_prediction = (x_prediction - x.mean()) / x.std()

if method_name == 'svr':
    # 時間短縮のため、最初だけグラム行列の分散を最大化することによる γ の最適化
    optimal_svr_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x, svr_gammas)

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
    # CV による ε の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
                               cv=fold_number)
    model_in_cv.fit(autoscaled_x, autoscaled_y)
    optimal_svr_epsilon = model_in_cv.best_params_['epsilon']
    # CV による C の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                               {'C': svr_cs}, cv=fold_number)
    model_in_cv.fit(autoscaled_x, autoscaled_y)
    optimal_svr_c = model_in_cv.best_params_['C']
    # CV による γ の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                               {'gamma': svr_gammas}, cv=fold_number)
    model_in_cv.fit(autoscaled_x, autoscaled_y)
    optimal_svr_gamma = model_in_cv.best_params_['gamma']
    # 最適化された C, ε, γ
    print('C : {0}\nε : {1}\nGamma : {2}'.format(optimal_svr_c, optimal_svr_epsilon, optimal_svr_gamma))
    # SVR
    model = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)  # モデルの宣言

model.fit(autoscaled_x, autoscaled_y)  # モデルの構築
if method_name == 'pls':
    # 標準回帰係数
    standard_regression_coefficients = pd.DataFrame(model.coef_, index=x.columns,
                                                    columns=['standard_regression_coefficients'])
    standard_regression_coefficients.to_csv(
        'pls_standard_regression_coefficients.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
    
# AD の設定
number_of_x_variables = int(np.ceil(x.shape[1] * rate_of_selected_x_variables))
print('各サブデータセットの説明変数の数 :', number_of_x_variables)
estimated_y_train_all = pd.DataFrame()  # 空の DataFrame 型を作成し、ここにサブモデルごとのトレーニングデータの y の推定結果を追加
selected_x_variable_numbers = []  # 空の list 型の変数を作成し、ここに各サブデータセットの説明変数の番号を追加
submodels = []  # 空の list 型の変数を作成し、ここに構築済みの各サブモデルを追加
for submodel_number in range(number_of_submodels):
    print(submodel_number + 1, '/', number_of_submodels)  # 進捗状況の表示
    # 説明変数の選択
    # 0 から 1 までの間に一様に分布する乱数を説明変数の数だけ生成して、その乱数値が小さい順に説明変数を選択
    random_x_variables = np.random.rand(x.shape[1])
    selected_x_variable_numbers_tmp = random_x_variables.argsort()[:number_of_x_variables]
    selected_autoscaled_x = autoscaled_x.iloc[:, selected_x_variable_numbers_tmp]
    selected_x_variable_numbers.append(selected_x_variable_numbers_tmp)

    if method_name == 'pls':
        # CV による成分数の最適化
        components = []  # 空の list の変数を作成して、成分数をこの変数に追加していきます同じく成分数をこの変数に追加
        r2_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の r2 をこの変数に追加
        for component in range(1, min(np.linalg.matrix_rank(selected_autoscaled_x),
                                      max_number_of_principal_components) + 1):
            # PLS
            submodel_in_cv = PLSRegression(n_components=component)  # PLS モデルの宣言
            estimated_y_in_cv = pd.DataFrame(cross_val_predict(submodel_in_cv, selected_autoscaled_x, autoscaled_y,
                                                               cv=fold_number))  # クロスバリデーション推定値の計算し、DataFrame型に変換
            estimated_y_in_cv = estimated_y_in_cv * y.std() + y.mean()  # スケールをもとに戻す
            r2_in_cv = metrics.r2_score(y, estimated_y_in_cv)  # r2 を計算
            r2_in_cv_all.append(r2_in_cv)  # r2 を追加
            components.append(component)  # 成分数を追加
        optimal_component_number = components[r2_in_cv_all.index(max(r2_in_cv_all))]
        # PLS
        submodel = PLSRegression(n_components=optimal_component_number)  # モデルの宣言
    elif method_name == 'svr':
        # ハイパーパラメータの最適化
        # グラム行列の分散を最大化することによる γ の最適化
        optimal_svr_gamma = sample_functions.gamma_optimization_with_variance(selected_autoscaled_x, svr_gammas)
        # CV による ε の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
                                   cv=fold_number, iid=False)
        model_in_cv.fit(selected_autoscaled_x, autoscaled_y)
        optimal_svr_epsilon = model_in_cv.best_params_['epsilon']
        # CV による C の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                                   {'C': svr_cs}, cv=fold_number, iid=False)
        model_in_cv.fit(selected_autoscaled_x, autoscaled_y)
        optimal_svr_c = model_in_cv.best_params_['C']
        # CV による γ の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                                   {'gamma': svr_gammas}, cv=fold_number, iid=False)
        model_in_cv.fit(selected_autoscaled_x, autoscaled_y)
        optimal_svr_gamma = model_in_cv.best_params_['gamma']
        # SVR
        submodel = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon,
                           gamma=optimal_svr_gamma)  # モデルの宣言
    submodel.fit(selected_autoscaled_x, autoscaled_y)  # モデルの構築
    submodels.append(submodel)
# サブデータセットの説明変数の種類やサブモデルを保存。同じ名前のファイルがあるときは上書きされるため注意
pd.to_pickle(selected_x_variable_numbers, 'selected_x_variable_numbers.bin')
pd.to_pickle(submodels, 'submodels.bin')

# 予測用データの y の推定
estimated_y_prediction = model.predict(autoscaled_x_prediction) * y.std() + y.mean()
estimated_y_prediction = pd.DataFrame(estimated_y_prediction, index=x_prediction.index, columns=['estimated_y'])
estimated_y_prediction.to_csv('estimated_y_prediction.csv')  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
# 予測用データの y の推定値の標準偏差の計算
estimated_y_prediction_all = pd.DataFrame()  # 空の DataFrame 型を作成し、ここにサブモデルごとの予測用データの y の推定結果を追加
for submodel_number in range(number_of_submodels):
    # 説明変数の選択
    selected_autoscaled_x_test = autoscaled_x_prediction.iloc[:, selected_x_variable_numbers[submodel_number]]
    # テストデータの y の推定
    estimated_y_prediction_sub = pd.DataFrame(
        submodels[submodel_number].predict(selected_autoscaled_x_test))  # テストデータの y の値を推定し、Pandas の DataFrame 型に変換
    estimated_y_prediction_sub = estimated_y_prediction_sub * y.std() + y.mean()  # スケールをもとに戻します
    estimated_y_prediction_all = pd.concat([estimated_y_prediction_all, estimated_y_prediction_sub], axis=1)
std_of_estimated_y_prediction = pd.DataFrame(estimated_y_prediction_all.std(axis=1))  # Series 型のため、行名と列名の設定は別に
std_of_estimated_y_prediction.index = x_prediction.index  # 行の名前をサンプル名に変更
std_of_estimated_y_prediction.columns = ['std_of_estimated_y']  # 列の名前を変更
std_of_estimated_y_prediction.to_csv(
    'std_of_estimated_y_prediction.csv')  # 推定値の標準偏差を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# y の推定値の時間プロット
plt.rcParams['font.size'] = 18
plt.plot(range(estimated_y_prediction.shape[0]), estimated_y_prediction.iloc[:, 0], 'b.-', label='estimated y')
plt.plot(range(estimated_y_prediction.shape[0]),
         estimated_y_prediction.iloc[:, 0] + 3 * std_of_estimated_y_prediction.iloc[:, 0], 'k.-', label='+3 sigma')
plt.plot(range(estimated_y_prediction.shape[0]),
         estimated_y_prediction.iloc[:, 0] - 3 * std_of_estimated_y_prediction.iloc[:, 0], 'k.-', label='−3 sigma')
plt.xlabel('time')
plt.ylabel(dataset_train.columns[0])
plt.xlim([0, estimated_y_prediction.shape[0]])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
plt.show()
# y の推定値の時間プロット (拡大)
plt.rcParams['font.size'] = 18
plt.plot(range(estimated_y_prediction.shape[0]), estimated_y_prediction.iloc[:, 0], 'b.-', label='estimated y')
plt.plot(range(estimated_y_prediction.shape[0]),
         estimated_y_prediction.iloc[:, 0] + 3 * std_of_estimated_y_prediction.iloc[:, 0], 'k.-', label='+3 sigma')
plt.plot(range(estimated_y_prediction.shape[0]),
         estimated_y_prediction.iloc[:, 0] - 3 * std_of_estimated_y_prediction.iloc[:, 0], 'k.-', label='−3 sigma')
plt.xlabel('time')
plt.ylabel(dataset_train.columns[0])
plt.xlim([170, 200])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
plt.show()
