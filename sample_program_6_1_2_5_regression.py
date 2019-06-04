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

method_name = 'svr'  # 'pls' or 'svr'
add_nonlinear_terms_flag = False  # True (二乗項・交差項を追加) or False (追加しない)
number_of_atom_types = 6  # 予測に用いる原子の種類の数
number_of_samples_in_prediction = 10000  # 予測するサンプル数
number_of_iterations = 100  # 予測を繰り返す回数
number_of_bins = 50  # y の推定値におけるヒストグラムのビンの数

fold_number = 2  # N-fold CV の N
max_number_of_principal_components = 30  # 使用する主成分の最大数
svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # C の候補
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # ε の候補
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補
ocsvm_nu = 0.003  # OCSVM における ν。トレーニングデータにおけるサンプル数に対する、サポートベクターの数の下限の割合
ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補

if method_name != 'pls' and method_name != 'svr':
    sys.exit('\'{0}\' という回帰分析手法はありません。method_name を見直してください。'.format(method_name))
    
dataset = pd.read_csv('unique_m.csv', index_col=-1)
dataset = dataset.sort_values('critical_temp', ascending=False).iloc[:4000, :]
y = dataset.iloc[:, 86].copy()
original_x = dataset.iloc[:, :86]
original_x = (original_x.T / original_x.T.sum()).T
# 標準偏差が 0 の説明変数を削除
original_x = original_x.drop(original_x.columns[original_x.std() == 0], axis=1)

if add_nonlinear_terms_flag:
    x = pd.read_csv('x_superconductor.csv', index_col=0)
    #    x = sample_functions.add_nonlinear_terms(original_x)  # 説明変数の二乗項や交差項を追加
    # 標準偏差が 0 の説明変数を削除
    std_0_nonlinear_variable_flags = x.std() == 0
    x = x.drop(x.columns[std_0_nonlinear_variable_flags], axis=1)
else:
    x = original_x.copy()

# オートスケーリング
autoscaled_original_x = (original_x - original_x.mean()) / original_x.std()
autoscaled_x = (x - x.mean()) / x.std()
autoscaled_y = (y - y.mean()) / y.std()

# グラム行列の分散を最大化することによる γ の最適化
optimal_ocsvm_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x, ocsvm_gammas)

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
    optimal_svr_gamma = optimal_ocsvm_gamma.copy()
    # CV による ε の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
                               cv=fold_number, iid=False, verbose=2)
    model_in_cv.fit(autoscaled_x, autoscaled_y)
    optimal_svr_epsilon = model_in_cv.best_params_['epsilon']
    # CV による C の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                               {'C': svr_cs}, cv=fold_number, iid=False, verbose=2)
    model_in_cv.fit(autoscaled_x, autoscaled_y)
    optimal_svr_c = model_in_cv.best_params_['C']
    # CV による γ の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                               {'gamma': svr_gammas}, cv=fold_number, iid=False, verbose=2)
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
    
# OCSVM による AD
print('最適化された gamma (OCSVM) :', optimal_ocsvm_gamma)
ad_model = svm.OneClassSVM(kernel='rbf', gamma=optimal_ocsvm_gamma, nu=ocsvm_nu)  # AD モデルの宣言
ad_model.fit(autoscaled_original_x)  # モデル構築

print('予測開始')
predicted_dataset = pd.DataFrame()
for iteration in range(number_of_iterations):
    print(iteration + 1, '/', number_of_iterations)
    # 予測用データの作成
    upper_x = original_x.max() + 0.1 * (original_x.max() - original_x.min())
    random_numbers = np.random.rand(number_of_samples_in_prediction, number_of_atom_types * 2)
    random_numbers[:, 0:number_of_atom_types] = np.ceil(
        random_numbers[:, 0:number_of_atom_types] * original_x.shape[1]) - 1
    random_numbers[:, 0] = original_x.columns.get_loc('O')  # O は必ず含める
    random_numbers[:, 1] = original_x.columns.get_loc('Ca')  # Ca は必ず含める
    random_numbers[:, 2] = original_x.columns.get_loc('Cu')  # Cu は必ず含める
    original_x_prediction = np.zeros([number_of_samples_in_prediction, original_x.shape[1]])
    for sample_number in range(number_of_samples_in_prediction):
        values_of_atoms = random_numbers[sample_number, number_of_atom_types:] * upper_x.iloc[
            random_numbers[sample_number, 0:number_of_atom_types]].values
        original_x_prediction[
            sample_number, random_numbers[sample_number, 0:number_of_atom_types].astype(np.int64)] = values_of_atoms
    original_x_prediction = pd.DataFrame(original_x_prediction)
    original_x_prediction.columns = original_x.columns
    original_x_prediction = (original_x_prediction.T / original_x_prediction.T.sum()).T

    # 予測用データに対して、AD の中か外かを判定
    autoscaled_original_x_prediction = (original_x_prediction - original_x.mean()) / original_x.std()
    data_density_prediction = ad_model.decision_function(autoscaled_original_x_prediction)  # データ密度 (f(x) の値)
    original_x_prediction_inside_ad = original_x_prediction.iloc[data_density_prediction >= 0, :]
    original_x_prediction_inside_ad = original_x_prediction_inside_ad.reset_index(drop=True)

    if add_nonlinear_terms_flag:
        x_prediction = sample_functions.add_nonlinear_terms(original_x_prediction_inside_ad)  # 説明変数の二乗項や交差項を追加
        x_prediction = x_prediction.drop(x_prediction.columns[std_0_nonlinear_variable_flags], axis=1)  # 標準偏差が 0 の説明変数を削除
    else:
        x_prediction = original_x_prediction_inside_ad.copy()

    # オートスケーリング
    autoscaled_x_prediction = (x_prediction - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    # 予測して、 positive なサンプルのみ保存
    predicted_y = pd.DataFrame(model.predict(autoscaled_x_prediction) * y.std() + y.mean())
    predicted_y.columns = [dataset.columns[86]]
    predicted_y_x = pd.concat([predicted_y, original_x_prediction_inside_ad], axis=1)
    predicted_dataset = pd.concat([predicted_dataset, predicted_y_x], axis=0)

predicted_dataset = predicted_dataset.reset_index(drop=True)
predicted_dataset = predicted_dataset.sort_values(dataset.columns[86], ascending=False)
predicted_dataset = predicted_dataset.reset_index(drop=True)
predicted_dataset.to_csv('predicted_critical_temp_dataset.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# 転移温度の推定値のヒストグラム
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.hist(predicted_dataset.iloc[:, 0], bins=number_of_bins)  # ヒストグラムの作成
plt.xlabel(predicted_dataset.columns[0])  # 横軸の名前
plt.ylabel('frequency')  # 縦軸の名前
plt.show()  # 以上の設定において、グラフを描画
