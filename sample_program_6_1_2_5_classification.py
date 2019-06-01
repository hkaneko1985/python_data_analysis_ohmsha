# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import math

import numpy as np
import pandas as pd
import sample_functions
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

method_name = 'rf'  # 'knn' or 'svm' or 'rf'
add_nonlinear_terms_flag = False  # True (二乗項・交差項を追加) or False (追加しない)
number_of_atom_types = 6  # 予測に用いる原子の種類の数
number_of_samples_in_prediction = 10000  # 予測するサンプル数
number_of_iterations = 100  # 予測を繰り返す回数

fold_number = 2  # N-fold CV の N
max_number_of_k = 20  # 使用する k の最大値
svm_cs = 2 ** np.arange(-5, 11, dtype=float)
svm_gammas = 2 ** np.arange(-20, 11, dtype=float)
rf_number_of_trees = 300  # RF における決定木の数
rf_x_variables_rates = np.arange(1, 11, dtype=float) / 10  # 1 つの決定木における説明変数の数の割合の候補
ocsvm_nu = 0.003  # OCSVM における ν。トレーニングデータにおけるサンプル数に対する、サポートベクターの数の下限の割合
ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補

dataset = pd.read_csv('unique_m.csv', index_col=-1)
dataset = dataset.sort_values('critical_temp', ascending=False).iloc[:4000, :]
y = dataset.iloc[:, 86].copy()
y[dataset.iloc[:, 86] >= 90] = 'positive'  # 転移温度 90 K 以上を高温超伝導体 (positive) とします
y[dataset.iloc[:, 86] < 90] = 'negative'
# 高温超電導体の数の調査
numbers = y.value_counts()
print('高温超電導体の数 :', numbers.iloc[1])
print('非高温超電導体の数 :', numbers.iloc[0])

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

autoscaled_original_x = (original_x - original_x.mean()) / original_x.std()  # オートスケーリング
autoscaled_x = (x - x.mean()) / x.std()  # オートスケーリング

# グラム行列の分散を最大化することによる γ の最適化
optimal_ocsvm_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x, ocsvm_gammas)

if method_name == 'knn':
    # CV による k の最適化
    accuracy_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の 正解率 をこの変数に追加していきます
    ks = []  # 同じく k の値をこの変数に追加していきます
    for k in range(1, max_number_of_k + 1):
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')  # k-NN モデルの宣言
        # クロスバリデーション推定値の計算し、DataFrame型に変換
        estimated_y_in_cv = pd.DataFrame(cross_val_predict(model, autoscaled_x, y, cv=fold_number))
        accuracy_in_cv = metrics.accuracy_score(y, estimated_y_in_cv)  # 正解率を計算
        print(k, accuracy_in_cv)  # k の値と r2 を表示
        accuracy_in_cv_all.append(accuracy_in_cv)  # r2 を追加
        ks.append(k)  # k の値を追加
    # k の値ごとの CV 後の正解率をプロットし、CV 後の正解率が最大のときを k の最適値に
    optimal_k = sample_functions.plot_and_selection_of_hyperparameter(ks, accuracy_in_cv_all, 'k',
                                                                      'cross-validated accuracy')

    print('\nCV で最適化された k :', optimal_k, '\n')
    # k-NN
    model = KNeighborsClassifier(n_neighbors=optimal_k, metric='euclidean')  # モデルの宣言
elif method_name == 'svm':
    optimal_svm_gamma = optimal_ocsvm_gamma.copy()
    # CV による C の最適化
    model_in_cv = GridSearchCV(svm.SVC(kernel='rbf', gamma=optimal_svm_gamma),
                               {'C': svm_cs}, cv=fold_number, iid=False, verbose=2)
    model_in_cv.fit(autoscaled_x, y)
    optimal_svm_c = model_in_cv.best_params_['C']
    # CV による γ の最適化
    model_in_cv = GridSearchCV(svm.SVC(kernel='rbf', C=optimal_svm_c),
                               {'gamma': svm_gammas}, cv=fold_number, iid=False, verbose=2)
    model_in_cv.fit(autoscaled_x, y)
    optimal_svm_gamma = model_in_cv.best_params_['gamma']
    print('CV で最適化された C :', optimal_svm_c)
    print('CV で最適化された γ:', optimal_svm_gamma)
    # SVM
    model = svm.SVC(kernel='rbf', C=optimal_svm_c, gamma=optimal_svm_gamma)  # モデルの宣言
elif method_name == 'rf':
    # OOB (Out-Of-Bugs) による説明変数の数の割合の最適化
    accuracy_oob = []
    for index, x_variables_rate in enumerate(rf_x_variables_rates):
        print(index + 1, '/', len(rf_x_variables_rates))
        model_in_validation = RandomForestClassifier(n_estimators=rf_number_of_trees, max_features=int(
            max(math.ceil(autoscaled_x.shape[1] * x_variables_rate), 1)), oob_score=True)
        model_in_validation.fit(autoscaled_x, y)
        accuracy_oob.append(model_in_validation.oob_score_)
    optimal_x_variables_rate = sample_functions.plot_and_selection_of_hyperparameter(rf_x_variables_rates,
                                                                                     accuracy_oob,
                                                                                     'rate of x-variables',
                                                                                     'accuracy for OOB')
    print('\nOOB で最適化された説明変数の数の割合 :', optimal_x_variables_rate)

    # RF
    model = RandomForestClassifier(n_estimators=rf_number_of_trees,
                                   max_features=int(
                                       max(math.ceil(autoscaled_x.shape[1] * optimal_x_variables_rate), 1)),
                                   oob_score=True)  # RF モデルの宣言
model.fit(autoscaled_x, y)  # モデルの構築
if method_name == 'rf':
    # 説明変数の重要度
    x_importances = pd.DataFrame(model.feature_importances_, index=x.columns, columns=['importance'])
    x_importances.to_csv('rf_x_importances.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
    
# OCSVM による AD
print('最適化された gamma (OCSVM) :', optimal_ocsvm_gamma)
ad_model = svm.OneClassSVM(kernel='rbf', gamma=optimal_ocsvm_gamma, nu=ocsvm_nu)  # AD モデルの宣言
ad_model.fit(autoscaled_original_x)  # モデル構築

print('予測開始')
predicted_positive_dataset = pd.DataFrame()
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
        x_prediction = x_prediction.drop(x_prediction.columns[std_0_nonlinear_variable_flags],
                                         axis=1)  # 標準偏差が 0 の説明変数を削除
    else:
        x_prediction = original_x_prediction_inside_ad.copy()

    # オートスケーリング
    autoscaled_x_prediction = (x_prediction - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    # 予測して、 positive なサンプルのみ保存
    predicted_y = model.predict(autoscaled_x_prediction)
    predicted_positive_dataset = pd.concat(
        [predicted_positive_dataset, original_x_prediction.iloc[predicted_y == 'positive', :]], axis=0)

predicted_positive_dataset = predicted_positive_dataset.reset_index(drop=True)
predicted_positive_dataset.to_csv('predicted_positive_dataset.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
