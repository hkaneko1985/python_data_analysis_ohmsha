# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sample_functions
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV

number_of_submodels = 30  # サブモデルの数
rate_of_selected_x_variables = 0.7  # 各サブデータセットで選択される説明変数の数の割合。0 より大きく 1 未満

svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # C の候補
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # ε の候補
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補
fold_number = 5  # N-fold CV の N

number_of_test_samples = 150  # テストデータのサンプル数
dataset = pd.read_csv('boston.csv', index_col=0)

# データ分割
y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数

# ランダムにトレーニングデータとテストデータとに分割
# random_state に数字を与えることで、別のときに同じ数字を使えば、ランダムとはいえ同じ結果にすることができます
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)

# オートスケーリング
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

# 時間短縮のため、最初だけグラム行列の分散を最大化することによる γ の最適化
optimal_svr_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x_train, svr_gammas)

number_of_x_variables = int(np.ceil(x_train.shape[1] * rate_of_selected_x_variables))
print('各サブデータセットの説明変数の数 :', number_of_x_variables)
estimated_y_train_all = pd.DataFrame()  # 空の DataFrame 型を作成し、ここにサブモデルごとのトレーニングデータの y の推定結果を追加
selected_x_variable_numbers = []  # 空の list 型の変数を作成し、ここに各サブデータセットの説明変数の番号を追加
submodels = []  # 空の list 型の変数を作成し、ここに構築済みの各サブモデルを追加
for submodel_number in range(number_of_submodels):
    print(submodel_number + 1, '/', number_of_submodels)  # 進捗状況の表示
    # 説明変数の選択
    # 0 から 1 までの間に一様に分布する乱数を説明変数の数だけ生成して、その乱数値が小さい順に説明変数を選択
    random_x_variables = np.random.rand(x_train.shape[1])
    selected_x_variable_numbers_tmp = random_x_variables.argsort()[:number_of_x_variables]
    selected_autoscaled_x_train = autoscaled_x_train.iloc[:, selected_x_variable_numbers_tmp]
    selected_x_variable_numbers.append(selected_x_variable_numbers_tmp)

    # ハイパーパラメータの最適化
    # CV による ε の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
                               cv=fold_number, iid=False)
    model_in_cv.fit(selected_autoscaled_x_train, autoscaled_y_train)
    optimal_svr_epsilon = model_in_cv.best_params_['epsilon']
    # CV による C の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                               {'C': svr_cs}, cv=fold_number, iid=False)
    model_in_cv.fit(selected_autoscaled_x_train, autoscaled_y_train)
    optimal_svr_c = model_in_cv.best_params_['C']
    # CV による γ の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                               {'gamma': svr_gammas}, cv=fold_number, iid=False)
    model_in_cv.fit(selected_autoscaled_x_train, autoscaled_y_train)
    optimal_svr_gamma = model_in_cv.best_params_['gamma']

    # SVR
    submodel = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)  # モデルの宣言
    submodel.fit(selected_autoscaled_x_train, autoscaled_y_train)  # モデルの構築
    submodels.append(submodel)

# サブデータセットの説明変数の種類やサブモデルを保存。同じ名前のファイルがあるときは上書きされるため注意
pd.to_pickle(selected_x_variable_numbers, 'selected_x_variable_numbers.bin')
pd.to_pickle(submodels, 'submodels.bin')

# サブデータセットの説明変数の種類やサブモデルを読み込み
# 今回は、保存した後にすぐ読み込んでいるため、あまり意味はありませんが、サブデータセットの説明変数の種類やサブモデルを
# 保存しておくことで、後で新しいサンプルを予測したいときにモデル構築の過程を省略できます
selected_x_variable_numbers = pd.read_pickle('selected_x_variable_numbers.bin')
submodels = pd.read_pickle('submodels.bin')

# テストデータの y の推定
estimated_y_test_all = pd.DataFrame()  # 空の DataFrame 型を作成し、ここにサブモデルごとのテストデータの y の推定結果を追加
for submodel_number in range(number_of_submodels):
    # 説明変数の選択
    selected_autoscaled_x_test = autoscaled_x_test.iloc[:, selected_x_variable_numbers[submodel_number]]

    # テストデータの y の推定
    estimated_y_test = pd.DataFrame(
        submodels[submodel_number].predict(selected_autoscaled_x_test))  # テストデータの y の値を推定し、Pandas の DataFrame 型に変換
    estimated_y_test = estimated_y_test * y_train.std() + y_train.mean()  # スケールをもとに戻します
    estimated_y_test_all = pd.concat([estimated_y_test_all, estimated_y_test], axis=1)

# テストデータの推定値の平均値
estimated_y_test = pd.DataFrame(estimated_y_test_all.median(axis=1))  # Series 型のため、行名と列名の設定は別に
#estimated_y_test = pd.DataFrame(estimated_y_test_all.mean(axis=1))  # Series 型のため、行名と列名の設定は別に
estimated_y_test.index = x_test.index
estimated_y_test.columns = ['estimated_y']
estimated_y_test.to_csv('estimated_y_test.csv')  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# テストデータの推定値の標準偏差
std_of_estimated_y_test = pd.DataFrame(estimated_y_test_all.std(axis=1))  # Series 型のため、行名と列名の設定は別に
std_of_estimated_y_test.index = x_test.index
std_of_estimated_y_test.columns = ['std_of_estimated_y']
std_of_estimated_y_test.to_csv('std_of_estimated_y_test.csv')  # 推定値の標準偏差を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# テストデータの推定値の標準偏差 vs. 推定値の誤差の絶対値プロット
plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.scatter(std_of_estimated_y_test.iloc[:, 0], abs(y_test - estimated_y_test.iloc[:, 0]), c='blue')  # 実測値 vs. 推定値プロット
plt.xlabel('std. of estimated y')  # x 軸の名前
plt.ylabel('absolute error of y')  # y 軸の名前
plt.show()  # 以上の設定で描画
