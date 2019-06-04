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
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV

# 下の y_name を、'boiling_point', 'logS', 'melting_point', 'pIC50', 'pIGC50' のいずれかにしてください。
# descriptors_with_[y_name].csv というファイルを dataset として読み込み計算します。
# さらに、y_name を別の名前に変えて、ご自身で別途 sample_program_6_8_0_csv.py もしくは
# sample_program_6_8_0_sdf.py で descriptors_with_[y_name].csv というファイルを、
# 他のファイルと同様の形式で準備すれば、同じように計算することができます。

y_name = 'melting_point'
# 'boiling_point' : 沸点のデータセットの場合
# 'logS' : 水溶解度のデータセットの場合
# 'melting_point' : 融点のデータセットの場合
# 'pIC50' : 薬理活性のデータセットの場合
# 'pIGC50' : 環境毒性のデータセットの場合

rate_of_test_samples = 0.25 # テストデータのサンプル数の割合。0 より大きく 1 未満
method_name = 'svr'  # 'pls' or 'svr'
number_of_submodels = 50  # サブモデルの数
rate_of_selected_x_variables = 0.8  # 各サブデータセットで選択される説明変数の数の割合。0 より大きく 1 未満
add_nonlinear_terms_flag = False  # True (二乗項・交差項を追加) or False (追加しない)

fold_number = 5  # N-fold CV の N
max_number_of_principal_components = 20  # 使用する主成分の最大数
svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # C の候補
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # ε の候補
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補

if method_name != 'pls' and method_name != 'svr':
    sys.exit('\'{0}\' という回帰分析手法はありません。method_name を見直してください。'.format(method_name))
    
dataset = pd.read_csv('descriptors_with_{0}.csv'.format(y_name), index_col=0)  # 物性・活性と記述子のデータセットの読み込み
y = dataset.iloc[:, 0].copy()
x = dataset.iloc[:, 1:]
x = x.replace(np.inf, np.nan).fillna(np.nan)  # inf を NaN に置き換え
nan_variable_flags = x.isnull().any()  # NaN を含む変数
x = x.drop(x.columns[nan_variable_flags], axis=1)  # NaN を含む変数を削除
number_of_test_samples = round(dataset.shape[0] * rate_of_test_samples)

# ランダムにトレーニングデータとテストデータとに分割
# random_state に数字を与えることで、別のときに同じ数字を使えば、ランダムとはいえ同じ結果にすることができます
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True,
                                                    random_state=0)
# 標準偏差が 0 の説明変数を削除
std_0_variable_flags = x_train.std() == 0
x_train = x_train.drop(x_train.columns[std_0_variable_flags], axis=1)
x_test = x_test.drop(x_test.columns[std_0_variable_flags], axis=1)

if add_nonlinear_terms_flag:
    x_train = pd.read_csv('x_train_{0}.csv'.format(y_name), index_col=0)  # 物性・活性と記述子のデータセットの読み込み
    x_test = pd.read_csv('x_test_{0}.csv'.format(y_name), index_col=0)  # 物性・活性と記述子のデータセットの読み込み
#    x_train = sample_functions.add_nonlinear_terms(x_train)  # 説明変数の二乗項や交差項を追加
#    x_test = sample_functions.add_nonlinear_terms(x_test)  # 説明変数の二乗項や交差項を追加
    # 標準偏差が 0 の説明変数を削除
    std_0_nonlinear_variable_flags = x_train.std() == 0
    x_train = x_train.drop(x_train.columns[std_0_nonlinear_variable_flags], axis=1)
    x_test = x_test.drop(x_test.columns[std_0_nonlinear_variable_flags], axis=1)

# オートスケーリング
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std()
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()

if method_name == 'svr':
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

    if method_name == 'pls':
        # CV による成分数の最適化
        components = []  # 空の list の変数を作成して、成分数をこの変数に追加していきます同じく成分数をこの変数に追加
        r2_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の r2 をこの変数に追加
        for component in range(1, min(np.linalg.matrix_rank(selected_autoscaled_x_train),
                                      max_number_of_principal_components) + 1):
            # PLS
            submodel_in_cv = PLSRegression(n_components=component)  # PLS モデルの宣言
            estimated_y_in_cv = pd.DataFrame(cross_val_predict(submodel_in_cv, selected_autoscaled_x_train, autoscaled_y_train,
                                                               cv=fold_number))  # クロスバリデーション推定値の計算し、DataFrame型に変換
            estimated_y_in_cv = estimated_y_in_cv * y_train.std() + y_train.mean()  # スケールをもとに戻す
            r2_in_cv = metrics.r2_score(y_train, estimated_y_in_cv)  # r2 を計算
            r2_in_cv_all.append(r2_in_cv)  # r2 を追加
            components.append(component)  # 成分数を追加
        optimal_component_number = components[r2_in_cv_all.index(max(r2_in_cv_all))]
        # PLS
        submodel = PLSRegression(n_components=optimal_component_number)  # モデルの宣言
    elif method_name == 'svr':
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
        submodel = svm.SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon,
                           gamma=optimal_svr_gamma)  # モデルの宣言
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
estimated_y_test = pd.DataFrame(estimated_y_test_all.mean(axis=1))  # Series 型のため、行名と列名の設定は別に
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
