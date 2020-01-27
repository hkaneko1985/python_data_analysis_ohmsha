# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import math
import sys

import numpy as np
import pandas as pd
import sample_functions
from sklearn import metrics, model_selection, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# 下の y_name を 'pIC50_class', 'pIGC50_class' のいずれかにしてください。
# descriptors_with_[y_name].csv というファイルを dataset として読み込み計算します。
# さらに、y_name を別の名前に変えて、ご自身で別途 sample_program_6_8_0_csv.py もしくは
# sample_program_6_8_0_sdf.py で descriptors_with_[y_name].csv というファイルを、
# 他のファイルと同様の形式で準備すれば、同じように計算することができます。

y_name = 'pIC50_class'
# 'pIC50_class' : クラス分類用の薬理活性のデータセットの場合
# 'pIGC50_class' : クラス分類用の環境毒性のデータセットの場合

rate_of_test_samples = 0.25  # テストデータのサンプル数の割合。0 より大きく 1 未満

method_name = 'rf'  # 'knn' or 'svm' or 'rf'
number_of_submodels = 50  # サブモデルの数
rate_of_selected_x_variables = 0.7  # 各サブデータセットで選択される説明変数の数の割合。0 より大きく 1 未満
add_nonlinear_terms_flag = False  # True (二乗項・交差項を追加) or False (追加しない)

fold_number = 5  # N-fold CV の N
max_number_of_k = 20  # 使用する k の最大値
svm_cs = 2 ** np.arange(-5, 11, dtype=float)
svm_gammas = 2 ** np.arange(-20, 11, dtype=float)
rf_number_of_trees = 300  # RF における決定木の数
rf_x_variables_rates = np.arange(1, 11, dtype=float) / 10  # 1 つの決定木における説明変数の数の割合の候補

if method_name != 'knn' and method_name != 'svm' and method_name != 'rf':
    sys.exit('\'{0}\' というクラス分類手法はありません。method_name を見直してください。'.format(method_name))
    
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

class_types = list(set(y_train))  # クラスの種類
class_types.sort(reverse=True)  # 並び替え

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
autoscaled_x_test = (x_test - x_train.mean()) / x_train.std()

if method_name == 'svm':
    # 時間短縮のため、最初だけグラム行列の分散を最大化することによる γ の最適化
    optimal_svm_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x_train, svm_gammas)

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

    if method_name == 'knn':
        # CV による k の最適化
        accuracy_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の 正解率 をこの変数に追加していきます
        ks = []  # 同じく k の値をこの変数に追加していきます
        for k in range(1, max_number_of_k + 1):
            model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')  # k-NN モデルの宣言
            # クロスバリデーション推定値の計算し、DataFrame型に変換
            estimated_y_in_cv = pd.DataFrame(
                model_selection.cross_val_predict(model, selected_autoscaled_x_train, y_train,
                                                  cv=fold_number))
            accuracy_in_cv = metrics.accuracy_score(y_train, estimated_y_in_cv)  # 正解率を計算
            accuracy_in_cv_all.append(accuracy_in_cv)  # r2 を追加
            ks.append(k)  # k の値を追加
        optimal_k = ks[accuracy_in_cv_all.index(max(accuracy_in_cv_all))]
        submodel = KNeighborsClassifier(n_neighbors=optimal_k, metric='euclidean')  # k-NN モデルの宣言
    elif method_name == 'svm':
        # CV による C の最適化
        model_in_cv = GridSearchCV(svm.SVC(kernel='rbf', gamma=optimal_svm_gamma),
                                   {'C': svm_cs}, cv=fold_number)
        model_in_cv.fit(selected_autoscaled_x_train, y_train)
        optimal_svm_c = model_in_cv.best_params_['C']
        # CV による γ の最適化
        model_in_cv = GridSearchCV(svm.SVC(kernel='rbf', C=optimal_svm_c),
                                   {'gamma': svm_gammas}, cv=fold_number)
        model_in_cv.fit(selected_autoscaled_x_train, y_train)
        optimal_svm_gamma = model_in_cv.best_params_['gamma']
        submodel = svm.SVC(kernel='rbf', C=optimal_svm_c, gamma=optimal_svm_gamma)  # SVM モデルの宣言
    elif method_name == 'rf':
        # OOB (Out-Of-Bugs) による説明変数の数の割合の最適化
        accuracy_oob = []
        for index, x_variables_rate in enumerate(rf_x_variables_rates):
            model_in_validation = RandomForestClassifier(n_estimators=rf_number_of_trees, max_features=int(
                max(math.ceil(selected_autoscaled_x_train.shape[1] * x_variables_rate), 1)), oob_score=True)
            model_in_validation.fit(selected_autoscaled_x_train, y_train)
            accuracy_oob.append(model_in_validation.oob_score_)
        optimal_x_variables_rate = rf_x_variables_rates[accuracy_oob.index(max(accuracy_oob))]
        submodel = RandomForestClassifier(n_estimators=rf_number_of_trees,
                                          max_features=int(max(math.ceil(
                                              selected_autoscaled_x_train.shape[1] * optimal_x_variables_rate), 1)),
                                          oob_score=True)  # RF モデルの宣言
    submodel.fit(selected_autoscaled_x_train, y_train)  # モデルの構築
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
# estimated_y_test_all = pd.DataFrame()  # 空の DataFrame 型を作成し、ここにサブモデルごとのテストデータの y の推定結果を追加
estimated_y_test_count = np.zeros([x_test.shape[0], len(class_types)])  # クラスごとに、推定したサブモデルの数をカウントして値をここに格納
for submodel_number in range(number_of_submodels):
    # 説明変数の選択
    selected_autoscaled_x_test = autoscaled_x_test.iloc[:, selected_x_variable_numbers[submodel_number]]

    # テストデータの y の推定
    estimated_y_test = pd.DataFrame(
        submodels[submodel_number].predict(selected_autoscaled_x_test))  # テストデータの y の値を推定し、Pandas の DataFrame 型に変換
    #    estimated_y_test_all = pd.concat([estimated_y_test_all, estimated_y_test], axis=1)
    for sample_number in range(estimated_y_test.shape[0]):
        estimated_y_test_count[sample_number, class_types.index(estimated_y_test.iloc[sample_number, 0])] += 1

# テストデータにおける、クラスごとの推定したサブモデルの数
estimated_y_test_count = pd.DataFrame(estimated_y_test_count, index=x_test.index, columns=class_types)
estimated_y_test_count.to_csv('estimated_y_test_count.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# テストデータにおける、クラスごとの確率
estimated_y_test_probability = estimated_y_test_count / number_of_submodels
estimated_y_test_probability.to_csv('estimated_y_test_probability.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# テストデータにおける、多数決で推定された結果
estimated_y_test = pd.DataFrame(estimated_y_test_count.idxmax(axis=1), columns=['estimated_class'])
estimated_y_test.to_csv('estimated_y_test.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# テストデータにおける、クラスごとの確率と推定結果の確認
y_test = pd.DataFrame(y_test)  # Series 型のため、行名と列名の設定は別に
y_test.columns = ['actual_class']
estimated_y_test_for_check = pd.concat([estimated_y_test_probability, y_test, estimated_y_test], axis=1)  # 結合
estimated_y_test_for_check.to_csv('estimated_y_test_for_check.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
