# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV

number_of_submodels = 30  # サブモデルの数
rate_of_selected_x_variables = 0.7  # 各サブデータセットで選択される説明変数の数の割合。0 より大きく 1 未満

svm_cs = 2 ** np.arange(-5, 11, dtype=float)
svm_gammas = 2 ** np.arange(-20, 10, dtype=float)
fold_number = 5  # N-fold CV の N
number_of_test_samples = 50  # テストデータのサンプル数
dataset = pd.read_csv('iris.csv', index_col=0)  # あやめのデータの読み込み
# 2 クラス 1 (positive), -1 (negative)  にします
dataset.iloc[0:100, 0] = 'positive'  # setosa と versicolor を 1 (positive) のクラスに
dataset.iloc[100:, 0] = 'negative'  # virginica を -1 (negative) のクラスに

# データ分割
y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数

# ランダムにトレーニングデータとテストデータとに分割
# random_state に数字を与えることで、別のときに同じ数字を使えば、ランダムとはいえ同じ結果にすることができます
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True,
                                                    random_state=21)

class_types = list(set(y_train))  # クラスの種類
class_types.sort(reverse=True)  # 並び替え

# オートスケーリング
autoscaled_x_train = (x_train - x_train.mean()) / x_train.std()
autoscaled_x_test = (x_test - x_test.mean()) / x_test.std()

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

    # ハイパーパラメータの最適化。CV による C, γ の最適化
    model_in_cv = GridSearchCV(svm.SVC(kernel='rbf'), {'C': svm_cs, 'gamma': svm_gammas}, cv=fold_number)
    model_in_cv.fit(selected_autoscaled_x_train, y_train)
    optimal_svm_c = model_in_cv.best_params_['C']
    optimal_svm_gamma = model_in_cv.best_params_['gamma']

    # SVM
    submodel = svm.SVC(kernel='rbf', C=optimal_svm_c, gamma=optimal_svm_gamma)  # モデルの宣言
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
