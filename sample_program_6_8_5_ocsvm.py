# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import numpy as np
import pandas as pd
import sample_functions
from sklearn import svm

# 下の y_name を、'boiling_point', 'logS', 'melting_point', 'pIC50', 'pIC50_class', 'pIGC50', 'pIGC50_class' のいずれかにしてください。
# descriptors_with_[y_name].csv というファイルを dataset として読み込み計算します。
# さらに、y_name を別の名前に変えて、ご自身で別途 sample_program_6_8_0_csv.py もしくは
# sample_program_6_8_0_sdf.py で descriptors_with_[y_name].csv というファイルを、
# 他のファイルと同様の形式で準備すれば、同じように計算することができます。

y_name = 'pIGC50_class'
# 'boiling_point' : 沸点のデータセットの場合
# 'logS' : 水溶解度のデータセットの場合
# 'melting_point' : 融点のデータセットの場合
# 'pIC50' : 薬理活性のデータセットの場合
# 'pIC50_class' : クラス分類用の薬理活性のデータセットの場合
# 'pIGC50' : 環境毒性のデータセットの場合
# 'pIGC50_class' : クラス分類用の環境毒性のデータセットの場合

ocsvm_nu = 0.05  # OCSVM における ν。トレーニングデータにおけるサンプル数に対する、サポートベクターの数の下限の割合
ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補

dataset = pd.read_csv('descriptors_with_{0}.csv'.format(y_name), index_col=0)  # 物性・活性と記述子のデータセットの読み込み
x = dataset.iloc[:, 1:]
x = x.replace(np.inf, np.nan).fillna(np.nan)  # inf を NaN に置き換え
nan_variable_flags = x.isnull().any()  # NaN を含む変数
x = x.drop(x.columns[nan_variable_flags], axis=1)  # NaN を含む変数を削除
# 標準偏差が 0 の説明変数を削除
std_0_variable_flags = x.std() == 0
x = x.drop(x.columns[std_0_variable_flags], axis=1)

autoscaled_x = (x - x.mean()) / x.std()  # オートスケーリング

# グラム行列の分散を最大化することによる γ の最適化
optimal_ocsvm_gamma = sample_functions.gamma_optimization_with_variance(autoscaled_x, ocsvm_gammas)
# 最適化された γ
print('最適化された gamma (OCSVM) :', optimal_ocsvm_gamma)

# OCSVM による AD
ad_model = svm.OneClassSVM(kernel='rbf', gamma=optimal_ocsvm_gamma, nu=ocsvm_nu)  # AD モデルの宣言
ad_model.fit(autoscaled_x)  # モデル構築

# トレーニングデータのデータ密度 (f(x) の値)
data_density_train = ad_model.decision_function(autoscaled_x)
number_of_support_vectors = len(ad_model.support_)
number_of_outliers_in_training_data = sum(data_density_train < 0)
print('\nトレーニングデータにおけるサポートベクター数 :', number_of_support_vectors)
print('トレーニングデータにおけるサポートベクターの割合 :', number_of_support_vectors / x.shape[0])
print('\nトレーニングデータにおける外れサンプル数 :', number_of_outliers_in_training_data)
print('トレーニングデータにおける外れサンプルの割合 :', number_of_outliers_in_training_data / x.shape[0])
data_density_train = pd.DataFrame(data_density_train, index=x.index, columns=['ocsvm_data_density'])
data_density_train.to_csv('ocsvm_data_density_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# トレーニングデータに対して、AD の中か外かを判定
inside_ad_flag_train = data_density_train >= 0  # AD 内のサンプルのみ TRUE
inside_ad_flag_train.columns = ['inside_ad_flag']
inside_ad_flag_train.to_csv('inside_ad_flag_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
