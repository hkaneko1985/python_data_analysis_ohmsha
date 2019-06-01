# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

number_of_bins = 50  # ヒストグラムのビンの数

# 下の y_name を、'boiling_point', 'logS', 'melting_point', 'pIC50', 'pIC50_class', 'pIGC50', 'pIGC50_class' のいずれかにしてください。
# descriptors_with_[y_name].csv というファイルを dataset として読み込み計算します。
# さらに、y_name を別の名前に変えて、ご自身で別途 sample_program_6_8_0_csv.py もしくは
# sample_program_6_8_0_sdf.py で descriptors_with_[y_name].csv というファイルを、
# 他のファイルと同様の形式で準備すれば、同じように計算することができます。

y_name = 'boiling_point'
# 'boiling_point' : 沸点のデータセットの場合
# 'logS' : 水溶解度のデータセットの場合
# 'melting_point' : 融点のデータセットの場合
# 'pIC50' : 薬理活性のデータセットの場合
# 'pIC50_class' : クラス分類用の薬理活性のデータセットの場合
# 'pIGC50' : 環境毒性のデータセットの場合
# 'pIGC50_class' : クラス分類用の環境毒性のデータセットの場合

dataset = pd.read_csv('descriptors_with_{0}.csv'.format(y_name), index_col=0)  # 物性・活性と記述子のデータセットの読み込み
dataset = dataset.replace(np.inf, np.nan).fillna(np.nan)  # inf を NaN に置き換え
dataset = dataset.drop(dataset.columns[dataset.isnull().any()], axis=1)  # NaN を含む変数を削除

if dataset.iloc[:, 0].dtype == 'float':  # 回帰分析
    dataset = dataset.drop(dataset.columns[dataset.std() == 0], axis=1)  # 標準偏差が 0 の特徴量 (記述子) を削除

    # y のヒストグラム
    plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
    plt.hist(dataset.iloc[:, 0], bins=number_of_bins)  # ヒストグラムの作成
    plt.xlabel(dataset.columns[0])  # 横軸の名前
    plt.ylabel('frequency')  # 縦軸の名前
    plt.show()  # 以上の設定において、グラフを描画
    
    # 相関行列
    correlation_coefficients = dataset.corr()  # 相関行列の計算
    correlation_coefficients.to_csv('correlation_coefficients_{0}.csv'.format(y_name))  # 相関行列を csv ファイルとして保存
    # 相関行列のヒートマップ (相関係数の値なし) 
    plt.rcParams['font.size'] = 12
    sns.heatmap(correlation_coefficients, vmax=1, vmin=-1, cmap='seismic', square=True, annot=False)
    plt.show()
    
    # y との相関係数のヒストグラム
    plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
    plt.hist(correlation_coefficients.iloc[1:, 0], bins=number_of_bins)  # ヒストグラムの作成
    plt.xlabel('correlation coef. with {0}'.format(dataset.columns[0]))  # 横軸の名前
    plt.ylabel('frequency')  # 縦軸の名前
    plt.show()  # 以上の設定において、グラフを描画
    
    # y と最も相関係数の絶対値の高い記述子と y の散布図
    correlation_wity_y_max_descriptor = abs(correlation_coefficients.iloc[1:, 0]).idxmax()
    plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
    plt.scatter(dataset[correlation_wity_y_max_descriptor], dataset.iloc[:, 0], c='blue')
    plt.xlabel(correlation_wity_y_max_descriptor)
    plt.ylabel(dataset.columns[0])
    plt.show()
else:
    x = dataset.iloc[:, 1:]
    x = x.drop(x.columns[x.std() == 0], axis=1)  # 標準偏差が 0 の特徴量 (記述子) を削除
    # 相関行列
    correlation_coefficients = x.corr()  # 相関行列の計算
    correlation_coefficients.to_csv('correlation_coefficients_{0}.csv'.format(y_name))  # 相関行列を csv ファイルとして保存
    # 相関行列のヒートマップ (相関係数の値なし) 
    plt.rcParams['font.size'] = 12
    sns.heatmap(correlation_coefficients, vmax=1, vmin=-1, cmap='seismic', square=True, annot=False)
    plt.show()
