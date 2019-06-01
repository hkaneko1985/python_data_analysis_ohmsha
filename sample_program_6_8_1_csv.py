# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# 下の y_name を、'boiling_point', 'logS', 'melting_point', 'pIC50', 'pIC50_class', 'pIGC50', 'pIGC50_class'  のいずれかにしてください。
# molecules_with_[y_name].csv というファイルを読み込み、分子ごとに記述子を計算し、物性・活性と記述子計算の
# 結果をつなげた結果を descriptors_with_[y_name].csv というファイルに保存します。
# さらに、y_name を別の名前に変えて、ご自身で別途 molecules_with_[y_name].csv というファイルを、
# 他のファイルと同様の形式で準備すれば、同じように記述子計算ができます。

y_name = 'boiling_point'
# 'boiling_point' : 沸点のデータセットの場合
# 'logS' : 水溶解度のデータセットの場合
# 'melting_point' : 融点のデータセットの場合
# 'pIC50' : 薬理活性のデータセットの場合
# 'pIC50_class' : クラス分類用の薬理活性のデータセットの場合
# 'pIGC50' : 環境毒性のデータセットの場合
# 'pIGC50_class' : クラス分類用の環境毒性のデータセットの場合

dataset = pd.read_csv('molecules_with_{0}.csv'.format(y_name), index_col=0)  # SMILES 付きデータセットの読み込み
smiles = dataset.iloc[:, 0]  # 分子の SMILES
y = dataset.iloc[:, 1]  # 物性・活性などの目的変数

# 計算する記述子名の取得
descriptor_names = []
for descriptor_information in Descriptors.descList:
    descriptor_names.append(descriptor_information[0])
print('計算する記述子の数 :', len(descriptor_names))

# 記述子の計算
descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
descriptors = []  # ここに計算された記述子の値を追加
print('分子の数 :', len(smiles))
for index, smiles_i in enumerate(smiles):
    print(index + 1, '/', len(smiles))
    molecule = Chem.MolFromSmiles(smiles_i)
    descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
descriptors = pd.DataFrame(descriptors, index=dataset.index, columns=descriptor_names)

# 保存
y = pd.DataFrame(y)  # Series のため列名の変更は別に
y.columns = [y_name]
descriptors_with_y = pd.concat([y, descriptors], axis=1)  # y と記述子を結合
#descriptors_with_y = descriptors_with_y.drop(['MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge'], axis=1)
descriptors_with_y.to_csv('descriptors_with_{0}.csv'.format(y_name))  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
