# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

y_name = 'boiling_point'

sdf = Chem.SDMolSupplier('boiling_point.sdf')  # sdf ファイルの読み込み

# 計算する記述子名の取得
descriptor_names = []
for descriptor_information in Descriptors.descList:
    descriptor_names.append(descriptor_information[0])
print('計算する記述子の数 :', len(descriptor_names))

# 記述子の計算
descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
# 分子ごとに、リスト型の変数 y に物性値を、descriptors に計算された記述子の値を、smiles に SMILES を追加
descriptors, y, smiles = [], [], []
print('分子の数 :', len(sdf))
for index, molecule in enumerate(sdf):
    print(index + 1, '/', len(sdf))
    y.append(float(molecule.GetProp(y_name)))
    descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    smiles.append(Chem.MolToSmiles(molecule))
descriptors = pd.DataFrame(descriptors, index=smiles, columns=descriptor_names)
y = pd.DataFrame(y, index=smiles, columns=[y_name])

# 保存
descriptors_with_y = pd.concat([y, descriptors], axis=1)  # y と記述子を結合
descriptors_with_y.to_csv('descriptors_with_y.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
