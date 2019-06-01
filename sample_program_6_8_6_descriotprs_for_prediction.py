# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# 下の file_name_for_prediction を、物性・活性を推定する化学構造の
# ファイル名にしてください。csv ファイルもしくは sdf ファイルです。 
# サンプルとして、molecules_for_prediction.csv と
# molecules_for_prediction.sdf がありますのでご確認ください。
# 記述子を計算した結果が descriptors_of_[file_name_for_prediction の拡張子以外].csv
# という名前の csv ファイルに保存されます。
# サンプルと同様の csv ファイルや sdf ファイルを作成して、
# file_name_for_prediction でファイル名を指定すれば、 
# 同じように記述子を計算して結果を保存することができます。

file_name_for_prediction = 'molecules_for_prediction.csv'  # 物性・活性の推定用のデータセットのファイル名

# 計算する記述子の準備
descriptor_names = []
for descriptor_information in Descriptors.descList:
    descriptor_names.append(descriptor_information[0])
print('計算する記述子の数 :', len(descriptor_names))
descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

# ファイルの読み込みおよび記述子計算
if file_name_for_prediction[-4:] == '.csv':
    dataset = pd.read_csv(file_name_for_prediction, index_col=0)  # SMILES 付きデータセットの読み込み
    smiles = dataset.iloc[:, 0]  # 分子の SMILES
    # 記述子の計算
    descriptors = []  # ここに計算された記述子の値を追加
    print('分子の数 :', len(smiles))
    for index, smiles_i in enumerate(smiles):
        print(index + 1, '/', len(smiles))
        molecule = Chem.MolFromSmiles(smiles_i)
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    descriptors = pd.DataFrame(descriptors, index=dataset.index, columns=descriptor_names)
elif file_name_for_prediction[-4:] == '.sdf':
    molecules = Chem.SDMolSupplier(file_name_for_prediction)  # sdf ファイルの読み込み
    # 分子ごとに、リスト型の変数 y に物性値を、descriptors に計算された記述子の値を、smiles に SMILES を追加
    descriptors, smiles = [], []
    print('分子の数 :', len(molecules))
    for index, molecule in enumerate(molecules):
        print(index + 1, '/', len(molecules))
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
        smiles.append(Chem.MolToSmiles(molecule))
    descriptors = pd.DataFrame(descriptors, index=smiles, columns=descriptor_names)

# 記述子の計算結果の保存
descriptors.to_csv('descriptors_of_{0}.csv'.format(file_name_for_prediction[:-4]))  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
