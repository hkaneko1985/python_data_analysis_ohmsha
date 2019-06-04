# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS

file_name_of_seed_structures = 'molecules_with_boiling_point.csv'  # サンプルとして、'molecules_with_boiling_point.csv' か 'boiling_point.sdf' などがあります。
number_of_generated_structures = 1000  # 生成する化学構造の数
output_file_type = 'smiles'  # 保存する化学構造のデータ形式。'smiles' もしくは 'sdf' です。

if file_name_of_seed_structures[-4:] == '.csv':  # SMILES で分子の読み込み
    dataset = pd.read_csv(file_name_of_seed_structures, index_col=0)
    smiles = dataset.iloc[:, 0]  # 分子の SMILES
    molecules = []
    for smiles_i in smiles:
        molecules.append(Chem.MolFromSmiles(smiles_i))
elif file_name_of_seed_structures[-4:] == '.sdf':  # SDF ファイルで分子の読み込み
    molecules = Chem.SDMolSupplier(file_name_of_seed_structures)

# フラグメントへの変換
print('読み込んだ分子の数 :', len(molecules))
print('フラグメントへの分解')
fragments = set()
for molecule in molecules:
    fragment = BRICS.BRICSDecompose(molecule, minFragmentSize=1)
    fragments.update(fragment)
print('生成されたフラグメントの数 :', len(fragments))

# 化学構造生成
generated_structures = BRICS.BRICSBuild([Chem.MolFromSmiles(fragment) for fragment in fragments])
if output_file_type == 'smiles':  # SMILES として保存
    smiles_of_generated_structures = []
    for index, generated_structure in enumerate(generated_structures):
        print(index + 1, '/', number_of_generated_structures)
        generated_structure.UpdatePropertyCache(True)
        smiles_of_generated_structures.append(Chem.MolToSmiles(generated_structure))
        if index + 1 >= number_of_generated_structures:
            break
    smiles_of_generated_structures = pd.DataFrame(smiles_of_generated_structures, columns=['SMILES'])
    smiles_of_generated_structures.to_csv('generated_structures.smi', header=False,
                                          index=False)  # smi ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
    # smiles_of_generated_structures.to_csv('generated_structures.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
elif output_file_type == 'sdf':  # SDF ファイルとして保存
    writer = Chem.SDWriter('generated_structures.sdf')
    for index, generated_structure in enumerate(generated_structures):
        print(index + 1, '/', number_of_generated_structures)
        generated_structure.UpdatePropertyCache(True)
        AllChem.Compute2DCoords(generated_structure)
        writer.write(generated_structure)
        if index + 1 >= number_of_generated_structures:
            break
    writer.close()
