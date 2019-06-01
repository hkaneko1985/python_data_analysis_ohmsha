# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import sample_functions
from rdkit import Chem
from rdkit.Chem import AllChem

file_name_of_main_fragments = 'sample_main_fragments.smi'  # 主骨格のフラグメントがあるファイル名。サンプルとして、'sample_main_fragments.smi' があります。
file_name_of_sub_fragments = 'sample_sub_fragments.smi'  # 側鎖のフラグメントがあるファイル名。サンプルとして、'sample_main_fragments.smi' があります
number_of_generated_structures = 10000  # 生成する化学構造の数
output_file_type = 'smiles'  # 保存する化学構造のデータ形式。'smiles' もしくは 'sdf' です。

# 化学構造生成
generated_structures = sample_functions.structure_generation_based_on_r_group_random(file_name_of_main_fragments,
                                                                                     file_name_of_sub_fragments,
                                                                                     number_of_generated_structures)

# 保存
if output_file_type == 'smiles':  # SMILES として保存
    str_ = '\n'.join(generated_structures)
    with open('generated_structures.smi', 'wt') as writer:
        writer.write(str_)
    writer.close()
elif output_file_type == 'sdf':  # SDF ファイルとして保存
    writer = Chem.SDWriter('generated_structures.sdf')
    for generated_structure in generated_structures:
        generated_molecule = Chem.MolFromSmiles(generated_structure)
        if generated_molecule is not None:
            generated_molecule.UpdatePropertyCache(True)
            AllChem.Compute2DCoords(generated_molecule)
            writer.write(generated_molecule)
    writer.close()
