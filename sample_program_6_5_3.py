# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import AllChem

fingerprint_type = 0  # 0: MACCS key, 1: RDKit, 2: Morgan (≒ECFP4), 3: Avalon

dataset = pd.read_csv('molecules_with_boiling_point.csv', index_col=0)  # SMILES 付きデータセットの読み込み
smiles = dataset.iloc[:, 0]  # 分子の SMILES
y = dataset.iloc[:, 1]  # 物性・活性などの目的変数

# フィンガープリントの計算
fingerprints = []  # ここに計算されたフィンガープリントを追加
print('分子の数 :', len(smiles))
for index, smiles_i in enumerate(smiles):
    print(index + 1, '/', len(smiles))
    molecule = Chem.MolFromSmiles(smiles_i)
    if fingerprint_type == 0:
        fingerprints.append(AllChem.GetMACCSKeysFingerprint(molecule))
    elif fingerprint_type == 1:
        fingerprints.append(Chem.RDKFingerprint(molecule))
    elif fingerprint_type == 2:
        fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048))
    elif fingerprint_type == 3:
        fingerprints.append(GetAvalonFP(molecule))
fingerprints = pd.DataFrame(np.array(fingerprints, int), index=dataset.index)

# 保存
fingerprints_with_y = pd.concat([y, fingerprints], axis=1)  # y と記述子を結合
fingerprints_with_y.to_csv('fingerprints_with_y.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
