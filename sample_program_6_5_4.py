# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import AllChem

y_name = 'boiling_point'
fingerprint_type = 0  # 0: MACCS key, 1: RDKit, 2: Morgan (≒ECFP4), 3: Avalon

sdf = Chem.SDMolSupplier('boiling_point.sdf')  # sdf ファイルの読み込み

# フィンガープリントの計算
# 分子ごとに、リスト型の変数 y に物性値を、fingerprints に計算されたフィンガープリントを、smiles に SMILES を追加
fingerprints, y, smiles = [], [], []
print('分子の数 :', len(sdf))
for index, molecule in enumerate(sdf):
    print(index + 1, '/', len(sdf))
    y.append(float(molecule.GetProp(y_name)))
    smiles.append(Chem.MolToSmiles(molecule))
    if fingerprint_type == 0:
        fingerprints.append(AllChem.GetMACCSKeysFingerprint(molecule))
    elif fingerprint_type == 1:
        fingerprints.append(Chem.RDKFingerprint(molecule))
    elif fingerprint_type == 2:
        fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048))
    elif fingerprint_type == 3:
        fingerprints.append(GetAvalonFP(molecule))
fingerprints = pd.DataFrame(np.array(fingerprints, int), index=smiles)
y = pd.DataFrame(y, index=smiles, columns=[y_name])

# 保存
fingerprints_with_y = pd.concat([y, fingerprints], axis=1)  # y と記述子を結合
fingerprints_with_y.to_csv('fingerprints_with_y.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
