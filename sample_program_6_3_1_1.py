# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

from rdkit import Chem
from rdkit.Chem import Draw

molecule = Chem.MolFromSmiles('CC(N)C(=O)O')  # SMILES の読み込み。'CC(N)C(=O)O' はアラニン
Draw.MolToFile(molecule, 'molecule.png')

# 右の IPython コンソールに
# Draw.MolToImage(molecule)
# と入力して実行しても、分子を描画できます
