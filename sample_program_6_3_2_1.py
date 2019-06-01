# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

from rdkit import Chem
from rdkit.Chem import Draw

molecule = Chem.MolFromMolFile('alanine.mol')  # MOL file の読み込み
Draw.MolToFile(molecule, 'molecule.png')

# 右の IPython コンソールに
# Draw.MolToImage(molecule)
# と入力して実行しても、分子を描画できます
