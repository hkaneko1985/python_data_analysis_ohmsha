# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd  # Pandas の取り込み。一般的に pd と名前を省略して取り込みます

dataset = pd.read_csv('iris.csv', encoding='SHIFT-JIS', index_col=0)  # データセットの読み込み
