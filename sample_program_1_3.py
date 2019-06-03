# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd

dataset = pd.read_csv('iris.csv', index_col=0)
dataset.to_csv('iris_new.csv')  # csv ファイルとして保存
