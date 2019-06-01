# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import pandas as pd

dataset = pd.read_csv('iris_without_species.csv', index_col=0)

autoscaled_dataset = (dataset - dataset.mean()) / dataset.std()  # オートスケーリング
autoscaled_dataset.to_csv('autoscaled_dataset.csv')

basic_statistics = pd.concat(
    [autoscaled_dataset.min(), autoscaled_dataset.median(), autoscaled_dataset.max(),
     autoscaled_dataset.mean(), autoscaled_dataset.var(), autoscaled_dataset.std()],
    axis=1).T
basic_statistics.index = ['min', 'median', 'max', 'mean', 'var', 'std']
basic_statistics.to_csv('basic_statistics_after_autoscaling.csv')
