# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:51:53 2023

@author: Mert
"""

from utils import find_csv_files_in_directory

import pandas as pd

import numpy as np

fold_results = find_csv_files_in_directory('./fold_results')
table = []
for fold_result in fold_results :

    fold_result_names = fold_result.split('\\')[-1].split('_')
    data = fold_result_names[0]
    model = fold_result_names[1]

    if 'REG' in fold_result:
        net = fold_result_names[9].split(',')[0].split(' ')[-1]
        beta = fold_result_names[9].split(',')[2].split(' ')[-1]
        act = fold_result_names[8].split(',')[1][1:].split(' ')[-1]
        model = 'VariationalShapley (REG {} {} {})'.format(net, act, beta)

    elif model == 'VariationalShapley':
        net = fold_result_names[8].split(',')[0].split(' ')[-1]
        beta = fold_result_names[8].split(',')[2].split(' ')[-1]
        act = fold_result_names[7].split(',')[1][1:].split(' ')[-1]
        model = 'VariationalShapley ({} {} {})'.format(net, act, beta)
    df = pd.read_csv(fold_result)
    item = [data, model] +  list(df.values[0]) + list(df.values[1:])
    table.append(item)
    
columns = ['Data', 'Model', 'Metric']
columns += ['Fold {}'.format(i) for i in range(1,6)]
table = pd.DataFrame(table, columns=columns)
mean = table.mean(numeric_only=True, axis=1)
std = table.std(numeric_only=True, axis=1) / np.sqrt(5)
table['mean'] = mean
table['std'] = std

a = table[table.Data == 'synthetic5'][['Model','mean','std']]
