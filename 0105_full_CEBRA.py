#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2024/03/22 23:57:02
@Author  :   Coen Berns 
@Version :   1.0
@Contact :   coenjpberns@gmail.com
@License :   (C)Copyright 2024-2025, Coen Berns
@Desc    :   None
'''
#%%
import sys

import pandas as pd
import numpy as np
import cebra
from cebra import CEBRA
import matplotlib.pyplot as plt
import cebra.grid_search
from sklearn.model_selection import train_test_split
import plotly
import plotly.graph_objs as go 

#%% variables and loading
max_iterations = 10000
egg_sw = cebra.load_data(file='0105_all_slowwave.h5', key='0105', columns=[f'Channel {i}' for i in range(8)]) #include timestamps or not??
dfreqs = cebra.load_data(file='0105_DomFreq_segs_of120s_sm_window_n4.h5', key='0105', columns=['DF_avg', 'SW_bool'])
egg_sw_short = egg_sw[0:11000,:]
dfreqs_short = dfreqs[0:11000,:]
#%%
# cebra time - slow wave only
CT_sw_model = CEBRA(model_architecture='offset10-model',
                        batch_size=1024,
                        learning_rate=1e-3,
                        # temperature=.2,
                        temperature_mode='auto',
                        min_temperature=1e-2,
                        output_dimension=6,
                        max_iterations=max_iterations,
                        max_adapt_iterations=1000,
                        distance='cosine',
                        conditional='time',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

# TRAINING
CT_sw_model.fit(egg_sw_short)

CB_sw_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.001,
                        # temperature=.2,
                        temperature_mode='auto',
                        min_temperature=1e-2,
                        output_dimension=6,
                        max_iterations=max_iterations,
                        max_adapt_iterations=1000,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

CB_sw_model.fit(egg_sw_short, dfreqs_short)

CH_sw_model = CEBRA(model_architecture='offset10-model',
                        batch_size=1024,
                        learning_rate=0.001,
                        # temperature=.2,
                        temperature_mode='auto',
                        min_temperature=1e-2,
                        output_dimension=6,
                        max_iterations=max_iterations,
                        max_adapt_iterations=1000,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        hybrid=True,
                        time_offsets=10)

CH_sw_model.fit(egg_sw_short, dfreqs_short)

# %%
CT_emb = CT_sw_model.transform(egg_sw_short)
cebra.plot_embedding_interactive(embedding=CT_emb, embedding_labels='time',cmap='cool')

#%%
CB_emb = CB_sw_model.transform(egg_sw_short)

#%%
CH_emb = CH_sw_model.transform(egg_sw_short)

sw = dfreqs_short[:,1] == 1
no_sw = dfreqs_short[:,1] == 0

ax1 = plt.subplot(projection='3d')
for dir, cmap in zip([sw,no_sw],['cool', 'viridis']):
    ax1 = cebra.plot_embedding_interactive(ax=ax1,embedding=CT_emb[dir,:], embedding_labels=dfreqs_short[dir,0],cmap=cmap)


# %%
