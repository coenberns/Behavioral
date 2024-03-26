#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2024/03/21 18:58:03
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
egg_sw_mmc = cebra.load_data(file='seg2_sw_mmc_3_5hrs_0104.h5', key='seg2') #include timestamps or not??
egg_sw_mmc_notime = cebra.load_data(file='seg2_sw_mmc_3_5hrs_0104.h5', key='seg2', columns=[f'Channel {i}_sw' for i in range(8)]+[f'Channel {i}_mmc' for i in range(8)])
# , columns=[f'Channel {i}' for i in range(8)]
egg_freq_sw = cebra.load_data(file='freq_data_seg2_0104.h5', key='freq')


#%% CEBRA TIME MODEL -- HYPOTHESIS DRIVEN
max_iterations=10000
cebra_time_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.003,
                        temperature=.2,
                        # temperature_mode='auto',
                        # min_temperature=1e-2,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        max_adapt_iterations=1000,
                        distance='cosine',
                        conditional='time',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=6)

# TRAINING TIME MODEL
cebra_time_model.fit(egg_sw_mmc)
# %%
egg_time_emb = cebra_time_model.transform(egg_sw_mmc)
fig1 = cebra.plot_embedding_interactive(egg_time_emb, embedding_labels='time',cmap='cool')
# %%
#%% CEBRA TIME MODEL -- timestamps not included
max_iterations=10000
cebra_time_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.003,
                        temperature=.2,
                        # temperature_mode='auto',
                        # min_temperature=1e-2,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        max_adapt_iterations=1000,
                        distance='cosine',
                        conditional='time',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=6)

# TRAINING TIME MODEL
cebra_time_model.fit(egg_sw_mmc_notime)
# %%
egg_notime_emb = cebra_time_model.transform(egg_sw_mmc_notime)
fig2 = cebra.plot_embedding_interactive(egg_notime_emb, embedding_labels='time',cmap='cool')

# %%
fig2.show()
# %% Behavioral model based on "continuous" frequency labels
max_iterations=10000
cebra_behavioral_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.003,
                        temperature=.2,
                        # temperature_mode='auto',
                        # min_temperature=1e-2,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        max_adapt_iterations=1000,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=6)

# TRAINING TIME MODEL
cebra_behavioral_model.fit(egg_sw_mmc, egg_freq_sw)

#%% Embedding
sw_mask = (egg_freq_sw[:, 0] > 2) & (egg_freq_sw[:, 0] < 5)
egg_sw_freq_emb = cebra_behavioral_model.transform(egg_sw_mmc)
cebra.plot_embedding_interactive(egg_sw_freq_emb, embedding_labels=egg_freq_sw[sw_mask], cmap='cool')

#%% Time model based solely on frequency
max_iterations=5000
cebra_freq_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.003,
                        temperature=.2,
                        # temperature_mode='auto',
                        # min_temperature=1e-2,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        max_adapt_iterations=1000,
                        distance='cosine',
                        conditional='time',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=6)

# TRAINING TIME MODEL
cebra_freq_model.fit(egg_freq_sw)

#%%
egg_freq_emb = cebra_freq_model.transform(egg_freq_sw)
cebra.plot_embedding_interactive(embedding=egg_freq_emb, embedding_labels='time', cmap='cool')
# %%


