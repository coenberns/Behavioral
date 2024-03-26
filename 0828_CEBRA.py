#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2024/03/21 23:39:11
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
# , columns=[f'Channel {i}_sw' for i in range(8)]+[f'Channel {i}_mmc' for i in range(8)]
max_iterations = 10000
egg_sw_mmc = cebra.load_data(file='egg_sw_mmc_0828_noDT.h5', key='0828')
egg_mmc = cebra.load_data(file='egg_sw_mmc_0828_noDT.h5', key='0828', columns=[f'Channel {i}_mmc' for i in range(8)])
egg_sw = cebra.load_data(file='egg_sw_mmc_0828_noDT.h5', key='0828', columns=[f'Channel {i}_sw' for i in range(8)])
ohe_behavior = cebra.load_data(file='OHE_behaviors_0828.h5', key='0828')
ohe_cats = cebra.load_data(file='OHE_categories_0828.h5', key='0828')
cats_factorized = cebra.load_data(file='categories_factorized_0_2_0828.npy')
egg_sw_dfreqs = cebra.load_data(file='0828_dfreqs_swbool.h5', key='0828')
feeding = cebra.load_data('feeding_0828.npy')
active = cebra.load_data('active_0828.npy')
inactive = cebra.load_data('inactive_0828.npy')
egg_sw_timestamps = cebra.load_data(file='egg_sw_mmc_0828_noDT.h5', key='0828', columns=['Synctime'])

# ohe_feeding = cebra.load_data(file='OHE_categories_0828.h5', key='0828', columns='Cat_0')
# ohe_active = cebra.load_data(file='OHE_categories_0828.h5', key='0828', columns='Cat_1')
# ohe_inactive = cebra.load_data(file='OHE_categories_0828.h5', key='0828', columns='Cat_2')


#%% CEBRA TIME MODELS -- HYPOTHESIS DRIVEN - slow wave and mmc
max_iterations=10000
CT_sw_mmc_model = CEBRA(model_architecture='offset10-model',
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

CT_sw_mmc_model.fit(egg_sw_mmc)
#%%
# cebra time - slow wave only
CT_sw_model = CEBRA(model_architecture='offset5-model',
                        batch_size=512,
                        learning_rate=0.003,
                        # temperature=.2,
                        temperature_mode='auto',
                        min_temperature=1e-2,
                        output_dimension=9,
                        max_iterations=max_iterations,
                        max_adapt_iterations=1000,
                        distance='cosine',
                        conditional='time',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=5)

# TRAINING TIME MODEL
CT_sw_model.fit(egg_sw)
#%%
# Cebra time - mmc only
CT_mmc_model = CEBRA(model_architecture='offset10-model',
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
CT_mmc_model.fit(egg_mmc)
# %%
sw_mmc_time_emb = CT_sw_mmc_model.transform(egg_sw_mmc)
sw_time_emb = CT_sw_model.transform(egg_sw)
mmc_time_emb = CT_mmc_model.transform(egg_mmc)

#%%
sw_time_emb = CT_sw_model.transform(egg_sw)
cebra.plot_embedding_interactive(embedding=sw_time_emb, embedding_labels='time', cmap='cool')
#%%
cebra.plot_embedding_interactive(sw_mmc_time_emb[inactive], embedding_labels='time',cmap='cool')
#%%
cebra.plot_embedding_interactive(sw_time_emb[feeding], embedding_labels='time',cmap='cool')
#%%
cebra.plot_embedding_interactive(mmc_time_emb[inactive], embedding_labels='time',cmap='cool')

# %% Behavioral model based on "continuous" frequency labels
max_iterations=8000
cebra_cats_sw_model = CEBRA(model_architecture='offset5-model',
                        batch_size=512,
                        learning_rate=0.003,
                        # temperature=.2,
                        temperature_mode='auto',
                        min_temperature=1e-2,
                        output_dimension=9,
                        num_hidden_units=64,
                        max_iterations=max_iterations,
                        max_adapt_iterations=1000,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        hybrid=True,
                        time_offsets=5)

# TRAINING TIME MODEL
feeding_active = feeding | active
cebra_cats_sw_model.fit(egg_sw, egg_sw_dfreqs, feeding_active)

#%% Embedding
sw_cat_emb = cebra_cats_sw_model.transform(egg_sw)
sw = egg_sw_dfreqs[:,0] == 1
non_sw = egg_sw_dfreqs[:,0] == 0


cebra.plot_embedding_interactive(embedding=sw_cat_emb, embedding_labels=egg_sw_dfreqs[:,1], cmap='winter', idx_order=(0,1,2))
#%%


fig = plt.figure(figsize=(8,8))

ax1=plt.subplot(projection='3d')

for dir,cmap in zip([sw, non_sw], ['winter',  'Spectral']):
    cebra.plot_embedding(ax=ax1,embedding=sw_cat_emb[dir,:],embedding_labels=egg_sw_dfreqs[:,1][dir],cmap=cmap,idx_order=(0,1,2),title='Cebra-Behavior (Categories)')
# %%
