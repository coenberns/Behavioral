#
# Created on Wed Jan 24 2024
#
# Copyright (c) 2024 Berns&Co
#
# GRID SEARCH FOR RIGHT CEBRA PARAMETERS - I.E. HYPER PARAMETER SWEEP
#%% imports
import sys

import pandas as pd
import numpy as np
import cebra
from cebra import CEBRA
import matplotlib.pyplot as plt
import cebra.grid_search
from sklearn.model_selection import train_test_split
import plotly


#%% DATA LOADING
max_iterations = 8000
egg_data = cebra.load_data(file='egg_time_beh_cat_new.h5', key='egg_time', columns=[f'Channel {i}' for i in range(8)]) #include timestamps or not??
cat_labels = cebra.load_data('egg_time_beh_cat_new.h5', key = 'category')
egg_sw = cebra.load_data(file='0105_all_slowwave.h5', key='0105',columns=[f'Channel {i}' for i in range(8)]) #include timestamps or not??
dfreqs = cebra.load_data(file='0105_DomFreq_segs_of120s_sm_window_n4.h5', key='0105', columns=['DF_avg', 'SW_bool'])
egg_sw_6hday = egg_sw[0:11000,:]
dfreqs_6hday = dfreqs[0:11000,:]
egg_sw_6hnight = egg_sw[11000:22000,:]
dfreqs_6hnight = dfreqs[11000:22000,:]


#%% Grid search (hyperparam sweep) for optimizing parameters
# TAKES A LONG TIME BASED ON AMOUNT OF PARAMETERS
params_grid = dict(
        model_architecture='offset5-model',
        batch_size = [512],
        learning_rate = [3e-3],
        temperature_mode= 'auto',
        min_temperature=1e-2,
        time_offsets = [5,10],
        num_hidden_units = [32,64],
        output_dimension = [9],
        max_iterations = [10000],
        verbose = True
    )

# 2. Fit the models generated from the list of parameters
grid_search = cebra.grid_search.GridSearch()
grid_models, parameter_grid = grid_search.generate_models(params_grid)

#%% WARNING THIS STEP TAKES A CONSIDERABLE AMOUNT OF TIME
# Training models for best determining optimal parameters
grid_training = grid_search.fit_models(datasets=
                                        {
                                        "0105_egg_dfreqs_day": (egg_sw_6hday, dfreqs_6hday),
                                        "0105_egg_dfreqs_night": (egg_sw_6hnight, dfreqs_6hnight)
                                        },
                       params=params_grid,
                       models_dir='grid_0325_offset5_0105_daynight')

#%%
# Determining best model
best_model, best_model_name = grid_search.get_best_model(scoring='infonce_loss', models_dir='grid_0325_offset5_0105_daynight')
# embedding = best_model.transform(egg_data)
# %%
