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
timesteps = cebra.load_data(file='egg_time_beh_cat_new.h5', key='egg_time', columns=['timestamps'])
cat_labels = cebra.load_data('egg_time_beh_cat_new.h5', key = 'category')
beh_labels_old = cebra.load_data('egg_time_beh_cat_new.h5', key = 'behavior')
beh_labels = cebra.load_data('egg_time_beh_cat_new.h5', key = 'behavior_ambulation_diff')
timesteps_np = np.array(np.arange(0, egg_data.shape[0]).flatten().tolist())

#%% Grid search (hyperparam sweep) for optimizing parameters
# TAKES A LONG TIME BASED ON AMOUNT OF PARAMETERS
params_grid = dict(
    batch_size = [512,1024],
    learning_rate = [3e-3],
    temperature_mode='auto',
    min_temperature=.1,
    time_offsets = [1,5,10],
    max_iterations = 8000,
    verbose = True)

# 2. Fit the models generated from the list of parameters
grid_search = cebra.grid_search.GridSearch()
grid_models, parameter_grid = grid_search.generate_models(params_grid)

#%% WARNING THIS STEP TAKES A CONSIDERABLE AMOUNT OF TIME
# Training models for best determining optimal parameters
grid_training = grid_search.fit_models(datasets={"egg_time_data": egg_data, "egg_cat_data": (egg_data, cat_labels), "egg_beh_data": (egg_data, beh_labels)},
                       params=params_grid,
                       models_dir='grid_search_0124')

# Determining best model
best_model, best_model_name = grid_search.get_best_model(scoring='infonce_loss', models_dir='grid_search_0124')
# embedding = best_model.transform(egg_data)