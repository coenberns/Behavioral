#
# Created on Thu Jan 04 2024
#
# Copyright (c) 2024 Berns&Co
#
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


#%% variables and loading
max_iterations = 8000
egg_data = cebra.load_data(file='egg_time_beh_cat_new.h5', key='egg_time', columns=[f'Channel {i}' for i in range(8)]) #include timestamps or not??
cat_labels = cebra.load_data('egg_time_beh_cat_new.h5', key = 'category')
beh_labels_old = cebra.load_data('egg_time_beh_cat_new.h5', key = 'behavior')
beh_labels = cebra.load_data('egg_time_beh_cat_new.h5', key = 'behavior_ambulation_diff')
# behavioral = cebra.load_data(file='', key=None, columns=['Behavior', 'Category'])

#%% Grid search (hyperparam sweep) for optimizing parameters
params_grid = dict(
    batch_size = [512],
    temperature = [0.01,0.05,0.075,0.1],
    learning_rate = [0.003],
    time_offsets = [1],
    max_iterations = 5000,
    verbose = True)

# 2. Fit the models generated from the list of parameters
grid_search = cebra.grid_search.GridSearch()
grid_models, parameter_grid = grid_search.generate_models(params_grid)

#%% Training models for best determining optimal parameters
grid_training = grid_search.fit_models(datasets={"egg_time_data": egg_data, "egg_behavior_data": (egg_data, cat_labels)},
                       params=params_grid,
                       models_dir='grid_search_additional')

#%% Determining best model
# 3. Get model with the best performances and use it as usual
best_model, best_model_name = grid_search.get_best_model(scoring='infonce_loss', models_dir='grid_search_additional')
# embedding = best_model.transform(egg_data)

#%% CEBRA TIME MODEL -- HYPOTHESIS DRIVEN
cebra_time_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.003,
                        temperature=.1,
                        temperature_mode='constant',
                        # min_temperature=.1,
                        output_dimension=3,
                        max_iterations=6000,
                        distance='cosine',
                        conditional='time',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=1)

#%% TRAINING TIME MODEL
cebra_time_model.fit(egg_data)

#%% Decode the data using the model to the latent space
egg_time = cebra_time_model.transform(egg_data)

#%% Plot the embedding of the CEBRA-Time model
cebra.plot_embedding(embedding=egg_time, embedding_labels='time', idx_order=(0,1,2), title='CEBRA-Time', cmap='cebra')

#%%
cebra.plot_loss(cebra_time_model)
cebra.plot_temperature(cebra_time_model)


#%% CEBRA CATEGORY TRAINED MODEL
max_iterations=8000
cebra_category_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.003,
                        # temperature=.5,
                        temperature_mode='auto',
                        min_temperature=1e-1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=6)

# Training with category labels (active,inactive,feeding/drinking)
cebra_category_model.fit(egg_data, cat_labels)
#%%
# Decode into embedding using both egg_time and category labels
egg_cat_emb = cebra_category_model.transform(egg_data)
# Plot the embedding
ax =cebra.plot_embedding(embedding=egg_cat_emb, embedding_labels=cat_labels,cmap='cebra',idx_order=(0,1,2),title='Cebra-Behavior (Categories)')
ax.set_title('CEBRA-Behavior on categorical labels', size=18)
ax.set_xlabel('Latent vector 1', size=15)
ax.set_ylabel('Latent vector 2', size=15)
plt.show()
#%%
# Plot loss and temperature
cebra.plot_loss(cebra_category_model)
cebra.plot_temperature(cebra_category_model)

#%% CEBRA CATEGORY SHUFFLED
max_iterations=8000
shuffled_labels = np.random.permutation(cat_labels)

cebra_cat_shuffled_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.003,
                        # temperature=.5,
                        temperature_mode='auto',
                        min_temperature=1e-1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=6)

cebra_cat_shuffled_model.fit(egg_data, shuffled_labels)

#%% Get shuffled embeddings
shuffled_cat_emb = cebra_cat_shuffled_model.transform(egg_data)
cebra.plot_embedding(embedding=shuffled_cat_emb,embedding_labels=cat_labels, title='CEBRA-Shuffled', idx_order=(0,1))

cebra.plot_loss(cebra_cat_shuffled_model)
cebra.plot_temperature(cebra_cat_shuffled_model)
#%% CEBRA CATEGORIAL MODEL WITH SPLITTING OF TRAIN AND TEST
from sklearn.model_selection import train_test_split

# 1. Split your neural data and auxiliary variable
(
    train_data,
    valid_data,
    train_discrete_label,
    valid_discrete_label,
) = train_test_split(egg_data,
                     cat_labels,
                     shuffle=False,
                     test_size=0.1)

# 2. Train a CEBRA-Behavior model on training data only
cebra_category_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.003,
                        # temperature=.5,
                        temperature_mode='auto',
                        min_temperature=1e-1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=6)

cebra_category_model.fit(train_data, train_discrete_label)


#%%
# 3. Get embedding for training and validation data
train_embedding = cebra_category_model.transform(train_data)
valid_embedding = cebra_category_model.transform(valid_data)

# 4. Train the decoder on training embedding and labels
decoder = cebra.KNNDecoder()
decoder.fit(train_embedding, train_discrete_label)

# 5. Compute the score on validation embedding and labels
score = decoder.score(valid_embedding, valid_discrete_label)
print(score)


cebra.plot_embedding(train_embedding,embedding_labels=train_discrete_label, title='Train Embedding')
cebra.plot_embedding(valid_embedding,embedding_labels=valid_discrete_label, title='Valid Embedding')

# %% USING THE NEW BEHAVIORAL LABELS WITHOUT THE LARGE AMBULATION LABEL
max_iterations=10000
cebra_behavior_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.003,
                        temperature=.5,
                        # temperature_mode='auto',
                        # min_temperature=1e-1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=6)

cebra_behavior_model.fit(egg_data, beh_labels, egg_data.continuous_index.numpy())

#%%
behavior_embedding = cebra_behavior_model.transform(egg_data)
cebra.plot_embedding_interactive(behavior_embedding, embedding_labels=beh_labels, title="CEBRA-Behavior", cmap='viridis')

#%%
cebra.plot_loss(cebra_behavior_model)
cebra.plot_temperature(cebra_behavior_model)
# %%
