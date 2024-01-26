#
# Created on Thu Jan 25 2024
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
max_iterations = 6000
egg_data2 = cebra.load_data(file='part2_0829_egg_time_beh_cat.h5', key='all', columns=[f'Channel {i}' for i in range(8)]) #include timestamps or not??
egg_data2_cat = cebra.load_data(file='part2_0829_egg_time_beh_cat.h5', key='all', columns=['cat_code']+[f'Channel {i}' for i in range(8)])
egg_data2_time = cebra.load_data(file='part2_0829_egg_time_beh_cat.h5', key='all', columns=['timestamps']+[f'Channel {i}' for i in range(8)])
time_cat_data = cebra.load_data(file='part2_0829_egg_time_beh_cat.h5', key='all', columns=['timestamps', 'cat_code'])
timesteps2 = cebra.load_data(file='part2_0829_egg_time_beh_cat.h5', key='all', columns=['timestamps'])
cat_labels2 = cebra.load_data('part2_0829_egg_time_beh_cat.h5', key = 'all', columns=['cat_code']).flatten()
beh_labels2 = cebra.load_data('part2_0829_egg_time_beh_cat.h5', key = 'all', columns=['beh_code']).flatten()
timesteps_np2 = np.array(np.arange(0, egg_data2.shape[0]).flatten().tolist())
# behavioral = cebra.load_data(file='', key=None, columns=['Behavior', 'Category'])

#%% CEBRA TIME MODEL -- HYPOTHESIS DRIVEN
max_iterations=6000
cebra_time_model_2 = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.003,
                        # temperature=.1,
                        temperature_mode='auto',
                        min_temperature=.1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=6)

# TRAINING TIME MODEL
cebra_time_model_2.fit(egg_data2)

#%% Decode the datato the latent space using the model
egg_time_emb = cebra_time_model_2.transform(egg_data2)
# Plot the embedding of the CEBRA-Time model
cebra.plot_embedding_interactive(embedding=egg_time_emb, embedding_labels='time', idx_order=(0,1,2), title='CEBRA-Time', cmap='cebra')

#%% CEBRA CATEGORY TRAINED MODEL
max_iterations=4000
cebra_category_model2 = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.003,
                        # temperature=.1,
                        temperature_mode='auto',
                        min_temperature=1e-1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=6,
                        hybrid=True)

# Training with category labels (active,inactive,feeding/drinking)
cebra_category_model2.fit(egg_data2,time_cat_data)
#%%
# Decode into embedding using both egg_time and category labels
egg_cat_emb2_hyb = cebra_category_model2.transform(egg_data2_cat)

#%%# Plot the embedding
active = egg_data2_cat[:,0] == 0
feed = egg_data2_cat[:,0] == 1
inactive = egg_data2_cat[:,0] == 2

fig = plt.figure(figsize=(10,10))

ax1 = plt.subplot(111, projection='3d')


for dir, cmap in zip([feed], ["rainbow"]):
    ax1=cebra.plot_embedding(ax=ax1, embedding=egg_cat_emb2_hyb[dir,:], embedding_labels=cat_labels2[dir], title='CEBRA-Behavior', cmap=cmap)
  
# cebra.plot_embedding_interactive(embedding=egg_cat_emb2_hyb, embedding_labels=cat_labels2,cmap='cebra',title='Cebra-Behavior (Categories)')
# ax.set_title('CEBRA-Behavior on categorical labels', size=18)
# ax.set_xlabel('Latent vector 1', size=15)
# ax.set_ylabel('Latent vector 2', size=15)
# plt.show()
#%%
# Plot loss and temperature
cebra.plot_loss(cebra_category_model2)
cebra.plot_temperature(cebra_category_model2)

#%% CEBRA CATEGORY SHUFFLED
max_iterations=5000
shuffled_cat2 = np.random.permutation(cat_labels2)

cebra_cat_shuffled_model2 = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.003,
                        # temperature=.1,
                        temperature_mode='auto',
                        min_temperature=1e-1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=6)

cebra_cat_shuffled_model2.fit(egg_data2, shuffled_cat2)

#%% Get shuffled embeddings
shuffled_cat_emb2 = cebra_cat_shuffled_model2.transform(egg_data2)
#%%
cebra.plot_embedding_interactive(embedding=shuffled_cat_emb2,embedding_labels=shuffled_cat2,cmap='cebra',title='CEBRA-Shuffled', idx_order=(0,1,2))
#%%
cebra.plot_loss(cebra_cat_shuffled_model2)

# %% USING THE NEW BEHAVIORAL LABELS WITHOUT THE LARGE AMBULATION LABEL
max_iterations= 8000
cebra_behavior_model2 = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.003,
                        temperature=.1,
                        # temperature_mode='auto',
                        # min_temperature=1e-1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=6)

cebra_behavior_model2.fit(egg_data2, beh_labels2)

#%%
behavior_embedding2 = cebra_behavior_model2.transform(egg_data2)
#%%
cebra.plot_embedding_interactive(behavior_embedding2, embedding_labels=beh_labels2, title="CEBRA-Behavior", cmap='viridis',idx_order=(0,1,2))

#%%
cebra.plot_loss(cebra_behavior_model2)
# cebra.plot_temperature(cebra_behavior_model)

# %% CEBRA Behavioral Labels with shuffling
max_iterations = 8000
shuffled_behavior2 = np.random.permutation(beh_labels2)

cebra_beh_shuffled_model2 = CEBRA(model_architecture='offset10-model',
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

cebra_beh_shuffled_model2.fit(egg_data2, shuffled_behavior2)

#%%
shuffled_beh_emb = cebra_beh_shuffled_model2.transform(egg_data2)
cebra.plot_embedding(embedding=shuffled_beh_emb,embedding_labels=shuffled_behavior2,cmap='viridis',
                        idx_order=(0,1,2),title='CEBRA-Behavior Shuffled')

#%%
cebra.plot_loss(cebra_beh_shuffled_model2)
cebra.plot_temperature(cebra_beh_shuffled_model2)