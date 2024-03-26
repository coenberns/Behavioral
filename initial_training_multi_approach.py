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
import plotly.graph_objs as go 

#%% variables and loading
max_iterations = 8000
egg_data = cebra.load_data(file='egg_time_beh_cat_new.h5', key='egg_time', columns=[f'Channel {i}' for i in range(8)]) #include timestamps or not??
timesteps = cebra.load_data(file='egg_time_beh_cat_new.h5', key='egg_time', columns=['timestamps'])
cat_labels = cebra.load_data('egg_time_beh_cat_new.h5', key = 'category')
beh_labels_old = cebra.load_data('egg_time_beh_cat_new.h5', key = 'behavior')
beh_labels = cebra.load_data('egg_time_beh_cat_new.h5', key = 'behavior_ambulation_diff')
timesteps_np = np.array(np.arange(0, egg_data.shape[0]).flatten().tolist())
part_2 = cebra.load_data(file='08292023_part2_pieces.h5',key='larger_filt', columns=[f'Channel {i}' for i in range(8)])
part_2_cats = cebra.load_data(file='08292023_part2_pieces.h5',key='larger_filt', columns=['cat_code']).flatten()

# behavioral = cebra.load_data(file='', key=None, columns=['Behavior', 'Category'])
c_feeding = cat_labels == 0
c_active = cat_labels == 1
c_inactive = cat_labels == 2

c2_feeding = part_2_cats == 0
c2_active = part_2_cats == 1
c2_inactive = part_2_cats == 2

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
cebra_time_model.fit(egg_data)

#%%
cebra_time_model_adapt = CEBRA.load('Trained models/cebra_time_model.pt')
cebra_time_model_adapt.fit(part_2,adapt=True)

# CEBRA.save(cebra_time_model, filename='Trained models/cebra_time_model.pt')
#%% Decode the datato the latent space using the model
egg_time_emb = cebra_time_model_adapt.transform(part_2)
cebra.plot_embedding_interactive(egg_time_emb, embedding_labels='time',cmap='cool')
# print(len(egg_time_emb)==len(part_2_cats))

#%% Plot the embedding of the CEBRA-Time model
fig1 = cebra.plot_embedding_interactive(embedding=egg_time_emb[c2_feeding], embedding_labels=part_2_cats[c2_feeding], idx_order=(0,1,2), title='CEBRA-Time', cmap='cool')
fig2 = cebra.plot_embedding_interactive(embedding=egg_time_emb[c2_active], embedding_labels=part_2_cats[c2_active], idx_order=(0,1,2), title='CEBRA-Time', cmap='viridis')
fig3 = cebra.plot_embedding_interactive(embedding=egg_time_emb[c2_inactive], embedding_labels=part_2_cats[c2_inactive], idx_order=(0,1,2), title='CEBRA-Time', cmap='magma')

#%% Interactive plotting with multiple 3D scatter plots
markersize=1
alpha=1
feeding_trace = go.Scatter3d(
    x=egg_time_emb[c2_feeding, 0],
    y=egg_time_emb[c2_feeding, 1],
    z=egg_time_emb[c2_feeding, 2],
    mode='markers',
    marker=dict(
        size=markersize,
        color='#db24ff', 
        colorscale='magma',
        opacity=alpha
    ),
    name='Feeding'  
)

active_trace = go.Scatter3d(
    x=egg_time_emb[c2_active,0],
    y=egg_time_emb[c2_active, 1],
    z=egg_time_emb[c2_active, 2],
    mode='markers',
    marker=dict(
        size=markersize,
        color='#24dbff',  
        colorscale='magma',  
        opacity=alpha
    ),
    name='Active'  
)

inactive_trace = go.Scatter3d(
    x=egg_time_emb[c2_inactive, 0],
    y=egg_time_emb[c2_inactive, 1],
    z=egg_time_emb[c2_inactive, 2],
    mode='markers',
    marker=dict(
        size=markersize,
        color='#46327e',  
        colorscale='Viridis',  
        opacity=alpha
    ),
    name='Inactive' 
)

fig = go.Figure()
fig.add_trace(feeding_trace)
fig.add_trace(active_trace)
fig.add_trace(inactive_trace)

# add layout
fig.update_layout(
    width=600,
    height=600,
    title=dict(
        text='CEBRA-Time - Colored on Category',
        font=dict(
            family="Arial, sans-serif", 
            size=24
        ),
        x=0.45
    ),
    legend=dict(
        title=dict(text='Categories'),
        itemsizing='constant',
        font=dict(
            size=16
        )
    ),
    scene=dict(
        bgcolor='white',
        xaxis=dict(title='Latent 0', title_font=dict(size=16), 
                   showbackground=False, showgrid=True, linecolor='black'),
        yaxis=dict(title='Latent 1', title_font=dict(size=16), 
                   showbackground=False, showgrid=True, linecolor='black'),
        zaxis=dict(title='Latent 2', title_font=dict(size=16), 
                   showbackground=False, showgrid=True, linecolor='black'),
    )    
    # scene=dict(
    #     xaxis=dict(title='Latent 0', title_font=dict(size=16)),
    #     yaxis=dict(title='Latent 1', title_font=dict(size=16)),
    #     zaxis=dict(title='Latent 2', title_font=dict(size=16))
    # )
)
# Show the figure
fig.show()
# fig.write_html('cebra_time.html')


# Plot the loss over time for this model, see if it converges
#cebra.plot_loss(cebra_time_model)
## Could also plot the temperature vs time, if it is not constant
# cebra.plot_temperature(cebra_time_model)


#%% CEBRA BEHAVIOR CATEGORY TRAINED MODEL
max_iterations=10000
indices = np.arange(egg_data.shape[0])
time_cat1 = np.column_stack((timesteps,cat_labels))
cebra_category_model = CEBRA(model_architecture='offset5-model',
                        batch_size=512,
                        learning_rate=3e-3,
                        # temperature=.2,
                        temperature_mode='auto',
                        min_temperature=1e-2,
                        output_dimension=9,
                        num_hidden_units=64,
                        max_iterations=max_iterations,
                        max_adapt_iterations=2000,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=5)

# Training with category labels (active,inactive,feeding/drinking)
cebra_category_model.fit(egg_data, cat_labels)

#%% Save trained model - adapt trained model after loading
CEBRA.save(cebra_category_model, filename='Trained models/cebra_cat_model_new0324.pt')
cebra_category_model = CEBRA.load("Trained models/cebra_cat_model_new0324.pt")
## Adapting a model for a different dataset
#cebra_category_model_adapt = CEBRA.load("Trained models/cebra_cat_model_new0324.pt")
# cebra_category_model_adapt.fit(part_2, adapt=True)

#%%
# Decode into embedding for the egg_data
embedding_categg = cebra_category_model.transform(egg_data)
cebra.plot_embedding_interactive(embedding=embedding_categg, embedding_labels='time', cmap='cool', idx_order=(2,3,4))
#%%
fig = plt.figure(figsize=(20,20))

ax1=plt.subplot(111,projection='3d')

for dir,cmap in zip([c_feeding,c_active,c_inactive], ['cool', 'spring', 'winter']):
    cebra.plot_embedding(ax=ax1,embedding=embedding_categg[dir,:],embedding_labels=cat_labels[dir],cmap=cmap,idx_order=(0,1,2),title='Cebra-Behavior (Categories)')
    # plt.legend(cmap)

# Set the view angle for better visualization
ax1.view_init(elev=10, azim=340)

# Customize the ticks on each axis for clarity
ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
ax1.set_zticks([-1, -0.5, 0, 0.5, 1])
#%%# Plot the embedding
combined = c_feeding | c_active

fig = cebra.plot_embedding_interactive(embedding=embedding_categg[combined], 
                                       embedding_labels=cat_labels[combined],
                                       idx_order=(0,1,2),cmap='cool',title='Cebra-Behavior (Categories)', name='Active/Feeding')

inactive_trace = go.Scatter3d(
    x=embedding_categg[c_inactive, 0],
    y=embedding_categg[c_inactive, 1],
    z=embedding_categg[c_inactive, 2],
    mode='markers',
    marker=dict(
        size=1,
        color=cat_labels[c_inactive],
        colorscale='viridis', 
        opacity=1
    ),
    name='Inactive',
    showlegend=True
)

# Add the inactive trace to the figure
fig.add_trace(inactive_trace)
# Show the figure
fig.show()
# ax.set_title('CEBRA-Behavior on categorical labels', size=18)
# ax.set_xlabel('Latent vector 1', size=15)
# ax.set_ylabel('Latent vector 2', size=15)
# plt.show()
#%% MANUALLY plot the three different categories in a interactive plot with labels
markersize=1
alpha=1
#3d scatter feeding
feeding_trace = go.Scatter3d(
    x=embedding_categg[c_feeding, 0],
    y=embedding_categg[c_feeding, 1],
    z=embedding_categg[c_feeding, 2],
    mode='markers',
    marker=dict(
        size=markersize,
        color='#db24ff',
        colorscale='magma',
        opacity=alpha
    ),
    name='Feeding'
)

#3d scatter inactive
active_trace = go.Scatter3d(
    x=embedding_categg[c_active,0],
    y=embedding_categg[c_active, 1],
    z=embedding_categg[c_active, 2],
    mode='markers',
    marker=dict(
        size=markersize,
        color='#24dbff',
        colorscale='magma',
        opacity=alpha
    ),
    name='Active'
)

#3d scatter inactive
inactive_trace = go.Scatter3d(
    x=embedding_categg[c_inactive, 0],
    y=embedding_categg[c_inactive, 1],
    z=embedding_categg[c_inactive, 2],
    mode='markers',
    marker=dict(
        size=markersize,
        color='#46327e',
        colorscale='Viridis',
        opacity=alpha
    ),
    name='Inactive'
)

#3D plotly figure with all traces
fig = go.Figure()
fig.add_trace(feeding_trace)
fig.add_trace(active_trace)
fig.add_trace(inactive_trace)

# add layout
fig.update_layout(
    width=600,
    height=600,
    title=dict(
        text='CEBRA-Behavior - Category labels',
        font=dict(
            family="Arial, sans-serif", 
            size=24
        ),
        x=0.45
    ),
    legend=dict(
        title=dict(text='Categories'),
        itemsizing='constant',
        font=dict(
            size=16
        )
    ),
    scene=dict(
        bgcolor='white',
        xaxis=dict(title='Latent 0', title_font=dict(size=16), 
                   showbackground=False, showgrid=True, linecolor='black'),
        yaxis=dict(title='Latent 1', title_font=dict(size=16), 
                   showbackground=False, showgrid=True, linecolor='black'),
        zaxis=dict(title='Latent 2', title_font=dict(size=16), 
                   showbackground=False, showgrid=True, linecolor='black'),
    )
)
# Show the figure
fig.show()
# fig.write_html('cebra_cat.html')


#%%
# Plot loss and temperature
cebra.plot_loss(cebra_category_model)
cebra.plot_temperature(cebra_category_model)

#%% CEBRA CATEGORY SHUFFLED
max_iterations=10000
shuffled_cat = np.random.permutation(cat_labels)

cebra_cat_shuffled_model = CEBRA(model_architecture='offset5-model',
                        batch_size=512,
                        learning_rate=0.003,
                        temperature=.2,
                        # temperature_mode='auto',
                        # min_temperature=1e-2,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=6)

cebra_cat_shuffled_model.fit(egg_data, shuffled_cat)

#%% Save shuffled model
# CEBRA.save(cebra_cat_shuffled_model, filename='Trained models/cebra_cat_shuffle_model.pt')

#%% Get shuffled embeddings
shuffled_cat_emb = cebra_cat_shuffled_model.transform(egg_data)
shuffled_feeding = shuffled_cat == 0
shuffled_active = shuffled_cat == 1
shuffled_inactive = shuffled_cat == 2
#%%
cebra.plot_embedding_interactive(embedding=shuffled_cat_emb,embedding_labels=shuffled_cat,cmap='cool',title='CEBRA-Shuffled', idx_order=(0,1,2))

#%%
markersize=1
alpha=1
feeding_trace_shuf = go.Scatter3d(
    x=shuffled_cat_emb[shuffled_feeding,  0],
    y=shuffled_cat_emb[shuffled_feeding,  1],
    z=shuffled_cat_emb[shuffled_feeding, 2],
    mode='markers',
    marker=dict(
        size=markersize,
        color='#24dbff',
        colorscale='magma',
        opacity=alpha
    ),
    name='Feeding'
)

active_trace_shuf = go.Scatter3d(
    x=shuffled_cat_emb[shuffled_active,0],
    y=shuffled_cat_emb[shuffled_active, 1],
    z=shuffled_cat_emb[shuffled_active, 2],
    mode='markers',
    marker=dict(
        size=markersize,
        color='#db24ff',  
        colorscale='magma', 
        opacity=alpha
    ),
    name='Active'
)

inactive_trace_shuf = go.Scatter3d(
    x=shuffled_cat_emb[shuffled_inactive, 0],
    y=shuffled_cat_emb[shuffled_inactive, 1],
    z=shuffled_cat_emb[shuffled_inactive, 2],
    mode='markers',
    marker=dict(
        size=markersize,
        color='#46327e',
        colorscale='Viridis', 
        opacity=alpha
    ),
    name='Inactive' 
)

#add all traces to 3D scatterplot
fig = go.Figure()
fig.add_trace(feeding_trace_shuf)
fig.add_trace(active_trace_shuf)
fig.add_trace(inactive_trace_shuf)

# add layout
fig.update_layout(
    width=600,
    height=600,
    title=dict(
        text='CEBRA-Behavior - Shuffled labels',
        font=dict(
            family="Arial, sans-serif", 
            size=24
        ),
        x=0.45
    ),
    legend=dict(
        title=dict(text='Categories'),
        itemsizing='constant',
        font=dict(
            size=16
        )
    ),
    scene=dict(
        bgcolor='white',
        xaxis=dict(title='Latent 0', title_font=dict(size=16), 
                   showbackground=False, showgrid=True, linecolor='black'),
        yaxis=dict(title='Latent 1', title_font=dict(size=16), 
                   showbackground=False, showgrid=True, linecolor='black'),
        zaxis=dict(title='Latent 2', title_font=dict(size=16), 
                   showbackground=False, showgrid=True, linecolor='black'),
    )
)
# Show the figure
fig.show()
fig.write_html('cebra_shuffle.html')

#%% Plotting the loss for the three different models  
# -- Additionally, we need a hybrid trained model based for instance on a tensor with:
# --> 

fig = plt.figure(figsize=(5,4))
ax = plt.subplot(111)

ax = cebra.plot_loss(cebra_time_model, color='deepskyblue', label='time only', ax=ax)
ax = cebra.plot_loss(cebra_category_model, color='deepskyblue', alpha=0.3, label='categorical labels', ax=ax)
ax = cebra.plot_loss(cebra_cat_shuffled_model, color='gray', alpha=0.6,label='shuffled labels', ax=ax)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('InfoNCE loss vs training iterations', fontdict=dict(family="Arial", size=14))
ax.set_xlabel('Iterations')
ax.set_ylabel('InfoNCE Loss')
plt.legend(bbox_to_anchor=(0.5,0.3), frameon = False)
plt.show()



#%%
#%% CEBRA CATEGORY TRAINED HYBRID MODEL
max_iterations=6000
time_cat1 = np.column_stack((timesteps,cat_labels))
hybrid_cat_model = CEBRA(model_architecture='offset10-model',
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

# Training with category labels (active,inactive,feeding/drinking)
hybrid_cat_model.fit(egg_data, cat_labels)
#%%
# Decode into embedding using both egg_time and category labels
egg_cat_emb_hyb = hybrid_cat_model.transform(egg_data)

#%%
active1 = time_cat1[:,1] == 0
feeding1 = time_cat1[:,1] == 1
inactive1 = time_cat1[:,1] == 2

cebra.plot_embedding_interactive(embedding=egg_cat_emb_hyb[inactive1,:], embedding_labels=time_cat1[inactive1,0])

#%%# Plot the embedding
cebra.plot_embedding_interactive(embedding=egg_cat_emb_hyb, embedding_labels=cat_labels,cmap='viridis',title='Cebra-Behavior (Categories)')
# ax.set_title('CEBRA-Behavior on categorical labels', size=18)
# ax.set_xlabel('Latent vector 1', size=15)
# ax.set_ylabel('Latent vector 2', size=15)
# plt.show()
#%%
# Plot loss and temperature
cebra.plot_loss(hybrid_cat_model)
cebra.plot_temperature(hybrid_cat_model)

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
max_iterations= 8000
cebra_behavior_model = CEBRA(model_architecture='offset10-model',
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

cebra_behavior_model.fit(egg_data, beh_labels)

#%%
behavior_embedding = cebra_behavior_model.transform(egg_data)
#%%
cebra.plot_embedding_interactive(behavior_embedding, embedding_labels=beh_labels, title="CEBRA-Behavior", cmap='viridis',idx_order=(0,1,2))

#%%
cebra.plot_loss(cebra_behavior_model)
# cebra.plot_temperature(cebra_behavior_model)

# %% CEBRA Behavioral Labels with shuffling
max_iterations = 8000
shuffled_behavior = np.random.permutation(beh_labels)

cebra_beh_shuffled_model = CEBRA(model_architecture='offset10-model',
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

cebra_beh_shuffled_model.fit(egg_data, shuffled_behavior)

#%%
shuffled_beh_emb = cebra_beh_shuffled_model.transform(egg_data)
cebra.plot_embedding(embedding=shuffled_beh_emb,embedding_labels=shuffled_behavior,cmap='viridis',
                        idx_order=(0,1,2),title='CEBRA-Behavior Shuffled')

#%%
cebra.plot_loss(cebra_beh_shuffled_model)
cebra.plot_temperature(cebra_beh_shuffled_model)

