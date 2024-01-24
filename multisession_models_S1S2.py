#
# Created on Wed Jan 24 2024
#
# Copyright (c) 2024 Berns&Co
#
# A multi-session model to use for the two subject pigs we used
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
import pickle

#%% Variable definition and data loading

hippocampus_a = cebra.datasets.init('rat-hippocampus-single-achilles')
hippocampus_b = cebra.datasets.init('rat-hippocampus-single-buddy')
hippocampus_c = cebra.datasets.init('rat-hippocampus-single-cicero')
hippocampus_g = cebra.datasets.init('rat-hippocampus-single-gatsby')

names = ["achilles", "buddy", "cicero", "gatsby"]
datas = [hippocampus_a.neural.numpy(), hippocampus_b.neural.numpy(), hippocampus_c.neural.numpy(), hippocampus_g.neural.numpy()]
labels = [hippocampus_a.continuous_index.numpy(), hippocampus_b.continuous_index.numpy(), hippocampus_c.continuous_index.numpy(), hippocampus_g.continuous_index.numpy()]

max_iter = 8000

#%% SINGLE SESSION
embeddings = dict()

# Single session training
for name, X, y in zip(names, datas, labels):
    # Fit one CEBRA model per session (i.e., per rat)
    print(f"Fitting CEBRA for {name}")
    cebra_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-3,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=max_iter,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

    cebra_model.fit(X, y)
    embeddings[name] = cebra_model.transform(X)


# Align the single session embeddings to the first rat
alignment = cebra.data.helper.OrthogonalProcrustesAlignment()
first_rat = list(embeddings.keys())[0]

for j, rat_name in enumerate(list(embeddings.keys())[1:]):
    embeddings[f"{rat_name}"] = alignment.fit_transform(
        embeddings[first_rat], embeddings[rat_name], labels[0], labels[j+1])


# Save embeddings in current folder
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)


#%% MULTISESSION
multi_embeddings = dict()

# Multisession training
multi_cebra_model = CEBRA(model_architecture='offset10-model',
                    batch_size=512,
                    learning_rate=3e-3,
                    temperature=1,
                    output_dimension=3,
                    max_iterations=max_iter,
                    distance='cosine',
                    conditional='time_delta',
                    device='cuda_if_available',
                    verbose=True,
                    time_offsets=10)

# Provide a list of data, i.e. datas = [data_a, data_b, ...]
multi_cebra_model.fit(datas, labels)

# Transform each session with the right model, by providing the corresponding session ID
for i, (name, X) in enumerate(zip(names, datas)):
    multi_embeddings[name] = multi_cebra_model.transform(X, session_id=i)

# Save embeddings in current folder
with open('multi_embeddings.pkl', 'wb') as f:
    pickle.dump(multi_embeddings, f)

#%% COMPARING THE EMBEDDINGS:
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)
with open('multi_embeddings.pkl', 'rb') as f:
    multi_embeddings = pickle.load(f)

# Plotting function: 
def plot_hippocampus(ax, embedding, label, gray = False, idx_order = (0,1,2)):
    r_ind = label[:,1] == 1
    l_ind = label[:,2] == 1

    if not gray:
        r_cmap = 'cool'
        l_cmap = 'viridis'
        r_c = label[r_ind, 0]
        l_c = label[l_ind, 0]
    else:
        r_cmap = None
        l_cmap = None
        r_c = 'gray'
        l_c = 'gray'

    idx1, idx2, idx3 = idx_order
    r=ax.scatter(embedding [r_ind,idx1],
               embedding [r_ind,idx2],
               embedding [r_ind,idx3],
               c=r_c,
               cmap=r_cmap, s=0.05, alpha=0.75)
    l=ax.scatter(embedding [l_ind,idx1],
               embedding [l_ind,idx2],
               embedding [l_ind,idx3],
               c=l_c,
               cmap=l_cmap, s=0.05, alpha=0.75)

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    return ax

fig = plt.figure(figsize=(10,6))

ax1 = plt.subplot(241, projection='3d')
ax2 = plt.subplot(242, projection='3d')
ax3 = plt.subplot(243, projection='3d')
ax4 = plt.subplot(244, projection='3d')
axs = [ax1, ax2, ax3, ax4]

ax5 = plt.subplot(245, projection='3d')
ax6 = plt.subplot(246, projection='3d')
ax7 = plt.subplot(247, projection='3d')
ax8 = plt.subplot(248, projection='3d')
axs_multi = [ax5, ax6, ax7, ax8]

for name, label, ax, ax_multi in zip(names, labels, axs, axs_multi):
    ax = plot_hippocampus(ax, embeddings[name], label)
    ax.set_title(f'Single-{name}', y=1, pad=-20)
    ax.axis('off')
    ax_multi = plot_hippocampus(ax_multi, multi_embeddings[name], label)
    ax_multi.set_title(f'Multi-{name}', y=1, pad=-20)
    ax_multi.axis('off')


plt.subplots_adjust(wspace=0,
                    hspace=0)
plt.show()

#%% COMPUTE CONSISTENCY MAPS
labels = [data.continuous_index[:, 0]
          for data in [hippocampus_a, hippocampus_b, hippocampus_c, hippocampus_g]]


# CEBRA (single animal)
scores, pairs, subjects = cebra.sklearn.metrics.consistency_score(embeddings=list(embeddings.values()),
                                                                                    labels=labels,
                                                                                    dataset_ids=names,
                                                                                    between="datasets")

# CEBRA (multiple animals)
multi_scores, multi_pairs, multi_subjects = cebra.sklearn.metrics.consistency_score(embeddings=list(multi_embeddings.values()),
                                                                                    labels=labels,
                                                                                    dataset_ids=names,
                                                                                    between="datasets")

# Display consistency maps
fig = plt.figure(figsize=(11, 4))

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

ax1 = cebra.plot_consistency(scores, pairs=pairs, datasets=subjects,
                                ax=ax1, title="single animal", colorbar_label=None)

ax2 = cebra.plot_consistency(multi_scores, pairs=multi_pairs, datasets=multi_subjects,
                                ax=ax2, title="multi-animal", colorbar_label=None)