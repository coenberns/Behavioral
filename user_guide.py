#%%
import numpy as np
import pandas as pd
import cebra
from cebra import CEBRA
#%%
X = np.random.normal(0,1,(100,3))
X_new = np.random.normal(0,1,(100,4))
np.savez("neural_data", neural = X, new_neural = X_new)

# Create a .h5 file, containing a pd.DataFrame

X_continuous = np.random.normal(0,1,(100,3))
X_discrete = np.random.randint(0,10,(100, ))
df = pd.DataFrame(np.array(X_continuous), columns=["continuous1", "continuous2", "continuous3"])
df["discrete"] = X_discrete
df.to_hdf("auxiliary_behavior_data.h5", key="auxiliary_variables")

#%%


# Load the .npz
neural_data = cebra.load_data(file="neural_data.npz", key="neural")

# ... and similarly load the .h5 file, providing the columns to keep
continuous_label = cebra.load_data(file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["continuous1", "continuous2", "continuous3"])
discrete_label = cebra.load_data(file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["discrete"]).flatten()


#%%
import cebra.helper as cebra_helper

X = np.random.normal(0,1,(100,3))
y = np.random.normal(0,1,(100,4))
np.savez("data", neural = X, trial = y)
# Create a .h5 file
url = "https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/Reaching-Mackenzie-2018-08-30/labeled-data/reachingvideo1/CollectedData_Mackenzie.h5?raw=true"
dlc_file = cebra_helper.download_file_from_url(url) # an .h5 example file
# Load data
X = cebra.load_data(file="data.npz", key="neural")
y_trial_id = cebra.load_data(file="data.npz", key="trial")
y_behavior = cebra.load_data(file=dlc_file, columns=["Hand", "Tongue"])


# %%
