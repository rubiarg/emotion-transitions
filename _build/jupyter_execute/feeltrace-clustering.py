#!/usr/bin/env python
# coding: utf-8

# # Emotion transitions study: Emotion responses cluster analysis
# Rubia Guerra
# 
# Last updated: Mar 31st 2022

# ### Module definitions

# In[1]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import glob
import re
import pandas as pd
import scipy.io as sio
import seaborn as sns

from scipy import signal
from statsmodels.tsa import stattools

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support

from tqdm import tqdm

plt.style.use("seaborn")

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Import data

# In[2]:


def load_and_split_dataset(data_dir = '../EEG/data/p*', split_size=100, random_seed=128, split_ratio = .7, subject_choice_seed=128):
    subject_data_files = glob.glob(os.path.join(data_dir, 'joystick.mat'))
    subject_data_files.sort()
    num_subjects = int(len(subject_data_files) * split_ratio)
    
    # group data, pick subjects randomly
    np.random.seed(subject_choice_seed)
    all_subjects = np.random.choice(subject_data_files, size=num_subjects, replace=False)
    subjects = re.findall('p\d+', ''.join(all_subjects))
    print(f"Training set participants: {subjects}")
    
    train = []
    test = []
    for subject_filename in subject_data_files:
        mat_contents = sio.loadmat(subject_filename)
        df = pd.DataFrame(mat_contents['var'], columns=['Timestamp', 'Feeltrace'])
        
        if subject_filename in all_subjects:
            train.append(df)
        else:
            test.append(df)
    
    return train, test


# In[3]:


[train, test] = load_and_split_dataset()


# ### Defining emotion dynamics features
# Refer to _Houben M, Van Den Noortgate W, Kuppens P. The relation between short-term emotion dynamics and psychological well-being: A meta-analysis. Psychological bulletin. 2015 Jul;141(4):901._
# 
# - **Emotional inertia:** refers to how well the intensity of an emotional state can be predicted from the emotional state at a previous moment.
# - **Emotional instability:** refers to the magnitude of emotional changes from one moment to the next. An individual characterized by high levels of instability experiences larger emotional shifts from one moment to the next, resulting in a more unstable emotional life.
# - **Emotional variability:** refers to the range or amplitude of someone’s emotional states across time. An individual characterized by higher levels of emotional variability experiences emotions that reach more extreme levels and shows larger emotional deviations from his or her average emotional level

# In[4]:


class EmotionDynamics:
    def __init__(self, Fs=30, interval=300):
        self.lag = int(Fs*interval*1e-1) # feeltrace sampling rate x 300 ms

    def emotional_variability(self, X):
        return np.std(X)

    def emotional_instability(self, X):
        return np.sum((X[1:] - X[:-1])**2)/(len(X)-1) # MSSD

    def emotional_inertia(self, X, lag=None):
        if lag is None:
            lag = self.lag
        try:
            return stattools.acf(X, nlags=lag)[lag] # Autocorrelation
        except IndexError:
            return stattools.acf(X, nlags=700)[700] # Autocorrelation

    
    def get_parameters(self, X):
        X = np.array(X)
        parameters = {'Inertia':'', 'Instability':'', 'Variability':''}
        parameters['Inertia'] = self.emotional_inertia(X)
        parameters['Instability'] = self.emotional_instability(X)
        parameters['Variability'] = self.emotional_variability(X)
        return parameters


# In[5]:


ED = EmotionDynamics()


# In[6]:


ED.get_parameters(train[1]['Feeltrace'])


# In[7]:


training_data = []

for subject in train:
    feeltrace = np.array(subject['Feeltrace'])
    training_data.append(ED.get_parameters(feeltrace))


# In[8]:


test_data = []
for subject in test:
    feeltrace = np.array(subject['Feeltrace'])
    test_data.append(ED.get_parameters(feeltrace))


# In[9]:


X_train = pd.DataFrame(training_data)
X_test = pd.DataFrame(test_data)
X_train


# In[10]:


X = X_train.append(X_test).reset_index(drop=True)
X.head()


# In[11]:


max_ = pd.Series(map(lambda x: x*1.2 if  x > 0 else x*0.8, X.max()), ['Inertia', 'Instability', 'Variability'])
min_ = pd.Series(map(lambda x: x*1.2 if  x < 0 else x*0.8, X.min()), ['Inertia', 'Instability', 'Variability'])


# #### Pairplot analysis

# In[12]:


grid = sns.pairplot(X);

# x-axis limits
grid.axes[0,0].set_xlim((min_.Inertia,max_.Inertia))
grid.axes[0,1].set_xlim((min_.Instability,max_.Instability))
grid.axes[0,2].set_xlim((min_.Variability,max_.Variability))
grid.axes[1,0].set_xlim((min_.Inertia,max_.Inertia))
grid.axes[1,1].set_xlim((min_.Instability,max_.Instability))
grid.axes[1,2].set_xlim((min_.Variability,max_.Variability))
grid.axes[2,0].set_xlim((min_.Inertia,max_.Inertia))
grid.axes[2,1].set_xlim((min_.Instability,max_.Instability))
grid.axes[2,2].set_xlim((min_.Variability,max_.Variability))

# y-axis limits
grid.axes[0,0].set_ylim((min_.Inertia,max_.Inertia))
grid.axes[0,1].set_ylim((min_.Inertia,max_.Inertia))
grid.axes[0,2].set_ylim((min_.Inertia,max_.Inertia))
grid.axes[1,0].set_ylim((min_.Instability,max_.Instability))
grid.axes[1,1].set_ylim((min_.Instability,max_.Instability))
grid.axes[1,2].set_ylim((min_.Instability,max_.Instability))
grid.axes[2,0].set_ylim((min_.Variability,max_.Variability))
grid.axes[2,1].set_ylim((min_.Variability,max_.Variability))
grid.axes[2,2].set_ylim((min_.Variability,max_.Variability))


# #### Data preprocessing: scaling
# Standardize features by removing the mean and scaling to unit variance.

# In[13]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns={'Inertia', 'Instability', 'Variability'})
X_scaled.head()


# In[14]:


max_scaled = pd.Series(map(lambda x: x*1.2 if  x > 0 else x*0.8, X_scaled.max()), ['Inertia', 'Instability', 'Variability'])
min_scaled = pd.Series(map(lambda x: x*1.2 if  x < 0 else x*0.8, X_scaled.min()), ['Inertia', 'Instability', 'Variability'])


# ### 3D scatterplot

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=130, auto_add_to_figure=False)
fig.add_axes(ax)

ax.scatter(X_scaled.Inertia, X_scaled.Instability, X_scaled.Variability, cmap=plt.cm.nipy_spectral, edgecolor="k")

ax.set_xlabel('Inertia')
ax.set_ylabel('Instability')
ax.set_zlabel('Variability')

plt.show()


# ### Principal Component Analysis

# In[16]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition

pca = decomposition.PCA(n_components=2)
pca.fit(X_scaled)
X_PCA = pca.transform(X_scaled)

sns.scatterplot(x=X_PCA[:, 0], y=X_PCA[:, 1]);


# ### Gaussian Mixture Model

# In[17]:


"""
================================
Gaussian Mixture Model Selection
================================

Gaussian Mixture Models using information-theoretic criteria (BIC).
Model selection concerns both the covariance type and the number of components in the model.
Unlike Bayesian procedures, such inferences are prior-free.

"""

import itertools
from scipy import linalg
from sklearn import mixture

lowest_bic = np.infty
bic = []
n_components_range = range(1, 10)
cv_types = ["spherical", "tied", "diag", "full"]
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(
            n_components=n_components, covariance_type=cv_type
        )
        gmm.fit(X_scaled)
        bic.append(gmm.bic(X_scaled))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(15, 15))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + 0.2 * (i - 2)
    bars.append(
        plt.bar(
            xpos,
            bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
            width=0.2,
            color=color,
        )
    )
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
plt.title("BIC score per model")
xpos = (
    np.mod(bic.argmin(), len(n_components_range))
    + 0.65
    + 0.2 * np.floor(bic.argmin() / len(n_components_range))
)
plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
spl.set_xlabel("Number of components")
spl.legend([b[0] for b in bars], cv_types)

# Plot the winner
splot = plt.subplot(2, 1, 2)
Y_ = clf.predict(X_scaled)
for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_, color_iter)):
    v, w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(X_scaled.Inertia.loc[Y_ == i], X_scaled.Instability.loc[Y_ == i], 50, color=color)

plt.legend(range(len(clf.means_)))
plt.xlabel('Inertia')
plt.ylabel('Instability')
plt.title(
    f"Selected GMM: {best_gmm.covariance_type} model, "
    f"{best_gmm.n_components} components"
)
plt.show()


# ### Within-participant analysis

# The main takeaway so far seems to be that there are no direct archetypes or patterns of response. This seems to suggest that personalized models make sense, given that each participant's response is distinct.
# 
# Below is an attempt to address KM's comment:
# 
# ```Is there any more evidence that it is indeed individualized? Is within-participant behavior consistent?```
# 
# Each feeltrace is split in 30 sec chunks, then analyzed within participant.

# In[18]:


participant_split = {}
n_samples = int(30*0.5*60) # sampling rate x n_seconds (3 min)
for index, subject in enumerate(train):
    feeltrace = np.array(subject['Feeltrace'])
    n_chunks = int(len(feeltrace)/n_samples)
    participant_split[index] = np.array_split(feeltrace, n_chunks)


# In[19]:


ed_participant = {}
grids = []
X_participant_dfs = []

for index, participant in enumerate(participant_split.items()):
    ed_participant[index] = list(map(ED.get_parameters, participant[1]))
    X_participant_dfs.append(pd.DataFrame(ed_participant[index]))

from functools import reduce
X_participants = reduce(lambda x, y: pd.concat([x,y], ignore_index = True, sort = False), X_participant_dfs)


# In[20]:


max_ = pd.Series(map(lambda x: x*1.2 if  x > 0 else x*0.8, X_participants.max()), ['Inertia', 'Instability', 'Variability'])
min_ = pd.Series(map(lambda x: x*1.2 if  x < 0 else x*0.8, X_participants.min()), ['Inertia', 'Instability', 'Variability'])


# In[21]:



for index, participant in enumerate(participant_split.items()):    
X_participant = pd.DataFrame(ed_participant[index])
grid = sns.pairplot(X_participant);# x-axis limits
grid.axes[0,0].set_xlim((min_.Inertia,max_.Inertia))
grid.axes[0,1].set_xlim((min_.Instability,max_.Instability))
grid.axes[0,2].set_xlim((min_.Variability,max_.Variability))
grid.axes[1,0].set_xlim((min_.Inertia,max_.Inertia))
grid.axes[1,1].set_xlim((min_.Instability,max_.Instability))
grid.axes[1,2].set_xlim((min_.Variability,max_.Variability))
grid.axes[2,0].set_xlim((min_.Inertia,max_.Inertia))
grid.axes[2,1].set_xlim((min_.Instability,max_.Instability))
grid.axes[2,2].set_xlim((min_.Variability,max_.Variability))

# y-axis limits
grid.axes[0,0].set_ylim((min_.Inertia,max_.Inertia))
grid.axes[0,1].set_ylim((min_.Inertia,max_.Inertia))
grid.axes[0,2].set_ylim((min_.Inertia,max_.Inertia))
grid.axes[1,0].set_ylim((min_.Instability,max_.Instability))
grid.axes[1,1].set_ylim((min_.Instability,max_.Instability))
grid.axes[1,2].set_ylim((min_.Instability,max_.Instability))
grid.axes[2,0].set_ylim((min_.Variability,max_.Variability))
grid.axes[2,1].set_ylim((min_.Variability,max_.Variability))
grid.axes[2,2].set_ylim((min_.Variability,max_.Variability))


# In[ ]:




