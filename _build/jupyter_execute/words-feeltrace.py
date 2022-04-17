#!/usr/bin/env python
# coding: utf-8

# # Feeltrace and calibrated words analysis
# Rubia Guerra
# 
# Last updated: Apr 16th 2022

# TODO: Fix participant mappings (P2, P3... P20) -> (P1...P16)

# ## Module definitions

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
import pandas as pd
import scipy.io as sio
import seaborn as sns
from scipy import signal
from statsmodels.tsa import stattools
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support

from tqdm import tqdm

plt.style.use("seaborn")

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import data

# In[2]:


def load_dataset(data_dir = '../EEG/data/p*'):
    feeltrace_data_files = glob.glob(os.path.join(data_dir, 'joystick.mat'))
    feeltrace_data_files.sort()
    
    feeltrace_data = []
    for subject_filename in feeltrace_data_files:
        mat_contents = sio.loadmat(subject_filename)
        df = pd.DataFrame(mat_contents['var'], columns=['Timestamps', 'Feeltrace'])
        p_number = [re.findall('p\d+', subject_filename)[0].strip('p')] * df.shape[0]
        df['p_number'] = p_number
        feeltrace_data.append(df)
    
    subject_data_files = glob.glob(os.path.join(data_dir, 'calibrated_words_calibrated_values.mat'))
    subject_data_files.sort()

    timestamps_data_files = glob.glob(os.path.join(data_dir, 'calibrated_words_time*.mat'))
    timestamps_data_files.sort()
    
    subjects_data = []
    for subject_filename, timestamp_filename in zip(subject_data_files, timestamps_data_files):
        subject_data = (sio.loadmat(subject_filename)['var'].ravel() + 10)*10
        timestamp_data = sio.loadmat(timestamp_filename)['var'].ravel()

        df_data = list(zip(timestamp_data, subject_data))        
        df = pd.DataFrame(df_data, columns=['Timestamps', 'Values'])

        p_number = [re.findall('p\d+', subject_filename)[0].strip('p')] * df.shape[0]
        df['p_number'] = p_number
        subjects_data.append(df)
    
    return subjects_data, feeltrace_data


# In[3]:


[words_list, feeltrace_list] = load_dataset()


# In[4]:


words_df = pd.concat(words_list)
feeltrace_df = pd.concat(feeltrace_list)


# In[5]:


feeltrace_df.p_number.unique()


# In[6]:


pd.concat(words_list).head()


# In[7]:


pd.concat(feeltrace_list).head()


# ## Single participant
# 
# Testing code with a single subject

# ### Downsampling the continuous annotation

# Before time series analysis can be performed, we must downsample the feeltrace to match the sample size of the interview words (or upsample the interview words):
# - $\text{Fs}_{\text{feeltrace}} = 30$Hz
# - $\text{Fs}_{\text{calibrated words}} \approx 0.05$Hz

# In[8]:


def scale(X, min_=0, max_=200):
    return (X - min_)/(max_ - min_)


# In[9]:


p10_words = words_list[0].copy()
p10_feeltrace = feeltrace_list[0].copy()

p10_words.Values = scale(p10_words.Values)
p10_feeltrace.Feeltrace = scale(p10_feeltrace.Feeltrace)


# In[10]:


def sampling_frequency(df):
    return df.Timestamps.count()/(df.Timestamps.iloc[-1]*1e-3)


# In[11]:


f'Fs Interview ~ %.2f Hz' % sampling_frequency(p10_words)


# In[12]:


f'Fs Feeltrace ~ %.2f Hz' % sampling_frequency(p10_feeltrace)


# Here, I attempt downsampling the feeltrace using aggregation strategies. I am using windows corresponding to midpoints in between sampled words. For instance, if the first interview word is recorded at $t=10$s and the next word is recorded at $t=15$s, the first window corresponds to $t=0$s to $t=12.5$s.
# 
# We can calculate windows using a rolling average on the interview timestamps.

# In[13]:


timestamps = p10_words.Timestamps.rolling(2).mean().dropna().reset_index(drop=True)
timestamps.head()


# In[14]:


plt.figure(figsize=(15,10))
sns.lineplot(data=p10_words, x='Timestamps', y='Values')
plt.ylim([-.1, 1.1])

for x in timestamps:
    plt.axvline(x, c='k', alpha=0.2, linestyle='dashed')


# In[15]:


plt.figure(figsize=(15,10))
sns.lineplot(data=p10_feeltrace, x='Timestamps', y='Feeltrace')
plt.ylim([-.1, 1.1])

for x in timestamps:
    plt.axvline(x, c='k', alpha=0.2, linestyle='dashed')


# In[16]:


from statsmodels.graphics.tsaplots import plot_acf

plt.rc('figure', figsize=(15,10))
plot_acf(p10_words['Values'], lags=range(len(p10_words)))
plt.ylim([-1.1, 1.1]);


# In[17]:


plt.rc('figure', figsize=(15,10))
plot_acf(p10_feeltrace['Feeltrace'], lags=range(len(p10_feeltrace)))
plt.ylim([-1.1, 1.1]);


# Extract indexes in the feeltrace by looking at the nearest timestamp that matches our window start points:

# In[18]:


idxs = []

for timestamp in timestamps:
    arr = p10_feeltrace.Timestamps.to_numpy().astype(int)
    dist = (arr - timestamp)**2
    idx = tuple(np.argwhere(dist == np.min(dist))[0])
    idxs.append(idx[0])

print(f'Indexes: %s' % idxs)


# Timestamps for the new feeltrace are defined as the mean timestamp of the window ((window start + window end) / 2):

# In[19]:


feeltrace_timestamps = list(p10_feeltrace.loc[idxs, 'Timestamps'].rolling(2).mean().reset_index(drop=True))

# add first timestamp
feeltrace_timestamps[0] = p10_feeltrace.loc[idxs[0], 'Timestamps'] / 2

# add last timestamp
feeltrace_timestamps.append((p10_feeltrace.Timestamps.iloc[-1] + p10_feeltrace.loc[idxs[-1], 'Timestamps']) / 2)

feeltrace_timestamps = pd.Series(feeltrace_timestamps)

print(f'Timestamps (ms): %s' % feeltrace_timestamps.head())


# In[20]:


# make sure both series are the same length
assert(len(feeltrace_timestamps) == len(p10_words.Timestamps))


# Now, I explore aggregating the feeltrace in two ways:
# - By taking the mean value of all points in window
# - By calculating the overall direction of change in the feeltrace (slope)

# #### Aggregating by taking the mean

# In[21]:


p10_feeltrace_agg_mean = pd.DataFrame()

mean_feeltrace = []

# first window
mean_feeltrace.append(p10_feeltrace.Feeltrace.loc[:idxs[0]].mean())

# middle windows
for (prev_idx, idx) in zip(idxs[:-1], idxs[1:]):
    mean_feeltrace.append(p10_feeltrace.Feeltrace.loc[prev_idx:idx].mean())

# last window
mean_feeltrace.append(p10_feeltrace.Feeltrace.loc[idxs[-1]:].mean())

# make sure both feeltrace and interview series are the same size
assert(len(mean_feeltrace) == len(p10_words.Values))

p10_feeltrace_agg_mean['Timestamps'] = feeltrace_timestamps
p10_feeltrace_agg_mean['Feeltrace'] = mean_feeltrace


# In[22]:


p10_feeltrace_agg_mean.head()


# In[23]:


plt.figure(figsize=(15,10))

sns.lineplot(data=p10_feeltrace, x='Timestamps', y='Feeltrace', marker='.', alpha=0.1)
sns.lineplot(data=p10_feeltrace_agg_mean, x='Timestamps', y='Feeltrace', marker='o')
sns.lineplot(data=p10_words, x='Timestamps', y='Values', marker='o')
plt.legend(['Original feeltrace', 'AggFeeltrace', 'Words']);

for x in timestamps:
    plt.axvline(x, c='k', alpha=0.2, linestyle='dashed')


# Checking autocorelation on aggregated data:

# In[24]:


plt.rc('figure', figsize=(15,10))
plot_acf(p10_feeltrace_agg_mean['Feeltrace'], lags=range(len(p10_feeltrace_agg_mean)))
plt.ylim([-1.1, 1.1]);


# #### Aggregating by taking the slope

# Trying to bin state transitions in three values:
# - Increase in stress level (+1)
# - Neutral (0)
# - Decrease in stress level (-1)
# 
# Tried a few attemps, settling for taking first and last value of window:

# In[25]:


def change_direction(x, y):
    def slope(x):
        slope = np.polyfit(range(len(x)), x, 1)[0]
        return slope
       
    def slope_angle(x, y):
        x = np.array(x)
        y = np.array(y)
        angle = np.arctan2(y[-1] - y[0], x[-1] - x[0])
        return angle
    
    def slope_sign(x, factor=1e3):
        if np.round(slope(x)*factor): 
            return np.sign(slope(x)*factor) 
        else: 
            return 0
        
    def first_last(x):
        diff = np.round(x.iloc[-1] - x.iloc[0],1)
        if diff:
            return np.sign(diff)
        else:
            return 0
       
    return first_last(x)


# Calculating for feeltrace:

# In[26]:


p10_feeltrace_agg_slope = pd.DataFrame()

slope_feeltrace = []

# first window
slope_feeltrace.append(change_direction(p10_feeltrace.Feeltrace[:idxs[0]], p10_feeltrace.Timestamps[:idxs[0]]))

# middle windows
for (prev_idx, idx) in zip(idxs[:-1], idxs[1:]):
    slope_feeltrace.append(change_direction(p10_feeltrace.Feeltrace.loc[prev_idx:idx], p10_feeltrace.Timestamps.loc[prev_idx:idx]))

# last window
slope_feeltrace.append(change_direction(p10_feeltrace.Feeltrace.loc[idxs[-1]:], p10_feeltrace.Timestamps.loc[idxs[-1]:]))

# make sure both feeltrace and interview series are the same size
assert(len(slope_feeltrace) == len(p10_words.Values))

p10_feeltrace_agg_slope['Timestamps'] = feeltrace_timestamps
p10_feeltrace_agg_slope['Feeltrace'] = slope_feeltrace
p10_feeltrace_agg_slope.head()


# Also calculating the slope values based on the aggregated feeltrace data:

# In[27]:


p10_feeltrace_agg_mean_slope = pd.DataFrame()

mean_feeltrace_words = []
indexes = list(p10_feeltrace_agg_mean.index)

# middle windows
for (prev_idx, idx) in zip(indexes[:-1], indexes[1:]):
    mean_feeltrace_words.append(change_direction(p10_feeltrace_agg_mean.Feeltrace.loc[prev_idx:idx], p10_feeltrace_agg_mean.Timestamps.loc[prev_idx:idx]))

p10_feeltrace_agg_mean_slope['Feeltrace'] = mean_feeltrace_words
p10_feeltrace_agg_mean_slope['Timestamps'] = timestamps
p10_feeltrace_agg_mean_slope.head()


# Taking the slopes for interview words:

# In[28]:


p10_words_agg_slope = pd.DataFrame()

slope_words = []
indexes = list(p10_words.index)

# middle windows
for (prev_idx, idx) in zip(indexes[:-1], indexes[1:]):
    slope_words.append(change_direction(p10_words.Values.loc[prev_idx:idx], p10_words.Timestamps.loc[prev_idx:idx]))

p10_words_agg_slope['Values'] = slope_words
p10_words_agg_slope['Timestamps'] = timestamps
p10_words_agg_slope.head()


# In[29]:


fig, axes = plt.subplots(2, 2, figsize=(25,10))

sns.lineplot(data=p10_words, x='Timestamps', y='Values', marker='.', alpha=0.5, ax=axes[0,0])
sns.lineplot(data=p10_words_agg_slope, y='Values', x='Timestamps', marker='o', ax=axes[0,0]);
axes[0,0].legend(['Original words', 'AggWords']);

sns.lineplot(data=p10_feeltrace, x='Timestamps', y='Feeltrace', marker='.', alpha=0.1, ax=axes[0,1])
sns.lineplot(data=p10_feeltrace_agg_slope, y='Feeltrace', x='Timestamps', marker='o', ax=axes[0,1]);
axes[0,1].legend(['Original feeltrace', 'AggSlopeFeeltrace']);

sns.lineplot(data=p10_feeltrace_agg_mean, x='Timestamps', y='Feeltrace', marker='.', alpha=0.5, ax=axes[1,0])
sns.lineplot(data=p10_feeltrace_agg_mean_slope, y='Feeltrace', x='Timestamps', marker='o', ax=axes[1,0]);
axes[1,0].legend(['AggMeanFeeltrace', 'AggMeanSlopeFeeltrace']);

fig.delaxes(axes[1,1])

for x in timestamps:
    axes[0,0].axvline(x, c='k', alpha=0.2, linestyle='dashed')
    axes[0,1].axvline(x, c='k', alpha=0.2, linestyle='dashed')
    axes[1,0].axvline(x, c='k', alpha=0.2, linestyle='dashed')


# To proceed with analysis, I will be comparing the transitions captured by word labels and by the aggregated feeltrace values.

# In[30]:


p10_agg_data = pd.DataFrame()
p10_agg_data['Timestamps'] = p10_words_agg_slope['Timestamps'].copy()
p10_agg_data['Words'] = p10_words_agg_slope['Values'].copy()
p10_agg_data['Feeltrace'] = p10_feeltrace_agg_mean_slope['Feeltrace'].copy()

p10_agg_data.head()


# Checking autocorelation on aggregated data:

# In[31]:


plt.rc('figure', figsize=(15,10))
plot_acf(p10_words_agg_slope['Values'], lags=range(len(p10_words_agg_slope)))
plt.ylim([-1.1, 1.1]);


# In[32]:


plt.rc('figure', figsize=(15,10))
plot_acf(p10_feeltrace_agg_mean_slope['Feeltrace'], lags=range(len(p10_feeltrace_agg_mean_slope)))
plt.ylim([-1.1, 1.1]);


# ### Joint time series analysis

# Refer to _Ernst AF, Timmerman ME, Jeronimus BF, Albers CJ. Insight into individual differences in emotion dynamics with clustering. Assessment. 2021 Jun;28(4):1186-206._
# 
# *Adapted from [BioSciEconomist/ex VAR.py](https://gist.github.com/BioSciEconomist/197bd86ea61e0b4a49707af74a0b9f9c).

# #### Granger Causality

# The results of Granger Causality for P10 are promising, suggeting a significant relationship* between interview words and feeltrace.
# 
# *(potentially a causality relationship, although this might be too strong of a claim, see `Limitations` section below)

# In[33]:


from statsmodels.tsa.stattools import grangercausalitytests
maxlag = 1
test = 'ssr_chi2test'

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [np.round(test_result[i+1][0][test][1], 5) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            if not min_p_value:
                raise ValueError
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

while True:
    try:
        grangers_causation_matrix(p10_agg_data[['Words', 'Feeltrace']], variables = p10_agg_data[['Words', 'Feeltrace']].columns, verbose=False) 
        maxlag+=1
    except ValueError:
        maxlag-=1
        grangers_causation_matrix(p10_agg_data[['Words', 'Feeltrace']], variables = p10_agg_data[['Words', 'Feeltrace']].columns, verbose=True) 
        print(maxlag)
        break

# notes: 
# The row are the Response (Y) and the columns are the predictor series (X).

# For example, if you take the value 0.0021 in (row 1, column 2), it refers to 
# the p-value of Feeltrace_x causing Words_y.


# ##### Limitations
# 
# (see https://en.wikipedia.org/wiki/Granger_causality#Limitations)
# 
# As its name implies, Granger causality is not necessarily true causality. In fact, the Granger-causality tests fulfill only the Humean definition of causality that identifies the cause-effect relations with constant conjunctions.[14] If both X and Y are driven by a common third process with different lags, one might still fail to reject the alternative hypothesis of Granger causality. Yet, manipulation of one of the variables would not change the other. Indeed, the Granger-causality tests are designed to handle pairs of variables, and may produce misleading results when the true relationship involves three or more variables. Having said this, it has been argued that given a probabilistic view of causation, Granger causality can be considered true causality in that sense, especially when Reichenbach's "screening off" notion of probabilistic causation is taken into account.[15] Other possible sources of misguiding test results are: (1) not frequent enough or too frequent sampling, (2) nonlinear causal relationship, (3) time series nonstationarity and nonlinearity and (4) existence of rational expectations.[14] A similar test involving more variables can be applied with vector autoregression. Recently [16] a fundamental mathematical study of the mechanism underlying the Granger method has been provided. By making use exclusively of mathematical tools (Fourier transformation and differential calculus), it has been found that not even the most basic requirement underlying any possible definition of causality is met by the Granger causality test: any definition of causality should refer to the prediction of the future from the past; instead by inverting the time series it can be shown that Granger allows one to ”predict” the past from the future as well. 

# #### Cointegration

# In[34]:


#-----------------------------------------
# cointegration
#-----------------------------------------

# When two or more time series are cointegrated, it means they have a long run, 
# statistically significant relationship.

# This is the basic premise on which Vector Autoregression(VAR) models is based on. 
# So, it’s fairly common to implement the cointegration test before starting to 
# build VAR models.

# more technically:
# Order of integration(d) is nothing but the number of differencing required to 
# make a non-stationary time series stationary.

# Now, when you have two or more time series, and there exists a linear combination
#  of them that has an order of integration (d) less than that of the individual 
# series, then the collection of series is said to be cointegrated.

from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df, alpha=0.05, verbose=False): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    results = []

    # Summary
    if verbose:
        print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        if verbose:
            print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
        results.append({col: [trace, cvt, trace > cvt]})
        
    return results

cointegration_test(p10_agg_data[['Words', 'Feeltrace']])


# #### Stationarity

# In[35]:


from statsmodels.tsa.stattools import adfuller

#------------------------------
# check for stationarity
#-----------------------------

# VAR model requires the time series you want to forecast to be stationary, 
# it is customary to check all the time series in the system for stationarity.

# Since, differencing reduces the length of the series by 1 and since all the 
# time series has to be of the same length, you need to difference all the series
#  in the system if you choose to difference at all.

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    if verbose:
        # Print Summary
        print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
        print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
        print(f' Significance Level    = {signif}')
        print(f' Test Statistic        = {output["test_statistic"]}')
        print(f' No. Lags Chosen       = {output["n_lags"]}')

        for key,val in r[4].items():
            print(f' Critical value {adjust(key)} = {round(val, 3)}')

        if p_value <= signif:
            print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
            print(f" => Series is Stationary.")
        else:
            print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
            print(f" => Series is Non-Stationary.")
    
    return output
        
 # ADF Test on each column
for name, column in p10_agg_data.iteritems():
    if 'Timestamps' in name:
        continue
    adfuller_test(column, name=column.name)
    print('\n')    
    
diff = p10_agg_data.diff().dropna()

# ADF Test on each column
for name, column in diff.iteritems():
    if 'Timestamps' in name:
        continue
    adfuller_test(column, name=column.name)
    print('\n')   


# #### Pearson's correlation

# In[36]:


p10_agg_data[['Words', 'Feeltrace']].corr()


# #### VAR Model

# ##### Training and validation data

# In[37]:


# The VAR model will be fitted on df_train and then used to forecast the next 4 
# observations. These forecasts will be compared against the actuals present in 
# test data.


nobs = 4
df_train, df_test = p10_agg_data[0:-nobs], p10_agg_data[-nobs:]

# Check size
print(df_train.shape)  # (119, 8)
print(df_test.shape)  # (4, 8)


# ##### Fitting

# In[38]:


#---------------------------------------
# fitting the order of the VAR
#--------------------------------------

# To select the right order of the VAR model, we iteratively fit increasing orders 
# of VAR model and pick the order that gives a model with least AIC.

df_train_diff = df_train.diff().dropna()

model = VAR(df_train_diff[['Words', 'Feeltrace']])
for i in [1,2,3,4,5, 6, 7]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')


# In[39]:


# In the above output, the AIC drops to lowest at lag 4, then increases at 
# lag 5 and then continuously drops further. (more negative = 'smaller' AIC)

#---------------------------------
# alterntative: auto fit
#---------------------------------

x = model.select_order(maxlags=7)
x.summary()


# In[40]:


#----------------------------------
# fit VAR(5)
#---------------------------------

model_fitted = model.fit(7)
model_fitted.summary()


# ##### Check for remaining serial correlation

# In[41]:


# Serial correlation of residuals is used to check if there is any leftover pattern 
# in the residuals (errors). If there is any correlation left in the residuals, then,
# there is some pattern in the time series that is still left to be explained by the
# model. In that case, the typical course of action is to either increase the order
# of the model or induce more predictors into the system or look for a different 
# algorithm to model the time series.

# A common way of checking for serial correlation of errors can be measured using 
# the Durbin Watson’s Statistic.

# The value of this statistic can vary between 0 and 4. The closer it is to the value 
# 2, then there is no significant serial correlation. The closer to 0, there is a 
# positive serial correlation, and the closer it is to 4 implies negative serial 
# correlation.

from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

# for col, val in zip(df.columns, out):
#    print(adjust(col), ':', round(val, 2))
    
for col, val in zip(p10_agg_data[['Words', 'Feeltrace']].columns, out):
    print(col, ':', round(val, 2))


# ##### Forecasting

# In[42]:


#--------------------------------------
# forecasting
#--------------------------------------

# In order to forecast, the VAR model expects up to the lag order number of 
# observations from the past data. This is because, the terms in the VAR model 
# are essentially the lags of the various time series in the dataset, so you 
# need to provide it as many of the previous values as indicated by the lag order
# used by the model.

# Get the lag order (we already know this)
lag_order = model_fitted.k_ar
print(lag_order)  #> 4

# Input data for forecasting
forecast_input = df_train_diff[['Words', 'Feeltrace']].values[-lag_order:]
forecast_input

# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs) # nobs defined at top of program
df_forecast = pd.DataFrame(fc, index=p10_agg_data[['Words', 'Feeltrace']].index[-nobs:], columns=p10_agg_data[['Words', 'Feeltrace']].columns + '_1d')
df_forecast


# In[43]:


# The forecasts are generated but it is on the scale of the training data used by 
# the model. So, to bring it back up to its original scale, you need to de-difference 
# it as many times you had differenced the original input data.

def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:     
        if 'Timestamp' in col:
            continue
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc


df_results = invert_transformation(df_train, df_forecast, second_diff=False)        
df_results.loc[:, ['Words_1d', 'Feeltrace_1d']]

#---------------------------
# plot forecasts
#---------------------------


fig, axes = plt.subplots(nrows=int(len(p10_agg_data.columns)/2), ncols=2, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(p10_agg_data[['Words', 'Feeltrace']].columns, axes.flatten())):
    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    df_test[col][-nobs:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();


# ##### Accuracy

# In[44]:


from statsmodels.tsa.stattools import acf

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})

print('Forecast Accuracy of: Words')
accuracy_prod = forecast_accuracy(df_results['Words_forecast'].values, df_test['Words'])
for k, v in accuracy_prod.items():
    print(k, ': ', round(v,4))

print('\nForecast Accuracy of: Feeltrace')
accuracy_prod = forecast_accuracy(df_results['Feeltrace_forecast'].values, df_test['Feeltrace'])
for k, v in accuracy_prod.items():
    print(k, ': ', round(v,4))


# In[45]:


model_fitted.coefs


# ## All participants

# In[46]:


def scale(X, min_=0, max_=200):
    return (X - min_)/(max_ - min_)


# In[47]:


def prepare_data(words, feeltrace):
    p_words = words.copy()
    p_feeltrace = feeltrace.copy()

    p_words.Values = scale(p_words.Values)
    p_words.p_number = p_num_map[p_words.p_number[0]]

    p_feeltrace.Feeltrace = scale(p_feeltrace.Feeltrace)
    p_feeltrace.p_number = p_num_map[p_feeltrace.p_number[0]]
        
    return p_words, p_feeltrace


# In[48]:


def get_indexes(feeltrace, words):
    idxs = []
    timestamps = words.Timestamps.rolling(2).mean().dropna().reset_index(drop=True)    

    for timestamp in timestamps:
        arr = feeltrace.Timestamps.to_numpy().astype(int)
        dist = (arr - timestamp)**2
        idx = tuple(np.argwhere(dist == np.min(dist))[0])
        idxs.append(idx[0])
        
    return idxs, timestamps


def get_feeltrace_timestamps(feeltrace, idxs):
    feeltrace_timestamps = list(feeltrace.loc[idxs, 'Timestamps'].rolling(2).mean().reset_index(drop=True))

    # add first timestamp
    feeltrace_timestamps[0] = feeltrace.loc[idxs[0], 'Timestamps'] / 2

    # add last timestamp
    feeltrace_timestamps.append((feeltrace.Timestamps.iloc[-1] + feeltrace.loc[idxs[-1], 'Timestamps']) / 2)

    feeltrace_timestamps = pd.Series(feeltrace_timestamps)
    
    return feeltrace_timestamps


# In[49]:


def aggregate_mean(feeltrace, idxs):
    feeltrace_agg_mean = pd.DataFrame()

    mean_feeltrace = []

    # first window
    mean_feeltrace.append(feeltrace.Feeltrace.loc[:idxs[0]].mean())

    # middle windows
    for (prev_idx, idx) in zip(idxs[:-1], idxs[1:]):
        mean_feeltrace.append(feeltrace.Feeltrace.loc[prev_idx:idx].mean())

    # last window
    mean_feeltrace.append(feeltrace.Feeltrace.loc[idxs[-1]:].mean())

    # make sure both feeltrace and interview series are the same size
    assert(len(mean_feeltrace) == len(p_words.Values))

    feeltrace_agg_mean['Timestamps'] = feeltrace_timestamps
    feeltrace_agg_mean['Feeltrace'] = mean_feeltrace
    
    return feeltrace_agg_mean

def aggregate_slope(feeltrace, idxs):
    feeltrace_agg_slope = pd.DataFrame()

    slope_feeltrace = []

    # first window
    slope_feeltrace.append(change_direction(feeltrace.Feeltrace[:idxs[0]], feeltrace.Timestamps[:idxs[0]]))

    # middle windows
    for (prev_idx, idx) in zip(idxs[:-1], idxs[1:]):
        slope_feeltrace.append(change_direction(feeltrace.Feeltrace.loc[prev_idx:idx], feeltrace.Timestamps.loc[prev_idx:idx]))

    # last window
    slope_feeltrace.append(change_direction(feeltrace.Feeltrace.loc[idxs[-1]:], feeltrace.Timestamps.loc[idxs[-1]:]))

    feeltrace_agg_slope['Timestamps'] = feeltrace_timestamps
    feeltrace_agg_slope['Feeltrace'] = slope_feeltrace
    feeltrace_agg_slope.head()
    
    return feeltrace_agg_slope


# Renaming participants:

# In[50]:


p_num_map = {'2': 1, '4': 2, '5': 3, '6': 4, 
             '7': 5, '8': 6, '9': 7, '10': 8, 
             '12': 9, '13': 10, '15': 11, '17': 12, 
             '19': 13, '20': 14, '22': 15, '23': 16}


# ### States

# In[51]:


from scipy.stats.stats import pearsonr

n_windows = []
fs = []
granger_p_values = []
stationarity = []
cointegration_p_values = []
pearsons_corr = []
var_models = []

for (p_words, p_feeltrace) in zip(words_list, feeltrace_list):
    [p_words, p_feeltrace] = prepare_data(p_words, p_feeltrace)
    
    fs.append({'Words': sampling_frequency(p_words), 
               'Feeltrace': sampling_frequency(p_feeltrace),
               'p_number': p_feeltrace.p_number[0]
              })
    
    [idxs, timestamps] = get_indexes(p_feeltrace, p_words)
    feeltrace_timestamps = get_feeltrace_timestamps(p_feeltrace, idxs)
    
    # make sure both timestamps are the same length
    assert(len(feeltrace_timestamps) == len(p_words.Timestamps))
    
    n_windows.append({'p_number': p_words.p_number[0], 'n_windows': len(feeltrace_timestamps)})
    
    p_feeltrace_agg_mean = aggregate_mean(p_feeltrace, idxs)
    
    # make sure both feeltrace and interview series are the same size
    assert(len(p_feeltrace_agg_mean.Feeltrace) == len(p_words.Values))
    
    p_agg_data = pd.DataFrame()
    p_agg_data['Timestamps'] = p_words['Timestamps'].copy()
    p_agg_data['Words'] = p_words['Values'].copy()
    p_agg_data['Feeltrace'] = p_feeltrace_agg_mean['Feeltrace'].copy()
    
    maxlag = 1
    
    # transitions
    while True:
        try:
            grangers_causation_matrix(p_agg_data[['Words', 'Feeltrace']], variables = p_agg_data[['Words', 'Feeltrace']].columns) 
            maxlag+=1
        except ValueError:
            maxlag-=1
            granger_p_values.append(grangers_causation_matrix(p_agg_data[['Words', 'Feeltrace']], variables = p_agg_data[['Words', 'Feeltrace']].columns))
            break
            
    stationarity.append({'p_number': p_feeltrace.p_number[0],
                         'Words': adfuller_test(p_agg_data['Words']), 
                         'Feeltrace': adfuller_test(p_agg_data['Feeltrace'])
                        })
    cointegration_p_values.append(cointegration_test(p_agg_data[['Words', 'Feeltrace']]))
    pearsons_corr.append(pearsonr(p_agg_data['Words'],p_agg_data['Feeltrace']))
    
    # The VAR model will be fitted on df_train and then used to forecast the next 4 
    # observations. These forecasts will be compared against the actuals present in 
    # test data.


    nobs = 4
    df_train, df_test = p_agg_data[0:-nobs], p_agg_data[-nobs:]

    # Check size
    model = VAR(df_train[['Words', 'Feeltrace']])

    maxlag = 1

    while True:
        try:
            x = model.select_order(maxlags=maxlag)
            maxlag+=1
        except ValueError:
            maxlag-=1
            x = model.select_order(maxlags=maxlag)
            model_fitted = model.fit(maxlag)
            break

    var_models.append([x, model_fitted])


# In[52]:


n_windows = pd.DataFrame(n_windows)


# For the following secionts, I perform multiple comparisons using Bonferroni-Holm correction:
# 
# ```
# from statsmodels.stats.multitest import multipletests
# rejected, p_adjusted, _, alpha_corrected = multipletests(raw_pvals, alpha=alpha, 
#                                method='bonferroni', is_sorted=False, returnsorted=False)
# np.sum(rejected)
# # 2
# alpha_corrected 
# # 0.0005
# 
# ```

# #### Granger Causality
# 
# TODO: calculate power / effect size

# In[53]:


words_to_feeltrace = []
feeltrace_to_words = []
for value in granger_p_values:
    words_to_feeltrace.append(value.Words_x.Feeltrace_y)
    feeltrace_to_words.append(value.Feeltrace_x.Words_y)


# In[54]:


from statsmodels.stats.multitest import multipletests
rejected, p_adjusted, _, alpha_corrected = multipletests(words_to_feeltrace, alpha=0.05, 
                                                         method='holm', 
                                                         is_sorted=False, returnsorted=False)
print(f'Rejected H0: %d' % np.sum(rejected))
print(f'Corrected alpha: %f' % alpha_corrected)

sns.kdeplot(words_to_feeltrace, color="red", shade=True, label='raw')
ax = sns.kdeplot(p_adjusted, color="green", shade=True, label='adujusted')
ax.set(xlim=(0, 1))
plt.title('Distribution of p-values')
plt.legend();


# In[55]:


n_windows.loc[p_adjusted < 0.05, 'p_number']


# In[56]:


n_windows.loc[p_adjusted >= 0.05, 'p_number']


# In[57]:


from statsmodels.stats.multitest import multipletests
rejected, p_adjusted, _, alpha_corrected = multipletests(feeltrace_to_words, alpha=0.05, 
                                                         method='holm', 
                                                         is_sorted=False, returnsorted=False)
print(f'Rejected H0: %d' % np.sum(rejected))
print(f'Corrected alpha: %f' % alpha_corrected)

sns.kdeplot(feeltrace_to_words, color="red", shade=True, label='raw')
ax = sns.kdeplot(p_adjusted, color="green", shade=True, label='adujusted')
ax.set(xlim=(0, 1))
plt.title('Distribution of p-values')
plt.legend();


# In[58]:


n_windows.loc[p_adjusted < 0.05, 'p_number']


# In[59]:


n_windows.loc[p_adjusted >= 0.05, 'p_number']


# #### Cointegration
# TODO: summarize

# In[60]:


cointegration_p_values # not p-values, different test statistic


# #### Pearson's correlation

# In[61]:


pearsons_corr


# In[62]:


pearsons_corr_values = [x[0] for x in pearsons_corr]
pearsons_corr_p_values = [x[1] for x in pearsons_corr]


# In[63]:


sns.kdeplot(pearsons_corr_values, color="red", shade=True, label='rho')
plt.title('Distribution of correlation values')
plt.legend();


# In[64]:


rejected, p_adjusted, _, alpha_corrected = multipletests(pearsons_corr_p_values, alpha=0.05, 
                                                         method='holm', 
                                                         is_sorted=False, returnsorted=False)
print(f'Rejected H0: %d' % np.sum(rejected))
print(f'Corrected alpha: %f' % alpha_corrected)

sns.kdeplot(feeltrace_to_words, color="red", shade=True, label='raw')
ax = sns.kdeplot(p_adjusted, color="green", shade=True, label='adujusted')
ax.set(xlim=(0, 1))
plt.title('Distribution of p-values')
plt.legend();


# In[65]:


n_windows.loc[p_adjusted < 0.05, 'p_number']


# In[66]:


n_windows.loc[p_adjusted >= 0.05, 'p_number']


# #### VAR Model
# 
# TODO: 
# - Summarize VAR model results
# - Cluster using VAR coefficients

# ### Transitions

# In[67]:


from scipy.stats.stats import pearsonr

n_windows = []
fs = []
granger_p_values = []
stationarity = []
cointegration_p_values = []
pearsons_corr = []
var_models = []

for (p_words, p_feeltrace) in zip(words_list, feeltrace_list):
    [p_words, p_feeltrace] = prepare_data(p_words, p_feeltrace)
    
    fs.append({'Words': sampling_frequency(p_words), 
               'Feeltrace': sampling_frequency(p_feeltrace),
               'p_number': p_feeltrace.p_number[0]
              })
    
    [idxs, timestamps] = get_indexes(p_feeltrace, p_words)
    feeltrace_timestamps = get_feeltrace_timestamps(p_feeltrace, idxs)
    
    # make sure both timestamps are the same length
    assert(len(feeltrace_timestamps) == len(p_words.Timestamps))
    
    n_windows.append({'p_number': p_words.p_number[0], 'n_windows': len(feeltrace_timestamps)})
    
    p_feeltrace_agg_mean = aggregate_mean(p_feeltrace, idxs)
    
    # make sure both feeltrace and interview series are the same size
    assert(len(p_feeltrace_agg_mean.Feeltrace) == len(p_words.Values))
    
    p_feeltrace_agg_slope = aggregate_slope(p_feeltrace, idxs)
    
    # make sure both feeltrace and interview series are the same size
    assert(len(p_feeltrace_agg_slope.Feeltrace) == len(p_words.Values))
    
    p_feeltrace_agg_mean_slope = pd.DataFrame()

    mean_feeltrace_words = []
    indexes = list(p_feeltrace_agg_mean.index)

    # middle windows
    for (prev_idx, idx) in zip(indexes[:-1], indexes[1:]):
        mean_feeltrace_words.append(change_direction(p_feeltrace_agg_mean.Feeltrace.loc[prev_idx:idx], p_feeltrace_agg_mean.Timestamps.loc[prev_idx:idx]))

    p_feeltrace_agg_mean_slope['Feeltrace'] = mean_feeltrace_words
    p_feeltrace_agg_mean_slope['Timestamps'] = timestamps
    p_feeltrace_agg_mean_slope.head()
    
    p_words_agg_slope = pd.DataFrame()

    slope_words = []
    indexes = list(p_words.index)

    # middle windows
    for (prev_idx, idx) in zip(indexes[:-1], indexes[1:]):
        slope_words.append(change_direction(p_words.Values.loc[prev_idx:idx], p_words.Timestamps.loc[prev_idx:idx]))

    p_words_agg_slope['Values'] = slope_words
    p_words_agg_slope['Timestamps'] = timestamps
    p_words_agg_slope.head()
    
    p_agg_data = pd.DataFrame()
    p_agg_data['Timestamps'] = p_words_agg_slope['Timestamps'].copy()
    p_agg_data['Words'] = p_words_agg_slope['Values'].copy()
    p_agg_data['Feeltrace'] = p_feeltrace_agg_mean_slope['Feeltrace'].copy()
    
    maxlag = 1
    
    # transitions
    while True:
        try:
            grangers_causation_matrix(p_agg_data[['Words', 'Feeltrace']], variables = p_agg_data[['Words', 'Feeltrace']].columns) 
            maxlag+=1
        except ValueError:
            maxlag-=1
            granger_p_values.append(grangers_causation_matrix(p_agg_data[['Words', 'Feeltrace']], variables = p_agg_data[['Words', 'Feeltrace']].columns))
            break
            
    stationarity.append({'p_number': p_feeltrace.p_number[0],
                         'Words': adfuller_test(p_agg_data['Words']), 
                         'Feeltrace': adfuller_test(p_agg_data['Feeltrace'])
                        })
    cointegration_p_values.append(cointegration_test(p_agg_data[['Words', 'Feeltrace']]))
    pearsons_corr.append(pearsonr(p_agg_data['Words'],p_agg_data['Feeltrace']))
    
    # The VAR model will be fitted on df_train and then used to forecast the next 4 
    # observations. These forecasts will be compared against the actuals present in 
    # test data.


    nobs = 4
    df_train, df_test = p_agg_data[0:-nobs], p_agg_data[-nobs:]

    # Check size
    model = VAR(df_train[['Words', 'Feeltrace']])

    maxlag = 1

    while True:
        try:
            x = model.select_order(maxlags=maxlag)
            maxlag+=1
        except ValueError:
            maxlag-=1
            x = model.select_order(maxlags=maxlag)
            model_fitted = model.fit(maxlag)
            break

    var_models.append([x, model_fitted])


# The number of windows corresponds to the number of words each participant used:

# In[68]:


n_windows = pd.DataFrame(n_windows)
n_windows['n_windows'].describe()


# In[69]:


plt.figure(figsize=(5,7));
sns.barplot(data=n_windows, x='p_number', y='n_windows', color='c');
plt.ylabel('# of windows');
plt.axhline(n_windows['n_windows'].mean(), c='k', alpha=0.2, linestyle='dashed');


# In[70]:


fs = pd.DataFrame(fs)
fs.describe()


# In[71]:


fs_long = fs.melt('p_number', var_name='Pass', value_name='Fs')


# In[72]:


plt.figure(figsize=(5,7));
sns.boxplot(data=fs, y='Words');


# In[73]:


plt.figure(figsize=(5,7));
sns.boxplot(data=fs, y='Feeltrace');


# All results in the next sections are ordered according to p_number:

# In[74]:


n_windows


# #### Granger Causality
# 
# TODO: calculate power / effect size

# In[75]:


words_to_feeltrace = []
feeltrace_to_words = []
for value in granger_p_values:
    words_to_feeltrace.append(value.Words_x.Feeltrace_y)
    feeltrace_to_words.append(value.Feeltrace_x.Words_y)


# In[76]:


from statsmodels.stats.multitest import multipletests
rejected, p_adjusted, _, alpha_corrected = multipletests(words_to_feeltrace, alpha=0.05, 
                                                         method='holm', 
                                                         is_sorted=False, returnsorted=False)
print(f'Rejected H0: %d' % np.sum(rejected))
print(f'Corrected alpha: %f' % alpha_corrected)

sns.kdeplot(words_to_feeltrace, color="red", shade=True, label='raw')
ax = sns.kdeplot(p_adjusted, color="green", shade=True, label='adujusted')
ax.set(xlim=(0, 1))
plt.title('Distribution of p-values')
plt.legend();


# In[77]:


n_windows.loc[p_adjusted < 0.05, 'p_number']


# In[78]:


n_windows.loc[p_adjusted >= 0.05, 'p_number']


# In[79]:


from statsmodels.stats.multitest import multipletests
rejected, p_adjusted, _, alpha_corrected = multipletests(feeltrace_to_words, alpha=0.05, 
                                                         method='holm', 
                                                         is_sorted=False, returnsorted=False)
print(f'Rejected H0: %d' % np.sum(rejected))
print(f'Corrected alpha: %f' % alpha_corrected)

sns.kdeplot(feeltrace_to_words, color="red", shade=True, label='raw')
ax = sns.kdeplot(p_adjusted, color="green", shade=True, label='adujusted')
ax.set(xlim=(0, 1))
plt.title('Distribution of p-values')
plt.legend();


# In[80]:


n_windows.loc[p_adjusted < 0.05, 'p_number']


# In[81]:


n_windows.loc[p_adjusted >= 0.05, 'p_number']


# #### Cointegration
# TODO: summarize

# In[82]:


cointegration_p_values


# #### Pearson's correlation

# In[83]:


pearsons_corr_values = [x[0] for x in pearsons_corr]
pearsons_corr_p_values = [x[1] for x in pearsons_corr]


# In[84]:


sns.kdeplot(pearsons_corr_values, color="red", shade=True, label='rho')
plt.title('Distribution of correlation values')
plt.legend();


# In[85]:


rejected, p_adjusted, _, alpha_corrected = multipletests(pearsons_corr_p_values, alpha=0.05, 
                                                         method='holm', 
                                                         is_sorted=False, returnsorted=False)
print(f'Rejected H0: %d' % np.sum(rejected))
print(f'Corrected alpha: %f' % alpha_corrected)

sns.kdeplot(feeltrace_to_words, color="red", shade=True, label='raw')
ax = sns.kdeplot(p_adjusted, color="green", shade=True, label='adujusted')
ax.set(xlim=(0, 1))
plt.title('Distribution of p-values')
plt.legend();


# In[86]:


n_windows.loc[p_adjusted < 0.05, 'p_number']


# In[87]:


n_windows.loc[p_adjusted >= 0.05, 'p_number']


# #### VAR Model
# 
# TODO: 
# - Summarize VAR model results
# - Cluster using VAR coefficients

# ## Emotion dynamics analysis
# 

# ### Defining emotion dynamics features
# Refer to _Houben M, Van Den Noortgate W, Kuppens P. The relation between short-term emotion dynamics and psychological well-being: A meta-analysis. Psychological bulletin. 2015 Jul;141(4):901._
# 
# - **Emotional inertia:** refers to how well the intensity of an emotional state can be predicted from the emotional state at a previous moment.
# - **Emotional instability:** refers to the magnitude of emotional changes from one moment to the next. An individual characterized by high levels of instability experiences larger emotional shifts from one moment to the next, resulting in a more unstable emotional life.
# - **Emotional variability:** refers to the range or amplitude of someone’s emotional states across time. An individual characterized by higher levels of emotional variability experiences emotions that reach more extreme levels and shows larger emotional deviations from his or her average emotional level

# In[88]:


class EmotionDynamics:
    def __init__(self, Fs=0.05, interval=300):
        self.lag = int(Fs*interval*1e-1) # feeltrace sampling rate x 300 ms

    def emotional_variability(self, X):
        return np.std(X)

    def emotional_instability(self, X):
        return np.sum((X[1:] - X[:-1])**2)/(len(X)-1) # MSSD

    def emotional_inertia(self, X, lag=None):
        if lag is None:
            lag = self.lag
        return stattools.acf(X, nlags=lag)[lag] # Autocorrelation
    
    def get_parameters(self, X):
        X = np.array(X)
        parameters = {'Inertia':'', 'Instability':'', 'Variability':''}
        parameters['Inertia'] = self.emotional_inertia(X)
        parameters['Instability'] = self.emotional_instability(X)
        parameters['Variability'] = self.emotional_variability(X)
        return parameters


# In[89]:


ED = EmotionDynamics(Fs=0.05)
ED.get_parameters(p10_words['Values'])


# In[90]:


words_data = []
for subject in words_list:
    calibrated_values = np.array(subject['Values'])
    ed = ED.get_parameters(calibrated_values)
    ed['p_number'] = subject['p_number'][0]
    ed['pass'] = 'interview'
    words_data.append(ed)
words_data = pd.DataFrame(words_data)
words_data.head()


# In[91]:


ED = EmotionDynamics(Fs=30)
feeltrace_data = []
for subject in feeltrace_list:
    feeltrace = np.array(subject['Feeltrace'])
    ed = ED.get_parameters(feeltrace)
    ed['p_number'] = subject['p_number'][0]
    ed['pass'] = 'feeltrace'
    feeltrace_data.append(ed)
    
feeltrace_data = pd.DataFrame(feeltrace_data)
feeltrace_data.head()


# In[92]:


X = pd.concat([words_data, feeltrace_data]).reset_index(drop=True)
X


# ### Data preprocessing: scaling
# Standardize features by removing the mean and scaling to unit variance.

# In[93]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[94]:


X_feeltrace = scaler.fit_transform(feeltrace_data[['Inertia', 'Instability', 'Variability']])
X_feeltrace = pd.DataFrame(X_feeltrace, columns=['Inertia', 'Instability', 'Variability'])
X_feeltrace['p_number'] = feeltrace_data['p_number']
X_feeltrace['pass'] = feeltrace_data['pass']
X_feeltrace


# In[95]:


X_words = scaler.fit_transform(words_data[['Inertia', 'Instability', 'Variability']])
X_words = pd.DataFrame(X_words, columns=['Inertia', 'Instability', 'Variability'])
X_words['p_number'] = words_data['p_number']
X_words['pass'] = words_data['pass']
X_words


# In[96]:


X_scaled = pd.concat([X_words, X_feeltrace]).reset_index(drop=True)


# ### Pairplot analysis

# In[97]:


sns.pairplot(X_scaled, hue='pass');


# In[98]:


abs(X_feeltrace[['Inertia', 'Instability', 'Variability']] - X_words[['Inertia', 'Instability', 'Variability']])


# ### 3D scatterplot

# TODO: color according to labelling pass

# In[99]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15,10))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=130, auto_add_to_figure=False, facecolor='w')
fig.add_axes(ax)

ax.scatter(X_scaled.Inertia, X_scaled.Instability, X_scaled.Variability, 
           cmap=plt.cm.nipy_spectral, edgecolor="k")

for (_, subject) in X_scaled.iterrows(): #plot each point + it's index as text above
    label = subject['p_number']
    ax.text(subject.Inertia,subject.Instability, subject.Variability,  '%s' % label, size=20, zorder=1,  
    color='k') 

ax.set_xlabel('Inertia')
ax.set_ylabel('Instability')
ax.set_zlabel('Variability')

plt.show()


# ### Principal Component Analysis

# In[100]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition

pca = decomposition.PCA(n_components=2)
pca.fit(X_scaled[['Inertia', 'Instability', 'Variability']])
X_PCA = pca.transform(X_scaled[['Inertia', 'Instability', 'Variability']])

plt.figure(figsize=(15,10))
sns.scatterplot(data=X, x=X_PCA[:, 0], y=X_PCA[:, 1], hue='pass');

for i, (_, subject) in enumerate(X_scaled.iterrows()): #plot each point + it's index as text above
    label = subject['p_number']
    plt.text(X_PCA[i, 0], X_PCA[i, 1],  '%s' % label, size=20, zorder=1,  
    color='k') 


# In[101]:


X_PCA = pd.DataFrame(X_PCA, columns=['PC1', 'PC2'])
X_PCA['p_number'] = X_scaled['p_number']
X_PCA['pass'] = X_scaled['pass']
X_PCA.head()


# In[102]:


# TODO: fix colors

from scipy.spatial import distance

pass_distance = []

for subject in X_PCA.groupby('p_number'):
    subject = subject[1]
    subject_interview = subject.loc[subject['pass'] == 'interview', ['PC1', 'PC2']]
    subject_feeltrace = subject.loc[subject['pass'] == 'feeltrace', ['PC1', 'PC2']]
    pass_distance.append({'p_number': subject.p_number.iloc[0], 'Distance': distance.euclidean(subject_interview, subject_feeltrace)})

pass_distance = pd.DataFrame(pass_distance)
sns.boxplot(data=pass_distance, y='Distance');


# ### Repeated Measures Analysis

# In[103]:


X_scaled.pivot(columns='pass').to_csv('scaled_ed_per_pass.csv', index=False)


# See Jamovi results.

# ### Gaussian Mixture Model

# In[104]:


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
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

lowest_bic = np.infty
bic = []
n_components_range = range(1, 16)
cv_types = ["spherical", "tied", "diag", "full"]
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(
            n_components=n_components, covariance_type=cv_type
        )
        gmm.fit(X_scaled[['Inertia', 'Instability', 'Variability']])
        bic.append(gmm.bic(X_scaled[['Inertia', 'Instability', 'Variability']]))
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
spl.legend([b[0] for b in bars], cv_types);


# In[105]:


# TODO: Fix legend

# Plot the winner
fig = plt.figure(figsize=(15,10))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=130, auto_add_to_figure=False, facecolor='w')
fig.add_axes(ax)
Y_ = clf.predict(X_scaled[['Inertia', 'Instability', 'Variability']])

for i, (mean, color) in enumerate(zip(clf.means_, color_iter)):
    if not np.any(Y_ == i):
        continue
    ax.scatter(X_scaled.Inertia.loc[Y_ == i], X_scaled.Instability.loc[Y_ == i], 
               X_scaled.Variability.loc[Y_ == i], s=100, color=color, edgecolor="k")

for (_, subject) in X_scaled.iterrows(): #plot each point + it's index as text above
    label = subject['p_number']
    ax.text(subject.Inertia,subject.Instability, subject.Variability,  '%s' % label, size=20, zorder=1,  
    color='k') 
    
plt.legend(range(len(clf.means_)))
ax.set_xlabel('Inertia')
ax.set_ylabel('Instability')
ax.set_zlabel('Variability')

plt.title(
    f"Selected GMM: {best_gmm.covariance_type} model, "
    f"{best_gmm.n_components} components"
)


plt.show()

