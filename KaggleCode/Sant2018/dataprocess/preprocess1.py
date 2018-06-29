import gc
import itertools
from copy import deepcopy

import numpy as np
import pandas as pd

from tqdm import tqdm

from scipy.stats import ks_2samp

from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
%matplotlib inline

%%time

# How many samples to take from both train and test
SAMPLE_SIZE = 4459

# Read train and test files
train_df = pd.read_csv('../../../../train.csv').sample(SAMPLE_SIZE)
test_df = pd.read_csv('../../../../test.csv').sample(SAMPLE_SIZE)

# Get the combined data
total_df = pd.concat([train_df.drop('target', axis=1), test_df], axis=0).drop('ID', axis=1)

# Columns to drop because there is no variation in training set
zero_std_cols = train_df.drop("ID", axis=1).columns[train_df.std() == 0]
total_df.drop(zero_std_cols, axis=1, inplace=True)
print(f">> Removed {len(zero_std_cols)} constant columns")

# Removing duplicate columns
# Taken from: https://www.kaggle.com/scirpus/santander-poor-mans-tsne
colsToRemove = []
colsScaned = []
dupList = {}
columns = total_df.columns
for i in range(len(columns)-1):
    v = train_df[columns[i]].values
    dupCols = []
    for j in range(i+1,len(columns)):
        if np.array_equal(v, train_df[columns[j]].values):
            colsToRemove.append(columns[j])
            if columns[j] not in colsScaned:
                dupCols.append(columns[j])
                colsScaned.append(columns[j])
                dupList[columns[i]] = dupCols
colsToRemove = list(set(colsToRemove))
total_df.drop(colsToRemove, axis=1, inplace=True)
print(f">> Dropped {len(colsToRemove)} duplicate columns")

# Go through the columns one at a time (can't do it all at once for this dataset)
total_df_all = deepcopy(total_df)
for col in total_df.columns:

    # Detect outliers in this column
    data = total_df[col].values
    data_mean, data_std = np.mean(data), np.std(data)
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    outliers = [x for x in data if x < lower or x > upper]

    # If there are crazy high values, do a log-transform
    if len(outliers) > 0:
        non_zero_idx = data != 0
        total_df.loc[non_zero_idx, col] = np.log(data[non_zero_idx])

    # Scale non-zero column values
    nonzero_rows = total_df[col] != 0
    total_df.loc[nonzero_rows, col] = scale(total_df.loc[nonzero_rows, col])

    # Scale all column values
    total_df_all[col] = scale(total_df_all[col])
    gc.collect()

# Train and test
train_idx = range(0, len(train_df))
test_idx = range(len(train_df), len(total_df))
total_df.to_csv("total_df.csv", index=False)
total_df_all.to_csv("total_df_all.csv", index=False)
