## Import the required python utilities
from plotly.offline import init_notebook_mode, iplot
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
import numpy as np

## Import sklearn important modules
from sklearn.decomposition import PCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.decomposition import TruncatedSVD, FastICA, NMF, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
import lightgbm as lgb

init_notebook_mode(connected=True)
path = "~/.kaggle/competitions/santander-value-prediction-challenge/"

train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
target = train['target']

train_id = train['ID']
test_id = test['ID']

n_comp = 1800
encoding_dim = 1000
removed_train = pd.read_csv(path+'removed_train.csv')
removed_test = pd.read_csv(path+'removed_test.csv')
svd_train = pd.read_csv(path+'encoded_svd_'+str(n_comp)+'d_train.csv')
svd_test = pd.read_csv(path+'encoded_svd_'+str(n_comp)+'d_test.csv')
encoded_train = pd.read_csv(path+'encoded_train_'+str(encoding_dim)+'d.csv')
encoded_test = pd.read_csv(path+'encoded_test_'+str(encoding_dim)+'d.csv')

bins = [0,1, 1e5, 1e6,1e7,1e100]
bins_label = ['zero', 'low', 'median','high','top']
train['target_group'] = pd.cut(train['target'], bins, labels=bins_label)
tgroup = train['target_group']
removed_train['target_group'] = tgroup
svd_train['target_group'] = tgroup
encoded_train['target_group'] = tgroup

removed_train.to_csv(path+'removed_train_fe.csv',index=False)
removed_test.to_csv(path+'removed_test_fe.csv',index=False)
svd_train.to_csv(path+'svd_'+str(n_comp)+'d_train_fe.csv',index=False)
svd_test.to_csv(path+'svd_'+str(n_comp)+'d_test_fe.csv',index=False)
encoded_train.to_csv(path+'encoded_train_'+str(encoding_dim)+'d_fe.csv',index=False)
encoded_test.to_csv(path+'encoded_test_'+str(encoding_dim)+'d_fe.csv',index=False)
