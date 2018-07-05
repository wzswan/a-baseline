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
train = train.drop(["target", "ID"], axis=1)
test = test.drop(["ID"], axis=1)

print ("Rows: " + str(train.shape[0]) + ", Columns: " + str(train.shape[1]))
train.head()

feature_df = train.describe().T
feature_df = feature_df.reset_index().rename(columns = {'index' : 'columns'})
feature_df['distinct_vals'] = feature_df['columns'].apply(lambda x : len(train[x].value_counts()))
feature_df['column_var'] = feature_df['columns'].apply(lambda x : np.var(train[x]))
feature_df['target_corr'] = feature_df['columns'].apply(lambda x : np.corrcoef(target, train[x])[0][1])
feature_df.head()

feature_df[feature_df['column_var'].astype(float) <= 1]

feature_df = feature_df.sort_values('column_var', ascending = True)
#normalize the var to [0,1]
feature_df['column_var'] = (feature_df['column_var'] - feature_df['column_var'].min()) / (feature_df['column_var'].max() - feature_df['column_var'].min())

# Check and remove duplicate columns
colsToRemove = []
colsScaned = []
dupList = {}

columns = train.columns

for i in range(len(columns)-1):
    v = train[columns[i]].values
    dupCols = []
    for j in range(i+1,len(columns)):
        if np.array_equal(v, train[columns[j]].values):
            colsToRemove.append(columns[j])
            if columns[j] not in colsScaned:
                dupCols.append(columns[j])
                colsScaned.append(columns[j])
                dupList[columns[i]] = dupCols

small_col_var_list = feature_df[abs(feature_df['column_var'].astype(float)) <= 0.00005]['columns'].tolist()
small_target_corr_list = feature_df[abs(feature_df['target_corr'].astype(float)) <= 0.0005]['columns'].tolist()

#drop duplicate colums
cols = [c for c in colsToRemove if c in train.columns]
train = train.drop(cols, axis=1)
test = test.drop(cols, axis=1)

print("Removed `{}` Duplicate Columns\n".format(len(dupList)))
print(dupList)

#drop insignificant columns
cols = [c for c in small_col_var_list if c in train.columns]
train = train.drop(cols,axis=1)
test = test.drop(cols, axis=1)
cols = [c for c in small_target_corr_list if c in train.columns]
train = train.drop(cols,axis=1)
test = test.drop(cols, axis=1)

print ("Train Rows: " + str(train.shape[0]) + ", Columns: " + str(train.shape[1]))
print ("Test Rows: " + str(test.shape[0]) + ", Columns: " + str(test.shape[1]))

col_names = train.columns.values
train_norm = StandardScaler().fit_transform(train.values)
test_norm = StandardScaler().fit_transform(test.values)

train_norm = pd.DataFrame(train_norm)
#train_norm = train_norm.add_prefix('feature_')
test_norm = pd.DataFrame(test_norm)
#test_norm = test_norm.add_prefix('feature_')

train_norm.columns = col_names
test_norm.columns = col_names
train_norm['target'] = target
train_norm['ID'] = train_id
test_norm['ID'] = test_id
train_norm.to_csv(path+'removed_train.csv',index=False)
test_norm.to_csv(path+'removed_test.csv',index=False)

ntrain = train.shape[0]
ntest = test.shape[0]

#print('final column list')
#print(list(train.columns.values))

print("Combine Train and Test")
df = pd.concat([train,test],axis=0)

#print(ntrain,ntest)
#df = StandardScaler().fit_transform(df.values)

n_comp = 1800
print('start svd = '+str(n_comp))

obj_svd = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_df = obj_svd.fit_transform(df)
print(obj_svd.explained_variance_ratio_.sum()) #total percentage of retained information from SVD-ed matrix

print(svd_df.shape)

#add the feature name back
encoded_train = pd.DataFrame(svd_df[0:ntrain])
encoded_train = encoded_train.add_prefix('svd_feature_')

encoded_test = pd.DataFrame(svd_df[ntrain:])
encoded_test = encoded_test.add_prefix('svd_feature_')

encoded_train['target'] = target
encoded_train['ID'] = train_id
encoded_test['ID'] = test_id

encoded_train.to_csv(path+'encoded_svd_'+str(n_comp)+'d_train.csv',index=False)
encoded_test.to_csv(path+'encoded_svd_'+str(n_comp)+'d_test.csv',index=False) 
