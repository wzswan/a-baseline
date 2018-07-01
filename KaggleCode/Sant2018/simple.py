import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
import copy
#print(os.listdir("../input"))

# Models Packages
#from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
#from sklearn import preprocessing

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
#
# Fradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold


print("\nData Load Stage")
train_df = pd.read_csv("../input/train.csv")
traindex = train_df.index
test_df = pd.read_csv("../input/test.csv")
testdex = test_df.index
test_id = test_df['ID'].values
y = train_df['target'].values

print('Train shape: {} Rows, {} Columns'.format(*train_df.shape))
print('Test shape: {} Rows, {} Columns'.format(*test_df.shape))

colsToRemove = []
for col in train_df.columns:
    if col != 'ID' and col != 'target':
        if train_df[col].std() == 0:
            colsToRemove.append(col)

# remove constant columns in the training set
train_df.drop(colsToRemove, axis=1, inplace=True)

# remove constant columns in the test set
test_df.drop(colsToRemove, axis=1, inplace=True)

print("Removed `{}` Constant Columns\n".format(len(colsToRemove)))

def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break

    return dups

colsToRemove = duplicate_columns(train_df)
#print(colsToRemove)

train_df.drop(colsToRemove, axis=1, inplace=True)

# remove duplicate columns in the testing set
test_df.drop(colsToRemove, axis=1, inplace=True)

print("Removed `{}` Duplicate Columns\n".format(len(colsToRemove)))

def drop_sparse(train, test):
    flist = [x for x in train.columns if not x in ['ID','target']]
    for f in flist:
        if len(np.unique(train[f]))<2:
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
    return train, test

%%time
train_df, test_df = drop_sparse(train_df, test_df)

gc.collect()
print("Train set size: {}".format(train_df.shape))
print("Test set size: {}".format(test_df.shape))

print('save into csv file')
train_df.to_csv("prepro_train_df.csv", index=False)
test_df.to_csv("prepro_total_df_all.csv", index=False)
print("Combine Train and Test")
total_df = pd.concat([train_df.drop('target', axis=1), test_df], axis=0).drop('ID', axis= 1)
#del train_df, test_df
gc.collect()

total_df_all = copy.deepcopy(total_df)
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

total_df_all['__data_sum'] = 0
for col in total_df_all.columns:

    data = total_df_all[col].values
    #total_df_all[col + '_data_mean'] = data_mean
    #total_df_all[col + '_data_std'] = data_std
    total_df_all['__data_sum'] += data

total_df_all['__data_mean'] = total_df_all['__data_sum'] / len(total_df_all)

print('save into csv file')
total_df.to_csv("total_df.csv", index=False)
total_df_all.to_csv("total_df_all.csv", index=False)

def test_pca(data, create_plots=True):
    """Run PCA analysis, return embedding"""

    # Create a PCA object, specifying how many components we wish to keep
    pca = PCA(n_components=100)

    # Run PCA on scaled numeric dataframe, and retrieve the projected data
    pca_trafo = pca.fit_transform(data)

    # The transformed data is in a numpy matrix. This may be inconvenient if we want to further
    # process the data, and have a more visual impression of what each column is etc. We therefore
    # put transformed/projected data into new dataframe, where we specify column names and index
    pca_df = pd.DataFrame(
        pca_trafo,
        index=total_df.index,
        columns=["PC" + str(i + 1) for i in range(pca_trafo.shape[1])]
    )

    # Only construct plots if requested
    if create_plots:

        # Create two plots next to each other
        _, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = list(itertools.chain.from_iterable(axes))

        # Plot the explained variance# Plot t
        axes[0].plot(
            pca.explained_variance_ratio_, "--o", linewidth=2,
            label="Explained variance ratio"
        )

        # Plot the cumulative explained variance
        axes[0].plot(
            pca.explained_variance_ratio_.cumsum(), "--o", linewidth=2,
            label="Cumulative explained variance ratio"
        )

        # Show legend
        axes[0].legend(loc="best", frameon=True)



    return pca_df

# Run the PCA and get the embedded dimension
#pca_df = test_pca(total_df)
pca_df_all = test_pca(total_df_all, create_plots=False)

print('save into csv file')
#total_df.to_csv("total_df.csv", index=False)
pca_df_all.to_csv("pca_df_all.csv", index=False)

X = df.loc[traindex,:].values
test_df = df.loc[testdex,:].values

for shape in [X,test_df]:
    print("{} Rows and {} Cols".format(*shape.shape))

print("Light Gradient Boosting Regressor")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    # 'max_depth': 15,
    'num_leaves': 270,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.75,
    'bagging_freq': 2,
    'learning_rate': 0.016,
    'verbose': 0
}

if VALID == False:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, random_state=23)

    # LGBM Dataset Formatting
    lgtrain = lgb.Dataset(X_train, y_train)
    lgvalid = lgb.Dataset(X_valid, y_valid)
    del X, X_train; gc.collect()

    # Go Go Go
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=20000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    print("Model Evaluation Stage")
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
    del X_valid ; gc.collect()

else:
    # LGBM Dataset Formatting
    lgtrain = lgb.Dataset(X, y)
    del X; gc.collect()
    # Go Go Go
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=1550,
        verbose_eval=100
    )

lgpred = lgb_clf.predict(test_df)

#Mixing lightgbm with ridge. I haven't really tested if this improves the score or not
#blend = 0.95*lgpred + 0.05*ridge_oof_test[:,0]
lgsub = pd.DataFrame(lgpred,columns=["target"],index=testdex)
lgsub['target'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
lgsub.to_csv("lgsub.csv",index=True,header=True)
