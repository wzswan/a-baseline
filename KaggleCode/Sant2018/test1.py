import numpy as np
import pandas as pd
import gc

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb
import xgboost as xgb

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

colsRemove = []
for col in train_df.columns:
    if col != 'ID' and col !='target':
        if train_df[col].std() == 0:
            colsRemove.append(col)

train_df.drop(colsRemove, axis =1, inplace = True)
test_df.drop(colsRemove, axis = 1, inplace = True)

print("Remove'{}' cols¥n'".format(len(colsRemove)))

colsToRemove = []
colsScaned = []
dupList = {}
columns = total_df.columsns
for i in range(len(columns)-1):
    v = train_df[columsns[i]].values
    dupCols = []
    for j in range(i+1, len(columsn)):
        if np.array_equal(v, train_df[j]i.values):
            colsToRemove.append(columns[j])
            if columns[j] not in colsScaned:
                dupCols.append(columns[j])
                dupList[columns[i]] = dupCols
colsToRemove = list(set(colsToRemove))
total_df.drop(colsToRemove, axis=1, inplace = True)


colsToRemove = []
colsScaned = []
dupList = {}
columns = train_df.columns
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
train_df.drop(colsToRemove, axis=1, inplace=True)
print(f">> Dropped {len(colsToRemove)} duplicate columns")

test_df.drop(colsToRemove, axis =1, inplace=True)

def drop_sparse(train, test):
    flist = [x for x in train.columns if not x in ['ID', 'target']]
    for f in flist:
        if len(np.unique(train[f]))<2:
            train.drop(f, axis =1, inplace = True)
            test.drop(f, axis =1, inplace = True)
    return train, test

def drop_sparse(train, test):
    flist = [x for x in train.columns if not x in ['ID','target']]
    for f in flist:
        if len(np.unique(train[f]))<2:
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
    return train,

train_df, test_df = drop_sparse(train_df, test_df)

X_train = train_df.drop(['ID','target'], axis=1)
y_train = np.log1p(train_df['target'].values)
#y_train_basic = train_df['target'].values
#y_train_("../tmp/y_train_basic_0704.csv", index=False)
X_test = test_df.drop(['ID'], axis =1)

total_df = pd.concat([X_train, X_test], axis=0)

import copy
from sklearn.preprocessing import scale

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
    total_df.loc[nonzero_rows, col] = scale(total_df.loc[nonzero_rows, col])# Here can choose different scale method

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

total_df.to_csv("../tmp/total_df_0704.csv", index=False)
total_df_all.to_csv("../tmp/total_df_all_0704.csv", index=False)

#pca_df = pd.read_csv('../tmp/total_df_0704.csv')
#pca_df_all = pd.read_csv('../tmp/total_df_all_0704.csv')

train_X_all = pca_df_all.iloc[:4459,:]
test_X_all = pca_df_all.iloc[4459:,:]

X_train = train_X_all
X_test = test_X_alldef run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
       "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "learning_rate" : 0.01,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=200, evals_result=evals_result)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
pred_test_full = 0
for dev_index, val_index in kf.split(X_train):
    dev_X, val_X = X_train.loc[dev_index,:], X_train.loc[val_index,:]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)
    pred_test_full += pred_test
pred_test_full /= 5.
pred_test_full = np.expm1(pred_test_full)

sub_df = pd.DataFrame({"ID":test_df["ID"].values})
sub_df["target"] = pred_test_full
sub_df.to_csv("baseline_lgb_pca.csv", index=False)
