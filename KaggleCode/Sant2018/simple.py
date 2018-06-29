import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
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
from sklearn.pipline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
#
# Fradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold

NFOLDS = 5
SEED = 42
VALID = True

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))



train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

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
print('save into csv file')
total_df.to_csv("total_df.csv", index=False)
total_df_all.to_csv("total_df_all.csv", index=False)

print('total_df_all rows and col', total_df_all.shape)

def test_pca(data, create_plots=True):
    """Run PCA analysis, return embedding"""

    # Create a PCA object, specifying how many components we wish to keep
    pca = PCA(n_components=1000)

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
pca_df = test_pca(total_df)
pca_df_all = test_pca(total_df_all, create_plots=False)
print('save pca file')
pca_df.to_csv("pca_df.csv", index=False)
pca_df_all.to_csv("pca_df_all.csv", index=False)

def test_prediction(data):
    """Try to classify train/test samples from total dataframe"""

    # Create a target which is 1 for training rows, 0 for test rows
    y = np.zeros(len(data))
    y[train_idx] = 1

    # Perform shuffled CV predictions of train/test label
    predictions = cross_val_predict(
        ExtraTreesClassifier(n_estimators=100, n_jobs=4),
        data, y,
        cv=StratifiedKFold(
            n_splits=10,
            shuffle=True,
            random_state=42
        )
    )

    # Show the classification report
    print(classification_report(y, predictions))

# Run classification on total raw data
test_prediction(total_df_all)
