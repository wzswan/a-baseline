import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.

import os
import gc
#print("Data:\n",os.listdir("../input"))
import time
notebookstart= time.time()

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

def cleanName(text):
    try:
        textProc = text.lower()
        # textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        #regex = re.compile(u'[^[:alpha:]]')
        #textProc = regex.sub(" ", textProc)
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except:
        return "name error"


print("\nData Load Stage (TFIDF)")
path = '~/.kaggle/competitions/avito-demand-prediction/train.csv'
i_path = '~/kaggle/avito_train_full.csv'
training = pd.read_csv('~/kaggle/avito_train_full.csv', index_col = "item_id", parse_dates = ["activation_date"])
testing = pd.read_csv('~/kaggle/avito_test_full.csv', index_col = "item_id", parse_dates = ["activation_date"])

#testcode #turn on this if you want to test the code only
#training = training[0:100]
#testing = testing[0:50]

print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print('Clean text data')
df['title'] = df['title'].apply(lambda x: cleanName(x))
df["description"]   = df["description"].apply(lambda x: cleanName(x))

print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
russian_stop = set(stopwords.words('russian'))

tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    "min_df":5,
    "max_df":.85,
    "smooth_idf":False
}

def get_col(col_name): return lambda x: x[col_name]
##I added to the max_features of the description. It did not change my score much but it may be worth investigating
vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=70000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('title',CountVectorizer(
            ngram_range=(1, 2),
            stop_words = russian_stop,
            max_df=0.85,
            min_df=1,
            max_features=30000,
            preprocessor=get_col('title')))
    ])

start_vect=time.time()

#Fit my vectorizer on the entire dataset instead of the training rows
#Score improved by .0001
vectorizer.fit(df.to_dict('records'))

ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()

print('write out csv now')
export_path = 'ready_df.npz'
#tfvocab.to_csv(export_path) #just the dictionary
from scipy import sparse
print(type(ready_df).__name__)
sparse.save_npz(export_path, ready_df)

with open("tfvocab.pickle", "wb") as fp:   #Pickling
    pickle.dump(tfvocab, fp)
print('saved tfvocab length: ' + str(len(tfvocab)))
print('generated ready_df shape: {} Rows, {} Columns'.format(*ready_df.shape))
