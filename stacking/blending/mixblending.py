# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import os

#print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.

import numpy as np

import pandas as pd

import math

import time

import os.path

d1=pd.read_csv('lgbtest2pop_price.csv')
#print(d1.shape)
#d1.head(5)
d2=pd.read_csv('result/subb2o21.csv')


d4=pd.read_csv('result/lgsub.csv')
d3=pd.read_csv('result/subb3_02219.csv')

d5=pd.read_csv('result/lgsub_100000b.csv')

#d4['deal_probability']=(d1['deal_probability']+d2['deal_probability'])/2

##d5['deal_probability']=(d4['deal_probability']+d5['deal_probability'])/2

def rmse(y, y0):

    assert len(y) == len(y0)

    return np.sqrt(np.mean(np.power((y - y0), 2)))


print("rmse with d1")

print (rmse(d1['deal_probability'], d2['deal_probability']),

rmse(d1['deal_probability'], d3['deal_probability']),

rmse(d1['deal_probability'], d4['deal_probability']),

rmse(d1['deal_probability'], d5['deal_probability']))



print("rmse with d2")

print(rmse(d2['deal_probability'], d3['deal_probability']),

rmse(d2['deal_probability'], d4['deal_probability']),

rmse(d2['deal_probability'], d5['deal_probability']),)
print("rmse with d3")
print (rmse(d3['deal_probability'], d4['deal_probability']))
rmse(d3['deal_probability'], d5['deal_probability'])
print("rmse with d4")
print (rmse(d4['deal_probability'], d5['deal_probability']))
#d1 and d3
d3['deal_probability']=(d1['deal_probability']+d3['deal_probability'])/2
d1['deal_probability']=(d1['deal_probability']+d3['deal_probability'])/2
d3['deal_probability']=(d1['deal_probability']+d3['deal_probability'])/2
d1['deal_probability']=(d1['deal_probability']+d3['deal_probability'])/2
d3['deal_probability']=(d1['deal_probability']+d3['deal_probability'])/2
d1['deal_probability']=(d1['deal_probability']+d3['deal_probability'])/2
d3['deal_probability']=(d1['deal_probability']+d3['deal_probability'])/2
d3['deal_probability']=(d1['deal_probability']+d3['deal_probability'])/2
d1['deal_probability']=(d1['deal_probability']+d3['deal_probability'])/2
d3['deal_probability']=(d1['deal_probability']+d3['deal_probability'])/2
d1['deal_probability']=(d1['deal_probability']+d3['deal_probability'])/2
d3['deal_probability']=(d1['deal_probability']+d3['deal_probability'])/2
d1['deal_probability']=(d1['deal_probability']+d3['deal_probability'])/2
d3['deal_probability']=(d1['deal_probability']+d3['deal_probability'])/2
#d2 and d4
d2['deal_probability']=(d4['deal_probability']+d2['deal_probability'])/2
d4['deal_probability']=(d4['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d4['deal_probability']+d2['deal_probability'])/2
d4['deal_probability']=(d4['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d4['deal_probability']+d2['deal_probability'])/2
d4['deal_probability']=(d4['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d4['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d4['deal_probability']+d2['deal_probability'])/2
d4['deal_probability']=(d4['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d4['deal_probability']+d2['deal_probability'])/2
d4['deal_probability']=(d4['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d4['deal_probability']+d2['deal_probability'])/2
d4['deal_probability']=(d4['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d4['deal_probability']+d2['deal_probability'])/2
#d2 and d3
d2['deal_probability']=(d3['deal_probability']+d2['deal_probability'])/2
d3['deal_probability']=(d3['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d3['deal_probability']+d2['deal_probability'])/2
d3['deal_probability']=(d3['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d3['deal_probability']+d2['deal_probability'])/2
d3['deal_probability']=(d3['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d3['deal_probability']+d2['deal_probability'])/2
d3['deal_probability']=(d3['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d3['deal_probability']+d2['deal_probability'])/2
d3['deal_probability']=(d3['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d3['deal_probability']+d2['deal_probability'])/2
d3['deal_probability']=(d3['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d3['deal_probability']+d2['deal_probability'])/2
rmse(d2['deal_probability'], d3['deal_probability'])
#d2 and d5
rmse(d2['deal_probability'], d5['deal_probability'])
d2['deal_probability']=(d5['deal_probability']+d2['deal_probability'])/2
d5['deal_probability']=(d5['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d5['deal_probability']+d2['deal_probability'])/2
d5['deal_probability']=(d5['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d5['deal_probability']+d2['deal_probability'])/2
d5['deal_probability']=(d5['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d5['deal_probability']+d2['deal_probability'])/2
d5['deal_probability']=(d5['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d5['deal_probability']+d2['deal_probability'])/2
d5['deal_probability']=(d5['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d5['deal_probability']+d2['deal_probability'])/2
d5['deal_probability']=(d5['deal_probability']+d2['deal_probability'])/2
d2['deal_probability']=(d5['deal_probability']+d2['deal_probability'])/2
rmse(d2['deal_probability'], d5['deal_probability'])
once = d2.copy()
once2 = once.copy()
d1=pd.read_csv('result/lgsub1.csv')

d2=pd.read_csv('lgbtest2pop_price.csv')


d4=pd.read_csv('result/subb2o21.csv')
d3=pd.read_csv('result/subb3_02219.csv')

d5=pd.read_csv('result/lgsub_100000b.csv')

d4['deal_probability']=(d1['deal_probability']+d2['deal_probability'])/2

d5['deal_probability']=(d4['deal_probability']+d3['deal_probability'])/2
d5['deal_probability']=(d4['deal_probability']+d5['deal_probability'])/2

print (rmse(once['deal_probability'], d1['deal_probability']),
rmse(once['deal_probability'], d2['deal_probability']),
rmse(once['deal_probability'], d3['deal_probability']),
rmse(once['deal_probability'], d4['deal_probability']),
rmse(once['deal_probability'], d5['deal_probability']))
d5['deal_probability']=(once['deal_probability']+d5['deal_probability'])/2
once['deal_probability']=(once['deal_probability']+d5['deal_probability'])/2
d5['deal_probability']=(once['deal_probability']+d5['deal_probability'])/2
once['deal_probability']=(once['deal_probability']+d5['deal_probability'])/2
d5['deal_probability']=(once['deal_probability']+d5['deal_probability'])/2
once['deal_probability']=(once['deal_probability']+d5['deal_probability'])/2
d5['deal_probability']=(once['deal_probability']+d5['deal_probability'])/2
once['deal_probability']=(once['deal_probability']+d5['deal_probability'])/2
d5['deal_probability']=(once['deal_probability']+d5['deal_probability'])/2
once['deal_probability']=(once['deal_probability']+d5['deal_probability'])/2
d5['deal_probability']=(once['deal_probability']+d5['deal_probability'])/2
once['deal_probability']=(once['deal_probability']+d5['deal_probability'])/2
d5['deal_probability']=(once['deal_probability']+d5['deal_probability'])/2
rmse(d5['deal_probability'], once['deal_probability'])
d5.to_csv("subb1w_19_2o_2pop_price.csv", index=False)
