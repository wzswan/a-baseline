import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


color = sns.color_palette()
%matplotlib inline


# not available on our server
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


#baisc data info
from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

train_df = pd.read_csv("../input/train.csv", parse_dates=[""])
test_df = pd.read_csv("../input/test.csv", parse_dates=[""])
#train_df = pd.read_csv("../input/train.csv", parse_dates=["activation_date"])
#est_df = pd.read_csv("../input/test.csv", parse_dates=["activation_date"])

print("Train file rows and columns are : ", train_df.shape)
print("Test file rows and columns are : ", test_df.shape)

train_df.head()
