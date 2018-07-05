#run this after remove column
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model

path = "~/.kaggle/competitions/santander-value-prediction-challenge/"

train = pd.read_csv(path+'removed_train.csv')
test = pd.read_csv(path+'removed_test.csv')

target = train['target']
train_id = train['ID']
test_id = test['ID']
train.drop(['target'], axis=1, inplace=True)

train.drop(['ID'], axis=1, inplace=True)
test.drop(['ID'], axis=1, inplace=True)

print('Train data shape', train.shape)
print('Test data shape', test.shape)

train_scaled = minmax_scale(train, axis = 0)
test_scaled = minmax_scale(test, axis = 0)

# define the number of features
ncol = train_scaled.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(train_scaled, target, train_size = 0.9, random_state = seed(2017))

### Define the encoder dimension
encoding_dim = 1000

input_dim = Input(shape = (ncol, ))

# Encoder Layers
#encoded1 = Dense(3000, activation = 'relu')(input_dim)
#encoded2 = Dense(2750, activation = 'relu')(encoded1)
#encoded3 = Dense(2500, activation = 'relu')(encoded2)
#encoded4 = Dense(2250, activation = 'relu')(encoded3)
#encoded5 = Dense(2000, activation = 'relu')(encoded4)
#encoded6 = Dense(1750, activation = 'relu')(encoded5)
#encoded7 = Dense(1500, activation = 'relu')(encoded6)
#encoded8 = Dense(1250, activation = 'relu')(encoded7)
#encoded9 = Dense(1000, activation = 'relu')(encoded8)
#encoded10 = Dense(750, activation = 'relu')(encoded9)
#encoded11 = Dense(600, activation = 'relu')(encoded10)
encoded12 = Dense(encoding_dim, activation = 'relu')(input_dim)

# Decoder Layers
#decoded1 = Dense(600, activation = 'relu')(encoded12)
#decoded2 = Dense(750, activation = 'relu')(decoded1)
#decoded3 = Dense(1000, activation = 'relu')(decoded2)
#decoded4 = Dense(1250, activation = 'relu')(decoded3)
#decoded5 = Dense(1500, activation = 'relu')(decoded4)
#decoded6 = Dense(1750, activation = 'relu')(decoded5)
#decoded7 = Dense(2000, activation = 'relu')(decoded6)
#decoded8 = Dense(2250, activation = 'relu')(decoded7)
#decoded9 = Dense(2500, activation = 'relu')(decoded8)
#decoded10 = Dense(2750, activation = 'relu')(decoded9)
#decoded11 = Dense(3000, activation = 'relu')(decoded10)
decoded12 = Dense(ncol, activation = 'sigmoid')(encoded12)

# Combine Encoder and Deocder layers
autoencoder = Model(inputs = input_dim, outputs = decoded12)

# Compile the Model
autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_logarithmic_error')

autoencoder.summary()

autoencoder.fit(X_train, X_train, nb_epoch = 10, batch_size = 32, shuffle = False, validation_data = (X_test, X_test))

encoder = Model(inputs = input_dim, outputs = encoded12)
encoded_input = Input(shape = (encoding_dim, ))

#add the feature name back
encoded_train = pd.DataFrame(encoder.predict(train_scaled))
encoded_train = encoded_train.add_prefix('feature_')

encoded_test = pd.DataFrame(encoder.predict(test_scaled))
encoded_test = encoded_test.add_prefix('feature_')

encoded_train['target'] = target
encoded_train['ID'] = train_id
encoded_test['ID'] = test_id
print(encoded_train.shape)
encoded_train.to_csv(path+'encoded_train_'+str(encoding_dim)+'d.csv',index=False)
encoded_test.to_csv(path+'encoded_test_'+str(encoding_dim)+'d.csv',index=False)
