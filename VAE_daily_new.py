"""
Created on Sun Mar  4 20:51:33 2018

@author: nicjm
"""

## VAE to cluster agents based on the strategy and object cluster

# Packages
#import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import norm
#from sklearn import preprocessing

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics

np.random.seed(0)

# Load Data
#workbook = xlrd.open_workbook('C:\\Users\\nicjm\\Documents\\Python\\VAEs\\ExpertWealth.xlsx')
#ExpertWealth = pd.read_excel('C:\\Users\\nicjm\\Documents\\Python\\VAEs\\Daily\\15_strats17_22-Oct-2009-29-Apr-2016_3001SH.xlsx',"Sheet1",header=None)
ExpertWealth = pd.read_excel('C:\\Users\\nicjm\\Documents\\Python\\VAEs\\Daily\\15_strats17_22-Oct-2009-29-Apr-2016_3001PnL_experts.xlsx',"Sheet1",header=None)
Params = pd.read_excel('C:\\Users\\nicjm\\Documents\\Python\\VAEs\\Daily\\15_strats17_3001parameters.xlsx',"Sheet1",header=None)
Paramsdf = Params
# Create dataframe of parameters- convert from strings from the excel file
#ignore the first 110 days as this is where no trading takes place
expwealth1 = ExpertWealth.iloc[:,:]
    
# Pre process
Params1 = Params.transpose()     
df1 = expwealth1.transpose()    

#mine
dflog = np.log(df1)
df2 = dflog.diff(1,1)
for i in range(df2.shape[1]):
    mean = np.mean(df2.iloc[:,i])
    std = np.std(df2.iloc[:,i])
    df2.iloc[:,i]= (df2.iloc[:,i]-mean)/std
df2 = np.nan_to_num(df2)
df2 = pd.DataFrame(data=df2,    # values
             index = df1.index,    # 1st column as index
             columns=df1.columns)


## Specify VAE parameters 
batch_size = 120
original_dim = df2.shape[1]
latent_dim = 2
intermediate_dim = 2560
epochs = 600
epsilon_std = 1.0


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='tanh')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='tanh')
decoder_mean = Dense(original_dim, activation='tanh')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
vae = Model(x, x_decoded_mean)

# Compute VAE loss
#binary_crossentropy
xent_loss = original_dim * metrics.mean_squared_error(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

# train the VAE on MNIST digits
x_train = df2#.iloc# [0:1000,:]
x_test = df2 #.iloc#[1000:,:]
      
strats = ['','EMAXover','Ichimoku','MACD','MovAveXover','ACC','Boll','Fast Stochastic'
          ,'MARSI','AntiBCRP','BCRP','PROC','RSI','SAR','Slow Stochastic','Williams%R']

history = vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size) 

plt.figure()
plt.plot(history.history['loss'])

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# get startegy names for all experts in expertparams1
y_test = [0 for i in range(df2.shape[0])]     
for m in range(0,len(df1)-1):
   for k in range(0,14):
     if k== int(Paramsdf.iloc[m,1]):
        y_test[m] = strats[k]

## display a 2D plot of the digit classes in the latent space
#Create differet sizes for markers based on cluster size and define markers for each stock
        # cluster
markersizes = Paramsdf.iloc[:,0].as_matrix().tolist()
markersizes = list(map(int, markersizes))  #convert all list items to integers
markers = ['s','d','o','2']

# define the markers for all the parameters
markersParams = [0 for i in range(df1.shape[0])]
for i in range(0,df1.shape[0]):
  markersParams[i] = markers[int(Paramsdf.iloc[i,0])-1]
     

#x and y coordinates of plot 
x_test_encoded = encoder.predict(df2, batch_size=batch_size)

#define figure and add points, markers etx to figure
fig, ax = plt.subplots()
for i in range(0,x_test_encoded.shape[0]):
    ax.scatter(x_test_encoded[i, 0],x_test_encoded[i, 1], marker = markersParams[i])
#ax.scatter(z, y)

## plot with colourmap for strategies and different sizes for different clusters
plt.figure(figsize=(4.7, 4.7))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], s = (200*markersizes)*2,c = Paramsdf.iloc[:,1],cmap=plt.cm.tab20)

plt.colorbar()
plt.show()
plt.savefig('strategies_daily.png', dpi=300)

## Plot with only colourmap for stock cluster and stategies
#colors = itertools.cycle(["r", "b", "g","y"])
cmap=plt.cm.rainbow
plt.figure(figsize=(4.7, 4.7))
#norm = matplotlib.colors.BoundaryNorm(np.arange(-2.5,3,1), cmap.N)
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1],cmap=cmap, s = (200*markersizes)*2,c = Paramsdf.iloc[:,0])
plt.colorbar()
plt.show()
plt.savefig('stockclusters_daily.png', dpi=300)