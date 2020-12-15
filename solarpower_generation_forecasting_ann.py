# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="UicUNHNs7wGC"
# # Solar Power Generation Forecasting By ANN
#
#

# %% [markdown] id="EUBRdZEs_Ucx"
# ## importing libraries

# %% id="-kMcwx2S7jOR"
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import regularizers
from keras.optimizers import RMSprop, Adam, SGD
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown] id="k6wzetIP_hgj"
# ## importing dataset

# %% id="UUYvoox9_gut" executionInfo={"status": "ok", "timestamp": 1601125742608, "user_tz": -330, "elapsed": 3081, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="e5fd778a-9df3-401b-f158-24b484072412" colab={"base_uri": "https://localhost:8080/", "height": 224}
#data_path = r'drive/My Drive/Proj/S.P.F./solarpowergeneration.csv'
dts = pd.read_csv('data/solarpowergeneration.csv')
dts.head(10)

# %% id="2p5Xv7sLGPVM" executionInfo={"status": "ok", "timestamp": 1601125746274, "user_tz": -330, "elapsed": 2721, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="20538b70-e5dd-43df-f181-5a510fb98c77" colab={"base_uri": "https://localhost:8080/", "height": 34} tags=[]
X = dts.iloc[:, :-1].values
y = dts.iloc[:, -1].values
print(X.shape, y.shape)
y = np.reshape(y, (-1,1))
y.shape

# %% id="ENid0bk0NJ-3" executionInfo={"status": "ok", "timestamp": 1601125748969, "user_tz": -330, "elapsed": 1521, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="ccd0d414-da4a-4f44-d528-503a83351b86" colab={"base_uri": "https://localhost:8080/", "height": 238}
X

# %% id="PFLG0kVFbbip" executionInfo={"status": "ok", "timestamp": 1601125750843, "user_tz": -330, "elapsed": 1455, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="e5461f56-cd8a-4e4e-f680-371cdb502f85" colab={"base_uri": "https://localhost:8080/", "height": 136}
y

# %% [markdown] id="Au0uOU0fJrmw"
# ## Splitting Training and Test sets

# %% id="lIlEtx-hJv3c" tags=[]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("Train Shape: {} {} \nTest Shape: {} {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

# %% [markdown] id="DxLkrF84POip"
# ## Feature Scaling

# %% id="puwEuG45PaBm"
from sklearn.preprocessing import StandardScaler
# input scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# outcome scaling:
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)    
y_test = sc_y.transform(y_test)

# %% id="7gSZs2z9VGxq" executionInfo={"status": "ok", "timestamp": 1601125757497, "user_tz": -330, "elapsed": 1380, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="89312473-af56-4502-eb56-47b4ac10201c" colab={"base_uri": "https://localhost:8080/", "height": 238}
X_train

# %% id="Rpk0HK7LVGjP" executionInfo={"status": "ok", "timestamp": 1601125759863, "user_tz": -330, "elapsed": 1362, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="e2f2b7b1-cc08-4db3-e34d-9cd431f5103b" colab={"base_uri": "https://localhost:8080/", "height": 238}
X_test

# %% id="famB9gS7c8Em" executionInfo={"status": "ok", "timestamp": 1601125762958, "user_tz": -330, "elapsed": 1794, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="8d1e41d0-6f5a-460d-8191-54a5a8755710" colab={"base_uri": "https://localhost:8080/", "height": 136}
y_train


# %% [markdown] id="6MwTUkzTWDTL"
# ## Creating Neural Network

# %% [markdown] id="SgPZtMeF86HH"
# ### defining accuracy function

# %% id="HPu2xxHcSQOB"
def create_spfnet(n_layers, n_activation, kernels):
  model = tf.keras.models.Sequential()
  for i, nodes in enumerate(n_layers):
    if i==0:
      model.add(Dense(nodes, kernel_initializer=kernels, activation=n_activation, input_dim=X_train.shape[1]))
      #model.add(Dropout(0.3))
    else:
      model.add(Dense(nodes, activation=n_activation, kernel_initializer=kernels))
      #model.add(Dropout(0.3))
  
  model.add(Dense(1))
  model.compile(loss='mse', 
                optimizer='adam',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
  return model



# %% id="4GmFjW2UePTI" tags=[]
spfnet = create_spfnet([32, 64], 'relu', 'normal')
spfnet.summary()

# %% tags=[]
from keras.utils.vis_utils import plot_model
plot_model(spfnet, to_file='spfnet_model.png', show_shapes=True, show_layer_names=True)

# %% id="lY4tgg3jjiqF" executionInfo={"status": "ok", "timestamp": 1601102450392, "user_tz": -330, "elapsed": 21938, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="b9756462-30db-4b7c-fab5-a39b088b0e32" colab={"base_uri": "https://localhost:8080/", "height": 1000} tags=[]
hist = spfnet.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test),epochs=150, verbose=2)

# %% id="f4co7KnVAdTH" executionInfo={"status": "ok", "timestamp": 1601102373259, "user_tz": -330, "elapsed": 2446, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="b37e337a-00df-4791-85af-760e67c5eda1" colab={"base_uri": "https://localhost:8080/", "height": 295}

plt.plot(hist.history['root_mean_squared_error'])
#plt.plot(hist.history['val_root_mean_squared_error'])
plt.title('Root Mean Squares Error')
plt.xlabel('Epochs')
plt.ylabel('error')
plt.show()

# %% tags=[]
spfnet.evaluate(X_train, y_train)

# %%
from sklearn.metrics import mean_squared_error

y_pred = spfnet.predict(X_test) # get model predictions (scaled inputs here)
y_pred_orig = sc_y.inverse_transform(y_pred) # unscale the predictions
y_test_orig = sc_y.inverse_transform(y_test) # unscale the true test outcomes

RMSE_orig = mean_squared_error(y_pred_orig, y_test_orig, squared=False)
RMSE_orig

# %%
train_pred = spfnet.predict(X_train) # get model predictions (scaled inputs here)
train_pred_orig = sc_y.inverse_transform(train_pred) # unscale the predictions
y_train_orig = sc_y.inverse_transform(y_train) # unscale the true train outcomes

mean_squared_error(train_pred_orig, y_train_orig, squared=False)

# %%
from sklearn.metrics import r2_score
r2_score(y_pred_orig, y_test_orig)

# %%
r2_score(train_pred_orig, y_train_orig)

# %%
np.concatenate((train_pred_orig, y_train_orig), 1)

# %%
np.concatenate((y_pred_orig, y_test_orig), 1)

# %%
plt.figure(figsize=(16,6))
plt.subplot(1,2,2)
plt.scatter(y_pred_orig, y_test_orig)
plt.xlabel('Predicted Generated Power on Test Data')
plt.ylabel('Real Generated Power on Test Data')
plt.title('Test Predictions vs Real Data')
#plt.scatter(y_test_orig, sc_X.inverse_transform(X_test)[:,2], color='green')
plt.subplot(1,2,1)
plt.scatter(train_pred_orig, y_train_orig)
plt.xlabel('Predicted Generated Power on Training Data')
plt.ylabel('Real Generated Power on Training Data')
plt.title('Training Predictions vs Real Data')
plt.show()

# %% tags=[]
x_axis = sc_X.inverse_transform(X_train)[:,-1]
x2_axis = sc_X.inverse_transform(X_test)[:,-1]
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.scatter(x_axis, y_train_orig, label='Real Generated Power')
plt.scatter(x_axis, train_pred_orig, c='red', label='Predicted Generated Power')
plt.ylabel('Predicted and real Generated Power on Training Data')
plt.xlabel('Solar Azimuth')
plt.title('Training Predictions vs Solar Azimuth')
plt.legend(loc='lower right')

plt.subplot(1,2,2)
plt.scatter(x2_axis, y_test_orig, label='Real Generated Power')
plt.scatter(x2_axis, y_pred_orig, c='red', label='Predicted Generated Power')
plt.ylabel('Predicted and real Generated Power on TEST Data')
plt.xlabel('Solar Azimuth')
plt.title('TEST Predictions vs Solar Azimuth')
plt.legend(loc='lower right')
plt.show()

# %%
results = np.concatenate((y_test_orig, y_pred_orig), 1)
results = pd.DataFrame(data=results)
results.columns = ['Real Solar Power Produced', 'Predicted Solar Power']
#results = results.sort_values(by=['Real Solar Power Produced'])
pd.options.display.float_format = "{:,.2f}".format
#results[800:820]
results[7:18]

# %%
sc = StandardScaler()
pred_whole = spfnet.predict(sc.fit_transform(X))
pred_whole_orig = sc_y.inverse_transform(pred_whole)
pred_whole_orig

# %%
y

# %%
r2_score(pred_whole_orig, y)

# %%
df_results = pd.DataFrame.from_dict({
    'R2 Score of Whole Data Frame': r2_score(pred_whole_orig, y),
    'R2 Score of Training Set': r2_score(train_pred_orig, y_train_orig),
    'R2 Score of Test Set': r2_score(y_pred_orig, y_test_orig),
    'Mean of Test Set': np.mean(y_pred_orig),
    'Standard Deviation pf Test Set': np.std(y_pred_orig),
    'Relative Standard Deviation': np.std(y_pred_orig) / np.mean(y_pred_orig),
},orient='index', columns=['Value'])
display(df_results.style.background_gradient(cmap='afmhot', axis=0))

# %%
corr = data.corr()
plt.figure(figsize=(22,22))
sns.heatmap(corr, annot=True, square=True);

# %% [markdown]
# **OBSERVATIONS**
# - High Correlation between Zenith and Agnle of Incidence of 0.71
# - Shortwave radiation backwards and Generate Power KW has corr of 0.56
# - Relative Humidity and Zenith are +ve corr (0.51)
# - Relative Humidity and Low Cloud Cover are + ve correlated (0.49)
# - Angle of Incidence and Zenith are -vely correlated with Genarted Power (-0.65)
# - -ve corr between Zenith and temperature of -0.55
# - High negative corr exists btw Shortwave radiation backwards and Zenith (-.8)
# - Shortwave radiation backwards and Relative humidity are -vely correlated (-.72)
# - Relative humidity and Temperature are -vely correlated (-.77)
# - 
#

# %%
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.001)

lasso.fit(X_train, y_train)

y_pred_lasso = lasso.predict(X_test)

lasso_coeff = pd.DataFrame({'Feature Importance':lasso.coef_}, index=data.columns[:-1])
lasso_coeff.sort_values('Feature Importance', ascending=False)

# %%
g = lasso_coeff[lasso_coeff['Feature Importance']!=0].sort_values('Feature Importance').plot(kind='barh',figsize=(6,6), cmap='winter')

# %%
