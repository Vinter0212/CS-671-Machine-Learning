import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import pandas as pd
import sklearn
import copy
from sklearn.datasets import fetch_california_housing

plt.rcParams['font.size'] = 14

# Download data

tmp = sklearn.datasets.fetch_california_housing()
num_samples   = tmp['data'].shape[0]
feature_names = tmp['feature_names']
y = tmp['target']
X = tmp['data']

data = {}
for n, feature in enumerate(feature_names):
    data[feature] = tmp['data'][:,n]
    
def varimp(X, y,num_samples,feature_names,data):
    bins = {}
    bin_idx = (np.arange(0,1.1,0.1)*num_samples).astype(np.int16)

    bin_idx[-1] = bin_idx[-1]-1

    for feature in (feature_names):
        bins[feature] = np.sort(data[feature])[bin_idx]
# decision stumps as weak classifiers
# 0 if not in bin, 1 if in bin
    stumps = {}
    for feature in feature_names:
        stumps[feature] = np.zeros([num_samples,len(bins[feature])-1])
        for n in range(len(bins[feature])-1):
            stumps[feature][:,n] =  data[feature]>bins[feature][n]
# stack the weak classifiers into a matrix
    H = np.hstack([stumps[feature] for feature in feature_names])
    H = np.hstack([np.ones([num_samples,1]),H])
# prepare the vector for storing weights
    alphas = np.zeros(H.shape[1])
    num_iterations = 30
    MSE = np.zeros(num_iterations) 
    for iteration in range(num_iterations):
        f = np.dot(H,alphas)
        r = y-f; MSE[iteration] = np.mean(r**2) # r = residual
        idx = np.min(np.where(abs(np.dot(np.transpose(r),H))==np.max(abs(np.dot(np.transpose(r),H)))))
        alphas[idx] = alphas[idx] + np.dot(np.transpose(r),H[:,idx])/np.sum(H[:,idx])# amount to move in optimal direction
    print(MSE[29])
    alphasf = {}
    start = 1
    for feature in feature_names:
        alphasf[feature] = alphas[start:(start+stumps[feature].shape[1])]
        start = start + stumps[feature].shape[1]
    alphasf['mean'] = alphas[0]
    i=1
    m=1
    fig = plt.figure()
    fig.subplots_adjust(hspace=1)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.suptitle('contribution to house price')
    for feature in feature_names:
        #plt.close("all")
        
        ax= fig.add_subplot(330 + m)
        ax.plot(data[feature],y-np.mean(y),'.',alpha=0.5,color=[0.9,0.9,0.9])
        
        # plot stuff
        ax.plot(data[feature],np.dot(H[:,i:i+10],alphasf[feature])-np.mean(np.dot(H[:,i:i+10],alphasf[feature])),'.',alpha=0.5,color=[0,0,0.9])
        ax.set_xlim([bins[feature][0],bins[feature][-2]])
        ax.title.set_text(feature)
        i=i+10
        m+=1
    plt.show()

varimp(X, y,num_samples,feature_names,data)# original data

# Variable Importance
for features in feature_names:
    newdata=copy.deepcopy(data)
    newdata[features] = np.random.permutation(newdata[features])
    print(features)
    varimp(X, y,num_samples,feature_names,newdata)

#Boosted Decision Tree 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
clf = GradientBoostingRegressor(loss="ls")
clf.fit(X,y)
plt.close("all")
plt.figure(figsize=[10,10])
ax = plt.gca()
plot_partial_dependence(clf, X, feature_names, feature_names, n_cols=3, ax=ax) 
plt.tight_layout()
plt.show()

#Linear Regression
from sklearn.linear_model import LinearRegression
clf2 = LinearRegression()
clf2.fit(X,y)


#Comparison in MSE
print(np.mean((y-clf2.predict(X))**2))
print(np.mean((y-clf.predict(X))**2))
