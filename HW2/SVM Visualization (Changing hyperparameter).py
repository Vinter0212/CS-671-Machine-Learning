import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import pandas as pd
import sklearn
import math
import csv
from sklearn.datasets import fetch_california_housing
from sklearn import metrics
from sklearn.svm import SVC
from scipy.interpolate import griddata

def visualizestuff(X,y,gamma,C,ax):
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y)
    limx = ax.set_xlim()
    limy = ax.set_ylim()
    x1 = np.linspace(limx[0], limy[1],30)
    y1 = np.linspace(limy[0], limy[1],30)
    Ygrid, Xgrid = np.meshgrid(y1, x1)
    XY = np.transpose(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    classifier= SVC(kernel='rbf',random_state=0,C=C,gamma=gamma)
    classifier.fit(X,y)
    Z = classifier.decision_function(XY).reshape(Xgrid.shape)
    ax.contourf(Xgrid,Ygrid,Z,cmap='RdBu')
    ax.contour(Xgrid, Ygrid, Z,levels=[-1, 0, 1],linestyles=['--', '-', '--'],alpha=0.5,cmap='RdBu')
    ax.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=80,edgecolors='k',alpha=1)
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.Paired,alpha=0.6)
    
df = pd.read_csv("train.csv",header=None)
df.columns = df.iloc[0]
df=df.drop(df.index[0])
df.x1 = pd.to_numeric(df.x1, errors='coerce')
df.x2 = pd.to_numeric(df.x2, errors='coerce')
df.y = pd.to_numeric(df.y, errors='coerce')
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

fig = plt.figure()
ax= fig.add_subplot(111)
visualizestuff(X,y,1,1,ax)
plt.show()

gamma = 1
C=10000
fig = plt.figure()
fig.subplots_adjust(hspace=0.5)
fig.set_figheight(15)
fig.set_figwidth(15)
plt.suptitle(gamma)
m=0
for i in range(9):
    C_i=C/(10**i)
    ax= fig.add_subplot(331 + m)
    visualizestuff(X,y,gamma,C_i,ax)
    m=m+1
    ax.title.set_text(C_i)
plt.show()

gamma = 10000
C=1
fig = plt.figure()
fig.subplots_adjust(hspace=0.5)
fig.set_figheight(15)
fig.set_figwidth(15)
plt.suptitle(C)
m=0
for i in range(9):
    gamma_i=gamma/(10**i)
    ax= fig.add_subplot(331 + m)
    visualizestuff(X,y,gamma_i,C,ax)
    m=m+1
    ax.title.set_text(gamma_i)
plt.show()