import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random as random
import pandas as pd
from matplotlib import cm

tmp = sio.loadmat("mousetracks.mat")
tracks = {}
for trackno in range(30):
    tracks[trackno] = tmp["num%d"%(trackno)]

plt.close("all")
for trackno in range(30):
    plt.plot(tracks[(trackno)][:,0],tracks[(trackno)][:,1],'.')
plt.axis("square")
plt.xlabel("meters")
plt.ylabel("meters")
plt.show()

X = np.zeros([30*50,2])

for trackno in range(30):
    X[(trackno*50):((trackno+1)*50),:] = tracks[trackno]

def kmeans(X,K=5,maxiter=100):
    maxval=np.amax(X,axis=0)
    minval=np.amin(X,axis=0)
    df=pd.DataFrame(X)
    df['Center']=0
    # initialize cluster centers
    random.seed(10)
    C =[[random.uniform(minval[j],maxval[j]) for j in range(2)] for i in range(5)]
    for iter in range(maxiter):      
        for i in range(1500):
            mindist= np.linalg.norm(X[i]-C[0])
            cent=1
            for j in range(1,5):
                if np.linalg.norm(X[i]-C[j])<mindist :
                    mindist= np.linalg.norm(X[i]-C[j])
                    cent= j+1
            df.iloc[i,2]= cent
        temp=df.groupby('Center')[[0,1]].mean()
        for k in range(K):
            C[k][0]=temp.iloc[k][0]
            C[k][1]=temp.iloc[k][1] 
    plt.close("all")
    df.columns=['x','y','Center']
    df.plot.scatter(x='x', y='y', c='Center',colormap=cm.get_cmap('Spectral'))
    C=np.asarray(C)
    plt.plot(C[:,0],C[:,1],'x',color='black',markersize=10)
    plt.axis("square")
    plt.xlabel("meters")
    plt.ylabel("meters")
    plt.show()
    return C

C=kmeans(X)
plt.close("all")
plt.plot(X[:,0],X[:,1],'.')
#uncomment to plot your cluster centers
plt.plot(C[:,0],C[:,1],'ro')
plt.axis("square")
plt.xlabel("meters")
plt.ylabel("meters")
plt.show()
print(C)


def kmedians(X,K=5,maxiter=100):
    maxval=np.amax(X,axis=0)
    minval=np.amin(X,axis=0)
    df=pd.DataFrame(X)
    df['Center']=0
    # initialize cluster centers
    random.seed(10)
    C =[[random.uniform(minval[j],maxval[j]) for j in range(2)] for i in range(5)]
    for iter in range(maxiter):      
        for i in range(1500):
            mindist= np.linalg.norm(X[i]-C[0],ord=1)
            cent=1
            for j in range(1,5):
                if np.linalg.norm(X[i]-C[j],ord=1)<mindist :
                    mindist= np.linalg.norm(X[i]-C[j],ord=1)
                    cent= j+1
            df.iloc[i,2]= cent
        temp=df.groupby('Center')[[0,1]].median()
        for k in range(K):
            C[k][0]=temp.iloc[k][0]
            C[k][1]=temp.iloc[k][1] 
    plt.close("all")
    df.columns=['x','y','Center']
    df.plot.scatter(x='x', y='y', c='Center',colormap=cm.get_cmap('Spectral'))
    C=np.asarray(C)
    plt.plot(C[:,0],C[:,1],'x',color='black',markersize=10)
    plt.axis("square")
    plt.xlabel("meters")
    plt.ylabel("meters")
    plt.show()
    return C

C=kmedians(X)
plt.close("all")
plt.plot(X[:,0],X[:,1],'.')
#uncomment to plot your cluster centers
plt.plot(C[:,0],C[:,1],'ro')
plt.axis("square")
plt.xlabel("meters")
plt.ylabel("meters")
plt.show()
print(C)

