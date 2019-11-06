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

# Calculates kernel value
def rbfker(x1,x2,gamma):
    return math.exp(-1*gamma*((np.linalg.norm(x1-x2))**2))

# Populates Gram matrix + 1/C* Identity matrix
def Kmatrix(df,gamma,length,C):
    kmatrix= [[rbfker(X.iloc[k,:].to_numpy(),X.iloc[j,:].to_numpy(),gamma) + 1/C if(k==j)else rbfker(X.iloc[k,:].to_numpy(),X.iloc[j,:].to_numpy(),gamma)  for j in range(length)] for k in range(length)] 
    return kmatrix

#Populates the coefficient matrix for the equations
def Inmatrix(kmatrix,length):
    inmatrix=np.zeros((length+1,length+1))
    for i in range(length+1):
        for j in range(length+1):
            if(i==0 and j==0): inmatrix[i][j]=0
            elif(i==0): inmatrix[i][j]=1
            elif(j==0): inmatrix[i][j]=1
            else: inmatrix[i][j]= kmatrix[i-1][j-1]
    return inmatrix

# Calculates the Predicted value from classifier
def calculate_fx(len_test,length,solution,X,X_test,gamma):
    f_x=np.zeros((len_test))
    for i in range(len_test):
        s=0
        for j in range(length):
            s+= solution[j+1]*rbfker(X.iloc[j,:].to_numpy(),X_test.iloc[i,:].to_numpy(),gamma)
        f_x[i]=s+ solution[0]
    return f_x

#Classifies a point using the sign of classifier
def classify_fx(f_x):
    y_pred= [1 if i>=0 else -1 for i in f_x]
    return y_pred

#Populates the dataframe for boxplot using different SVM kernels as well as LS-SVM
def populatebox(A,boxval,gamma,C,X,y,length,X_test,y_test,len_test):
    kmatrix=Kmatrix(X,gamma,length,C)
    inmatrix=np.zeros((length+1,length+1))
    inmatrix = Inmatrix(kmatrix,length)
    rhs=[0]
    rhs.extend(y.to_numpy())
    solution = np.linalg.solve(inmatrix, rhs)
    
    f_x= calculate_fx(len_test,length,solution,X,X_test,gamma)
    y_pred=classify_fx(f_x)
    boxval['LS-SVM-Test'].iloc[A]= 1-metrics.accuracy_score(y_test, y_pred)
    
    f_x_train= calculate_fx(length,length,solution,X,X,gamma)
    y_pred_train=classify_fx(f_x_train)
    boxval['LS-SVM-Train'].iloc[A]= 1-metrics.accuracy_score(y, y_pred_train)
    
    classifier= SVC(kernel='rbf',random_state=0,C=C,gamma=gamma)
    classifier.fit(X,y)
    y_pred_SVM=classifier.predict(X_test)
    boxval['Rbf-SVM-Test'].iloc[A]= 1-metrics.accuracy_score(y_test, y_pred_SVM)
    
    y_pred_train_SVM=classifier.predict(X)
    boxval['Rbf-SVM-Train'].iloc[A]= 1-metrics.accuracy_score(y, y_pred_train_SVM)
    
    classifier1= SVC(kernel='linear',random_state=0,C=C,gamma=gamma)
    classifier1.fit(X,y)
    y_pred_SVM1=classifier1.predict(X_test)
    boxval['Linear-SVM-Test'].iloc[A]= 1-metrics.accuracy_score(y_test, y_pred_SVM1)
    
    y_pred_train_SVM1=classifier1.predict(X)
    boxval['Linear-SVM-Train'].iloc[A]= 1-metrics.accuracy_score(y, y_pred_train_SVM1)
    
    classifier2= SVC(kernel='poly',random_state=0,C=C,gamma=gamma)
    classifier2.fit(X,y)
    y_pred_SVM2=classifier2.predict(X_test)
    boxval['Polynomial-SVM-Test'].iloc[A]= 1-metrics.accuracy_score(y_test, y_pred_SVM2)
    
    y_pred_train_SVM2=classifier2.predict(X)
    boxval['Polynomial-SVM-Train'].iloc[A]= 1-metrics.accuracy_score(y, y_pred_train_SVM2)
    
    classifier3= SVC(kernel='sigmoid',random_state=0,C=C,gamma=gamma)
    classifier3.fit(X,y)
    y_pred_SVM3=classifier3.predict(X_test)
    boxval['Sigmoid-SVM-Test'].iloc[A]= 1-metrics.accuracy_score(y_test, y_pred_SVM3)
    
    y_pred_train_SVM3=classifier3.predict(X)
    boxval['Sigmoid-SVM-Train'].iloc[A]= 1-metrics.accuracy_score(y, y_pred_train_SVM3)

#Read Training data 
df = pd.read_csv("train.csv",header=None)
df.columns = df.iloc[0]
df=df.drop(df.index[0])
df.x1 = pd.to_numeric(df.x1, errors='coerce')
df.x2 = pd.to_numeric(df.x2, errors='coerce')
df.y = pd.to_numeric(df.y, errors='coerce')
length= df.shape[0]
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#Read Test Data
df_test = pd.read_csv("test.csv",header=None)
df_test.columns = df_test.iloc[0]
df_test=df_test.drop(df_test.index[0])
df_test.x1 = pd.to_numeric(df_test.x1, errors='coerce')
df_test.x2 = pd.to_numeric(df_test.x2, errors='coerce')
df_test.y = pd.to_numeric(df_test.y, errors='coerce')
X_test=df_test.iloc[:,:-1]
y_test=df_test.iloc[:,-1]
len_test=df_test.shape[0]

#fix C and vary gamma
gamma = 100
C=1
N_rows=7
A=0
boxval1 = pd.DataFrame(np.zeros((N_rows, 12)),columns =['Gamma','C','LS-SVM-Train', 'LS-SVM-Test', 'Rbf-SVM-Train','Rbf-SVM-Test','Linear-SVM-Train','Linear-SVM-Test','Polynomial-SVM-Train','Polynomial-SVM-Test','Sigmoid-SVM-Train','Sigmoid-SVM-Test'])
for j in range(7):
    gamma_j=gamma/(10**j)
    boxval1['C'].iloc[A]= C
    boxval1['Gamma'].iloc[A]= gamma_j
    populatebox(A,boxval1,gamma_j,C,X,y,length,X_test,y_test,len_test)
    A+=1
print(boxval1)

boxval1.boxplot(column=['LS-SVM-Train', 'LS-SVM-Test', 'Rbf-SVM-Train','Rbf-SVM-Test','Linear-SVM-Train','Linear-SVM-Test','Polynomial-SVM-Train','Polynomial-SVM-Test','Sigmoid-SVM-Train','Sigmoid-SVM-Test'])
plt.xticks(rotation=90)
plt.show()

gamma = 1
C=10000
N_rows=9
A=0
pd.set_option('precision', 5)
boxval = pd.DataFrame(np.zeros((N_rows, 12)),columns =['Gamma','C','LS-SVM-Train', 'LS-SVM-Test', 'Rbf-SVM-Train','Rbf-SVM-Test','Linear-SVM-Train','Linear-SVM-Test','Polynomial-SVM-Train','Polynomial-SVM-Test','Sigmoid-SVM-Train','Sigmoid-SVM-Test'])
for i in range(9):
    C_i=C/(10**i)
    boxval['C'].iloc[A]= C_i
    boxval['Gamma'].iloc[A]= gamma
    populatebox(A,boxval,gamma,C_i,X,y,length,X_test,y_test,len_test)
    A+=1

print(boxval)
boxval.boxplot(column=['LS-SVM-Train', 'LS-SVM-Test', 'Rbf-SVM-Train','Rbf-SVM-Test','Linear-SVM-Train','Linear-SVM-Test','Polynomial-SVM-Train','Polynomial-SVM-Test','Sigmoid-SVM-Train','Sigmoid-SVM-Test'])
plt.xticks(rotation=90)
plt.show()