import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import scipy.io as sio
from PIL import Image
from scipy import ndimage
import pylab
np.random.seed(1)

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache


def sigmoid_diff(Z):
    
    s = 1/(1+np.exp(-Z))
    dZ =  s * (1-s)
    return dZ

def initialize_parameters_deep(layer_dims):   
    parameters = {}
    L = len(layer_dims)            
    np.random.seed(1)
    for l in range(1, L):
       
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W,A)+b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b):
   
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = sigmoid(Z)
   
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2                
    

    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev,parameters['W' + str(l)] , parameters['b' + str(l)])
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)] , parameters['b' + str(L)])
    caches.append(cache)
            
    return AL, caches

def compute_cost(AL,Y):
    m=Y.shape[1]
    cost=-(1/m)*(np.dot(Y,np.transpose(np.log(AL)))+np.dot(1-Y,np.transpose(np.log(1-AL))))
    cost=np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m=A_prev.shape[1]
    dW = (1/m)*(np.dot(dZ,np.transpose(A_prev)))
    db = (1/m)*(np.dot(dZ,np.ones((dZ.shape[1],1))))
    dA_prev = np.dot(np.transpose(W),dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache):
    
    linear_cache, activation_cache = cache


    dZ = dA*sigmoid_diff(activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
     
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
   
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache)   

    for l in reversed(range(L-1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-(learning_rate*grads["dW" + str(l + 1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-(learning_rate*grads["db" + str(l + 1)])

    return parameters

def L_layer_model(X, Y, layers_dims, check,learning_rate = 0.1,num_iterations=3000):
    

    costs = []                         
    parameters = initialize_parameters_deep(layers_dims)


    for i in range(0, num_iterations):
  
     
        AL, caches = L_model_forward(X, parameters)
 

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        if(check==1):
            return np.linalg.norm(grads['dW1'],ord='fro')

        parameters = update_parameters(parameters, grads, learning_rate)
    

    return parameters

def predict(X, y, parameters):

    
    m = X.shape[1]
    n = len(parameters) 
    p = np.zeros((1,m))
    
 
    probas, caches = L_model_forward(X, parameters)

 
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
 
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

tmp = sio.loadmat("mnist_all.mat")
x1=tmp['train0']
x2=tmp['train1']
x3=tmp['test0']
x4=tmp['test1']

y1=[[0] for i in range(x1.shape[0])]
y2=[[1] for i in range(x2.shape[0])]
y3=[[0] for i in range(x3.shape[0])]
y4=[[1] for i in range(x4.shape[0])]

train_x_orig=np.concatenate((x1,x2))
test_x_orig=np.concatenate((x3,x4))
train_y=np.concatenate((y1,y2))

test_y=np.concatenate((y3,y4))


layers_dims = [784]
d=2
for j in range(d):
    layers_dims.append(20)
layers_dims.append(1)

train_y=np.transpose(train_y)
test_y=np.transpose(test_y)

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))



train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T


train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

parameters = L_layer_model(train_x, train_y, layers_dims,0, num_iterations = 3000)



pred_train = predict(train_x, train_y, parameters)

pred_test = predict(test_x, test_y, parameters)

layers_dims = [784]
for i in range(1,11):

    layers_dims.append(20)
    layers_dims.append(1)
    if i==1:
        normval=[L_layer_model(train_x, train_y, layers_dims, 1,num_iterations = 1)]
    else:
        normval.append(L_layer_model(train_x, train_y, layers_dims,1, num_iterations = 1))
    layers_dims.pop()
    
print(normval)


fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(2, 2, 1)

line, = ax.plot(normval, color='blue', lw=2)

ax.set_yscale('log')
plt.xlabel('Number of hidden layers')
plt.ylabel('Log of frobenius norm of dW1')
pylab.show()


