import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def incomplete_data(y):
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if math.isnan(y[i, j]) == True:  # if the data doesn't exists
                y[i, j] = 0
    return y



def multitask_lasso(X, y, L, lr=0.1, alpha=0.01,beta = 0.00, alpha_t=0.01, max_iter=100,
                    min_gap=0.001, temporal_smooth=False, analytic_expression=False,sw = 0):
    
    loss_train_record = []
    weights = np.zeros((X.shape[1], y.shape[1]))
    # Set necessary factors
    H = np.zeros((y.shape[1], y.shape[1] - 1))
    for i in range(y.shape[1]):
        for j in range(y.shape[1] - 1):
            if i == j:
                H[i, j] = 1
            elif i == j + 1:
                H[i, j] = -1
    # Set H matrix for TSP
    y = incomplete_data(y)
    # Dealing the Nan data of output

    for iter in range(max_iter):
        if (temporal_smooth == True and analytic_expression == False):
            weights = weights - lr * (np.dot(np.dot(X.T, X), weights) - np.dot(X.T, y) + 2 * alpha * weights + alpha_t * np.dot(np.dot(weights, H), H.T))
            loss_train = np.linalg.norm((np.dot(X, weights) - y) ** 2) + alpha * np.linalg.norm(
                weights ** 2) + alpha_t * np.linalg.norm(np.dot(weights, H) ** 2)

        elif (temporal_smooth == True and analytic_expression == True):
            m1 = np.dot(X.T, X) + alpha * np.identity(X.shape[1])
            m2 = alpha_t * np.dot(H, H.T)
            derta1, Q1 = np.linalg.eig(m1)
            derta2, Q2 = np.linalg.eig(m2)
            D = np.dot(np.dot(np.dot(Q1.T, X.T), y), Q2)
            W = np.zeros((derta1.shape[0], derta2.shape[0]))
            for i in range(derta1.shape[0]):
                for j in range(derta2.shape[0]):
                    W[i][j] = D[i][j] / (derta1[i] + derta2[j])
            weights = np.dot(np.dot(Q1, W), Q2.T)
            loss_train = np.linalg.norm((np.dot(X, weights) - y) ** 2) + alpha * np.linalg.norm(
                weights ** 2) + alpha_t * np.linalg.norm(np.dot(weights, H) ** 2)
          
        elif (temporal_smooth == False and analytic_expression == False):
  
           
            #∂J(W)/∂W = X.T * X * W -X.T * y + 2 * alpha * W +2L(W.T)(X.T)X 
            weights = weights - lr * (
                    np.dot(np.dot(X.T, X), weights) - np.dot(X.T, y) + 2 * alpha * weights +  2 * beta * np.dot(np.dot(np.dot(L,weights.T),X.T),X).T)
            # loss_train = np.linalg.norm((np.dot(X, weights) - y) ** 2) + alpha * np.abs(weights).sum()+beta * np.dot(np.dot(np.dot(np.dot(X,weights),L),weights.T),X.T).T.trace()
            loss_train = np.linalg.norm((np.dot(X, weights) - y) ** 2) + alpha * np.abs(weights).sum()+beta * np.dot(np.dot(np.dot(np.dot(X,weights),L),weights.T),X.T).T.trace()
          

        else:
            ae1 = np.dot(X.T, X) + alpha * np.identity(X.shape[1])
            ae2 = np.dot(X.T, y)
            weights = np.dot(np.linalg.inv(ae1), ae2)
            loss_train = np.linalg.norm((np.dot(X, weights) - y) ** 2) + alpha * np.abs(
                weights).sum()
            # print(loss_train)
            # W = (X.T * X + θ1 * I)^(-1) * X.T * Y
        if analytic_expression == True:
            loss_train_record.append(loss_train)
            break
        else:
            loss_train_record.append(loss_train)
        if sw:
      
            if iter != 0:
                if (loss_train_record[iter - 1] - loss_train_record[iter]) < min_gap:
                    break
    # Train the Multitask Lasso Regression by gradient descent
    return weights,loss_train_record


def mutitaskLasso(X, y, L,alpha,beta =0.01,tol = 1e-24,sw = 0):
   
    weights,trl= multitask_lasso(X, y, L, lr=0.001, alpha=alpha, beta =beta, alpha_t=0.001, max_iter=1500,
                                                         min_gap=tol, temporal_smooth=False,
                                                         # min_gap=0.001, temporal_smooth=False,
                                                         analytic_expression=False, sw = sw)
 

    prediction = np.dot(X, weights)
  
    return weights, prediction