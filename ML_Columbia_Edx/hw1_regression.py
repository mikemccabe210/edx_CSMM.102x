# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 19:35:02 2017

@author: Mike
"""
from __future__ import print_function

import sys
import numpy as np

class ridge_regression(object):
    '''Ridge regression model
    
    Solves for w = (I*lambda + X.T * X)^-1 * X.T * y
    then used w for prediction
    
    Does not assume weights are normalized since last column seems to be intercept
    '''
    def __init__(self, lam, sigma2):
        self.lam = lam
        self.sigma2 = sigma2
        self.train_mu = None
        self.train_sigma = None
        self.cov = None
        
    def fit(self, X, y):
        '''Returns weights w corresponding to the ridge regression solution for
        X, y, and lambda. 
        
        Args:
            X - Data
            y - labels
            
        returns:
            w - weight matrix
        '''
#        self.train_mu = X.mean(axis = 0)
#        self.train_sigma = X.std(axis = 0)
#        data = (X - self.train_mu) / self.train_sigma
#        data[:, -1] = 1
        data = X
        self.data = data
        reg_mat = np.eye(data.shape[1]) * lam
        reg_mat[-1, -1] = 0
        self.reg_mat = reg_mat

#       
        self.weights = np.linalg.inv((reg_mat + data.T.dot(data))).dot(data.T).dot(y)
        self.cov = np.linalg.inv(reg_mat + 1/self.sigma2 * data.T.dot(data))
        
        return self.weights
    
        
    def predict(self, X):
        return self.weights.dot(X)
        
    def active_learn(self, X):
        '''Takes in data X and identifies the next 10 (1-indexed) locations of the next 
        data point that should be used. 
        '''
        
        r_vals = []
        used = set()
        for j in range(10):
            validate = []

            max_val = -float('inf')
            max_ind = -1
#            print(self.cov)
            for i, x in enumerate(X):
                update = x.T.dot(self.cov).dot(x)
                validate.append((i, update))
                if update > max_val and i not in used:
                    max_val = update
                    max_ind = i
                    x0 = x
#                    print(i, update, max_ind)
            r_vals.append(max_ind + 1)
#            print(self.cov)
#            print(np.outer(x0, x0))
#            self.cov = np.linalg.inv(np.linalg.inv(self.cov)
#                                        +  1/self.sigma2 * np.outer(x0, x0))
            used.add(max_ind)

            self.cov = np.linalg.inv(self.reg_mat + 1/self.sigma2 
                                     * (np.outer(x0, x0) + self.data.T.dot(self.data)))
#            print(sorted(validate, key = lambda x: -x[1])[:10])

        return r_vals
            
        
    
if __name__ == '__main__':
    #Args
    args = sys.argv
#    print(args)
    if len(args) != 6:
#    assert len(args) == 5, 'Not enough args'
        lam = 2
        sigma2 = 3
        x_train = np.genfromtxt('D:\\Mike\\Documents\\Coursera\\Columbia ML\\Data\\X_train.csv',
                                delimiter = ',')
        y_train = np.genfromtxt('D:\\Mike\\Documents\\Coursera\\Columbia ML\\Data\\y_train.csv',
                                delimiter = ',')
        x_test = np.genfromtxt('D:\\Mike\\Documents\\Coursera\\Columbia ML\\Data\\X_test.csv',
                               delimiter = ',')
    else:
        lam = float(args[1])
        sigma2 = float(args[2])
        x_train = np.genfromtxt(args[3], delimiter = ',')
        y_train = np.genfromtxt(args[4], delimiter = ',')
        x_test = np.genfromtxt(args[5], delimiter = ',')

    cls = ridge_regression(lam, sigma2)
    cls.fit(x_train, y_train)
#    print(cls.weights)
    np.savetxt('wRR_%s.csv' % (int(lam)), cls.weights, delimiter = ',')
#    print(cls.active_learn(x_test))
    np.savetxt('active_%s_%s.csv' % (int(lam), int(sigma2)), cls.active_learn(x_test), delimiter = ',')
    
    
    
    
    