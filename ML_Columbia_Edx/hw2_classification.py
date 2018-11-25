# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 10:53:11 2017

@author: Mike
"""
from __future__ import print_function
from __future__ import division
from scipy.stats import multivariate_normal as norm

import sys
import numpy as np


class bayes_classifier(object):
    ''' Implementation of Bayes classifier assuming: 
    
    y ~ Disrete(pi), x|y ~ Normal(mu_y, cov_y)
    
    '''
    def __init__(self):
        self.pi = None
        self.covs = {}
        self.mus = {}
        
    
    def fit(self, X, y):
        k, pi = np.unique(y, return_counts = True)
        N, m = X.shape
        self.k = k.size
        self.pi = pi/N
        
        N, m = X.shape
        for i in range(self.k):
            X_sub = X[y == i]
            self.covs[i] = np.cov(X_sub.T, bias = True)
            self.mus[i] = np.mean(X_sub, axis = 0)
            
    def predict_proba(self, X):
        preds = np.zeros((X.shape[0], self.k))
        for j, x in enumerate(X):
#            max_l = 0
#            max_ret = None
            for i in range(self.k):
                preds[j, i] = self.pi[i] * norm.pdf(x, self.mus[i], self.covs[i])
#                
#                if like > max_l:
#                    max_l = like
#                    max_ret = i
#            preds[j] = max_ret
        return preds
            
    


if __name__ == '__main__':
    #Args
    args = sys.argv
#    print(args)
    if len(args) != 4:
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

        x_train = np.genfromtxt(args[1], delimiter = ',')
        y_train = np.genfromtxt(args[2], delimiter = ',')
        x_test = np.genfromtxt(args[3], delimiter = ',')

    cls = bayes_classifier()
    cls.fit(x_train, y_train)
#    print(cls.weights)
#    np.savetxt('wRR_%s.csv' % (int(lam)), cls.weights, delimiter = ',')
#    print(cls.active_learn(x_test))
    a = cls.predict_proba(x_test)
#    print(cls.predict_proba(x_test))
    np.savetxt('probs_test.csv', cls.predict_proba(x_test), delimiter = ',')
    
    