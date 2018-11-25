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


class k_means(object):
    ''' Implementation of Bayes classifier assuming: 
    
    y ~ Disrete(pi), x|y ~ Normal(mu_y, cov_y)
    
    '''
    def __init__(self, X):
        self.centroids = X[np.random.choice(X.shape[0], 5, replace = False),:]
        self.assignments = np.zeros(X.shape[0])
        self.X = X      
    
    def step(self):
        self.assign()
        self.update()
        
    def assign(self):
        for i, row in enumerate(self.X):
            min_dist = np.inf
            min_assign = None
            for j, cent in enumerate(self.centroids):
                dist = np.linalg.norm(row - cent)
                if dist < min_dist:
                    min_dist = dist
                    min_assign = j
            self.assignments[i] = min_assign
            
    def update(self):
        for i, centroid in enumerate(self.centroids):
            self.centroids[i, :] = np.mean(self.X[self.assignments == i, :], axis = 0)
            
            
class GMM(object):
    ''' Implementation of Bayes classifier assuming: 
    
    y ~ Disrete(pi), x|y ~ Normal(mu_y, cov_y)
    
    '''
    def __init__(self, X):
        self.mus = X[np.random.choice(X.shape[0], 5, replace = False),:]
        self.assignments = np.zeros(X.shape[0])
        self.X = X     
        self.phi = np.zeros((5, X.shape[0]))
        self.pi = np.ones(5)/5
        self.sigma = np.array([np.eye(X.shape[1]).tolist()] * 5)



    
    def step(self):
        self.E()
        self.M()
        
    def E(self):
        for i, row in enumerate(self.X):
            for j, prior in enumerate(self.pi):
#                print(norm.pdf(row, self.mus[j,:], 
#                                    self.sigma[j, :, :]))
                self.phi[j,i] = self.pi[j] * norm.pdf(row, self.mus[j,:], 
                                    self.sigma[j, :, :])
#            print()
#        print(self.phi)
        self.phi = np.divide(self.phi, self.phi.sum(axis = 0))
#        print(self.phi)
                
            
    def M(self):
        nk = self.phi.sum(axis = 1)

#        print(nk.shape)
        N, M = self.X.shape
        self.sigma = np.zeros((5, M, M))
        for i, centroid in enumerate(self.pi):
            self.pi[i] = nk[i]/N
#            print(self.pi[i])
#            print(self.mus[i, :].shape, self.phi[i,:] * self.X.T)

            self.mus[i, :] = 1/nk[i] * (self.phi[i,:] * self.X.T).sum(axis = 1)
#            print(self.mus[i, :])
            for j, row in enumerate(self.X):
#                print(1/N * (self.phi[i, j] * \
#                    np.outer(row - self.mus[i, :],row - self.mus[i, :])))
                self.sigma[i, :, :] += 1/nk[i] * (self.phi[i, j] * \
                    np.outer(row - self.mus[i, :],row - self.mus[i, :]))
#        print(self.sigma)

            
            
    


if __name__ == '__main__':
    #Args
    args = sys.argv
#    print(args)
    if len(args) != 2:
#    assert len(args) == 5, 'Not enough args'

        X = np.genfromtxt('D:\\Mike\\Documents\\Coursera\\Columbia ML\\Data\\X.csv',
                                delimiter = ',')

    else:
        X = np.genfromtxt(args[1], delimiter = ',')

    km = k_means(X)
    gmm = GMM(X)
    for i in np.arange(1, 11):
        km.step()
        gmm.step()
        np.savetxt('centroids-%s.csv' % i, km.centroids, delimiter = ',')  
        np.savetxt('pi-%s.csv' % i, gmm.pi, delimiter = ',') 
        np.savetxt('mu-%s.csv' % i, gmm.mus, delimiter = ',')        
        for j, cluster in enumerate(gmm.sigma):
            np.savetxt('Sigma-%s-%s.csv' % (j+1, i), cluster, delimiter = ',')        