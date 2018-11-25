# -*- coding: utf-8 -*-
"""
Created on Sun Apr 09 01:58:07 2017

@author: Mike
"""
import sys
import numpy as np
from collections import defaultdict
class PMF():
    ''' Implementation of Bayes classifier assuming: 
    
    y ~ Disrete(pi), x|y ~ Normal(mu_y, cov_y)
    
    '''
    def __init__(self, X):
#        print(X.shape)
#        print(X)
        self.valid_U = defaultdict(list)
        self.valid_V = defaultdict(list)
        self.M = self.build_M(X)
        N, P = self.M.shape
        self.U = np.random.random((N, 5))
        self.V = np.random.random((5, P))
        self.sigma2 = .1
        self.lamb = 2    
        self.loss = self.L()

        
    def build_M(self, X):
        M = np.zeros((int(X.max(axis = 0)[0]), int(X.max(axis = 0)[1])))
        for row in X:
            M[int(row[0] - 1), int(row[1] - 1)] = row[2]
            self.valid_U[int(row[0] - 1)].append(int(row[1] - 1))
            self.valid_V[int(row[1] - 1)].append(int(row[0] - 1))
        return M

    def L(self):
#        Mhat = self.U.dot(self.V)
        mhat_diff = 0
        for u, V in self.valid_U.items():
            for v in V:
                mhat_diff += (self.M[u,v] - self.U[u,:].dot(self.V[:,v])) ** 2
                
        self.loss = - 1/(2*self.sigma2) * mhat_diff - self.lamb/2 \
                        * np.sum(np.sqrt(np.sum(self.U **2, axis = 1))) \
                        - self.lamb/2 * np.sum(np.sqrt(np.sum(self.V **2, axis = 0)))

    def step(self):
        self.Uup()
        self.Vup()
        self.L()
        
    def Uup(self):
        for i, row in enumerate(self.M):
            temp_sum = np.eye(5) * self.sigma2 * self.lamb
            other = np.zeros(5)
            for j in self.valid_U[i]:
                temp_sum += np.outer(self.V[:, j], self.V[:, j])
                other += self.M[i, j] * self.V[:, j]
            self.U[i, :] = np.linalg.inv(temp_sum).dot(other)
                
            
    def Vup(self):
        for i, row in enumerate(self.M.T):
            temp_sum = np.eye(5) * self.sigma2 * self.lamb
            other = np.zeros(5)
            for j in self.valid_V[i]:
                temp_sum += np.outer(self.U[j, :], self.U[j,:])
                other += self.M.T[i, j] * self.U[j, :]
            self.V[:, i] = np.linalg.inv(temp_sum).dot(other)
            
            
    


if __name__ == '__main__':
    #Args
    args = sys.argv
#    print(args)
    if len(args) != 2:
#    assert len(args) == 5, 'Not enough args'

        X = np.genfromtxt('D:\\Mike\\Documents\\Coursera\\Columbia ML\\Data\\ratings.csv',
                                delimiter = ',')

    else:
        X = np.genfromtxt(args[1], delimiter = ',')

    pmf = PMF(X)
    key_iters = [10, 25, 50]
    loss = []
    for i in np.arange(1, 51):
        pmf.step()
        loss.append(pmf.loss)
#        print( pmf.loss)
        if i in key_iters:
            np.savetxt('U-%s.csv' % i, pmf.U, delimiter = ',') 
            np.savetxt('V-%s.csv' % i, pmf.V.T, delimiter = ',')        
    np.savetxt('objective.csv', loss, delimiter = ',')        
 