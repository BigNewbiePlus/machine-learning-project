# -*- coding:utf-8 -*-
import sys
import numpy as np

x = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])

y = np.array([[0,1,1,0]]).T

syn0 = 2 * np.random.random((3,4)) - 1
syn1 = 2 * np.random.random((4,1)) - 1

for j in xrange(60000):
    if j%6000==0:
        sys.stdout.write('\r已训练%%%d'%(j*100.0/60000))
        sys.stdout.flush()
        
    l1 = 1/(1+np.exp(-(np.dot(x, syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1, syn1))))
    l2_delta = (y-l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T)*(l1*(1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += x.T.dot(l1_delta)
print()
print(l2)