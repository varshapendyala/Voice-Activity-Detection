# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:13:56 2016

@author: varsha
"""

import numpy as np
#import python_speech_features as features
#import scipy.spatial.distance as dist
#import math
import scipy
import matplotlib.pyplot
#import scipy.io.wavfile as wav
import scipy.io
from sklearn import mixture
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D


f1=scipy.io.loadmat('knock_lowd.mat')
f1=f1['knock_lowd']
a=f1.shape[0]
b=f1.shape[1]
f1=f1.reshape((a*b,4))
f2=scipy.io.loadmat('key_lowd.mat')
f2=f2['key_lowd']
a=f2.shape[0]
b=f2.shape[1]
f2=f2.reshape((a*b,4))

lowd=np.concatenate((f1,f2),axis=0)

c1=lowd[lowd[:,3]==1]
c2=lowd[lowd[:,3]==2]

gmm1 = mixture.GMM(n_components=10,n_init=3,verbose=2)
gmm1.fit(c1[:,0:3])
#L1=gmm1.score_samples(c1)

gmm2=mixture.GMM(n_components=20,n_init=3,verbose=2)
gmm2.fit(c2[:,0:3])
#L2=gmm2.score_samples(c2)

#L10=L1[0]
#L11=L1[1]
#
#L20=L2[0]
#L21=L2[1]

means1=gmm1.means_
covars1=gmm1.covars_
weights1=gmm1.weights_

means2=gmm2.means_
covars2=gmm2.covars_
weights2=gmm2.weights_

############################################################################################
def likelihood_calc(data_point):
    #Class 1:
    P1=0
    P2=0
    for i in range(10):
        P1=P1+weights1[i]*multivariate_normal.pdf(data_point,means1[i,:], np.diag(covars1[i,:]))
    
    for i in range(20):
        P2=P2+weights2[i]*multivariate_normal.pdf(data_point,means2[i,:], np.diag(covars2[i,:]))
        
    return P1,P2
 
       
estimlabel=np.zeros(55250)

for i in range(55250):
    [p1,p2]=likelihood_calc(lowd[i,0:3])
    if (p1>p2):
        estimlabel[i]=1
    else:
        if(p2>p1):
            estimlabel[i]=2
#    
#fig=matplotlib.pyplot.figure(1)
#ax = fig.add_subplot(111, projection='3d')
#b=ax.scatter(c1[:,0],c1[:,1],c1[:,2],c='b',marker='o')
#g=ax.scatter(c2[:,0],c2[:,1],c2[:,2],c='g',marker='^')
#
#fig=matplotlib.pyplot.figure(2)
#ax = fig.add_subplot(111, projection='3d')
#b=ax.scatter(temp1[:,0],temp1[:,1],temp1[:,2],c='b',marker='o')
#g=ax.scatter(temp2[:,0],temp2[:,1],temp2[:,2],c='g',marker='^')
#
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')

