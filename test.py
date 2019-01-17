# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 23:15:36 2016

@author: varsha
"""

import numpy as np
import python_speech_features as features
import scipy.spatial.distance as dist
import math
import scipy
import matplotlib.pyplot
import scipy.io.wavfile as wav
import scipy.io
from scipy.stats import multivariate_normal
import time
##############################################################################################

def likelihood_calc(data_point):
    #Class 1:
    P1=0
    P2=0
    for i in range(10):
        P1=P1+weights1[0,i]*multivariate_normal.pdf(data_point,means1[i,:], np.diag(covars1[i,:]))
    
    for i in range(20):
        P2=P2+weights2[0,i]*multivariate_normal.pdf(data_point,means2[i,:], np.diag(covars2[i,:]))
        
    return P1,P2
    
#################################################################################################    
keyboard_knock_basis=scipy.io.loadmat('F:/Semester VII/speech_project/created_me/Training/Nstationary/SpeechTransient_Transient/keyboard_knock_basis.mat')
data_knock_sa2=keyboard_knock_basis['data_knock_sa2']
data_key_sa2=keyboard_knock_basis['data_key_sa2']
lowd_knock_sa2=keyboard_knock_basis['lowd_knock_sa2']
lowd_key_sa2=keyboard_knock_basis['lowd_key_sa2']
inv_cov_knock_sa2=keyboard_knock_basis['inv_cov_knock_sa2']
inv_cov_key_sa2=keyboard_knock_basis['inv_cov_key_sa2']

gmm_param=scipy.io.loadmat('F:/Semester VII/speech_project/created_me/Training/Nstationary/SpeechTransient_Transient/gmm_param2.mat')
weights1=gmm_param['weights1']
weights2=gmm_param['weights2']
means1=gmm_param['means1']
means2=gmm_param['means2']
covars1=gmm_param['covars1']
covars2=gmm_param['covars2']

#################################################################################################
# Basis_data
basis_data=np.concatenate((data_knock_sa2,data_key_sa2),axis=0)
inv_cov_mat=np.concatenate((inv_cov_knock_sa2,inv_cov_key_sa2),axis=2)
R=np.concatenate((lowd_knock_sa2,lowd_key_sa2),axis=0)
no_eig=3
#################################################################################################

#Consider a test file:
#(rate,sig)=wav.read('C:/Users/Varsha/Documents/MATLAB/si999.wav')
#(rate,sig) = wav.read('F:/Semester VII/speech_project/created_me/Training/SpeechTransient_Transient/keyboard-typing/fcjf0_si1027.wav')
(rate,sig) = wav.read('F:/Semester VII/speech_project/created_me/Training/SpeechTransient_Transient_Speech/keyboard-typing/fdml0_sx69.wav')
mfcc_feat=features.mfcc(sig,rate,winlen=0.032,winstep=0.016)

buffer_size=15
prev_mfccs=mfcc_feat[0:buffer_size,:]
store_lowd=np.array([]).reshape(0,no_eig+1)
inv_cov_mat1=np.zeros((13,13))
cov_mat=np.zeros((13,13))
Unew=np.zeros((1,4))
delta=1.5
start = time.time() 
for itr in range(mfcc_feat.shape[0]):    
    
    if(min(mfcc_feat.shape[0],itr+buffer_size)==mfcc_feat.shape[0]):
        break
    
    affine_data=mfcc_feat[itr+buffer_size : min(itr+1+buffer_size,mfcc_feat.shape[0]),:]
    ########################################################################################
    # Getting Covariance matrix:    
    cov_mat[:,:]=np.cov((prev_mfccs.T))
    inv_cov_mat1[:,:]=np.linalg.inv(cov_mat[:,:])
    prev_mfccs=np.concatenate((prev_mfccs[1:buffer_size,:],affine_data[0:1,:]),axis=0)
    
    ########################################################################################
   
    # Calculating B matrix:
    N=5  
    epsilon=50 
    B=np.zeros(basis_data.shape[0])
    temp2=np.zeros(basis_data.shape[0])
    for i in range(basis_data.shape[0]):
        temp2[i]=np.power(dist.mahalanobis(affine_data,basis_data[i,:],inv_cov_mat1+inv_cov_mat[:,:,i]),2)/2.0
        temp2[i]=math.exp(-temp2[i]/epsilon)
    ind=np.argpartition(temp2, -N)[-N:]
    B[ind]=temp2[ind]
    temp=1.0/np.sum(B)*B
    temp=np.dot(temp,R)
    
    #######################################################################################

    #Estimating labels:
    [p1,p2]=likelihood_calc(temp)
    if (delta*p1>p2):
        label=1
    else:
        if(p2>delta*p1):
            label=2
            
    #Storing low dimensional representation and their labels:    
    Unew[0,0:3]=temp
    Unew[0,3] =label  
    store_lowd=np.concatenate((store_lowd,Unew),axis=0)
print 'It took', time.time()-start, 'seconds.'

#############################################################################################

matplotlib.pyplot.figure(1)
matplotlib.pyplot.scatter(store_lowd[0:161,1],store_lowd[0:161,2],c='y')
matplotlib.pyplot.scatter(store_lowd[161:161+508,1],store_lowd[161:161+508,2],c='r')
matplotlib.pyplot.scatter(store_lowd[161+508:828,1],store_lowd[161+508:828,2],c='y')

matplotlib.pyplot.figure(1)
matplotlib.pyplot.scatter(R[0:135,1],R[0:135,2],c='y')
matplotlib.pyplot.scatter(R[437:437+135,1],R[437:437+135,2],c='y')
matplotlib.pyplot.scatter(R[135:437,1],R[135:437,2],c='r')
matplotlib.pyplot.scatter(R[437+135:135+437+553,1],R[437+135:135+437+533,2],c='r')

#matplotlib.pyplot.scatter(store_lowd[:,1],store_lowd[:,2],c='b')
#############################################################################################   
