# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 20:24:18 2016

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

store_lowd=[]
#remaining_lowd=[]
winlen=0.032
winstep=0.016
newdata_mat=scipy.io.loadmat('filenames.mat')
newdata_mat=newdata_mat['Filenames']
################################################################################################
#(rate,sig) = wav.read('F:/Semester VII/speech_project/created_me/Training/Nstationary/SpeechTransient_Transient/keyboard-typing/fcjf0_sa2.wav')
#mfcc_feat=features.mfcc(sig,rate,winlen=0.032,winstep=0.016)
#basis_data=mfcc_feat
#
## Getting Covariances:
#inv_cov_mat=np.zeros((13,13,basis_data.shape[0]))
#cov_mat=np.zeros((13,13))
#buf=15
#for i in range(basis_data.shape[0]):
#    cov_mat[:,:]=np.cov((basis_data[max(0,i-buf):min(i+buf,basis_data.shape[0]-1),:].T))
#    inv_cov_mat[:,:,i]=np.linalg.inv(cov_mat[:,:])
#
#no_eig=3 
## Calculating K and M matrices:
#epsilon=40 
#K=np.zeros((basis_data.shape[0],basis_data.shape[0]))
#M=np.zeros((basis_data.shape[0],basis_data.shape[0]))
##temp1=np.zeros((basis_data.shape[0],basis_data.shape[0]))
#for i in range(basis_data.shape[0]):
#    for j in range(basis_data.shape[0]):
#        temp1=dist.mahalanobis(basis_data[i,:], basis_data[j,:], inv_cov_mat[:,:,i]+inv_cov_mat[:,:,j])
##        temp1[i,j]=dist.euclidean(basis_data[i,:], basis_data[j,:])                
#        temp1=(np.power(temp1,2))/2.0
#        K[i,j]=math.exp(-temp1/epsilon)
#D=np.diag(np.sum(K,axis=1))
#M=np.dot(np.linalg.inv(D),K)
#[H,R]=scipy.linalg.eigh(M,eigvals=(basis_data.shape[0]-no_eig,basis_data.shape[0]-1))
############################################################################################
basis_data=np.concatenate((data_knock_sa2,data_key_sa2),axis=0)
inv_cov_mat=np.concatenate((inv_cov_knock_sa2,inv_cov_key_sa2),axis=2)
R=np.concatenate((lowd_knock_sa2,lowd_key_sa2),axis=0)
no_eig=3
###########################################################################################
for file_num in range(50):
    (rate1,sig1) = wav.read('F:/Semester VII/speech_project/created_me/Training/Nstationary/SpeechTransient_Transient/keyboard-typing/'+str(newdata_mat[file_num,0][0]))
    mfcc_feat=features.mfcc(sig1,rate1,winlen=0.032,winstep=0.016)
    new_data=mfcc_feat

    # Getting Covariances:
    inv_cov_mat1=np.zeros((13,13,new_data.shape[0]))
    cov_mat=np.zeros((13,13))
    buf=15
    for i in range(new_data.shape[0]):
        cov_mat[:,:]=np.cov((new_data[max(0,i-buf):min(i+buf,new_data.shape[0]-1),:].T))
        inv_cov_mat1[:,:,i]=np.linalg.inv(cov_mat[:,:])
    N=15  
    # Calculating B matrix:
    epsilon=50 
    B=np.zeros((new_data.shape[0],basis_data.shape[0]))
    temp2=np.zeros((new_data.shape[0],basis_data.shape[0]))
    #temp3=np.zeros((new_data.shape[0],basis_data.shape[0]))
    for i in range(new_data.shape[0]):
        for j in range(basis_data.shape[0]):
            temp2[i,j]=dist.mahalanobis(new_data[i,:], basis_data[j,:], inv_cov_mat1[:,:,i]+inv_cov_mat[:,:,j])
            #        temp2[i,j]=dist.euclidean(basis_data[i,:], basis_data[j,:])                
            temp2[i,j]=(np.power(temp2[i,j],2))/2.0
            temp2[i,j]=math.exp(-temp2[i,j]/epsilon)
        ind = np.argpartition(temp2[i,:], -N)[-N:]
        B[i,ind]=temp2[i,ind]    
            
    Unew=np.zeros((basis_data.shape[0],no_eig))
    temp=np.diag(1/(np.sum(B,axis=1)))
    temp=np.dot(temp,B)
    Unew=np.dot(temp,R)
                
###################################################################################################
#    label_mat=scipy.io.loadmat('F:/Semester VII/speech_project/created_me/Training/Nstationary/SpeechTransient_Transient/keyboard-typing/keyboard_labels.mat')
#    label_mat=label_mat['keyboard']
    
    label_mat=scipy.io.loadmat('F:/Semester VII/speech_project/created_me/Training/Nstationary/SpeechTransient_Transient/knock-on-the-door/knock_labels.mat')
    label_mat=label_mat['knock']
    
#    label_mat=scipy.io.loadmat('F:/Semester VII/speech_project/created_me/Training/Nstationary/SpeechTransient_Transient/coughing/coughing_labels.mat')
#    label_mat=label_mat['coughing']
    
#    label_mat=scipy.io.loadmat('F:/Semester VII/speech_project/created_me/Training/Nstationary/SpeechTransient_Transient/baby9/baby9_labels.mat')
#    label_mat=label_mat['baby9']
    
    labels=label_mat[file_num,:]
    sample_labels=np.zeros(sig1.shape[0])
    sample_labels[0:labels[1]+1]=1
    sample_labels[labels[1]+1:labels[2]+1]=2
    
    real_label=np.zeros((new_data.shape[0],1))
    for i in range(new_data.shape[0]):
        temp=sample_labels[int(winstep*16000*i):int(winstep*16000*i+winlen*16000+1)] 
        if(np.array_equal(temp,np.ones(temp.shape[0]))):
            real_label[i,0]=1
        if(np.array_equal(temp,2*np.ones(temp.shape[0]))):
            real_label[i,0]=2
        if(np.any(temp==1) and np.any(temp==2)):
            real_label[i,0]=1
                                
    Unew=np.concatenate((Unew,real_label),axis=1)
    store_lowd.append(Unew)
#    remaining_lowd.append(Unew)
#####################################################################################################
    
#####################################################################################################
matplotlib.pyplot.figure(1)
b=matplotlib.pyplot.scatter(R[0:135,1],R[0:135,2],c='b')
g=matplotlib.pyplot.scatter(R[135:437,1],R[135:437,2],c='g')
b=matplotlib.pyplot.scatter(R[437:572,1],R[437:572,2],c='b')
g=matplotlib.pyplot.scatter(R[572:1105,1],R[572:1105,2],c='g')

matplotlib.pyplot.figure(1)
matplotlib.pyplot.scatter(Unew[0:183,1],Unew[0:183,2],c='y')
matplotlib.pyplot.scatter(Unew[183:668,1],Unew[183:668,2],c='r')
########################################################################
                                
