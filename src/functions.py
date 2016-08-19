"""
Helper functions class
"""

import scipy.sparse.linalg as lin
import scipy.sparse as sparse
import scipy.interpolate as interpolate
from matplotlib.pyplot import *

def conc_sv(C):
    SVS = []
    for i, c in zip(range(len(C)),C):
        SVS.append(np.linalg.svd(c, full_matrices=True)[1])
    return SVS

def plot_interpolate_1d(patterns, Y_recalls, overSFac = 20, plotrange = 30):
    
    Driver_int = np.zeros([(plotrange-1)*overSFac])
    Recall_int = np.zeros([(len(Y_recalls[0])-1)*overSFac])
    NRMSEsAlign = np.zeros([len(patterns)])
        
    for i,p in zip(range(len(patterns)),patterns):
        
        p = np.vectorize(p)        
        
        Driver = p(np.linspace(0,plotrange-1,plotrange))
        Recall = np.squeeze(Y_recalls[i])      
            
        fD = interpolate.interp1d(range(plotrange),Driver,kind='cubic')
        fR = interpolate.interp1d(range(len(Recall)),Recall,kind='cubic') 

        Driver_int = fD(np.linspace(0,(len(Driver_int)-1.)/overSFac,len(Driver_int)))
        Recall_int = fR(np.linspace(0,(len(Recall_int)-1.)/overSFac,len(Recall_int)))
            
        L = len(Recall_int)
        M = len(Driver_int)    
    
        phasematches = np.zeros([L-M])  
        
        for s in range(L-M):
            phasematches[s] = np.linalg.norm(Driver_int-Recall_int[s:s+M])
            
        pos = np.argmin(phasematches)  
        Recall_PL = Recall_int[np.linspace(pos,pos+overSFac*(plotrange-1),plotrange).astype(int)]  
        Driver_PL = Driver_int[np.linspace(0,overSFac*(plotrange-1)-1,plotrange).astype(int)]
 
        NRMSEsAlign[i] = NRMSE(np.reshape(Recall_PL,(1,len(Recall_PL))),np.reshape(Driver_PL ,(1,len(Driver_PL ))))        
            
        subplot(len(patterns),1,i+1)
        xspace = np.linspace(0,plotrange-1,plotrange)
        plot(xspace,Driver_PL)
        plot(xspace,Recall_PL)
      
    print(NRMSEsAlign)

def checkRecall(patterns, Y_recalls, evalRange = 50):
    
    meanError = np.zeros([len(patterns)])
    
    for i,p in enumerate(patterns):
        
        target = np.argmax(p[0:evalRange,:], axis = 1)
        recall = np.argmax(Y_recalls[i], axis = 1)
        
        L = len(recall)
        M = len(target)
        phasematches = np.zeros([L-M])
        for s in range(L-M):
            phasematches[s] = np.linalg.norm(target-recall[s:s+M])
        
        pos = np.argmin(phasematches)
        recall_pm = recall[pos:pos+evalRange]  
        target_pm = target[0:evalRange]
        
        meanError[i] = np.mean(recall_pm != target_pm)
    
    return meanError
    
    
def IntWeights(N, M,connectivity):    
    
    succ = False
    while not succ:    
        try:
            W_raw =  sparse.rand(N, M ,format='lil', density=connectivity )
            rows,cols = W_raw.nonzero()
            for row,col in zip(rows,cols):
                W_raw[row,col] = np.random.randn()
            specRad,eigenvecs = np.abs(lin.eigs(W_raw,1))
            W_raw =  np.squeeze(np.asarray(W_raw/specRad))
            succ = True
            return W_raw
        except:
            pass
  
def NRMSE(output, target):
    combinedVar = 0.5 * (np.var(target,1) + np.var(output,1))
    error = output-target
    
    return np.sqrt(np.mean(error**2,axis=1)/combinedVar)
       
def RidgeWout(TrainArgs, TrainOuts, TychonovAlpha):
    n_n = len(TrainArgs[:,1])
    return np.transpose(np.dot(np.linalg.inv(np.dot(TrainArgs,TrainArgs.T)+TychonovAlpha*np.eye(n_n)),np.dot(TrainArgs,TrainOuts.T)))

def RidgeWload(TrainOldArgs, W_targets, TychonovAlpha):
    n_n = len(TrainOldArgs[:,1])
    return np.transpose(np.dot(np.linalg.pinv(np.dot(TrainOldArgs,TrainOldArgs.T)+TychonovAlpha*np.eye(n_n)),np.dot(TrainOldArgs,W_targets.T)))

def AND(C,B):
    dim = len(C)
    tol = 1e-12
    
    Uc,Sc,Vc = np.linalg.svd(C, full_matrices=True)      
    Ub,Sb,Vb = np.linalg.svd(B, full_matrices=True) 

    if np.diag(Sc[Sc > tol]).size:
        numRankC = np.linalg.matrix_rank(np.diag(Sc[Sc > tol]))
    else:
        numRankC = 0
    if np.diag(Sb[Sb > tol]).size:    
        numRankB = np.linalg.matrix_rank(np.diag(Sb[Sb > tol]))  
    else:
        numRankB = 0 
        
    Uc0 = Uc[:, numRankC:]
    Ub0 = Uc[:, numRankB:]
    
    W,Sig,Wt = np.linalg.svd(np.dot(Uc0,Uc0.T)+np.dot(Ub0,Ub0.T), full_matrices=True)      
    if np.diag(Sig[Sig > tol]).size: 
        numRankSig = np.linalg.matrix_rank(np.diag(Sig[Sig > tol]))
    else: 
        numRankSig = 0
    Wgk = W[:, numRankSig:]
    arg = np.linalg.pinv(C,tol)+np.linalg.pinv(B,tol)-np.eye(dim)
    
    return np.dot(np.dot(Wgk,np.linalg.inv(np.dot(Wgk.T,np.dot(arg,Wgk)))),Wgk.T)
    
def NOT(C):
    dim = len(C)
    return np.eye(dim) - C
    
def OR(C,B):
    return NOT(AND(NOT(C), NOT(B)))
    
def  sPHI(c,gamma):
    d = np.zeros([len(c)])
    for i in range(len(c)):
        if (gamma == 0):
            if (c[i] < 1):  d[i] = 0
            if (c[i] == 1): d[i] = 1
        else:
            d[i] = c[i]/(c[i]+(gamma**-2)*(1-c[i]))
    return d
        
def sNOT(c):
    return 1-c
    
def sAND(c,b):
    d = np.zeros([len(c)])
    for i in range(len(c)):
        if (c[i] == 0 and b[i] == 0): d[i] = 0
        else:                         d[i] = c[i]*b[i]/(c[i]+b[i]-c[i]*b[i])
    return d
    
def sOR(c,b):
    d = np.zeros([len(c)])
    for i in range(len(c)):
        if (c[i] == 1 and c[i] == b[i]): d[i] = 1
        else:                            d[i] = (c[i]+b[i]-2*c[i]*b[i])/(1-c[i]*b[i])   
    return d
    
def phi(C, gamma):
    return np.dot(C, np.linalg.inv(C+gamma**(-2)*(np.eye(len(C))-C)))