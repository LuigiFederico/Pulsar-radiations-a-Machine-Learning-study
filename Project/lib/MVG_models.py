import lib.model_evaluation as ev
import numpy
import scipy
import numpy.matlib 

def vrow(v):
    return v.reshape((1, v.size))

def vcol(v):
    return v.reshape((v.size, 1))

def logpdf_GAU_ND(X, mu, C):
    
    """
    Multivariate Gaussian log-density

    Parameters
    ----------
    X : Matrix with N column vectors xi. Those xi are the feature vectors with shape (M,1)
    
    mu : Mean of the multivariate gaussian. It is a numpy array with shape (M,1)
    
    C : Covariance matrix. It is a numpy array with shape (MxM).
        
    Returns
    -------
    Numpy array that contains the log-densities of the X's samples.
    
    References
    ----------
    ML05 - Probability and density estimation
    ML - Lab04

    """
    
    C_inverse = numpy.linalg.inv(C)        # Inverse of the covariance matrix
    _, C_logdet = numpy.linalg.slogdet(C)  # log-determinant log|C| di C
    M = mu.shape[0]
    Y = []

    a = -0.5 * M * numpy.log(2*numpy.pi)       # First constant element
    b = -0.5 * C_logdet                    # Secondo constant element
    for i in range(X.shape[1]):
        xi = vcol(X[:, i])
        c = -0.5 * numpy.dot((xi-mu).T, numpy.dot(C_inverse, (xi-mu)))[0][0]  
        logMVG = a+b+c
        Y.append(logMVG)
    
    return numpy.array(Y).ravel()        
    
def empirical_mean(X):
    return vcol(X.mean(1))

def empirical_cov(X):
    mu = empirical_mean(X)
    cov = numpy.dot((X-mu), (X-mu).T) / X.shape[1]
    return cov    

#-------------------#
#  MVG classifiers  #
#-------------------#



def MVG_Full(DT,LT,DE,LE,prior):
    
    mean0=empirical_mean(DT[:, LT == 0]) # Mean of the Gaussian Curve for Class 0
    mean1=empirical_mean(DT[:, LT == 1]) # Mean of the Gaussian Curve for Class 1
    
    sigma0=empirical_cov(DT[:, LT == 0]) # Sigma of the Gaussian Curve for Class 0
    sigma1=empirical_cov(DT[:, LT == 1]) # Sigma of the Gaussian Curve for Class 1
    
    LS0 = logpdf_GAU_ND(DE, mean0, sigma0) # Log Densities with parameters of Class 0
    LS1 = logpdf_GAU_ND(DE, mean1, sigma1) # Log Densities with parameters of Class 1
    
    SJoint = numpy.zeros((2, DE.shape[1]))
    
    SJoint[0, :] = numpy.exp(LS0) * (1-prior)        #Product Between Densities LS0 and PriorProb
    SJoint[1, :] = numpy.exp(LS1) * (prior)          #Product Between Densities LS1 and PriorProb
    
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    LabelPred=SPost.argmax(0)
    
    llr = LS1-LS0 # log-likelihood ratios
    
    return LabelPred, llr



def MVG_Diag(DT,LT,DE,LE,prior):
    
    mean0=empirical_mean(DT[:, LT == 0]) # Mean of the Gaussian Curve for Class 0
    mean1=empirical_mean(DT[:, LT == 1]) # Mean of the Gaussian Curve for Class 1
    
    sigma0=numpy.diag(numpy.diag(empirical_cov(DT[:, LT == 0]) )) # Diagonal Sigma of the Gaussian Curve for Class 0
    sigma1=numpy.diag(numpy.diag(empirical_cov(DT[:, LT == 1])))  # Diagonal Sigma of the Gaussian Curve for Class 1
  
    LS0 = logpdf_GAU_ND(DE, mean0, sigma0) # Log Densities with parameters of Class 0
    LS1 = logpdf_GAU_ND(DE, mean1, sigma1) # Log Densities with parameters of Class 1
    
    SJoint = numpy.zeros((2, DE.shape[1]))
    
    SJoint[0, :] = numpy.exp(LS0) * (1-prior)        #Product Between Densities LS0 and PriorProb
    SJoint[1, :] = numpy.exp(LS1) * (prior)          #Product Between Densities LS1 and PriorProb
   
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    LabelPred=SPost.argmax(0)
    
    llr = LS1-LS0 # log-likelihood ratios
   
    return LabelPred, llr



def MVG_TiedFull(DT,LT,DE,LE,prior):
    
    mean0=empirical_mean(DT[:, LT == 0]) # Mean of the Gaussian Curve for Class 0
    mean1=empirical_mean(DT[:, LT == 1]) # Mean of the Gaussian Curve for Class 1
    
    sigma0=empirical_cov(DT[:, LT == 0]) # Sigma of the Gaussian Curve for Class 0
    sigma1=empirical_cov(DT[:, LT == 1]) # Sigma of the Gaussian Curve for Class 1
    
    sigma = 1/(DT.shape[1])*(DT[:, LT == 0].shape[1]*sigma0+DT[:, LT == 1].shape[1]*sigma1) # Shared Sigma
    
    LS0 = logpdf_GAU_ND(DE, mean0, sigma) # Log Densities with parameters of Class 0
    LS1 = logpdf_GAU_ND(DE, mean1, sigma) # Log Densities with parameters of Class 1
    
    SJoint = numpy.zeros((2, DE.shape[1]))
    
    SJoint[0, :] = numpy.exp(LS0) * (1-prior)        #Product Between Densities LS0 and PriorProb
    SJoint[1, :] = numpy.exp(LS1) * (prior)          #Product Between Densities LS1 and PriorProb
    
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    LabelPred=SPost.argmax(0)
    
    llr = LS1-LS0 # log-likelihood ratios
    
    return LabelPred, llr



def MVG_TiedDiag(DT,LT,DE,LE,prior):
    
    mean0=empirical_mean(DT[:, LT == 0]) # Mean of the Gaussian Curve for Class 0
    mean1=empirical_mean(DT[:, LT == 1]) # Mean of the Gaussian Curve for Class 1
    
    sigma0=empirical_cov(DT[:, LT == 0]) # Sigma of the Gaussian Curve for Class 0
    sigma1=empirical_cov(DT[:, LT == 1]) # Sigma of the Gaussian Curve for Class 1
    
    sigma = numpy.diag(numpy.diag(1/(DT.shape[1])*(DT[:, LT == 0].shape[1]*sigma0+DT[:, LT == 1].shape[1]*sigma1))) # Shared Sigma
    
    LS0 = logpdf_GAU_ND(DE, mean0, sigma) # Log Densities with parameters of Class 0
    LS1 = logpdf_GAU_ND(DE, mean1, sigma) # Log Densities with parameters of Class 1
    
    SJoint = numpy.zeros((2, DE.shape[1]))
    
    SJoint[0, :] = numpy.exp(LS0) * (1-prior)        #Product Between Densities LS0 and PriorProb
    SJoint[1, :] = numpy.exp(LS1) * (prior)          #Product Between Densities LS1 and PriorProb
    
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    LabelPred=SPost.argmax(0)
    
    llr = LS1-LS0 # log-likelihood ratios
    
    return LabelPred, llr



#------------------------#
#  Kfold implementation  #
#------------------------#

def kfold_MVG_Full(k_subsets, K, prior=0.5):
    
    scores = []
    LE = []

    for i in range(K):
        DT_k, LT_k = k_subsets[i][0]  # Data and Label Train
        DE_k, LE_k = k_subsets[i][1]  # Data and Label Test
        
        PredLables, llRateos = MVG_Full(DT_k,LT_k,DE_k,LE_k,prior) # Classify the DE_k data
        scores.append(llRateos) 
        LE.append(LE_k)
    
    LE = numpy.concatenate(LE).ravel()    
    scores = numpy.concatenate(scores).ravel()
    minDCF = ev.computeMinDCF(LE, scores, prior, numpy.array([[0,1],[1,0]])) # Compute the minDCF
    
    return minDCF     
 

   
def kfold_MVG_Diag(k_subsets, K, prior=0.5):
    
    scores = []
    LE = []

    for i in range(K):
        DT_k, LT_k = k_subsets[i][0]  # Data and Label Train
        DE_k, LE_k = k_subsets[i][1]  # Data and Label Test
        
        PredLables, llRateos = MVG_Diag(DT_k,LT_k,DE_k,LE_k,prior) # Classify the DE_k data
        scores.append(llRateos) 
        LE.append(LE_k)
    
    LE = numpy.concatenate(LE).ravel()    
    scores = numpy.concatenate(scores).ravel()
    minDCF=ev.computeMinDCF(LE, scores, prior, numpy.array([[0,1],[1,0]])) # Compute the minDCF
    
    return minDCF    
 
    
 
def kfold_MVG_TiedFull(k_subsets, K, prior=0.5):
    
    scores = []
    LE = []

    for i in range(K):
        DT_k, LT_k = k_subsets[i][0]  # Data and Label Train
        DE_k, LE_k = k_subsets[i][1]  # Data and Label Test
        
        PredLables, llRateos = MVG_TiedFull(DT_k,LT_k,DE_k,LE_k,prior) # Classify the DE_k data
        scores.append(llRateos) 
        LE.append(LE_k)
    
    LE = numpy.concatenate(LE).ravel()    
    scores = numpy.concatenate(scores).ravel()
    minDCF=ev.computeMinDCF(LE, scores, prior, numpy.array([[0,1],[1,0]])) # Compute the minDCF
    
    return minDCF     

    

def kfold_MVG_TiedDiag(k_subsets, K, prior=0.5):
    
    scores = []
    LE = []

    for i in range(K):
        DT_k, LT_k = k_subsets[i][0]  # Data and Label Train
        DE_k, LE_k = k_subsets[i][1]  # Data and Label Test
        
        PredLables, llRateos = MVG_TiedDiag(DT_k,LT_k,DE_k,LE_k,prior) # Classify the DE_k data
        scores.append(llRateos) 
        LE.append(LE_k)
    
    LE = numpy.concatenate(LE).ravel()    
    scores = numpy.concatenate(scores).ravel()
    minDCF = ev.computeMinDCF(LE, scores, prior, numpy.array([[0,1],[1,0]])) # Compute the minDCF
    
    return minDCF    








