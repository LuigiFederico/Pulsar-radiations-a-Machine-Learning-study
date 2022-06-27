
import numpy
import scipy

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



















#------------------------#
#  Kfold implementation  #
#------------------------#

def kfold_MVG_Full(k_subsets, K, prior=0.5, usePCA=False, mPCA=7):
    
    scores = []

    for i in range(K):
        DT_k, LT_k = k_subsets[i][0]  # Data and Label Train
        DE_k, LE_k = k_subsets[i][1]  # Data and Label Test
        if(usePCA):
            DT_PCA = dim_red.PCA(DT_k, mPCA)
            DE_PCA = dim_red.PCA(DE_k, mPCA)
            
        # TODO: Training model
        
        # TODO: Scores from the Test set
        # scores.append(...)
        
    # TODO: Merge the scores
    
    # TODO: Compute the minDCF
    
    # RETURN minDCF     
    False
    
def kfold_MVG_Diag(k_subsets, K, prior=0.5, usePCA=False, mPCA=7):
    
    scores = []

    for i in range(K):
        DT_k, LT_k = k_subsets[i][0]  # Data and Label Train
        DE_k, LE_k = k_subsets[i][1]  # Data and Label Test
        if(usePCA):
            DT_PCA = dim_red.PCA(DT_k, mPCA)
            DE_PCA = dim_red.PCA(DE_k, mPCA)
            
        # TODO: Training model
        
        # TODO: Scores from the Test set
        # scores.append(...)
        
    # TODO: Merge the scores
    
    # TODO: Compute the minDCF
    
    # RETURN minDCF     
    False
  
def kfold_MVG_TiedFull(k_subsets, K, prior=0.5, usePCA=False, mPCA=7):
    
    scores = []

    for i in range(K):
        DT_k, LT_k = k_subsets[i][0]  # Data and Label Train
        DE_k, LE_k = k_subsets[i][1]  # Data and Label Test
        if(usePCA):
            DT_PCA = dim_red.PCA(DT_k, mPCA)
            DE_PCA = dim_red.PCA(DE_k, mPCA)
            
        # TODO: Training model
        
        # TODO: Scores from the Test set
        # scores.append(...)
        
    # TODO: Merge the scores
    
    # TODO: Compute the minDCF
    
    # RETURN minDCF     
    False
  
def kfold_MVG_TiedDiag(k_subsets, K, prior=0.5, usePCA=False, mPCA=7):
    
    scores = []

    for i in range(K):
        DT_k, LT_k = k_subsets[i][0]  # Data and Label Train
        DE_k, LE_k = k_subsets[i][1]  # Data and Label Test
        if(usePCA):
            DT_PCA = dim_red.PCA(DT_k, mPCA)
            DE_PCA = dim_red.PCA(DE_k, mPCA)
            
        # TODO: Training model
        
        # TODO: Scores from the Test set
        # scores.append(...)
        
    # TODO: Merge the scores
    
    # TODO: Compute the minDCF
    
    # RETURN minDCF     
    False








