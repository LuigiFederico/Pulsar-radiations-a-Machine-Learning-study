# Last edit: 09/05/2022 - Luigi

import numpy
import scipy.linalg


def vcol(v):
    return v.reshape((v.size, 1))

def covariance_matrix(D):
    """
    Computes the covariance matrix of a no-centered dataset

    Parameters
    ----------
    D : Type: numpy.ndarray
        Description: Dataset (not centered)

    Returns
    -------
    C : Type: numpy.ndarray
        Description: Covariance Matrix

    """
    mu = vcol(D.mean(1))                       # Dataset mean
    DC = D - mu                                # Matrix of centered data
    C = numpy.dot(DC, DC.T) / DC.shape[1]      # Covariance matrix
    return C


 
def PCA(D, m):
    """
    Principal Component Analysis

    Parameters
    ----------
    D : Type: numpy.ndarray
        Description: Input dataset  
    m : Type: int
        Description: Number of leading eigenvectors

    Returns
    -------
    DP : Type: numpy.ndarray
         Description: Reduced matrix
         
    References
    ----------
    ML04 - Dimensionality Reduction
    ML - Lab03

    """    
    
    C = covariance_matrix(D)        # Covariance matrix
    U, s, Vh = numpy.linalg.svd(C)  # Singular Value Decomposition
    P = U[:, 0:m]                   # U = eigenvectors, s = eigenvalues (sorted)
    DP = numpy.dot(P.T, D)
  
    return DP


def LDA(D, L, m, n=2):
    """
    Linear Disccriminant Analysis

    Parameters
    ----------
    D : Type: numpy.ndarray
        Description: Input dataset
    L : Type: numpy.array
        Description: Labels
    m : Type: int
        Description: 
    n : Type: int
        Description: Number of classes

    Returns
    -------
    W : Type: numpy.ndarray
        Description: LDA directions
   
    References
    ----------
    ML04 - Dimensionality Reduction
    ML - Lab03    
    """
    
    Sb = 0                  # Between-class covariance matrix
    Sw = 0                  # Within-class covariance matrix
    mu = vcol(D.mean(1))    # Dataset mean
    
    for i in range(n):
        Di = D[:, L==i]
        nc = Di.shape[1]
        mui = vcol(Di.mean(1))
        Sb += ( nc * (mui-mu) * (mui-mu).T )
        Sw += ( covariance_matrix(Di) * nc )
    Sb = Sb / D.shape[1]
    Sw = Sw / D.shape[1]
    
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    
    return W
    

