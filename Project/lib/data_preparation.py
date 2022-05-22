# Last edit: 15/05/2022 - Alex
import numpy
import sys

# -------------------- #
#     SPLIT DATASET    #
# -------------------- #
def split_db_2to1(D, L, seed=0, ratio=2.0/3.0):
    """
    Randomly splits the dataset and the labelset in two parts:
        - (D.shape[1]*ratio) random samples will be the traning data
        - (D.shape[1]*(1-ratio)) random samples will be the evaluation data
        
    Parameters
    ----------
    D : Dataset to split    
    
    L : Labels of the input dataset
    
    seed : optional (default = 0). Seed the legacy random number generator.
        
    ratio: number of traning data over the total number of data (default=2/3)
        
    Returns
    -------
    Two tuple: (TraningData, TraningLabels), (TestData, TestLabels)
    """
    
    nTrain = int(D.shape[1] * ratio)
    numpy.random.seed(seed) # Seed the legacy random number generator.
    idx = numpy.random.permutation(D.shape[1]) # Crea un array con elementi da 0 a 149 ordinati randomicamente
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]    # Traning data
    DTE = D[:, idxTest]     # Test data
    LTR = L[idxTrain]       # Training labels
    LTE = L[idxTest]        # Test labels
    
    return (DTR, LTR), (DTE, LTE)

def k_fold(D, L, seed=0, K=5):
    """
    K-fold algorithm

    Parameters
    ----------
    D : Dataset to split    
    
    L : Labels of the input dataset
    
    seed : optional (default = 0). Seed the legacy random number generator.
        
    K : number of output subset, optional (default=5)

    Returns
    -------
    subsets : array of tuple ((DTrain_i, LTrain_i), (DTest_i, LTest_i))
    """ 
    
    folds   = [] # array of tuple (D_i, L_i)
    subsets = [] # array of tuple ((DTrain_i, LTrain_i), (DTest_i, LTest_i) )
    
    # Split the dataset into k folds
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    l = 0
    r0 = int(D.shape[1] / K)
    r = r0
    for i in range(K):
        if i == K-1:
            r = D.shape[1]
        subset_i = idx[l:r]
        D_i = D[:, subset_i]
        L_i = L[subset_i]
        folds.append((D_i, L_i))
        l = r
        r = r + r0

    # Generate the k subsets
    for i in range(K):
        test_i = folds[i]
        first  = True
        for j in range(K):
            if j!=i:
                if first:
                    Dtrain_i = folds[j][0]
                    Ltrain_i = folds[j][1]
                    first = False
                else:
                    Dtrain_i = numpy.hstack([Dtrain_i, folds[j][0]])
                    Ltrain_i = numpy.hstack([Ltrain_i, folds[j][1]])
        subsets.append( ((Dtrain_i, Ltrain_i), test_i) )

    return subsets

def k_fold_LOO(D, L, seed=0):
    """
    K-fold Leave One Out Algorithm. It is K-fold with K=D.shape[1]

    Parameters
    ----------
    D : Dataset to split    
    
    L : Labels of the input dataset
    
    seed : optional (default = 0). Seed the legacy random number generator.

    Returns
    -------
    subsets : array of tuple ((DTrain_i, LTrain_i), (DTest_i, LTest_i))
    """ 
    
    folds   = [] # array of tuple (D_i, L_i)
    subsets = [] # array of tuple ((DTrain_i, LTrain_i), (DTest_i, LTest_i) )
    K = D.shape[1]
    
    # Split the dataset into k folds
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    l = 0
    r0 = int(D.shape[1] / K)
    r = r0
    for i in range(K):
        if i == K-1:
            r = D.shape[1]
        subset_i = idx[l:r]
        D_i = D[:, subset_i]
        L_i = L[subset_i]
        folds.append((D_i, L_i))
        l = r
        r = r + r0

    # Generate the k subsets
    for i in range(K):
        test_i = folds[i]
        first  = True
        for j in range(K):
            if j!=i:
                if first:
                    Dtrain_i = folds[j][0]
                    Ltrain_i = folds[j][1]
                    first = False
                else:
                    Dtrain_i = numpy.hstack([Dtrain_i, folds[j][0]])
                    Ltrain_i = numpy.hstack([Ltrain_i, folds[j][1]])
        subsets.append( ((Dtrain_i, Ltrain_i), test_i) )

    return subsets


# ---------------------- #
#     CLASS UNBLANACE    #
# ---------------------- #

def over_sampling(D, L, numOfCopies=2):
    '''
    Over Sapmling Algorithm. Select the class that has the minor number of samples and makes a specific number of copies.
    
    Parameters
    ----------
    D : Dataset to over_sampling    
    L : Labels of the input dataset
    numOfCopies : Number of copies requested.The default is 2.

    Returns
    -------
    D_final : Oversampled Dataset on Minor Class
    L_final : Labels of the output dataset
    '''
    
    unique, counts = numpy.unique(L, return_counts=True)
    samplesDistribution = dict(zip(unique, counts))
    minorClass = min(samplesDistribution, key=samplesDistribution.get)
    # With that we know the class that we need to duplicate
    
    # Now we split D and L into 2 parts,
    # D_minor is the D that we want to duplicate
    D_minor = D[:, L==minorClass]
    L_minor = L[L==minorClass]
    # D_major is the D that we don't want to duplicate
    D_major = D[:, L!=minorClass]
    L_major = L[L!=minorClass]
    
    # We replicate the D_minor
    D_replicated = numpy.repeat(D_minor, numOfCopies, 1)
    L_replicated = numpy.repear(L_minor, numOfCopies)
    
    # Now we concatenate the D_replicated and L_replicated with D_major and L_major
    D_united = numpy.concatenate((D_replicated, D_major))
    L_united = numpy.concatenate((L_replicated, L_major))
    
    # Define a permutation 
    if( D_united.shape[1]==L_united.shape[0]):
        P = numpy.random.permutation(D_united.shape[1])
    else:
        print("Error with shape of Matrices while Over Sampling")
        sys.exit()
    
    # Applying permutation to D_final and L_final
    D_final = D_united[:,P]
    L_final = L_united[P]
    
    return D_final, L_final






