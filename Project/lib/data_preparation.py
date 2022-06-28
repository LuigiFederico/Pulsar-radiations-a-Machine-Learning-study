import numpy
import sys
from scipy.stats import norm
from dim_reduction import PCA

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
    numpy.random.seed(seed)  # Seed the legacy random number generator.
    # Crea un array con elementi da 0 a 149 ordinati randomicamente
    idx = numpy.random.permutation(D.shape[1])
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

    folds = []  # array of tuple (D_i, L_i)
    subsets = []  # array of tuple ((DTrain_i, LTrain_i), (DTest_i, LTest_i) )

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
        first = True
        for j in range(K):
            if j != i:
                if first:
                    Dtrain_i = folds[j][0]
                    Ltrain_i = folds[j][1]
                    first = False
                else:
                    Dtrain_i = numpy.hstack([Dtrain_i, folds[j][0]])
                    Ltrain_i = numpy.hstack([Ltrain_i, folds[j][1]])
        subsets.append(((Dtrain_i, Ltrain_i), test_i))

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

    folds = []  # array of tuple (D_i, L_i)
    subsets = []  # array of tuple ((DTrain_i, LTrain_i), (DTest_i, LTest_i) )
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
        first = True
        for j in range(K):
            if j != i:
                if first:
                    Dtrain_i = folds[j][0]
                    Ltrain_i = folds[j][1]
                    first = False
                else:
                    Dtrain_i = numpy.hstack([Dtrain_i, folds[j][0]])
                    Ltrain_i = numpy.hstack([Ltrain_i, folds[j][1]])
        subsets.append(((Dtrain_i, Ltrain_i), test_i))

    return subsets


def gaussianization (D_Train, D_Evaluation=0):
    
    # If D_Evaluation is equal to 0 we are using the function to gaussianize Training Data
    if(D_Evaluation != 0):
        
        # Create a temporary array to takes the stack values
        Stack = numpy.zeros([D_Evaluation.shape[0], D_Evaluation.shape[1]])
        
        # Iterate over each samples and calculate tha stack
        for row_count,(i,row_samples) in D_Train, enumerate(D_Evaluation):
            for (j,x) in enumerate(row_samples):
                Stack[i,j] = ((row_count<x).sum() + 1 )/(D_Train.shape[1]+2)
    
    else:
        
        # Create a temporary array to takes the stack values
        Stack = numpy.zeros([D_Train.shape[0], D_Train.shape[1]])
        
        # Iterate over each samples and calculate tha stack
        for (i,row) in enumerate(D_Train):
            for (j,x) in enumerate(row):
                Stack[i,j] = (((row < x).sum() + 1 ) / (D_Train.shape[1]+2))
                
    # Create the Final Matrix that will contain the Gaussianized Data
    D_Gaussianized = numpy.zeros([Stack.shape[0], Stack.shape[1]])
    
    # Iterate over each row to aplly the ppf function
    for (i,row) in enumerate(Stack):
        temp = numpy.array([norm.ppf(row)])
        D_Gaussianized[i,:] = temp
        
    return D_Gaussianized


def ksplit(D, L, K, idx):
    
    folds = []   # array of tuple (D_i, L_i)
    subsets = [] # array of tuple ((DTrain_i, LTrain_i), (DTsest_i, LTest_i))
    
    l = 0
    r0 = int(D.shape[1] / K)
    r = r0
    
    for i in range(K):
        if i == K-1:
            r = D.shape[1]
        subset_i = idx[l:r]
        D_i = D[:, subset_i]
        L_i  = L[subset_i]
        folds.append((D_i, L_i))
        l = r
        r = r + r0

    # Generate the k subsets
    for i in range(K):
        test_i = folds[i]
        first = True
        
        for j in range(K):
            if j != i:
                if first:
                    DTrain_i = folds[j][0]
                    LTrain_i = folds[j][1]
                    first = False
                else:
                    DTrain_i = numpy.hstack([DTrain_i, folds[j][0]])
                    LTrain_i = numpy.hstack([LTrain_i, folds[j][1]])
        
        subsets.append(((DTrain_i, LTrain_i), test_i))
        
    return subsets


def kfold_computeAll(DTrain, DGauss, L, K=5):
    """
    Returns 4 elements corresponding to subsets 
    of DTrain, DGauss, DGaussPCA7, DGaussPCA6.
    The subsets are array of K tuple organized as follows:
        [( (DTrain_i, LTrain_i), (DTsest_i, LTest_i) ), ....]
    """
    
    # PCA on the gaussianized dataset
    D_PCA7 = PCA(DGauss, 7)
    D_PCA6 = PCA(DGauss, 6)

    # Split the dataset into k folds
    numpy.random.seed(0)
    idx = numpy.random.permutation(DTrain.shape[1])
    
    subsets_T = ksplit(DTrain, L, K, idx)
    subsets_G = ksplit(DGauss, L, K, idx)
    subsets_PCA7 = ksplit(D_PCA7, L, K, idx)
    subsets_PCA6 = ksplit(D_PCA6, L, K, idx)

    return subsets_T, subsets_G, subsets_PCA7, subsets_PCA6
    