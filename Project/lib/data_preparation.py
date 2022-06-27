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
    # if D_Evaluation is equal to 0 we are using the function to gaussianize train data
    if(D_Evaluation!=0):
        D_Gaussianized = numpy.zeros([D_Evaluation.shape[0], D_Evaluation.shape[1]])
        for row_count,(i,row_samples) in D_Train, enumerate(D_Evaluation):
            for (j,x) in enumerate(row_samples):
                D_Gaussianized[i,j]=((row_count<x).sum() + 1 )/(D_Train.shape[1]+2)
    
    else:
        D_Gaussianized = numpy.zeros([D_Train.shape[0], D_Train.shape[1]])
        for (i,row) in enumerate(D_Train):
            for (j,x) in enumerate(row):
                D_Gaussianized[i,j]=(((row < x).sum() + 1 )/(D_Train.shape[1]+2))
                
        
    return D_Gaussianized
