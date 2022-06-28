import numpy
import sys
from scipy.stats import norm
from lib.dim_reduction import PCA

# -------------------- #
#     SPLIT DATASET    #
# -------------------- #

def single_split(D, L, idxTrain, idxTest):
    DTR = D[:, idxTrain]    # Traning data
    DTE = D[:, idxTest]     # Test data
    LTR = L[idxTrain]       # Training labels
    LTE = L[idxTest]        # Test labels

    return (DTR, LTR), (DTE, LTE)


def single_split_computeAll(DTrain, DGauss, L):
    """
    Returns 4 elements corresponding to a single split of:
        - the training dataset, 
        - the training dataset with PCA m = 7,
        - the training dataset with PCA m = 6,
        - the gaussianized dataset,
        - the gaussianized dataset with PCA m = 7,
        - the gaussianized dataset with PCA m = 6,
        - the gaussianized dataset with PCA m = 5.
    
    Every element contains two tuple as follows:
        (DTrain, LTrain), (DTest, LTest)
    """
    
    # PCA on the gaussianized dataset
    DT_PCA7 = PCA(DTrain, 7)
    DT_PCA6 = PCA(DTrain, 6)
    DG_PCA7 = PCA(DGauss, 7)
    DG_PCA6 = PCA(DGauss, 6)
    
    # Split seed
    nTrain = int(DTrain.shape[1] * 2.0/3.0)
    numpy.random.seed(0)
    idx = numpy.random.permutation(DTrain.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    # Single split
    split_T = single_split(DTrain, L, idxTrain, idxTest)
    split_T_PCA7 = single_split(DT_PCA7, L, idxTrain, idxTest)
    split_T_PCA6 = single_split(DT_PCA6, L, idxTrain, idxTest)
    
    split_G = single_split(DGauss, L, idxTrain, idxTest)
    split_G_PCA7 = single_split(DG_PCA7, L, idxTrain, idxTest)
    split_G_PCA6 = single_split(DG_PCA6, L, idxTrain, idxTest)
    
    return split_T, split_T_PCA7, split_T_PCA6, split_G, split_G_PCA7, split_G_PCA6


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
    Returns 4 elements corresponding to subsets of:
        - Training dataset, 
        - Training dataset with PCA m = 7,
        - Training dataset with PCA m = 6,
        - Gaussianized dataset,
        - Gaussianized dataset with PCA m = 7,
        - Gaussianized dataset with PCA m = 6
        
    The subsets are array of K tuple organized as follows:
        [( (DTrain_i, LTrain_i), (DTsest_i, LTest_i) ), ....]
    """
    
    # PCA
    DT_PCA7 = PCA(DTrain, 7)
    DT_PCA6 = PCA(DTrain, 6)
    DG_PCA7 = PCA(DGauss, 7)
    DG_PCA6 = PCA(DGauss, 6)
    
    # Split seed
    numpy.random.seed(0)
    idx = numpy.random.permutation(DTrain.shape[1])
    
    # K-fold split 
    subsets_T = ksplit(DTrain, L, K, idx)
    subsets_T_PCA7 = ksplit(DT_PCA7, L, K, idx)
    subsets_T_PCA6 = ksplit(DT_PCA6, L, K, idx)
    
    subsets_G = ksplit(DGauss, L, K, idx)
    subsets_G_PCA7 = ksplit(DG_PCA7, L, K, idx)
    subsets_G_PCA6 = ksplit(DG_PCA6, L, K, idx)

    return subsets_T, subsets_T_PCA7, subsets_T_PCA6, subsets_G, subsets_G_PCA7, subsets_G_PCA6
    