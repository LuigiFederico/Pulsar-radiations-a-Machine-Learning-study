import numpy
import lib.plots as plt
import lib.dim_reduction as dim_red
import lib.data_preparation as prep
import lib.MVG_models as MVG
import lib.LR_models as LR


#########################
#   ANALYSIS SECTION    #
#########################

def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:8]
                attrs = numpy.array([float(i) for i in attrs])
                attrs = attrs.reshape(8, 1)
                label = line.split(',')[8]
                DList.append(attrs)
                labelsList.append(label)
            except:
                print("An error occurred inside the function: 'load(fname)'")

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

def feature_analysis(D, L, title=''):
    # Histogram plot
    plt.plot_hist(D, L)
    
    # Heat plot of the abs value of Pearson coef
    P = numpy.abs(numpy.corrcoef(D))
    P = numpy.round(P, 2)
    plt.plot_heatmap(P, title)


#########################
#   TRAINING SECTION    #
#########################

# Generative models

def kfold_MVG_compute(k_subsets, K, prior, f):
    """
    Wrapper function to compute minDCF of a MVG classifier as prior varies,
    with/without PCA (m=7, m=6)
    
    Parameters
    ----------
    k_subsets : array retrieved form prop.Kfold function
    K : K-fold parameter
    prior : array of prior probabilities with len = 3
    f : callback meaning the classifier to invoke

    Returns
    -------
    An array minDCF_MVG containing the minDCF obtained as follow:
       - minDCF_MVG[0] = f(k_subsets, K, prior[0])
       - minDCF_MVG[1] = f(k_subsets, K, prior[1])
       - minDCF_MVG[2] = f(k_subsets, K, prior[2])
       - minDCF_MVG[3] = f(k_subsets, K, prior[0], usePCA=True, mPCA=7)
       - minDCF_MVG[4] = f(k_subsets, K, prior[1], usePCA=True, mPCA=7)
       - minDCF_MVG[5] = f(k_subsets, K, prior[2], usePCA=True, mPCA=7)
       - minDCF_MVG[6] = f(k_subsets, K, prior[0], usePCA=True, mPCA=6)
       - minDCF_MVG[7] = f(k_subsets, K, prior[1], usePCA=True, mPCA=6)
       - minDCF_MVG[8] = f(k_subsets, K, prior[2], usePCA=True, mPCA=6)
        
    """
    minDCF_MVG = [] # 0,1,2 -> prior, noPCA ; 3,4,5 -> prior, PCA m=7; 6,7,8 -> prior, PCA m=6
    
    minDCF_MVG.append( f(k_subsets, K, prior[0]) )
    minDCF_MVG.append( f(k_subsets, K, prior[1]) )
    minDCF_MVG.append( f(k_subsets, K, prior[2]) )
  
    print (numpy.around(minDCF_MVG, 3)) # rounded
    #print(minDCF_MVG)
    return minDCF_MVG


def single_split_MVG_compute(split, prior, f):
    minDCF_MVG = []
    
    minDCF_MVG.append( f(split, prior[0]) )
    minDCF_MVG.append( f(split, prior[1]) )
    minDCF_MVG.append( f(split, prior[2]) )
    
    print (numpy.around(minDCF_MVG, 3)) # rounded
    #print(minDCF_MVG)
    return minDCF_MVG
    
        
def MVG_models(subsets, splits, prior, K):

    print('########   MVG   ########\n')
    
    print('------- K FOLD -------')
    k_subsets, k_subsets_PCA7, k_subsets_PCA6, k_gauss_subsets, k_gauss_PCA7_subs, k_gauss_PCA6_subs = subsets 
    
    print('\nTraining dataset')
    kfold_MVG_compute(k_subsets, K, prior, MVG.kfold_MVG_Full)
    kfold_MVG_compute(k_subsets, K, prior, MVG.kfold_MVG_Diag)
    kfold_MVG_compute(k_subsets, K, prior, MVG.kfold_MVG_TiedFull)
    kfold_MVG_compute(k_subsets, K, prior, MVG.kfold_MVG_TiedDiag)
    
    print('\nTraining dataset with PCA m = 7')
    kfold_MVG_compute(k_subsets_PCA7, K, prior, MVG.kfold_MVG_Full)
    kfold_MVG_compute(k_subsets_PCA7, K, prior, MVG.kfold_MVG_Diag)
    kfold_MVG_compute(k_subsets_PCA7, K, prior, MVG.kfold_MVG_TiedFull)
    kfold_MVG_compute(k_subsets_PCA7, K, prior, MVG.kfold_MVG_TiedDiag)
    
    print('\nTraining dataset with PCA m = 6')
    kfold_MVG_compute(k_subsets_PCA6, K, prior, MVG.kfold_MVG_Full)
    kfold_MVG_compute(k_subsets_PCA6, K, prior, MVG.kfold_MVG_Diag)
    kfold_MVG_compute(k_subsets_PCA6, K, prior, MVG.kfold_MVG_TiedFull)
    kfold_MVG_compute(k_subsets_PCA6, K, prior, MVG.kfold_MVG_TiedDiag)
    
    print('\nGaussianized dataset')
    kfold_MVG_compute(k_gauss_subsets, K, prior, MVG.kfold_MVG_Full)
    kfold_MVG_compute(k_gauss_subsets, K, prior, MVG.kfold_MVG_Diag)
    kfold_MVG_compute(k_gauss_subsets, K, prior, MVG.kfold_MVG_TiedFull)
    kfold_MVG_compute(k_gauss_subsets, K, prior, MVG.kfold_MVG_TiedDiag)

    print('\nGaussianized with PCA m = 7')
    kfold_MVG_compute(k_gauss_PCA7_subs, K, prior, MVG.kfold_MVG_Full)
    kfold_MVG_compute(k_gauss_PCA7_subs, K, prior, MVG.kfold_MVG_Diag)
    kfold_MVG_compute(k_gauss_PCA7_subs, K, prior, MVG.kfold_MVG_TiedFull)
    kfold_MVG_compute(k_gauss_PCA7_subs, K, prior, MVG.kfold_MVG_TiedDiag)

    print('\nGaussianized with PCA m = 6')
    kfold_MVG_compute(k_gauss_PCA6_subs, K, prior, MVG.kfold_MVG_Full)
    kfold_MVG_compute(k_gauss_PCA6_subs, K, prior, MVG.kfold_MVG_Diag)
    kfold_MVG_compute(k_gauss_PCA6_subs, K, prior, MVG.kfold_MVG_TiedFull)
    kfold_MVG_compute(k_gauss_PCA6_subs, K, prior, MVG.kfold_MVG_TiedDiag)
    
    print('----------------------\n')
    
    
    print('------- SINGLE SPLIT -------')
    train_split, train_PCA7_split, train_PCA6_split, gauss_split, gauss_PCA7_split, gauss_PCA6_split = splits
    
    print('\nTraining dataset')
    single_split_MVG_compute(train_split, prior, MVG.single_split_MVG_Full)
    single_split_MVG_compute(train_split, prior, MVG.single_split_MVG_Diag)
    single_split_MVG_compute(train_split, prior, MVG.single_split_MVG_TiedFull)
    single_split_MVG_compute(train_split, prior, MVG.single_split_MVG_TiedDiag)
    
    print('\nTraining dataset with PCA m = 7')
    single_split_MVG_compute(train_PCA7_split, prior, MVG.single_split_MVG_Full)
    single_split_MVG_compute(train_PCA7_split, prior, MVG.single_split_MVG_Diag)
    single_split_MVG_compute(train_PCA7_split, prior, MVG.single_split_MVG_TiedFull)
    single_split_MVG_compute(train_PCA7_split, prior, MVG.single_split_MVG_TiedDiag)
    
    print('\nTraining dataset  with PCA m = 6')
    single_split_MVG_compute(train_PCA6_split, prior, MVG.single_split_MVG_Full)
    single_split_MVG_compute(train_PCA6_split, prior, MVG.single_split_MVG_Diag)
    single_split_MVG_compute(train_PCA6_split, prior, MVG.single_split_MVG_TiedFull)
    single_split_MVG_compute(train_PCA6_split, prior, MVG.single_split_MVG_TiedDiag)
    
    print('\nGaussianized dataset')
    single_split_MVG_compute(gauss_split, prior, MVG.single_split_MVG_Full)
    single_split_MVG_compute(gauss_split, prior, MVG.single_split_MVG_Diag)
    single_split_MVG_compute(gauss_split, prior, MVG.single_split_MVG_TiedFull)
    single_split_MVG_compute(gauss_split, prior, MVG.single_split_MVG_TiedDiag)
    
    print('\nGaussianized with PCA m = 7')
    single_split_MVG_compute(gauss_PCA7_split, prior, MVG.single_split_MVG_Full)
    single_split_MVG_compute(gauss_PCA7_split, prior, MVG.single_split_MVG_Diag)
    single_split_MVG_compute(gauss_PCA7_split, prior, MVG.single_split_MVG_TiedFull)
    single_split_MVG_compute(gauss_PCA7_split, prior, MVG.single_split_MVG_TiedDiag)
    
    print('\nGaussianized with PCA m = 6')
    single_split_MVG_compute(gauss_PCA6_split, prior, MVG.single_split_MVG_Full)
    single_split_MVG_compute(gauss_PCA6_split, prior, MVG.single_split_MVG_Diag)
    single_split_MVG_compute(gauss_PCA6_split, prior, MVG.single_split_MVG_TiedFull)
    single_split_MVG_compute(gauss_PCA6_split, prior, MVG.single_split_MVG_TiedDiag)
    
    print('----------------------------\n')


# Discriminative models

def kfold_LR_compute(k_subsets, K, prior, f):
    
    minDCF_LR = [] # 0,1,2 -> prior, noPCA ; 3,4,5 -> prior, PCA m=7; 6,7,8 -> prior, PCA m=6
    lambdas=[]
    
    for p in prior :
        minDCF_values, lambdas = f(k_subsets, K, p)
        minDCF_LR.append(minDCF_values)
    
    plt.plot_DCF(lambdas, minDCF_LR, "λ")
    print (numpy.around(minDCF_LR, 3)) # rounded
    
    return minDCF_LR


def single_split_LR_compute(split, prior, f):
    
    minDCF_LR = [] # 0,1,2 -> prior, noPCA ; 3,4,5 -> prior, PCA m=7; 6,7,8 -> prior, PCA m=6
    lambdas=[]
    
    for p in prior :
        minDCF_values, lambdas = f( split , p)
        minDCF_LR.append(minDCF_values)
    
    plt.plot_DCF(lambdas, minDCF_LR, "λ")
    print (numpy.around(minDCF_LR, 3)) # rounded
    
    return minDCF_LR


def LR_models(subsets, splits, prior, K):
    print('########   LR   ########\n')
    
    print('------- K FOLD -------')
    k_subsets, k_subsets_PCA7, k_subsets_PCA6, k_gauss_subsets, k_gauss_PCA7_subs, k_gauss_PCA6_subs = subsets 
    
    print('\nTraining dataset')
    kfold_LR_compute(k_subsets, K, prior, LR.kfold_LogReg)
    
    print('\nGaussianized dataset')
    kfold_LR_compute(k_gauss_subsets, K, prior, LR.kfold_LogReg)
    
    
    print('----------------------\n')
    
    
    print('------- SINGLE SPLIT -------')
    train_split, train_PCA7_split, train_PCA6_split, gauss_split, gauss_PCA7_split, gauss_PCA6_split = splits
    
    print('\nTraining dataset')
    single_split_LR_compute(train_split, prior, LR.single_split_LogReg)
    
    print('\nGaussianized dataset')
    single_split_LR_compute(gauss_split, prior, LR.single_split_LogReg)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

if __name__ == '__main__':

    #-----------------#
    #  Data analysis  #
    #-----------------#
    D_Train, L_Train = load('data/Train.txt')
    feature_analysis(D_Train, L_Train, 'Feature correlation')

    # Gaussianization to clear the outliers
    D_Gaussianization = prep.gaussianization(D_Train)
    feature_analysis(D_Gaussianization, L_Train, 'Gaussianized features')

    
    #------------------------------#
    #  Model training with K-FOLD  #
    #------------------------------#
    K = 5  # k-fold 
    prior = [0.5, 0.9, 0.1]
    pi_T = [0.5, 0.9, 0.1]
    
    # K-fold
    subsets = prep.kfold_computeAll(D_Train, D_Gaussianization, L_Train, K)
    
    # Single split 80-20
    splits = prep.single_split_computeAll(D_Train, D_Gaussianization, L_Train)
    
    # MVG 
    MVG_models(subsets, splits, prior, K)
    
    
    # LR
    LR_models(subsets, splits, prior, K)
    # Without gaussianization
    # With gaussianization

    
    # SVM
    # Without gaussianization
    # With gaussianization
    
    
    # GMM
    # Without gaussianization
    # With gaussianization
   


    
        
    #----------------------------------#  
    #  Choice of the candidate models  #
    #----------------------------------#
    
    # ROC and DET curve
    # TODO: ROC curve plot, DET curve plot
    
    # Evaluation with other metrics
    # TODO: actual DCF
    # TODO: Bayes error plot
    
    # Merging models
    # TODO: fuse the most promising models
    # TODO: minDCF evaluation, actDCF, ROC, Bayesian error plot
    
    
    # TODO: Choose the best and final model!
        
        




    ###########################
    #   EVALUATION SECTION    #
    ###########################
    # D_Test, L_Test = load('data/Test.txt')
    
    # We need to replicate the analisys done before
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
