# Last edit: 26/06/2022 - Luigi

import numpy
import lib.plots as plt
import lib.dim_reduction as dim_red
import lib.data_preparation as prep


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


if __name__ == '__main__':

    
    #-----------------#
    #  Data analysis  #
    #-----------------#
    D_Train, L_Train = load('data/Train.txt')

    # Histogram plot
    plt.plot_hist(D_Train, L_Train)
    
    # Heat plot of the abs value of Pearson coef
    P = numpy.abs(numpy.corrcoef(D_Train))
    P = numpy.round(P, 2)
    plt.plot_heatmap(P, title='Feature correlation')
    
    # Gaussianization to clear the outliers
    #D_Gaussianization=prep.gaussianization(D_Train)
    # Histogram plot of the new features Gaussianized
    #plt.plot_hist(D_Gaussianization, L_Train)


    #########################
    #   TRAINING SECTION    #
    #########################
    m = 6  # PCA and LDA parameter
    hLabels = { 0: 'Non-Pulsar', 1: 'Pulsar' }
    
    #------------------------------#
    #  Model training with K-FOLD  #
    #------------------------------#
    K = 5  # k-fold parameter

    Kfold_subsets = prep.k_fold(D_Train, L_Train, K)

    for i in range(K):
        DT_kfold = Kfold_subsets[i][0][0]  # Data Train
        LT_kfold = Kfold_subsets[i][0][1]  # Label Train
        DE_kfold = Kfold_subsets[i][1][0]  # Data Test (evaluation)
        LE_kfold = Kfold_subsets[i][1][1]  # Label Test (evaluation)

        # Dimentionality reduction: PCA / LDA
        DT_PCA = dim_red.PCA(DT_kfold, m)
        W_LDA = dim_red.LDA(DT_kfold, LT_kfold, m)
        DT_LDA = numpy.dot(W_LDA.T, DT_kfold)
        
        # Gaussianized dataset
        # TODO: Gaussianization of the k-fold dataset fragmet

        # MVG
        # TODO: GMV full-cov, diag-cov, tied full-cov, tied diag-cov
                
        # LR
        # TODO: LR with pi_T=[0.5, 0.1, 0.9], Quadratic LR
        
        # SVM
        # TODO: SVM with pi_T as above, Quadratic SVM, RBF kernel SVM
        
        # GMM
        # TODO: GMM full-cov, diag-cov, tied full-cov, tied diag-cov
        
        
        
        
    #----------------------------#
    #  Model quality evaluation  #
    #----------------------------#    
    
    # MVG
    # TODO: minDCF evaluation
    
    # LR
    # TODO: minDCF evaluation, comparison with the previous ones
    
    # SVM
    # TODO: minDCF evaluation, comparison with the previous ones
    
    # GMM
    # TODO: minDCF evaluation, comparison with the previous ones


    
        
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
