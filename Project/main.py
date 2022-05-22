# Last edit: 14/05/2022 - Luigi

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

    #########################
    #   TRAINING SECTION    #
    #########################
    m = 6  # PCA and LDA parameter
    hLabels = {
        0: 'Non-Pulsar',
        1: 'Pulsar'
    }

    #--------------------#
    #  Data preparation  #
    #--------------------#
    D_Train, L_Train = load('data/Train.txt')

    # Plot evaluation
    #plt.plot_scatter_matrix(D_Train, L_Train, 'Train set')

    # handle class unbalance
    D_Train, L_Train = prep.random_under_sampling(D_Train, L_Train, n_kill=0.5)
    

    #------------------------------#
    #  Model training with K-FOLD  #
    #------------------------------#
    K = 10  # k-fold parameter

    Kfold_subsets = prep.k_fold(D_Train, L_Train, K)

    for i in range(K):
        DT_kfold = Kfold_subsets[i][0][0]  # Data Train
        LT_kfold = Kfold_subsets[i][0][1]  # Label Train
        DE_kfold = Kfold_subsets[i][1][0]  # Data Test (evaluation)
        LE_kfold = Kfold_subsets[i][1][1]  # Label Test (evaluation)

        # Dimentionality reduction: PCA / LDA
        # NOTA: valutare se vanno usate o meno, dai plot direi di no
        DT_PCA = dim_red.PCA(DT_kfold, m)
        W_LDA = dim_red.LDA(DT_kfold, LT_kfold, m)
        DT_LDA = numpy.dot(W_LDA.T, DT_kfold)

        # Model training

        # Model evaluation

    #----------------------------------#
    #  Model training with SPLIT 2TO1  #
    #----------------------------------#
    (DT_2to1, LT_2to1), (DE_2to1, LE_2to1) = prep.split_db_2to1(D_Train, L_Train)

    # Dimentionality reduction: PCA / LDA
    # NOTA: valutare se vanno usate o meno, dai plot direi di no
    DT_PCA = dim_red.PCA(DT_2to1, m)
    W_LDA = dim_red.LDA(DT_2to1, LT_2to1, m)
    DT_LDA = numpy.dot(W_LDA.T, DT_2to1)

    # Model training

    # Model evaluation

    #####################
    #   TEST SECTION    #
    #####################
    # D_Test, L_Test = load('data/Test.txt')
