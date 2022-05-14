# Last edit: 14/05/2022 - Luigi

import numpy
import lib.plots as plt
from lib.dim_reduction import PCA, LDA

def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:8]
                attrs = numpy.array([float (i) for i in attrs])
                attrs = attrs.reshape(8,1)
                label = line.split(',')[8]
                DList.append(attrs)
                labelsList.append(label)
            except:
                print("An error occurred inside the function: 'load(fname)'")

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


if __name__ == '__main__':
    
    #-----------------------#
    #   TRAINING SECTION    #
    #-----------------------#
    m = 8
    hLabels = {
         0:'Non-Pulsar',
         1:'Pulsar'
        }
    D_Train, L_Train = load('data/Train.txt')
    
    # Data observation
    #plt.plot_hist(D_Train, L_Train, showClass=False, title='DATASET')
    #plt.plot_scatter(D_Train, L_Train)
    #plt.plot_hist(D_Train, L_Train, title='CLASSES')
    plt.plot_scatter_matrix(D_Train, L_Train, 'Train set')
    
    D_PCA = PCA(D_Train, m)
    D_LDA = LDA(D_Train, L_Train, m, n=2)
    D_LDA = numpy.dot(D_LDA.T, D_Train)
    
    plt.plot_scatter_matrix(D_PCA, L_Train, 'PCA')
    plt.plot_scatter_matrix(D_LDA, L_Train)

    #-------------------------#
    #   EVALUATION SECTION    #
    #-------------------------#
    ## D_Test, L_Test = load('data/Test.txt')











