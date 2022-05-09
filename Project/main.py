# Last edit: /05/2022 - Alex

import numpy
from lib.plots import plot_hist, plot_scatter


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

    hLabels = {
         0:'Non-Pulsar',
         1:'Pulsar'
        }
    
    D_Train,L_Train = load('data/Train.txt')
    D_Test,L_Test = load('data/Test.txt')
    
    plot_hist(D_Train,L_Train)
    plot_scatter(D_Train,L_Train)




