# Last edit: 13/05/2022 - Alex

import numpy
import pylab

def computeConfusionMatrix(TrueLabels,PredictedLabels,NumOfClass):
    '''
    Parameters
    ----------
    TrueLabels : Type: Numpy Array 
                 Description: Array of correct labels.
    
    PredictedLabels : Type: Numpy Array 
                      Description: Array of predicted labels.
                
    NumOfClass : Type: Integer
                 Description: Number of different classes.

    Returns
    -------
    ConfusionMatrix : Type: Numpy Multidimensional Array 
                      Description: Confusion Matrix.
    '''
    ConfusionMatrix=numpy.zeros((NumOfClass,NumOfClass))
    for i in range(NumOfClass):
        for j in range(NumOfClass):
            ConfusionMatrix[i,j]=((PredictedLabels==i)*(TrueLabels==j)).sum()
    return ConfusionMatrix


def computeROC(llRateos,TrueLabels,NumOfClass=2):
    '''
    Parameters
    ----------
    llRateos : Type: Numpy Array 
               Description: Array of LogLikelihood Rateos.
    TrueLabels : Type: Numpy Array 
                 Description: Array of correct labels.
    NumOfClass : Type: Integer
                 Description: Number of different classes.

    Returns
    -------
    None.
    '''
    
    thresolds=numpy.concatenate([ numpy.array([-numpy.inf]), llRateos, numpy.array([numpy.inf])])
    thresolds.sort()
    FPR=numpy.zeros(thresolds.size)
    TPR=numpy.zeros(thresolds.size)
    
    for z,t in enumerate(thresolds):
        PredictedLabels=numpy.int32(llRateos>t)
        Conf=computeConfusionMatrix(TrueLabels,PredictedLabels,NumOfClass)
        FPR[z]= Conf[1,0] / (Conf[1,0]+Conf[0,0])
        TPR[z]= Conf[1,1] / (Conf[1,1]+Conf[0,1])
    pylab.plot(FPR,TPR)
    pylab.show()
        
