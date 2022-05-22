# Last edit: 22/05/2022 - Alex

import numpy
import pylab

def computeConfusionMatrix(TrueLabels, PredictedLabels, NumOfClass):
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


def computeROC(llRateos, TrueLabels, NumOfClass=2):
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
        

def computeDCFu(TrueLabel, PredLabel, pi, CostMatrix):
    '''
    Compute the DCFu value. This function can work in binary case and also in multiclass
    case. In Binary case we suggest to insert for pi values the float number of the prior
    probability for True Class, but, if not possible, can also be used the prior vec probabilites
    formatted in this way: [False Class Prob, True Class Prob]

    Parameters
    ----------
    TrueLabels : Type: Numpy Array 
                 Description: Array of correct labels.
    PredictedLabels : Type: Numpy Array 
                      Description: Array of predicted labels.
    pi : Type: Numpy Array or Float Single Value 
         Description: Array of prior probabilies or Single probabilities of TrueClass.
    CostMatrix : Type: Numpy Array 
                 Description: Matrix of costs. Depends from the application.

    Returns
    -------
    DCFu : Type: Float Value
           Description: DCFu Value.

    '''
    
    # Compute the number of classes == dimension of the CostMatrix
    dimension = CostMatrix.shape[0]
    
    # Compute Confusion Matrix using TrueLabels and PredLabesl
    ConfusionM = computeConfusionMatrix(TrueLabel, PredLabel, dimension)
    
    # Compute MissclassRateos by dividing each element by the sum of values of its column
    MisclassRateos = ConfusionM / ConfusionM.sum(axis=0)
    
    # If Dimension is 2 and pi is only a float, then we need to create the pi vec
    # Else we don't need to create nothing
    if(dimension==2 and (isinstance(pi, float))):
        pi_vec = numpy.array([(1-pi), pi])
    else:
        pi_vec = pi
    
    # Calculate the product between MisclassRateos and CostMatrix
    # We use the MisclassRateos transposed because the product is calculated
    # In the formula colum by column, and this can be replicated in Matrix product
    # Transposing one of the 2 matrices
    # In conclusion we takes only the diagonal elements because corrisponding
    # To the correct products
    # What we obtain correspond to the following formula
    # SemFin=Sum_{i=1}^k R_{ij}C_{ij}
    # With R the MisclassRateos and C the CostMatrix
    SemFin = numpy.dot(MisclassRateos.T, CostMatrix).diagonal()
    
    # Return the product by SemFin and pi_vec
    # Sum_{j=1}^k pi_{j} * SemFin
    return numpy.dot(SemFin, pi_vec.T)


def computeNormalizedDCF(TrueLabel, PredLabel, pi, CostMatrix):
    '''
    Compute the Normalized DCF value. This function can work in binary case and also in multiclass
    case. In Binary case we suggest to insert for pi values the float number of the prior
    probability for True Class, but, if not possible, can also be used the prior vec probabilites
    formatted in this way: [False Class Prob, True Class Prob]

    Parameters
    ----------
    TrueLabels : Type: Numpy Array 
                 Description: Array of correct labels.
    PredictedLabels : Type: Numpy Array 
                      Description: Array of predicted labels.
    pi : Type: Numpy Array or Float Single Value 
         Description: Array of prior probabilies or Single probabilities of TrueClass.
    CostMatrix : Type: Numpy Array 
                 Description: Matrix of costs. Depends from the application.

    Returns
    -------
    DCF : Type: Float Value
           Description: DCF Normalized Value.

    '''
    
    # Calculate the DCFu value
    dcf_u = computeDCFu(TrueLabel, PredLabel, pi, CostMatrix)
    
    # If Dimension is 2 and pi is only a float, then we need to create the pi vec
    # Else we don't need to create nothing
    if(CostMatrix.shape[0]==2 and (isinstance(pi, float))):
        pi_vec = numpy.array([(1-pi),pi])
    else:
        pi_vec = pi
    
    # Return the DCFu value divided by the minimum values from the product 
    # By CostMatrix and pi_vec
    return dcf_u / numpy.min(numpy.dot(CostMatrix, pi_vec))


def computeMinDCF(llRateos,TrueLabel,pi,CostMatrix):
    '''
    Compute the minimum DCF value. This function can work only in binary case We suggest to insert 
    for pi values the float number of the prior probability for True Class, but, if not possible, 
    can also be used the prior vec probabilites formatted in this way: [False Class Prob, True Class Prob]

    Parameters
    ----------
    TrueLabels : Type: Numpy Array 
                 Description: Array of correct labels.
    llRateos : Type: Numpy Array 
               Description: Array of posterior probabilities rateos.
    pi : Type: Numpy Array or Float Single Value 
         Description: Array of prior probabilies or Single probabilities of TrueClass.
    CostMatrix : Type: Numpy Array 
                 Description: Matrix of costs. Depends from the application.

    Returns
    -------
    Min DCF : Type: Float Value
              Description: Min DCF Value.
    '''
    
    # Concatenate the ordered llRateos with -inf and +inf
    T = numpy.concatenate([numpy.array([-numpy.inf]), numpy.sort(llRateos) , numpy.array([numpy.inf])])
    
    # Set the minimum value of DCF to +inf
    minDCF = numpy.inf
    
    # Iterate over all the llRateos plus +inf and -inf
    # For all this values used as threashold to classificate labels
    # We calculate the Normalized DCF and after we select the minimum 
    # Between all the DCF generated
    for z,t in enumerate(T):
        PredictedLabel = numpy.int32(llRateos > t)
        DCF = computeNormalizedDCF(TrueLabel, PredictedLabel, pi, CostMatrix)
        minDCF = min(DCF, minDCF)
    
    # Return the minimu DCF
    return minDCF