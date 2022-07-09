import numpy
import pylab
import lib.LR_models as LR

def vrow(v):
    return v.reshape((1, v.size))


def computeConfusionMatrix(TrueLabels, PredictedLabels):
    '''
    Parameters
    ----------
    TrueLabels : Type: Numpy Array
                 Description: Array of correct labels.

    PredictedLabels : Type: Numpy Array
                      Description: Array of predicted labels.

    Returns
    -------
    ConfusionMatrix : Type: Numpy Multidimensional Array
                      Description: Confusion Matrix.
    '''
    ConfusionMatrix = numpy.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ConfusionMatrix[i, j] = (
                (PredictedLabels == i)*(TrueLabels == j)).sum()
    return ConfusionMatrix


def computeROC(TrueLabels, llRateos, NumOfClass=2):
    '''
    Compute ROC Curve
    
    Parameters
    ----------
    TrueLabels : Type: Numpy Array
                 Description: Array of correct labels.
    llRateos : Type: Numpy Array
               Description: Array of LogLikelihood Rateos.
    NumOfClass : Type: Integer
                 Description: Number of different classes.

    Returns
    -------
    None.
    '''

    thresolds = numpy.concatenate(
        [numpy.array([-numpy.inf]), llRateos, numpy.array([numpy.inf])])
    thresolds.sort()
    FPR = numpy.zeros(thresolds.size)
    TPR = numpy.zeros(thresolds.size)

    for z, t in enumerate(thresolds):
        PredictedLabels = numpy.int32(llRateos > t)
        Conf = computeConfusionMatrix(TrueLabels, PredictedLabels)
        FPR[z] = Conf[1, 0] / (Conf[1, 0]+Conf[0, 0])
        TPR[z] = Conf[1, 1] / (Conf[1, 1]+Conf[0, 1])
    pylab.plot(FPR, TPR)
    pylab.show()


def computeDCFu(TrueLabel, PredLabel, pi, cfn=1, cfp=1):
    '''
    Compute the DCFu value. 

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

    # Compute Confusion Matrix using TrueLabels and PredLabesl
    ConfusionM = computeConfusionMatrix(TrueLabel, PredLabel)

    # Compute MissclassRateos by dividing each element by the sum of values of its column
    FNR = ConfusionM[0][1]/(ConfusionM[0][1]+ConfusionM[1][1])
    FPR = ConfusionM[1][0]/(ConfusionM[0][0]+ConfusionM[1][0])
    
    # cfn=CostMatrix[0][1]
    # cfp=CostMatrix[1][0]
   
    return (pi*cfn*FNR +(1-pi)*cfp*FPR)


def computeNormalizedDCF(TrueLabel, PredLabel, pi, CostMatrix):
    '''
    Compute the Normalized DCF value. 

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
    cfn=CostMatrix[0][1]
    cfp=CostMatrix[1][0]
    
    # Calculate the DCFu value
    dcf_u = computeDCFu(TrueLabel, PredLabel, pi, cfn, cfp)
    
    denomin = numpy.array([pi*cfn, (1-pi)*cfp])
    index = numpy.argmin (denomin) 
    
    return dcf_u/denomin[index]


def computeMinDCF(TrueLabel, llRateos, pi, CostMatrix):
    '''
    Compute the minimum DCF value. 

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


def computeActualDCF(TrueLabel, llRateos, pi, Predicted_Label, cfn=1, cfp=1):
    
    CostMatrix=numpy.array([[0,cfn],[cfp,0]])
    
    Predicted_Label = (llRateos > (-numpy.log(pi/(1-pi)))).astype(int)
    
    N_DCF = computeNormalizedDCF(TrueLabel, Predicted_Label, pi, CostMatrix)
    
    return N_DCF


def calibrateScores(s, L, lambd, prior=0.5):
    
    s=vrow(s)
    alpha, beta = LR.LogRegForCalibration(s,L,lambd,prior)
    calib_scores = alpha*s+beta-numpy.log(prior/(1-prior))
    
    return calib_scores

 
def computeFPR_TPR_FNR(llrs, TrueLabels, t):
    
    predLabels = (llrs > t).astype(int)
    confM = computeConfusionMatrix(TrueLabels, predLabels)
    
    FNR = confM[0][1] / (confM[0][1] + confM[1][1])
    TPR = 1-FNR
    FPR = confM[1][0] / (confM[0][0] + confM[1][0])
    
    return FPR, TPR, FNR
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
