import lib.model_evaluation as ev
import numpy
import scipy
import numpy.matlib 

def vrow(v):
    return v.reshape((1, v.size))

def vcol(v):
    return v.reshape((v.size, 1))

def logpdf_GAU_ND(X, mu, C): #Multivariate Gaussian log-density
    
    C_inverse = numpy.linalg.inv(C)        # Inverse of the covariance matrix
    _, C_logdet = numpy.linalg.slogdet(C)  # log-determinant log|C| di C
    M = mu.shape[0]
    Y = []

    a = -0.5 * M * numpy.log(2*numpy.pi)       # First constant element
    b = -0.5 * C_logdet                    # Secondo constant element
    for i in range(X.shape[1]):
        xi = vcol(X[:, i])
        c = -0.5 * numpy.dot((xi-mu).T, numpy.dot(C_inverse, (xi-mu)))[0][0]  
        logMVG = a+b+c
        Y.append(logMVG)
    
    return numpy.array(Y).ravel()        
    
def empirical_mean(X):
    return vcol(X.mean(1))

def empirical_cov(X):
    mu = empirical_mean(X)
    cov = numpy.dot((X-mu), (X-mu).T) / X.shape[1]
    return cov    


#-------------------#
#  MVG classifiers  #
#-------------------#

def MVG_Full(DT,LT,DE,LE,prior):
    
    mean0=empirical_mean(DT[:, LT == 0]) # Mean of the Gaussian Curve for Class 0
    mean1=empirical_mean(DT[:, LT == 1]) # Mean of the Gaussian Curve for Class 1
    
    sigma0=empirical_cov(DT[:, LT == 0]) # Sigma of the Gaussian Curve for Class 0
    sigma1=empirical_cov(DT[:, LT == 1]) # Sigma of the Gaussian Curve for Class 1
    
    LS0 = logpdf_GAU_ND(DE, mean0, sigma0) # Log Densities with parameters of Class 0
    LS1 = logpdf_GAU_ND(DE, mean1, sigma1) # Log Densities with parameters of Class 1
    
    SJoint = numpy.zeros((2, DE.shape[1]))
    
    SJoint[0, :] = numpy.exp(LS0) * (1-prior) #Product Between Densities LS0 and PriorProb
    SJoint[1, :] = numpy.exp(LS1) * (prior)   #Product Between Densities LS1 and PriorProb
    
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    LabelPred=SPost.argmax(0)
    
    llr = LS1-LS0 # log-likelihood ratios
    
    return llr, LabelPred


def MVG_Diag(DT,LT,DE,LE,prior):
    
    mean0=empirical_mean(DT[:, LT == 0]) # Mean of the Gaussian Curve for Class 0
    mean1=empirical_mean(DT[:, LT == 1]) # Mean of the Gaussian Curve for Class 1
    
    sigma0=numpy.diag(numpy.diag(empirical_cov(DT[:, LT == 0]) )) # Diagonal Sigma of the Gaussian Curve for Class 0
    sigma1=numpy.diag(numpy.diag(empirical_cov(DT[:, LT == 1])))  # Diagonal Sigma of the Gaussian Curve for Class 1
  
    LS0 = logpdf_GAU_ND(DE, mean0, sigma0) # Log Densities with parameters of Class 0
    LS1 = logpdf_GAU_ND(DE, mean1, sigma1) # Log Densities with parameters of Class 1
    
    SJoint = numpy.zeros((2, DE.shape[1]))
    
    SJoint[0, :] = numpy.exp(LS0) * (1-prior)        #Product Between Densities LS0 and PriorProb
    SJoint[1, :] = numpy.exp(LS1) * (prior)          #Product Between Densities LS1 and PriorProb
   
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    LabelPred=SPost.argmax(0)
    
    llr = LS1-LS0 # log-likelihood ratios
   
    return llr,LabelPred


def MVG_TiedFull(DT,LT,DE,LE,prior):
    
    mean0=empirical_mean(DT[:, LT == 0]) # Mean of the Gaussian Curve for Class 0
    mean1=empirical_mean(DT[:, LT == 1]) # Mean of the Gaussian Curve for Class 1
    
    sigma0=empirical_cov(DT[:, LT == 0]) # Sigma of the Gaussian Curve for Class 0
    sigma1=empirical_cov(DT[:, LT == 1]) # Sigma of the Gaussian Curve for Class 1
    
    sigma = 1/(DT.shape[1])*(DT[:, LT == 0].shape[1]*sigma0+DT[:, LT == 1].shape[1]*sigma1) # Shared Sigma
    
    LS0 = logpdf_GAU_ND(DE, mean0, sigma) # Log Densities with parameters of Class 0
    LS1 = logpdf_GAU_ND(DE, mean1, sigma) # Log Densities with parameters of Class 1
    
    SJoint = numpy.zeros((2, DE.shape[1]))
    
    SJoint[0, :] = numpy.exp(LS0) * (1-prior)        #Product Between Densities LS0 and PriorProb
    SJoint[1, :] = numpy.exp(LS1) * (prior)          #Product Between Densities LS1 and PriorProb
    
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    LabelPred=SPost.argmax(0)
    
    llr = LS1-LS0 # log-likelihood ratios
    
    return llr,LabelPred


def MVG_TiedDiag(DT,LT,DE,LE,prior):
    
    mean0=empirical_mean(DT[:, LT == 0]) # Mean of the Gaussian Curve for Class 0
    mean1=empirical_mean(DT[:, LT == 1]) # Mean of the Gaussian Curve for Class 1
    
    sigma0=empirical_cov(DT[:, LT == 0]) # Sigma of the Gaussian Curve for Class 0
    sigma1=empirical_cov(DT[:, LT == 1]) # Sigma of the Gaussian Curve for Class 1
    
    sigma = numpy.diag(numpy.diag(1/(DT.shape[1])*(DT[:, LT == 0].shape[1]*sigma0+DT[:, LT == 1].shape[1]*sigma1))) # Shared Sigma
    
    LS0 = logpdf_GAU_ND(DE, mean0, sigma) # Log Densities with parameters of Class 0
    LS1 = logpdf_GAU_ND(DE, mean1, sigma) # Log Densities with parameters of Class 1
    
    SJoint = numpy.zeros((2, DE.shape[1]))
    
    SJoint[0, :] = numpy.exp(LS0) * (1-prior)        #Product Between Densities LS0 and PriorProb
    SJoint[1, :] = numpy.exp(LS1) * (prior)          #Product Between Densities LS1 and PriorProb
    
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    LabelPred=SPost.argmax(0)
    
    llr = LS1-LS0 # log-likelihood ratios
    
    return llr,LabelPred


#----------#
#  Splits  #
#----------#

MVG_models = {
    'full': MVG_Full,
    'diag': MVG_Diag,
    'tied-full': MVG_TiedFull,
    'tied-diag': MVG_TiedDiag }


def kfold_MVG(k_subsets, K, prior, MVG_train, getScores=False):
    
    minDCF_final = []
    scores_final = []
    
    
    for p in prior:
        scores = []
        LE = []
        for i in range(K):
            DT_k, LT_k = k_subsets[i][0]  # Data and Label Train
            DE_k, LE_k = k_subsets[i][1]  # Data and Label Test
            
            llRateos, _ = MVG_train(DT_k, LT_k, DE_k, LE_k, p) # Classify the DE_k data
            scores.append(llRateos) 
            LE.append(LE_k)
        
        LE = numpy.concatenate(LE).ravel()    
        scores = numpy.concatenate(scores).ravel()
        if getScores: scores_final.append(vcol(scores))
        minDCF = ev.computeMinDCF(LE, scores, p, numpy.array([[0,1],[1,0]])) # Compute the minDCF
        minDCF_final.append(minDCF)
    
    if getScores:
        return scores_final, LE 
    return minDCF_final  
   
def single_split_MVG(split, prior, MVG_train):
    
    DT, LT = split[0] # Train Data and Labels
    DE, LE = split[1] # Test Data and Labels
    minDCF_final = []

    for p in prior:
        llRateos,_ = MVG_train(DT, LT, DE, LE, p)
        minDCF = ev.computeMinDCF(LE, llRateos, p, numpy.array([[0,1],[1,0]]))
        minDCF_final.append(minDCF)

    return minDCF_final


def kfold_MVG_actDCF(k_subsets, K, prior, MVG_train): 
    actDCF_final = []
    minDCF_final = []
    
    for p in prior:
        scores = []
        LE = []
        for i in range(K):
            DT_k, LT_k = k_subsets[i][0]  # Data and Label Train
            DE_k, LE_k = k_subsets[i][1]  # Data and Label Test
            
            llRateos, _ = MVG_train(DT_k, LT_k, DE_k, LE_k, p) # Classify the DE_k data
            scores.append(llRateos) 
            LE.append(LE_k)
        
        LE = numpy.concatenate(LE).ravel()    
        scores = numpy.concatenate(scores).ravel()
        actDCF = ev.computeActualDCF(LE, scores, p, 1, 1) # Compute the actDCF
        minDCF = ev.computeMinDCF(LE, scores, p, numpy.array([[0,1],[1,0]])) # Compute the minDCF
        actDCF_final.append(actDCF)
        minDCF_final.append(minDCF)
        print('MVG Tied-Full (prior=%.1f): actDCF=%.3f' % (p, actDCF))
        
    return actDCF_final, minDCF_final


def kfold_MVG_actDCF_Calibrated(k_subsets, K, prior, MVG_train, lambd=1e-4, getScores=False): 
    
    actDCF_final = []
    scores_final = []
    
    for p in prior:
        scores = []
        LE = []
        for i in range(K):
            DT_k, LT_k = k_subsets[i][0]  # Data and Label Train
            DE_k, LE_k = k_subsets[i][1]  # Data and Label Test
            
            llRateos, _ = MVG_train(DT_k, LT_k, DE_k, LE_k, p) # Classify the DE_k data
            scores.append(llRateos) 
            LE.append(LE_k)
        
        LE = numpy.concatenate(LE).ravel()    
        scores = numpy.concatenate(scores).ravel()
        scores = ev.calibrateScores(scores,LE,lambd,p)
        if getScores: scores_final.append(vcol(scores))
        actDCF = ev.computeActualDCF(LE, scores, p, 1, 1) # Compute the actDCF
        actDCF_final.append(actDCF)
        #print('MVG Tied-Full (prior=%.1f): actDCF=%.3f' % (p, actDCF))
        
    if getScores:
        return scores_final, LE

    return actDCF_final


def MVG_EVALUATION(split, prior, MVG_train, lambd_calib=1e-4, mode="full"):
    
    DT, LT = split[0] # Train Data and Labels
    DE, LE = split[1] # Test Data and Labels
    minDCF_final = []
    actDCF_final = []
    actDCFCalibrated_final = []

    for p in prior:
        llRateos,_ = MVG_train(DT, LT, DE, LE, p)
        actDCF = ev.computeActualDCF(LE, llRateos, p, 1, 1) # Compute the actDCF
        minDCF = ev.computeMinDCF(LE, llRateos, p, numpy.array([[0,1],[1,0]]))

        
        llRTrain,_=MVG_train(DT, LT, DT, LT, p)
        llR_Calib = ev.calibrateScoresForEvaluation(llRTrain,LT,llRateos,lambd_calib,p)
        actDCFCalibrated = ev.computeActualDCF(LE, llR_Calib, p, 1, 1) # Compute the actDCF
        
        minDCF_final.append(minDCF)
        actDCF_final.append(actDCF)
        actDCFCalibrated_final.append(actDCFCalibrated)
        print ("MVG- %s with prior = %.1f , minDCF = %.3f , actDCF = %.3f, actDCF (Calibrated) = %.3f" % (mode,p,minDCF,actDCF,actDCFCalibrated))
        

    return minDCF_final, actDCF_final, actDCFCalibrated_final







