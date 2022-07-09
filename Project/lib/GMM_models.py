import numpy
import scipy
import numpy.matlib
import lib.model_evaluation as ev
import lib.MVG_models as mvg



def vrow(v):
    return v.reshape((1, v.size))

def vcol(v):
    return v.reshape((v.size, 1))
    
def logpdf_GMM(X, gmm):
    S = numpy.zeros((len(gmm), X.shape[1]))
    
    for g in range(len(gmm)):
        (w, mu, C) = gmm[g]
        S[g, :] = mvg.logpdf_GAU_ND(X, mu, C) + numpy.log(w)
        
    logdens = scipy.special.logsumexp(S, axis=0)
    
    return S, logdens

#-------------------#
#  MVG classifiers  #
#-------------------#

def GMM_EM_Full(X, gmm, psi = 0.01):
    thNew = None
    thOld = None
    N = X.shape[1]
    D = X.shape[0]
    
    while thOld == None or thNew - thOld > 1e-6:
        thOld = thNew
        logSj, logSjMarg = logpdf_GMM(X, gmm)
        thNew = logSjMarg.sum() / N # log-likelihood
       
        # E-step: postirior prob for each component of the GMM
        P = numpy.exp(logSj - logSjMarg)
        
        # M-step: update the model parameters
        newGmm = []
        for i in range(len(gmm)):
            gamma = P[i, :]
            Z = gamma.sum()
            F = (vrow(gamma) * X).sum(1)
            S = numpy.dot(X, (vrow(gamma) * X).T)
            w = Z / N
            mu = vcol(F / Z)
            sigma = S / Z - numpy.dot(mu, mu.T)
            U, s, _ = numpy.linalg.svd(sigma)
            s[s < psi] = psi
            sigma = numpy.dot(U, vcol(s) * U.T)
            newGmm.append((w, mu, sigma))
        gmm = newGmm
    
    return gmm

def GMM_EM_Diag(X, gmm, psi = 0.01):
    thNew = None
    thOld = None
    N = X.shape[1]
    D = X.shape[0]
    
    while thOld == None or thNew - thOld > 1e-6:
        thOld = thNew
        logSj, logSjMarg = logpdf_GMM(X, gmm)
        thNew = logSjMarg.sum() / N # log-likelihood
       
        # E-step: postirior prob for each component of the GMM
        P = numpy.exp(logSj - logSjMarg)
        
        # M-step: update the model parameters
        newGmm = []
        for i in range(len(gmm)):
            gamma = P[i, :]
            Z = gamma.sum()
            F = (vrow(gamma) * X).sum(1)
            S = numpy.dot(X, (vrow(gamma) * X).T)
            w = Z / N
            mu = vcol(F / Z)
            sigma = S/Z - numpy.dot(mu, mu.T)
            sigma *= numpy.eye(sigma.shape[0]) # Diag cov
            U, s, _ = numpy.linalg.svd(sigma)
            s[s < psi] = psi
            sigma = numpy.dot(U, vcol(s) * U.T)
            newGmm.append((w, mu, sigma))
        gmm = newGmm
    
    return gmm

def GMM_EM_TiedFull(X, gmm, psi=0.01):
    thNew = None
    thOld = None
    N = X.shape[1]
    D = X.shape[0]
    
    while thOld == None or thNew - thOld > 1e-6:
        thOld = thNew
        logSj, logSjMarg = logpdf_GMM(X, gmm)
        thNew = logSjMarg.sum() / N # log-likelihood
       
        # E-step: postirior prob for each component of the GMM
        P = numpy.exp(logSj - logSjMarg)
        
        # M-step: update the model parameters
        newGmm = []
        sigmaTied = numpy.zeros((D, D))
        for i in range(len(gmm)):
            gamma = P[i, :]
            Z = gamma.sum()
            F = (vrow(gamma) * X).sum(1)
            S = numpy.dot(X, (vrow(gamma) * X).T)
            w = Z/N
            mu = vcol(F/Z)
            sigma = S/Z - numpy.dot(mu, mu.T)
            sigmaTied += Z * sigma
            newGmm.append((w, mu))   
        gmm = newGmm
        sigmaTied /= N
        U, s, _ = numpy.linalg.svd(sigmaTied)
        s[s<psi] = psi
        sigmaTied = numpy.dot(U, vcol(s) * U.T)
        
        newGmm = []
        for i in range(len(gmm)):
            (w, mu) = gmm[i]
            newGmm.append((w, mu, sigmaTied))
            
        gmm = newGmm
    
    return gmm

def GMM_EM_TiedDiag(X, gmm, psi=0.01):
    thNew = None
    thOld = None
    N = X.shape[1]
    D = X.shape[0]
    
    while thOld == None or thNew - thOld > 1e-6:
        thOld = thNew
        logSj, logSjMarg = logpdf_GMM(X, gmm)
        thNew = logSjMarg.sum() / N # log-likelihood
       
        # E-step: postirior prob for each component of the GMM
        P = numpy.exp(logSj - logSjMarg)
        
        # M-step: update the model parameters
        newGmm = []
        sigmaTied = numpy.zeros((D, D))
        for i in range(len(gmm)):
            gamma = P[i, :]
            Z = gamma.sum()
            F = (vrow(gamma) * X).sum(1)
            S = numpy.dot(X, (vrow(gamma) * X).T)
            w = Z/N
            mu = vcol(F/Z)
            sigma = S/Z - numpy.dot(mu, mu.T)
            sigmaTied += Z * sigma
            newGmm.append((w, mu))   
        gmm = newGmm
        sigmaTied /= N
        sigmaTied *= numpy.eye(sigma.shape[0]) # Diag Cov
        U, s, _ = numpy.linalg.svd(sigmaTied)
        s[s<psi] = psi
        sigmaTied = numpy.dot(U, vcol(s) * U.T)
        
        newGmm = []
        for i in range(len(gmm)):
            (w, mu) = gmm[i]
            newGmm.append((w, mu, sigmaTied))
            
        gmm = newGmm
    
    return gmm


def GMM_LBG(X, alpha, nComponents, GMM_EM_train, psi = 0.01):
    # GMM_EM_train is a callback retrived from the GMM_EM_models dictionary
    
    gmm = [(1, mvg.empirical_mean(X), mvg.empirical_cov(X))]
    
    while len(gmm) <= nComponents:
        # # print(f'\nGMM has {len(gmm)} components')
        gmm = GMM_EM_train(X, gmm, psi)
                
        if len(gmm) == nComponents:
            break
        
        newGmm = []
        for i in range(len(gmm)):
            (w, mu, sigma) = gmm[i]
            U, s, Vh = numpy.linalg.svd(sigma)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            newGmm.append((w/2, mu + d, sigma))
            newGmm.append((w/2, mu - d, sigma))
        gmm = newGmm
            
    return gmm


#----------#
#  Splits  #
#----------#

GMM_EM_models = {
    'full': GMM_EM_Full,
    'diag': GMM_EM_Diag,
    'tied-full': GMM_EM_TiedFull,
    'tied-diag': GMM_EM_TiedDiag }

def kfold_GMM(subset, K, prior, mode, alpha, nComponents, psi=0.01): 
    minDCF_final = []  
    
    for Comp in nComponents:
        scores = []
        LE = []
        
        for i in range(K):
            DT_k, LT_k = subset[i][0]  # Data and Label Train
            DE_k, LE_k = subset[i][1]  # Data and Label Test
            DT_0 = DT_k[:, LT_k==0]
            DT_1 = DT_k[:, LT_k==1]
            
            # Train        
            gmm_c0 = GMM_LBG(DT_0, alpha, Comp, GMM_EM_models[mode], psi)
            gmm_c1 = GMM_LBG(DT_1, alpha, Comp, GMM_EM_models[mode], psi)
            
            # Test
            _, llr_0 = logpdf_GMM(DE_k, gmm_c0)
            _, llr_1 = logpdf_GMM(DE_k, gmm_c1)
            
            scores.append(llr_1 - llr_0)
            LE.append(LE_k)
            
        LE = numpy.concatenate(LE).ravel()
        scores = numpy.concatenate(scores).ravel()
        minDCF = ev.computeMinDCF(LE, scores, prior, numpy.array([[0,1],[1,0]]))
        minDCF_final.append(minDCF)
        print("[%d K-Fold] GMM - %s, α=%.2f, %d gau, prior = %.2f, minDCF = %.3f" % (K,mode, alpha, Comp, prior, minDCF))
    
    return minDCF_final

def single_split_GMM(split, prior, mode, alpha, nComponents, psi=0.01):
    DT, LT = split[0] # Train Data and Labels
    DE, LE = split[1] # Test Data and Labels
    DT_0 = DT[:, LT==0]
    DT_1 = DT[:, LT==1]
    minDCF_final = []  

    for Comp in nComponents:
        # Train        
        gmm_c0 = GMM_LBG(DT_0, alpha, prior, Comp, GMM_EM_models[mode], psi)
        gmm_c1 = GMM_LBG(DT_1, alpha, prior, Comp, GMM_EM_models[mode], psi)
        
        # Test
        _, llr_0 = logpdf_GMM(DE, gmm_c0)
        _, llr_1 = logpdf_GMM(DE, gmm_c1)
        llRateos = llr_1 - llr_0
        
        minDCF = ev.computeMinDCF(LE, llRateos, prior, numpy.array([[0,1],[1,0]]))
        minDCF_final.append(minDCF)
        print("[Single_Split] GMM - %s, α=%.2f, %d gau, prior = %.2f, minDCF = %.3f" % (mode, alpha, Comp, prior, minDCF))
    
    return minDCF_final
    




