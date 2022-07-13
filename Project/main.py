import numpy
import lib.plots as plt
import lib.dim_reduction as dim_red
import lib.data_preparation as prep
import lib.MVG_models as MVG
import lib.LR_models as LR
import lib.SVM_models as SVM
import lib.GMM_models as GMM
import lib.model_evaluation as ev


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


########################
#   MODELS TRAINING    #
#      Pulsar=1        #
#     NonPulsar=0      #
#  Pi_T->ClassLabel=1  #
########################

      
def MVG_models(subsets, splits, prior, K):

    def kfold_MVG_compute(k_subsets, K, prior, title=''):
        
        print(title)
        for k in MVG.MVG_models.keys():
            minDCF = numpy.round( MVG.kfold_MVG(k_subsets, K, prior, MVG.MVG_models[k]), 3)
            for i,p in enumerate(prior):
                print("- %s, prior = %.2f, minDCF = %.3f" % (k, p, minDCF[i]))
        

    def single_split_MVG_compute(split, prior, title=''):
        
        print(title)
        for k in MVG.MVG_models.keys():
            minDCF = numpy.round( MVG.single_split_MVG(split, prior, MVG.MVG_models[k]), 3)
            for i,p in enumerate(prior):
                print("- %s, prior = %.2f, minDCF = %.3f" % (k, p, minDCF[i]))
    
    TrainMod = ['Raw dataset',
                'Raw dataset + PCA m = 7',
                'Raw dataset + PCA m = 6',
                'Gaussianized dataset',
                'Gaussianized dataset + PCA m = 7',
                'Gaussianized dataset + PCA m = 6']

    print('########   MVG   ########\n')
    
    print('------- %d-FOLD -------\n' % K)
    for (i, TrainLabel) in enumerate(TrainMod):
        kfold_MVG_compute(subsets[i], K, prior, TrainLabel)
     
    #print('----- SINGLE SPLIT -----\n')
    #for(i, TrainLabel) in enumerate(TrainMod):
    #    single_split_MVG_compute(splits[i], prior, TrainLabel)

    
def LR_models(subsets, splits, prior, K, lambdas, quadratic=False, pi_t=0.5):
    
    def kfold_LR_compute(k_subsets, K, lambdas, prior , pi_t, f):   
        minDCF_LR = []
        
        for pi_T in pi_t:
            minDCF_values = f(k_subsets, K, lambdas, prior, pi_T)
            minDCF_LR.append(minDCF_values)
            #plt.plot_DCF(lambdas, minDCF_LR, "λ")
            #print (numpy.around(minDCF_LR, 3)) # rounded
        
        return minDCF_LR
    
    
    def single_split_LR_compute(split, lambdas, prior, pi_t, f):
        minDCF_LR = [] 
        for pi_T in pi_t:
            minDCF_values  = f( split , lambdas, prior, pi_T)
            minDCF_LR.append(minDCF_values)
            #plt.plot_DCF(lambdas, minDCF_LR, "λ")
            #print (numpy.around(minDCF_LR, 3)) # rounded
        
        return minDCF_LR    
    
    
    TrainMod=["Raw dataset",
              "Raw dataset with PCA m = 7",
              "Raw dataset with PCA m = 6",
              "Gaussianized dataset","Gaussianized with PCA m = 7",
              "Gaussianized with PCA m = 6"]    
        
    if quadratic == False:
        print('########   LR   ########\n')
          
        print('-------  %d-FOLD  -------\n' % K)
        for (i, TrainLabel) in enumerate(TrainMod):
            print('\n',TrainLabel)
            kfold_LR_compute(subsets[i], K, lambdas, prior, pi_t, LR.kfold_LogReg)
        
        #print('------- SINGLE SPLIT -------')
        #for (i,TrainLabel) in enumerate(TrainMod):
        #    print('\n',TrainLabel)
        #    single_split_LR_compute(splits[i], lambdas, prior, pi_t, LR.single_split_LogReg)
            
    else:
        print('####  Quadratic LR  ####\n')
        
        print('-------  %d-FOLD  -------\n' % K)
        k_subsets, k_subsets_PCA7, k_subsets_PCA6, k_gauss_subsets, k_gauss_PCA7_subs, k_gauss_PCA6_subs = subsets 
        
        print('\nRaw dataset')
        kfold_LR_compute(k_subsets, K, lambdas, prior, pi_t, LR.kfold_QuadLogReg)
        
        print('\nGaussianized dataset')
        kfold_LR_compute(k_gauss_subsets, K, lambdas, prior, pi_t, LR.kfold_QuadLogReg)
        
        #print('------- SINGLE SPLIT -------')
        #train_split, train_PCA7_split, train_PCA6_split, gauss_split, gauss_PCA7_split, gauss_PCA6_split = splits
        
        #print('\nRaw dataset')
        #single_split_LR_compute(train_split, lambdas, prior, pi_t, LR.single_split_QuadLogReg)
        
        #print('\nGaussianized dataset')
        #single_split_LR_compute(gauss_split, lambdas, prior, pi_t, LR.single_split_QuadLogReg)
    
        
def SVM_models(subsets, splits, prior, K, Cs, pi_t=0.5, mode='linear'):
    
    def single_split_SVM(split, prior, Cs, pi_t, f, mode):
        minDCF_SVM = []
        
        if mode == 'balanced-linear':
            for pi_T in pi_t:
                for p in prior:
                    minDCF_values, LabelPredicetd = f(split, Cs, pi_T, p, mode)
        else:
            minDCF_SVM, LabelPredicetd = f(split, Cs, pi_t, prior, mode)
            
        return minDCF_SVM
    
    
    def kfold_SVM(k_subsets, prior, K, Cs, pi_t, f, mode='linear', title=''):
        minDCF_SVM = []
        
        if mode == 'balanced-linear':
            for pi_T in pi_t:
                minDCF_values = f(k_subsets, Cs, pi_T, prior, K, mode)
                minDCF_SVM.append(minDCF_values)
            print (numpy.around(minDCF_SVM, 3)) # rounded
        
        elif mode=='poly': # prior=0.5
            c_vec = [0, 1, 10, 30]
            for c in c_vec:
                minDCF_values = f(k_subsets, Cs, c=c, mode=mode)
                minDCF_SVM.append(minDCF_values)
            plt.plot_DCF(Cs, minDCF_SVM, "C", title)
            print (numpy.around(minDCF_SVM), 3)

        elif mode=='RBF':
            gammas=numpy.logspace(-3,-1,3) # prior=0.5
            for gamma in gammas:
                minDCF_values = f(k_subsets, Cs, pi_t, [0.5], K, mode, gamma)
                minDCF_SVM.append(minDCF_values)
            print (numpy.around(minDCF_SVM, 3)) # rounded
            plt.plot_DCF(Cs, minDCF_SVM, "C")
            print (numpy.around(minDCF_SVM), 3)
            
            return minDCF_SVM
            
        else:               
            minDCF_SVM = f(k_subsets, Cs, pi_t, prior, K, mode)
            print (numpy.around(minDCF_SVM, 3)) # rounded
        
        return minDCF_SVM
    
    
    print('########  %s SVM  ########\n' % mode)
    
    #print('------- SINGLE SPLIT -------')
    #train_split, _, _, gauss_split, _, _ = splits
    
    #print('\nRaw dataset')
    #single_split_SVM(train_split, prior, Cs, pi_t, SVM.single_split_SVM, mode)
    
    #print('\nGaussianized dataset')
    #single_split_SVM(gauss_split, prior, Cs, pi_t, SVM.single_split_SVM, mode)
    
    
    print('-------  %d-FOLD  -------\n' % K)
    
    k_subsets, k_subsets_PCA7, k_subsets_PCA6, k_gauss_subsets, k_gauss_PCA7_subs, k_gauss_PCA6_subs = subsets    
    
    print('\nRaw dataset')
    kfold_SVM(k_subsets, prior, K, Cs, pi_t, SVM.kfold_SVM, mode, title="SVM_Raw_prior5")
    
    #print('\nGaussianized dataset')
    #kfold_SVM(k_guass, prior, K, Cs, pi_t, SVM.kfold_SVM, mode)
    
    print('\nGaussianized dataset + PCA m=6')
    kfold_SVM(k_gauss_PCA6_subs, prior, K, Cs, pi_t, SVM.kfold_SVM, mode, title="SVM_GaussPCA6_prior5")
    
  
def GMM_models(subsets, splits, prior, K, alpha, nComponents, mode='full', psi=0.01 ):
    
    def single_split_GMM(split, prior, alpha, nComponents, mode='full', psi=0.01):

        minDCF_GMM = []
        minDCF_GMM = GMM.single_split_GMM(split, prior, mode, 
                                          alpha, nComponents, psi)
        print (numpy.around(minDCF_GMM, 3)) # rounded
        #plt.plot_DCF(nComponents, minDCF_GMM, "GMM Components",2)
    
    def kfold_GMM(subset, K, prior, alpha, nComponents, mode='full', psi=0.01, title=''):
        
        minDCF_GMM = []
        minDCF_GMM = GMM.kfold_GMM(subset, K, prior, mode,
                                   alpha, nComponents, psi)

        print (numpy.around(minDCF_GMM, 3)) # rounded
        #plt.plot_DCF(nComponents, minDCF_GMM, "GMM Components", 2, title)        
        

    print('########  GMM - %s-cov  ########\n' % mode)
     
    # print('------- SINGLE SPLIT -------')
    # train_split, _, _, gauss_split, _, _ = splits
    
    # print('\nRaw dataset')    
    # single_split_GMM(train_split, prior, alpha, nComponents, mode, psi)
    
    # print('\nGaussianized dataset')    
    # single_split_GMM(gauss_split, prior, alpha, nComponents, mode, psi)
    
    print('-------  %d-FOLD  -------\n' % K)
    k_subset, _, _, k_guass, _, _ = subsets    
    
    print('\nRaw dataset')    
    kfold_GMM(k_subset, K, prior, alpha, 
              nComponents, mode, psi, 'GMM-'+mode+' Train')
    
    print('\nGaussianized dataset')      
    kfold_GMM(k_guass, K, prior, alpha,
              nComponents, mode, psi, 'GMM-'+mode+' Gauss')
    
  
def score_calibration(subsets, prior, K):
    k_raw, _, _, _, _, k_gauss_PCA6 = subsets    
    numberOfPoints=21
    effPriorLogOdds = numpy.linspace(-3, 3, numberOfPoints)
    effPriors = 1/(1+numpy.exp(-1*effPriorLogOdds))
    C = 0.1
    pi_T = 0.5
    lambd = 0.00001
    pi_T_LogReg = 0.1
    
    # MVG Tied Full-Cov - Raw
    print("Start MVG Tied Full-Cov 5-folds on raw features...")
    actDCF_MVG = MVG.kfold_MVG_actDCF(k_raw, K, prior, MVG.MVG_models['tied-full'])
    print("End")
    
    actualDCFs=[]
    minDCFs=[]
    for i in range(numberOfPoints):
      actDCF, minDCF = MVG.kfold_MVG_actDCF(k_raw, K, [effPriors[i]], MVG.MVG_models['tied-full'])
      actDCF=actDCF[0]
      minDCF=minDCF[0]
      actualDCFs.append(actDCF)
      minDCFs.append(minDCF)
      print("At iteration", i, "the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
    plt.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "MVG Tied Full-Cov")
    
    
    #------------------------------------------------##
    
    
    # LogReg (lambda=1e-5, pi_T=0.1) - Raw

    print("Start LogReg (λ=1e-5, pi_T=0.1) 5-folds on raw features...")
    actDCF_LR = LR.kfold_LogReg_actDCF(k_raw, K, lambd, prior, pi_T)
    print("End")
    
    actualDCFs=[]
    minDCFs=[]
    for i in range(numberOfPoints):
      actDCF, minDCF = LR.kfold_LogReg_actDCF(k_raw, K, lambd, [effPriors[i]], pi_T_LogReg)
      actDCF=actDCF[0]
      minDCF=minDCF[0]
      actualDCFs.append(actDCF)
      minDCFs.append(minDCF)
      print("At iteration", i, "the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
    plt.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "LogReg")
    
    
    ##------------------------------------------------##
        
    
    # linear SVM (C=0.1, pi_T=0.5) - gauss PCA6

    print("Start Linear balanced SVM (C=0.1, pi_T=0.5) 5-folds on gaussianized features with PCA m=6...")
    actDCF_SVM, minDCF_SVM = SVM.kfold_SVM_actDCF(k_gauss_PCA6, C, pi_T, prior, K, mode='balanced-linear')
    print(actDCF_SVM)
    print("End")
    
    actualDCFs=[]
    minDCFs=[]
    for i in range(numberOfPoints):
      actDCF, minDCF = SVM.kfold_SVM_actDCF(k_gauss_PCA6, C, pi_T, [effPriors[i]], K, mode='balanced-linear')
      actDCF=actDCF[0]
      minDCF=minDCF[0]
      actualDCFs.append(actDCF)
      minDCFs.append(minDCF)
      print("At iteration", i, "the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
    plt.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "Linear-SVM")
    
    
    ##------------------------------------------------##
    ##              CALIBRATED VERSION                ##
    ##------------------------------------------------##
    
    lamb_calibration=1e-4
    
    ##------------------------------------------------##
    
    #print("Calibrated MVG Tied Full-Cov 5-folds on raw features")
    actualDCFs=[]
    minDCFs=[]
    for i in range(numberOfPoints):
      minDCF = MVG.kfold_MVG(k_raw, K, [effPriors[i]], MVG.MVG_models['tied-full'])
      actDCF = MVG.kfold_MVG_actDCF_Calibrated(k_raw, K, [effPriors[i]], MVG.MVG_models['tied-full'],lamb_calibration)
      actDCF=actDCF[0]
      minDCF=minDCF[0][0]
      actualDCFs.append(actDCF)
      minDCFs.append(minDCF)
      print("At iteration after Calibration", i, "the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
    plt.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "MVG Tied Full-Cov,", True,lamb_calibration)
    
    ##------------------------------------------------##
    
    print("Calibrated LogReg (λ=1e-5, pi_T=0.1) 5-folds on raw features")
    actualDCFs=[]
    minDCFs=[]
    for i in range(numberOfPoints):
      minDCF=LR.kfold_LogReg(k_raw, K, [lambd], effPriors[i], pi_T_LogReg)
      actDCF=LR.kfold_LogReg_actDCF_Calibrated(k_raw, K, lambd, [effPriors[i]], pi_T_LogReg, lamb_calibration)
      actDCF=actDCF[0]
      minDCF=minDCF[0][0]
      actualDCFs.append(actDCF)
      minDCFs.append(minDCF)
      print("At iteration", i, " after Calibration with effPriors ="  ,effPriors[i] ,"the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
    plt.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "LogReg,", True,lamb_calibration)
    
    
    ##------------------------------------------------##
    
    print("Calibrated Linear SVM (C=0.1, pi_T=0.5) 5-folds on gaussianized features with PCA m=6")
    actualDCFs=[]
    minDCFs=[]
    for i in range(numberOfPoints):
      minDCF=SVM.kfold_SVM(k_gauss_PCA6, [C], pi_T, effPriors[i], K=5, mode='balanced-linear')
      actDCF=SVM.kfold_SVM_actDCF_Calibrated(k_gauss_PCA6, C, pi_T, [effPriors[i]], K=5, mode='balanced-linear', lambd_calib=1e-4 )
      actDCF=actDCF[0]
      minDCF=minDCF[0][0]
      actualDCFs.append(actDCF)
      minDCFs.append(minDCF)
      print("At iteration", i, " after Calibration with effPriors ="  ,effPriors[i] ,"the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
    plt.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "Linear SVM,", True,lamb_calibration)
    
    False
    

def ROC(subsets, K, prior, calibrated=True):
    k_raw, _, _, _, _, k_gauss_PCA6 = subsets    
    
    
    # MVG Tied Full-Cov - Raw
    print("Start MVG....")
    FPR_MVG = []
    TPR_MVG = []
    FNR_MVG = []

    if not calibrated:
        scores_MVG, LE_MVG = MVG.kfold_MVG(
                        k_raw, K, prior, 
                        MVG.MVG_models['tied-full'], getScores=True)        
    else:
        scores_MVG, LE_MVG = MVG.kfold_MVG_actDCF_Calibrated(
                        k_raw, K, prior, 
                        MVG.MVG_models['tied-full'], getScores=True)
    scores_MVG = scores_MVG[0].flatten()
    sortedScores_MVG = numpy.sort(scores_MVG)

    for t in sortedScores_MVG:
        FPR, TPR, FNR = ev.computeFPR_TPR_FNR(scores_MVG, LE_MVG, t)
        FPR_MVG.append(FPR)
        TPR_MVG.append(TPR)
        FNR_MVG.append(FNR)
    print("- MVG done.")
    
    
    # Linear LR lambda=1e-5, pi_T=0.5
    print("Start LR...")
    FPR_LR = []
    TPR_LR = []
    FNR_LR = []
    pi_T_LR = 0.5
    
    if not calibrated:
        lambd = [0.00001]
        scores_LR, LE_LR = LR.kfold_LogReg(
                        k_raw, K, lambd, prior, pi_T_LR, getScores=True)
    else:
        lambd = 0.00001
        scores_LR, LE_LR = LR.kfold_LogReg_actDCF_Calibrated(
                        k_raw, K, lambd, prior, pi_T_LR, getScores=True)
    scores_LR = scores_LR[0].flatten()
    print(scores_LR.shape)
    sortedScores_LR = numpy.sort(scores_LR)
    
    for t in sortedScores_LR:
        FPR, TPR, FNR = ev.computeFPR_TPR_FNR(scores_LR, LE_LR, t)
        FPR_LR.append(FPR)
        TPR_LR.append(TPR)
        FNR_LR.append(FNR)
    print("- LR done.")
    
    
    # Linear SVM C=0.1, pi_T=0.5
    print("Start SVM....")
    FPR_SVM = []
    TPR_SVM = []
    FNR_SVM = []
    pi_T_SVM = 0.5
    
    if not calibrated:
        C = [0.1]
        scores_SVM, LE_SVM = SVM.kfold_SVM(
                        k_gauss_PCA6, C, pi_T_SVM, prior,
                        K, mode='balanced-linear', getScores=True)
    else:
        C = 0.1
        scores_SVM, LE_SVM = SVM.kfold_SVM_actDCF_Calibrated(
                        k_gauss_PCA6, C, pi_T_SVM, prior,
                        K, mode='balanced-linear', getScores=True)
    scores_SVM = scores_SVM[0].flatten()
    print(scores_SVM.shape)
    sortedScores_SVM = numpy.sort(scores_SVM)
    
    for t in sortedScores_SVM:
        FPR, TPR, FNR = ev.computeFPR_TPR_FNR(scores_SVM, LE_SVM, t)
        FPR_SVM.append(FPR)
        TPR_SVM.append(TPR)
        FNR_SVM.append(FNR)
    print("- SVM done.")


    # ROC plot
    print("Plotting ROC ...")
    plt.plotROC(FPR_MVG, TPR_MVG,
                FPR_LR,  TPR_LR,
                FPR_SVM, TPR_SVM)
    # plt.plotDET(FPR_MVG, FNR_MVG,
    #             FPR_LR,  FNR_LR,
    #             FPR_SVM, FNR_SVM)
    print("- ROC plot done.")





if __name__ == '__main__':

    #-----------------#
    #  Data analysis  #
    #-----------------#
    D_Train, L_Train = load('data/Train.txt')
    #feature_analysis(D_Train, L_Train, 'Feature correlation')

    # Gaussianization to clear the outliers
    D_Gaussianization = prep.gaussianization(D_Train)
    #feature_analysis(D_Gaussianization, L_Train, 'Gaussianized features')

    
    #------------------------------#
    #  Model training with K-FOLD  #
    #------------------------------#
    K = 2  # k-fold 
    prior = [0.5, 0.9, 0.1]
    pi_t = [0.5, 0.9, 0.1]
    
    lambdas=numpy.logspace(-5,-5, 1)        #For Normal Use
    #lambdas=numpy.logspace(-5, 2, num=30)  #For Graphichs Use
    
    Cs = numpy.logspace(-1, -1, num=1)      #For Normal Use
    #Cs = numpy.logspace(-4, -1, num=10)    #For Graphichs Use
    
    nComponents = [8]                       #For Normal Use
    #nComponents = [2, 4, 8, 16, 32]        #For Graphichs Use
    
    # K-fold
    subsets = prep.kfold_computeAll(D_Train, D_Gaussianization, L_Train, K)
    
    # Single split 80-20
    splits = prep.single_split_computeAll(D_Train, D_Gaussianization, L_Train)
     
    # MVG 
    MVG_models(subsets, splits, prior, K)
    
    
    # LR
    LR_models(subsets, splits, prior, K , lambdas, pi_t=[0.5])
    
    
    # QLR
    LR_models(subsets, splits, prior, K, lambdas, quadratic=True, pi_t=[0.5])
    
    
    # SVM
    SVM_models(subsets, splits, prior, K, Cs, pi_t, 'linear')
    SVM_models(subsets, splits, prior, K, Cs, pi_t, 'balanced-linear')
    SVM_models(subsets, splits, prior, K, Cs, pi_t, 'poly')
    SVM_models(subsets, splits, prior, K, Cs, pi_t, 'RBF')
    
    
    # GMM    
    GMM_models(subsets, splits, prior, K, alpha=0.1, nComponents=nComponents, mode='full', psi=0.01)
    GMM_models(subsets, splits, prior, K, alpha=0.1, nComponents=nComponents, mode='diag', psi=0.01)
    GMM_models(subsets, splits, prior, K, alpha=0.1, nComponents=nComponents, mode='tied-full', psi=0.01)
    GMM_models(subsets, splits, prior, K, alpha=0.1, nComponents=nComponents, mode='tied-diag', psi=0.01)
    
        
    #-----------------------------#  
    #  Score calibration section  #
    #-----------------------------#
    
    # Evaluation with other metrics
    score_calibration(subsets, prior, K)
    
    
    # ROC and DET curve
    ROC(subsets, K, [0.5])
    
    

    ###########################
    #   EVALUATION SECTION    #
    ###########################
    
    D_Test, L_Test = load('data/Test.txt')
    #D_Train, L_Train = load('data/Train.txt') #Already Loaded butto re-member the names
    
    D_Gauss_Test =  prep.gaussianization(D_Train, D_Test)
    
    D_Gauss_Test_PCA6 = dim_red.PCA_On_Test(D_Train, D_Test,6) #Compute PCA Over TestSet
    D_Gauss_Train_PCA6 = dim_red.PCA(D_Train, 6) #Compute PCA Over TrainSet
    
    Split_Raw_Test = [[D_Train, L_Train], [D_Test, L_Test]]
    Split_GaussPCA_Test = [[D_Gauss_Train_PCA6, L_Train], [D_Gauss_Test_PCA6, L_Test]]

    
    # MVG
    print ("\n \n#----------Raw Data and MVG----------# \n")
    
    MVG.MVG_EVALUATION(Split_Raw_Test, prior, MVG.MVG_models['full'], 1e-4, "full")
    MVG.MVG_EVALUATION(Split_Raw_Test, prior, MVG.MVG_models['diag'], 1e-4, "diag")
    MVG.MVG_EVALUATION(Split_Raw_Test, prior, MVG.MVG_models['tied-full'], 1e-4, "tied-full")
    MVG.MVG_EVALUATION(Split_Raw_Test, prior, MVG.MVG_models['tied-diag'], 1e-4, "tied-diag")
    
    print ("\n \n#----------Gauss Data + PCA6 and MVG----------# \n")
    
    MVG.MVG_EVALUATION(Split_GaussPCA_Test, prior, MVG.MVG_models['full'], 1e-4, "full")
    MVG.MVG_EVALUATION(Split_GaussPCA_Test, prior, MVG.MVG_models['diag'], 1e-4, "diag")
    MVG.MVG_EVALUATION(Split_GaussPCA_Test, prior, MVG.MVG_models['tied-full'], 1e-4, "tied-full")
    MVG.MVG_EVALUATION(Split_GaussPCA_Test, prior, MVG.MVG_models['tied-diag'], 1e-4, "tied-diag")
    
    
    # LR
    print ("\n \n#----------Raw Data and LR----------# \n")
    
    LR.LR_EVALUATION(Split_Raw_Test, prior, 0.1, 1e-5, lambd_calib=1e-4)
    LR.LR_EVALUATION(Split_Raw_Test, prior, 0.5, 1e-5, lambd_calib=1e-4)
    LR.LR_EVALUATION(Split_Raw_Test, prior, 0.9, 1e-5, lambd_calib=1e-4)
    
    print ("\n \n#----------Gauss Data + PCA6 and LR----------# \n")
    
    LR.LR_EVALUATION(Split_GaussPCA_Test, prior, 0.1, 1e-5, lambd_calib=1e-4)
    LR.LR_EVALUATION(Split_GaussPCA_Test, prior, 0.5, 1e-5, lambd_calib=1e-4)
    LR.LR_EVALUATION(Split_GaussPCA_Test, prior, 0.9, 1e-5, lambd_calib=1e-4)
    
    
    # SVM
    print ("\n \n#----------Raw Data and SVM----------# \n")
    
    SVM.SVM_EVALUATION(Split_Raw_Test, C=0.1, pi_t=0.1, prior=prior, lambd_calib=1e-4, mode='linear')
    SVM.SVM_EVALUATION(Split_Raw_Test, C=0.1, pi_t=0.1, prior=prior, lambd_calib=1e-4, mode='balanced-linear')
    SVM.SVM_EVALUATION(Split_Raw_Test, C=0.1, pi_t=0.5, prior=prior, lambd_calib=1e-4, mode='balanced-linear')
    SVM.SVM_EVALUATION(Split_Raw_Test, C=0.1, pi_t=0.9, prior=prior, lambd_calib=1e-4, mode='balanced-linear')
    
    print ("\n \n#----------Gauss Data + PCA6 and SVM----------# \n")
    
    SVM.SVM_EVALUATION(Split_GaussPCA_Test, C=0.1, pi_t=0.1, prior=prior,lambd_calib=1e-4, mode='linear')
    SVM.SVM_EVALUATION(Split_GaussPCA_Test, C=0.1, pi_t=0.1, prior=prior,lambd_calib=1e-4, mode='balanced-linear')
    SVM.SVM_EVALUATION(Split_GaussPCA_Test, C=0.1, pi_t=0.5, prior=prior,lambd_calib=1e-4, mode='balanced-linear')
    SVM.SVM_EVALUATION(Split_GaussPCA_Test, C=0.1, pi_t=0.9, prior=prior,lambd_calib=1e-4, mode='balanced-linear')
    
    
    
    # GMM    
    print ("\n \n#----------Raw Data and GMM----------# \n")
    
    GMM.GMM_EVALUATION(Split_Raw_Test, prior, mode="full", alpha=0.1, Component=2, psi=0.01, lambd_calib=1e-4)
    GMM.GMM_EVALUATION(Split_Raw_Test, prior, mode="diag", alpha=0.1, Component=4, psi=0.01, lambd_calib=1e-4)
    GMM.GMM_EVALUATION(Split_Raw_Test, prior, mode="tied-full", alpha=0.1, Component=8, psi=0.01, lambd_calib=1e-4)
    GMM.GMM_EVALUATION(Split_Raw_Test, prior, mode="tied-diag", alpha=0.1, Component=16, psi=0.01, lambd_calib=1e-4)
    
    print ("\n \n#----------Gauss Data + PCA6 and GMM----------# \n")
    
    GMM.GMM_EVALUATION(Split_GaussPCA_Test, prior, mode="full", alpha=0.1, Component=2, psi=0.01, lambd_calib=1e-4)
    GMM.GMM_EVALUATION(Split_GaussPCA_Test, prior, mode="diag", alpha=0.1, Component=4, psi=0.01, lambd_calib=1e-4)
    GMM.GMM_EVALUATION(Split_GaussPCA_Test, prior, mode="tied-full", alpha=0.1, Component=8, psi=0.01, lambd_calib=1e-4)
    GMM.GMM_EVALUATION(Split_GaussPCA_Test, prior, mode="tied-diag", alpha=0.1, Component=16, psi=0.01, lambd_calib=1e-4)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
