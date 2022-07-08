import numpy
import lib.plots as plt
import lib.dim_reduction as dim_red
import lib.data_preparation as prep
import lib.MVG_models as MVG
import lib.LR_models as LR
import lib.SVM_models as SVM
import lib.GMM_models as GMM


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
            for p in prior:
                print("- %s, prior = %.2f, minDCF = %.3f" % (k, p, minDCF))
        print()
        

    def single_split_MVG_compute(split, prior, title=''):
        
        print(title)
        for k in MVG.MVG_models.keys():
            minDCF = numpy.round( MVG.single_split_MVG(split, prior, MVG.MVG_models[k]), 3)
            for p in prior:
                print("- %s, prior = %.2f, minDCF = %.3f" % (k, p, minDCF))
        print()
    
    TrainMod = ['Training dataset',
                'Training dataset + PCA m = 7',
                'Training dataset + PCA m = 6',
                'Gaussianized dataset',
                'Gaussianized dataset + PCA m = 7',
                'Gaussianized dataset + PCA m = 6']

    print('########   MVG   ########\n')
    
    print('------- %d-FOLD -------\n' % K)
    for (i, TrainLabel) in enumerate(TrainMod):
        kfold_MVG_compute(subsets[i], K, prior, TrainLabel)
     
    print('----- SINGLE SPLIT -----\n')
    for(i, TrainLabel) in enumerate(TrainMod):
        single_split_MVG_compute(splits[i], prior, TrainLabel)

    
def LR_models(subsets, splits, prior, K, lambdas, quadratic=False, pi_t=0.5):
    
    def kfold_LR_compute(k_subsets, K, lambdas, prior , pi_t, f):   
        minDCF_LR = []
        
        for pi_T in pi_t:
            for p in prior :
                minDCF_values,_ = f(k_subsets, K, lambdas, p, pi_T)
                minDCF_LR.append(minDCF_values)
            #plt.plot_DCF(lambdas, minDCF_LR, "λ")
            print (numpy.around(minDCF_LR, 3)) # rounded
        
        return minDCF_LR
    
    
    def single_split_LR_compute(split, lambdas, prior, pi_t, f):
        minDCF_LR = [] 
        for pi_T in pi_t:
            for p in prior :
                minDCF_values,_  = f( split , lambdas, p, pi_T)
                minDCF_LR.append(minDCF_values)
            #plt.plot_DCF(lambdas, minDCF_LR, "λ")
            print (numpy.around(minDCF_LR, 3)) # rounded
        
        return minDCF_LR    
    
    
    TrainMod=["Training dataset",
              "Training dataset with PCA m = 7",
              "Training dataset with PCA m = 6",
              "Gaussianized dataset","Gaussianized with PCA m = 7",
              "Gaussianized with PCA m = 6"]    
        
    if quadratic == False:
        print('########   LR   ########\n')
          
        print('-------  %d-FOLD  -------\n' % K)
        for (i, TrainLabel) in enumerate(TrainMod):
            print('\n',TrainLabel)
            kfold_LR_compute(subsets[i], K, lambdas, prior, pi_t, LR.kfold_LogReg)
        
        print('------- SINGLE SPLIT -------')
        for (i,TrainLabel) in enumerate(TrainMod):
            print('\n',TrainLabel)
            single_split_LR_compute(splits[i], lambdas, prior, pi_t, LR.single_split_LogReg)
            
    else:
        print('####  Quadratic LR  ####\n')
        
        print('------- K FOLD -------')
        k_subsets, k_subsets_PCA7, k_subsets_PCA6, k_gauss_subsets, k_gauss_PCA7_subs, k_gauss_PCA6_subs = subsets 
        
        print('\nTraining dataset')
        kfold_LR_compute(k_subsets, K, lambdas, prior, pi_t, LR.kfold_QuadLogReg)
        
        print('\nGaussianized dataset')
        kfold_LR_compute(k_gauss_subsets, K, lambdas, prior, pi_t, LR.kfold_QuadLogReg)
        
        print('------- SINGLE SPLIT -------')
        train_split, train_PCA7_split, train_PCA6_split, gauss_split, gauss_PCA7_split, gauss_PCA6_split = splits
        
        print('\nTraining dataset')
        single_split_LR_compute(train_split, lambdas, prior, pi_t, LR.single_plit_QuadLogReg)
        
        print('\nGaussianized dataset')
        single_split_LR_compute(gauss_split, lambdas, prior, pi_t, LR.single_split_QuadLogReg)
    
        
def SVM_models(subsets, splits, prior, K, Cs, pi_t=0.5, mode='linear'):
    
    def single_split_SVM(split, prior, Cs, pi_t, f, mode):
        minDCF_SVM = []
        
        if mode == 'balanced-linear':
            for pi_T in pi_t:
                for p in prior:
                    minDCF_values, LabelPredicetd = f(split, Cs, pi_T, p, mode)
            print (numpy.around(minDCF_SVM, 3)) # rounded
        else:
            for p in prior :
                minDCF_values, LabelPredicetd = f(split, Cs, pi_t, p, mode)
                minDCF_SVM.append(minDCF_values)
            print (numpy.around(minDCF_SVM, 3)) # rounded
        
        return minDCF_SVM
    
    
    def kfold_SVM(k_subsets, prior, K, Cs, pi_t, f, mode='linear', title=''):
        minDCF_SVM = []
        
        if mode == 'balanced-linear':
            for pi_T in pi_t:
                for p in prior:
                    minDCF_values, LabelPredicetd = f(k_subsets, Cs, pi_T, p, K, mode)
                    minDCF_SVM.append(minDCF_values)
            print (numpy.around(minDCF_SVM, 3)) # rounded
        
        elif mode=='poly': # prior=0.5
            c_vec = [0, 1, 10, 30]
            for c in c_vec:
                minDCF_values, LabelPredicted = f(k_subsets, Cs, c=c, mode=mode)
                minDCF_SVM.append(minDCF_values)
            plt.plot_DCF(Cs, minDCF_SVM, "C", title)
            print (numpy.around(minDCF_SVM), 3)

        elif mode=='RBF':
            gammas=numpy.logspace(-3,-1,3) # prior=0.5
            for gamma in gammas:
                minDCF_values, LabelPredicetd = f(k_subsets, Cs, pi_t, 0.5, K, mode, gamma)
                minDCF_SVM.append(minDCF_values)
            print (numpy.around(minDCF_SVM, 3)) # rounded
            plt.plot_DCF(Cs, minDCF_SVM, "C")
            print (numpy.around(minDCF_SVM), 3)
            
            return minDCF_SVM
            
                    
        else:            
            for p in prior:
                minDCF_values, LabelPredicetd = f(k_subsets, Cs, pi_t, p, K, mode)
                minDCF_SVM.append(minDCF_values)
            print (numpy.around(minDCF_SVM, 3)) # rounded
        
        return minDCF_SVM
    
    
    print('########  %s SVM  ########\n' % mode)
    
    # print('------- SINGLE SPLIT -------')
    # train_split, _, _, gauss_split, _, _ = splits
    
    # print('\nTraining dataset')
    # single_split_SVM(train_split, prior, Cs, pi_t, SVM.single_split_SVM, mode)
    
    # print('\nGaussianized dataset')
    # single_split_SVM(gauss_split, prior, Cs, pi_t, SVM.single_split_SVM, mode)
    
    
    print('-------  %d-FOLD  -------\n' % K)
    
    k_subsets, k_subsets_PCA7, k_subsets_PCA6, k_gauss_subsets, k_gauss_PCA7_subs, k_gauss_PCA6_subs = subsets    
    
    print('\nTraining dataset')
    kfold_SVM(k_subsets, prior, K, Cs, pi_t, SVM.kfold_SVM, mode, title="SVM_Raw_prior5")
    
    # print('\nGaussianized dataset')
    # kfold_SVM(k_guass, prior, K, Cs, pi_t, SVM.kfold_SVM, mode)
    
    print('\nGaussianized dataset + PCA m=6')
    kfold_SVM(k_gauss_PCA6_subs, prior, K, Cs, pi_t, SVM.kfold_SVM, mode, title="SVM_GaussPCA6_prior5")
    
  
def GMM_models(subsets, splits, prior, K, alpha, nComponents, mode='full', psi=0.01 ):
    
    def single_split_GMM(split, prior, alpha, nComponents, mode='full', psi=0.01):

        minDCF_GMM = []
        for p in prior:
            minDCF = GMM.single_split_GMM(split, p, 
                                          mode, 
                                          alpha, nComponents, psi)
            minDCF_GMM.append(minDCF)
        print (numpy.around(minDCF_GMM, 3)) # rounded
        #plt.plot_DCF(nComponents, minDCF_GMM, "GMM Components",2)
    
    def kfold_GMM(subset, K, prior, alpha, nComponents, mode='full', psi=0.01, title=''):
        
        minDCF_GMM = []
        for p in prior:
            minDCF = GMM.kfold_GMM(subset, K, p,
                                   mode,
                                   alpha, nComponents, psi)
            minDCF_GMM.append(minDCF)
        print (numpy.around(minDCF_GMM, 3)) # rounded
        plt.plot_DCF(nComponents, minDCF_GMM, 
                     "GMM Components", 2, title)        
        

    print('########  GMM - %s-cov  ########\n' % mode)
     
    # print('------- SINGLE SPLIT -------')
    # train_split, _, _, gauss_split, _, _ = splits
    
    # print('\nTraining dataset')    
    # single_split_GMM(train_split, prior, alpha, nComponents, mode, psi)
    
    # print('\nGaussianized dataset')    
    # single_split_GMM(gauss_split, prior, alpha, nComponents, mode, psi)
    
    print('-------  %d-FOLD  -------\n' % K)
    k_subset, _, _, k_guass, _, _ = subsets    
    
    print('\nTraining dataset')    
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
    #print("Start MVG Tied Full-Cov 5-folds on raw features...")
    #actDCF_MVG = MVG.kfold_MVG_actDCF(k_raw, K, prior, MVG.MVG_models['tied-full'])
    #print("End")
    
    actualDCFs=[]
    minDCFs=[]
    #for i in range(numberOfPoints):
      #actDCF, minDCF = MVG.kfold_MVG_actDCF(k_raw, K, [effPriors[i]], MVG.MVG_models['tied-full'])
      #actDCF=actDCF[0]
      #minDCF=minDCF[0]
      #actualDCFs.append(actDCF)
      #minDCFs.append(minDCF)
      #print("At iteration", i, "the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
    #plt.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "MVG Tied Full-Cov")
    
    
    ##------------------------------------------------##
    
    
    # # LogReg (lambda=1e-5, pi_T=0.1) - Raw

    #print("Start LogReg (λ=1e-5, pi_T=0.1) 5-folds on raw features...")
    #actDCF_LR = LR.kfold_LogReg_actDCF(k_raw, K, lambd, prior, pi_T)
    #print("End")
    
    #actualDCFs=[]
    #minDCFs=[]
    #for i in range(numberOfPoints):
      #actDCF, minDCF = LR.kfold_LogReg_actDCF(k_raw, K, lambd, [effPriors[i]], pi_T_LogReg)
      #actDCF=actDCF[0]
      #minDCF=minDCF[0]
      #actualDCFs.append(actDCF)
      #minDCFs.append(minDCF)
      #print("At iteration", i, "the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
    #plt.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "LogReg")
    
    
    ##------------------------------------------------##
        
    
    # linear SVM (C=0.1, pi_T=0.5) - gauss PCA6

    #print("Start Linear SVM (C=0.1, pi_T=0.5) 5-folds on gaussianized features with PCA m=6...")
    #actDCF_SVM, minDCF_SVM = SVM.kfold_SVM_actDCF(k_gauss_PCA6, C, pi_T, prior, K, mode='balanced-linear')
    #print("End")
    
    #actualDCFs=[]
    #minDCFs=[]
    #for i in range(numberOfPoints):
      #actDCF, minDCF = SVM.kfold_SVM_actDCF(k_gauss_PCA6, C, pi_T, [effPriors[i]], K, mode='balanced-linear')
      #actDCF=actDCF[0]
      #minDCF=minDCF[0]
      #actualDCFs.append(actDCF)
      #minDCFs.append(minDCF)
      #print("At iteration", i, "the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
    #plt.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "Linear-SVM")
    
    
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
    K = 5  # k-fold 
    prior = [0.5, 0.9, 0.1]
    pi_t = [0.5, 0.9, 0.1]
    
    lambdas=numpy.logspace(-5,-5, 1)         #For Normal Use
    #lambdas=numpy.logspace(-5, 2, num=30)  #For Graphichs Use
    
    Cs = numpy.logspace(-1, -1, num=1)  #For Normal Use
    #Cs = numpy.logspace(-4, -1, num=10) #For Graphichs Use
    
    nComponents = [8] #For Normal Use
    nComponents = [2, 4, 8, 16, 32] #For Normal Use
    
    # K-fold
    subsets = prep.kfold_computeAll(D_Train, D_Gaussianization, L_Train, K)
    
    # Single split 80-20
    splits = prep.single_split_computeAll(D_Train, D_Gaussianization, L_Train)
     
    # MVG 
    #MVG_models(subsets, splits, prior, K)
    
    # LR
    #LR_models(subsets, splits, prior, K , lambdas, pi_t=0.5)
    
    # QLR
    #LR_models(subsets, splits, prior, K, lambdas, quadratic=True, pi_t=0.5)
    
    # SVM
    #SVM_models(subsets, splits, prior, K, Cs, pi_t, 'linear')
    #SVM_models(subsets, splits, prior, K, Cs, pi_t, 'balanced-linear')
    #SVM_models(subsets, splits, prior, K, Cs, pi_t, 'poly')
    #SVM_models(subsets, splits, prior, K, Cs, pi_t, 'RBF')

    
    # GMM    
    #GMM_models(subsets, splits, prior, K, alpha=0.1, nComponents=nComponents, mode='full', psi=0.01)
    #GMM_models(subsets, splits, prior, K, alpha=0.1, nComponents=nComponents, mode='diag', psi=0.01)
    #GMM_models(subsets, splits, prior, K, alpha=0.1, nComponents=nComponents, mode='tied-full', psi=0.01)
    #GMM_models(subsets, splits, prior, K, alpha=0.1, nComponents=nComponents, mode='tied-diag', psi=0.01)
    
        
    #----------------------------------#  
    #  Choice of the candidate models  #
    #----------------------------------#
    
    # Score calibration
    score_calibration(subsets, prior, K)
    
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
