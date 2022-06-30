import numpy
import lib.plots as plt
import lib.dim_reduction as dim_red
import lib.data_preparation as prep
import lib.MVG_models as MVG
import lib.LR_models as LR
import lib.SVM_models as SVM


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
########################

# Generative models      
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

    
# Discriminative models
def LR_models(subsets, splits, prior, K, quadratic=False):
    
    def kfold_LR_compute(k_subsets, K, prior, f):   
        minDCF_LR = []
        lambdas = []
        
        for p in prior :
            minDCF_values, lambdas = f(k_subsets, K, p)
            minDCF_LR.append(minDCF_values)
        #plt.plot_DCF(lambdas, minDCF_LR, "λ")
        print (numpy.around(minDCF_LR, 3)) # rounded
        
        return minDCF_LR
    
    
    def single_split_LR_compute(split, prior, f):
        minDCF_LR = [] 
        lambdas=[]
        
        for p in prior :
            minDCF_values, lambdas = f( split , p)
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
            kfold_LR_compute(subsets[i], K, prior, LR.kfold_LogReg)
        
        print('------- SINGLE SPLIT -------')
        for (i,TrainLabel) in enumerate(TrainMod):
            print('\n',TrainLabel)
            single_split_LR_compute(splits[i], prior, LR.single_split_LogReg)
            
    else:
        print('####  Quadratic LR  ####\n')
        
        print('------- K FOLD -------')
        k_subsets, k_subsets_PCA7, k_subsets_PCA6, k_gauss_subsets, k_gauss_PCA7_subs, k_gauss_PCA6_subs = subsets 
        
        print('\nTraining dataset')
        kfold_LR_compute(k_subsets, K, prior, LR.kfold_QuadLogReg)
        
        print('\nGaussianized dataset')
        kfold_LR_compute(k_gauss_subsets, K, prior, LR.kfold_QuadLogReg)
        
        print('------- SINGLE SPLIT -------')
        train_split, train_PCA7_split, train_PCA6_split, gauss_split, gauss_PCA7_split, gauss_PCA6_split = splits
        
        print('\nTraining dataset')
        single_split_LR_compute(train_split, prior, LR.single_plit_QuadLogReg)
        
        print('\nGaussianized dataset')
        single_split_LR_compute(gauss_split, prior, LR.single_split_QuadLogReg)
    
        
    
def single_split_SVM(split, prior, f):
    
    minDCF_SVM = []
    Cs=[]
    
    for p in prior :
        minDCF_values, Cs = f( split , p)
        minDCF_SVM.append(minDCF_values)
    
    #plt.plot_DCF(Cs, minDCF_SVM, "C")
    print (numpy.around(minDCF_SVM, 3)) # rounded
    
    return minDCF_SVM
    

def SVM_models(subsets, splits, prior, K):
    
    print('########   SVM   ########\n')
    
    print('------- SINGLE SPLIT -------')
    train_split, train_PCA7_split, train_PCA6_split, gauss_split, gauss_PCA7_split, gauss_PCA6_split = splits
    
    print('\nTraining dataset')
    single_split_SVM(train_split, prior, SVM.single_split_SVM)
    
    print('\nGaussianized dataset')
    single_split_SVM(gauss_split, prior, SVM.single_split_SVM)
    
    
       

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
    pi_T = [0.5, 0.9, 0.1]
    
    # K-fold
    subsets = prep.kfold_computeAll(D_Train, D_Gaussianization, L_Train, K)
    
    # Single split 80-20
    splits = prep.single_split_computeAll(D_Train, D_Gaussianization, L_Train)
    
    # MVG 
    #MVG_models(subsets, splits, prior, K)
    
    # LR
    LR_models(subsets, splits, prior, K)
    
    # QLR
    #LR_models(subsets, splits, prior, K, quadratic=True)
    
    # SVM
    #SVM_models(subsets, splits, prior, K)
    
    
    # GMM
    # Without gaussianization
    # With gaussianization
   


    
        
    #----------------------------------#  
    #  Choice of the candidate models  #
    #----------------------------------#
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
