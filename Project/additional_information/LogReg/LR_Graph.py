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

########################
#   MODELS TRAINING    #
#      Pulsar=1        #
#     NonPulsar=0      #
#  Pi_T->ClassLabel=1  #
########################

# Discriminative models
def LR_models(subsets, splits, prior, K, lambdas, quadratic=False, pi_t=0.5):
    
    def kfold_LR_compute(k_subsets, K, lambdas, prior , pi_t, f):   
        minDCF_LR = []
        
        for p in prior :
            minDCF_values,_ = f(k_subsets, K, lambdas, p, pi_t)
            minDCF_LR.append(minDCF_values)
        #plt.plot_DCF(lambdas, minDCF_LR, "λ")
        #print (numpy.around(minDCF_LR, 3)) # rounded
        
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
        
            
    else:
        print('####  Quadratic LR  ####\n')
        
        print('------- K FOLD -------')
        k_subsets, k_subsets_PCA7, k_subsets_PCA6, k_gauss_subsets, k_gauss_PCA7_subs, k_gauss_PCA6_subs = subsets 
        
        print('\nTraining dataset')
        kfold_LR_compute(k_subsets, K, lambdas, prior, pi_t, LR.kfold_QuadLogReg)
        
        print('\nGaussianized dataset')
        kfold_LR_compute(k_gauss_subsets, K, lambdas, prior, pi_t, LR.kfold_QuadLogReg)

    
     
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
    
    lambdas=numpy.logspace(-5,-5,1)         #For Normal Use
    #lambdas=numpy.logspace(-5, 2, num=20)  #For Graphichs Use
    
    Cs = numpy.logspace(-2, -2, num=1)  #For Normal Use
    #Cs = numpy.logspace(-4, -1, num=20) #For Graphichs Use
    
    nComponents = [8] #For Normal Use
    nComponents = [2, 4, 8, 16, 32] #For Normal Use
    
    # K-fold
    subsets = prep.kfold_computeAll(D_Train, D_Gaussianization, L_Train, K)
    
    # Single split 80-20
    splits = prep.single_split_computeAll(D_Train, D_Gaussianization, L_Train)
    
    
    
    # MVG 
    #MVG_models(subsets, splits, prior, K)
    
    # LR
    print("------------LR pi_t 0.5---------------")
    LR_models(subsets, splits, prior, K , lambdas, pi_t=0.5)
    
    print("------------LR pi_t 0.1---------------")
    LR_models(subsets, splits, prior, K , lambdas, pi_t=0.1)
    
    print("------------LR pi_t 0.9---------------")
    LR_models(subsets, splits, prior, K , lambdas, pi_t=0.9)
    
    # QLR
    LR_models(subsets, splits, prior, K, lambdas, quadratic=True, pi_t=0.5)
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
