import numpy as np
import matplotlib.pyplot as plt
import random
import numpy
import pylab
import lib.model_evaluation as ev

hFea = {
    0: 'Mean of the integrated profile',
    1: 'Standard deviation of the integrated profile',
    2: 'Excess kurtosis of the integrated profile',
    3: 'Skewness of the integrated profil',
    4: 'Mean of the DM-SNR curve',
    5: 'Standard deviation of the DM-SNR curve',
    6: 'Excess kurtosis of the DM-SNR curve',
    7: 'Skewness of the DM-SNR curve'
}


def plot_hist(D, L, title='', showClass=True):
    '''
    Plot Histograms

    Parameters
    ----------
    D : Matrix of features.
    L : Array of labels.
    bi : Number of bins to use in plot, default equal to 10.

    Returns
    -------
    None.

    '''
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for idx in range(D.shape[0]):
        plt.figure()
        plt.suptitle(title, fontsize=16)
        plt.xlabel(hFea[idx])
        if showClass:
            plt.hist(D0[idx, :], bins=70, histtype='stepfilled',
                     density=True, label='Non-Pulsar', color='blue', alpha=0.5)
            plt.hist(D1[idx, :], bins=70, histtype='stepfilled',
                     density=True, label='Pulsar', color='red', alpha=0.5)
        else:
            plt.hist(D[idx, :], bins=70, histtype='stepfilled',
                     density=True, label='Dataset', color='green', alpha=0.5)
        plt.legend()
        plt.tight_layout()

    plt.show()


def plot_scatter(D, L):
    '''
    Plot Scatter

    Parameters
    ----------
    D : Matrix of features.
    L : Array of labels.

    Returns
    -------
    None.

    '''

    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for idx1 in range(D.shape[0]):
        for idx2 in range(D.shape[0]):
            if idx1 == idx2:
                continue
            plt.figure()
            plt.xlabel(hFea[idx1])
            plt.ylabel(hFea[idx2])
            plt.scatter(D0[idx1, :], D0[idx2, :],
                        label='Non-Pulsar', color='blue', linewidths=1)
            plt.scatter(D1[idx1, :], D1[idx2, :],
                        label='Pulsar', color='red', linewidths=1)
            plt.legend()
            plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
            #plt.savefig('scatter_%d_%d.pdf' % (idx1, idx2))
        plt.show()


def plot_scatter_matrix(D, L, subtitles=True):

    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    fig, axs = plt.subplots(D.shape[0], D.shape[0])
    for idx1 in range(D.shape[0]):
        for idx2 in range(D.shape[0]):
            if idx1 == idx2:
                axs[idx1, idx2].hist(
                    D0[idx1, :], bins=70, histtype='stepfilled', label='Non-Pulsar', color='blue', alpha=0.5)
                axs[idx1, idx2].hist(
                    D1[idx1, :], bins=70, histtype='stepfilled', label='Pulsar', color='red', alpha=0.5)
                continue
            axs[idx1, idx2].scatter(
                D0[idx1, :], D0[idx2, :], label='Non-Pulsar', color='blue')
            axs[idx1, idx2].scatter(
                D1[idx1, :], D1[idx2, :], label='Pulsar', color='red')
            axs[idx1, idx2].set(xlabel=hFea[idx1], ylabel=hFea[idx2])
    if subtitles:
        plt.legend()
    plt.tight_layout()
    plt.rcParams["figure.figsize"] = (50, 50)
    plt.show()


def plot_heatmap(D, title=''):
    L = hFea.values()
    fig, ax = plt.subplots()
    im = ax.imshow(D)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(L)))
    ax.set_yticks(np.arange(len(L)))

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(len(L)):
        for j in range(len(L)):
            text = ax.text(j, i, D[i, j], ha="center", va="center", color="white")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_DCF(x, y, xlabel,base=10, title=''):
    if title=='':
        name= (str(random.uniform(0, 10))+".png")
    else: name = title
    print(name)
    plt.figure()
    #plt.suptitle(title, fontsize=16)
    plt.plot(x, y[0], label='min DCF prior=0.5', color='b')
    plt.plot(x, y[1], label='min DCF prior=0.9', color='r')
    plt.plot(x, y[2], label='min DCF prior=0.1', color='g')
    plt.xlim([min(x), max(x)])
    plt.xscale("log", base=base)
    plt.legend(["min DCF prior=0.5", "min DCF prior=0.9", "min DCF prior=0.1"],loc='lower right')
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.savefig(name)
    return



def computeBayesErrorPlot(TrueLabel_1 ,llRateos_1 ,CostMatrix ,Num=21 ,TrueLabel_2=None ,llRateos_2=None):
    '''
    Compute the Bayes Error Plot for one or two Models. In the first case the function takes a maximum of 4 values, in the second 
    case the function takes 6 vslues. The red and blue lines are for the first model, the yellow and green lines for the second model.

    Parameters
    ----------
    TrueLabels : Type: Numpy Array 
                 Description: Array of correct labels.
    llRateos_1 : Type: Numpy Array 
                 Description: Array of LogLikelihood Rateos.
    CostMatrix : Type: Numpy Array 
                 Description: Matrix of costs. Depends from the application.
    Num :        Type: Int
                 Description: Number of samples into Numpy Linspace
    TrueLabel_2 : Type: Numpy Array 
                  Description: Array of correct labels.
    llRateos_2 : Type: Numpy Array 
                 Description: Array of LogLikelihood Rateos.

    Returns
    -------
    None.

    '''    
    
    effPriorLogOdds = numpy.linspace(-3, 3, Num)
    PlotLegend=[]
    
    # Create 2 array of 0 that will contain the DCF and minDCF values from the first Model
    actdcf_1=numpy.zeros(Num)
    mindcf_1=numpy.zeros(Num)
    
    # Check if the user provides 1 model to plot or 2 models
    if ( llRateos_2 is None and TrueLabel_2 is None) :
        
        for i,p in enumerate(effPriorLogOdds):
            
           # Define the threshold using the value p from the effPriorLogOdds and assign a label using this t
           pi=1/(1+numpy.exp(-p))
           t=-numpy.log(((pi*CostMatrix[0,1])/((1-pi)*CostMatrix[1,0])))
           PredLabel_1=numpy.int32(llRateos_1>t)

           # Compute the DCF Normalized and minDCF
           actdcf_1[i]=ev.computeActualDCF(TrueLabel_1, llRateos_1,pi,PredLabel_1)
           mindcf_1[i]=ev.computeMinDCF(TrueLabel_1, llRateos_1 ,pi,CostMatrix)
           
           # Create the plot Legend
           PlotLegend=["Act DCF Model1", "minDCF Model1"]
    else:
        
        # Create other 2 array of 0 that will contain the DCF and minDCF values from the secon Model
        actdcf_2=numpy.zeros(Num)
        mindcf_2=numpy.zeros(Num)
        
        for i,p in enumerate(effPriorLogOdds):
            
           # Define the threshold using the value p from the effPriorLogOdds and assign a label using this t
           pi=1/(1+numpy.exp(-p))
           t=-numpy.log(((pi*CostMatrix[0,1])/((1-pi)*CostMatrix[1,0])))
           PredLabel_1=numpy.int32(llRateos_1>t)
           PredLabel_2=numpy.int32(llRateos_2>t)

           # Compute the DCF Normalized and minDCF for both Models
           actdcf_1[i]=ev.computeActualDCF(TrueLabel_1, llRateos_1,pi,PredLabel_1)
           mindcf_1[i]=ev.computeMinDCF(TrueLabel_1, llRateos_1 , pi, CostMatrix)
           actdcf_2[i]=ev.computeActualDCF(TrueLabel_2, llRateos_2, pi, PredLabel_2)
           mindcf_2[i]=ev.computeMinDCF(TrueLabel_2, llRateos_2 , pi, CostMatrix)
           
        # Plot the DCF and minDCF for the secon Model
        pylab.plot(effPriorLogOdds, actdcf_2, label='Act DCF', color='g') 
        pylab.plot(effPriorLogOdds, mindcf_2, label='min DCF', color='y') 
        
        # Create the plot Legend
        PlotLegend=["Act DCF Model2", "minDCF Model2","Act DCF Model1", "minDCF Model1"]
        
    # Plot the DCF and minDCF for the first Model. This part of code is out of if statement
    # Because it will be executed in both of cases
    pylab.plot(effPriorLogOdds, actdcf_1, label='DCF', color='r') 
    pylab.plot(effPriorLogOdds, mindcf_1, label='min DCF', color='b') 
    pylab.legend(PlotLegend)
    pylab.ylim([0, 1.1])
    pylab.xlim([-3, 3])


