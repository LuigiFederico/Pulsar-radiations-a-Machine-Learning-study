# Last edit: 14/05/2022 - Luigi

import matplotlib.pyplot as plt

hFea = {
    0:'Mean of the integrated profile',
    1:'Standard deviation of the integrated profile',
    2:'Excess kurtosis of the integrated profile',
    3:'Skewness of the integrated profil',
    4:'Mean of the DM-SNR curve',
    5:'Standard deviation of the DM-SNR curve',
    6:'Excess kurtosis of the DM-SNR curve',
    7:'Skewness of the DM-SNR curve'
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
    D0 = D[:, L==0]
    D1 = D[:, L==1]
     
    for idx in range(D.shape[0]):
       plt.figure()
       plt.suptitle(title, fontsize=16)
       plt.xlabel(hFea[idx])
       if showClass:
           plt.hist(D0[idx, :], bins=50, histtype='stepfilled', density=True, label='Non-Pulsar', color='blue', alpha=0.4)
           plt.hist(D1[idx, :], bins=50, histtype='stepfilled', density=True, label='Pulsar', color='red', alpha=0.4)
       else:
           plt.hist(D[idx, :], bins=50, histtype='stepfilled', density=True, label='Dataset', color='green', alpha=0.4)
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
            plt.scatter(D0[idx1, :], D0[idx2, :], label='Non-Pulsar', color='blue', linewidths=1)
            plt.scatter(D1[idx1, :], D1[idx2, :], label='Pulsar', color='red', linewidths=1)
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
            if idx1==idx2:
                axs[idx1, idx2].hist(D0[idx1, :], bins=70, histtype='stepfilled', label='Non-Pulsar', color='blue', alpha=0.5)
                axs[idx1, idx2].hist(D1[idx1, :], bins=70, histtype='stepfilled', label='Pulsar', color='red', alpha=0.5)
                continue    
            axs[idx1, idx2].scatter(D0[idx1, :], D0[idx2, :], label='Non-Pulsar', color='blue')
            axs[idx1, idx2].scatter(D1[idx1, :], D1[idx2, :], label='Pulsar', color='red')
            axs[idx1, idx2].set(xlabel = hFea[idx1], ylabel=hFea[idx2])
    if subtitles:
        plt.legend()
    plt.tight_layout()
    plt.rcParams["figure.figsize"] = (50, 50)
    plt.show() 



