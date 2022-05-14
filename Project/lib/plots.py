# Last edit: 10/05/2022 - Alex

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


def plot_hist(D, L, bi=10, title=''):
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
       #plt.hist(D[idx, :], bins=bi, density=True, alpha=0.4, label='Dataset')
       plt.hist(D0[idx, :], bins=bi, density=True, alpha=0.4, label='Non-Pulsar')
       plt.hist(D1[idx, :], bins=bi, density=True, alpha=0.4, label='Pulsar')
       plt.legend()
       plt.suptitle("Number of Bins:" + str(bi))
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
                continue  #salto le coppie x1==x2
            plt.figure()
            plt.xlabel(hFea[idx1])
            plt.ylabel(hFea[idx2])
            plt.scatter(D0[idx1, :], D0[idx2, :], label='Non-Pulsar')
            plt.scatter(D1[idx1, :], D1[idx2, :], label='Pulsar')

            plt.legend()
            plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
            #plt.savefig('scatter_%d_%d.pdf' % (idx1, idx2))
        plt.show()
