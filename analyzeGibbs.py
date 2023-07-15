#%%
import numpy as np
from matplotlib import pyplot as plt
import tqdm
import timeit

from SERSGenerator import SERSGenerator
from SERSDataset import load_dataset
from plot_map import *
from nmfGibbs import *


def plot_probabilities(probs, burnin=0.1, figsize=(15,5), inset=True, plot_prior=False):
    M = len(probs['log_likelihood'])
    zoom_offset = int(burnin*M)
    
    ### Log Likelihood
    fig, ax = plt.subplots(1,1, figsize=figsize)
    y = probs['log_likelihood']
    ax.plot(y, color='C0')
    ax.set_title(r'Log-likelihood: $\log P(X|A,B,\sigma^2)$', fontsize=20)
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_xticks(np.arange(0,M+1,500))
    ax.set_xlim(right=M)
    if inset:
        s = 0.143
        axin = ax.inset_axes([s, 0.1, 1-s, 0.63])
        axin.plot(y, color='C0')
        # Zoom in on the noisy data in the inset axis
        axin.set_xlim([zoom_offset, M])
        axin.set_ylim(bottom=min(y[zoom_offset:]), top=max(y[zoom_offset:]))
        axin.set_xticks([])
        #axin.set_yticks([])
        ax.indicate_inset_zoom(axin)
    plt.tight_layout()
    plt.show()
    
    ### Log Posterior
    fig, ax = plt.subplots(1,1, figsize=figsize)
    y = probs['log_posterior']
    ax.plot(y, color='C1')
    ax.set_title(r'Log-posterior: $\log P(A,B,\sigma^2 | X)$', fontsize=20)
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_xticks(np.arange(0,M+1,500))
    ax.set_xlim(right=M)
    if inset:
        s = 0.143
        axin = ax.inset_axes([s, 0.1, 1-s, 0.63])
        axin.plot(y, color='C1')
        # Zoom in on the noisy data in the inset axis
        axin.set_xlim([zoom_offset, M])
        axin.set_ylim(bottom=min(y[zoom_offset:]), top=max(y[zoom_offset:]))
        axin.set_xticks([])
        #axin.set_yticks([])
        ax.indicate_inset_zoom(axin)
    plt.tight_layout()
    plt.show()
    
    if plot_prior:
        fig, axes = plt.subplots(1,3, figsize=figsize)
        axes[0].plot(probs['log_prior_A'], color='C2')
        axes[0].set(title=r'$\log P(A)$')
        axes[1].plot(probs['log_prior_B'], color='C5')
        axes[1].set(title=r'$\log P(B)$')
        axes[2].plot(probs['log_prior_sigma2'], color='C4')
        axes[2].set(title=r'$\log P(\sigma^2)$')
        plt.tight_layout()
        plt.show()
    return


def MSE_reconstruction(X, trace, burnin=0.1):
    M = trace['AB'].shape[2]
    start = int(burnin*M)
    ABs = trace['AB'][:,:, start:]
    X_hat = np.mean(ABs, axis=2)
    mse = np.mean((X-X_hat)**2)
    return mse, X_hat


def compare_spectrum(i, X, X_hat, AB_perc5, AB_perc95):
    N, W = X.shape
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(X[i,:], color='C0', label='Spectrum index {}'.format(i))
    ax.plot(X_hat[i,:], color='red',  label='Reconstructed spectrum (mean)')
    ax.fill_between(np.arange(W), AB_perc5[i,:], AB_perc95[i,:], alpha=0.25,
                    color='orange', label='90% credibility interval')
    ax.plot(AB_perc5[i,:], color='orange', lw=0.5, alpha=0.5)
    ax.plot(AB_perc95[i,:], color='orange', lw=0.5, alpha=0.5)

    ax.legend()
    ax.set_xlim([0,W])
    ax.set(title='Comparison for spectrum index {}'.format(i),
           xlabel='Wavenumber', ylabel='Spectrum intensity')
    ax.set_ylim(top=np.max(X))
    plt.tight_layout()
    plt.show()
    return


def compare_reconstruction(X, trace, burnin=0.1):
    N, W = X.shape
    M = trace['AB'].shape[2]
    # Burn-in: discard beginning samples
    start = int(burnin*M)
    x = trace['AB'][:,:, start:]
    
    X_hat = np.mean(x, axis=2)
    AB_perc5 = np.percentile(x, 5, axis=2)
    AB_perc95 = np.percentile(x, 95, axis=2)
    
    plot_reconstruction(X, X_hat, figsize=(10,20), cmap='Greys')
    plot_reconstruction_3d(X, X_hat, figsize=(15,10),
                           elev=67.5, angle=270, grid=False, transparent=False)
    
    mse = np.mean((X-X_hat)**2)
    print('MSE:', mse)
    
    # Highest spectrum index
    n = np.argmax(X)//W  
    compare_spectrum(n, X, X_hat, AB_perc5, AB_perc95)
    compare_spectrum(0, X, X_hat, AB_perc5, AB_perc95)
    
    return X_hat


def plot_parameters_B(trace, Vp, burnin=0.1, X=None):
    M = trace['B'].shape[2]
    W = trace['B'].shape[1]
    start = int(burnin*M)
    Bs = trace['B'][0,:,start:]
    
    # Aggregate samples
    B_mean = np.mean(Bs, axis=1)
    B_std = np.std(Bs, axis=1)
    B_perc5 = np.percentile(Bs, 5, axis=1)
    B_perc95 = np.percentile(Bs, 95, axis=1)
    
    w_peak = np.argmax(B_mean)
    
    Vp = Vp.flatten()
    s =  scaling_constant(B_mean, Vp)
    print('s',s)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(7,4))
    if X is not None: 
        ax2 = ax.twinx()
        for n in range(X.shape[0]-1):
            ax2.plot(X[n,:], color='C0', alpha=0.1, zorder=0)
        line4, = ax2.plot(X[-1,:], color='C0', alpha=0.1, label='Observed spectra', zorder=0)
        #line3, = ax2.plot(X[np.argmax(X)//W,:], color='C0', alpha=0.1, label='Highest observation', zorder=0)
    
    line1, = ax.plot(B_mean, color='red',  label='NMF Gibbs (mean)', zorder=1)
    fill = ax.fill_between(np.arange(W),B_perc5, B_perc95, alpha=0.25,
                           color='orange', label='NMF Gibbs\n(90% credibility interval)', zorder=1)
    ax.plot(B_perc5, color='orange', lw=0.5, alpha=0.5, zorder=1)
    ax.plot(B_perc95, color='orange', lw=0.5, alpha=0.5, zorder=1)
    line5, = ax.plot(s*Vp, ls=(0, (5, 5)), label='Scaled pure Voigt component')

    ax.set_xlim([0,W])
    ax.set_title('Weights in B distribution')
    ax.set_ylabel(r'Weights in $B$')#, color='r')
    #ax.tick_params('y', colors='r')
    ax.set(xlabel='Index in B (Wavenumber)')
    handles = [line1, fill, line5]
    if X is not None:
        ax2.set_ylabel('Spectrum intensity', color='C0')
        ax2.tick_params('y', colors='C0')
        ax2.set_ylim(bottom=-0.1)
        handles += [line4]

    labels = [handle.get_label() for handle in handles]
    ax.legend(handles, labels, loc='upper left')
    #ax.legend(handles, labels)
    
    # Plot Noise
    fig, ax = plt.subplots(1, 1, figsize=(7,4))
    ax.plot(B_std, color='orange', lw=2)
    ax.set(title='Weights in B standard deviation', xlabel='Index in B (Wavenumber)')
    ax.set_ylabel('Standard deviation')#, color='orange')
    #ax.tick_params('y', colors='orange')
    ax.set_xlim([0,W])

    plt.tight_layout()
    plt.show()
    
    return w_peak


def plot_trace(trace, burnin=0.1, figsize = (15,4)):
    M = trace["A"].shape[2]
    start = int(burnin*M)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(x=start, c='grey', ls='--')
    for i in range(trace["A"].shape[0]):
        ax.plot(trace["A"][i,0,:], lw=0.1)
    ax.set(title='Traces: samples of elements of A',  xlabel='Iteration')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(x=start, c='grey', ls='--')
    for i in range(trace["B"].shape[1]):
        ax.plot(trace["B"][0,i,:], lw=0.2)
    ax.set(title='Traces: sample of elements of B', xlabel='Iteration')
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(x=start, c='grey', ls='--')
    ax.plot(trace["sigma2"])
    ax.set(title=r'Trace: sample of $\sigma^2$', xlabel='Iteration')
    plt.tight_layout()
    plt.show()
    return


def scaling_constant(x, xhat):
    c =  (x.T @ xhat) / (xhat.T @ xhat)
    return c


def plot_distribution_A(trace, hyperparams, X, w_peak, alphas, burnin=0.1):
    M = trace['A'].shape[2]
    N = trace['A'].shape[0]
    start = int(burnin*M)
    As = trace['A'][:,0,start:]
    
    # Normalize the mean values between 0 and 1 to map to colormap
    means = np.mean(As, axis=1).flatten()
    colormap = plt.cm.viridis
    norm = plt.Normalize(min(means), max(means))
    
    ### Boxplot
    fig, ax = plt.subplots(figsize=(15,5))
    box = ax.boxplot(As.T, patch_artist=True, medianprops={'color': 'black'})
    ax.set_xticklabels([r'$A_{{{}}}$'.format(i) for i in range(N)])
    # Set the colors for each boxplot
    for patch, median in zip(box['boxes'], means):
        color = colormap(norm(median))
        patch.set_facecolor(color)
    plt.tight_layout()
    plt.show()
    
    ### Histograms Grid
    fig, axes = plt.subplots(5,5, figsize=(15,15))
    max_height = 0
    max_x = 0
    # Loop through each subplot
    for k in range(25):
        i, j = (k // 5), (k % 5)  # Compute i,j indices
        ax = axes[i,j]
        data = As[k,:]
        n, bins, patches = ax.hist(data, bins=10, density=True, alpha=0.7, color=colormap(norm(means[k])),
                                   edgecolor='black', linewidth=1.2, label='Posterior')
        max_height = max(max_height, np.max(n))
        max_x = max(max_x, np.max(data))
        
    # Compute the PDF values of an exponential distribution
    x = np.linspace(0, max_x, 100)
    pdf = expon.pdf(x, scale=1/hyperparams['alpha'])
    for k in range(25):
        i, j = (k // 5), (k % 5)  # Compute i,j indices
        ax = axes[i,j]
        ax.plot(x, pdf, '-', c='C9', linewidth=2, label='Exponential Prior')
        if j == 0:
            ax.set_ylabel('Density', fontsize=18)
        if j != 0:
            ax.set_yticks([])
        if i != 4:
            ax.set_xticks([])
        #ax.set_title(r'Histogram of $A_{{{}}}$'.format(k))
        ax.text(0.05, 35, r'$A_{{{}}}$'.format(k), fontsize=20)
        ax.set_ylim([0,max_height])
    
        #ax.legend(loc="upper left")     
    plt.tight_layout()
    plt.show()
    
    ### Map at peak:
    # Map at peak 1
    n = int(np.sqrt(N))
    fig, ax = plt.subplots(figsize=(4,4))
    mappable = ax.imshow(X[:, w_peak].reshape((n,n)), cmap='viridis', interpolation='none')
    ax.set_title(r'Map at $w_{{peak}}$ = {}'.format(w_peak))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(mappable=mappable, ax=ax, shrink=0.7)
    cbar.set_label('Spectrum intensity')
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(4,4))
    mappable = ax.imshow(means.reshape((n,n)), cmap='viridis', interpolation='none')
    ax.set_title(r'Mean of $A_n$')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(mappable=mappable, ax=ax, shrink=0.7)
    cbar.set_label('')
    plt.tight_layout()
    plt.show()
    
    stds = np.std(As, axis=1).flatten()
    fig, ax = plt.subplots(figsize=(4,4))
    mappable = ax.imshow(stds.reshape((n,n)), cmap='viridis', interpolation='none')
    ax.set_title(r'Standard deviation of $A_n$')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(mappable=mappable, ax=ax, shrink=0.7)
    cbar.set_label('')
    plt.tight_layout()
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=(4,4))
    mappable = ax.imshow(alphas.reshape((n,n)), cmap='viridis', interpolation='none')
    ax.set_title(r'Truth $A_n$')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(mappable=mappable, ax=ax, shrink=0.7)
    cbar.set_label('')
    plt.tight_layout()
    plt.show()
    
    print('s', scaling_constant(means, alphas))

    fig, ax = plt.subplots(figsize=(4,4))
    x = alphas.flatten()
    y = means
    sortIdx = x.argsort()
    x = x[sortIdx]
    y = y[sortIdx]
    coef = np.polyfit(x,y,1)
    print('coef', coef)
    poly1d_fn = np.poly1d(coef) 
    # poly1d_fn is now a function which takes in x and returns an estimate for y
    ax.scatter(x,y, color=colormap(norm(y)))
    ax.plot(x, poly1d_fn(x), ls='--', color='grey', label='Linear regression\ny={}x+{}'.format(round(coef[0],3), round(coef[1],3)))
    #ax.scatter(alphas, means)
    ax.set(xlabel=r'Truth $A_n$', ylabel=r'Mean of $A_n$', title='Comparison truth and Gibbs values')
    plt.legend()
    plt.tight_layout()
    plt.show()
    

    ### Violin plot
    """
    As_df = pd.DataFrame(As[:,0,:].T, columns=[r'$A_{{{}}}$'.format(i) for i in range(N)])
    plt.figure(figsize=(20, 5))
    ax = sns.violinplot(As_df, cut=0, inner='quartiles')
    for l in ax.lines:
        l.set_linestyle('--')
        l.set_linewidth(0.6)
        l.set_color('red')
        l.set_alpha(0.8)
    for l in ax.lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(1.2)
        l.set_color('black')
        l.set_alpha(0.8)
    plt.tight_layout()
    plt.show()
    """
    return


def plot_distribution_B(trace, hyperparams, w_peak, burnin=0.1):
    M = trace['B'].shape[2]
    W = trace['B'].shape[1]
    start = int(burnin*M)
    Bs = trace['B'][0,:,start:]
    
    idx_1 = np.linspace(0, w_peak, num=12, dtype=int)
    idx_2 = np.linspace(w_peak+1, W-1, num=12, dtype=int)
    #idx_1 = np.sort(np.random.choice(peak, 12, replace=False))
    #idx_2 = np.sort(np.random.choice(W-peak, 12, replace=False)) + peak
    idx = np.concatenate((idx_1, [w_peak], idx_2))
    
    ### Histograms Grid
    fig, axes = plt.subplots(5,5, figsize=(15,15))
    max_height = 0
    max_x = 0
    # Loop through each subplot
    for k in range(5**2):
        i, j = (k // 5), (k % 5)  # Compute i,j indices
        ax = axes[i,j]
        data = Bs[idx[k],:]
        n, bins, patches = ax.hist(data, bins=10, density=True, alpha=0.7, color='C3', edgecolor='black', linewidth=1.2, label='Posterior')
        max_height = max(max_height, np.max(n))
        max_x = max(max_x, np.max(data))
        
    # Compute the PDF values of an exponential distribution
    x = np.linspace(0, max_x, 100)
    pdf = expon.pdf(x, scale=1/hyperparams['beta'])
    for k in range(5**2):
        i, j = (k // 5), (k % 5)  # Compute i,j indices
        ax = axes[i,j]
        ax.plot(x, pdf, '-', c='C9', linewidth=2, label='Exponential Prior')
        if j == 0:
            ax.set_ylabel('Density')
        ax.set_title(r'Histogram of $B_{{{}}}$'.format(idx[k]))
        ax.set_ylim([0,max_x])
        ax.set_ylim([0,max_height])
        ax.legend(loc="upper right")  
    plt.tight_layout()
    plt.show()


def plot_distribution_sigma2(trace, hyperparams, burnin=0.1):
    sigma2s = np.copy(trace["sigma2"])
    M = len(sigma2s)
    # Burn-in: discard beginning samples
    start = int(burnin*M)
    sigma2s = sigma2s[start:]
 
    ### Zoomed-in histogram
    fig, ax  = plt.subplots()
    data = sigma2s
    ax.hist(data, bins=10, density=True, alpha=0.7, color='C4', edgecolor='black', linewidth=1.2, label='Posterior')
    # Compute the PDF values of an gamma distribution
    #ax2 = ax.twinx()
    x = np.linspace(np.min(data), np.max(data), 100)
    pdf = invgamma.pdf(x, a=hyperparams['k'], scale=hyperparams['theta'])
    #ax2.plot(x, pdf, '-', c='C9', linewidth=2, label='Inverse Gamma Prior')
    # Set labels and title
    ax.set_ylabel('Density (Histogram)')#, c='C4')
    ax.axvline(x=0.05**2, ls='--', c='red')
    #ax.tick_params('y', colors='C4')
    #ax2.set_ylabel('Probability Density (PDF)', c='C9')
    #ax2.tick_params('y', colors='C9')
    ax.set_title(r'Gibbs: Empiricial posterior distribution of $p(\sigma^2$|X)')
    ax.set_xlabel(r'Value of $\sigma^2$')
    #ax.legend(loc="upper left") 
    #ax2.legend(loc="upper right") 

    plt.tight_layout()
    plt.show()
    
    ### Zoomed-out histogram
    """
    fig, ax  = plt.subplots()
    data = sigma2s
    ax.hist(data, bins=10, density=True, alpha=0.7, color='C4', edgecolor='black', linewidth=1.2, label='Posterior')
    # Compute the PDF values of an gamma distribution
    ax2 = ax.twinx()
    x = np.linspace(0, 1, 100)
    pdf = invgamma.pdf(x, a=hyperparams['k'], scale=hyperparams['theta'])
    ax2.plot(x, pdf, '-', c='C9', linewidth=2, label='Inverse Gamma Prior')
    # Set labels and title
    ax.set_ylabel('Density (Histogram)', c='C4')
    ax.tick_params('y', colors='C4')
    ax2.set_ylabel('Probability Density (PDF)', c='C9')
    ax2.tick_params('y', colors='C9')
    ax.set_title(r'Histogram of $\sigma^2$ zoomed out')
    ax.legend(loc="upper left") 
    ax2.legend(loc="upper right") 
    plt.tight_layout()
    plt.show()
    """
    
    ### Boxplot
    #plt.boxplot(sigma2s)
    #plt.title('Boxplot of $\sigma^2$')
    #plt.show()
    #sns.violinplot(pd.DataFrame(trace["sigma2"][int(0.25*M):,]), cut=0, bw=0.1, inner='quartiles')
    #plt.show()
    
    return




#%%
if __name__ == "__main__":
    # Set seed for reproducibility 
    np.random.seed(0)

    ### Multiple Maps
    mapsize= (5,5)
    N = mapsize[0]*mapsize[1]
    W = 200
    dataset = load_dataset('SERSTestsetNew.pkl')

    #%%
    X = dataset[2]['X']
    
    ### Example 1 MAP
    #gen = SERSGenerator(mapsize=mapsize, Nw=W, seed=2)
    #X = gen.generate(N_hotspots=2, K=1, sig=0.05, sbr=1, plot=False, background='none')
    #X = np.clip(X, 0, None)

    plot_SERS(X)
    plot_highest_spectrum(X, figsize=(10,2))

    #%%
    # Run Gibbs sampling
    M = 5000
    K = 1
    hyperparams = {'alpha':10, 'beta':1, 'k':1, 'theta':1}
    burnin = 0.1
    
    trace = nmf_gibbs(X, K, M, hyperparams, verbose=True)
    trace = gibbs_probabilities(X, trace, hyperparams, verbose=True)
   
    #%%
    # Plot probabilities
    plot_probabilities(trace, figsize=(7,5),burnin=burnin)
    
    #%%
    # Reconstruct
    X_hat = compare_reconstruction(X, trace, burnin=burnin)

    #%%
    w_peak = plot_parameters_B(trace, dataset[2]['Vp'], burnin=burnin, X=None)
    
    #%%
    # Plot parameters distributions
    plot_distribution_A(trace, hyperparams, X, w_peak, dataset[2]['alpha'], burnin=burnin)
    #%%
    #plot_distribution_B(trace, hyperparams, w_peak, burnin=burnin)
    plot_distribution_sigma2(trace, hyperparams, burnin=burnin)
    
    #%%
    plot_trace(trace, burnin=0.1, figsize=(10,4))


    #%%
    # Run algo on test set
    S = len(dataset)
    keys = ['A', 'B', 'sigma2', 'log_likelihood', 'log_posterior']
    for i in tqdm(range(S)):
        X = dataset[i]['X']
        
        if i in [6,7, 10, 11, 12]:
            print(np.max(X))
            plot_data_3d(X, elev=67.5, angle=270, grid=False, transparent=False)
        
        trace = nmf_gibbs(X, K, M, hyperparams)
        trace = gibbs_probabilities(X, trace, hyperparams)
        save_nmf_gibbs('gibbs_traces_new/trace_{}.pkl'.format(i), trace, keys)

        #mse, X_hat = MSE_reconstruction(X, trace, burnin=burnin)
        #mean_log_likelihood = np.mean(trace['log_likelihood'][int(burnin*M):])
        #mean_log_posterior = np.mean(trace['log_posterior'][int(burnin*M):])
        
        # Record result
        #traces[i] = trace
        #measurements[i,:] = np.array([mse, mean_log_likelihood, mean_log_posterior])

    #%%
    #plt.hist(measurements[:,0])
    #plt.show()
    #plt.hist(measurements[:,1])
    #plt.show()
    #plt.hist(measurements[:,2])
    #plt.show()
# %%
