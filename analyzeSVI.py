#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal, LogNormal, Exponential

from SERSGenerator import SERSGenerator
from SERSDataset import load_dataset
from plot_map import *
from nmfSVI import *


def plot_measurements(out, figsize=(15,5), zoom_offset=1000):
    M = len(out['log_likelihood'])
    lw = 0.1
    
    ### Log Likelihood
    fig, ax = plt.subplots(1,1, figsize=figsize)
    y = out['log_likelihood']
    ax.plot(y, color='C0', lw=lw*2)
    ax.set_title(r'Log-likelihood: $\log P(X|z)$', fontsize=20)
    ax.set_xlabel('Iteration', fontsize=14)
    #ax.set_xticks(np.arange(0,M+1,500))
    ax.set_xlim(right=M)
    
    # Inset -- ZOOM
    s = 0.143
    axin = ax.inset_axes([s, 0.1, 1-s, 0.63])
    axin.plot(y, color='C0', lw=lw)
    # Zoom in on the noisy data in the inset axis
    axin.set_xlim([zoom_offset, M])
    axin.set_ylim(bottom=min(y[zoom_offset:]), top=max(y[zoom_offset:]))
    # Hide inset axis ticks
    axin.set_xticks([zoom_offset])
    #axin.set_yticks([])
    # Add the lines to indicate where the inset axis is coming from
    ax.indicate_inset_zoom(axin)
    plt.tight_layout()
    plt.show()
    
    ### Log Posterior
    fig, ax = plt.subplots(1,1, figsize=figsize)
    y = out['log_posterior']
    ax.plot(y, color='C1', lw=lw*2)
    ax.set_title(r'Log-posterior: $\log P(z | X)$', fontsize=20)
    ax.set_xlabel('Iteration', fontsize=14)
    #ax.set_xticks(np.arange(0,M+1,500))
    ax.set_xlim(right=M)
    
    # Inset -- ZOOM
    s = 0.143
    axin = ax.inset_axes([s, 0.1, 1-s, 0.63])
    axin.plot(y, color='C1', lw=lw)
    # Zoom in on the noisy data in the inset axis
    axin.set_xlim([zoom_offset, M])
    axin.set_ylim(bottom=min(y[zoom_offset:]), top=max(y[zoom_offset:]))
    # Hide inset axis ticks
    axin.set_xticks([zoom_offset])
    #axin.set_yticks([])
    # Add the lines to indicate where the inset axis is coming from
    ax.indicate_inset_zoom(axin)
    plt.tight_layout()
    plt.show()
    
    ### Loss
    fig, ax = plt.subplots(1,1, figsize=figsize)
    y = out['losses']
    ax.plot(y, color='k', lw=lw*2)
    ax.set_title('SVI Loss', fontsize=20)
    ax.set_xlabel('Iteration', fontsize=14)
    #ax.set_xticks(np.arange(0,M+1,500))
    ax.set_xlim(right=M)
    
    # Inset -- ZOOM
    s = 0.143
    axin = ax.inset_axes([s, 0.37-0.1, 1-s, 0.63])
    axin.plot(y, color='k', lw=lw)
    # Zoom in on the noisy data in the inset axis
    axin.set_xlim([zoom_offset, M])
    axin.set_ylim(bottom=min(y[zoom_offset:]), top=max(y[zoom_offset:]))
    # Hide inset axis ticks
    axin.set_xticks([zoom_offset])
    #axin.set_yticks([])
    # Add the lines to indicate where the inset axis is coming from
    ax.indicate_inset_zoom(axin)
    plt.tight_layout()
    plt.show()
    return


def compare_spectrum(i, X, X_hat, AB_perc5, AB_perc95):
    N, W = X.shape
    fig, ax = plt.subplots(figsize=(10,5))
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
    plt.tight_layout()
    plt.show()
    return


def compare_reconstruction(X, out, num_samples):
    N, W = X.shape
    X = X.numpy()
    
    # Reconstruct data with posterior mean
    mu = out['variational_family'].mu_var
    sigma = torch.exp(out['variational_family'].log_sigma_var)
    mean = LogNormal(mu, sigma).mean.unsqueeze(0)
    _, _, _, AB = out['likelihood_model'].reconstruct(mean)
    X_hat = AB.squeeze(0).numpy()
    
    _, _, z = out['variational_family'].rsample((num_samples,))
    _, _, _, ABs = out['likelihood_model'].reconstruct(z)
    ABs = ABs.numpy()

    #X_hat2 = np.mean(ABs, axis=0)
    AB_perc5 = np.percentile(ABs, 5, axis=0)
    AB_perc95 = np.percentile(ABs, 95, axis=0)
    
    plot_reconstruction(X, X_hat, figsize=(10,20))
    plot_reconstruction_3d(X, X_hat, figsize=(15,10),
                           elev=67.5, angle=270, grid=False, transparent=False)
    
    mse = np.mean((X-X_hat)**2)
    print('MSE:', mse)
    
    # Highest spectrum index
    n = np.argmax(X)//W  
    compare_spectrum(n, X, X_hat, AB_perc5, AB_perc95)
    
    return X_hat


def scaling_constant(x, xhat):
    c =  (x.T @ xhat) / (xhat.T @ xhat)
    return c



def plot_parameters_B(out, Vp, num_samples, X=None):
    N, W = X.shape
    X = X.numpy()
    
    # Reconstruct data with posterior mean
    mu = out['variational_family'].mu_var[N:N+W]
    sigma = torch.exp(out['variational_family'].log_sigma_var[N:N+W])
    dist = LogNormal(mu, sigma)
    B_mean = dist.mean.numpy()
    #B_scale = dist.scale.numpy()

    _, _, z = out['variational_family'].rsample((num_samples,))
    _, Bs, _, _ = out['likelihood_model'].reconstruct(z)
    Bs = Bs.squeeze(1).numpy()
    
    # Aggregate samples
    B_std = np.std(Bs, axis=0)
    B_perc5 = np.percentile(Bs, 5, axis=0)
    B_perc95 = np.percentile(Bs, 95, axis=0)
    
    w_peak = np.argmax(B_mean)
    
    s =  scaling_constant(Vp, B_mean)
    print('s',s)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10,4))
    if False: #X is not None: 
        ax2 = ax.twinx()
        for n in range(X.shape[0]-1):
            ax2.plot(X[n,:], color='C0', alpha=0.1, zorder=0)
        line4, = ax2.plot(X[-1,:], color='C0', alpha=0.1, label='Observed spectra', zorder=0)
        #line3, = ax2.plot(X[np.argmax(X)//W,:], color='C0', alpha=0.1, label='Highest observation', zorder=0)
    
    line1, = ax.plot(B_mean, color='red',  label='NMF SVI (mean)', zorder=1)
    fill = ax.fill_between(np.arange(W),B_perc5, B_perc95, alpha=0.25,
                           color='orange', label='NMF SVI (90% credibility interval)', zorder=1)
    ax.plot(B_perc5, color='orange', lw=0.5, alpha=0.5, zorder=1)
    ax.plot(B_perc95, color='orange', lw=0.5, alpha=0.5, zorder=1)
    line5, = ax.plot((1/s)*Vp, ls=(0, (5, 5)), label='Scaled pure Voigt component')

    ax.set_xlim([0,W])
    ax.set_title('Weights in B distribution')
    ax.set_ylabel(r'Weights in $B$')#, color='r')
    #ax.tick_params('y', colors='r')
    ax.set(xlabel='Index in B (Wavenumber)')
    handles = [line1, fill, line5]
    if False:#X is not None:
        ax2.set_ylabel('Spectrum intensity', color='C0')
        ax2.tick_params('y', colors='C0')
        handles += [line4]

    labels = [handle.get_label() for handle in handles]
    ax.legend(handles, labels, loc='upper left')
    
    # Plot Noise
    fig, ax = plt.subplots(1, 1, figsize=(10,4))
    ax.plot(B_std, color='orange')
    ax.set(title='Standard deviation of weights in B', xlabel='Index in B (Wavenumber)')
    ax.set_ylabel('Standard deviation')#,color='orange')
    #ax.tick_params('y', colors='orange')
    ax.set_xlim([0,W])

    plt.tight_layout()
    plt.show()
    
    # Plot Noise
    term1 = torch.sqrt(torch.exp(2*mu+sigma**2))
    term2 = torch.sqrt(torch.exp(sigma**2)-1)
    std = term1 * term2
    fig, ax  = plt.subplots(figsize=(10,4))
    line1, = ax.plot(term2, label=r'$\sqrt{\exp(\sigma^2)-1}$', color='C4')
    ax2 = ax.twinx()
    line2, = ax2.plot(term1, label=r'$\sqrt{\exp(2\mu+\sigma^2)}$', color='C2')
    line3, = ax.plot(std, label=r'$\sqrt{(\exp(\sigma^2)-1)\exp(2\mu+\sigma^2)}$', color='orange')
    ax.set(title='Standard deviation decomposition of B', xlabel='Index in B (Wavenumber)')
    ax.set_ylabel(r'Standard deviation and $\sqrt{\exp(\sigma^2)-1}$', color='k')
    ax.tick_params('y', colors='k')
    ax2.set_ylabel(r'$\sqrt{\exp(2\mu+\sigma^2)}$', color='C2')
    ax2.tick_params('y', colors='C2')
    ax.set_xlim([0,W])
    
    handles = [line1, line2, line3]
    labels = [handle.get_label() for handle in handles]
    ax.legend(handles, labels, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    return w_peak


def plot_distribution_A(out, X, w_peak):
    N, W = X.shape
    mu = out['variational_family'].mu_var[:N]
    sigma = torch.exp(out['variational_family'].log_sigma_var[:N])
    dist = LogNormal(mu, sigma)
    means = dist.mean.numpy()
    stds = dist.stddev.numpy()
    
    # Normalize the means values between 0 and 1 to map to colormap
    colormap = plt.cm.viridis
    norm = plt.Normalize(min(means), max(means))
    
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    max_x = np.max(means+3*stds)
    x = np.linspace(0.001, max_x, 200)     
    #y_prior = torch.exp(out['prior_model'].distr_A.log_prob(torch.tensor(x))).numpy()
    max_height = 0
    for k in range(25):
        i, j = (k // 5), (k % 5)  # Compute i,j indices
        # Compute the LogNormal PDF values
        distr = LogNormal(mu[k], sigma[k])
        y = torch.exp(distr.log_prob(torch.tensor(x))).numpy()
        max_height = max(max_height, np.max(y))
        
        #axes[i, j].plot(x, y_prior, c='C9', label="Exponential Prior")
        axes[i, j].fill_between(x, 0, y, color=colormap(norm(means[k])),
                                 label=r"LogNormal($\mu=${:.2f}, $\sigma=${:.2f})".format(mu[k], sigma[k]))
        #axes[i, j].set_title(r'PDF of $A_{{{}}}$'.format(k))
        #axes[i, j].legend(handlelength=0, fontsize=8, loc='upper left')

    for k in range(25):
        i, j = (k // 5), (k % 5)
        ax = axes[i,j]
        ax.set_ylim([0,max_height])
        if j == 0:
            ax.set_ylabel('Probability Density', fontsize=18)
        if j != 0:
            ax.set_yticks([])
        if i != 4:
            ax.set_xticks([])
            
        ax.text(0.05, 35, r'$A_{{{}}}$'.format(k), fontsize=20)
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
    return


def plot_distribution_B(out, w_peak):
    N = out['prior_model'].N
    W = out['prior_model'].W
    ### Visualize distributions of parameter B
    mu = out['variational_family'].mu_var[N:N+W]
    sigma = torch.exp(out['variational_family'].log_sigma_var[N:N+W])
    dist = LogNormal(mu, sigma)
    means = dist.mean.numpy()
    stds = dist.stddev.numpy()
    
    idx_1 = np.linspace(0, w_peak, num=12, dtype=int)
    idx_2 = np.linspace(w_peak+1, W-1, num=12, dtype=int)
    idx = np.concatenate((idx_1, [w_peak], idx_2))

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    max_x = np.max(means+3*stds)
    x = np.linspace(0.001, max_x, 200)
    #y_prior = torch.exp(out['prior_model'].distr_B.log_prob(torch.tensor(x))).numpy()
    max_height = 0
    for k in range(25):
        i, j = (k // 5), (k % 5)  # Compute i,j indices
        # Compute the LogNormal PDF values
        distr = LogNormal(mu[idx[k]], sigma[idx[k]])
        y = torch.exp(distr.log_prob(torch.tensor(x))).numpy()
        max_height = max(max_height, np.max(y))
        
        #axes[i, j].plot(x, y_prior, c='C9', label="Prior")
        axes[i, j].fill_between(x, 0, y, color='C3', alpha=0.7,
                                 label=r"LogNormal($\mu=${:.2f}, $\sigma=${:.2f})".format(mu[idx[k]], sigma[idx[k]]))
        axes[i, j].set_title(r'PDF of $B_{{{}}}$'.format(idx[k]))
        axes[i, j].legend(handlelength=0, fontsize=8, loc='upper left')
        if j==0:
            axes[i,j].set_ylabel("Probability density")
    for k in range(25):
        i, j = (k // 5), (k % 5)
        axes[i,j].set_ylim([0,max_height])
    plt.tight_layout()
    plt.show()
    return


def plot_distribution_sigma2(out):
    ### Sigma 2 distribution
    mu = out['variational_family'].mu_var[-1]
    sigma = torch.exp(out['variational_family'].log_sigma_var[-1])
    distr = LogNormal(mu, sigma)
    x = np.linspace(distr.mean-5*distr.stddev, distr.mean+5*distr.stddev, 200)
    # Compute the LogNormal PDF values
    y = torch.exp(distr.log_prob(torch.tensor(x))).numpy()
    #y_prior = torch.exp(out['prior_model'].distr_sigma2.log_prob(torch.tensor(x))).numpy()
    
     # Plot the PDF in the current subplot
    fig, ax = plt.subplots()
    ax.fill_between(x, 0, y, color='C4',
                    label=r"LogNormal($\mu=${:.2f}, $\sigma=${:.2f})".format(mu, sigma))

    # Set labels and title
    ax.set_title(r'PDF of $\sigma^2$')
    ax.legend(handlelength=0, fontsize=8, loc='upper left')
    ax.set_ylabel("Probability density")
    ax.set_xticks(np.linspace(min(x),max(x),5))
    plt.tight_layout()
    plt.show()
    return
    

def plot_trace_param(out, idx, figsize=(10,4)):
    x = out['mu'][:,idx]
    y = out['log_sigma'][:,idx]
    
    fig, ax = plt.subplots(1,1, figsize=figsize)
    for j, i  in enumerate(idx):
        ax.plot(x[:,j], label=r'$\mu_{{{}}}$'.format(i))
        ax.plot(y[:,j], label=r'$\log \sigma_{{{}}}$'.format(i))
    ax.legend()
    ax.set(xlabel='Iteration')
    plt.show()
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
    X = torch.tensor(X)
    # Generate some synthetic data
    #gen = SERSGenerator(mapsize=mapsize, Nw=W, seed=2)
    #X = gen.generate(N_hotspots=2, K=1, sig=0.05, sbr=1, plot=False, background='none')
    #X = np.clip(X, 0, None).astype('float32')
    #X = torch.tensor(X)
    

    plot_SERS(X)
    plot_highest_spectrum(X, figsize=(10,2))

    #%% 
    # Run SVI
    num_epochs = 20000
    num_samples = 1
    hyperparams = {'alpha':10, 'beta':1, 'k':1, 'theta':1}

    out = nmf_SVI(X, num_epochs, num_samples, hyperparams, lr=0.01)
    
    #%%
    # Plot measurements
    plot_measurements(out, figsize=(7,5), zoom_offset=7000)
    
    #%%
    # Reconstruct
    X_hat = compare_reconstruction(X, out, num_samples=1000)
    
    #%%
    w_peak = plot_parameters_B(out, dataset[2]['Vp'].flatten(), num_samples=1000, X=X)
    
    #%% 
    # Plot traces
    plot_trace_param(out, [0,25, 25+108, N+W], figsize=(15,4)) 
    
    #%%
    # Plot parameters distributions
    plot_distribution_A(out, X, w_peak)
    #plot_distribution_B(out, w_peak)
    plot_distribution_sigma2(out)
    


    #%%
    ### Multiple Maps
    dataset = load_dataset('SERSTestsetNew.pkl')
    
    #%%
    # Run algo on test set
    S = len(dataset)
    for i in tqdm(range(S)):
        X = torch.tensor(dataset[i]['X'])
        out = nmf_SVI(X, num_epochs, num_samples, hyperparams, lr=0.01)
        save_nmf_svi('svi_models_new/model_{}.pkl'.format(i), out['variational_family'])
# %%
