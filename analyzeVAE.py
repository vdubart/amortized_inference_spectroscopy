#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal, LogNormal, Exponential
import seaborn as sns

from SERSGenerator import SERSGenerator
from SERSDataset import load_dataset
from plot_map import *
from nmfSVI import *
from nmfVAE import *


def plot_measurements(out, figsize=(15,5), zoom_offset=1000):
    M = len(out['log_likelihood'])
    lw = 1
    
    ### Log Likelihood
    fig, ax = plt.subplots(1,1, figsize=figsize)
    y = out['log_likelihood']
    ax.plot(y, color='C0', lw=lw*2)
    ax.set(title=r'Log-likelihood: $\log P(X|z)$',
           xlabel='Iteration')
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
    ax.set(title=r'Log-posterior: $\log P(z | X)$',
           xlabel='Iteration')
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
    ax.set(title='Loss', xlabel='Iteration')
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


#%%
if __name__ == "__main__":

    # Set seed for reproducibility 
    np.random.seed(0)

    ### Multiple Maps
    mapsize= (5,5)
    N = mapsize[0]*mapsize[1]
    W = 200
    dataset = load_dataset('SERSTestsetNew.pkl')
    hyperparams = {'alpha':10, 'beta':1, 'k':1, 'theta':1}
    
    #%%
    # Instatiate models
    # Define models
    prior_model = PriorModelNMF(N, W, alpha=hyperparams['alpha'], beta=hyperparams['beta'],
                             k=hyperparams['k'], theta=hyperparams['theta'])
    prior_model.eval()
    likelihood_model = LikelihoodModelNMF(N, W)
    likelihood_model.eval()

    # Load VAE
    vae_filename = "vae_models/07-15_13-53-06_epoch_500000_DONE.pkl"
    vae = VAE_NMF(N, W)
    vae.load_state_dict(torch.load(vae_filename, map_location=torch.device('cpu')))
    vae.eval()
    
    out = load_training_stats("vae_models/sub_out_New.pkl")

    #%%
    X = dataset[2]['X']
    X = torch.tensor(X)
    plot_SERS(X)
    plot_highest_spectrum(X, figsize=(10,2))
    
    
    #%%
    num_samples = 200
    
    with torch.no_grad():
        mu, sigma, z = vae(X_tensor, num_samples)
        distr = LogNormal(mu, sigma)
        mean = LogNormal(mu, sigma).mean
        
        _, _, _, AB = likelihood_model.reconstruct(mean)
        X_hat = AB.squeeze(0).numpy()
        
        mse = np.mean((X-X_hat)**2)
        
        plot_reconstruction(X, X_hat, figsize=(10,20))
        plot_reconstruction_3d(X, X_hat, figsize=(15,10),
                            elev=67.5, angle=270, grid=False, transparent=False)
        
        
    #%%
    fig, ax  = plt.subplots(figsize=(10,3))
    sns.stripplot(x=out['peaks'], size=2, ax=ax, jitter=0.25)
    ax.set(title='Jitter plot of map peaks during training', xlabel='Wavenumber', ylabel='Jitter')
    plt.tight_layout()
    plt.show()
    
    
    #%%
    fig, axes = plt.subplots(1,2, figsize=(15,5))
    ax = axes[0]
    ax.plot(out['log_likelihood'], color='C0')
    ax.set(title=r'Log-likelihood: $\log P(X|z)$', xlabel='Iteration')

    ax = axes[1]
    ax.plot(out['log_posterior'], color='C1')
    ax.set(title=r'Log-Posterior: $\log P(z|X)$', xlabel='Iteration')
    plt.show()

    fig, axes = plt.subplots(1,2, figsize=(15,5))
    ax = axes[0]
    ax.plot(out['losses'], color='C0')
    ax.set(title=r'Loss', xlabel='Iteration')

    ax = axes[1]
    ax.plot(out['ELBOs'], color='C1')
    ax.set(title=r'ELBO', xlabel='Iteration')
    plt.show()