#%%
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Normal, LogNormal, Exponential
from scipy.stats import lognorm

from SERSDataset import load_dataset
from plot_map import *
from nmfGibbs import *
from nmfSVI import *
from nmfVAE import *


def gibbs_reconstruct(X, trace, burnin=0.1):
    M = trace['B'].shape[2]
    W = trace['B'].shape[1]
    N = trace['A'].shape[0]
    start = int(burnin*M)
    Bs = trace['B'][:,:,start:]
    As = trace['A'][:,:,start:]
    
    # Aggregate samples
    B_mean = np.mean(Bs, axis=2)
    A_mean = np.mean(As, axis=2)
    X_hat = A_mean@B_mean
    
    mse = np.mean((X-X_hat)**2)
    return mse, X_hat


## Normalized NMF
def scaling_constant(x, xhat):
    c =  (x.T @ xhat) / (xhat.T @ xhat)
    return c


def scaled_MSE(x, xhat):
    c =  (x.T @ xhat) / (xhat.T @ xhat)
    sMSE = np.mean((x - c*xhat)**2)
    return sMSE


#%%
if __name__ == "__main__":
    ### Multiple Maps
    dataset = load_dataset('SERSTestsetNew.pkl')
    
    N = 25
    W = 200
    M = 5000
    burnin = 0.1
    hyperparams = {'alpha':10, 'beta':1, 'k':1, 'theta':1}
    
    # Define models
    prior_model = PriorModelNMF(N, W, alpha=hyperparams['alpha'], beta=hyperparams['beta'],
                             k=hyperparams['k'], theta=hyperparams['theta'])
    prior_model.eval()
    likelihood_model = LikelihoodModelNMF(N, W)
    likelihood_model.eval()
    
    S = len(dataset)
    num_samples = 300

    #%%
    # Load GIBBS on test set
    measurements_gibbs = np.zeros((S, 5))
    X_hat_gibbs = np.zeros((S, num_samples, N*W))
    A_gibbs = np.zeros((S, num_samples, N))
    B_gibbs = np.zeros((S, num_samples, W))
    sigma2_gibbs = np.zeros((S, num_samples))
    scaling_constants_gibbs = np.zeros((S, num_samples))
    gibbs_peak_accuracy = np.zeros((4,))
    gibbs_peaks = np.zeros((S,2))
    for i in tqdm(range(S)):
        X = dataset[i]['X']
        alphas = dataset[i]['alpha'].flatten()
        Vp = dataset[i]['Vp'].flatten()
        trace = load_nmf_gibbs('gibbs_traces_new/trace_{}.pkl'.format(i))
        
        
        As = trace['A'][:,0,-num_samples:]
        Bs = trace['B'][0,:,-num_samples:]
        sigma2s = trace['sigma2'][-num_samples:]
        As_tensor = torch.tensor(As.T).unsqueeze(-1)
        Bs_tensor = torch.tensor(Bs.T).unsqueeze(1)
        ABs = torch.bmm(As_tensor, Bs_tensor).numpy().reshape((num_samples, -1))
        X_hat_gibbs[i,:,:] = ABs
        A_gibbs[i,:,:] = As.T
        B_gibbs[i,:,:] = Bs.T
        sigma2_gibbs[i,:] = sigma2s
        
        A_mean = np.mean(trace['A'][:,0,-num_samples:], axis=1)
        B_mean = np.mean(trace['B'][0,:,-num_samples:], axis=1)
        sMSE_A = scaled_MSE(alphas, A_mean)
        sMSE_B = scaled_MSE(Vp, B_mean)
        
        w_peak = np.argmax(B_mean)
        c = dataset[i]['c_data']
        diff = np.abs(w_peak-c)
        if diff >= 3:
            gibbs_peak_accuracy[-1] += 1
        else:
            gibbs_peak_accuracy[diff] += 1
        
        gibbs_peaks[i,0] = w_peak
        gibbs_peaks[i,1] = dataset[i]['c']
        
        mse, X_hat = gibbs_reconstruct(X, trace, burnin=burnin)
        mean_log_likelihood = np.mean(trace['log_likelihood'][int(burnin*M):])
        mean_log_posterior = np.mean(trace['log_posterior'][int(burnin*M):])
        
        # Record result
        measurements_gibbs[i,:] = np.array([mse, mean_log_likelihood, mean_log_posterior,
                                            sMSE_A, sMSE_B])

    #%%
    measures = ['MSE', 'Log-like', 'Log-post', 'sMSE(A)', 'sMSE(B)']
    for i in range(5):
        mean_ = np.mean(measurements_gibbs[:,i])
        std_ = np.std(measurements_gibbs[:,i])
        print(measures[i], mean_, std_)


    #%%
    # Load SVI on test set
    measurements_svi = np.zeros((S, 3))
    X_hat_svi = np.zeros((S, num_samples, N*W))
    A_svi = np.zeros((S, num_samples, N))
    B_svi = np.zeros((S, num_samples, W))
    sigma2_svi = np.zeros((S, num_samples))
    scaling_constants_svi = np.zeros((S, num_samples))
    for i in tqdm(range(S)):
        X = dataset[i]['X']
        X_tensor = torch.tensor(X)
        
        ## SVI
        with torch.no_grad():
            variational_family = load_nmf_svi('svi_models_new/model_{}.pkl'.format(i))
            # Reconstruct data with posterior mean
            mu = variational_family.mu_var
            sigma = torch.exp(variational_family.log_sigma_var)
            mean = LogNormal(mu, sigma).mean.unsqueeze(0)
            _, _, _, AB = likelihood_model.reconstruct(mean)
            X_hat = AB.squeeze(0).numpy()
            mse = np.mean((X-X_hat)**2)
            mean_log_likelihood = likelihood_model(X_tensor, mean).numpy()
            mean_log_prior = prior_model(mean).numpy()
            mean_log_posterior = mean_log_likelihood + mean_log_prior
            measurements_svi[i,:] = np.array([mse, mean_log_likelihood, mean_log_posterior])
            
            #
            _, _, z = variational_family.rsample((num_samples,))
            As, Bs, sigma2s, ABs = likelihood_model.reconstruct(z)
            X_hat_svi[i,:,:] = ABs.numpy().reshape((num_samples,-1))
            A_svi[i,:,:] = As.squeeze(-1).numpy()
            B_svi[i,:,:] = Bs.squeeze(1).numpy()
            sigma2_svi[i,:] = sigma2s.numpy()
            
            for j in range(num_samples):
                scaling_constants_svi[i,j] = scaling_constant(X.reshape(-1), X_hat_svi[i,j,:])
            
            #plot_reconstruction(X, X_hat, figsize=(10,20))
            #plot_reconstruction_3d(X, X_hat, figsize=(15,10),
            #                    elev=67.5, angle=270, grid=False, transparent=False)

    
    #%%
    # Run VAE on test set
    vae_filename = "vae_models/07-04_09-37-40_epoch_200000_DONE.pkl"
    vae = VAE_NMF(N, W)
    vae.load_state_dict(torch.load(vae_filename, map_location=torch.device('cpu')))
    vae.eval()
    measurements_vae = np.zeros((S, 3))
    X_hat_vae = np.zeros((S, num_samples, N*W))
    A_vae = np.zeros((S, num_samples, N))
    B_vae = np.zeros((S, num_samples, W))
    sigma2_vae = np.zeros((S, num_samples))
    for i in tqdm(range(S)):
        X = dataset[i]['X'].astype('float32')
        X_tensor = torch.tensor(X)
        ## VAE
        with torch.no_grad():
            mu, log_sigma, z = vae(X_tensor, num_samples)
            sigma = torch.exp(log_sigma)
            distr = LogNormal(mu, sigma)
            mean = LogNormal(mu, sigma).mean
    
            _, _, _, AB = likelihood_model.reconstruct(mean)
            X_hat = AB.squeeze(0).numpy()
            mse = np.mean((X-X_hat)**2)
            mean_log_likelihood = likelihood_model(X_tensor, mean).numpy()
            mean_log_prior = prior_model(mean).numpy()
            mean_log_posterior = mean_log_likelihood + mean_log_prior
            measurements_vae[i,:] = np.array([mse, mean_log_likelihood, mean_log_posterior])
            
            As, Bs, sigma2s, ABs = likelihood_model.reconstruct(z)
            X_hat_vae[i,:,:] = ABs.numpy().reshape((num_samples,-1))
            A_vae[i,:,:] = As.squeeze(-1).numpy()
            B_vae[i,:,:] = Bs.squeeze(1).numpy()
            sigma2_vae[i,:] = sigma2s.numpy()
            
            #if i%10 == 0:
            #    plot_reconstruction(X, X_hat, figsize=(10,20))
            #    plot_reconstruction_3d(X, X_hat, figsize=(15,10),
            #                    elev=67.5, angle=270, grid=False, transparent=False)
    
    #%%
    # Compare Xhat
    for i in [2, 80]:
        X = dataset[i]['X']
        X_gibbs = X_hat_gibbs[i,0,:].reshape((N,W))
        X_svi = X_hat_svi[i,0,:].reshape((N,W))
        X_vae = X_hat_vae[i,0,:].reshape((N,W))

        figsize=(5,5)
        plot_data_3d(X, figsize=figsize, elev=67.5, angle=270, grid=False, transparent=False)
        plot_data_3d(X_gibbs, figsize=figsize, elev=67.5, angle=270, grid=False, transparent=False)
        plot_data_3d(X_svi, figsize=figsize, elev=67.5, angle=270, grid=False, transparent=False)
        plot_data_3d(X_vae, figsize=figsize, elev=67.5, angle=270, grid=False, transparent=False)
    
    
    #%%
    fig, axes = plt.subplots(2,3, figsize=(15,6))
    bins = 15
    ## MSE
    min_ = min(min(measurements_gibbs[:,0]), min(measurements_svi[:,0]))
    max_ = max(max(measurements_gibbs[:,0]), max(measurements_svi[:,0]))
    ax = axes[0,0]
    n1, _, _ = ax.hist(measurements_gibbs[:,0], bins=bins,alpha=1.0, color='green', edgecolor='black', linewidth=1.2)
    ax.set_xlim([min_,max_])
    ax.set_title('Histogram of MSE', fontsize=15)
    ax.set_ylabel('Gibbs', rotation=0, fontsize=15, labelpad=25)
    ax = axes[1,0]
    n2, _, _  = ax.hist(measurements_svi[:,0], bins=bins, alpha=0.5, color='green', edgecolor='black', linewidth=1.2)
    ax.set_xlim([min_,max_])
    ax.set_ylabel('SVI', rotation=0, fontsize=15, labelpad=25)
    axes[0,0].set_ylim(top=max(max(n1),max(n2)))
    axes[1,0].set_ylim(top=max(max(n1),max(n2)))

    ## Log-likelihood
    min_ = min(min(measurements_gibbs[:,1]), min(measurements_svi[:,1]))
    max_ = max(max(measurements_gibbs[:,1]), max(measurements_svi[:,1]))
    ax = axes[0,1]
    n1, _, _  = ax.hist(measurements_gibbs[:,1], bins=bins, alpha=1, color='C0', edgecolor='black', linewidth=1.2)
    ax.set_xlim([min_,max_])
    ax.set_title('Histogram of Log-likelihood', fontsize=15)
    #ax.set_ylabel('Gibbs')
    ax = axes[1,1]
    n2, _, _  =ax.hist(measurements_svi[:,1], bins=bins, alpha=0.5, color='C0', edgecolor='black', linewidth=1.2)
    ax.set_xlim([min_,max_])
    #ax.set_ylabel('SVI')
    axes[0,1].set_ylim(top=max(max(n1),max(n2)))
    axes[1,1].set_ylim(top=max(max(n1),max(n2)))
    
    ## Log-Posterior
    min_ = min(min(measurements_gibbs[:,2]), min(measurements_svi[:,2]))
    max_ = max(max(measurements_gibbs[:,2]), max(measurements_svi[:,2]))
    ax = axes[0,2]
    n1, _, _  = ax.hist(measurements_gibbs[:,2], bins=bins, alpha=1, color='C1', edgecolor='black', linewidth=1.2)
    ax.set_xlim([min_,max_])
    ax.set_title('Histogram of Log-posterior', fontsize=15)
    #ax.set_ylabel('Gibbs')
    ax = axes[1,2]
    n2, _, _  = ax.hist(measurements_svi[:,2], bins=bins, alpha=0.5, color='C1', edgecolor='black', linewidth=1.2)
    ax.set_xlim([min_,max_])
    #ax.set_ylabel('SVI')
    axes[0,2].set_ylim(top=max(max(n1),max(n2)))
    axes[1,2].set_ylim(top=max(max(n1),max(n2)))
        
    plt.show()
    
    
    #%%
    fig, axes = plt.subplots(3,3, figsize=(15,6))
    bins = 15
    ## MSE
    min_ = min(min(measurements_gibbs[:,0]), min(measurements_svi[:,0]), min(measurements_vae[:,0]))
    max_ = max(max(measurements_gibbs[:,0]), max(measurements_svi[:,0]), max(measurements_vae[:,0]))
    ax = axes[0,0]
    n1, _, _ = ax.hist(measurements_gibbs[:,0], bins=bins,alpha=1.0, color='green', edgecolor='black', linewidth=1.2)
    ax.set_xlim([min_,max_])
    ax.set_title('Histogram of MSE', fontsize=15)
    ax.set_ylabel('Gibbs', rotation=0, fontsize=15, labelpad=25)
    ax = axes[1,0]
    n2, _, _  = ax.hist(measurements_svi[:,0], bins=bins, alpha=0.5, color='green', edgecolor='black', linewidth=1.2)
    ax.set_xlim([min_,max_])
    ax.set_ylabel('SVI', rotation=0, fontsize=15, labelpad=25)
    ax = axes[2,0]
    n3, _, _  = ax.hist(measurements_vae[:,0], bins=bins, alpha=0.3, color='green', edgecolor='black', linewidth=1.2)
    ax.set_xlim([min_,max_])
    ax.set_ylabel('VAE', rotation=0, fontsize=15, labelpad=25)
    height = max(max(n1), max(n2), max(n3))
    axes[0,0].set_ylim(top=height)
    axes[1,0].set_ylim(top=height)
    axes[2,0].set_ylim(top=height)
    
    ## Log-likelihood
    min_ = min(min(measurements_gibbs[:,1]), min(measurements_svi[:,1]), min(measurements_vae[:,1]))
    max_ = max(max(measurements_gibbs[:,1]), max(measurements_svi[:,1]), max(measurements_vae[:,1]))
    ax = axes[0,1]
    n1, _, _ = ax.hist(measurements_gibbs[:,1], bins=bins,alpha=1.0, color='C0', edgecolor='black', linewidth=1.2)
    ax.set_xlim([min_,max_])
    ax.set_title('Histogram of Log-likelihood', fontsize=15)
    ax = axes[1,1]
    n2, _, _  = ax.hist(measurements_svi[:,1], bins=bins, alpha=0.5, color='C0', edgecolor='black', linewidth=1.2)
    ax.set_xlim([min_,max_])
    ax = axes[2,1]
    n3, _, _  = ax.hist(measurements_vae[:,1], bins=bins, alpha=0.3, color='C0', edgecolor='black', linewidth=1.2)
    ax.set_xlim([min_,max_])
    height = max(max(n1), max(n2), max(n3))
    axes[0,1].set_ylim(top=height)
    axes[1,1].set_ylim(top=height)
    axes[2,1].set_ylim(top=height)

    ## Log-Posterior
    min_ = min(min(measurements_gibbs[:,2]), min(measurements_svi[:,2]), min(measurements_vae[:,2]))
    max_ = max(max(measurements_gibbs[:,2]), max(measurements_svi[:,2]), max(measurements_vae[:,2]))
    ax = axes[0,2]
    n1, _, _ = ax.hist(measurements_gibbs[:,2], bins=bins,alpha=1.0, color='C1', edgecolor='black', linewidth=1.2)
    ax.set_xlim([min_,max_])
    ax.set_title('Histogram of Log-posterior', fontsize=15)
    ax = axes[1,2]
    n2, _, _  = ax.hist(measurements_svi[:,2], bins=bins, alpha=0.5, color='C1', edgecolor='black', linewidth=1.2)
    ax.set_xlim([min_,max_])
    ax = axes[2,2]
    n3, _, _  = ax.hist(measurements_vae[:,2], bins=bins, alpha=0.3, color='C1', edgecolor='black', linewidth=1.2)
    ax.set_xlim([min_,max_])
    height = max(max(n1), max(n2), max(n3))
    axes[0,2].set_ylim(top=height)
    axes[1,2].set_ylim(top=height)
    axes[2,2].set_ylim(top=height)
    plt.show()
    
    
    
    #%%
    std_gibbs = np.zeros((S,N*W))
    std_svi = np.zeros((S, N*W))
    for i in tqdm(range(S)):
        tmp  = X_hat_gibbs[i,:,:]
        std_gibbs[i,:] = np.std(tmp, axis=0)
        
        tmp  = X_hat_svi[i,:,:]
        std_svi[i,:] = np.std(tmp, axis=0)
    

    std_std_gibbs = np.std(std_gibbs, axis=0)
    mean_std_gibbs = np.mean(std_gibbs, axis=0)
    
    std_std_svi = np.std(std_svi, axis=0)
    mean_std_svi = np.mean(std_svi, axis=0)

    
    #%%
    fig, ax = plt.subplots(figsize=(10,10))
    idx = 0
    x = np.linspace(min(std_gibbs[idx,:]), max(std_gibbs[idx,:]), 100)
    ax.plot(x,x, color='grey', alpha=0.5)
    ax.scatter(std_gibbs[idx,:], std_svi[idx,:], s=0.5)
    ax.set(title='Comparison of the std of the value of reconstructed $\hat{X}$ for 1 Map',
           xlabel='Std for the Gibbs method',
           ylabel='Std for the SVI method')
    plt.show()
    
    #%%
    fig, ax = plt.subplots(figsize=(10,10))
    x = np.linspace(min(mean_std_gibbs), max(mean_std_gibbs), 100)
    ax.plot(x,x, color='grey', alpha=0.5)
    ax.scatter(mean_std_gibbs, mean_std_svi, s=0.5)
    ax.set(title='Comparison of the std of the value of reconstructed $\hat{X}$ \n (mean for 100 maps)',
           xlabel='Std for the Gibbs method',
           ylabel='Std for the SVI method')
    plt.show()
    
    
    #%%
    fig, ax = plt.subplots(figsize=(10,10))
    x = np.linspace(min(mean_std_gibbs), max(mean_std_gibbs), 100)
    ax.plot(x,x, color='grey', alpha=0.5)
    ax.scatter(mean_std_gibbs, mean_std_svi, s=0.5)
    ax.errorbar(x=mean_std_gibbs[::500], y=mean_std_svi[::500], yerr=std_std_svi[::500],
                lw=1, alpha=0.5, color='green', ecolor='red', fmt="o", capsize=1)
    ax.set(title='Comparison of the std of the value of reconstructed $\hat{X}$',
           xlabel='Std for the Gibbs method',
           ylabel='Std for the SVI method')
    plt.show()
    
    #%%
    fig, ax = plt.subplots(figsize=(10,10))
    x = np.linspace(min(mean_std_gibbs), max(mean_std_gibbs), 100)
    ax.plot(x,x, color='grey', alpha=0.5)
    ax.scatter(mean_std_gibbs, mean_std_svi, s=0.5)
    ax.errorbar(x=mean_std_gibbs[::500], y=mean_std_svi[::500], xerr=std_std_gibbs[::500],
                lw=1, alpha=0.5, color='green', ecolor='red', fmt="o", capsize=1)
    ax.set(title='Comparison of the std of the value of reconstructed $\hat{X}$',
           xlabel='Std for the Gibbs method',
           ylabel='Std for the SVI method')
    plt.show()
    

    #%%
    fig, ax = plt.subplots(figsize=(10,10))
    x = np.linspace(min(mean_std_gibbs), max(mean_std_gibbs), 100)
    ax.plot(x,x, color='grey', alpha=0.5)
    ax.scatter(mean_std_gibbs, mean_std_svi, s=0.5)
    ax.errorbar(x=mean_std_gibbs[::500], y=mean_std_svi[::500],
                xerr=std_std_gibbs[::500], yerr=std_std_svi[::500],
                lw=1, alpha=0.5, color='green', ecolor='red', fmt="o", capsize=2)
    ax.set(title='Comparison of the std of the value of reconstructed $\hat{X}$',
           xlabel='Std for the Gibbs method',
           ylabel='Std for the SVI method')
    plt.show()
    
    #%%
    A_std_gibbs = np.std(A_gibbs, axis=1)
    A_mean_std_gibbs = np.mean(A_std_gibbs, axis=0)
    A_std_std_gibbs = np.std(A_std_gibbs, axis=0)

    A_std_svi = np.std(A_svi, axis=1)
    A_mean_std_svi = np.mean(A_std_svi, axis=0)
    A_std_std_svi = np.std(A_std_svi, axis=0)
    
    
    fig, ax = plt.subplots(figsize=(10,10))
    x = np.linspace(min(A_mean_std_gibbs), max(A_mean_std_gibbs), 100)
    #ax.plot(x,x, color='grey', alpha=0.5)
    ax.scatter(A_mean_std_gibbs, A_mean_std_svi)
    #ax.errorbar(x=A_mean_std_gibbs, y=A_mean_std_svi,
    #            xerr=A_std_std_gibbs, yerr=A_std_std_svi,
    #            lw=1, alpha=0.5, color='green', ecolor='red', fmt="o", capsize=2)
    ax.set(title='Comparison of the std of the parameter A',
           xlabel='Std for the Gibbs method',
           ylabel='Std for the SVI method')
    plt.show()


    #%%
    B_std_gibbs = np.std(B_gibbs, axis=1)
    B_mean_std_gibbs = np.mean(B_std_gibbs, axis=0)
    B_std_std_gibbs = np.std(B_std_gibbs, axis=0)

    B_std_svi = np.std(B_svi, axis=1)
    B_mean_std_svi = np.mean(B_std_svi, axis=0)
    B_std_std_svi = np.std(B_std_svi, axis=0)
    
    fig, ax = plt.subplots(figsize=(10,10))
    x = np.linspace(min(B_mean_std_gibbs), max(B_mean_std_gibbs), 100)
    ax.plot(x,x, color='grey', alpha=0.5)
    ax.scatter(B_mean_std_gibbs, B_mean_std_svi)
    #ax.errorbar(x=B_mean_std_gibbs[::10], y=B_mean_std_svi[::10],
    #            xerr=B_std_std_gibbs[::10], yerr=B_std_std_svi[::10],
    #            lw=1, alpha=0.5, color='green', ecolor='red', fmt="o", capsize=2)
    ax.set(title='Comparison of the std of the parameter B',
           xlabel='Std for the Gibbs method',
           ylabel='Std for the SVI method')
    plt.show()
    
    
    #%%
    sigma2_std_gibbs = np.std(sigma2_gibbs, axis=1)
    sigma2_std_svi = np.std(sigma2_svi, axis=1)
    
    fig, ax = plt.subplots()
    ax.boxplot([sigma2_std_gibbs, sigma2_std_svi])
    ax.set_xticklabels(['Gibbs', 'SVI'])
    ax.set(title=r'Standard deviation of parameter $\sigma^2$',
           xlabel='Method',
           ylabel='Std of the noise distribution $\sigma^2$')
    plt.show()
    
    
    #%%
    fig, ax = plt.subplots()
    for i in range(200):
        x = B_gibbs[-1, :, i]
        y = B_svi[-1,:,i]
        ax.scatter(x, y, s=1)
    # Line
    #x_ = np.linspace(min(x), max(x), 2)
    x_ = np.linspace(0,3,2)
    ax.plot(x_,x_, color='grey', ls='--', alpha=0.5)
    ax.set(title='Value of B for 200 different samples (1 map)',
           xlabel='Value in Gibbs', ylabel='Value in SVI')
    plt.show()
    
    #%%
    fig, ax = plt.subplots()
    for i in range(200):
        x = B_gibbs[:, 0, i]
        y = B_svi[:, 0, i]
        ax.scatter(x, y, s=1)
    # Line
    #x_ = np.linspace(min(x), max(x), 2)
    x_ = np.linspace(0,6,2)
    ax.plot(x_,x_, color='grey', alpha=0.5)
    ax.set(title='Value of B for 100 maps (1 sample)',
           xlabel='Value in Gibbs', ylabel='Value in SVI')
    plt.show()

    
    #%%
    fig, ax = plt.subplots()
    for i in range(25):
        x = A_gibbs[-1, :, i]
        y = A_svi[-1,:,i]
        ax.scatter(x, y, s=1)
    # Line
    #x_ = np.linspace(min(x), max(x), 2)
    x_ = np.linspace(0.06,0.11,2)
    ax.plot(x_,x_, color='grey', ls='--', alpha=0.5)
    ax.set(title='Value of A for 200 different samples (1 map)',
           xlabel='Value in Gibbs', ylabel='Value in SVI')
    plt.show()

    
    
    #%%
    # LogNormal parameters can't be too large
    mu = 5
    log_sigma = 1.5
    sigma = 1.5
    log_sigma = np.log(sigma)
    sigma = np.exp(log_sigma)
    dist = lognorm(s=sigma, scale=np.exp(mu))
    x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 100)
    fig, ax = plt.subplots()
    ax.plot(x, dist.pdf(x), 'k-', lw=2, label='frozen pdf')
    ax.set(title=r'LogNormal PDF $\mu$={}, $\log\sigma$={}'.format(mu, log_sigma))
    plt.show()

    #%%
    def generalized_sigmoid(x, a, b, scale_factor):
        return a + (b - a) / (1 + torch.exp(-scale_factor * x))
    
    x = torch.linspace(-10, 4, 1000)
    sigmoid = nn.Sigmoid()
    a = 10e-5
    b = 1.5
    y = generalized_sigmoid(x, a, b, 0.5)
    y2 = torch.exp(x)
    plt.plot(x,y)
    plt.plot(x,y2)
    plt.show()


# %%
