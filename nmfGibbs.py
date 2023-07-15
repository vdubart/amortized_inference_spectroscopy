import numpy as np
from scipy.special import erfc, erfcinv
from scipy.stats import norm, expon, invgamma
from tqdm import tqdm
import pickle


def randr(m, s, l):
    """ Draw random numbers from rectified normal density
    p(x)=K*exp(-(x-m)^2/s-l'x), x>=0 

    Input:
        m           Means  (N,)
        s           Variance
        l           Scales (N,)
    Output:
        x           Random numbers sampled (N,)

    Copyright 2007 Mikkel N. Schmidt, ms@it.dk, www.mikkelschmidt.dk
    """
    A = (l * s - m) / np.sqrt(2 * s)
    a = A > 26
    x = np.zeros_like(m)

    y = np.random.rand(*m.shape)
    x[a] = -np.log(y[a]) / ((l[a] * s - m[a]) / s)

    R = erfc(np.abs(A[~a]))
    x[~a] = erfcinv(y[~a] * R - (A[~a] < 0) * (2 * y[~a] + R - 2)) * np.sqrt(2 * s) + m[~a] - l[~a] * s

    x[np.isnan(x)] = 0
    x[x < 0] = 0
    x[np.isinf(x)] = 0
    x = x.real

    return x


def nmf_gibbs(X, K, M, hyperparams, verbose=False):
    """ Non-negative matrix factorization Gibbs sampler

    Args:
        X       Data matrix (IxJ)
        K       Number of components (int>0)
        M       Number of Gibbs samples to compute (int>0)
        hyperparams: dictionary containing keys: 
            alpha   Prior scale for A (float > 0)     (Exp. distr.)
            beta    Prior scale for B (float > 0)     (Exp. distr.)
            theta   Prior scale for sigma squared (float > 0) (Gamma distr.)
            k       Prior shape for sigma squared (float > 0) (Gamma distr.)
        verbose Bool to display loading bar

    Returns:
        trace: dictionnary containing keys:
            As      Trace of samples of A (I x K x M)
            Bs      Trace of samples of B (K x J x M)
            sigma2s Trace of samples of sigma squared (M)
    """
    I, J = X.shape

    # Transform input parameters
    alpha = hyperparams["alpha"]
    beta = hyperparams["beta"]
    theta = hyperparams["theta"]
    k = hyperparams["k"]
    
    alpha = alpha * np.ones((I,K))
    beta = beta * np.ones((K,J))
    
    # Initialization
    A = np.random.rand(I, K) #Initial value for A (I x K)
    B = np.random.rand(K, J) #Initial value for B (K x J)
    sigma2 = 0.001           #Initial value for noise variance (sigma^2)

    # Containers for all samples
    As = np.zeros((I, K, M))
    Bs = np.zeros((K, J, M))
    sigma2s = np.zeros((M,))
    
    # Put initial value
    As[:,:,0] = A
    Bs[:,:,0] = B
    sigma2s[0]= sigma2
     
    # Loop over gibbs samples
    for m in tqdm(range(1,M), desc="Gibbs sampling", disable=not verbose):
    
        # Sample A: Loop over columns of A
        C = B@B.T
        D = X@B.T
        for n in range(K):
            # all columns except n
            nn = np.concatenate((np.arange(n), np.arange(n+1, K)))

            # Sample from rectified normal distr.
            mu = (D[:, n] - (A[:, nn]@C[nn, n]) - (alpha[:,n]*sigma2)) / C[n, n]
            var = sigma2 / C[n, n]
            scale = alpha[:, n]
            A[:, n] = randr(mu, var, scale)

        # Sample sigma squared from inverse gamma distr.
        sigma2 = invgamma.rvs(a=(I*J)/2 + k + 1, loc=0, scale=0.5*np.sum((X-(A@B))**2)+theta)

        # Sample B: Loop over rows of B
        E = A.T@A
        F = A.T@X
        for n in range(K):
            # all rows except n
            nn = np.concatenate((np.arange(n), np.arange(n+1, K)))

            # Sample from rectified normal distr.
            mu = (F[n,:] - (E[n,nn]@B[nn,:]) - (beta[n,:]*sigma2)) / E[n,n] 
            var = sigma2 / E[n,n]
            scale = beta[n,:]
            B[n,:] = randr(mu, var, scale)
        
        # Traces
        As[:,:,m] = A 
        Bs[:,:,m] = B
        sigma2s[m] = sigma2

    return {"A":As,
            "B":Bs,
            "sigma2":sigma2s}


def gibbs_probabilities(X, trace, hyperparams, verbose=False):
    N, W = X.shape
    As = trace["A"]
    Bs = trace["B"]
    sigma2s = trace["sigma2"]
    M = As.shape[2]
    
    ABs = np.zeros((N, W, M))
    log_likelihood = np.zeros((M,))
    log_posterior = np.zeros((M,))
    log_prior_A = np.zeros((M,))
    log_prior_B = np.zeros((M,))
    log_prior_sigma2 = np.zeros((M,))
    
    # Loop over gibbs samples
    for m in tqdm(range(M), desc="Computing probabilities", disable=not verbose):
        A = As[:,:,m]
        B = Bs[:,:,m]
        sigma2 = sigma2s[m]
        
        # Log-Likelihood p(X|AB,sigma2)
        AB = A@B
        ABs[:,:,m] = AB
        log_likelihood[m]  = np.sum(norm.logpdf(X, loc=AB, scale=sigma2**(1/2)))
        
        # Log-Priors
        log_prior_A[m] = np.sum(expon.logpdf(A, scale=1/hyperparams['alpha']))
        log_prior_B[m] = np.sum(expon.logpdf(B, scale=1/hyperparams['beta']))
        log_prior_sigma2[m] = invgamma.logpdf(sigma2, a=hyperparams['k'], scale=hyperparams['theta'])
                
        # Log-Posterior
        log_posterior[m] = log_likelihood[m] + log_prior_A[m] + log_prior_B[m] + log_prior_sigma2[m]
    
    trace['AB'] = ABs
    trace['log_likelihood'] = log_likelihood
    trace['log_posterior'] = log_posterior
    trace['log_prior_A'] = log_prior_A
    trace['log_prior_B'] = log_prior_B
    trace['log_prior_sigma2'] = log_prior_sigma2
    
    return trace


def save_nmf_gibbs(filename, trace, keys):
    out = {k:trace[k] for k in keys}
    pickle.dump(out, open(filename, "wb"))
    return

def load_nmf_gibbs(filename):
    out = pickle.load(open(filename, "rb")) 
    return out