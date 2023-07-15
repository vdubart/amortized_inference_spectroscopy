#%%
import torch
import torch.nn as nn
from torch.distributions import Normal, LogNormal, Exponential
from torch.optim import Adam
from tqdm import tqdm

from inverse_gamma import InverseGamma

# Define the probabilistic model
class LikelihoodModelNMF(nn.Module):
    def __init__(self, N, W):
        super(LikelihoodModelNMF, self).__init__()
        self.N = N
        self.W = W
        #self.sigma = torch.tensor(sigma)  # Fixed model parameter

    def reconstruct(self, z):
        A = z[:, :self.N]
        B = z[:, self.N:self.N+self.W]
        sigma2 = z[:, -1]
        A = A.unsqueeze(-1)
        B = B.unsqueeze(1)
        AB = torch.bmm(A, B)
        return A, B, sigma2, AB

    def forward(self, x, z):
        _, _, sigma2, AB = self.reconstruct(z)
        distr = Normal(AB, torch.sqrt(sigma2))
        return distr.log_prob(x).sum()

"""
# Define the prior model for matrices A, B and noise sigma^2
class PriorModelNMF(nn.Module):
    def __init__(self, N, W, alpha, beta, k, theta):
        super(PriorModelNMF, self).__init__()
        self.N = N
        self.W = W
        self.alpha = torch.tensor(alpha)  # Fixed model parameter
        self.beta = torch.tensor(beta)  # Fixed model parameter
        # Fixed prior distributions 
        self.distr_A = Exponential(self.alpha)
        self.distr_B = Exponential(self.beta) 
        self.distr_sigma2 = InverseGamma(concentration=k, rate=theta)

    def forward(self, z):
        A = z[:, :self.N]
        B = z[:, self.N:self.N+self.W]
        sigma2 = z[:, -1]
        return self.distr_A.log_prob(A).sum() \
             + self.distr_B.log_prob(B).sum() \
             + self.distr_sigma2.log_prob(sigma2).sum() 
"""
# Define the prior model for matrices A, B and noise sigma^2
class PriorModelNMF(nn.Module):
    def __init__(self, N, W, alpha, beta, k, theta, device='cpu'):
        super(PriorModelNMF, self).__init__()
        self.N = N
        self.W = W
        self.alpha = torch.tensor(alpha)  # Fixed model parameter
        self.beta = torch.tensor(beta)  # Fixed model parameter
        # Fixed prior distributions 
        self.distr_A = Exponential(self.alpha)
        self.distr_B = Exponential(self.beta)
        self.k = torch.tensor(k).to(device)
        self.theta = torch.tensor(theta).to(device)
        self.distr_sigma2 = InverseGamma(concentration=self.k, rate=self.theta)

    def forward(self, z):
        A = z[:, :self.N]
        B = z[:, self.N:self.N+self.W]
        sigma2 = z[:, -1]
        return self.distr_A.log_prob(A).sum() \
             + self.distr_B.log_prob(B).sum() \
             + self.distr_sigma2.log_prob(sigma2).sum() 


# Define the log-normal variational family
class VariationalFamilyNMF(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalFamilyNMF, self).__init__()
        self.latent_dim = latent_dim
        self.mu_var = nn.Parameter(torch.zeros(latent_dim))  # Variational parameter
        self.log_sigma_var = nn.Parameter(torch.zeros(latent_dim))  # Variational parameter

    def rsample(self, size):
        sigma_var = torch.exp(self.log_sigma_var)
        distr = LogNormal(self.mu_var, sigma_var)
        return self.mu_var, self.log_sigma_var, distr.rsample(size)
    
    def forward(self, z):
        sigma_var = torch.exp(self.log_sigma_var)
        distr = LogNormal(self.mu_var, sigma_var)
        return distr.log_prob(z).sum()


def nmf_SVI(X, num_epochs, num_samples, hyperparams, lr=0.01):
    N, W = X.shape

    # Define models
    prior_model = PriorModelNMF(N, W, alpha=hyperparams['alpha'], beta=hyperparams['beta'],
                             k=hyperparams['k'], theta=hyperparams['theta'])
    likelihood_model = LikelihoodModelNMF(N, W)
    variational_family = VariationalFamilyNMF(latent_dim=N+W+1)

    # Define the optimizer
    optimizer = Adam(variational_family.parameters(), lr=lr)

    # Perform stochastic optimization
    losses = []
    elbos = []
    log_likelihood = []
    log_posterior = []
    trace_mu = torch.zeros(num_epochs, N+W+1)
    trace_log_sigma = torch.zeros(num_epochs, N+W+1)
    trace_z = torch.zeros(num_epochs, N+W+1)
    for epoch in tqdm(range(num_epochs), desc="Stochastic optimization"):
        
        optimizer.zero_grad()
        
        # Sample from q
        mu, log_sigma, z = variational_family.rsample((num_samples,))

        # Compute the objective function -- Monte Carlo approximation
        log_like = likelihood_model(X, z)
        log_q = variational_family(z)
        log_p = prior_model(z)

        ELBO = (1.0/num_samples) * (log_like - log_q + log_p)
        loss = - ELBO

        # Update the variational parameters
        loss.backward()
        optimizer.step()

        # Print progress
        #if (epoch + 1) % (num_epochs//10) == 0:
            #print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

        # Record
        losses += [loss.item()]
        elbos += [ELBO.item()]
        log_likelihood += [log_like.item()]
        log_posterior += [log_like.item() + log_p.item()]
        trace_mu[epoch] = mu.detach()
        trace_log_sigma[epoch] = log_sigma.detach()
        trace_z[epoch] = z.detach()


    # Inference completed, disable gradient computation
    for param in variational_family.parameters():
        param.requires_grad = False

    return {'prior_model': prior_model,
            'likelihood_model': likelihood_model,
            'variational_family': variational_family,
            'log_likelihood': log_likelihood,
            'log_posterior': log_posterior,
            'losses': losses,
            'ELBOs': elbos,
            'mu': trace_mu,
            'log_sigma': trace_log_sigma,
            'z': trace_z}
    
    
def save_nmf_svi(filename, model):
    torch.save(model.state_dict(), filename)
    return


def load_nmf_svi(filename):
    N = 25
    W = 200
    model = VariationalFamilyNMF(latent_dim=N+W+1)
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model