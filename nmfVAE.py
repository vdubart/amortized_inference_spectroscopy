#%%
import torch
import torch.nn as nn
from torch.distributions import LogNormal
from torch.optim import Adam
from tqdm import tqdm
import datetime
import pickle


from SERSDataset import generate_map
from plot_map import *
from nmfSVI import LikelihoodModelNMF, PriorModelNMF


class VAE_NMF(nn.Module):
    def __init__(self, N, W):
        super(VAE_NMF, self).__init__()
        self.N = N
        self.W = W
        self.input_dim = N*W  # Flattened map
        self.latent_dim = N+W+1
        self.lower = 10e-5         # min value for sigma of LogNormal
        self.upper = 1.5           # max value for sigma of LogNormal
        self.scale_factor = 0.5
        # encoder NN
        self.fc1 = nn.Linear(self.input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc31 = nn.Linear(512, self.latent_dim)
        self.fc32 = nn.Linear(512, self.latent_dim)
        
    def generalized_sigmoid(self, x, lower, upper, scale_factor):
        return lower + (upper - lower) / (1 + torch.exp(-scale_factor * x))
        
    def encoder(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.fc31(h)
        logit_sigma = self.fc32(h)
        sigma = self.generalized_sigmoid(logit_sigma, self.lower, self.upper, self.scale_factor)
        return mu, sigma
    
    def rsample(self, mu, sigma, size):
        distr = LogNormal(mu, sigma)
        z = distr.rsample(size)
        return z.squeeze(1)
    
    def forward(self, x, num_samples):
        x = x.view(-1, self.N*self.W)
        mu, sigma = self.encoder(x)  
        z = self.rsample(mu, sigma, (num_samples,))
        return mu, sigma, z
    
    def variational_log_prob(self, z, mu, sigma):
        distr = LogNormal(mu, sigma)
        return distr.log_prob(z).sum()


#%%
def nmf_VAE(N, W, num_epochs, num_samples, hyperparams, plot_every=1000, save=False, save_every=10000):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mapsize = (int(np.sqrt(N)), int(np.sqrt(N)))
    # Define models
    prior_model = PriorModelNMF(N, W, alpha=hyperparams['alpha'], beta=hyperparams['beta'],
                             k=hyperparams['k'], theta=hyperparams['theta'], device=device)
    prior_model = prior_model.to(device)
    likelihood_model = LikelihoodModelNMF(N, W)
    likelihood_model = likelihood_model.to(device)
    vae = VAE_NMF(N, W)
    vae = vae.to(device)

    # Define the optimizer
    optimizer = Adam(vae.parameters(), lr=0.00001) 
    #0.0001 oK --> rsample z element 0 --> log_prob(z) invalid
    #0.01 BAD --> log_sigma too large --> rsample -> inf/0 --> nan/inf Xhat
    # --> clamp log_sigma?? --> nan from the gradient?
    # 0.00001 OK
    losses = []
    elbos = []
    log_likelihood = []
    log_posterior = []
    log_prior = []
    log_variational = []
    trace_mu = torch.zeros(10, N+W+1).to(device) # num_epochs
    trace_sigma = torch.zeros(10, N+W+1).to(device)
    trace_z = torch.zeros(10, N+W+1).to(device)
    peaks = []
    # Perform stochastic optimization
    for epoch in tqdm(range(num_epochs), desc="Training VAE"):
        try:
            # Generate new map
            X, label = generate_map(mapsize, W, threshold=0.5, seed=None)
            X = torch.tensor(X)
            X = X.to(device)
            peaks += [label[0]]
            
            # Loop over training set:
            with torch.enable_grad():
                
                optimizer.zero_grad()
                
                mu, sigma, z = vae(X, num_samples)
                
                #print(epoch, mu.shape, sigma.shape, z.shape)
                #print(epoch, torch.isnan(z).sum(), torch.isinf(z).sum())

                trace_mu[epoch%10] = mu.squeeze(0).detach()
                trace_sigma[epoch%10] = sigma.squeeze(0).detach()
                trace_z[epoch%10] = z.squeeze(0).detach()
                
                # Compute the objective function -- Monte Carlo approximation
                log_like = likelihood_model(X, z)
                #print('log_like', log_like)
                log_q = vae.variational_log_prob(z, mu, sigma)
                #print('log_q', log_q)
                log_p = prior_model(z)
                #print('log_p', log_p)

                ELBO = (1.0/num_samples) * (log_like - log_q + log_p)
                loss = - ELBO
            
                # Update the NN encoder parameters
                loss.backward()
                optimizer.step()
                    
            # Print progress
            if (epoch+1) % plot_every == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
                _, _, _, AB = likelihood_model.reconstruct(z)
                X_hat = AB.squeeze(0).detach().to('cpu').numpy()
                plot_reconstruction_3d(X.to('cpu').numpy(), X_hat, figsize=(15,10),
                                    elev=67.5, angle=270, grid=False, transparent=False)
        
            if save and (epoch+1) % save_every == 0: 
                    ## Save checkpoint
                    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
                    filename = "vae_models/"+timestamp+"_epoch_{}.pkl".format(epoch)
                    torch.save(vae.state_dict(), filename)

            # Record
            losses += [loss.item()]
            elbos += [ELBO.item()]
            log_likelihood += [log_like.item()]
            log_posterior += [log_like.item() + log_p.item()]
            log_prior += [log_p.item()]
            log_variational += [log_q.item()]
            
        except Exception as e:
            print(f"Exception occurred: {e}")
            break
        
    out = {'prior_model': prior_model,
            'likelihood_model': likelihood_model,
            'vae': vae,
            'log_likelihood': log_likelihood,
            'log_posterior': log_posterior,
            'losses': losses,
            'ELBOs': elbos,
            'peaks': peaks,
            'mu': trace_mu,
            'sigma': trace_sigma,
            'z': trace_z}
        
    ## Save model:
    if save: 
        timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        filename = "vae_models/"+timestamp+"_epoch_{}_DONE.pkl".format(epoch)
        torch.save(vae.state_dict(), filename)
        save_training_stats("vae_models/"+timestamp+"_epoch_{}_OUT.pkl".format(epoch), out,
                            ['log_likelihood', 'log_posterior', 'losses', 'ELBOs', 'peaks'])
    
    return out


def save_training_stats(filename, out, keys):
    sub_out = {k: out[k] for k in keys}
    pickle.dump(sub_out, open(filename, "wb"))
    return


def load_training_stats(filename):
    out = pickle.load(open(filename, "rb")) 
    return out


#%%
if __name__ == "__main__":
    
    N = 25
    W = 200
    mapsize = (int(np.sqrt(N)), int(np.sqrt(N)))
    hyperparams = {'alpha':10, 'beta':1, 'k':1, 'theta':1}

    num_epochs = 50000
    num_samples = 1

    # Training VAE
    out = nmf_VAE(N, W, num_epochs, num_samples, hyperparams, plot_every=1000, save=True, save_every=10000)
    

# %%
