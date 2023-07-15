
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from SERSGenerator import SERSGenerator
from plot_map import *


def generate_map(mapsize, W, threshold=0.5, seed=None):
    gen = SERSGenerator(mapsize=mapsize, Nw=W, seed=seed)
    X = gen.generate(N_hotspots=2, K=1, sig=0.05, sbr=1, plot=False, background='none')
    X = X.astype('float32')

    if np.max(X) < threshold: # Try again
        return generate_map(mapsize, W, threshold, seed)

    else:  # Record map if there is a sufficient signal
        peak_data = int(np.argmax(np.max(X, axis=0)))
        peak_truth = float(gen.c[0])
        label = [peak_data, peak_truth]
        return X, label
    

def create_dataset(mapsize, W, S, threshold=0.5, seed=None):
    np.random.seed(seed)
    #dataset = np.zeros((S, mapsize[0]*mapsize[1], W))
    #labels = np.zeros((S, 2))
    dataset = {}
    i = 0
    n = 0
    while(n < S):
        # Generate one map
        gen = SERSGenerator(mapsize=mapsize, Nw=W, seed=i)
        X = gen.generate(N_hotspots=2, K=1, sig=0.05, sbr=1, plot=False, background='none')
        X = X.astype('float32')
        
        # Record map if there is a sufficient signal
        if np.max(X) > threshold:
            dataset[n] = {}
            dataset[n]['seed'] = i
            dataset[n]['X'] = X
            dataset[n]['c'] = float(gen.c[0])
            dataset[n]['c_data'] = int(np.argmax(np.max(X, axis=0)))
            dataset[n]['alpha'] = gen.alpha
            dataset[n]['Vp'] = gen.Vp
            
            n = n + 1
            #peak_data = int(np.argmax(np.max(X, axis=0)))
            #peak_truth = float(gen.c[0])
            #dataset[n,:,:] = X
            #labels[n,:] = np.array([peak_data, peak_truth])

        i = i + 1
    
    return dataset


def save_dataset(filename, dataset):
    pickle.dump(dataset, open(filename, "wb"))
    return


def load_dataset(filename):
    dataset = pickle.load(open(filename, "rb"))
    return dataset


#%%
if __name__ == "__main__":
    print('Main')
    
    #%%
    # Create dataset
    mapsize = (5,5)
    W = 200
    S = 100
    #testset, labels = create_dataset(mapsize, W, S, threshold=0.5, seed=0)
    dataset = create_dataset(mapsize, W, S, threshold=0.5, seed=2)
    
    #%%
    # Visualize peaks
    peaks = [dataset[n]['c'] for n in range(S)]
    #peaks = labels[:,0]
    fig, ax  = plt.subplots(figsize=(10,3))
    sns.stripplot(x=peaks, size=5, ax=ax, jitter=0.25)
    ax.set(title='Jitter plot of dataset peak', xlabel='Wavenumber', ylabel='Jitter')
    ax.set_xlim([0,W])
    plt.tight_layout()
    plt.show()

    fig, ax  = plt.subplots(figsize=(10,3))
    bin_width = 10
    bin_edges = np.arange(0, W+bin_width, bin_width)
    ax.hist(peaks, bins=bin_edges, alpha=0.7, color='C0',
             edgecolor='black', linewidth=1.2)
    ax.set(title='Histogram plot of dataset peak', xlabel='Wavenumber', ylabel='Count')
    ax.set_xlim([0,W])
    plt.tight_layout()
    plt.show()


    #%%
    filename = 'SERSTestsetNew.pkl'
    save_dataset(filename, dataset)
    dataset = load_dataset(filename)
    
    #%%
    # Example map background
    gen = SERSGenerator(mapsize=(10,10), Nw=W, seed=10)
    X = gen.generate(N_hotspots=2, K=2, sig=0.05, sbr=1, plot=False, background='default')
    n = np.argmax(X)//W
    N = X.shape[0]
    c1 = int(round(gen.c[0]))
    c2 = int(round(gen.c[1]))
    
    # Imshow
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    im = ax.imshow(X, cmap='Greys', interpolation='none')
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Spectrum index')
    ax.set_title('SERS map', fontsize=16)
    ax.grid(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.axhline(y=n, xmin=0, xmax=W, color='blue', ls='--')
    ax.axvline(x=c1, ymin=0, ymax=N, color='C1', ls='--')
    ax.axvline(x=c2, ymin=0, ymax=N, color='C1', ls='--')
    plt.colorbar(im, cax=cax, label='Intensity', fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()

    # 3D
    elev=67.5
    angle=270
    grid=False
    transparent=False
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7,7))
    xline = np.arange(W)
    yline = np.arange(N)
    for i in range(N):

        zline = X[i,:]
        yline = np.ones((W,)) * i
        if i == n:
            ax.plot3D(xline, yline, zline, 'blue', lw=1, alpha=1)
        elif i%3 == 0:
            ax.plot3D(xline, yline, zline, 'black', lw=1, alpha=0.5)
        else:
            continue
    
    ax.set(xlabel='Wavenumber', ylabel='Spectrum index', zlabel='Intensity')
    ax.view_init(elev, angle, vertical_axis='z')
    ax.grid(grid)
    ax.set_zlim(top=1.2)
    if transparent:
        ax.w_xaxis.pane.fill = False
        ax.w_yaxis.pane.fill = False
        ax.w_zaxis.pane.fill = False
    plt.tight_layout() 
    plt.show()

    # highest spectrum
    fig, ax = plt.subplots(figsize=(5,5))
    signal = X[n,:]
    ax.plot(signal)
    plt.plot(gen.BB[n,:], color='grey', ls='--')
    ax.axvline(x=c1, color='C1', ls='--', alpha=0.5)
    ax.axvline(x=c2,color='C1', ls='--', alpha=0.5)
    ax.set_xlabel('Wavenumber', fontsize=12)
    ax.set_title('Spectrum at index {}'.format(n), fontsize=16)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_xlim([0,W])
    plt.tight_layout() 
    plt.show()
    
    fig, ax = plt.subplots(figsize=(5,5))
    for i in range(gen.Vp.shape[0]):
        ax.plot(gen.Vp[i,:], color='C0')
    ax.set_xlabel('Wavenumber', fontsize=12)
    ax.set_title('Pure Voigt line shape', fontsize=16)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.axvline(x=c1, color='C1', ls='--', alpha=0.5)
    ax.axvline(x=c2,color='C1', ls='--', alpha=0.5)
    ax.set_xlim([0,W])
    plt.tight_layout() 
    plt.show()
    
    fig, ax = plt.subplots(figsize=(5,5))
    mappable = ax.imshow(X[:, int(round(gen.c[1]))].reshape(*gen.mapsize), cmap='Greys', interpolation='none')
    ax.set_title(f'Map for wavenumber {round(gen.c[0])}', fontsize=16)
    ax.grid(False)
    plt.colorbar(mappable=mappable, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout() 
    plt.show()
    
    
    
    #%%
    gen = SERSGenerator(mapsize=mapsize, Nw=W, seed=2)
    X = gen.generate(N_hotspots=2, K=1, sig=0.05, sbr=1, plot=False, background='none')
    n = np.argmax(X)//W
    N = X.shape[0]
    c1 = int(round(gen.c[0]))
    
    # Imshow
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    im = ax.imshow(X, cmap='Greys', interpolation='none')
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Spectrum index')
    ax.set_title('SERS map', fontsize=16)
    ax.grid(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.axhline(y=n, xmin=0, xmax=W, color='blue', ls='--')
    ax.axvline(x=c1, ymin=0, ymax=N, color='C1', ls='--')
    plt.colorbar(im, cax=cax, label='Intensity', fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()

    # 3D
    elev=67.5
    angle=270
    grid=False
    transparent=False
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7,7))
    xline = np.arange(W)
    yline = np.arange(N)
    for i in range(N):
        zline = X[i,:]
        yline = np.ones((W,)) * i
        if i == n:
            ax.plot3D(xline, yline, zline, 'blue', lw=1, alpha=1)
        else:
            ax.plot3D(xline, yline, zline, 'black', lw=1, alpha=0.5)

    ax.set(xlabel='Wavenumber', ylabel='Spectrum index', zlabel='Intensity')
    ax.view_init(elev, angle, vertical_axis='z')
    ax.grid(grid)
    ax.set_zlim(top=1.2)
    if transparent:
        ax.w_xaxis.pane.fill = False
        ax.w_yaxis.pane.fill = False
        ax.w_zaxis.pane.fill = False
    plt.tight_layout() 
    plt.show()

    # highest spectrum
    fig, ax = plt.subplots(figsize=(5,5))
    signal = X[n,:]
    ax.plot(signal)
    plt.plot(gen.BB[n,:], color='grey', ls='--')
    ax.axvline(x=c1, color='C1', ls='--', alpha=0.5)
    ax.set_xlabel('Wavenumber', fontsize=12)
    ax.set_title('Spectrum at index {}'.format(n), fontsize=16)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_xlim([0,W])
    plt.tight_layout() 
    plt.show()
    
    fig, ax = plt.subplots(figsize=(5,5))
    for i in range(gen.Vp.shape[0]):
        ax.plot(gen.Vp[i,:], color='C0')
    ax.set_xlabel('Wavenumber', fontsize=12)
    ax.set_title('Pure Voigt line shape', fontsize=16)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.axvline(x=c1, color='C1', ls='--', alpha=0.5)
    ax.set_xlim([0,W])
    plt.tight_layout() 
    plt.show()
    
    fig, ax = plt.subplots(figsize=(5,5))
    mappable = ax.imshow(X[:, int(round(gen.c[0]))].reshape(*gen.mapsize), cmap='Greys', interpolation='none')
    ax.set_title(f'Map for wavenumber {round(gen.c[0])}', fontsize=16)
    ax.grid(False)
    plt.colorbar(mappable=mappable, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout() 
    plt.show()
# %%
