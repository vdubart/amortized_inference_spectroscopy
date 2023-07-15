import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_SERS(X, figsize=(10,4)):
    # SERS Matrix
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(X, cmap='viridis', interpolation='none')
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Spectrum index')
    ax.set_title('X')
    ax.grid(False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    fig.tight_layout()
    plt.show()


def plot_slice(X, w, figsize=(8,8)):
    N = X.shape[0]
    # Map at wavenumber w
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    mappable = ax.imshow(X[:, int(w)].reshape((np.sqrt(N), np.sqrt(N))),
                         cmap='viridis', interpolation='none')
    ax.set_title(f'w={w}')
    ax.grid(False)
    plt.colorbar(mappable=mappable, ax=ax)
    fig.tight_layout()
    plt.show()


def plot_spectrum(X, n, figsize=(8,8)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(X[n,:])
    ax.set_xlabel('Wavenumber')
    ax.set_title('Spectrum at index {}'.format(n))
    fig.tight_layout()
    plt.tight_layout()
    plt.show()


def plot_highest_spectrum(X, figsize=(8,8)):
    W = X.shape[1]
    n = np.argmax(X)//W
    plot_spectrum(X, n, figsize)


def plot_full(gen, X, figsize=(8,8)):
    fig, axs = plt.subplots(2,2, figsize=figsize)

    # SERS Matrix
    axs[0,0].imshow(X, cmap='viridis', interpolation='none')
    axs[0,0].set_xlabel('Wavenumber')
    axs[0,0].set_ylabel('Spectrum index')
    axs[0,0].set_title('X')
    axs[0,0].grid(False)

    # Map at peak 1
    mappable = axs[0,1].imshow(X[:, int(round(gen.c[0]))].reshape(*gen.mapsize), cmap='viridis', interpolation='none')
    axs[0,1].set_title(f'c={round(gen.c[0])}')
    axs[0,1].grid(False)
    plt.colorbar(mappable=mappable, ax=axs[0,1])

    # One spectrum
    idx = np.argmax(X)//gen.Nw
    axs[1,0].plot(X[idx,:])
    axs[1,0].set_xlabel('Wavenumber')
    axs[1,0].set_title('Highest spectrum at index {}'.format(idx))

    # Pure Voigt of peak 1
    for i in range(gen.Vp.shape[0]):
        axs[1,1].plot(gen.Vp[i,:], label='Pure Voigt {}'.format(i+1))
    axs[1,1].set_xlabel('Wavenumber')
    axs[1,1].set_title('Pure Voigt')
    if gen.Vp.shape[0] > 1:
        axs[1,1].legend()

    fig.tight_layout()
    plt.tight_layout()
    plt.show()


def plot_reconstruction(X, X_hat, cmap='Greys', figsize=(10,20)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # Original 
    ax = axes[0]
    im = ax.imshow(X, cmap=cmap, interpolation='none')
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Spectrum index')
    ax.set_title(r'Original $X$')
    #ax.yaxis.set_major_locator(MultipleLocator(100))
    #ax.grid(which='major', axis='both')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # Reconstructed
    ax = axes[1]
    im = ax.imshow(X_hat, cmap=cmap, interpolation='none')
    ax.set_xlabel('Wavenumber')
    #ax.set_ylabel('Spectrum index')
    ax.set_title(r'Reconstructed $X$')
    #ax.yaxis.set_major_locator(MultipleLocator(100))
    #ax.grid(which='major', axis='both')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    plt.tight_layout()
    plt.show()


def plot_reconstruction_3d(X, X_hat, figsize=(20,10),  elev=70, angle=270, grid=True, transparent=False):
    N, W = X.shape
    fig, axes = plt.subplots(1, 2,  figsize=figsize, subplot_kw={"projection": "3d"})
    xline = np.arange(W)
    
    # Original
    ax = axes[0]
    for i in range(N):
        zline = X[i,:]
        yline = np.ones((W,)) * i
        ax.plot3D(xline, yline, zline, 'black', lw=1, alpha=1)
    ax.set(xlabel='Wavenumber', ylabel='Spectrum index', zlabel='Intensity')
    ax.view_init(elev, angle, vertical_axis='z')
    ax.grid(grid) 
    ax.set_zlim(top=1.2)
    if transparent:
        ax.w_xaxis.pane.fill = False
        ax.w_yaxis.pane.fill = False
        ax.w_zaxis.pane.fill = False
    
    # Reconstructed
    ax = axes[1]
    for i in range(N):
        zline = X_hat[i,:]
        yline = np.ones((W,)) * i
        ax.plot3D(xline, yline, zline, 'black', lw=1, alpha=1)
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


def plot_data_3d(X, figsize=(7,7), elev=70, angle=270, grid=True, transparent=False):
    N, W = X.shape
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)
    xline = np.arange(W)
    yline = np.arange(N)
    for i in range(N):
        zline = X[i,:]
        yline = np.ones((W,)) * i
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


def plot_surface_3d(X, elev=10, angle=220, grid=True, transparent=False):
    N, W = X.shape
    XX, YY = np.meshgrid(np.arange(W), np.arange(N))
    ZZ = X
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(XX, YY, ZZ, cmap="viridis", linewidth=0, antialiased=False)
    ax.set(xlabel='Wavenumber', ylabel='Spectrum index')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev, angle, vertical_axis='z')
    ax.grid(grid) 
    if transparent:
        ax.w_xaxis.pane.fill = False
        ax.w_yaxis.pane.fill = False
        ax.w_zaxis.pane.fill = False
    plt.tight_layout()
    plt.show()