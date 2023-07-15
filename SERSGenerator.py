"""
Copyright © 2023 David Frich Hansen (dfha@dtu.dk), Tommy Sonne Alstrøm (tsal@dtu.dk)

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the “Software”), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""



#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
import logging


class SERSGenerator:
    def __init__(self, mapsize, Nw, seed=None, c=None, gamma=None, eta=None):
        """Object to generate synthetic SERS data.
        Args:
            mapsize (iterable): Iterable (tuple, list etc.) of length 2 - physical size of the substrate (pixels x pixels)
            Nw (int): Number of wavenumbers (Raman shift) as np.arange(Nw)
            seed (int): Seed for RNG
            c (np.ndarray): If peak positions are fixed, put them here
            gamma (np.ndarray): If peak widths (FWHM) are fixed, put them here
            eta (np.ndarray): If Lorenzianities are fixed, put them here
        """
        self.mapsize = mapsize
        self.Nw = Nw
        self.w = np.arange(Nw)

        self.sbr = None

        self.seed = seed
        if c is not None and gamma is not None and eta is not None:
            assert len(c) == len(gamma) == len(eta)
            self.K = len(c)

            self.c = c
            self.gamma = gamma
            self.eta = eta

        else:
            self.c = None
            self.gamma = None
            self.eta = None
            self.K = None

    def pseudo_voigt(self, w, c, gamma, eta):
        """Computes the (scaled) Pseudo-Voigt function over wavenumbers w with parameters c, gamma, eta efficiently.

        V = eta * Lorentzian + (1-eta) * Gaussian

        Args:
            w (np.array): Measured wavenumbers (sorted) (W x 1)
            c (np.array): Centers of Voigt curves (K x 1)
            gamma (np.array): Full-width-at-half-maximum (FWHM) of Voigt curves (K x 1)
            eta (np.array): Mixture coefficients of Lorentzian and Gaussian (0 <= eta <= 1) (K x 1)

        Returns:
            np.ndarray: Computed Voigt curves, (K x W)
        """
        K = len(c)
        assert len(gamma) == K and len(eta) == K
        W = len(w)

        xdata = np.tile(w, (K, 1))
        c_arr = np.tile(c, (W, 1)).T
        gamma_arr = np.tile(gamma, (W, 1)).T
        eta_arr = np.tile(eta, (W, 1)).T

        diff = xdata - c_arr
        kern = diff / gamma_arr
        diffsq = kern * kern

        # lorentzian
        L = 1 / (1 + diffsq)

        # gaussian
        G_kern = -0.5 * diffsq
        G = np.exp(G_kern)

        # vectorized voigts
        Vo = eta_arr * L + (1 - eta_arr) * G

        return Vo

    def generate(self, N_hotspots, K, sig, sbr, alpha=None, p_outlier=0, background='default', plot=True, smooth=False, verbose=False,
                 *bgargs,
                 **bgkwargs):
        """Generates a SERS map. Also sets several attributes in the object with generated quantities for
        later retrieval.


        Args:
            N_hotspots: Number of hotspots on the plate (int > 0)
            K:          Number of peaks (int > 0)
            sig:        Measurement error (float > 0)
            sbr:        Approximate signal-to-background ratio (float > 0)
            alpha:      Relative amplitudes of peaks (np.ndarray of size self.K) or None (recommended)
            p_outlier:  Probability of a given spectrum being an outlier (ie outside hotspot) (0 <= float <= 1)
            background: String, np.ndarray or callable.
                        String options:
                            'default' generates background as AR(1) process.
                            'none' generates no background.
                        Callable should have signature
                        background(self.w, *bgargs, **bgkwargs) -> np.ndarray of size self.Nw x 1.
                        Note that in this case no further computations are done on the background signal - only basic
                        input checks are done.
                        In particular, this means that all inherent stochastic elements of such a signal *must* be
                        handled by the callable itself.
                        If it is an np.ndarray, the array is used for the background spectrum. Only input checks are perfomed

            plot:       Plot generated qunatities on generation (boolean)
            smooth:     Should Savitzky-Golay background smoothing be applied? (boolean)
                        Only available if background == 'default'
            *bgargs:    Any extra arguments to background if callable(background) == True.
            **bgkwargs: Any extra keyword-only arguments to background if callable(background) == True

        Returns:
            X:          Generated SERS map. (np.ndarray of size (self.nw, np.prod(mapsize))
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        self.sbr = sbr

        if self.c is None and K is None:
            raise ValueError('K needs to be specified if c, gamma and eta are not prespecified')

        if self.c is not None:
            if K is not None:
                if verbose:
                    logging.warning('c, gamma and eta are prespecified, so argument for K is ignored')
            K = len(self.c)
        self.K = K
        N = np.prod(self.mapsize)
        DD = np.zeros((N, self.Nw))
        LL = np.zeros((N, self.Nw))

        # measurement noise
        self.sigma = sig

        ### Generate outliers
        if p_outlier > 0:
            self.outliers = True
            NL = np.random.binomial(N, p_outlier)
            eta = np.random.randn(NL, self.Nw)
            L = np.zeros_like(eta)
            L[:, 0] = np.random.rand(NL)
            # c could be changed here
            c = 0
            phi = 1
            for w in range(1, self.Nw):
                L[:, w] = c + phi * L[:, w - 1] + eta[:, w]

            L -= repmat(np.min(L, axis=1)[:, np.newaxis], 1, self.Nw)
            L /= repmat(np.max(L, axis=1)[:, np.newaxis], 1, self.Nw)

            l = np.random.exponential(1e-2, size=(NL, 1))
            L *= repmat(l, 1, self.Nw)

            if plot:
                plt.figure()
                plt.title('Outliers')
                plt.plot(L.T)
                plt.show()

            inx = np.random.choice(N, NL)
            LL[inx, :] = L
            self.N_outliers = NL

        ### Generate background

        if type(background) == str and background == 'default':
            # this can be changed based on application
            eta = np.random.randn(self.Nw, 1)
            B = np.zeros_like(eta)
            B[0] = np.random.rand()
            c = 0.2
            phi = 0.995
            for w in range(1, self.Nw):
                B[w] = c + phi * B[w - 1] + eta[w]
            if smooth:
                from scipy.signal import savgol_filter
                B = savgol_filter(B.ravel(), window_length=149, polyorder=2)

            if plot:
                plt.figure()
                plt.title('Background spectrum')
                plt.plot(range(self.Nw), B)
                plt.show()

            if len(B) > 1:
                B -= np.min(B)
                B /= np.max(B)
                # B += np.random.rand()



        elif type(background) == str and background == 'none':
            if smooth:
                if verbose:
                    logging.warning('Smoothing available only for default background. Ignoring')
            B = np.zeros((1, self.Nw))

        elif callable(background):
            if smooth:
                if verbose:
                    logging.warning('Smoothing available only for default background. Ignoring')

            B = background(self.w, *bgargs, **bgkwargs)
            assert B.ndim == 1
            assert len(B) == self.Nw
            assert (B >= 0).all()

            if plot:
                plt.figure()
                plt.title('Background spectrum')
                plt.plot(B)
                plt.show()

        elif type(background) == np.ndarray:
            assert background.ndim == 1
            assert len(background) == self.Nw
            assert (background >= 0).all()

            B = background

        else:
            raise ValueError("Illegal input for 'background'. Should be 'default', 'none', a np.ndarray or a callable")

        self.B = B
        B = np.reshape(B, -1)
        B = repmat(B, N, 1)
        b = np.random.beta(100, 100, size=(N, 1))
        self.b = b

        b = repmat(b, 1, self.Nw)
        BB = b * B

        if plot:
            plt.matshow(B)
            plt.title('Background map')
            plt.colorbar()
            plt.show()

        ### Generate hotspots (signal)
        if N_hotspots > 0:
            mu = repmat(self.mapsize, N_hotspots, 1) * np.random.rand(N_hotspots, 2)
            r = 5 * np.random.rand(N_hotspots, 1) + 2
            A = np.random.rand(N_hotspots, 1)
            X = np.arange(self.mapsize[0])
            Y = np.arange(self.mapsize[1])
            XX, YY = np.meshgrid(X, Y)

            P = np.array([XX.reshape((-1)), YY.reshape(-1)]).T

            D = np.zeros(N)

            for h in range(N_hotspots):
                inner = (repmat(mu[h, :], N, 1) - P) ** 2
                D = D + A[h].item() * np.exp(-np.sum(inner, axis=1) / (r[h] * r[h]))

            # generate voigts
            if alpha is None:
                mina = sbr / 2
                alpha = mina + mina * np.random.rand(K)
            else:
                if alpha.ndim == 2:
                    assert alpha.shape == (N, self.K)
                else:
                    assert alpha.ndim == 1 and len(alpha) == self.K

            if self.gamma is None:
                gamma = np.random.gamma(21, 0.5, size=K)
                self.gamma = gamma
            else:
                gamma = self.gamma
            if self.c is None:
                c = gamma + (self.Nw - 2 * gamma) * np.random.rand(K)
                self.c = c
            else:
                c = self.c
            if self.eta is None:
                eta = np.random.rand(K)
                self.eta = eta
            else:
                eta = self.eta

            Vp = self.pseudo_voigt(self.w, c, gamma, eta)
            self.Vp = Vp
            
            if plot:
                plt.figure()
                plt.title('Voigt profiles')
                plt.plot(Vp.T)
                plt.show()

            spec = np.sum(Vp, axis=0).T

            if plot:
                plt.figure()
                plt.title('True underlying spectrum')
                plt.plot(self.w, spec)
                plt.show()
            if alpha.ndim == 1:
                A = repmat(alpha[:, np.newaxis], 1, N).T * repmat(D[:, np.newaxis], 1, K)
            else:
                A = alpha
            self.alpha = A

            DD = A @ Vp

        # generate noise
        eta = self.sigma * np.random.randn(N, self.Nw)

        self.real_noise = eta

        self.DD = DD
        self.BB = BB
        self.LL = LL

        X = DD + BB + LL + eta

        self.X = X

        if plot:
            plt.matshow(X)
            plt.colorbar()
            plt.title('Data matrix')
            plt.show()

        return X


def pseudo_voigt(w, c, gamma, eta):
    """Computes the (scaled) Pseudo-Voigt function over wavenumbers w with parameters c, gamma, eta efficiently.

    V = eta * Lorentzian + (1-eta) * Gaussian

    Args:
        w (np.array): Measured wavenumbers (sorted) (W x 1)
        c (np.array): Centers of Voigt curves (K x 1)
        gamma (np.array): Full-width-at-half-maximum (FWHM) of Voigt curves (K x 1)
        eta (np.array): Mixture coefficients of Lorentzian and Gaussian (0 <= eta <= 1) (K x 1)
    Returns:
        np.ndarray: Computed Voigt curves, (K x W)
    """
    K = len(c)
    assert len(gamma) == K and len(eta) == K
    W = len(w)

    xdata = np.tile(w, (K, 1))
    c_arr = np.tile(c, (W, 1)).T
    gamma_arr = np.tile(gamma, (W, 1)).T
    eta_arr = np.tile(eta, (W, 1)).T

    diff = xdata - c_arr
    kern = diff / gamma_arr
    diffsq = kern * kern

    # lorentzian
    L = 1 / (1 + diffsq)

    # gaussian
    G_kern = -0.5 * diffsq
    G = np.exp(G_kern)

    # vectorized voigts
    V = eta_arr * L + (1 - eta_arr) * G

    return V

# %%
# generate some "realistic" Raman data
if __name__ == "__main__":
    
    gen = SERSGenerator(mapsize=(20,20), Nw=500, seed=100, c=400, eta=[0,0])
    
    X = gen.generate(2,2,0.1,1, plot=False, background='none')
    fig, axs = plt.subplots(2,3, figsize=(11,6))
    
    c = gen.c
    mapsize = gen.mapsize
    
    # SERS Matrix
    axs[0,0].matshow(X)
    axs[0,0].set_xlabel('Wavenumber')
    axs[0,0].set_ylabel('Spectrum index')
    axs[0,0].set_title('X')
    
    
    # Loadings of peak 1
    mappable = axs[0,1].matshow(X[:, int(round(gen.c[0]))].reshape(*mapsize))
    axs[0,1].set_title(f'c={round(gen.c[0])}')
    plt.colorbar(mappable=mappable, ax=axs[0,1])
    
    # Loadings of peak 2
    mappable = axs[0,2].matshow(X[:, int(round(gen.c[1]))].reshape(*mapsize))
    axs[0,2].set_title(f'c={round(gen.c[1])}')
    plt.colorbar(mappable=mappable, ax=axs[0,2])
    
    axs[1,0].plot(gen.B)
    axs[1,0].set_xlabel('Wavenumber')
    axs[1,0].set_title('Normalized baseline')
    
    
    axs[1,1].plot(gen.Vp[0,:])
    axs[1,1].set_xlabel('Wavenumber')
    axs[1,1].set_title('Pure Voigt 1')
    
    axs[1,2].plot(gen.Vp[1,:])
    axs[1,2].set_xlabel('Wavenumber')
    axs[1,2].set_title('Pure Voigt 2')
    
    
    
    fig.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot((gen.alpha @ gen.Vp)[::100,:].T)
    ax.set_title('Pure simulated spectra')
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Intensity (A.U)')
    
    
    """
    # save data
    with open('data.pkl', 'wb') as f:
        dill.dump(gen, f)
    # similarly many maps can be saved in one file, if you add all SERSGenerator 
    # objects in eg. an np.ndarray or list 

    #with open('data.pkl', 'wb') as f:
    #    dill.dump(iterable_of_sersgenerators, f)
    
    # data can be loaded as
    with open('data.pkl', 'rb') as f:
        data = dill.load(f)
    """
