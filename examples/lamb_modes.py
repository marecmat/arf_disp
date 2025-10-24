import sys
import numpy as np
sys.path.extend(['./src/', '../src/'])
from compute import compute_dispersion



class lamb_modes:
    """
    Computes the modes in an elastic plate with free (vanishing stress) boundary conditions
    """

    def __init__(self, freqs, cl, ct, H):
        self.frequencies = freqs
        self.cl = cl    # Longitudinal velocity
        self.ct = ct    # Transverse velocity
        self.H = H      # layer thickness
        self.omega = self.frequencies[-1]*2*np.pi

    def update_frequency(self, omega):
        self.omega = omega
        return None 

    def get_matrix(self, k):
        kl = np.sqrt((self.omega**2/self.cl**2) - k**2 + 0j)
        kt = np.sqrt((self.omega**2/self.ct**2) - k**2 + 0j)

        expr = np.tan(kt*self.H)/np.tan(kl*self.H) + (4*k**2*kt*kl)/(kt**2 - k**2)**2
        return expr




if __name__ == '__main__':
    import matplotlib.pyplot as plt


    freqs  = np.linspace(1, 5e3, 800)[::-1]
    ALU = {'name': 'Aluminium', 'rho': 2700, 'cp': 6091, 'cs': 3134}
    method = lamb_modes(freqs, ALU['cp'], ALU['cs'], H=1)

    disp = compute_dispersion(
        method, 
        muller_params={
            'dx': .1+.1j, 
            'eps1': 1e-4, 
            'eps2': 1e-7, 
            'eps3': 1e8,
            'nbIter': 200, 
            'deflated': False, 
            'interval': ((0, 5), (-5, 5))
        }, 
        det_type='expr', 
        nb_sol=8, 
        map_size=(200, 100), 
        grid_interval=((0, 5), (-5, 5)), 
        verbose=1
    )

    roots, freqs = disp.compute()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')
    ax[0].scatter(np.real(roots), freqs, s=2, c='k')
    ax[1].scatter(np.imag(roots), freqs, s=2, c='k')

    plt.show()
